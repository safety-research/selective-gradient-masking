import argparse
import datetime
import os
import random
import time
from collections import Counter, defaultdict

import numpy as np
import torch
import torch._dynamo as dynamo
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
import wandb
import yaml
from datasets import Dataset, load_from_disk
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm
from transformers import DataCollatorForLanguageModeling, GPT2Tokenizer

from sgtm.train.utils import MultiDataLoader, float_range, load_model

dynamo.config.optimize_ddp = False


def logit_calibration(model, data_loader, local_rank, args):
    optimizer = torch.optim.AdamW([model.module.lm_head.bias], lr=args.logit_calibration_lr)

    scheduler = optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=1.0,
        end_factor=0.1,
        total_iters=args.logit_calibration_steps,
    )

    for _ in range(args.logit_calibration_steps):
        losses = {}
        optimizer.zero_grad()

        for source in ("forget", "adjacent", "retain"):
            batch = data_loader.get_batch(source)
            if batch is None:
                continue
            batch = {k: v.to(local_rank) for k, v in batch.items()}

            # Forward pass with mixed precision
            with torch.amp.autocast("cuda", enabled=True, dtype=torch.bfloat16):
                outputs = model(return_dict=True, **batch)
                losses[source] = outputs.loss

        loss = (
            losses.get("forget", 0)
            + args.logit_beta * losses.get("adjacent", 0)
            + args.logit_alpha * losses.get("retain", 0)
        )

        loss.backward()
        optimizer.step()
        scheduler.step()


def partition_dataset_by_category(dataset, top_level_only=False):
    df = dataset.to_pandas()
    if "category" not in df:
        df["category"] = "main"

    if top_level_only:
        df["category"] = df["category"].str.split(".").str[0]

    category_groups = df.groupby("category")

    category_datasets = {}
    for category, group in category_groups:
        category_datasets[category] = Dataset.from_pandas(group.reset_index(drop=True))

    return category_datasets


def evaluate_by_category(
    model,
    dataset,
    data_collator,
    local_rank,
    world_size,
    top_level_only=False,
):
    """Evaluate model on each category separately and return per-category metrics."""
    category_datasets = partition_dataset_by_category(dataset, top_level_only)

    category_metrics = defaultdict(dict)
    all_losses = []
    all_weights = []

    for category, cat_dataset in category_datasets.items():
        cat_dataset = cat_dataset.select_columns(["input_ids", "attention_mask"])
        sampler = DistributedSampler(cat_dataset, num_replicas=world_size, rank=local_rank % world_size, shuffle=False)

        data_loader = DataLoader(
            cat_dataset,
            batch_size=32,
            sampler=sampler,
            collate_fn=data_collator,
            num_workers=4,
            pin_memory=True,
        )

        all_eval_losses = evaluate(
            model=model,
            data_iter=data_loader,
            local_rank=local_rank,
            world_size=world_size,
        )

        avg_eval_loss = sum([loss.item() for loss in all_eval_losses]) / world_size
        category_metrics[category]["loss"] = avg_eval_loss
        category_metrics[category]["perplexity"] = np.exp(avg_eval_loss)
        category_metrics[category]["num_samples"] = len(cat_dataset)

        all_losses.append(avg_eval_loss)
        all_weights.append(len(cat_dataset))

    return category_metrics, all_losses, all_weights


def evaluate(model, data_iter, local_rank, world_size):
    model.eval()
    eval_losses = []

    with torch.no_grad():
        for eval_batch in data_iter:
            eval_batch = {k: v.to(local_rank) for k, v in eval_batch.items()}

            with torch.amp.autocast("cuda", enabled=True, dtype=torch.bfloat16):
                eval_outputs = model(return_dict=True, **eval_batch)
                eval_loss = eval_outputs.loss
            eval_losses.append(eval_loss.item())

    # Gather eval losses from all processes
    all_eval_losses = [torch.zeros(1).to(local_rank) for _ in range(world_size)]
    eval_loss = torch.tensor(sum(eval_losses) / len(eval_losses)).to(local_rank)
    dist.all_gather(all_eval_losses, eval_loss)

    model.train()
    return all_eval_losses


def evaluate_all_datasets(
    model,
    datasets,
    data_collator,
    local_rank,
    world_size,
    top_level_only=False,
):
    """Evaluate model on retain and forget datasets and return metrics dict."""
    metrics = defaultdict(dict)

    for label, dataset in datasets.items():
        if len(dataset) == 0:
            continue
        
        category_metrics, all_losses, _ = evaluate_by_category(
            model=model,
            dataset=dataset,
            data_collator=data_collator,
            local_rank=local_rank,
            world_size=world_size,
            top_level_only=top_level_only and label != "forget",
        )
        metrics[label]["val_loss"] = np.mean(all_losses)
        metrics[label]["val_ppl"] = np.exp(np.mean(all_losses))
        metrics[label].update(category_metrics)

    return metrics


def train(args):
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    dist.init_process_group("nccl")
    torch.cuda.set_device(local_rank)

    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    if args.forget_adjacent_dataset_path:
        forget_adjacent_dataset = load_from_disk(args.forget_adjacent_dataset_path)
    else:
        forget_adjacent_dataset = {
            "train": Dataset.from_dict({"input_ids": [], "attention_mask": []}),
            "test": Dataset.from_dict({"input_ids": [], "attention_mask": []}),
        }

    retain_dataset = load_from_disk(args.retain_dataset_path)
    forget_dataset = load_from_disk(args.forget_dataset_path)

    model = load_model(
        hidden_size=args.hidden_size,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        context_size=args.context_size,
        tie_weights=args.tie_weights,
        clean_model=args.clean_model,
        forget_mlp_dim=args.forget_mlp_dim,
        forget_attn_heads=args.forget_attn_heads,
        forget_param_perc=args.forget_param_perc,
        masked_layers=args.masked_layers,
        masking_strategy=args.masking_strategy,
        split_masked_weights=True,
        sgtm_mask_embeddings=args.mask_embeddings,
        do_print=rank == 0,
        finetune_from=args.finetune_from,
        finetune_ablate=True,
        not_trainable_ablate=True,
    )

    if rank == 0:
        print("Model architecture:")
        print(model)

    # Print total number of parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if rank == 0:
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")

    model = model.to(local_rank)

    model = DDP(model, device_ids=[local_rank], find_unused_parameters=False, broadcast_buffers=False)
    model = torch.compile(model)

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    effective_batch_size = args.physical_batch_size * world_size
    if args.logical_batch_size % effective_batch_size != 0:
        raise ValueError(
            f"Logical batch size ({args.logical_batch_size}) must be divisible by "
            f"effective batch size (physical_batch_size * world_size = {effective_batch_size})"
        )
    gradient_accumulation_steps = args.logical_batch_size // effective_batch_size

    # Calculate proportions based on dataset sizes
    total_dataset_size = (
        len(forget_adjacent_dataset["train"]) + len(retain_dataset["train"]) + len(forget_dataset["train"])
    )
    data_split_order = (
        ["adjacent"] * int(len(forget_adjacent_dataset["train"]) * args.upsample_adjacent_set)
        + ["retain"] * int(len(retain_dataset["train"]) * args.upsample_retain_set)
        + ["forget"] * int(len(forget_dataset["train"]) * args.upsample_forget_set)
    )

    rng = random.Random(42)
    rng.shuffle(data_split_order)
    data_split_order = data_split_order[: args.total_steps]

    steps_per_epoch = total_dataset_size // args.logical_batch_size

    data_split_counts = Counter(data_split_order)

    true_forget = data_split_counts["forget"]
    true_retain = data_split_counts["retain"] + data_split_counts["adjacent"]

    if args.forget_fpr is not None and args.forget_tpr is not None:
        if rank == 0:
            print(f"Using TPR/FPR configuration: TPR={args.forget_tpr}, FPR={args.forget_fpr}")

        true_positives = args.forget_tpr * true_forget
        false_posities = args.forget_fpr * true_positives
        true_negatives = true_retain - false_posities
        false_negatives = true_forget - true_positives
    else:
        if rank == 0:
            print(
                f"Using precision/recall configuration: precision={args.forget_precision}, recall={args.forget_recall}"
            )

        true_positives = args.forget_recall * true_forget
        false_posities = true_positives * ((1 / args.forget_precision) - 1)
        true_negatives = true_retain - false_posities
        false_negatives = true_forget - true_positives

    prob_false_negative = false_negatives / data_split_counts["forget"] if data_split_counts["forget"] > 0 else 0
    prob_false_positive = false_posities / data_split_counts["adjacent"] if data_split_counts["adjacent"] > 0 else 0

    if rank == 0:
        print(f"Physical batch size per GPU: {args.physical_batch_size}")
        print(f"World size: {world_size}")
        print(f"Effective batch size per step: {effective_batch_size}")
        print(f"Logical batch size: {args.logical_batch_size}")
        print(f"Gradient accumulation steps: {gradient_accumulation_steps}")
        print(f"Forget-adjacent dataset: {len(forget_adjacent_dataset['train'])} train samples")
        print(f"Retain dataset: {len(retain_dataset['train'])} train samples")
        print(f"Forget dataset: {len(forget_dataset['train'])} train samples")
        print(f"Total training steps: {args.total_steps}")
        print(
            f"Steps per data split: "
            f"adjacent={data_split_counts['adjacent']} ({data_split_counts['adjacent'] / len(data_split_order) * 100:.1f}%), "
            f"retain={data_split_counts['retain']} ({data_split_counts['retain'] / len(data_split_order) * 100:.1f}%),"
            f"forget={data_split_counts['forget']} ({data_split_counts['forget'] / len(data_split_order) * 100:.1f}%)"
        )
        print(f"Using warmup for {args.warmup_steps} steps")
        print(f"Inject forget data: {args.inject_forget}")
        print(f"Upsample forget factor: {args.upsample_forget_set}")
        print(f"Upsample adjacent factor: {args.upsample_adjacent_set}")
        print(f"Upsample retain factor: {args.upsample_retain_set}")
        print(f"Forget retain percentage: {args.forget_retain_perc}")
        print(f"Adjacent retain percentage: {args.adjacent_retain_perc}")
        print(f"Retain retain percentage: {args.retain_retain_perc}")
        print("Confusion matrix calculations:")
        print(f"  True forget samples: {true_forget}")
        print(f"  True retain samples: {true_retain}")
        print(f"  True positives: {true_positives:.1f}")
        print(f"  False positives: {false_posities:.1f}")
        print(f"  True negatives: {true_negatives:.1f}")
        print(f"  False negatives: {false_negatives:.1f}")
        print(f"P(False Negative) = {prob_false_negative:.4f}")
        print(f"P(False Positive) = {prob_false_positive:.4f}")

        epoch_percentage = (args.total_steps / steps_per_epoch) * 100 if steps_per_epoch > 0 else 0
        print(f"Steps per epoch: {steps_per_epoch} ({epoch_percentage:.1f}% of epoch)")

    train_loader = MultiDataLoader(
        datasets={
            "adjacent": forget_adjacent_dataset["train"].select_columns(["input_ids", "attention_mask"]),
            "retain": retain_dataset["train"].select_columns(["input_ids", "attention_mask"]),
            "forget": forget_dataset["train"].select_columns(["input_ids", "attention_mask"]),
        },
        rank=rank,
        world_size=world_size,
        batch_size=args.physical_batch_size,
        shuffle=True,
        collate_fn=data_collator,
        num_workers=8,
        pin_memory=True,
    )

    calibration_loader = MultiDataLoader(
        datasets={
            "adjacent": forget_adjacent_dataset["train"].select_columns(["input_ids", "attention_mask"]),
            "retain": retain_dataset["train"].select_columns(["input_ids", "attention_mask"]),
            "forget": forget_dataset["train"].select_columns(["input_ids", "attention_mask"]),
        },
        rank=rank,
        world_size=world_size,
        batch_size=args.physical_batch_size,
        shuffle=True,
        collate_fn=data_collator,
    )

    optimizers = {}
    if args.optimizer_strategy == "legacy":
        optimizers["joint"] = optim.AdamW(
            model.parameters(),
            lr=args.learning_rate,
            weight_decay=0.1,
            betas=(0.9, 0.95),
        )
        if rank == 0:
            num_params = sum(p.numel() for p in model.parameters())
            print(f"Number of parameters in joint optimizer: {num_params:,}")

    elif args.optimizer_strategy == "split":
        for param_type in ("joint", "retain", "forget"):
            optimizers[param_type] = optim.AdamW(
                model.module.parameters_split(sgtm_split=param_type),
                lr=args.learning_rate,
                weight_decay=0.1,
                betas=(0.9, 0.95),
            )
            if rank == 0:
                num_params = sum(p.numel() for p in model.module.parameters_split(sgtm_split=param_type))
                print(f"Number of parameters in {param_type} optimizer: {num_params:,}")
    else:
        raise ValueError(
            f"Unsupported optimizer strategy: {args.optimizer_strategy}. Supported strategies are: 'legacy', 'split'"
        )

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"{args.run_name}_{timestamp}"
    output_dir = os.path.join(args.output_root, args.wandb_project, run_name, "output")

    if rank == 0:
        os.makedirs(output_dir, exist_ok=True)
        wandb.init(project=args.wandb_project, name=run_name, config=vars(args))

    model.train()

    global_step = 0
    last_log_time = time.time()
    stats = Counter()

    schedulers = {}
    if not args.constant_lr:
        for optim_k, optim_v in optimizers.items():
            warmup_scheduler = LinearLR(optim_v, start_factor=0.1, end_factor=1.0, total_iters=args.warmup_steps)
            cosine_scheduler = CosineAnnealingLR(optim_v, T_max=args.total_steps - args.warmup_steps, eta_min=0)
            schedulers[optim_k] = SequentialLR(
                optim_v, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[args.warmup_steps]
            )

    for global_step in tqdm(range(args.total_steps + 1), disable=rank != 0):
        for optimizer in optimizers.values():
            optimizer.zero_grad(set_to_none=True)
        # Evaluation
        if (
            (global_step > 0 or args.finetune_from)
            and (global_step % args.eval_steps == 0 or global_step == args.total_steps)
            # ) or (args.finetune_from and global_step < args.eval_steps):
        ):
            # Evaluate on all datasets
            eval_metrics = {}
            eval_metrics["eval"] = evaluate_all_datasets(
                model=model,
                datasets={
                    "forget": forget_dataset["test"],
                    "default": forget_adjacent_dataset["test"],
                    "retain": retain_dataset["test"],
                },
                local_rank=local_rank,
                world_size=world_size,
                data_collator=data_collator,
            )

            saved_state = {k: v.clone() for k, v in model.state_dict().items()}
            if not args.clean_model:
                model.module.ablate(trainable=False)

                eval_metrics["eval_ablated"] = evaluate_all_datasets(
                    model=model,
                    datasets={
                        "forget": forget_dataset["test"],
                        "adjacent": forget_adjacent_dataset["test"],
                        "retain": retain_dataset["test"],
                    },
                    local_rank=local_rank,
                    world_size=world_size,
                    data_collator=data_collator,
                )

            if (args.logit_calibration_steps is not None) and (
                args.logit_on_intermediate or global_step == args.total_steps
            ):
                calibration_model = load_model(
                    hidden_size=args.hidden_size,
                    num_heads=args.num_heads,
                    num_layers=args.num_layers,
                    context_size=args.context_size,
                    tie_weights=args.tie_weights,
                    clean_model=args.clean_model,
                    forget_mlp_dim=args.forget_mlp_dim,
                    forget_attn_heads=args.forget_attn_heads,
                    forget_param_perc=args.forget_param_perc,
                    masked_layers=args.masked_layers,
                    masking_strategy=args.masking_strategy,
                    split_masked_weights=True,
                    sgtm_mask_embeddings=args.mask_embeddings,
                    do_print=False,
                )

                # Copy the current model's state to the calibration model
                for param in calibration_model.parameters():
                    param.requires_grad = False
                calibration_model.lm_head.bias.requires_grad = True

                calibration_model = calibration_model.to(local_rank)
                calibration_model = DDP(
                    calibration_model, device_ids=[local_rank], find_unused_parameters=False, broadcast_buffers=False
                )
                calibration_model = torch.compile(calibration_model)
                calibration_model.load_state_dict(saved_state)
                if not args.clean_model:
                    calibration_model.module.ablate(trainable=False)

                generator = torch.Generator(device=f"cuda:{local_rank}")
                generator.manual_seed(42)
                torch.nn.init.normal_(
                    calibration_model.module.lm_head.bias, mean=0.0, std=args.logit_bias_std, generator=generator
                )

                logit_calibration(
                    model=calibration_model, data_loader=calibration_loader, local_rank=local_rank, args=args
                )
                eval_metrics["eval_calibrated"] = evaluate_all_datasets(
                    model=calibration_model,
                    datasets={
                        "forget": forget_dataset["test"],
                        "adjacent": forget_adjacent_dataset["test"],
                        "retain": retain_dataset["test"],
                    },
                    local_rank=local_rank,
                    world_size=world_size,
                    data_collator=data_collator,
                )

            model.load_state_dict(saved_state)
            if rank == 0:
                wandb.log(eval_metrics, step=global_step)

            model.train()

        if global_step == args.total_steps:
            break

        total_loss = 0
        batch_true_label = data_split_order[global_step]

        batch_applied_label = None
        if batch_true_label == "forget":
            if rng.random() < prob_false_negative:
                batch_applied_label = "default"
            elif rng.random() < args.forget_retain_perc / 100:
                batch_applied_label = "retain"
            else:
                batch_applied_label = "forget"

        if batch_true_label == "adjacent":
            if rng.random() < prob_false_positive:
                batch_applied_label = "forget"
            elif rng.random() < args.adjacent_retain_perc / 100:
                batch_applied_label = "retain"
            else:
                batch_applied_label = "default"

        if batch_true_label == "retain":
            if rng.random() < args.retain_retain_perc / 100:
                batch_applied_label = "retain"
            else:
                batch_applied_label = "default"

        stats[f"sgtm/batch_{batch_true_label}_to_{batch_applied_label}"] += 1

        if batch_applied_label == "forget" and not args.inject_forget:
            for scheduler in schedulers.values():
                scheduler.step()
            continue

        for _ in range(gradient_accumulation_steps):
            batch = train_loader.get_batch(batch_true_label)
            batch = {k: v.to(local_rank) for k, v in batch.items()}

            if not args.clean_model:
                batch["sgtm_mode"] = batch_applied_label

                stats[f"sgtm/planned_{batch_true_label}"] += 1
                stats[f"sgtm/applied_{batch_applied_label}"] += 1

            # Forward pass with mixed precision
            with torch.amp.autocast("cuda", enabled=True, dtype=torch.bfloat16):
                outputs = model(return_dict=True, **batch)
                loss = outputs.loss / gradient_accumulation_steps  # Scale loss

            # Backward pass
            loss.backward()
            total_loss += loss.item()

        if not args.clean_model:
            model.module.adjust_gradients(sgtm_mode=batch["sgtm_mode"])

        optimizers["joint"].step()
        stats["sgtm/opt_joint"] += 1
        if args.optimizer_strategy == "split":
            if batch["sgtm_mode"] == "default":
                optimizers["forget"].step()
                optimizers["retain"].step()

                stats["sgtm/opt_forget"] += 1
                stats["sgtm/opt_retain"] += 1
            elif batch["sgtm_mode"] == "forget":
                optimizers["forget"].step()
                stats["sgtm/opt_forget"] += 1
            elif batch["sgtm_mode"] == "retain":
                optimizers["retain"].step()
                stats["sgtm/opt_retain"] += 1

        for scheduler in schedulers.values():
            scheduler.step()

        if rank == 0 and global_step > 0 and global_step % args.logging_steps == 0:
            current_time = time.time()
            time_elapsed = current_time - last_log_time
            tokens_per_step = args.physical_batch_size * world_size * args.context_size * gradient_accumulation_steps
            tokens_since_last_log = args.logging_steps * tokens_per_step
            tokens_per_second = tokens_since_last_log / time_elapsed
            last_log_time = current_time
            epoch_float = global_step / steps_per_epoch

            total_norm = 0.0
            for p in model.parameters():
                if p.grad is not None:
                    param_norm = p.grad.detach().data.norm(2)
                    total_norm += param_norm.item() ** 2
            total_norm = total_norm**0.5

            # Calculate total tokens processed and FLOPs
            tokens_processed = args.logical_batch_size * global_step * args.context_size
            flops = 6 * total_params * tokens_processed

            log_dict = {
                "train/loss": total_loss,
                "train/tokens_per_second": tokens_per_second,
                "train/epoch": epoch_float,
                "train/tokens_processed": tokens_processed,
                "train/flops": flops,
                "train/grad_norm": total_norm,
                "train/lr": optimizers["joint"].param_groups[0]["lr"],
                "train/step": global_step,
                "train/forget_epoch": train_loader.epochs.get("forget", 0),
            }
            log_dict.update(stats)

            wandb.log(log_dict, step=global_step)

    # Save final model
    if rank == 0:
        final_checkpoint_path = os.path.join(output_dir, "final-checkpoint")
        os.makedirs(final_checkpoint_path, exist_ok=True)

        model_to_save = model.module if hasattr(model, "module") else model
        model_to_save.save_pretrained(final_checkpoint_path)
        tokenizer.save_pretrained(final_checkpoint_path)

        # Save optimizer state(s)
        optimizer_state_path = os.path.join(final_checkpoint_path, "optimizer.pt")
        optimizer_states = {k: v.state_dict() for k, v in optimizers.items()}
        torch.save(optimizer_states, optimizer_state_path)

        # Save training state for resuming
        training_state_path = os.path.join(final_checkpoint_path, "training_state.pt")
        training_state = {"global_step": global_step, "optimizer_strategy": args.optimizer_strategy, "args": vars(args)}
        torch.save(training_state, training_state_path)

    # Final evaluation on validation set
    model.eval()

    if rank == 0:
        # Report total tokens processed and FLOPs
        total_tokens = args.logical_batch_size * global_step * args.context_size
        total_flops = 6 * total_params * total_tokens
        print(f"\nTotal tokens processed: {total_tokens:,} ({total_tokens / 1e9:.3f}B)")
        print(f"Total FLOPs: {total_flops:.2e} ({total_flops / 1e18:.3f} PFLOPs)")

        wandb.summary["final/tokens_processed"] = total_tokens
        wandb.summary["final/flops"] = total_flops

        wandb.finish()

    dist.destroy_process_group()


def parse_args():
    parser = argparse.ArgumentParser(description="Train a model on retain dataset with forget injection")
    parser.add_argument(
        "--model-config", type=str, default=None, help="Path to YAML config file for model architecture"
    )
    parser.add_argument("--output-root", type=str, help="Root directory for output files")
    parser.add_argument("--clean-model", action="store_true", default=False, help="No gradient masking if set to true")
    parser.add_argument(
        "--forget-mlp-dim",
        type=int,
        default=None,
        help="MLP dimension dedicated to forget data (use this OR forget-param-perc)",
    )
    parser.add_argument(
        "--forget-attn-heads",
        type=int,
        default=None,
        help="Number of attention heads dedicated to forget data (use this OR forget-param-perc)",
    )
    parser.add_argument(
        "--forget-param-perc",
        type=float,
        default=None,
        help="Percentage of parameters dedicated to forget data (0-100). Automatically calculates forget-mlp-dim and forget-attn-heads",
    )
    parser.add_argument(
        "--masked-layers", type=int, nargs="+", default=None, help="Layer indices to apply gradient masking"
    )
    parser.add_argument(
        "--masking-strategy",
        type=str,
        default=None,
        help="Masking strategy (gradient_routing, activation_masking, parameter_masking)",
    )
    parser.add_argument(
        "--mask-embeddings",
        action="store_true",
        default=False,
        help="Whether to apply gradient masking to embeddings",
    )
    parser.add_argument("--physical-batch-size", type=int, default=None, help="Total batch size for gradient updates")
    parser.add_argument("--logical-batch-size", type=int, default=None, help="Total batch size for gradient updates")
    parser.add_argument("--learning-rate", type=float, default=None, help="Learning rate for training")
    parser.add_argument("--num-layers", type=int, default=None, help="Number of transformer layers")
    parser.add_argument("--hidden-size", type=int, default=None, help="Hidden size of the model")
    parser.add_argument("--num-heads", type=int, default=None, help="Number of attention heads")
    parser.add_argument("--tie-weights", action="store_true", default=None, help="Tie input and output embeddings")
    parser.add_argument(
        "--no-tie-weights", dest="tie_weights", action="store_false", help="Don't tie input and output embeddings"
    )
    parser.add_argument("--eval-steps", type=int, default=None, help="Number of steps between evaluations")
    parser.add_argument("--logging-steps", type=int, default=50, help="Number of steps between logging")
    parser.add_argument("--wandb-project", type=str, default="generic_ddp", help="Wandb project name")
    parser.add_argument("--run-name", type=str, default="generic_ddp", help="Name for this run")
    parser.add_argument(
        "--total-steps", type=int, default=None, help="Total number of training steps. If not set, train for full epoch"
    )
    parser.add_argument("--context-size", type=int, default=None, help="Context size for the model")
    parser.add_argument(
        "--warmup-steps", type=int, default=None, help="Number of warmup steps for learning rate scheduler"
    )
    parser.add_argument(
        "--forget-adjacent-dataset-path", type=str, default=None, help="Path to forget-adjacent dataset"
    )
    parser.add_argument("--retain-dataset-path", type=str, required=True, help="Path to pre-tokenized retain dataset")
    parser.add_argument("--forget-dataset-path", type=str, required=True, help="Path to forget dataset")
    parser.add_argument(
        "--upsample-forget-set",
        type=float,
        default=1.0,
        help="Factor to upsample forget dataset (default 1.0 = no upsampling)",
    )
    parser.add_argument(
        "--upsample-adjacent-set",
        type=float,
        default=1.0,
        help="Factor to upsample forget-adjacent dataset (default 1.0 = no upsampling)",
    )
    parser.add_argument(
        "--upsample-retain-set",
        type=float,
        default=1.0,
        help="Factor to upsample retain dataset (default 1.0 = no upsampling)",
    )
    parser.add_argument(
        "--forget-retain-perc",
        type=float,
        default=0.0,
    )
    parser.add_argument(
        "--adjacent-retain-perc",
        type=float,
        default=0.0,
    )
    parser.add_argument(
        "--retain-retain-perc",
        type=float,
        default=0.0,
    )
    parser.add_argument(
        "--forget-precision",
        type=float_range(0, 1),
        default=1,
    )
    parser.add_argument(
        "--forget-recall",
        type=float_range(0, 1),
        default=1,
    )
    parser.add_argument(
        "--forget-tpr",
        type=float_range(0, 1),
        default=None,
    )
    parser.add_argument(
        "--forget-fpr",
        type=float_range(0, 1),
        default=None,
    )
    parser.add_argument(
        "--inject-forget",
        action="store_true",
        default=False,
        help="Whether to inject forget data during training",
    )
    parser.add_argument(
        "--logit-calibration-steps",
        type=int,
        default=None,
        help="Number of steps for logit calibration",
    )
    parser.add_argument(
        "--logit-calibration-lr",
        type=float,
        default=1e-2,
        help="Learning rate for logit calibration",
    )
    parser.add_argument(
        "--logit-alpha",
        type=float,
        default=100.0,
        help="Alpha parameter for logit calibration loss weighting",
    )
    parser.add_argument(
        "--logit-beta",
        type=float,
        default=1.0,
        help="Alpha parameter for logit calibration loss weighting",
    )
    parser.add_argument(
        "--logit-bias-std",
        type=float,
        default=0.1,
        help="Standard deviation for logit bias initialization during calibration",
    )
    parser.add_argument(
        "--logit-on-intermediate",
        action="store_true",
        default=False,
        help="Whether to apply logit calibration on intermediate layers",
    )
    parser.add_argument("--optimizer-strategy", type=str, default="legacy")
    parser.add_argument("--finetune-from", type=str, default=None, help="Path to checkpoint to finetune from")
    parser.add_argument(
        "--constant-lr",
        action="store_true",
        default=False,
        help="Use constant learning rate, disabling all LR scheduling",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    # Load model config from YAML if provided
    if args.model_config:
        with open(args.model_config, "r") as f:
            config = yaml.safe_load(f)

        if "model" in config:
            model_config = config["model"]
            for attr in [
                "num_layers",
                "hidden_size",
                "num_heads",
                "tie_weights",
                "learning_rate",
                "warmup_steps",
                "context_size",
                "physical_batch_size",
                "logical_batch_size",
                "total_steps",
                "eval_steps",
            ]:
                if attr in model_config:
                    if getattr(args, attr) is not None:
                        print(
                            f"Warning: {attr} value set via command line argument ({getattr(args, attr)}) "
                            f"overrides config value ({model_config[attr]})"
                        )
                        continue
                    setattr(args, attr, model_config[attr])

    if args.clean_model:
        if (
            args.forget_mlp_dim
            or args.forget_attn_heads
            or args.forget_param_perc
            or args.masking_strategy
            or args.masked_layers
        ):
            raise ValueError(
                "Cannot specify masking arguments (forget_mlp_dim, forget_attn_heads, forget_param_perc, masking_strategy) "
                "when using a clean model (no gradient masking)"
            )
    else:
        if not args.masking_strategy:
            raise ValueError("When using gradient masking, must specify masking_strategy")

        if (args.forget_param_perc is None) and (args.forget_mlp_dim is None or args.forget_attn_heads is None):
            raise ValueError(
                "When using gradient masking, must specify either forget_param_perc OR (forget_mlp_dim and forget_attn_heads)"
            )

        if (args.forget_param_perc is not None) and (
            args.forget_mlp_dim is not None or args.forget_attn_heads is not None
        ):
            raise ValueError(
                "Cannot use --forget-param-perc with --forget-mlp-dim or --forget-attn-heads. Choose one method."
            )

        if args.masked_layers is None:
            args.masked_layers = list(range(args.num_layers))

    train(args)

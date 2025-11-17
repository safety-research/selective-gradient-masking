import argparse
import numpy as np
import torch
import wandb
from transformers import DataCollatorForLanguageModeling, GPT2Tokenizer
from sgtm.model import GPTNeoForCausalLMSGTM
from datasets import load_from_disk
from sgtm.train.utils import MultiDataLoader
import torch.optim as optim
from tqdm import tqdm
import torch.nn.functional as F
import datetime
import os
from collections import defaultdict
from sgtm.train.trainer import partition_dataset_by_category
from torch.utils.data import DataLoader


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=str, help="Root directory for output files")
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--retain-dataset-path",
        type=str,
    )
    parser.add_argument(
        "--forget-dataset-path",
        type=str,
    )

    parser.add_argument("--alpha", type=int, default=100, help="retain weight")
    parser.add_argument(
        "--steering-coeff",
        type=int,
        default=20,
    )
    parser.add_argument("--lr", type=float, default=5e-5, help="learning rate")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--total-steps", type=int, default=80)
    parser.add_argument("--layer-id", type=int, required=True, help="layer to unlearn")
    parser.add_argument("--layer-ids", type=int, nargs="+", required=True, help="update layers")
    parser.add_argument("--param-ids", type=int, nargs="+", default=[6], help="update params")
    parser.add_argument("--seed", type=int, default=42, help="Seed")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use (cuda/cpu)")
    parser.add_argument("--wandb-project", type=str, default="generic_ddp", help="Wandb project name")
    parser.add_argument("--run-name", type=str, default="generic_ddp", help="Name for this run")

    return parser.parse_args()


def evaluate_by_category(
    model,
    dataset,
    data_collator,
    local_rank,
    top_level_only=False,
):
    """Evaluate model on each category separately and return per-category metrics."""
    category_datasets = partition_dataset_by_category(dataset, top_level_only)

    category_metrics = defaultdict(dict)
    all_losses = []
    all_weights = []

    for category, cat_dataset in tqdm(category_datasets.items()):
        cat_dataset = cat_dataset.select_columns(["input_ids", "attention_mask"])

        data_loader = DataLoader(
            cat_dataset,
            batch_size=32,
            collate_fn=data_collator,
            num_workers=4,
            pin_memory=True,
        )

        all_eval_losses = evaluate(
            model=model,
            data_iter=data_loader,
            local_rank=local_rank,
        )

        avg_eval_loss = np.mean(all_eval_losses)
        category_metrics[category]["loss"] = avg_eval_loss
        category_metrics[category]["perplexity"] = np.exp(avg_eval_loss)
        category_metrics[category]["num_samples"] = len(cat_dataset)

        all_losses.append(avg_eval_loss)
        all_weights.append(len(cat_dataset))

    return category_metrics, all_losses, all_weights


def evaluate(model, data_iter, local_rank):
    model.eval()
    eval_losses = []

    with torch.no_grad():
        for eval_batch in data_iter:
            eval_batch = {k: v.to(local_rank) for k, v in eval_batch.items()}

            with torch.amp.autocast("cuda", enabled=True, dtype=torch.bfloat16):
                eval_outputs = model(return_dict=True, **eval_batch)
                eval_loss = eval_outputs.loss
            eval_losses.append(eval_loss.item())

    model.train()
    return eval_losses


def evaluate_all_datasets(
    model,
    datasets,
    data_collator,
    device,
    top_level_only=False,
):
    """Evaluate model on retain and forget datasets and return metrics dict."""
    metrics = defaultdict(dict)

    for label, dataset in datasets.items():
        category_metrics, all_losses, _ = evaluate_by_category(
            model=model,
            dataset=dataset,
            data_collator=data_collator,
            local_rank=device,
            top_level_only=top_level_only and label != "forget",
        )
        metrics[label]["val_loss"] = np.mean(all_losses)
        metrics[label]["val_ppl"] = np.exp(np.mean(all_losses))
        metrics[label].update(category_metrics)

    return metrics


def get_params(model, layer_ids, param_ids):
    params = []
    for layer_id in layer_ids:
        for i, (pname, p) in enumerate(model.transformer.h[layer_id].named_parameters()):
            if i in param_ids:
                print(f"Adding parameter {i} from layer {layer_id}: {pname}")
                params.append(p)
    return params


def forward_with_cache(model, inputs, module, no_grad=True):
    # define a tensor with the size of our cached activations
    cache = []

    def hook(module, input, output):
        if isinstance(output, tuple):
            cache.append(output[0])
        else:
            cache.append(output)
        return None

    hook_handle = module.register_forward_hook(hook)

    if no_grad:
        with torch.no_grad():
            _ = model(**inputs)
    else:
        _ = model(**inputs)

    hook_handle.remove()

    return cache[0]


def train(args):
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    frozen_model = GPTNeoForCausalLMSGTM.from_pretrained(args.model_path).to(args.device)
    updated_model = GPTNeoForCausalLMSGTM.from_pretrained(args.model_path).to(args.device)

    retain_dataset = load_from_disk(args.retain_dataset_path)
    retain_train = retain_dataset["train"].select_columns(["input_ids", "attention_mask"])
    retain_test = retain_dataset["test"]

    forget_dataset = load_from_disk(args.forget_dataset_path)
    forget_train = forget_dataset["train"].select_columns(["input_ids", "attention_mask"])
    forget_test = forget_dataset["test"]

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    train_loader = MultiDataLoader(
        datasets={"retain": retain_train, "forget": forget_train},
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=data_collator,
        num_workers=8,
        pin_memory=True,
    )

    updated_model = updated_model.train()
    params = get_params(updated_model, args.layer_ids, args.param_ids)
    optimizer = optim.AdamW(params, lr=args.lr)
    frozen_module = frozen_model.transformer.h[args.layer_id]
    updated_module = updated_model.transformer.h[args.layer_id]

    random_vector = torch.rand(
        1, 1, updated_model.config.hidden_size, dtype=updated_model.dtype, device=updated_model.device
    )
    control_vec = random_vector / torch.norm(random_vector) * args.steering_coeff

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"{args.run_name}_{timestamp}"
    output_dir = os.path.join(args.output_root, args.wandb_project, run_name, "output")
    wandb.init(project=args.wandb_project, name=run_name, config=vars(args))

    for step in tqdm(range(args.total_steps)):
        unlearn_batch = train_loader.get_batch("forget")
        unlearn_batch = {k: v.to(args.device) for k, v in unlearn_batch.items()}
        retain_batch = train_loader.get_batch("retain")
        retain_batch = {k: v.to(args.device) for k, v in retain_batch.items()}

        updated_forget_activations = forward_with_cache(
            updated_model, unlearn_batch, module=updated_module, no_grad=False
        )

        unlearn_loss = F.mse_loss(updated_forget_activations, control_vec)

        updated_retain_activations = forward_with_cache(
            updated_model, retain_batch, module=updated_module, no_grad=False
        )
        frozen_retain_activations = forward_with_cache(frozen_model, retain_batch, module=frozen_module, no_grad=True)

        retain_loss = torch.nn.functional.mse_loss(updated_retain_activations, frozen_retain_activations)
        retain_loss *= args.alpha

        loss = unlearn_loss + retain_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        frozen_forget_activations = forward_with_cache(frozen_model, unlearn_batch, module=frozen_module, no_grad=True)
        unlearn_cosine = torch.nn.functional.cosine_similarity(
            updated_forget_activations, frozen_forget_activations, dim=-1
        ).mean()
        retain_cosine = torch.nn.functional.cosine_similarity(
            updated_retain_activations, frozen_retain_activations, dim=-1
        ).mean()

        log_dict = {
            "train/loss": loss,
            "train/unlearn_loss": unlearn_loss,
            "train/retain_loss": retain_loss,
            "train/param_change": params[0].grad.abs().mean().item(),
            "train/unlearn_cosine": unlearn_cosine.item(),
            "train/retain_cosine": retain_cosine.item(),
            "train/updated_forget_activations": torch.mean(
                updated_forget_activations.norm(dim=-1).mean(dim=1), dim=0
            ).item(),
            "train/frozen_forget_activations": torch.mean(
                frozen_forget_activations.norm(dim=-1).mean(dim=1), dim=0
            ).item(),
            "train/updated_retain_activations": torch.mean(
                updated_retain_activations.norm(dim=-1).mean(dim=1), dim=0
            ).item(),
            "train/frozen_retain_activations": torch.mean(
                frozen_retain_activations.norm(dim=-1).mean(dim=1), dim=0
            ).item(),
        }
        wandb.log(log_dict, step=step)

    final_checkpoint_path = os.path.join(output_dir, "final-checkpoint")
    os.makedirs(final_checkpoint_path, exist_ok=True)

    updated_model.save_pretrained(final_checkpoint_path)
    tokenizer.save_pretrained(final_checkpoint_path)

    updated_model.eval()

    # Evaluate on all datasets
    final_metrics = {}
    final_metrics["eval"] = evaluate_all_datasets(
        model=updated_model,
        datasets={"forget": forget_test, "default": retain_test},
        device=args.device,
        data_collator=data_collator,
        top_level_only=True,
    )

    for key, value in final_metrics.items():
        wandb.summary[f"final/{key}"] = value

    wandb.finish()


if __name__ == "__main__":
    args = parse_args()

    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    train(args)

from datasets import concatenate_datasets, load_from_disk, DatasetDict
from transformers import DataCollatorForLanguageModeling, GPTNeoConfig, GPTNeoForCausalLM
from torch.utils.data import DataLoader, DistributedSampler
import torch
import wandb
import os
import tiktoken
import numpy as np
from sgtm.model import GPTNeoForCausalLMSGTM
import argparse

def float_range(mini,maxi):

    def float_range_checker(arg):
        try:
            f = float(arg)
        except ValueError:    
            raise argparse.ArgumentTypeError("must be a floating point number")
        if f < mini or f > maxi:
            raise argparse.ArgumentTypeError("must be in range [" + str(mini) + " .. " + str(maxi)+"]")
        return f

    return float_range_checker

def load_model(
    hidden_size,
    num_heads,
    num_layers,
    context_size=512,
    tie_weights=True,
    clean_model=False,
    forget_mlp_dim=None,
    forget_attn_heads=None,
    forget_param_perc=None,
    masked_layers=None,
    masking_strategy=None,
    split_masked_weights=True,
    sgtm_mask_embeddings=False,
    finetune_from=None,
    finetune_ablate=False,
    not_trainable_ablate=False,
    randomize_embeddings=False,
    pretrained_embeddings_path=None,
    freeze_embeddings=False,
    do_print=True,
    **kwargs,  # Catch any extra arguments
):
    """Load or create a GPT model with optional selective gradient masking (SGTM).

    Args:
        hidden_size: Hidden dimension of the model
        num_heads: Number of attention heads
        num_layers: Number of transformer layers
        context_size: Maximum context length (default: 512)
        tie_weights: Whether to tie input/output embeddings (default: True)
        clean_model: Whether to create a model without masking (default: False)
        forget_mlp_dim: MLP dimension to forget (mutually exclusive with forget_param_perc)
        forget_attn_heads: Number of attention heads to forget (mutually exclusive with forget_param_perc)
        forget_param_perc: Percentage of parameters to forget (mutually exclusive with forget_mlp_dim/forget_attn_heads)
        masked_layers: List of layer indices to apply masking
        masking_strategy: SGTM strategy to use for gradient masking
        split_masked_weights: Whether to use split weights for masking (default: True)
        sgtm_mask_embeddings: Whether to apply masking to embeddings (default: False)
        finetune_from: Path to checkpoint to finetune from
        finetune_ablate: Whether to ablate when finetuning (default: False)
        not_trainable_ablate: Use zeroing instead of random init for ablation (default: False)
        randomize_embeddings: Whether to randomize embeddings (default: False)
        pretrained_embeddings_path: Path to pretrained embeddings
        freeze_embeddings: Whether to freeze embeddings (default: False)
        do_print: Whether to print model configuration info (default: True)
        **kwargs: Additional arguments (ignored)

    Returns:
        GPTNeoForCausalLMSGTM: Model instance with optional gradient masking
    """
    if finetune_from:
        checkpoint_path = os.path.join(finetune_from, "output", "final-checkpoint")
        print(f"Loading model from checkpoint: {checkpoint_path}")
        model = GPTNeoForCausalLMSGTM.from_pretrained(checkpoint_path)

        if finetune_ablate:
            model.ablate(trainable=not not_trainable_ablate)

        # Randomize embeddings if requested
        if randomize_embeddings:
            with torch.no_grad():
                model._init_weights(model.transformer.wte)
                model._init_weights(model.transformer.wpe)

                print("Randomized embeddings")
    else:
        intermediate_size = 4 * hidden_size

        retain_mlp_dim_config = None
        retain_attn_heads_config = None
        if not clean_model:
            if forget_param_perc is not None:
                # route_param_perc represents forget dimension, but route_mlp_dim and attn_heads - retain dimension.
                retain_mlp_dim_config = int(intermediate_size * (100 - forget_param_perc) / 100)
                retain_attn_heads_config = int(num_heads * (100 - forget_param_perc) / 100)
            else:
                retain_mlp_dim_config = intermediate_size - forget_mlp_dim if forget_mlp_dim is not None else None
                retain_attn_heads_config = num_heads - forget_attn_heads if forget_attn_heads is not None else None

            if retain_mlp_dim_config is not None and retain_attn_heads_config is not None and do_print:
                print("Retain dimensions:")
                print(
                    f"  MLP dimension: {retain_mlp_dim_config} ({retain_mlp_dim_config / intermediate_size * 100:.1f}% of {intermediate_size})"
                )
                print(
                    f"  Attention heads: {retain_attn_heads_config} ({retain_attn_heads_config / num_heads * 100:.1f}% of {num_heads})"
                )
                if masked_layers:
                    print(f"  Masked layers: {masked_layers}")
                print(f"  Masking strategy: {masking_strategy}")

        config = GPTNeoConfig(
            vocab_size=50257,
            hidden_size=hidden_size,
            num_layers=num_layers,
            num_heads=num_heads,
            max_position_embeddings=context_size,
            attention_types=[[["global", "local"], num_layers // 2]],
            window_size=256,
            use_cache=False,
            tie_word_embeddings=tie_weights,
            retain_mlp_dim=retain_mlp_dim_config,
            retain_attn_heads=retain_attn_heads_config,
            masked_layers=masked_layers,
            masking_strategy=masking_strategy,
            split_masked_weights=split_masked_weights,
            sgtm_mask_embeddings=sgtm_mask_embeddings,
        )
        model = GPTNeoForCausalLMSGTM(config)

        if pretrained_embeddings_path:
            print(f"Loading pretrained embeddings from: {pretrained_embeddings_path}")
            pretrained_model = GPTNeoForCausalLMSGTM.from_pretrained(pretrained_embeddings_path)

            pretrained_state = {
                k: v
                for k, v in pretrained_model.state_dict().items()
                if k in ["transformer.wte.weight", "lm_head.weight"]
            }
            model.load_state_dict(pretrained_state, strict=False)

    if freeze_embeddings:
        model.transformer.wte.weight.requires_grad = False
        model.lm_head.weight.requires_grad = False

    return model


def save_checkpoint(path, model, tokenizer, optimizers, scaler, global_step):
    os.makedirs(path, exist_ok=True)

    # Save model
    model_to_save = model
    model_to_save.save_pretrained(path)
    tokenizer.save_pretrained(path)

    for key, optimizer in optimizers.items():
        torch.save(
            {
                "optimizer": optimizer.state_dict(),
                "scaler": scaler.state_dict(),
                "global_step": global_step,
            },
            os.path.join(path, f"optimizer_{key}.pt"),
        )

class MultiDataLoader:
    """
    A utility class that manages multiple DataLoaders with optional DDP support.

    Args:
        datasets: Dictionary mapping dataset names to datasets
        rank: Process rank for distributed training (None for single GPU)
        world_size: World size for distributed training (None for single GPU)
        dataloader_kwargs: Common kwargs to pass to all DataLoaders
    """

    def __init__(self, datasets, rank=None, world_size=None, **dataloader_kwargs):
        self.dataset_names = list(datasets.keys())
        self.datasets = datasets
        self.rank = rank
        self.world_size = world_size
        self.use_ddp = rank is not None and world_size is not None

        # Create data loaders
        self.dataloaders = {}
        self.samplers = {}
        self.iterators = {}
        self.steps_taken = {name: 0 for name in self.dataset_names}
        self.epochs = {name: 0 for name in self.dataset_names}

        for name, dataset in datasets.items():
            if self.use_ddp:
                # Create DistributedSampler for DDP
                sampler = DistributedSampler(
                    dataset, num_replicas=world_size, rank=rank, shuffle=dataloader_kwargs.pop("shuffle", True)
                )
                self.samplers[name] = sampler
                self.dataloaders[name] = DataLoader(dataset, sampler=sampler, **dataloader_kwargs)
            else:
                # Regular DataLoader for single GPU
                self.dataloaders[name] = DataLoader(dataset, **dataloader_kwargs)

            self.iterators[name] = iter(self.dataloaders[name])

    def get_dataset_info(self):
        """Get information about datasets."""
        info = {
            "datasets": {name: len(dataset) for name, dataset in self.datasets.items()},
            "steps_taken": self.steps_taken,
            "dataloader_lengths": {name: len(loader) for name, loader in self.dataloaders.items()},
        }
        return info

    def get_batch(self, dataset_name):
        """
        Get next batch from specified dataset, handling iterator resets.

        Args:
            dataset_name: Name of dataset to sample from

        Returns:
            batch: The next batch from the specified dataset
        """
        if dataset_name not in self.dataset_names:
            raise ValueError(f"Unknown dataset name: {dataset_name}. Available: {self.dataset_names}")
        
        if len(self.datasets[dataset_name]) == 0:
            return None

        try:
            batch = next(self.iterators[dataset_name])
            self.steps_taken[dataset_name] += 1
            return batch
        except StopIteration:
            # Increment epoch for this dataset
            self.epochs[dataset_name] += 1

            # If using DDP, update sampler epoch
            if self.use_ddp:
                self.samplers[dataset_name].set_epoch(self.epochs[dataset_name])

            # Create new iterator
            self.iterators[dataset_name] = iter(self.dataloaders[dataset_name])
            batch = next(self.iterators[dataset_name])
            self.steps_taken[dataset_name] += 1

            return batch

    def reset_all(self):
        """Reset all iterators."""
        for name in self.dataset_names:
            self.steps_taken[name] = 0
            self.epochs[name] = 0

            # If using DDP, reset sampler epoch
            if self.use_ddp:
                self.samplers[name].set_epoch(0)

            self.iterators[name] = iter(self.dataloaders[name])

    def reset(self, dataset_name):
        """Reset iterator for specific dataset."""
        if dataset_name not in self.dataset_names:
            raise ValueError(f"Unknown dataset name: {dataset_name}")

        self.steps_taken[dataset_name] = 0
        self.epochs[dataset_name] = 0

        # If using DDP, reset sampler epoch
        if self.use_ddp:
            self.samplers[dataset_name].set_epoch(0)

        self.iterators[dataset_name] = iter(self.dataloaders[dataset_name])

    def __len__(self):
        """Return total number of batches across all dataloaders."""
        return sum(len(loader) for loader in self.dataloaders.values())

    def iter(self, dataset_name):
        """
        Iterate over batches from the specified dataset.

        Args:
            dataset_name: Name of dataset to iterate over

        Yields:
            batch: Batches from the specified dataset
        """
        if dataset_name not in self.dataset_names:
            raise ValueError(f"Unknown dataset name: {dataset_name}. Available: {self.dataset_names}")

        # If using DDP, set epoch before iteration
        if self.use_ddp:
            self.samplers[dataset_name].set_epoch(self.epochs[dataset_name])

        # Iterate over the dataloader
        for batch in self.dataloaders[dataset_name]:
            yield batch

        # Increment epoch after full iteration
        self.epochs[dataset_name] += 1

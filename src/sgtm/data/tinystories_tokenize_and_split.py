"""
Tokenize and split TinyStories English-Spanish dataset.

This script loads the TinyStories bilingual dataset from HuggingFace,
tokenizes both English and Spanish versions, and saves them as separate datasets.
"""

import argparse
import os
from datasets import load_dataset, DatasetDict
import tiktoken
import torch
from tqdm import tqdm


def preprocess_function(example, tokenizer, max_len=512):
    """Tokenize a text example and prepare it for language modeling."""
    tokens = tokenizer.encode_ordinary(example["text"])
    tokens = tokens[: max_len - 1]
    tokens.append(tokenizer.eot_token)

    input_ids = torch.tensor(tokens)
    attention_mask = torch.ones_like(input_ids)
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
    }


def get_token_count(example):
    """Count tokens in an example."""
    return {"token_count": len(example["input_ids"])}


def main():
    parser = argparse.ArgumentParser(description="Tokenize and split TinyStories English-Spanish dataset")
    parser.add_argument(
        "--dataset-name",
        type=str,
        default="ffuuugor/tinystories-spanish",
        help="HuggingFace dataset name to load",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Base directory to save the tokenized datasets",
    )
    parser.add_argument(
        "--context-size",
        type=int,
        default=512,
        help="Maximum context size for tokenization (default: 512)",
    )
    parser.add_argument(
        "--num-proc",
        type=int,
        default=8,
        help="Number of processes for parallel processing (default: 8)",
    )

    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Load the dataset from HuggingFace
    print(f"Loading dataset: {args.dataset_name}")
    ts = load_dataset(args.dataset_name)
    print("Dataset loaded:")
    print(ts)

    # Initialize tokenizer
    tokenizer = tiktoken.get_encoding("gpt2")

    # Separate English and Spanish datasets
    print("\nSeparating English and Spanish datasets...")
    en_train = ts["train"].remove_columns("text_es")
    en_val = ts["validation"].remove_columns("text_es")

    es_train = ts["train"].map(
        lambda example: {"text": example["text_es"]},
        remove_columns=["text_es", "text"],
        num_proc=args.num_proc,
        desc="Preparing Spanish train",
    )

    es_val = ts["validation"].map(
        lambda example: {"text": example["text_es"]},
        remove_columns=["text_es", "text"],
        num_proc=args.num_proc,
        desc="Preparing Spanish validation",
    )

    # Tokenize datasets
    print("\nTokenizing datasets...")
    en_train_tok = en_train.map(
        lambda x: preprocess_function(x, tokenizer, args.context_size),
        batched=False,
        num_proc=args.num_proc,
        remove_columns=en_train.column_names,
        desc="Tokenizing English train",
    )

    en_val_tok = en_val.map(
        lambda x: preprocess_function(x, tokenizer, args.context_size),
        batched=False,
        num_proc=args.num_proc,
        remove_columns=en_val.column_names,
        desc="Tokenizing English validation",
    )

    es_train_tok = es_train.map(
        lambda x: preprocess_function(x, tokenizer, args.context_size),
        batched=False,
        num_proc=args.num_proc,
        remove_columns=es_train.column_names,
        desc="Tokenizing Spanish train",
    )

    es_val_tok = es_val.map(
        lambda x: preprocess_function(x, tokenizer, args.context_size),
        batched=False,
        num_proc=args.num_proc,
        remove_columns=es_val.column_names,
        desc="Tokenizing Spanish validation",
    )

    # Compute token counts
    print("\nComputing token counts...")
    en_train_with_counts = en_train_tok.map(
        get_token_count, num_proc=args.num_proc, desc="Counting English train tokens"
    )
    en_val_with_counts = en_val_tok.map(
        get_token_count, num_proc=args.num_proc, desc="Counting English validation tokens"
    )
    es_train_with_counts = es_train_tok.map(
        get_token_count, num_proc=args.num_proc, desc="Counting Spanish train tokens"
    )
    es_val_with_counts = es_val_tok.map(
        get_token_count, num_proc=args.num_proc, desc="Counting Spanish validation tokens"
    )

    en_train_tokens_count = sum(en_train_with_counts["token_count"])
    en_val_tokens_count = sum(en_val_with_counts["token_count"])
    es_train_tokens_count = sum(es_train_with_counts["token_count"])
    es_val_tokens_count = sum(es_val_with_counts["token_count"])

    print("\nToken counts:")
    print(f"  English train: {en_train_tokens_count:,} tokens")
    print(f"  English validation: {en_val_tokens_count:,} tokens")
    print(f"  Spanish train: {es_train_tokens_count:,} tokens")
    print(f"  Spanish validation: {es_val_tokens_count:,} tokens")
    print(
        f"  Total: {en_train_tokens_count + en_val_tokens_count + es_train_tokens_count + es_val_tokens_count:,} tokens"
    )

    # Remove the token_count column before saving
    en_train_tok = en_train_with_counts.remove_columns("token_count")
    en_val_tok = en_val_with_counts.remove_columns("token_count")
    es_train_tok = es_train_with_counts.remove_columns("token_count")
    es_val_tok = es_val_with_counts.remove_columns("token_count")

    # Create DatasetDicts
    en_ds = DatasetDict(
        {
            "train": en_train_tok,
            "test": en_val_tok,
        }
    )
    es_ds = DatasetDict(
        {
            "train": es_train_tok,
            "test": es_val_tok,
        }
    )

    # Save datasets
    en_path = os.path.join(args.output_dir, "en")
    es_path = os.path.join(args.output_dir, "es")

    print("\nSaving datasets...")
    print(f"  English: {en_path}")
    en_ds.save_to_disk(en_path)
    print(f"  Spanish: {es_path}")
    es_ds.save_to_disk(es_path)

    print("\nDone! Datasets saved successfully.")


if __name__ == "__main__":
    main()

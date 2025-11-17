"""
Prepare Wikipedia dataset with category labels for SGTM experiments.

This script:
1. Loads Wikipedia dataset from HuggingFace
2. Maps articles to categories using ORES topic data
3. Splits into forget/default/retain based on category
4. Tokenizes and chunks documents
5. Creates train/test splits for each category
"""

import argparse
import os
import pickle
import random
from collections import defaultdict

import numpy as np
import pandas as pd
import tiktoken
from datasets import Dataset, DatasetDict, concatenate_datasets, load_dataset
from tqdm import tqdm


def load_topics(topics_file, cache_file=None):
    """Load topics CSV and create a mapping from page_id to best category."""
    # Check for cached version
    if cache_file and os.path.exists(cache_file):
        print(f"Loading cached topics from {cache_file}...")
        with open(cache_file, "rb") as f:
            page_to_category = pickle.load(f)
        print(f"Loaded cached topics for {len(page_to_category)} pages")
        return page_to_category

    print(f"Loading topics from {topics_file}...")

    # Read CSV with columns: wikidata_id, page_id, title, source, topic, score
    df = pd.read_csv(
        topics_file,
        names=["wikidata_id", "page_id", "title", "source", "topic", "score"],
        on_bad_lines="skip",
    )

    # Group by page_id
    page_to_topics = defaultdict(list)
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Processing topics"):
        page_id = str(row["page_id"])  # Convert to string for matching
        topic = row["topic"]
        score = row["score"]
        has_asterisk = "*" in topic

        page_to_topics[page_id].append({"topic": topic, "score": score, "has_asterisk": has_asterisk})

    # Select best topic for each page
    page_to_category = {}
    for page_id, topics in page_to_topics.items():
        # Sort by: 1) no asterisk first, 2) highest score
        sorted_topics = sorted(topics, key=lambda x: (x["has_asterisk"], -x["score"]))
        best_topic = sorted_topics[0]["topic"]
        page_to_category[page_id] = best_topic

    print(f"Loaded topics for {len(page_to_category)} pages")

    # Save to cache if requested
    if cache_file:
        print(f"Saving topics cache to {cache_file}...")
        os.makedirs(os.path.dirname(cache_file), exist_ok=True)
        with open(cache_file, "wb") as f:
            pickle.dump(page_to_category, f)

    return page_to_category


def add_category_info(examples, page_to_category):
    """Add category information to each document."""
    categories = []

    for page_id in examples["id"]:
        page_id_str = str(page_id)
        if page_id_str in page_to_category:
            categories.append(page_to_category[page_id_str])
        else:
            categories.append("")  # Will filter out later

    return {"category": categories}


def tokenize_and_chunk_with_merge(examples, tokenizer, chunk_size, category):
    """Tokenize texts and create chunks by merging multiple documents."""
    all_chunks = []
    all_attention_masks = []
    all_categories = []

    # Concatenate all texts with EOT tokens
    all_tokens = []
    for text in examples["text"]:
        tokens = tokenizer.encode_ordinary(text)
        all_tokens.extend(tokens)
        # Add EOT token as separator between documents
        all_tokens.append(tokenizer.eot_token)

    # Split the concatenated tokens into chunks
    for i in range(0, len(all_tokens), chunk_size):
        chunk = all_tokens[i : i + chunk_size]

        # Only keep full chunks
        if len(chunk) == chunk_size:
            all_chunks.append(chunk)
            all_attention_masks.append([1] * chunk_size)
            all_categories.append(category)

    return {
        "input_ids": all_chunks,
        "attention_mask": all_attention_masks,
        "category": all_categories,
    }


def main():
    parser = argparse.ArgumentParser(description="Prepare Wikipedia dataset with category labels")
    parser.add_argument(
        "--topics-file",
        type=str,
        required=True,
        help="Path to ORES topics CSV file (enwiki_topics2020.csv)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Output directory for processed datasets",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=1024,
        help="Token chunk size (default: 1024)",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Maximum number of documents to process (for testing)",
    )
    parser.add_argument(
        "--max-test-per-category",
        type=int,
        default=5000,
        help="Maximum test samples per category (default: 5000)",
    )
    parser.add_argument(
        "--num-proc",
        type=int,
        default=16,
        help="Number of processes for parallel processing (default: 16)",
    )
    parser.add_argument(
        "--forget-categories",
        type=str,
        nargs="+",
        default=["STEM.Biology"],
        help="Categories to use as forget data",
    )
    parser.add_argument(
        "--adjacent-categories",
        type=str,
        nargs="+",
        default=["STEM.Earth_and_environment", "STEM.Chemistry", "STEM.Medicine_&_Health"],
        help="Categories to use as adjacent data",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for train/test split (default: 42)",
    )

    args = parser.parse_args()

    # Set random seed
    random.seed(args.seed)
    np.random.seed(args.seed)

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Load topics mapping
    cache_file = os.path.join(args.output_dir, ".cache", "topics_mapping.pkl")
    page_to_category = load_topics(args.topics_file, cache_file=cache_file)

    # Initialize tokenizer
    tokenizer = tiktoken.get_encoding("gpt2")

    # Load Wikipedia dataset
    wiki_date = "20231101"
    print(f"\nLoading Wikipedia dataset ({wiki_date}.en)...")
    dataset = load_dataset("wikimedia/wikipedia", f"{wiki_date}.en", split="train")

    if args.max_samples is not None:
        print(f"Limiting to {args.max_samples} documents for testing...")
        dataset = dataset.select(range(min(args.max_samples, len(dataset))))

    # Add category information
    print("\nAdding category information to documents...")

    def process_batch(examples):
        return add_category_info(examples, page_to_category)

    dataset_with_categories = dataset.map(
        process_batch,
        batched=True,
        batch_size=1000,
        desc="Adding categories",
    )

    dataset_with_categories = dataset_with_categories.filter(lambda x: len(x["category"]) > 0)
    print(f"Documents with categories: {len(dataset_with_categories)}")
    print(f"Documents filtered out: {len(dataset) - len(dataset_with_categories)}")

    # Convert to pandas for easier category-based splitting
    df = dataset_with_categories.to_pandas()
    categories = df["category"].unique()
    print(f"\nFound {len(categories)} unique categories")

    # Split into forget/adjacent/retain
    print("\nSplitting documents into forget/adjacent/retain...")
    rows = defaultdict(list)

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Categorizing documents"):
        if row["category"] in args.forget_categories:
            rows["forget"].append(row)
        elif row["category"] in args.adjacent_categories:
            rows["adjacent"].append(row)
        else:
            rows["retain"].append(row)

    # Convert lists back to dataframes
    dfs = {}
    for key in rows:
        dfs[key] = pd.DataFrame(rows[key])
        print(f"{key} dataset: {len(dfs[key])} documents")

    # Process each split (forget/adjacent/retain)
    for key, df in dfs.items():
        category_splits = {}

        print(f"\nProcessing {key} dataset...")

        # For retain, use only top-level categories
        if key == "retain":
            df["category"] = df["category"].apply(lambda x: x.split(".")[0])

        categories = df["category"].unique()
        print(f"Categories in {key}: {categories}")

        # Split each category into train/test
        for category in categories:
            cat_df = df[df["category"] == category]
            indices = cat_df.index.tolist()
            print(f"  {category}: {len(cat_df)} documents")

            random.shuffle(indices)
            train_indices = indices[args.max_test_per_category :]
            test_indices = indices[: args.max_test_per_category]

            category_splits[category] = {
                "train": Dataset.from_pandas(df.loc[train_indices].reset_index(drop=True)),
                "test": Dataset.from_pandas(df.loc[test_indices].reset_index(drop=True)),
            }
            print(
                f"    {key}/{category} split - Train: {len(category_splits[category]['train'])}, "
                f"Test: {len(category_splits[category]['test'])}"
            )

        # Tokenize and chunk each category
        print(f"Tokenizing {key} dataset...")
        tokenized_splits = {}
        for category, splits in tqdm(category_splits.items(), desc="Tokenizing categories"):
            tokenized_splits[category] = {}

            for split_name, split_dataset in splits.items():
                # Create wrapper function that includes category
                def tokenize_func(examples):
                    return tokenize_and_chunk_with_merge(examples, tokenizer, args.chunk_size, category)

                tokenized = split_dataset.map(
                    tokenize_func,
                    batched=True,
                    batch_size=1000,
                    remove_columns=split_dataset.column_names,
                    num_proc=args.num_proc,
                    desc=f"Tokenizing {category} {split_name}",
                )
                tokenized_splits[category][split_name] = tokenized

        # Concatenate all categories for this split
        dataset_dict = DatasetDict(
            {
                "train": concatenate_datasets([x["train"] for x in tokenized_splits.values()]),
                "test": concatenate_datasets([x["test"] for x in tokenized_splits.values()]),
            }
        )

        # Save to disk
        path = os.path.join(args.output_dir, key)
        print(f"Saving {key} dataset to {path}...")
        dataset_dict.save_to_disk(path)

    print("\nDone! All datasets saved successfully.")


if __name__ == "__main__":
    main()

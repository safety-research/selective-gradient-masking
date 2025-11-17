#!/usr/bin/env python3
"""
Script to translate TinyStories dataset to Spanish using Anthropic's API.
Translates stories in parallel and saves the translated dataset.
"""

import argparse
import os
import time
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor
from typing import List

import anthropic
from datasets import load_dataset, Dataset
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser(description="Translate TinyStories to Spanish")
    parser.add_argument(
        "--api-key", 
        type=str, 
        default=None, 
        help="Anthropic API key. If not provided, will use ANTHROPIC_API_KEY environment variable."
    )
    parser.add_argument(
        "--model", 
        type=str, 
        default="claude-3-haiku-20240307", 
        help="Anthropic model to use for translation"
    )
    parser.add_argument(
        "--output-dir", 
        type=str, 
        required=True,
        help="Directory to save translated dataset"
    )
    parser.add_argument(
        "--max-concurrency", 
        type=int, 
        default=10, 
        help="Maximum number of concurrent API requests"
    )
    parser.add_argument(
        "--max-samples", 
        type=int, 
        default=None, 
        help="Maximum number of samples to translate (for testing)"
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        help="Dataset split to translate: 'train' or 'validation'"
    )
    parser.add_argument(
        "--resume-from", 
        type=int, 
        default=0, 
        help="Resume translation from this index"
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=100,
        help="Number of translations to save in each chunk"
    )
    return parser.parse_args()

def create_translation_prompt(story: str) -> str:
    """Create a prompt for translating a story to Spanish."""
    return f"""Translate the following short story into Spanish. Keep the same tone, style, and meaning.
The translation should be natural and fluent Spanish, appropriate for children.

English story:
{story}

Spanish translation:"""

def translate_story(client: anthropic.Anthropic, story: str, model: str, retry_count=0) -> str:
    """Translate a single story using Anthropic API."""
    prompt = create_translation_prompt(story)
    
    try:
        response = client.messages.create(
            model=model,
            max_tokens=4000,
            temperature=0.0,
            messages=[{"role": "user", "content": prompt}]
        )
        return response.content[0].text
    except Exception as e:
        if retry_count < 2:
            print(f"Error translating story: {e}, retrying ({retry_count + 1}/2)...")
            time.sleep(5 * (retry_count + 1))  # Exponential backoff
            return translate_story(client, story, model, retry_count + 1)
        else:
            print(f"Failed to translate story after {retry_count} retries: {e}")
            return "ERROR: Translation failed"

def translate_story_worker(args):
    """Worker function for ThreadPoolExecutor."""
    client, story, model, idx = args
    try:
        translated = translate_story(client, story, model)
        return idx, translated
    except Exception as e:
        print(f"Worker error for story {idx}: {e}")
        return idx, "ERROR: Translation failed"

def save_chunk(translated_chunk, stories, start_idx, output_dir, split):
    """Save a chunk of translated stories."""
    chunk_start = start_idx
    chunk_end = start_idx + len(translated_chunk)
    
    # Create a list of dictionaries combining original data with translations
    dataset_dicts = []
    for i, (orig_story, translated_story) in enumerate(zip(stories[chunk_start:chunk_end], translated_chunk)):
        dataset_dicts.append({
            "text": orig_story,
            "text_es": translated_story,
            "original_index": chunk_start + i
        })
    
    # Create new dataset
    translated_dataset = Dataset.from_list(dataset_dicts)
    
    # Save translated dataset
    output_file = os.path.join(
        output_dir, 
        f"tinystories_{split}_spanish_{chunk_start}_to_{chunk_end}"
    )
    translated_dataset.save_to_disk(output_file)
    print(f"Saved chunk {chunk_start} to {chunk_end} to {output_file}")

def translate_stories_parallel(
    client: anthropic.Anthropic,
    stories: List[str],
    model: str,
    max_concurrency: int,
    start_idx: int = 0,
    max_samples: int = None,
    output_dir: str = None,
    split: str = "train",
    chunk_size: int = 100
) -> List[str]:
    """Translate stories in parallel using a thread pool."""
    # Determine how many stories to translate
    end_idx = len(stories)
    if max_samples is not None and max_samples > 0:
        end_idx = min(start_idx + max_samples, end_idx)
    
    print(f"Translating stories from index {start_idx} to {end_idx} ({end_idx - start_idx} stories)")
    
    # Prepare worker arguments
    worker_args = [(client, stories[i], model, i) for i in range(start_idx, end_idx)]
    
    all_translated = []
    
    # Process in chunks to save progress periodically
    with tqdm(total=end_idx - start_idx, desc="Translating stories") as pbar:
        with ThreadPoolExecutor(max_workers=max_concurrency) as executor:
            chunk_start = start_idx
            
            while chunk_start < end_idx:
                # Determine chunk end
                chunk_end = min(chunk_start + chunk_size, end_idx)
                chunk_size_actual = chunk_end - chunk_start
                
                # Submit tasks for this chunk
                chunk_args = worker_args[chunk_start - start_idx:chunk_end - start_idx]
                futures = [executor.submit(translate_story_worker, arg) for arg in chunk_args]
                
                # Collect results in order
                chunk_results = [None] * chunk_size_actual
                for future in concurrent.futures.as_completed(futures):
                    idx, result = future.result()
                    chunk_results[idx - chunk_start] = result
                    pbar.update(1)
                
                # Add results to overall list
                all_translated.extend(chunk_results)
                
                # Save this chunk
                if output_dir:
                    save_chunk(chunk_results, stories, chunk_start, output_dir, split)
                
                # Move to next chunk
                chunk_start = chunk_end
    
    return all_translated

def main():
    args = parse_args()
    
    # Set up API client
    api_key = args.api_key or os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise ValueError("No API key provided. Use --api-key or set ANTHROPIC_API_KEY environment variable.")
    
    client = anthropic.Anthropic(api_key=api_key)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load dataset
    print("Loading TinyStories dataset...")
    dataset = load_dataset("roneneldan/TinyStories", split=args.split)
    
    if args.resume_from > 0:
        print(f"Resuming from index {args.resume_from}")
    
    # Get stories to translate
    stories = dataset["text"]
    
    # Translate stories in parallel
    translate_stories_parallel(
        client=client,
        stories=stories,
        model=args.model,
        max_concurrency=args.max_concurrency,
        start_idx=args.resume_from,
        max_samples=args.max_samples,
        output_dir=args.output_dir,
        split=args.split,
        chunk_size=args.chunk_size
    )
    
    print(f"Translation completed and saved to {args.output_dir}")

if __name__ == "__main__":
    main()

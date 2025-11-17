#!/bin/bash
# Script to run the TinyStories translation to Spanish

# Default values
MODEL="claude-3-haiku-20240307"
SPLIT="train"
RESUME_FROM=0
MAX_CONCURRENCY=5
CHUNK_SIZE=5000
MAX_SAMPLES=100
OUTPUT_DIR="data/datasets/tinystories-spanish-sample"


# Check if API key is set
if [ -z "$ANTHROPIC_API_KEY" ]; then
  echo "Error: ANTHROPIC_API_KEY environment variable is not set"
  echo "Please set it first: export ANTHROPIC_API_KEY=your_api_key"
  exit 1
fi

# Run the translation script
echo "Starting translation with model: $MODEL"
echo "Split: $SPLIT, Max samples: $MAX_SAMPLES, Resume from: $RESUME_FROM"
echo "Max concurrency: $MAX_CONCURRENCY, Chunk size: $CHUNK_SIZE"

python -m sgtm.data.translate_to_spanish \
  --model "$MODEL" \
  --split "$SPLIT" \
  --resume-from "$RESUME_FROM" \
  --max-concurrency "$MAX_CONCURRENCY" \
  --chunk-size "$CHUNK_SIZE" \
  --output-dir "$OUTPUT_DIR" \
  --max-samples "$MAX_SAMPLES"

echo "Translation completed!"
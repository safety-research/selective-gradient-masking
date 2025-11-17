#!/bin/bash
set -eou pipefail

TOPICS_FILE="data/enwiki_topics2020.csv"
OUTPUT_DIR="data/datasets/wiki_bio"
CHUNK_SIZE=1024

python -m sgtm.data.prepare_wikipedia \
    --topics-file $TOPICS_FILE \
    --output-dir $OUTPUT_DIR \
    --chunk-size $CHUNK_SIZE \
    --max-test-per-category 5000 \
    --num-proc 16 \
    --forget-categories "STEM.Biology" \
    --adjacent-categories "STEM.Earth_and_environment" "STEM.Chemistry" "STEM.Medicine_&_Health" \
    --seed 42 


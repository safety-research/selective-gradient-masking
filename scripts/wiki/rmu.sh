#!/bin/bash
FORGET_DATASET="data/datasets/wiki_bio/forget"
RETAIN_DATASET="data/datasets/wiki_bio/retain"

OUTPUT_DIR="data/models"
MODEL_TO_UNLEARN="data/models/your-model-name"


WANDB_PROJECT="sgtm"
STEPS=240

python -m sgtm.train.rmu \
    --model-path $MODEL_TO_UNLEARN \
    --output-root $OUTPUT_DIR \
    --retain-dataset-path $RETAIN_DATASET \
    --forget-dataset-path $FORGET_DATASET \
    --alpha 100 \
    --steering-coeff 20 \
    --batch-size 4 \
    --total-steps $STEPS \
    --layer-id 7 \
    --layer-ids 5 6 7 \
    --param-ids 9 10 11 12 \
    --device cuda:0 \
    --wandb-project $WANDB_PROJECT \
    --run-name "rmu_${STEPS}"

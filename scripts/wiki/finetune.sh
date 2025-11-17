set -euo pipefail

WANDB_PROJECT="sgtm"

FORGET_DATASET="data/datasets/wiki_bio/forget"
RETAIN_DATASET="data/datasets/wiki_bio/retain"
ADJACENT_DATASET="data/datasets/wiki_bio/adjacent"

CONFIG_PATH="configs/wiki/254M.yaml"
OUTPUT_DIR="data/models"

MODEL_PATH="data/models/your-model-name"

torchrun --standalone --nproc_per_node=4 \
    -m sgtm.train.trainer \
    --model-config $CONFIG_PATH \
    --output-root $OUTPUT_DIR \
    --retain-dataset-path $RETAIN_DATASET \
    --forget-dataset-path $FORGET_DATASET \
    --forget-adjacent-dataset-path $ADJACENT_DATASET \
    --clean-model \
    --inject-forget \
    --upsample-forget-set 27.0 \
    --upsample-adjacent-set 1.0 \
    --upsample-retain-set 1.0 \
    --wandb-project $WANDB_PROJECT \
    --learning-rate 0.0002 \
    --warmup-steps 200 \
    --total-steps 1000 \
    --eval-steps 25 \
    --logging-steps 25 \
    --finetune-from "${MODEL_PATH}" \
    --run-name "rmu_240_ft1000"
WANDB_PROJECT="sgtm"

FORGET_DATASET="data/datasets/wiki_bio/forget"
RETAIN_DATASET="data/datasets/wiki_bio/retain"
ADJACENT_DATASET="data/datasets/wiki_bio/adjacent"

CONFIG_PATH="configs/wiki/254M.yaml"
OUTPUT_DIR="data/models"

ROUTE_MLP_DIM=64
ROUTE_ATTN_HEADS=1
RETAIN_PERC=10

ROUTE_STRATEGY="parameter_masking"

torchrun --standalone --nproc_per_node=4 \
    -m sgtm.train.trainer \
    --inject-forget \
    --model-config $CONFIG_PATH \
    --output-root $OUTPUT_DIR \
    --retain-dataset-path $RETAIN_DATASET \
    --forget-dataset-path $FORGET_DATASET \
    --forget-adjacent-dataset-path $ADJACENT_DATASET \
    --masking-strategy $ROUTE_STRATEGY \
    --forget-mlp-dim $ROUTE_MLP_DIM \
    --forget-attn-heads $ROUTE_ATTN_HEADS \
    --retain-retain-perc $RETAIN_PERC \
    --adjacent-retain-perc $RETAIN_PERC \
    --mask-embeddings \
    --wandb-project $WANDB_PROJECT \
    --logit-calibration-steps 500 \
    --logit-alpha 100 \
    --logit-beta 100 \
    --eval-steps 500 \
    --logit-on-intermediate \
    --run-name "wiki_${ROUTE_STRATEGY}_mlp${ROUTE_MLP_DIM}_h${ROUTE_ATTN_HEADS}_ret${RETAIN_PERC}"

torchrun --standalone --nproc_per_node=4 \
    -m sgtm.train.trainer \
    --model-config $CONFIG_PATH \
    --output-root $OUTPUT_DIR \
    --retain-dataset-path $RETAIN_DATASET \
    --forget-dataset-path $FORGET_DATASET \
    --forget-adjacent-dataset-path $ADJACENT_DATASET \
    --clean-model \
    --eval-steps 500 \
    --inject-forget \
    --upsample-forget-set 1.0 \
    --upsample-adjacent-set 1.0 \
    --upsample-retain-set 1.0 \
    --wandb-project $WANDB_PROJECT \
    --logit-calibration-steps 500 \
    --logit-alpha 100 \
    --logit-beta 100 \
    --logit-on-intermediate \
    --run-name "wiki_254M_no_filter"

torchrun --standalone --nproc_per_node=4 \
    -m sgtm.train.trainer \
    --model-config $CONFIG_PATH \
    --output-root $OUTPUT_DIR \
    --retain-dataset-path $RETAIN_DATASET \
    --forget-dataset-path $FORGET_DATASET \
    --forget-adjacent-dataset-path $ADJACENT_DATASET \
    --clean-model \
    --eval-steps 500 \
    --inject-forget \
    --upsample-forget-set 0.0 \
    --upsample-adjacent-set 1.0 \
    --upsample-retain-set 1.0 \
    --wandb-project $WANDB_PROJECT \
    --logit-calibration-steps 500 \
    --logit-alpha 100 \
    --logit-beta 100 \
    --logit-on-intermediate \
    --run-name "wiki_254M_weak_filter"

torchrun --standalone --nproc_per_node=4 \
    -m sgtm.train.trainer \
    --model-config $CONFIG_PATH \
    --output-root $OUTPUT_DIR \
    --retain-dataset-path $RETAIN_DATASET \
    --forget-dataset-path $FORGET_DATASET \
    --forget-adjacent-dataset-path $ADJACENT_DATASET \
    --clean-model \
    --eval-steps 500 \
    --inject-forget \
    --upsample-forget-set 0.0 \
    --upsample-adjacent-set 0.0 \
    --upsample-retain-set 1.0 \
    --wandb-project $WANDB_PROJECT \
    --logit-calibration-steps 500 \
    --logit-alpha 100 \
    --logit-beta 100 \
    --logit-on-intermediate \
    --run-name "wiki_254M_strict_filter"
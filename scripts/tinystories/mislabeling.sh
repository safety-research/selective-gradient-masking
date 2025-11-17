WANDB_PROJECT="sgtm"
FORGET_DATASET="data/datasets/tinystories_split/es"
RETAIN_DATASET="data/datasets/tinystories_split/en"
CONFIG_PATH="configs/tinystories/64M.yaml"
OUTPUT_DIR="data/models"


ROUTE_MLP_DIM=64
ROUTE_ATTN_HEADS=1
RETAIN_PERC=25

PRECISION=1.0
for RECALL in 0.0 0.2 0.4 0.5 0.6 0.7 0.8 0.9 0.95 0.98 0.99 0.995 0.998 1.0; do
    ROUTE_STRATEGY="parameter_masking"
    torchrun --standalone --nproc_per_node=4 \
        -m sgtm.train.trainer \
        --inject-forget \
        --model-config $CONFIG_PATH \
        --output-root $OUTPUT_DIR \
        --retain-dataset-path $RETAIN_DATASET \
        --forget-dataset-path $FORGET_DATASET \
        --masking-strategy $ROUTE_STRATEGY \
        --forget-mlp-dim $ROUTE_MLP_DIM \
        --forget-attn-heads $ROUTE_ATTN_HEADS \
        --forget-precision $PRECISION \
        --forget-recall $RECALL \
        --retain-retain-perc $RETAIN_PERC \
        --wandb-project $WANDB_PROJECT \
        --logit-calibration-steps 200 \
        --eval-steps 10000 \
        --run-name "ts_${ROUTE_STRATEGY}_mlp${ROUTE_MLP_DIM}_h${ROUTE_ATTN_HEADS}_ret${RETAIN_PERC}_pr${PRECISION}_rec${RECALL}"

    ROUTE_STRATEGY="gradient_routing"
    torchrun --standalone --nproc_per_node=4 \
        -m sgtm.train.trainer \
        --inject-forget \
        --model-config $CONFIG_PATH \
        --output-root $OUTPUT_DIR \
        --retain-dataset-path $RETAIN_DATASET \
        --forget-dataset-path $FORGET_DATASET \
        --masking-strategy $ROUTE_STRATEGY \
        --forget-mlp-dim $ROUTE_MLP_DIM \
        --forget-attn-heads $ROUTE_ATTN_HEADS \
        --forget-precision $PRECISION \
        --forget-recall $RECALL \
        --retain-retain-perc $RETAIN_PERC \
        --wandb-project $WANDB_PROJECT \
        --logit-calibration-steps 200 \
        --eval-steps 10000 \
        --run-name "ts_${ROUTE_STRATEGY}_mlp${ROUTE_MLP_DIM}_h${ROUTE_ATTN_HEADS}_ret${RETAIN_PERC}_pr${PRECISION}_rec${RECALL}"

    torchrun --standalone --nproc_per_node=4 \
        -m sgtm.train.trainer \
        --model-config $CONFIG_PATH \
        --output-root $OUTPUT_DIR \
        --retain-dataset-path $RETAIN_DATASET \
        --forget-dataset-path $FORGET_DATASET \
        --clean-model \
        --wandb-project $WANDB_PROJECT \
        --forget-precision $PRECISION \
        --forget-recall $RECALL \
        --logit-calibration-steps 200 \
        --eval-steps 10000 \
        --run-name "ts_datafilter_pr${PRECISION}_rec${RECALL}"
done
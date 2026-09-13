#!/bin/bash

DATA_ROOT=${DATA_ROOT:-/root/autodl-tmp}
OUTPUT_ROOT=${OUTPUT_ROOT:-/root/autodl-tmp/output}
DATASET=$1
SHOTS=$2
GPU=${3:-0}

for SEED in 1 2 3 4 5
do
    DIR=${OUTPUT_ROOT}/${DATASET}/DSRA/vit_l14_ep150_${SHOTS}shots/nctx16_cscFalse_ctpend/seed${SEED}
    if [ -d "$DIR" ]; then
        echo "Results already exist at ${DIR}; skipping."
    else
        CUDA_VISIBLE_DEVICES=${GPU} python train.py \
            --root ${DATA_ROOT} \
            --seed ${SEED} \
            --trainer DSRA \
            --dataset-config-file configs/datasets/${DATASET}.yaml \
            --config-file configs/trainers/DSRA/vit_l14_ep150.yaml \
            --output-dir ${DIR} \
            TRAINER.DSRA.N_CTX 16 \
            TRAINER.DSRA.CSC False \
            TRAINER.DSRA.CLASS_TOKEN_POSITION end \
            DATASET.NUM_SHOTS ${SHOTS}
    fi
done

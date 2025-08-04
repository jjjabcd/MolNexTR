#!/bin/bash

NUM_NODES=1
NUM_GPUS_PER_NODE=1
NODE_RANK=0
BATCH_SIZE=64
ACCUM_STEP=1
MASTER_PORT=$(shuf -n 1 -i 10000-65535)
DATESTR=$(date +"%m-%d-%H-%M")

SAVE_PATH="revision_druglike_results/non_augmentation/"

mkdir -p ${SAVE_PATH}

set -x

torchrun \
    --nproc_per_node=$NUM_GPUS_PER_NODE --nnodes=$NUM_NODES --node_rank $NODE_RANK --master_addr localhost --master_port $MASTER_PORT \
    main.py \
    --data_path data \
    --train_file ../../MolScribe/data/DECIMER/DECIMER/filtered_DECIMER_train_train.csv \
    --valid_file ../../MolScribe/data/DECIMER/DECIMER/filtered_DECIMER_train_val.csv \
    --vocab_file MolNexTR/vocab/vocab_chars.json \
    --formats chartok_coords,edges \
    --dynamic_indigo --augment --mol_augment \
    --include_condensed \
    --coord_bins 64 --sep_xy \
    --input_size 256 \
    --encoder_lr 4e-4 \
    --decoder_lr 4e-4 \
    --load_ckpt ckpts/molnextr_best.pth \
    --resume \
    --save_path $SAVE_PATH --save_mode all \
    --label_smoothing 0.1 \
    --epochs 40 \
    --batch_size $((BATCH_SIZE / NUM_GPUS_PER_NODE / ACCUM_STEP)) \
    --gradient_accumulation_steps $ACCUM_STEP \
    --use_checkpoint \
    --warmup 0.02 \
    --print_freq 200 \
    --do_train --do_valid \
    --fp16 --backend gloo 2>&1


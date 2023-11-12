#!/bin/bash

# MCLI run
# 2:4 sparse Llama2-7b, pruned on C4
# 1 epoch on train_mid
# LR = 1e-4

export EXT_DISK=
export S3_BUCKET=s3://eldar-mosaic

# export HF_DATASETS_CACHE=/localhome/ekurtic/hf_cache
# export CUDA_VISIBLE_DEVICES=4,5,6,7
export WANDB_ENTITY=eldarkurtic
export WANDB_DISABLED=False

export DATA_REMOTE=${S3_BUCKET}/llama2_c4_small_1024
export DATA_LOCAL=./local_llama2_c4_small_1024
export MDL=${PWD}/llama2_7b_c4_oneshot_sparsegpt_sp2_4_nsamples512_seqlen2048
export MAX_SEQ_LEN=1024
export SPARSITY=2_4
export NUM_EPOCHS=1
export LR=1e-4

# export MAX_DURATION=${NUM_EPOCHS}ep
export MAX_DURATION=4ba
export BS=32
export PER_DEVICE_BS=1
export GRAD_ACCUM=1
export WARMUP=100ba

export WANDB_PROJECT=llama2_c4_trainmid30M_sp${SPARSITY}_${NUM_EPOCHS}ep
export RUN_NAME=CE1.0_sp${SPARSITY}_${MAX_DURATION}_lr${LR}_bs${BS}_noGradClip_warmup${WARMUP}
export SAVE_FOLDER=${S3_BUCKET}/output_dir/${WANDB_PROJECT}/${RUN_NAME}

composer train_sparse.py \
    yamls/pretrain/llama2/mcli_c4trainmid30M.yaml \
    model_name_or_path=${MDL} \
    max_seq_len=${MAX_SEQ_LEN} \
    max_duration=${MAX_DURATION} \
    run_name=${RUN_NAME} \
    optimizer.lr=${LR} \
    global_train_batch_size=${BS} \
    device_train_microbatch_size=${PER_DEVICE_BS} \
    device_train_grad_accum=${GRAD_ACCUM} \
    eval_first=False \
    scheduler.t_warmup=${WARMUP} \
    data_remote=${DATA_REMOTE} \
    data_local=${DATA_LOCAL} \
    save_folder=${SAVE_FOLDER} \
    eval_subset_num_batches=2

aws s3 --profile ista sync output_dir/${WANDB_PROJECT}/${RUN_NAME} ${S3_BUCKET}/output_dir/${WANDB_PROJECT}/${RUN_NAME}

# KD params
# export TEACHER=meta-llama/Llama-2-7b-hf
# export HARDNESS_CE=1.0
# export HARDNESS_SQUAREHEAD=1.0
# export HARDNESS_KLDIV=0.0
# export KLDIV_TEMPERATURE=1.0
    # knowledge_distillation.teacher_name_or_path=${TEACHER} \
    # knowledge_distillation.hardness_ce=${HARDNESS_CE} \
    # knowledge_distillation.hardness_squarehead=${HARDNESS_SQUAREHEAD} \
    # knowledge_distillation.temperature=${KLDIV_TEMPERATURE} \
    # knowledge_distillation.hardness_kldiv=${HARDNESS_KLDIV} \
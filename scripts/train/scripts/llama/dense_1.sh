#!/bin/bash

#export HF_DATASETS_CACHE=/localhome/ekurtic/hf_cache
export EXT_DISK=/nm/drive0/eldar/llm-foundry
export WANDB_DIR=${EXT_DISK}/wandb

export CUDA_VISIBLE_DEVICES=4,5,6,7

export WANDB_ENTITY=eldarkurtic
export WANDB_DISABLED=False

export SPARSITY=0

for NUM_EPOCHS in 2;
do
    export WANDB_PROJECT=llama2-13b_gsm8k_sp${SPARSITY}
    export MDL=/home/eldar/checkpoints/Llama-2-13b-hf

    for LR in 1e-6 5e-6;
    do
        export MAX_DURATION=${NUM_EPOCHS}ep
        export BS=32
        export PER_DEVICE_BS=8
        export WARMUP=20ba
        export RUN_NAME=sp${SPARSITY}_${MAX_DURATION}_lr${LR}_bs${BS}_warmup${WARMUP}

        composer train_sparse.py \
            yamls/finetune/llama2/dense_yesGradNorm.yaml \
            model_name_or_path=${MDL} \
            max_duration=${MAX_DURATION} \
            run_name=${RUN_NAME} \
            optimizer.lr=${LR} \
            global_train_batch_size=${BS} \
            device_train_microbatch_size=${PER_DEVICE_BS} \
            device_eval_batch_size=${PER_DEVICE_BS} \
            eval_first=True \
            scheduler.t_warmup=${WARMUP}
    done
done



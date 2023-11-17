#!/bin/bash

# Mistral Sparse-Finetuning
# sparsity = {50, 60, 70, 80, 2_4}
# num_epochs = {1, 2, 3, 4}
# LRs = {5e-7, 1e-6, 3e-6, 5e-6, 1e-5, 3e-5}

export EXT_DISK=/nm/drive1/eldar/llm-foundry

export CUDA_VISIBLE_DEVICES=4,5,6,7

for SPARSITY in 70 80 2_4;
do
    for NUM_EPOCHS in 1 2 3 4;
    do
        for LR in 5e-7 1e-6 3e-6 5e-6 1e-5 3e-5;
        do
            export WANDB_ENTITY=eldarkurtic
            export WANDB_DISABLED=False
            export WANDB_PROJECT=mistral_7b_gsm8k_sp${SPARSITY}_${NUM_EPOCHS}ep

            export MDL=/nm/drive0/eldar/neuralmagicml/mistral_7b_base/gsm8k/from_dense_finetuned_acc52/oneshot_sparsegpt_sp${SPARSITY}
            export TEACHER=/nm/drive0/eldar/llm-foundry/output_dir/mistral-7b-v01_gsm8k_sp0/sp0_2ep_lr5e-6_bs32_warmup20ba

            export HARDNESS_CE=1.0
            export HARDNESS_SQUAREHEAD=1.0
            export HARDNESS_KLDIV=0.0
            export KLDIV_TEMPERATURE=1.0

            export MAX_DURATION=${NUM_EPOCHS}ep
            export BS=32
            export PER_DEVICE_BS=8
            export WARMUP=20ba
            export RUN_NAME=CE${HARDNESS_CE}_SquareHead${HARDNESS_SQUAREHEAD}_oneshot_sparsegpt_sp${SPARSITY}_${MAX_DURATION}_lr${LR}_bs${BS}_noGradClip_warmup${WARMUP}

            composer train_sparse.py \
                yamls/finetune/llama2/FT_gsm8k_noGradClip_KD_linearLR.yaml \
                model_name_or_path=${MDL} \
                max_duration=${MAX_DURATION} \
                run_name=${RUN_NAME} \
                optimizer.lr=${LR} \
                global_train_batch_size=${BS} \
                device_train_microbatch_size=${PER_DEVICE_BS} \
                device_eval_batch_size=${PER_DEVICE_BS} \
                knowledge_distillation.teacher_name_or_path=${TEACHER} \
                knowledge_distillation.hardness_ce=${HARDNESS_CE} \
                knowledge_distillation.hardness_squarehead=${HARDNESS_SQUAREHEAD} \
                knowledge_distillation.temperature=${KLDIV_TEMPERATURE} \
                knowledge_distillation.hardness_kldiv=${HARDNESS_KLDIV} \
                eval_first=True \
                scheduler.t_warmup=${WARMUP}
        done
    done
done

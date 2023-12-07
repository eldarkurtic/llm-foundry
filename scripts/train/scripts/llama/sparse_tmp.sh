#!/bin/bash

export EXT_DISK=/data/eldar/llm-foundry
export CUDA_VISIBLE_DEVICES=0,1,2,3

# === 80% sparse model from SFT70 ===
# LRs = 3e-5, 5e-5, 8e-5 1e-4
# eps = 1, 2, 3, 4, 5
# v1)
# hardness_ce = 1.0
# hardness_squarehead = 1.0
# hardness_cosine_dist = 1.0
# v2)
# hardness_ce = 1.0
# hardness_squarehead = 0.0
# hardness_cosine_dist = 1.0


for SPARSITY in 80;
do
    for NUM_EPOCHS in 1 2 3 4 5;
    do
        for LR in 3e-5 5e-5 8e-5 1e-4;
        do
            export WANDB_ENTITY=eldarkurtic
            export WANDB_DISABLED=False
            export WANDB_PROJECT=llama2_gsm8k_sp${SPARSITY}_fromSFT70_${NUM_EPOCHS}ep

	        export MDL=/data/eldar/neuralmagicml/llama2_7b_base/gsm8k/from_SFT70_acc3685/oneshot_sparsegpt_sp80_fromSFT70_nsamples2048_seqlen512_OWL_lamda0.08_M10_IgnorePadding
            export TEACHER=/home/eldar/checkpoints/llama2_7b_gsm8k

            export HARDNESS_CE=1.0
            export HARDNESS_SQUAREHEAD=1.0
            export HARDNESS_KLDIV=0.0
            export KLDIV_TEMPERATURE=1.0
            export HARDNESS_COSINE_DIST=1.0

            export MAX_DURATION=${NUM_EPOCHS}ep
            export BS=32
            export PER_DEVICE_BS=8
            export WARMUP=20ba
            export RUN_NAME=CE${HARDNESS_CE}_SquareHead${HARDNESS_SQUAREHEAD}_CosineDist${HARDNESS_COSINE_DIST}_oneshot_sparsegpt_sp${SPARSITY}_${MAX_DURATION}_lr${LR}_bs${BS}_noGradClip_warmup${WARMUP}

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
                knowledge_distillation.hardness_cosine_dist=${HARDNESS_COSINE_DIST} \
                eval_first=True \
                scheduler.t_warmup=${WARMUP}
        done
    done
done
#!/bin/bash

# etc
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export CUDA_VISIBLE_DEVICES="0,1,2,3"
export AIHUB_ID="jsb10121249@gmail.com"
export AIHUB_PASS="AI_hub_account(123)"
export OMP_NUM_THREADS=10
export TORCHDYNAMO_DISABLE="1"

# wandb
export WANDB_RUN_GROUP="multimodal"
export WANDB_DISABLE_CODE="false"
export WANDB_WATCH="gradients"
export WANDB_DISABLED="false"

deepspeed --num_gpus=4 \
    /root/workspace/main.py \
    --run_name=clip \
    --output_dir=/root/output_dir \
    --model_name_or_path=/root/dual_vision_encoder_model \
    --dataset_name=/root/workspace/data/Outside_Knowledge_based_Multimodal_QA_Data.py \
    --preprocessing_num_workers=2 \
    --seed=42 \
    --image_column=IMAGE \
    --caption_column=CAPTION \
    --per_device_train_batch_size=32 \
    --gradient_accumulation_steps=1 \
    --per_device_eval_batch_size=32 \
    --num_train_epochs=8 \
    --evaluation_strategy=steps \
    --eval_steps=50 \
    --save_strategy=epoch \
    --logging_strategy=steps \
    --logging_steps=1 \
    --lr_scheduler_type=cosine \
    --learning_rate=2e-6 \
    --warmup_ratio=0.1 \
    --optim=adamw_torch \
    --report_to=wandb \
    --dataloader_num_workers=2 \
    --do_train=true \
    --do_eval=true \
    --group_by_length=true \
    --ddp_find_unused_parameters=false \
    --tf32=true \
    --bf16=true \
    --deepspeed=/root/workspace/ds_config/zero_2.json
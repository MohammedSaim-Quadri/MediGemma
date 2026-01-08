#!/bin/bash

# --- CONFIGURATION FOR RTX 4090 (24GB) PART 1 ---
# FIX: Switched to ZeRO Stage 0 (Safety Mode).
# WHY: We have enough RAM with 4-bit. Disabling sharding stops the crashes.

DEEPSPEED_ARGS="--master_port=12345"

python3 -m deepspeed.launcher.runner $DEEPSPEED_ARGS \
    phase3_training/LLaVA/llava/train/train_mem.py \
    --lora_enable True --lora_r 64 --lora_alpha 128 --mm_projector_lr 2e-5 \
    --deepspeed phase3_training/LLaVA/scripts/zero0.json \
    --model_name_or_path liuhaotian/llava-v1.5-7b \
    --version v1 \
    --data_path phase3_training/dataset_train.json \
    --image_folder phase3_training \
    --vision_tower openai/clip-vit-large-patch14-336 \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --group_by_modality_length True \
    --bf16 True \
    --bits 4 \
    --output_dir phase3_training/checkpoints/llava-v1.5-7b-wound-finetune \
    --num_train_epochs 3 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 16 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 500 \
    --save_total_limit 2 \
    --learning_rate 2e-4 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --model_max_length 2048 \
    --dataloader_num_workers 0 \
    --lazy_preprocess True \
    --report_to tensorboard
#!/bin/bash

export CUDA_VISIBLE_DEVICES=0,3,6,7
WANDB_MODE=offline torchrun --nproc_per_node=4 --master_port=20000 train/main.py \
    --model_name_or_path /H1/zhouhongli/JudgePO/models/vicuna-7b-v1.5 \
    --model_type "vicuna" \
    --data_path train/LLaMA-Factory/data/helpsteer_dpo.json \
    --bf16 True \
    --output_dir "/H1/zhouhongli/JudgePO/output/vicuna-7b-v1.5-pairwise_et" \
    --num_train_epochs 3 \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 8 \
    --eval_strategy "no" \
    --save_strategy "steps" \
    --save_steps 1000 \
    --save_total_limit 1 \
    --learning_rate 1e-5 \
    --weight_decay 0 \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 False \
    --model_max_length 1024 \
    --gradient_checkpointing True \
    --lazy_preprocess True \
    --fsdp "full_shard auto_wrap offload" \
    --fsdp_transformer_layer_cls_to_wrap 'LlamaDecoderLayer'

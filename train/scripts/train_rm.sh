#!/bin/bash

export NCCL_P2P_DISABLE=1
export WANDB_MODE=offline
export CUDA_VISIBLE_DEVICES=3,4,5,6


dataset_name='train/data/helpsteer.json'
base_model='/H1/zhouhongli/PORM/output/Llama-2-13B-chat-helpsteer_sft_2e-5'
log_dir='save_reward_models'
n_gpu=4
learning_rate=1e-5
max_length=1024
num_train_epochs=3
per_device_train_batch_size=2
gradient_accumulation_steps=16

accelerate launch --num_processes ${n_gpu} \
    --config_file "/H1/zhouhongli/JudgePO/train/LLaMA-Factory/configs/accelerate/fsdp_config.yaml" \
    train/reward_models/run_reward_models_train.py \
    --base_model ${base_model} \
    --log_dir ${log_dir} \
    --num_train_epochs ${num_train_epochs} \
    --per_device_train_batch_size ${per_device_train_batch_size} \
    --gradient_accumulation_steps ${gradient_accumulation_steps} \
    --max_length ${max_length} \
    --use_lora False \
    --learning_rate ${learning_rate} \
    --dataset ${dataset_name} \
    --bf16 True \
    --gradient_checkpointing True 
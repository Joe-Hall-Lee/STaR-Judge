#!/bin/bash

export NCCL_P2P_DISABLE=1
export WANDB_MODE=offline
export CUDA_VISIBLE_DEVICES=6,7


dataset_name='data/arena_rm_llama.jsonl'
base_model='model/Llama-3.2-3B-Instruct'
log_dir='output'
main_process_port=12541

n_gpu=2
learning_rate=1e-5
max_length=1024
num_train_epochs=3
per_device_train_batch_size=8
gradient_accumulation_steps=8

accelerate launch --num_processes ${n_gpu} \
    --main_process_port ${main_process_port} \
    --config_file accelerate/fsdp_config.yaml \
    train/reward_models/run_reward_models_train.py \
    --base_model ${base_model} \
    --log_dir ${log_dir} \
    --num_train_epochs ${num_train_epochs} \
    --per_device_train_batch_size ${per_device_train_batch_size} \
    --gradient_accumulation_steps ${gradient_accumulation_steps} \
    --max_length ${max_length} \
    --learning_rate ${learning_rate} \
    --dataset ${dataset_name} \
    --bf16 True \
    --gradient_checkpointing True

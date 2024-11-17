#!/bin/bash
#SBATCH --job-name=Mistral-7B-Instruct-v0.3_lr_1e-6  # 任务名字
#SBATCH --nodes=1  # 节点
#SBATCH --ntasks=4  # 需要多少个 cpu
#SBATCH --gres=gpu:4  # 使用 gpu 数量，如果 2 个 gpu 写成 gpu:2
#SBATCH --partition=gpus # 分区默认即可
source ~/.bashrc
WANDB_MODE=offline CUDA_VISIBLE_DEVICES="1,3,6,7"  accelerate launch \
    --config_file ../accelerate/fsdp_config.yaml \
    --main_process_port=12542 \
    train/LLaMA-Factory/src/train.py configs/train/mistral_full_sft.yaml

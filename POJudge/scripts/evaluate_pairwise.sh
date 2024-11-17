#!/bin/bash
#SBATCH --job-name=Llama-2-13b-chat_et_lr_2e-5_eval  # 任务名字
#SBATCH --nodes=1  # 节点
#SBATCH --ntasks=4  # 需要多少个 cpu
#SBATCH --gres=gpu:1  # 使用 gpu 数量，如果 2 个 gpu 写成 gpu:2
#SBATCH --partition=gpus  # 分区默认即可

export CUDA_VISIBLE_DEVICES=2
python eval/run_bench.py \
    --name Llama-2-13b-chat_et_lr_2e-5 \
    --config configs/eval/llama2-13b.yaml \
    --benchmarks llmbar,hhh,mtbench,biasbench,judgelm \
    --data-path ../data

#!/bin/bash
#SBATCH --job-name=Mistral-7B-Instruct-v0.3-helpsteer2_et_lr_1e-6_eval  # 任务名字
#SBATCH --nodes=1  # 节点
#SBATCH --ntasks=4  # 需要多少个 cpu
#SBATCH --gres=gpu:1  # 使用 gpu 数量，如果 2 个 gpu 写成 gpu:2
#SBATCH --partition=gpus  # 分区默认即可

export CUDA_VISIBLE_DEVICES=5
python eval/run_bench.py \
    --name gemma-2b-it_test_len1024_fulltrain_2e-06_dataunified_dpo_gemma.json \
    --config configs/eval/gemma-2b.yaml \
    --benchmark llmbar,hhh,mtbench,rewardbench,unified-feedback \
    --data-path data

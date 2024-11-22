#!/bin/bash
#SBATCH --job-name=yolov5  # 任务名字
#SBATCH --nodes=1  # 节点
#SBATCH --ntasks=4  # 需要多少个 cpu
#SBATCH --gres=gpu:1  #使用 gpu 数量，如果 2 个 gpu 写成 gpu:2
#SBATCH --partition=gpus  # 分区默认即可

export CUDA_VISIBLE_DEVICES="6" 
python -m eval.rewardbench.rewardbench \
    --model="save_reward_models/gemma-2b-it_test_len1024_fulltrain_2e-06_dataunified_dpo.json/logs/checkpoint-78" \
    --dataset="data/rewardbench/filtered.json" \
    --batch_size=32 \
    --load_json \
    --save_all

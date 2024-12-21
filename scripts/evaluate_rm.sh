#!/bin/bash
#SBATCH --job-name=yolov5  # 任务名字
#SBATCH --nodes=1  # 节点
#SBATCH --ntasks=4  # 需要多少个 cpu
#SBATCH --gres=gpu:1  #使用 gpu 数量，如果 2 个 gpu 写成 gpu:2
#SBATCH --partition=gpus  # 分区默认即可

export CUDA_VISIBLE_DEVICES="3" 
python -m eval.rewardbench.rewardbench \
    --model="output/Llama-3.2-3B-Instruct_test_len1024_fulltrain_1e-05_datawebgpt_dpo.json" \
    --dataset="data/webgpt_dpo.json" \
    --output_dir="result/webgpt" \
    --batch_size=32 \
    --load_json \
    --save_all

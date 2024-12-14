#!/bin/bash
#SBATCH --job-name=Llama-2-13b-chat_lr_2e-5  # 任务名字
#SBATCH --nodes=1  # 节点
#SBATCH --gres=gpu:1  # 使用 gpu 数量，如果 2 个 gpu 写成 gpu:2
#SBATCH --partition=gpus # 分区默认即可

RESULT_FILE="../result/Llama-3.2-3B-Instruct/arena.json"
DISTILL_ET_FILE="../result/Llama-3.2-3B-Instruct-arena_distlled_et_lr_1e-5_epoch_3/arena.json"
OUTPUT_FILE="../data/arena_cot_2.json"
DATA_KEY="arena"
MODEL_NAME="../output/Llama-3.2-3B-Instruct-arena_generate_cot_lr_1e-5_epoch_3"
TEMPERATURE=0.0
MAX_TOKENS=512

CUDA_VISIBLE_DEVICES="5" python generate_rationale_judge.py \
    --result "$RESULT_FILE" \
    --result_distill_et "$DISTILL_ET_FILE" \
    --output "$OUTPUT_FILE" \
    --data "$DATA_KEY" \
    --model "$MODEL_NAME" \
    --temperature "$TEMPERATURE" \
    --max_tokens "$MAX_TOKENS"

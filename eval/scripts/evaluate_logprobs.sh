#!/bin/bash
export CUDA_VISIBLE_DEVICES=1
python evaluate_logprobs.py \
    --model-name-or-path "/H1/zhouhongli/JudgePO/LLaMA-Factory/output/Llama-3-8B-Instruct-helpsteer_sft" \
    --files llmbar hhh mtbench biasbench judgelm \
    --data-dir /H1/zhouhongli/Gen4Eval/eval/result/Llama-3-8B-Instruct-helpsteer_sft
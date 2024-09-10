#!/bin/bash
export CUDA_VISIBLE_DEVICES=3
python evaluate_logprobs.py \
    --model-name-or-path "/H1/zhouhongli/JudgePO/models/Llama-2-7b-chat-hf" \
    --files llmbar hhh mtbench biasbench judgelm \
    --data-dir /H1/zhouhongli/Gen4Eval/eval/result/Llama-2-7b-chat-hf
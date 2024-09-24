#!/bin/bash
export CUDA_VISIBLE_DEVICES=2
python eval/evaluate_logprobs.py \
    --model-name-or-path "/H1/zhouhongli/JudgePO/models/Llama-2-13B-chat" \
    --files llmbar hhh mtbench biasbench judgelm \
    --data-dir /H1/zhouhongli/Gen4Eval/eval/result/Llama-2-13B-chat
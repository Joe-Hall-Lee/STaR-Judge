#!/bin/bash
export CUDA_VISIBLE_DEVICES=7
python evaluate_logprobs.py \
    --model-name-or-path "/H1/zhouhongli/JudgePO/LLaMA-Factory/output/Llama-2-13B-chat-helpsteer_simpo_2.0_1.6_lr_1e-5_epoch_3/checkpoint-870" \
    --files llmbar hhh mtbench biasbench judgelm \
    --data-dir /H1/zhouhongli/Gen4Eval/eval/result/Llama-2-13B-chat-helpsteer_simpo_2.0_1.6_lr_1e-5_epoch_3
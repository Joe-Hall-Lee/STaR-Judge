export CUDA_VISIBLE_DEVICES="1" 
python -m eval.rewardbench.rewardbench \
    --model="/H1/zhouhongli/JudgePO/save_reward_models/Llama-2-7b-chat-hf_test_len1024_fulltrain_1e-05_datahelpsteer.json/logs/checkpoint-3" \
    --dataset="eval/data/rewardbench/filtered.json" \
    --batch_size=16 \
    --load_json \
    --debug
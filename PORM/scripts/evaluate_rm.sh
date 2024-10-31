export CUDA_VISIBLE_DEVICES="0" 
python -m eval.rewardbench.rewardbench \
    --model="save_reward_models/Qwen2.5-3B-Instruct_test_len1024_fulltrain_1e-05_datahelpsteer2_dpo.json/logs" \
    --batch_size=32 \
    --load_json \
    --debug
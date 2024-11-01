export CUDA_VISIBLE_DEVICES="0" 
python -m eval.rewardbench.rewardbench \
    --model="/H1/zhouhongli/POEnhancer/PORM/save_reward_models/Llama-2-13B-chat_test_len1024_fulltrain_1e-05_datahelpsteer2_dpo.json/logs/checkpoint-1" \
    --batch_size=32 \
    --load_json \
    --debug
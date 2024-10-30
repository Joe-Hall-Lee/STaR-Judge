export CUDA_VISIBLE_DEVICES="1" 
python -m eval.rewardbench.rewardbench \
    --model="/H1/zhouhongli/POEnhancer/PORM/save_reward_models/Llama-3.1-8B-Instruct_test_len1024_fulltrain_1e-05_datahelpsteer2_dpo.json/logs" \
    --batch_size=64 \
    --load_json
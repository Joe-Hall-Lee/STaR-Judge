export CUDA_VISIBLE_DEVICES="1" 
python -m eval.rewardbench.rewardbench \
    --model="save_reward_models/Mistral-7B-Instruct-v0.3-helpsteer2_orpo_0.01_lr_1e-5_epoch_3_test_len1024_fulltrain_1e-05_datahelpsteer2_dpo.json/logs/checkpoint-156" \
    --batch_size=32 \
    --load_json
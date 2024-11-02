export CUDA_VISIBLE_DEVICES="3" 
python -m eval.rewardbench.rewardbench \
    --model="save_reward_models/Llama-2-13B-chat-helpsteer2_simpo_2.0_1.6_lr_1e-5_epoch_3_test_len1024_fulltrain_1e-05_datahelpsteer2_dpo.json/logs/checkpoint-156" \
    --batch_size=32 \
    --load_json
CUDA_VISIBLE_DEVICES="0,1,3,4" accelerate launch \
    --config_file train/LLaMA-Factory/configs/accelerate/fsdp_config.yaml \
    --main_process_port=12541 \
    train/LLaMA-Factory/src/train.py train/LLaMA-Factory/configs/vicuna_full_dpo.yaml

WANDB_MODE=offline CUDA_VISIBLE_DEVICES="3,4,5,6" accelerate launch \
    --config_file train/LLaMA-Factory/configs/accelerate/fsdp_config.yaml \
    --main_process_port=12541 \
    train/LLaMA-Factory/src/train.py train/LLaMA-Factory/configs/llama2_full_dpo.yaml

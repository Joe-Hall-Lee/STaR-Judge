WANDB_MODE=offline CUDA_VISIBLE_DEVICES="4,5,6,7" accelerate launch \
    --config_file ../accelerate/fsdp_config.yaml \
    --main_process_port=12542 \
    train/LLaMA-Factory/src/train.py configs/train/mistral_full_sft.yaml

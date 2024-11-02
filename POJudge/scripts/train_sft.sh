WANDB_MODE=offline CUDA_VISIBLE_DEVICES="0,1" accelerate launch \
    --config_file ../accelerate/fsdp_config.yaml \
    --main_process_port=12542 \
    train/LLaMA-Factory/src/train.py configs/train/qwen2.5_full_sft.yaml

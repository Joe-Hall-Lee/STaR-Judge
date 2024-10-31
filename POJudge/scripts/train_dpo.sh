WANDB_MODE=offline CUDA_VISIBLE_DEVICES="4,5,6,7  " accelerate launch \
    --config_file configs/accelerate/fsdp_config.yaml \
    --main_process_port=12541 \
    train/LLaMA-Factory/src/train.py configs/llama2_full_dpo.yaml

CUDA_VISIBLE_DEVICES="2,6" accelerate launch \
    --config_file configs/accelerate/fsdp_config.yaml \
    src/train.py configs/llama3_full_dpo.yaml

CUDA_VISIBLE_DEVICES="1,3" accelerate launch \
    --config_file configs/accelerate/fsdp_config.yaml \
    src/train.py configs/llama2_full_dpo.yaml

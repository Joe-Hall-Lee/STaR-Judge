CUDA_VISIBLE_DEVICES="0,1,2,3" accelerate launch \
    --config_file configs/accelerate/fsdp_config.yaml \
    src/train.py configs/llama3_full_dpo.yaml

CUDA_VISIBLE_DEVICES="0,7" accelerate launch \
    --config_file configs/accelerate/fsdp_config.yaml \
    --main_process_port=12548 \
    src/train.py configs/llama2_full_dpo.yaml

CUDA_VISIBLE_DEVICES="0,1,3,4" accelerate launch \
    --config_file configs/accelerate/fsdp_config.yaml \
    --main_process_port=12541 \
    src/train.py configs/llama2_full_dpo.yaml

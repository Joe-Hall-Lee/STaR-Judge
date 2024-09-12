CUDA_VISIBLE_DEVICES="0,1,3,7" accelerate launch \
    --config_file configs/accelerate/fsdp_config.yaml \
    --main_process_port=12547 \
    src/train.py configs/llama3_full_sft.yaml

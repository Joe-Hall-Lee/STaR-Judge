CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7" accelerate launch \
    --config_file configs/accelerate/fsdp_config.yaml \
    --main_process_port=12542 \
    src/train.py configs/llama2_full_sft.yaml

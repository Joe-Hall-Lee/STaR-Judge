WANDB_MODE=offline CUDA_VISIBLE_DEVICES="0,1,3,4" accelerate launch \
    --config_file /H1/zhouhongli/JudgePO/train/LLaMA-Factory/configs/accelerate/deepspeed_zero3.yaml \
    --main_process_port=12542 \
    train/LLaMA-Factory/src/train.py train/LLaMA-Factory/configs/llama2_full_rm.yaml

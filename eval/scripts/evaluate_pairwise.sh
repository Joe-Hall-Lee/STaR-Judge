export CUDA_VISIBLE_DEVICES=3
python run_bench.py \
    --name Llama-2-13B-chat-helpsteer_orpo_0.2_lr_1e-5_epoch_3 \
    --config config/llama2-13b.yaml \
    --benchmarks llmbar,hhh,mtbench,biasbench,judgelm \
    --data-path "/H1/zhouhongli/Gen4Eval/eval/data"

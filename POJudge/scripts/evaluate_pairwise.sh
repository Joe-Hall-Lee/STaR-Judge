export CUDA_VISIBLE_DEVICES=4
python eval/run_bench.py \
    --name Llama-2-13B-chat-helpsteer_simpo_2.0_1.6_lr_1e-5_epoch_3_et \
    --config eval/config/llama2-13b.yaml \
    --benchmarks llmbar,hhh,mtbench,biasbench,judgelm \
    --data-path eval/data

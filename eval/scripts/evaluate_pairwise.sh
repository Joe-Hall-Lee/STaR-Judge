export CUDA_VISIBLE_DEVICES=4
python eval/run_bench.py \
    --name Llama-2-13B-chat-offsetbias_orpo_0.1_lr_2e-5_epoch_3 \
    --config eval/config/llama2-13b.yaml \
    --benchmarks llmbar,hhh,mtbench,biasbench,judgelm \
    --data-path "/H1/zhouhongli/Gen4Eval/eval/data"

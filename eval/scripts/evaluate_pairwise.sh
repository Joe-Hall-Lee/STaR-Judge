export CUDA_VISIBLE_DEVICES=4
python eval/run_bench.py \
    --name Llama-2-13B-chat-offsetbias_et \
    --config eval/config/llama2-13b.yaml \
    --benchmarks llmbar,hhh,mtbench,biasbench,judgelm \
    --data-path eval/data

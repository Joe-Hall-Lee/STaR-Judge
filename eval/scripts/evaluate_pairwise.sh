export CUDA_VISIBLE_DEVICES=0
python eval/run_bench.py \
    --name test \
    --config eval/config/llama2-13b.yaml \
    --benchmarks llmbar,hhh,mtbench,biasbench,judgelm \
    --data-path eval/data

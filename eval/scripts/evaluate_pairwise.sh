export CUDA_VISIBLE_DEVICES=0
python run_bench.py \
    --name Llama-2-13B-chat \
    --config config/llama2-13b.yaml \
    --benchmarks llmbar,hhh,mtbench,biasbench,judgelm \
    --data-path "/H1/zhouhongli/Gen4Eval/eval/data"

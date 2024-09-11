export CUDA_VISIBLE_DEVICES=1
python run_bench.py \
    --name Llama-3-8B-Instruct-helpsteer_sft\
    --config config/llama3-8b.yaml \
    --benchmarks llmbar,hhh,mtbench,biasbench,judgelm \
    --data-path "/H1/zhouhongli/Gen4Eval/data"

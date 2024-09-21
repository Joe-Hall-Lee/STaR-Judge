export CUDA_VISIBLE_DEVICES=1
python run_bench.py \
    --name Vicuna-7B \
    --config config/vicuna-7b.yaml \
    --benchmarks llmbar,hhh,mtbench,biasbench,judgelm \
    --data-path "/H1/zhouhongli/Gen4Eval/eval/data"

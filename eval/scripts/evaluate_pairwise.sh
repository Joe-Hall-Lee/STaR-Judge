export CUDA_VISIBLE_DEVICES=3
python eval/run_bench.py \
    --name vicuna-7b-union \
    --config eval/config/vicuna-7b.yaml \
    --benchmarks llmbar,hhh,mtbench,biasbench,judgelm \
    --data-path "/H1/zhouhongli/Gen4Eval/eval/data"

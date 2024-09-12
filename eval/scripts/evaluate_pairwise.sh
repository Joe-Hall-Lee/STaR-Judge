export CUDA_VISIBLE_DEVICES=0
python run_bench.py \
    --name Llama-3-8B-Instruct-helpsteer_orpo_0.2_lr_1e-6\
    --config config/llama3-8b.yaml \
    --benchmarks llmbar,hhh,mtbench,biasbench,judgelm \
    --data-path "/H1/zhouhongli/Gen4Eval/data"

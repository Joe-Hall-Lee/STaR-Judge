export CUDA_VISIBLE_DEVICES=1
python run_bench.py \
    --name Llama-2-7b-chat-hf-offsetbias_orpo_0.2 \
    --config config/llama2-7b.yaml \
    --benchmarks llmbar,hhh,mtbench,biasbench,judgelm \
    --data-path "/H1/zhouhongli/Gen4Eval/data"

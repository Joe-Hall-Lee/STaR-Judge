## Usage

### 1. Set Up the Environment

Install the required dependencies:

```bash
pip install -r requirements.txt
```

### 2. Generate CoT Evaluation Results

Run the `scripts/evaluate_judge.sh` script to generate the model's Chain-of-Thought (CoT) evaluation results. Ensure the `prompt` in the configuration file is set to `cot`:

```bash
python eval/run_bench.py \
    --name {result_name} \
    --config {config_path} \
    --benchmark rewardbench \
    --data-path data
```

### 3. Train the ET Model

Use the `scripts/train_sft.sh` script to train the ET model. Update the configuration files accordingly. For the `arena` benchmark, it is recommended to set `cutoff_len` to 2048.

### 4. Evaluate the ET Model

Run the `scripts/evaluate_judge.sh` script again to generate the ET model's evaluation results. For this step, set the `prompt` in the configuration file to `llmbar`:

```bash
python eval/run_bench.py \
    --name {result_name} \
    --config {config_path} \
    --benchmark rewardbench,arena \
    --data-path data
```

### 5. Train the Distilled ET Model

Navigate to the `disilled` folder and execute `distill_judge.py`. Afterward, use the `scripts/train_sft.sh` script to train the Distilled ET model.

### 6. Train the Rationale Generation Model

In the `rationale` folder:

1. Run `generate_cot.py` to generate rationale outputs.
2. Use the resulting `cot_output_file` to train the rationale generation model.

### 7. Train the CoT ET Model

In the `rationale` folder:

1. Run the `generate_rationale_judge.sh` script, then merge the `OUTPUT_FILE` from `generate_rationale_judge.sh` and the `et_output_file` from `generate_cot.py`.
2. Use the `scripts/train_sft.sh` script to train the CoT ET model.

### 8. Generate CoT ET Model Evaluation Results

Run the `scripts/evaluate_judge.sh` script to generate the CoT ET model's evaluation results.

### 9. Prepare Reward Model Training Data

Navigate to the `disilled` folder and run `distill_judge_rm.py` to prepare the reward model (RM) training dataset.

### 10. Train the Reward Model

Use the `scripts/train_rm.sh` script to train the reward model.

### 11. Evaluate the Reward Model

Run the `scripts/evaluate_rm.sh` script to evaluate the reward model.

## Acknowledgements

This repo benefits from [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory), [OffsetBias](https://github.com/ncsoft/offsetbias), [GRM](https://github.com/YangRui2015/Generalizable-Reward-Model) and [RewardBench](https://github.com/allenai/reward-bench). Thanks for their wonderful works!


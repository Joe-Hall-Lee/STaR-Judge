## Environment Setup
This project requires two distinct environments for training and evaluation:
```
git clone git@github.com:Joe-Hall-Lee/POEnhancer.git

# For **SFT & xPO & Evaluation Tuning** and **Judge Evluation**
cd POEnhancer/POJudge
conda create -n POJudge python=3.10
pip install -r requirements.txt

# For **Reward modeling** and **RM Evluation**
cd POEnhancer/PORM
conda create -n PORM python=3.10
pip install -r requirements.txt
```
## Usage
Run the corresponding script in the scripts folder. If nothing goes wrong, only the .sh script and config file need to be changed.

### ET
Taking Qwen2.5 as an example, set `dataset` to `helpsteer2_et` in POJudge/configs/train/qwen2.5_full_sft.yaml, and set the config file parameter of `train_sft.sh` to `configs/train/qwen2.5_full_sft.yaml`.
```
cd POEnhancer/POJudge
nohup bash scripts/train_sft.sh > et.log 2>&1 &
```

### SFT
Taking Qwen2.5 as an example, set `dataset` to `helpsteer2_sft` in `POJudge/configs/train/qwen2.5_full_sft.yaml`, and set the config file parameter of `train_sft.sh` to `configs/train/qwen2.5_full_sft.yaml`.
```
cd POEnhancer/POJudge
nohup bash scripts/train_sft.sh > sft.log 2>&1 &
```

### xPO
```
cd POEnhancer/POJudge
nohup bash scripts/train_dpo.sh > dpo.log 2>&1 &
```

### Evaluating Judge
Modify the model path in the config file in the `POJudge/configs/eval` folder, and set the config file parameter of `evaluate_pairwise.sh` correctly.
```
cd POEnhancer/POJudge
nohup bash scripts/evaluate_pairwise.sh > eval.log 2>&1 &
```
The evaluation results will be in the `result` folder.
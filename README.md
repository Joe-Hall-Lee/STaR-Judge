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
Run the corresponding script in the scripts folder. Note that for POJudge, parameters in the configs folder also need to be modified.
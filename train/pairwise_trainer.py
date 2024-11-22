import torch.nn.functional as F
from transformers import Trainer
from transformers.trainer_pt_utils import LabelSmoother
from utils import rank0_print

IGNORE_TOKEN_ID = LabelSmoother.ignore_index


class PairwiseTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def compute_loss(self, model, inputs, return_outputs=False):
        # Extract inputs
        better_sft_input_ids = inputs["better_sft_input_ids"]
        better_sft_attention_mask = inputs["better_sft_attention_mask"]
        worse_sft_input_ids = inputs["worse_sft_input_ids"]
        worse_sft_attention_mask = inputs["worse_sft_attention_mask"]
        better_et_input_ids = inputs["better_et_input_ids"]
        better_et_attention_mask = inputs["better_et_attention_mask"]
        worse_et_input_ids = inputs["worse_et_input_ids"]
        worse_et_attention_mask = inputs["worse_et_attention_mask"]
        better_sft_labels = inputs["better_sft_labels"]
        worse_sft_labels = inputs["worse_sft_labels"]
        better_et_labels = inputs["better_et_labels"]
        worse_et_labels = inputs["worse_et_labels"]

        def compute_sft_loss(input_ids, labels, attention_mask):
            outputs = model(input_ids=input_ids,
                            attention_mask=attention_mask, labels=labels)
            return outputs.loss

        better_et_loss = compute_sft_loss(
            better_et_input_ids, better_et_labels, better_et_attention_mask)
        worse_et_loss = compute_sft_loss(
            worse_et_input_ids, worse_et_labels, worse_et_attention_mask)
        better_sft_loss = compute_sft_loss(
            better_sft_input_ids, better_sft_labels, better_sft_attention_mask)
        worse_sft_loss = compute_sft_loss(
            worse_sft_input_ids, worse_sft_labels, worse_sft_attention_mask)

        better_sft_logps = -better_sft_loss
        worse_sft_logps = -worse_sft_loss

        beta = 2.0
        simpo_gamma = 1.6
        pi_logratios = better_sft_logps - worse_sft_logps
        gamma_logratios = simpo_gamma / beta
        logits = pi_logratios - gamma_logratios
        simpo_loss = -F.logsigmoid(beta * logits)
        weight = 0.1
        better_sft_loss = weight * better_sft_loss
        simpo_loss = weight * simpo_loss

        et_loss = better_et_loss + worse_et_loss

        loss = better_sft_loss + simpo_loss + et_loss

        rank0_print(
            f"\nSFT loss: {better_sft_loss.item():.4f}\tDPO loss: {simpo_loss.item():.4f}\tET loss: {et_loss.item():.4f}")

        return loss

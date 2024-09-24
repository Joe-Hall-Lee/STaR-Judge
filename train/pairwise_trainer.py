import torch
import torch.nn.functional as F
from transformers import Trainer
from transformers.trainer_pt_utils import LabelSmoother

IGNORE_TOKEN_ID = LabelSmoother.ignore_index


class PairwiseTrainer(Trainer):
    def __init__(self, alpha=15, pad_token_id=0, disable_prompt_loss=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pad_token_id = pad_token_id
        self.alpha = alpha
        self.loss_fct = torch.nn.CrossEntropyLoss(
            reduction='none', ignore_index=self.pad_token_id)  # Added ignore_index
        self.disable_prompt_loss = disable_prompt_loss

    def compute_logps(self, input_ids, attention_mask, logits):
        mask = attention_mask[:, :-1]
        per_token_logps = torch.gather(logits[:, :-1, :].log_softmax(-1), dim=2,
                                       index=(input_ids[:, 1:]).unsqueeze(2)).squeeze(2)
        return torch.mul(per_token_logps, mask).sum(dim=1) / mask.sum(dim=1)

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
            return outputs.loss, outputs.logits

        better_sft_loss, better_sft_logits = compute_sft_loss(
            better_sft_input_ids, better_sft_labels, better_sft_attention_mask)
        worse_sft_loss, worse_sft_logits = compute_sft_loss(
            worse_sft_input_ids, worse_sft_labels, worse_sft_attention_mask)
        better_et_loss, better_et_logits = compute_sft_loss(
            better_et_input_ids, better_et_labels, better_et_attention_mask)
        worse_et_loss, worse_et_logits = compute_sft_loss(
            worse_et_input_ids, worse_et_labels, worse_et_attention_mask)

        better_sft_logps = self.compute_logps(
            better_sft_input_ids, better_sft_attention_mask, better_sft_logits)
        worse_sft_logps = self.compute_logps(
            worse_sft_input_ids, worse_sft_attention_mask, worse_sft_logits)

        beta = 2.0
        simpo_gamma = 1.6
        pi_logratios = better_sft_logps - worse_sft_logps
        gamma_logratios = simpo_gamma / beta
        logits = pi_logratios - gamma_logratios
        simpo_loss = -F.logsigmoid(beta * logits)

        loss = better_sft_loss + simpo_loss + better_et_loss + worse_et_loss
        loss = better_et_loss + worse_et_loss

        return loss

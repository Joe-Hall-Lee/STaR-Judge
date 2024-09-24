import json
import torch
from dataclasses import dataclass
from typing import Dict, Union, List
from torch.utils.data import Dataset
import transformers
from utils import format_instruction_sft, format_instruction_et, rank0_print


def preprocess(
    sources,
    instruction_et,
    instruction_sft
) -> Dict:


    better_sft_prompts = []
    better_sft_labels = []
    better_et_prompts = []
    better_et_labels = []

    worse_sft_prompts = []
    worse_sft_labels = []
    worse_et_prompts = []
    worse_et_labels = []

    for i, source in enumerate(sources):
        better_sft_prompt = format_instruction_sft(instruction_sft, source)
        better_sft_prompts.append(better_sft_prompt)
        better_sft_labels.append(source["chosen"])

        better_et_prompt = format_instruction_et(instruction_et, source)
        better_et_prompts.append(better_et_prompt)
        better_et_labels.append("Output (a)")

        source["chosen"], source["rejected"] = source["rejected"], source["chosen"]

        worse_sft_prompt = better_sft_prompt
        worse_sft_prompts.append(worse_sft_prompt)
        worse_sft_labels.append(source["rejected"])

        worse_et_prompt = format_instruction_et(instruction_et, source)
        worse_et_prompts.append(worse_et_prompt)
        worse_et_labels.append("Output (b)")

    return better_sft_prompts, better_sft_labels, worse_sft_prompts, worse_sft_labels, better_et_prompts, better_et_labels, worse_et_prompts, worse_et_labels


class LazySupervisedDataset(Dataset):
    def __init__(self, raw_data, tokenizer: transformers.PreTrainedTokenizer, instruction_et, instruction_sft, data_path):
        super(LazySupervisedDataset, self).__init__()
        self.tokenizer = tokenizer
        self.raw_data = raw_data
        self.cached_data_dict = {}
        self.instruction_et = instruction_et
        self.instruction_sft = instruction_sft
        self.data_path = data_path


class PairwiseLazySupervisedDataset(LazySupervisedDataset):
    def __len__(self):
        return len(self.raw_data)

    def __getitem__(self, i):
        better_sft_prompts, better_sft_labels, worse_sft_prompts, worse_sft_labels, better_et_prompts, better_et_labels, worse_et_prompts, worse_et_labels = preprocess(
            [self.raw_data[i]],
            self.instruction_et,
            self.instruction_sft
        )

        def _tokenize(prompt, label):
            conv_labels = [prompt + label]
            tokenized = self.tokenizer(
                conv_labels,
                return_tensors="pt",
                padding="max_length",
                max_length=self.tokenizer.model_max_length,
                truncation=True,
            )
            labels = tokenized.input_ids.clone()
            source_tokenized = self.tokenizer(
                [prompt],
                return_tensors="pt",
                padding="max_length",
                max_length=self.tokenizer.model_max_length,
                truncation=True,
            )
            source_len = source_tokenized.input_ids[0].ne(
                self.tokenizer.pad_token_id).sum().item()
            labels[0][:source_len] = -100  # IGNORE_TOKEN_ID

            return dict(
                input_ids=tokenized.input_ids[0],
                labels=labels[0],
                attention_mask=tokenized.attention_mask[0],
            )

        better_sft = _tokenize(better_sft_prompts[0], better_sft_labels[0])
        worse_sft = _tokenize(worse_sft_prompts[0], worse_sft_labels[0])
        better_et = _tokenize(better_et_prompts[0], better_et_labels[0])
        worse_et = _tokenize(worse_et_prompts[0], worse_et_labels[0])

        return {
            "better_sft_input_ids": better_sft["input_ids"],
            "better_sft_labels": better_sft["labels"],
            "better_sft_attention_mask": better_sft["attention_mask"],
            "worse_sft_input_ids": worse_sft["input_ids"],
            "worse_sft_labels": worse_sft["labels"],
            "worse_sft_attention_mask": worse_sft["attention_mask"],
            "better_et_input_ids": better_et["input_ids"],
            "better_et_labels": better_et["labels"],
            "better_et_attention_mask": better_et["attention_mask"],
            "worse_et_input_ids": worse_et["input_ids"],
            "worse_et_labels": worse_et["labels"],
            "worse_et_attention_mask": worse_et["attention_mask"],
        }


def make_supervised_data_module(
    tokenizer: transformers.PreTrainedTokenizer,
    data_args,
    instruction_et,
    instruction_sft
) -> Dict:
    dataset_cls = PairwiseLazySupervisedDataset

    rank0_print(f"The data path is {data_args.data_path}")
    rank0_print("Loading data...")
    train_data = []

    with open(data_args.data_path, "r", encoding="utf-8") as fin:
        train_data = json.load(fin)
    train_dataset = dataset_cls(train_data,
                                tokenizer=tokenizer,
                                instruction_et=instruction_et,
                                instruction_sft=instruction_sft,
                                data_path=data_args.data_path,
                                )

    rank0_print("Loading data finished")

    return dict(train_dataset=train_dataset)


@dataclass
class PairwiseDataCollator:
    def __call__(self, examples: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        if not isinstance(examples[0], dict):
            raise ValueError(f"Unexpected input type: {type(examples[0])}")

        batch = {
            "better_sft_input_ids": torch.stack([example["better_sft_input_ids"] for example in examples]),
            "better_sft_labels": torch.stack([example["better_sft_labels"] for example in examples]),
            "better_sft_attention_mask": torch.stack([example["better_sft_attention_mask"] for example in examples]),
            "worse_sft_input_ids": torch.stack([example["worse_sft_input_ids"] for example in examples]),
            "worse_sft_labels": torch.stack([example["worse_sft_labels"] for example in examples]),
            "worse_sft_attention_mask": torch.stack([example["worse_sft_attention_mask"] for example in examples]),
            "better_et_input_ids": torch.stack([example["better_et_input_ids"] for example in examples]),
            "better_et_labels": torch.stack([example["better_et_labels"] for example in examples]),
            "better_et_attention_mask": torch.stack([example["better_et_attention_mask"] for example in examples]),
            "worse_et_input_ids": torch.stack([example["worse_et_input_ids"] for example in examples]),
            "worse_et_labels": torch.stack([example["worse_et_labels"] for example in examples]),
            "worse_et_attention_mask": torch.stack([example["worse_et_attention_mask"] for example in examples]),
        }

        return batch

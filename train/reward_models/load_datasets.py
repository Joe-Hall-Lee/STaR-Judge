import numpy as np
import os
import json
import torch.nn as nn
from datasets import load_dataset, Dataset


# for vanilla chosen and reject style dataset, such as dendrydong/preference_700K
def build_dataset(data_path, tokenizer, split='train', size=None, model_name=''):

    def convert_to_chat_format(example):
        new_example = {}
        chosen_messages = []
        rejected_messages = []

        # Add the instruction as the first user message
        chosen_messages.append(
            {"role": "user", "content": example["instruction"]})
        rejected_messages.append(
            {"role": "user", "content": example["instruction"]})

        # Add the chosen and rejected responses as assistant messages
        chosen_messages.append(
            {"role": "assistant", "content": example["chosen"]})
        rejected_messages.append(
            {"role": "assistant", "content": example["rejected"]})

        new_example["chosen"] = chosen_messages
        new_example["rejected"] = rejected_messages

        return new_example

    ds = []


    with open(data_path, 'r') as file:
        ds = json.load(file)

    if size is not None:
        ds = ds[:size]

    def formatting_func(example):
        kwargs = {"padding": "max_length", "truncation": True,
                  "max_length": tokenizer.max_length, "return_tensors": "pt"}

        # convert example to chat format
        example = convert_to_chat_format(example)
        chosen_messages = example['chosen']
        rejected_messages = example['rejected']

        prompt_plus_chosen_response = tokenizer.apply_chat_template(
            chosen_messages, tokenize=False)
        prompt_plus_rejected_response = tokenizer.apply_chat_template(
            rejected_messages, tokenize=False)
        tokens_chosen = tokenizer.encode_plus(
            prompt_plus_chosen_response, **kwargs)
        tokens_rejected = tokenizer.encode_plus(
            prompt_plus_rejected_response, **kwargs)

        if 'GRM' in model_name:
            # add label mask for sft and dpo training
            prompt = example['chosen'][:-1]
            prompt_template = tokenizer.apply_chat_template(
                prompt, tokenize=False, add_generation_prompt=True)
            tokens_prompt = tokenizer.encode_plus(
                prompt_template, **kwargs)['input_ids'][0]
            label_chosen = tokens_chosen["input_ids"][0].clone()
            label_chosen[:len(tokens_prompt)] = -100
            label_rejected = tokens_rejected["input_ids"][0].clone()
            label_rejected[:len(tokens_prompt)] = -100
            return {
                "input_ids_chosen": tokens_chosen["input_ids"][0], "attention_mask_chosen": tokens_chosen["attention_mask"][0],
                "input_ids_rejected": tokens_rejected["input_ids"][0], "attention_mask_rejected": tokens_rejected["attention_mask"][0],
                "label_chosen": label_chosen, 'label_rejected': label_rejected
            }
        else:
            return {
                "input_ids_chosen": tokens_chosen["input_ids"][0], "attention_mask_chosen": tokens_chosen["attention_mask"][0],
                "input_ids_rejected": tokens_rejected["input_ids"][0], "attention_mask_rejected": tokens_rejected["attention_mask"][0],
            }

    ds = Dataset.from_list(ds)

    ds = ds.map(formatting_func, batched=False, num_proc=16)

    remove_columns = []
    for col in ds.column_names:
        if 'input' not in col and 'attention' not in col and 'label' not in col:
            remove_columns.append(col)
    ds = ds.remove_columns(remove_columns)
    ds.set_format(type="torch")
    return ds


def load_train_eval_dataset(data_path, tokenizer, size=None, mode='', model_name=''):
    dataset = build_dataset(
        data_path, tokenizer, split='train', size=size, model_name=model_name)
    dataset_split = dataset.train_test_split(test_size=0.01)
    train_dataset, eval_dataset = dataset, dataset_split['test']
    return train_dataset, eval_dataset

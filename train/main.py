import pathlib
import torch
import transformers
from args import ModelArguments, DataArguments, TrainingArguments
from data import make_supervised_data_module, PairwiseDataCollator
from pairwise_trainer import PairwiseTrainer
from utils import create_prompt_et, create_prompt_sft, set_local_rank


def train():

    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    set_local_rank(training_args.local_rank)

    config = transformers.AutoConfig.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        trust_remote_code=model_args.trust_remote_code,
    )
    config.use_cache = False

    MODEL_CLS = transformers.AutoModelForCausalLM

    model = MODEL_CLS.from_pretrained(
        model_args.model_name_or_path,
        config=config,
        cache_dir=training_args.cache_dir,
        trust_remote_code=model_args.trust_remote_code,
        ignore_mismatched_sizes=True,
    )

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side=model_args.padding_side,
        use_fast=False,
        trust_remote_code=model_args.trust_remote_code,
    )


    model = model.to(torch.bfloat16) if training_args.bf16 else model.to(torch.float16)
    if tokenizer.pad_token != tokenizer.unk_token:
        tokenizer.pad_token = tokenizer.unk_token

    instruction_et = create_prompt_et(model_args.model_type)
    instruction_sft = create_prompt_sft(model_args.model_type)

    data_module = make_supervised_data_module(
        tokenizer=tokenizer,
        data_args=data_args,
        instruction_et=instruction_et,
        instruction_sft=instruction_sft
    )

    data_collator = PairwiseDataCollator()

    trainer = PairwiseTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        data_collator=data_collator,
        **data_module
    )


    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()

    model.config.use_cache = True
    trainer.save_state()
    if trainer.is_deepspeed_enabled:
        trainer.save_model()
    else:
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
        from torch.distributed.fsdp import StateDictType, FullStateDictConfig

        save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
        with FSDP.state_dict_type(
            trainer.model, StateDictType.FULL_STATE_DICT, save_policy
        ):
            trainer.save_model()


if __name__ == "__main__":
    train()

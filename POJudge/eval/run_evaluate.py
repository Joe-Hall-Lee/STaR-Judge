import argparse
import copy
import json
import os
import random
import warnings
from collections import defaultdict
from pathlib import Path


from prometheus_eval.prompts import ABS_SYSTEM_PROMPT, REL_SYSTEM_PROMPT
from prometheus_eval.prompts import RELATIVE_PROMPT as R2R_PROMPT
from parser import parse_output
from utils import calculate_results, extract_sections
from prometheus_eval.vllm import VLLM
from tqdm import tqdm
from transformers import AutoTokenizer

DEBUG = False
CACHE_DIR = ""


class EvalDataLoader:
    # List of allowed filenames
    ALLOWED_FILES = [
        "feedback_collection_ood_test.json",
        "feedback_collection_test.json",
        "preference_collection_ood_test.json",
        "flask_eval.json",
        "hhh_alignment_eval.json",
        "mt_bench_eval.json",
        "mt_bench_human_judgement_eval.json",
        "vicuna_eval.json",
        "autoj_pairwise.json",
        "alpaca_eval.json",
    ]

    def __init__(self, data_name):
        """
        Initializes the EvalDataLoader with the name of the data file (without extension).

        :param data_name: The name of the data file to load (without '.json').
        """
        # Construct the filename by appending '.json' extension
        filename = f"{data_name}.json"

        # Check if the constructed filename is in the list of allowed files
        if filename not in self.ALLOWED_FILES:
            raise ValueError(
                f"Filename '{filename}' is not allowed. Please choose from the allowed list."
            )

        # Use __file__ to determine the directory of the current script and construct the absolute path
        self.data_name = data_name
        script_dir = os.path.dirname(__file__)
        self.data_path = os.path.join(script_dir, "data")
        self.file_path = os.path.join(
            script_dir, "data", filename
        )  # Assuming the files are in a 'data' subdirectory
        self.records = []

    def _read_records(self):
        """
        Reads and parses JSON objects from the file. Supports both a single JSON object/array
        for the entire file and one JSON object per line.
        """
        try:
            with open(self.file_path, "r") as file:
                # Attempt to load the entire file content as a single JSON object/array
                try:
                    self.records = json.load(file)
                except json.JSONDecodeError:
                    # If the above fails, revert to reading the file line by line
                    file.seek(0)  # Reset file pointer to the beginning
                    self.records = [json.loads(line)
                                    for line in file if line.strip()]
            print(
                f"Successfully loaded {len(self.records)} records from {self.file_path}."
            )
        except FileNotFoundError:
            print(f"Error: The file '{self.file_path}' was not found.")
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON from the file '{self.file_path}': {e}")

    def _parse_records(self):
        """
        Augments each record with additional key-values extracted from the 'instruction' field using the extract_sections function.
        """
        if self.data_name in [
            "flask_eval",
            "mt_bench_eval",
            "vicuna_eval",
            "feedback_collection_ood_test",
        ]:
            for record in self.records:
                if (
                    isinstance(record, dict) and "instruction" in record
                ):  # Check if the record is the expected format
                    record["instruction"] = record["instruction"].strip().rstrip(
                        '",')
                    extracted_sections = extract_sections(
                        record["instruction"])
                    record.update(extracted_sections)

        else:
            raise NotImplementedError(
                "Parsing records for this data is not implemented yet."
            )

    def get_records(self):
        """
        Returns the list of parsed JSON records.

        :return: A list of dictionaries, each representing a JSON object.
        """
        self._read_records()
        self._parse_records()
        return self.records


# Moddel inference (Use offline batching)
def batch_completions_with_retries(
    model,
    inputs,
    params,
    batch_size,
    mode,
    parse_output,
    max_retries=5,
):
    # DEBUG: Debugging purposes
    if DEBUG:
        inputs = inputs[:10]
    batched_outputs = []

    # Adjust batch size to fit the number of inputs
    # VLLM supports adaptive batch size already
    batch_size = len(inputs)
    total_batches = len(inputs) // batch_size + (
        1 if len(inputs) % batch_size > 0 else 0
    )
    total_len = len(inputs)

    print("Processing initial batches...")
    for i in tqdm(
        range(0, len(inputs), batch_size), total=total_batches, desc="Initial Batches"
    ):
        batch_inputs = inputs[i : i + batch_size]
        batch_outputs = model.completions(batch_inputs, **params, use_tqdm=True)
        batched_outputs.extend(batch_outputs)

    # Identify failed instances and prepare for retries
    to_retry_inputs = []
    to_retry_indices = []
    for i, output in enumerate(batched_outputs):
        feedback, score = parse_output(output, mode=mode)
        if feedback is None:  # Parsing failed
            to_retry_inputs.append(inputs[i])
            to_retry_indices.append(i)

    # Retry logic with progress bar
    retries = 0
    while to_retry_inputs and retries < max_retries:
        retries += 1
        print(f"Retrying failed batches: Attempt {retries}/{max_retries}")
        retry_outputs = []
        for i in tqdm(
            range(0, len(to_retry_inputs), batch_size), desc=f"Retry Attempt {retries}"
        ):
            batch_inputs = to_retry_inputs[i : i + batch_size]
            batch_outputs = model.completions(batch_inputs, **params, use_tqdm=True)

            assert len(batch_outputs) == len(batch_inputs)
            retry_outputs.extend(batch_outputs)

        new_to_retry_inputs = []
        new_to_retry_indices = []
        for idx, (retry_idx, output) in enumerate(zip(to_retry_indices, retry_outputs)):
            feedback, score = parse_output(output, mode=mode)
            if feedback is None:  # Still failing
                new_to_retry_inputs.append(to_retry_inputs[idx])
                new_to_retry_indices.append(to_retry_indices[idx])
            else:
                batched_outputs[retry_idx] = output  # Update with successful retry

        to_retry_inputs = new_to_retry_inputs
        to_retry_indices = new_to_retry_indices

    # Final aggregation and printing
    outputs_len = len(batched_outputs)
    print(f"Processed {outputs_len}/{total_len} instances.")

    if outputs_len < total_len:
        warnings.warn("Some instances failed to generate feedback.")
        warnings.warn("They will be written as None in the output file.")
        raise Exception(
            f"Failed to generate feedback for {total_len - outputs_len} instances."
        )

    feedbacks = []
    scores = []

    for output in tqdm(batched_outputs, desc="Finalizing"):
        feedback, score = parse_output(output, mode=mode)
        if feedback is not None and score is not None:
            feedbacks.append(feedback)
            scores.append(score)
        else:
            feedbacks.append(feedback)
            scores.append(3)
            print(
                f"Parsing failed for output: {output}. Feedback: {feedback}, Score: {score}"
            )

    return feedbacks, scores


def collect_and_zip_feedbacks_and_scores(
    model, inputs, records, params, parse_output, batch_size=128, runs=3, mode="a2a"
):
    all_feedbacks = []
    all_scores = []

    # Execute batch_completions_with_retries multiple times and collect results
    for _ in range(runs):
        print(f"Starting run: {len(all_feedbacks) + 1}/{runs}")
        feedbacks, scores = batch_completions_with_retries(
            model, inputs, params, batch_size, mode, parse_output
        )

        if mode == "a2r":
            _scores = copy.deepcopy(scores)

            _accepted_scores = [_scores[i] for i in range(len(_scores)) if i % 2 == 0]
            _rejected_scores = [_scores[i] for i in range(len(_scores)) if i % 2 != 0]

            to_retry_inputs = []
            to_retry_indices = []
            assert len(_accepted_scores) == len(_rejected_scores)
            for i in range(len(_accepted_scores)):
                if _accepted_scores[i] is None or _rejected_scores[i] is None:
                    continue
                if (
                    _accepted_scores[i] == _rejected_scores[i]
                    and records[i]["tie"] != 1
                ):
                    to_retry_inputs.append(inputs[i * 2])
                    to_retry_indices.append(i * 2)
                    to_retry_inputs.append(inputs[i * 2 + 1])
                    to_retry_indices.append(i * 2 + 1)

            retries = 0
            while to_retry_inputs and retries < 2:
                retries += 1
                print(f"Retrying a2r: Attempt {retries}/2")

                _, retry_scores = batch_completions_with_retries(
                    model, to_retry_inputs, params, batch_size, mode, parse_output
                )

                _accepted_scores = [
                    retry_scores[i] for i in range(len(retry_scores)) if i % 2 == 0
                ]
                _rejected_scores = [
                    retry_scores[i] for i in range(len(retry_scores)) if i % 2 != 0
                ]

                new_to_retry_inputs = []
                new_to_retry_indices = []

                for i in range(len(_accepted_scores)):
                    if _accepted_scores[i] is None or _rejected_scores[i] is None:
                        continue
                    k = to_retry_indices[i * 2] // 2
                    if (
                        _accepted_scores[i] == _rejected_scores[i]
                        and records[k]["tie"] != 1
                    ):
                        new_to_retry_inputs.append(to_retry_inputs[i * 2])
                        new_to_retry_indices.append(to_retry_indices[i * 2])
                        new_to_retry_inputs.append(to_retry_inputs[i * 2 + 1])
                        new_to_retry_indices.append(to_retry_indices[i * 2 + 1])
                    else:
                        scores[to_retry_indices[i * 2]] = _accepted_scores[i]
                        scores[to_retry_indices[i * 2 + 1]] = _rejected_scores[i]

                to_retry_inputs = new_to_retry_inputs
                to_retry_indices = new_to_retry_indices

        all_feedbacks.append(feedbacks)
        all_scores.append(scores)

    # Zip feedbacks and scores
    zipped_feedbacks = list(zip(*all_feedbacks))
    zipped_scores = list(zip(*all_scores))

    # Combine feedbacks for each input across runs
    combined_feedbacks = [list(feedback_group) for feedback_group in zipped_feedbacks]
    combined_scores = [list(score_group) for score_group in zipped_scores]

    if mode == "a2r":
        accepted_feedbacks = [
            combined_feedbacks[i] for i in range(len(combined_feedbacks)) if i % 2 == 0
        ]
        rejected_feedbacks = [
            combined_feedbacks[i] for i in range(len(combined_feedbacks)) if i % 2 != 0
        ]

        accepted_scores = [
            combined_scores[i] for i in range(len(combined_scores)) if i % 2 == 0
        ]
        rejected_scores = [
            combined_scores[i] for i in range(len(combined_scores)) if i % 2 != 0
        ]

        combined_feedbacks = list(zip(accepted_feedbacks, rejected_feedbacks))
        combined_scores = list(zip(accepted_scores, rejected_scores))

    elif mode == "a2a":
        pass
    elif mode == "r2r":
        pass
    else:
        raise ValueError("Invalid mode. Must be 'a2a', 'a2r', or 'r2r'.")

    return combined_feedbacks, combined_scores


def prepare_inputs(records, tokenizer, mode="a2a"):
    inputs = []
    # System prompt is the same for all records

    if mode == "a2a":
        system_message = ABS_SYSTEM_PROMPT
        for record in records:
            # TODO: Check if tokenizer.chat_template is correct or tokenizer.default_chat_template is correct
            if tokenizer.chat_template is not None: 
                if "system" in tokenizer.chat_template:
                    messages = [
                        {"role": "system", "content": system_message},
                        {"role": "user", "content": record["instruction"]},
                    ]
            else:
                messages = [
                    {"role": "user", "content": system_message + record["instruction"]},
                ]
            input_str = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            inputs.append(input_str)
    elif mode == "a2r":
        system_message = ABS_SYSTEM_PROMPT
        for record in records:
            if "system" in tokenizer.default_chat_template:
                messages_A = [
                    {"role": "system", "content": system_message},
                    {
                        "role": "user",
                        "content": record["chosen_instruction"],
                    },
                ]

                messages_B = [
                    {"role": "system", "content": system_message},
                    {
                        "role": "user",
                        "content": record["rejected_instruction"],
                    },
                ]

            else:
                messages_A = [
                    {
                        "role": "user",
                        "content": system_message + record["chosen_instruction"],
                    },
                ]

                messages_B = [
                    {
                        "role": "user",
                        "content": system_message + record["rejected_instruction"],
                    },
                ]

            input_str_A = tokenizer.apply_chat_template(
                messages_A, tokenize=False, add_generation_prompt=True
            )

            input_str_B = tokenizer.apply_chat_template(
                messages_B, tokenize=False, add_generation_prompt=True
            )
            # odd index: chosen, even index: rejected
            inputs.append(input_str_A)
            inputs.append(input_str_B)
    elif mode == "r2r":
        system_message = REL_SYSTEM_PROMPT
        for record in records:
            orig_instruction = record["orig_instruction"]
            score_rubric = record["score_rubric"].split("\n")[0]
            response_A = record["orig_response_A"]
            response_B = record["orig_response_B"]

            input_str = R2R_PROMPT.format(
                orig_instruction=orig_instruction,
                response_A=response_A,
                response_B=response_B,
                score_rubric=score_rubric,
            )
            input_str = input_str.strip()
            if "system" in tokenizer.chat_template:
                messages = [
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": input_str},
                ]
            else:
                messages = [{"role": "user", "content": system_message + input_str}]

            input_str = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )

            inputs.append(input_str)
    else:
        raise ValueError("Invalid mode. Must be 'a2a', 'a2r', or 'r2r'.")

    random_inputs = random.sample(inputs, 3)
    for input_str in random_inputs:
        print("Random input:")
        print(input_str)
        print()

    return inputs


def main(
    model_name,
    eval_data_names: list,
    force_rerun=False,
    debug=False
):
    cache_dir = CACHE_DIR
    model_id = model_name.split("/")[-1]
    data_path = os.path.join(os.path.dirname(__file__), "outputs")
    report_path = os.path.join(os.path.dirname(__file__), "reports")

    global DEBUG
    DEBUG = debug

    print("Running evaluation...")
    print(f"Debug Mode: {DEBUG}")

    print(f"Model Name: {model_name}")
    print(f"Eval Data Names: {eval_data_names}")
    print(f"Force Rerun: {force_rerun}")
    print(f"Cache Dir: {cache_dir}")
    print(f"Data Path: {data_path}")
    print(f"Report Path: {report_path}")

    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
    model = VLLM(model_name)

    eval_runs = []
    for eval_data_name in eval_data_names:
        eval_runs.append((eval_data_name, "a2a", 0.0))

    overall_results = defaultdict(dict)

    for eval_data_name, mode, temperature in eval_runs:
        result_key = f"{eval_data_name}_{mode}_temp{temperature}"
        print(f"Running inference for {eval_data_name} in {mode} mode...")

        data_loader = EvalDataLoader(eval_data_name)
        records = data_loader.get_records()

        output_file_path = os.path.join(
            data_path,
            f"{model_id}-outputs",
            f"{result_key}_output.json",
        )

        output_path = Path(output_file_path)

        if output_path.exists() and not force_rerun:
            print("Output file already exists. Skipping inference.")
            sub_results = calculate_results(output_file_path, mode=mode)
            print(sub_results)
            overall_results[result_key] = sub_results
            continue

        output_path.parent.mkdir(parents=True, exist_ok=True)

        inputs = prepare_inputs(records, tokenizer, mode=mode)

        assert parse_output is not None

        params = {
            "max_tokens": 1024,
            "repetition_penalty": 1.03,
            "best_of": 1,
            "temperature": temperature,
            "top_p": 0.9,
        }

        feedbacks, scores = collect_and_zip_feedbacks_and_scores(
            model,
            inputs,
            records,
            params,
            parse_output,
            batch_size=1024,
            # batch_size=1, # [DEBUG] Use batch_size=1 when debugging
            runs=1,
            mode=mode,
        )

        with output_path.open("w") as file:
            for i, record in enumerate(records):
                record["prometheus_output"] = feedbacks[i]
                record["prometheus_score"] = scores[i]
                file.write(json.dumps(record) + "\n")

        sub_results = calculate_results(output_file_path, mode=mode)
        print(sub_results)
        overall_results[result_key] = sub_results

    def format_results(results):
        for eval_name, eval_data in results.items():
            print(f"{eval_name}:")
            for category, values in eval_data.items():
                if isinstance(values, float):
                    # Format averages with .4f
                    print(f"  {category}: {values*100:.2f}")
                else:
                    print(f"  {category}:")
                    for metric, value in values.items():
                        # Format correlation values with .3f
                        # import pdb; pdb.set_trace()
                        if isinstance(value, float):
                            print(f"    {metric}: {value:.3f}")
                        else:
                            print(f"    {metric}: {value}")

    format_results(overall_results)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run model evaluation.")
    parser.add_argument(
        "--model_name",
        type=str,
        default="/H1/zhouhongli/JudgePO/models/Llama-2-7b-chat-hf",
        help="Name of the model to evaluate",
    )
    parser.add_argument(
        "--eval_data_names",
        nargs="+",  # This allows multiple eval data names to be provided
        default=[
            "vicuna_eval",
            "mt_bench_eval"
        ],
        help="List of evaluation data names",
    )
    parser.add_argument(
        "--rerun",
        action="store_true",
        help="Use system prompt during evaluation",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        default=False,
        help="Enable debug mode"
    )

    args = parser.parse_args()

    main(
        args.model_name,
        args.eval_data_names,
        force_rerun=args.rerun,
        debug=args.debug
    )

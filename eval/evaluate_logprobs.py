import vllm
import torch
import json
import os
import argparse


model = None


def batched_evaluation(
    prompts,
    temperature=0.0,
    top_p=1.0,
):
    sampling_params = vllm.SamplingParams(
        temperature=temperature,
        max_tokens=1,
        top_p=top_p,
        prompt_logprobs=0,
    )
    global model
    if model is None:
        torch.cuda.empty_cache()
        print("Start loading VLLM model!")
        model = vllm.LLM(model=args.model_name_or_path,
                         gpu_memory_utilization=0.7)
        print("VLLM model loaded!")

    pred_list = model.generate(prompts, sampling_params)

    prompt_logprobs = []
    for pl in pred_list:
        if pl is None or pl.prompt_logprobs is None:
            prompt_logprobs.append([0])
        else:
            prompt_logprobs.append([list(lp.items())[0][1]
                                   for lp in pl.prompt_logprobs[1:]])

    return prompt_logprobs


def calculate_consistency(subset, temperature=0.0, top_p=1.0):
    prompts = []
    predictions = []
    labels = []

    for item in subset:
        prompts.append(item['instruction'] + item['response1'])
        prompts.append(item['instruction'] + item['response2'])
        predictions.append(item['result']['orig']['prediction'])
        predictions.append(item['result']['swap']['prediction'])
        labels.append(item['label'])

    prompt_logprobs = batched_evaluation(prompts, temperature, top_p)

    pred_scores = []
    for pl in prompt_logprobs:
        total = 0.0
        for item in pl:
            if isinstance(item, (int, float)):
                total += item
            else:
                total += item.logprob
        # pred_scores.append(total / len(pl))
        pred_scores.append(total)

    # Split the scores for response1 and response2
    pred_scores_a = pred_scores[0::2]
    pred_scores_b = pred_scores[1::2]

    # Create a new list to store predictions based on scores
    logprobs_predictions = []

    logprobs_labels = []
    for score_a, score_b in zip(pred_scores_a, pred_scores_b):
        if score_a > score_b:
            logprobs_predictions.append(1)  # Response1 is chosen
            logprobs_predictions.append(2)

            logprobs_labels.append(1)
        else:
            logprobs_predictions.append(2)  # Response2 is chosen
            logprobs_predictions.append(1)

            logprobs_labels.append(2)

    # Calculate accuracy against the original predictions
    correct_count = sum(1 for i in range(len(
        predictions)) if predictions[i] == logprobs_predictions[i])

    consistency_rate = correct_count / (len(predictions))

    # Calculate accuracy against the original labels
    logprobs_labels_accuracy = sum(1 for i in range(
        len(labels)) if labels[i] == logprobs_labels[i]) / len(labels)

    return consistency_rate, logprobs_labels_accuracy, logprobs_predictions


def load_datasets_from_files(data_dir, selected_files):
    datasets = {}
    for filename in selected_files:
        filepath = os.path.join(data_dir, f"{filename}.json")
        if os.path.exists(filepath):
            with open(filepath, 'r') as f:
                dataset = json.load(f)
                for subset_name in dataset.keys():
                    subset = []
                    for i, row in enumerate(dataset[subset_name]):
                        subset.append(row)
                    datasets[subset_name] = subset
        else:
            print(f"Warning: {filepath} does not exist.")
    return datasets


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-name-or-path', type=str, required=True,
                        help='Path to the model checkpoint')
    parser.add_argument('--data-dir', type=str, required=True,
                        help='Directory containing dataset JSON files')
    parser.add_argument('--files', type=str, nargs='+', required=True,
                        help='List of dataset filenames (without .json)')
    parser.add_argument('--temperature', type=float, default=0.0,
                        help='Temperature for sampling')
    parser.add_argument('--top-p', type=float, default=1.0,
                        help='Top-p sampling threshold')

    args = parser.parse_args()
    datasets = load_datasets_from_files(args.data_dir, args.files)

    all_consistency_rates = []
    all_logprobs_labels_accuracies = []

    for dataset_name, dataset in datasets.items():
        consistency_rate, logprobs_labels_accuracy, logprobs = calculate_consistency(
            dataset, args.temperature, args.top_p)

        all_consistency_rates.append((dataset_name, consistency_rate))
        all_logprobs_labels_accuracies.append(
            (dataset_name, logprobs_labels_accuracy))

    for dataset_name, rate in all_consistency_rates:
        print(f"Consistency Rate for {dataset_name}: {rate}")
    for dataset_name, rate in all_logprobs_labels_accuracies:
        print(f"logprobs_labels Accuracy for {dataset_name}: {rate}")

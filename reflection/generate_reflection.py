import json
import argparse
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer


def generate_reflection_with_vllm(result_file, distill_et_file, output_file, data_key, model_name, temperature, max_tokens):
    """
    使用 vllm 批量生成反思，并构造指定格式的数据。

    Args:
        result_file (str): 原始模型结果文件路径。
        distill_et_file (str): Distill ET 结果文件路径。
        output_file (str): 输出文件路径。
        data_key (str): 用于提取数据的键。
        model_name (str): vllm 模型名称。
        temperature (float): 生成随机性温度。
        max_tokens (int): 最大生成的 token 数量。
    """
    # 加载数据
    with open(result_file, 'r') as f1:
        data1 = json.load(f1)
    with open(distill_et_file, 'r') as f2:
        data2 = json.load(f2)

    # 确保两个文件的长度一致
    if len(data1[data_key]) != len(data2[data_key]):
        raise ValueError(f"两个文件的 '{data_key}' 数据长度不一致，无法比较！")

    # 初始化 vllm 模型和 tokenizer
    llm = LLM(model=model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    sampling_params = SamplingParams(
        temperature=temperature, max_tokens=max_tokens, skip_special_tokens=False)

    # 构造 prompts 列表
    original_prompts = []
    distill_prompts = []
    data_pairs = []
    different_predictions_count = 0  # 添加计数器

    for item1, item2 in zip(data1[data_key], data2[data_key]):
        if item1["result"]["orig"]["prediction"] == item2["result"]["orig"]["prediction"]:
            continue
        different_predictions_count += 1  # 统计预测不同的数据数量
        instruction = item1["instruction"]
        response1 = item1["response1"]
        response2 = item1["response2"]

        # 处理原始模型的结果
        original_prompt = construct_prompt(instruction, response1, response2)
        original_prompts.append(original_prompt)

        # 处理 Distill ET 的结果
        distill_result = item2["result"]["orig"]["prediction"]
        distill_conclusion = "Output (a)" if distill_result == 1 else "Output (b)"

        # 构建 Distill ET 的 prompt
        distill_prompt = construct_prompt_with_distill_result(
            instruction, response1, response2, distill_conclusion)
        distill_prompts.append(distill_prompt)
        data_pairs.append((item1, item2))

    if not original_prompts:
        print("没有需要生成的数据，已跳过。")
        return


    # Tokenize prompts without special tokens
    original_tokens = [tokenizer.encode(
        prompt, add_special_tokens=False) for prompt in original_prompts]
    distill_tokens = [tokenizer.encode(
        prompt, add_special_tokens=False) for prompt in distill_prompts]

    # Batch generate responses
    original_results = llm.generate(
        prompt_token_ids=original_tokens, sampling_params=sampling_params)
    distill_results = llm.generate(
        prompt_token_ids=distill_tokens, sampling_params=sampling_params)
    
    # Print the first conversation prompt and its tokenized form
    print(
        f"Sampled original conversation:\n{tokenizer.decode(original_tokens[0], skip_special_tokens=False)}", end='')
    # Print the first generated text and its tokenized form
    print(f"{original_results[0].outputs[0].text.strip()}")

    # Print the first conversation prompt and its tokenized form
    print(
        f"\n\nSampled distill conversation:\n{tokenizer.decode(distill_tokens[0], skip_special_tokens=False)}", end='')
    # Print the first generated text and its tokenized form
    print(f"{distill_results[0].outputs[0].text.strip()}")


    # 构造新数据
    formatted_data = []
    for (item1, item2), original_result, distill_result in zip(data_pairs, original_results, distill_results):
        original_generated_text = original_result.outputs[0].text.strip()
        original_conclusion, original_critique = extract_conclusion_and_critique(
            original_generated_text)

        distill_critique = distill_result.outputs[0].text.strip()
        distill_result_prediction = item2["result"]["orig"]["prediction"]
        distill_conclusion_generated = "Output (a)" if distill_result_prediction == 1 else "Output (b)"

        if "Output (a)" in original_conclusion and item1["label"] == 1:
            if distill_conclusion_generated == "Output (a)":
                continue
            chosen = original_critique + "\nTherefore, Output (a) is better."
            rejected = distill_critique + "\nTherefore, Output (b) is better."
        elif "Output (b)" in original_conclusion and item1["label"] == 2:
            if distill_conclusion_generated == "Output (b)":
                continue
            chosen = original_critique + "\nTherefore, Output (b) is better."
            rejected = distill_critique + "\nTherefore, Output (a) is better."
        elif "Output (a)" in original_conclusion and item1["label"] == 2:
            if distill_conclusion_generated == "Output (a)":
                continue
            chosen = distill_critique + "\nTherefore, Output (b) is better."
            rejected = original_critique + "\nTherefore, Output (a) is better."
        elif "Output (b)" in original_conclusion and item1["label"] == 1:
            if distill_conclusion_generated == "Output (b)":
                continue
            chosen = distill_critique + "\nTherefore, Output (a) is better."
            rejected = original_critique + "\nTherefore, Output (b) is better."
        else:
            continue


        # 构造符合格式的数据
        formatted_data.append({
            "system": "You are a helpful assistant in evaluating the quality of the outputs for a given instruction. Your goal is to select the best output for the given instruction.",
            "instruction": f"""After giving a brief explanation, select the Output (a) or Output (b) that is better for the given instruction. The two outputs are generated by two different AI chatbots respectively.

Here are some rules of the evaluation:
(1) You should prioritize evaluating whether the output honestly/precisely/closely executes the instruction, then consider its helpfulness, accuracy, level of detail, harmlessness, etc.
(2) Outputs should NOT contain more/less than what the instruction asks for, as such outputs do NOT precisely execute the instruction.
(3) You should avoid any potential bias and your judgment should be as objective as possible. For example, the order in which the outputs were presented should NOT affect your judgment, as Output (a) and Output (b) are **equally likely** to be the better.

You should first provide a brief explanation of your evaluation, and then always end your response with either "Therefore, Output (a) is better." or "Therefore, Output (b) is better." verbatim.
Do NOT say both / neither are good.
Do NOT output any other words.

# Instruction:
{item1["instruction"]}

# Output (a):
{item1["response1"]}

# Output (b):
{item1["response2"]}

# Decision (Give a brief explanation of your evaluation followed by either "Therefore, Output (a) is better." or "Therefore, Output (b) is better." verbatim. In your explanation, you should always use "Output (a)" or "Output (b)" to refer to the two outputs respectively.):""",
            "chosen": chosen,
            "rejected": rejected
        })

    # 保存结果到文件
    with open(output_file, 'w') as f:
        json.dump(formatted_data, f, indent=4)

    print(f"生成的数据已保存到 {output_file}")
    print(f"原始预测不同的数据数量: {different_predictions_count}")
    print(f"最终保存的数据数量: {len(formatted_data)}")


def construct_prompt(instruction, response1, response2):
    """
    构造原始模型的 prompt。

    Args:
        instruction (str): 指令。
        response1 (str): 输出 a。
        response2 (str): 输出 b。

    Returns:
        str: prompt 文本。
    """
    return f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are a helpful assistant in evaluating the quality of the outputs for a given instruction. Your goal is to select the best output for the given instruction.<|eot_id|><|start_header_id|>user<|end_header_id|>
Select the Output (a) or Output (b) that is better for the given instruction. The two outputs are generated by two different AI chatbots respectively.

Here are some rules of the evaluation:
(1) You should prioritize evaluating whether the output honestly/precisely/closely executes the instruction, then consider its helpfulness, accuracy, level of detail, harmlessness, etc.
(2) Outputs should NOT contain more/less than what the instruction asks for, as such outputs do NOT precisely execute the instruction.
(3) You should avoid any potential bias and your judgment should be as objective as possible. For example, the order in which the outputs were presented should NOT affect your judgment, as Output (a) and Output (b) are **equally likely** to be the better.

On the first line you should answer using ONLY "Output (a)" or "Output (b)" and do NOT output any other words.
On the second line provide a brief explanation for your choice.
Do NOT say both / neither are good.


# Instruction:
{instruction}

# Output (a):
{response1}

# Output (b):
{response2}

# Decision (You should answer using ONLY "Output (a)" or "Output (b)" on the first line, then provide a brief explanation on the second line. In your explanation, you should always use "Output (a)" or "Output (b)" to refer to the two outputs respectively.):<|eot_id|><|start_header_id|>assistant<|end_header_id|>"""


def construct_prompt_with_distill_result(instruction, response1, response2, distill_conclusion):
    """
    构造 Distill ET 的 prompt，并拼接 Distill ET 的结论。

    Args:
        instruction (str): 指令。
        response1 (str): 输出 a。
        response2 (str): 输出 b。
        distill_conclusion (str): Distill ET 预测的结论 (Output (a) 或 Output (b))。

    Returns:
        str: prompt 文本。
    """
    return f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are a helpful assistant in evaluating the quality of the outputs for a given instruction. Your goal is to select the best output for the given instruction.<|eot_id|><|start_header_id|>user<|end_header_id|>
Select the Output (a) or Output (b) that is better for the given instruction. The two outputs are generated by two different AI chatbots respectively.

Here are some rules of the evaluation:
(1) You should prioritize evaluating whether the output honestly/precisely/closely executes the instruction, then consider its helpfulness, accuracy, level of detail, harmlessness, etc.
(2) Outputs should NOT contain more/less than what the instruction asks for, as such outputs do NOT precisely execute the instruction.
(3) You should avoid any potential bias and your judgment should be as objective as possible. For example, the order in which the outputs were presented should NOT affect your judgment, as Output (a) and Output (b) are **equally likely** to be the better.

On the first line you should answer using ONLY "Output (a)" or "Output (b)" and do NOT output any other words.
On the second line provide a brief explanation for your choice.
Do NOT say both / neither are good.


# Instruction:
{instruction}

# Output (a):
{response1}

# Output (b):
{response2}

# Decision (You should answer using ONLY "Output (a)" or "Output (b)" on the first line, then provide a brief explanation on the second line. In your explanation, you should always use "Output (a)" or "Output (b)" to refer to the two outputs respectively.):<|eot_id|><|start_header_id|>assistant<|end_header_id|><|eot_id|><|start_header_id|>assistant<|end_header_id|>{distill_conclusion}\n"""


def extract_conclusion_and_critique(generated_text):
    """
    从生成的文本中提取结论和反思。

    Args:
        generated_text (str): 模型生成的完整文本。

    Returns:
        tuple: (conclusion, critique)
    """
    try:
        # 提取结论
        first_newline_index = generated_text.find('\n')
        conclusion = generated_text[:first_newline_index].strip()

        # 提取解释
        critique = generated_text[first_newline_index:].strip()
        return conclusion, critique

    except ValueError:
        return None, generated_text.strip()


if __name__ == "__main__":
    # 使用 argparse 解析命令行参数
    parser = argparse.ArgumentParser(
        description="Use vllm to generate reflections and format the data.")
    parser.add_argument("--result", type=str, required=True,
                        help="Path to the original model results JSON file.")
    parser.add_argument("--result_distill_et", type=str, required=True,
                        help="Path to the Distill ET results JSON file.")
    parser.add_argument("--output", type=str, required=True,
                        help="Path to the output file.")
    parser.add_argument("--data", type=str, default="arena",
                        help="Key for accessing data (default: arena).")
    parser.add_argument("--model", type=str, default="gpt-4",
                        help="Name of the vllm model to use (default: gpt-4).")
    parser.add_argument("--temperature", type=float, default=0.7,
                        help="Temperature for generation randomness.")
    parser.add_argument("--max_tokens", type=int, default=512,
                        help="Maximum number of tokens to generate.")

    # 解析命令行参数
    args = parser.parse_args()

    # 调用主函数
    generate_reflection_with_vllm(
        result_file=args.result,
        distill_et_file=args.result_distill_et,
        output_file=args.output,
        data_key=args.data,
        model_name=args.model,
        temperature=args.temperature,
        max_tokens=args.max_tokens
    )


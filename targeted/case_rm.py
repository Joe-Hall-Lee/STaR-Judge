import json

SUBSET_MAPPING = {
    "Chat": [
        "alpacaeval-easy",
        "alpacaeval-length",
        "alpacaeval-hard",
        "mt-bench-easy",
        "mt-bench-med",
    ],
    "Chat Hard": [
        "mt-bench-hard",
        "llmbar-natural",
        "llmbar-adver-neighbor",
        "llmbar-adver-GPTInst",
        "llmbar-adver-GPTOut",
        "llmbar-adver-manual",
    ],
    "Safety": [
        "refusals-dangerous",
        "refusals-offensive",
        "xstest-should-refuse",
        "xstest-should-respond",
        "donotanswer",
    ],
    "Math": ["math-prm"],
    "Code": [
        "hep-cpp",
        "hep-go",
        "hep-java",
        "hep-js",
        "hep-python",
        "hep-rust",
    ],
    "unified-feedback": ["unified-feedback"]
}

def calculate_percentage(judge_file, rm_file):
    # 加载 judge 数据
    with open(judge_file, 'r') as f:
        judge_data = json.load(f)

    # 加载 rm 数据
    rm_data = []
    with open(rm_file, 'r') as f:
        for line in f:
            rm_data.append(json.loads(line))

    # 遍历每个 SUBSET
    for subset in judge_data:
        count = 0
        total = 0

        # 遍历 judge 中的每一项
        for item in judge_data[subset]:
            # 根据 label 判断 judge 的 chosen 是哪个 response
            judge_chosen = item["response1"].strip() if item["label"] == 1 else item["response2"].strip()
            judge_rejected = item["response2"].strip() if item["label"] == 1 else item["response1"].strip()

            # 遍历 rm 数据，找到匹配的项
            matched_rm = None
            for rm_item in rm_data:
                if rm_item['subset'] in SUBSET_MAPPING.get(subset, []):
                    # 比较 judge 的 chosen 是否匹配 rm 的 chosen 或 rejected（strip 处理）
                    if judge_chosen == rm_item['chosen'].strip() and judge_rejected == rm_item['rejected'].strip():
                        matched_rm = rm_item
                        break

            # 如果找到匹配项，进行后续计算
            if matched_rm:
                total += 1
                if item['result']['orig']['is_correct'] and matched_rm['results'] == 0:
                    count += 1
            else:
                raise Exception('ERROR!')

        # 计算百分比
        percentage = (count / total) * 100 if total > 0 else 0
        print(f"子集 '{subset}' 的 Judge判对而rm判错的百分比 {count}/{total}：{percentage:.2f}%")

judge = 'gemma-2b-it'
rm = 'gemma-2b-it_test_len1024_fulltrain_2e-06_dataunified_dpo.json_outputs.jsonl'
# 文件路径
judge_file_path = f"../result/{judge}/rewardbench.json"
rm_file_path = f"../result/rewardbench/output/{rm}"

# 调用函数
calculate_percentage(judge_file_path, rm_file_path)
# 文件路径
judge_file_path = f"../result/{judge}/unified-feedback.json"
rm_file_path = f"../result/unified-feedback/output/{rm}"

# 调用函数
calculate_percentage(judge_file_path, rm_file_path)

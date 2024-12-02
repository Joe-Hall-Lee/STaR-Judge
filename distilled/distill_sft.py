import json
import vllm
import argparse

parser = argparse.ArgumentParser(description='VLLM推理脚本')
parser.add_argument('--model', type=str, required=True, help='模型路径')
parser.add_argument('--input_file', type=str, required=True, help='输入文件路径')
parser.add_argument('--output_file', type=str, required=True, help='输出文件路径')
args = parser.parse_args()

# 加载模型
model = vllm.LLM(model=args.model)

# 加载输入文件
with open(args.input_file, 'r', encoding='utf-8') as f:
    data = json.load(f)

# 定义采样参数
sampling_params = vllm.SamplingParams(
    temperature=0.0,
    max_tokens=1024,
    top_p=1.0
)

# 进行批量推理
pred_list = model.generate([item['instruction'] for item in data], sampling_params)

# 更新输出
for i, item in enumerate(data):
    item['output'] = pred_list[i].outputs[0].text  # 提取生成的文本

# 保存输出文件
with open(args.output_file, 'w', encoding='utf-8') as f:
    json.dump(data, f, ensure_ascii=False, indent=4)

from transformers import AutoTokenizer, GPT2LMHeadModel 
import torch
# 加载预训练模型和分词器
model_path = r"E:\gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = GPT2LMHeadModel.from_pretrained(model_path).eval()
# 遍历所有子模块并注册钩子
hooks = []


# 手动构建张量
inputs = torch.tensor([[0]])
model.eval()
model.generate(inputs, max_length=50)

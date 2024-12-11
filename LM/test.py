from transformers import AutoTokenizer, GPT2LMHeadModel 
import torch
def print_tensor_elements(tensor, label="Tensor", num_elements=5):
    if isinstance(tensor, torch.Tensor):
        elements = tensor.flatten()
        print(f"{label}:")
        print("First 5 elements:", elements[:num_elements].tolist())
        print("Last 5 elements:", elements[-num_elements:].tolist() if len(elements) >= num_elements else elements.tolist())
    else:
        print(f"{label}: Not a Tensor")

def print_nested_tensors(data, prefix=""):
    if isinstance(data, (tuple, list)):
        for idx, item in enumerate(data):
            new_prefix = f"{prefix}.Output {idx+1}" if prefix else f"Output {idx+1}"
            print_nested_tensors(item, new_prefix)
    elif isinstance(data, dict):
        for key, value in data.items():
            new_prefix = f"{prefix}.{key}" if prefix else key
            print_nested_tensors(value, new_prefix)
    else:
        print_tensor_elements(data, label=prefix)

def hook_fn(module, input, output):
    print(f"Layer: {module.__class__.__name__}")

    # 处理输入张量
    if input and isinstance(input[0], torch.Tensor):
        print_tensor_elements(input[0], label="Input")

    # 处理输出张量
    print_nested_tensors(output)

    print("-" * 50)

# 加载预训练模型和分词器
model_path = r"E:\gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = GPT2LMHeadModel.from_pretrained(model_path).eval()
# 遍历所有子模块并注册钩子
hooks = []
for name, module in model.named_modules():
    if not isinstance(module, (torch.nn.ModuleList, torch.nn.Sequential)):
        hooks.append(module.register_forward_hook(hook_fn))
        # if name == 'transformer.h.0.attn':
        #     hooks.append(module.register_forward_hook(hook_fn))

# 手动构建张量
inputs = torch.tensor([[0]])
model.eval()
model.generate(inputs, max_length=50)

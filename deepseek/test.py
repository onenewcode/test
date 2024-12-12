from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
def print_tensor_elements(tensor, label="Tensor", num_elements=5):
    if isinstance(tensor, torch.Tensor):
        elements = tensor.flatten()
        print(f"{label}: shape={tensor.shape}")
        print("First 5 elements:", elements[:num_elements].tolist())
        print("Last 5 elements:", elements[-num_elements:].tolist() if len(elements) >= num_elements else elements.tolist())
    else:
        print(f"{label}: Not a Tensor")

def hook_fn(module, input, output):
    print(f"Layer: {module.__class__.__name__}")

    # 处理输入张量
    if input and isinstance(input[0], torch.Tensor):
        print_tensor_elements(input[0], label="Input")
    else:
        print("error")
    # 处理输出张量
    if input and isinstance(output[0], torch.Tensor):
        print_tensor_elements(output[0], label="Input")
    else:
        print("error")


    print("-" * 50)

# 加载预训练模型和分词器
model_path = "/home/ztf/cpm"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model =  AutoModelForCausalLM.from_pretrained(model_path,trust_remote_code=True)
model.eval()
# 遍历所有子模块并注册钩子
hooks = []
for name, module in model.named_modules():
    if not isinstance(module, (torch.nn.ModuleList, torch.nn.Sequential)):
        hooks.append(module.register_forward_hook(hook_fn))
        # if name == 'transformer.ln_f':
        #     hooks.append(module.register_forward_hook(hook_fn))

# 手动构建张量并进行推理
inputs = "Once upon a time,"
generated_tokens = torch.tensor([[59422]])
# 将文本转换为模型输入
input_ids = tokenizer(inputs, return_tensors='pt').input_ids

# 使用模型进行推理
with torch.no_grad():  # 确保推理过程中不计算梯度以节省内存
    outputs = model.generate(generated_tokens, max_length=2, do_sample=True)
print(outputs)
for i in range(outputs.shape[0]):  # 遍历所有生成的序列
    print(tokenizer.decode(outputs[i], skip_special_tokens=True))
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# 加载预训练模型和分词器
model_path = r"F:\edged\qwen2"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path).eval()

# 设定初始输入
input_context = "你好，世界"  # 您可以更改这个字符串为任何您想要开始推理的内容

# 将输入编码为模型可理解的格式
input_ids = tokenizer.encode(input_context, return_tensors='pt')

# 设定生成参数
max_length = 100  # 最大生成长度

# 开始生成循环
with torch.no_grad():  # 禁用梯度计算以节省内存
    for _ in range(max_length):
        # 获取模型预测的下一个token
        outputs = model(input_ids)
        next_token_logits = outputs.logits[:, -1, :]  # 取最后一个token的logits
        
        # 直接选择最可能的下一个token
        next_token = torch.argmax(next_token_logits, dim=-1).unsqueeze(0)

        # 将新token添加到序列中
        input_ids = torch.cat([input_ids, next_token], dim=-1)

        # 实时输出新生成的字符
        generated_text = tokenizer.decode(input_ids[0])
        print(generated_text, end='\r')  # 使用'\r'覆盖上一行，实现“实时”效果

        # 如果模型生成了结束标记，则停止
        if tokenizer.eos_token_id and next_token.item() == tokenizer.eos_token_id:
            break

print("\n推理完成")
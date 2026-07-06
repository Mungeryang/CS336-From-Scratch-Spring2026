# LoRA微调 核心配置文件
# 学习资料参考： 李宏毅老师 - 机器学习

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import peft

max_seq_length = 2048

model, tokenizer = AutoModelForCausalLM.from_pretrained(
    model_name = " ",
    dtype = torch.bfloat16,
    max_seq_length = max_seq_length,
)







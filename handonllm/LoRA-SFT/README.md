## 基于Llama-7B的模型LoRA微调实践

### 📊 训练数据集 gsm8k_train.jsonl

gsm8k_train.jsonl 一共 7473 行,每一行是一个 QA 样本:

```shell
{
  "question": "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?",

  "answer": "Natalia sold 48/2 = <<48/2=24>>24 clips in May.\nNatalia sold 48+24 = <<48+24=72>>72 clips altogether in April and May.\n#### 72"
}

```

这是普通的QA格式：

```shell
question: 题目
answer: 推理过程 + 最终答案
```

但是 Llama-3.2-1B-Instruct 是 chat/instruct 模型，所以训练时要变成：

```shell
user: 问题
assistant: 回答
```

### 🔧 训练样本构造

微调 Llama-3.2-1B-Instruct 模型前, 需要将 gsm8k_train.jsonl 训练数据集中 QA 数据集转换为 few-shot 的 chat 格式, 每条训练样本都会额外随机抽 1 条 GSM8K 样本作为 few-shot 示例:

```shell
[
  {
    "role": "user",
    "content": "Q: For every 12 cans you recycle, you receive $0.50..."
  },
  {
    "role": "assistant",
    "content": "A: There are 144/12 = <<144/12=12>>12 ...\n#### 12"
  },
  {
    "role": "user",
    "content": "Q: Natalia sold clips to 48 of her friends in April... Let's think step by step. At the end, you MUST write the answer as an integer after '####'."
  },
  {
    "role": "assistant",
    "content": "A: Natalia sold 48/2 = <<48/2=24>>24 clips in May.\nNatalia sold 48+24 = <<48+24=72>>72 clips altogether in April and May.\n#### 72"
  }
]
```

一条训练样本实际变成了：

```shell
示例问题 -> 示例答案
当前问题 + 格式要求 -> 当前答案
```

这不是单纯的 question -> answer，而是 1-shot prompt -> answer。

### 加上 Llama chat template:

```python
train_sample = tokenizer.apply_chat_template(chats, tokenize=False)
```

这一步会把上面的 Python list 转成 Llama-3.2-Instruct 风格的文本:

```shell
<|start_header_id|>user<|end_header_id|>

Q: For every 12 cans you recycle...
<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>

A: There are 144/12 = #### 12
<|eot_id|>
<|start_header_id|>user<|end_header_id|>

Q: Natalia sold clips ... Let's think step by step. At the end, you MUST write the answer as an integer after '####'.
<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>

A: Natalia sold 48/2 = ...
#### 72
<|eot_id|>
```

### 📦 包装 formatted_gsm8k

```shell
Dataset([
    {
        "text": "<|start_header_id|>user<|end_header_id|>\n\nQ: ...<|eot_id|>..."
    },
    {
        "text": "<|start_header_id|>user<|end_header_id|>\n\nQ: ...<|eot_id|>..."
    },
    ...
])
```

SFTTrainer 会从每条样本中取 text 字段，然后 tokenizer 编码成 token，做 causal language modeling 训练。

训练的本质是 LoRA + SFT + GSM8K chain-of-thought 格式监督微调。




 



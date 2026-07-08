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

### 训练样本构造





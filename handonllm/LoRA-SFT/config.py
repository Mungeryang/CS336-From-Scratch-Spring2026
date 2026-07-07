# LoRA微调 核心配置文件
# 学习资料参考： 李宏毅老师 - 机器学习

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import (
    LoraConfig,
    get_peft_model,
    PeftModel
)

import os
import json
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
device = torch.device('cuda:0')

from datasets import Dataset
from trl import SFTConfig, SFTTrainer
from tqdm import tqdm
import random
random.seed(42)

# 加载模型 和 tokenizer 的 backbone
model = AutoModelForCausalLM.from_pretrained(
    'meta-llama/Llama-3.2-1B-Instruct',
    dtype=torch.bfloat16,
    device_map='auto'
)
tokenizer = AutoTokenizer.from_pretrained(
    'meta-llama/Llama-3.2-1B-Instruct'
)

tokenizer.add_special_tokens({'pad_token': '[PAD]'})

# LoRA 配置
peft_config = LoraConfig(
    r=8,
    lora_alpha=16,
    bias='none',
    task_type='CAUSAL_LM',
    target_modules=['up_proj','down_proj','gate_proj','q_proj','k_proj','v_proj','o_proj']
)
peft_model = get_peft_model(model, peft_config)


# 数据转换
def load_jsonlines(file_name: str):
    f = open(file_name, 'r')
    return [json.loads(line) for line in f]

def nshot_chats(nshot_data: list, n: int, question: str, answer: any, mode: str) -> dict: # Function to create n-shot chats
    if mode not in ['train', 'test']:
        raise AssertionError('Undefined Mode!!!')

    chats = []
    # TODO: Use fixed few-shot examples
    for qna in random.sample(nshot_data, n): # Samples n examples from the n-shot data
        chats.append(
            {
                'role': 'user',
                'content': f'Q: {qna["question"]}' # Creates a user message with the question
            }
        )
        chats.append(
            {
                'role': 'assistant',
                'content': f'A: {qna["answer"]}' # Creates an assistant message with the answer
            }
        )

    chats.append(
        {
            'role': 'user',
            'content': f'Q: {question} Let\'s think step by step. At the end, you MUST write the answer as an integer after \'####\'.' # Creates a user message with the question and instructions
        }
    )
    if mode == 'train':
        chats.append(
            {
                'role': 'assistant',
                'content': f'A: {answer}' # Creates an assistant message with the answer
            }
        )

    return chats # Returns the list of chats

gsm8k_train = load_jsonlines('/home/ygm/gsm8k_train.jsonl')

formatted_gsm8k = []
TRAIN_N_SHOT = 1 # TODO: Give model more examples
max_token_len = 0 # Record token length of dataset and prevent data from truncation
for qna in gsm8k_train: # Iterates over the GSM8K training data
    chats = nshot_chats(nshot_data=gsm8k_train, n=TRAIN_N_SHOT, question=qna['question'], answer=qna['answer'], mode='train') # Creates n-shot chats for the current example
    train_sample = tokenizer.apply_chat_template(chats, tokenize=False) # Applies the chat template to the chats
    train_sample = train_sample[train_sample.index("<|eot_id|>") + len("<|eot_id|>"):] # Remove Cutting Knowledge Date in prompt template
    formatted_gsm8k.append( # Appends the formatted example to the list
        {
            'text': train_sample # Adds the text of the example
        }
    )
    max_token_len = max(max_token_len, len(tokenizer(train_sample)['input_ids'])) # Updates the maximum token length

formatted_gsm8k = Dataset.from_list(formatted_gsm8k)

training_args = SFTConfig(
    seed=1126,
    data_seed=1126,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    optim="paged_adamw_32bit",
    num_train_epochs=1, # TODO: If you use fixed few-shot examples, increase epoch
    logging_strategy="steps",
    logging_steps=0.1,
    save_strategy="steps",
    save_steps=0.1,
    lr_scheduler_type='linear',
    learning_rate=2e-4, # TODO: Decrease learning rate
    # TODO: Add weight decay
    bf16=True,
    group_by_length=True,
    max_seq_length=max_token_len,
    dataset_text_field='text',
    report_to='none',
)

trainer = SFTTrainer( # Creates the SFT trainer
    model=peft_model,
    train_dataset=formatted_gsm8k,
    peft_config=peft_config,
    processing_class=tokenizer,
    args=training_args,
)
trainer.train() # Starts the training proces







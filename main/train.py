from modelscope import AutoTokenizer, AutoModelForCausalLM
from transformers import DataCollatorForSeq2Seq, TrainingArguments, Trainer
from datasets import Dataset
import pandas as pd
import torch
from peft import LoraConfig, TaskType, get_peft_model
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.abspath(os.path.join(current_dir, "../model/LLM-Research/Meta-Llama-3___1-8B-Instruct"))

df = pd.read_json('data/huanhuan.json')
ds = Dataset.from_pandas(df)

tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False, trust_remote_code=True)
tokenizer.padding_side = "right" 
tokenizer.pad_token = tokenizer.eos_token

def process_func(example):
    MAX_LENGTH = 384
    input_ids, attention_mask, labels = [], [], []
    instruction = tokenizer(f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nCutting Knowledge Date: December 2023\nToday Date: 26 Jul 2025\n\n现在你要扮演皇帝身边的女人--甄嬛<|eot_id|><|start_header_id|>user<|end_header_id|>{example["instruction"] + example["input"]}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n", add_special_tokens=False)
    response = tokenizer(f"{example["output"]}<|eot_id|>", add_special_tokens=False)
    input_ids = instruction["input_ids"] + response["input_ids"] + [tokenizer.eos_token_id]
    attention_mask = instruction["attention_mask"] + response["attention_mask"] + [1]
    labels = [-100] * len(instruction["input_ids"]) + response["input_ids"] + [tokenizer.eos_token_id]
    if len(input_ids) > MAX_LENGTH:
        input_ids = input_ids[:MAX_LENGTH]
        attention_mask = attention_mask[:MAX_LENGTH]
        labels = labels[:MAX_LENGTH]
        
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels
    }
    
tokenizer_ds = ds.map(process_func, remove_columns=ds.column_names)

model = AutoModelForCausalLM.from_pretrained(model_path, 
                                             torch_dtype=torch.half,
                                             load_in_4bit=True,
                                             bnb_4bit_compute_dtype=torch.half,
                                             bnb_4bit_quant_type="nf4",
                                             bnb_4bit_use_double_quant=True
                                             )

model.config.pad_token_id = tokenizer.pad_token_id

config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    inference_mode=False,
    r=8,
    lora_alpha=32,
    lora_dropout=0.1
)

model.enable_input_require_grads()

model = get_peft_model(model, config)

model.print_trainable_parameters()

args = TrainingArguments(
    output_dir='./output/llama3_1_instruct_lora',
    per_device_train_batch_size=1,
    gradient_accumulation_steps=16,
    logging_steps=10,
    num_train_epochs=3,
    save_steps=100,
    learning_rate=1e-4,
    save_on_each_node=True,
    gradient_checkpointing=True,
    optim="paged_adamw_32bit"
)

trainer = Trainer(
    args=args,
    model=model,
    train_dataset=tokenizer_ds,
    data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer,padding=True)
)

trainer.train()
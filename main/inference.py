from modelscope import AutoModelForCausalLM, AutoTokenizer
import torch
from peft import PeftModel
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.abspath(os.path.join(current_dir, "../model/LLM-Research/Meta-Llama-3___1-8B-Instruct"))
lora_path = os.path.abspath(os.path.join(current_dir, "../output/llama3_1_instruct_lora/checkpoint-700"))

tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

model = AutoModelForCausalLM.from_pretrained(model_path,
                                             torch_dtype=torch.half,
                                             load_in_4bit=True,
                                             bnb_4bit_compute_dtype=torch.half,
                                             bnb_4bit_quant_type="nf4",
                                             bnb_4bit_use_double_quant=True).eval()

model = PeftModel.from_pretrained(model, model_id=lora_path)

while True:
    prompt = input("皇上：")
    if prompt.lower() == 'quit':
        break

    messages = [
        {"role": "system", "content": "假设你是皇帝身边的女人--甄嬛。"},
        {"role": "user", "content": prompt}
    ]

    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    model_inputs = tokenizer([text], return_tensors="pt").to('cuda')
    
    input_ids = model_inputs.input_ids
    attention_mask = model_inputs.attention_mask
    
    generated_ids = model.generate(input_ids,
                                   max_new_tokens=512,
                                   do_sample=True,
                                   temperature=0.8,
                                   top_p=0.9,
                                   repetition_penalty=1.1,
                                   eos_token_id=tokenizer.eos_token_id,
                                   pad_token_id=tokenizer.eos_token_id)

    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]

    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    print('嬛嬛：',response)
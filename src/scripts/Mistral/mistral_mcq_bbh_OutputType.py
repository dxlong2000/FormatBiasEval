from transformers import AutoModelForCausalLM, AutoTokenizer
import time
import json
from tqdm import tqdm
import pandas as pd
from random import sample
import csv 
import random

device = "cuda" # the device to load the model onto

model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2")
tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2")

#########
def get_mistral_answer(prompt, role=""):
    messages = [{"role": "user", "content": role + " " + prompt}]
    encodeds = tokenizer.apply_chat_template(messages, return_tensors="pt")
    model_inputs = encodeds.to(device)
    model.to(device)
    generated_ids = model.generate(model_inputs, max_new_tokens=1024, top_p=0.9)
    assistant_message = tokenizer.batch_decode(generated_ids)[0]
    return assistant_message.split("[/INST]")[1].strip().split("</s>")[0]

###### Loading data ######
def extract_character_answer(answer, cot=False):
    if cot:
        if "<ANSWER>" in answer and "</ANSWER>" in answer:
            answer = answer.split("<ANSWER>")[1].split("</ANSWER>")[0].strip()
        else: 
            answer = " "
    return answer[0].strip()

def extract_textual_value_answer(answer, cot=False):
    if cot:
        if "<ANSWER>" in answer and "</ANSWER>" in answer:
            answer = answer.split("<ANSWER>")[1].split("</ANSWER>")[0].strip()
        else: 
            answer = " "
    return answer.strip()

saved_rows = []
with open("./EvalData/mcq_bbh_test_1.csv") as file:
    csvreader = csv.reader(file)
    header = next(csvreader) # topic,question,choices_texts,input_data,correct_key,correct_text
    
    zero_shot_character_cnt = 0
    zero_shot_value_cnt = 0
    
    cot_character_cnt = 0
    cot_text_cnt = 0
    
    all_cnt = 0
    for row in tqdm(csvreader):
        input_data = row[list(header).index("input_data")]
        correct_key = row[list(header).index("correct_key")]
        correct_text = row[list(header).index("correct_text")]
        
        zero_shot_character_instruct = "Answer the following multiple-choice question by outputting only the designated character identifier."
        zero_shot_text_val_instruct = "Answer the following multiple-choice question by outputting the textual value of your choice without the character identifier."  
        
        cot_character_instruct = "Answer the following multiple-choice question step-by-step by outputting only the designated character identifier."
        cot_text_val_instruct = "Answer the following multiple-choice question step-by-step by outputting the textual value of your choice without the character identifier."  
        cot_wrapping = "Wrap your final answer by <ANSWER> and </ANSWER>."
                
        raw_zs_character_result = get_mistral_answer(zero_shot_character_instruct + "\n" + input_data)
        zero_shot_character_result = extract_character_answer(raw_zs_character_result)
        zero_shot_character_cnt += int(zero_shot_character_result==correct_key)
        
        raw_zs_text_result = get_mistral_answer(zero_shot_text_val_instruct + "\n" + input_data)
        zero_shot_text_result = extract_textual_value_answer(raw_zs_text_result)
        zero_shot_value_cnt += int(zero_shot_text_result==correct_text)
        
        raw_cot_character_result = get_mistral_answer(cot_character_instruct + "\n" + input_data + "\n" + cot_wrapping)
        cot_character_result = extract_character_answer(raw_cot_character_result, cot=True)
        cot_character_cnt += int(cot_character_result==correct_key)
        
        raw_cot_text_result = get_mistral_answer(cot_text_val_instruct + "\n" + input_data + "\n" + cot_wrapping)
        cot_text_result = extract_character_answer(raw_cot_text_result, cot=True)
        cot_text_cnt += int(cot_text_result==correct_text)
        
        all_cnt += 1
        
        tmp_row = row 
        tmp_row.extend([
            raw_zs_character_result, zero_shot_character_result, 
            raw_zs_text_result, zero_shot_text_result, 
            raw_cot_character_result, cot_character_result, 
            raw_cot_text_result, cot_text_result
        ])
        saved_rows.append(tmp_row)

with open("./src/output/mistral_mcq_bbh_OutputType.csv", "w"):
    csvwriter = csv.writer(file)
    csvwriter.writerow([
        "topic", "question", "choices_texts", "input_data",
        "correct_key", "correct_text", 
        "raw_zs_character_result", "zs_character_result", 
        "raw_zs_text_result", "zs_text_result", 
        "raw_cot_character_result", "cot_character_result", 
        "raw_cot_text_result", "cot_text_result"
    ])
    csvwriter.writerows(saved_rows)

print(f"zero_shot_character_cnt: {zero_shot_character_cnt/all_cnt}")
print(f"zero_shot_value_cnt: {zero_shot_value_cnt/all_cnt}")
print(f"cot_character_cnt: {cot_character_cnt/all_cnt}")
print(f"cot_value_cnt: {cot_text_cnt/all_cnt}")    
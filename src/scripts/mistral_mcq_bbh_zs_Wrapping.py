from transformers import AutoModelForCausalLM, AutoTokenizer
import time
import json
from tqdm import tqdm
import pandas as pd
from random import sample
import csv 
import random
import re

device = "cuda" # the device to load the model onto

model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2")
tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2")

#########
def get_mistral_answer(prompt, role=""):
    messages = [{"role": "user", "content": role + " " + prompt}]
    encodeds = tokenizer.apply_chat_template(messages, return_tensors="pt")
    model_inputs = encodeds.to(device)
    model.to(device)
    generated_ids = model.generate(model_inputs, max_new_tokens=1500, num_beams=5)
    assistant_message = tokenizer.batch_decode(generated_ids)[0]
    return assistant_message.split("[/INST]")[1].strip().split("</s>")[0]

###### Loading data ######
def extract_character_free_form_answer(answer, cot=False):
    if "<ANSWER>" in answer and "</ANSWER>" in answer:
        answer = answer.split("<ANSWER>")[1].split("</ANSWER>")[0].strip()
    else: 
        answer = " "
    return answer[0].strip()
    
def extract_character_bolding_answer(answer, cot=False):
    try:
        pattern = r'\*\*(.*?)\*\*'
        matches = re.findall(pattern, answer)
        return matches[0][0]
    except:
        return ""

def extract_character_italicizing_answer(input_string):
    try:
        pattern = r'\*(.*?)\*'
        matches = re.findall(pattern, input_string)
        return matches[0][0]
    except:
        return ""

def extract_character_brackets_answer(input_string):
    try:
        pattern = r'\[\[(.*?)\]\]'
        matches = re.findall(pattern, input_string)
        return matches[0][0]
    except:
        return ""

def extract_character_parenthesis_answer(input_string):
    try:
        pattern = r'\(\((.*?)\)\)'
        matches = re.findall(pattern, input_string)
        return matches[0][0]
    except:
        return ""

def extract_character_placeholder_answer(input_text):
    try:
        pattern = r'So the answer is: ([^}]+)'
        matches = re.findall(pattern, input_text)
        return matches[0][0]
    except:
        return ""

def extract_character_quoting_answer(input_string):
    try:
        pattern = r'\'\'\'(.*?)\'\'\'|\"\"\"(.*?)\"\"\"'
        matches = re.findall(pattern, input_string, re.DOTALL)
        return [match[0] or match[1] for match in matches][0][0]
    except:
        return ""

saved_rows = []
with open("./EvalData/mcq_bbh_test_1.csv") as file:
    csvreader = csv.reader(file)
    header = next(csvreader) # topic,question,choices_texts,input_data,correct_key,correct_text
    
    zs_free_form_cnt = 0
    zs_bolding_cnt = 0
    zs_italicizing_cnt = 0
    zs_brackets_cnt = 0
    zs_parenthesis_cnt = 0
    zs_placeholder_cnt = 0
    zs_quoting_cnt = 0
    
    all_cnt = 0
    for row in tqdm(csvreader):
        input_data = row[list(header).index("input_data")]
        correct_key = row[list(header).index("correct_key")]
        
        zero_shot_character_instruct = "Answer the following multiple-choice question by outputting only the designated character identifier."
        cot_character_instruct = "Answer the following multiple-choice question step-by-step by outputting only the designated character identifier."
        
        free_form_wrapping = "Wrap your final answer by <ANSWER> and </ANSWER>."
        bolding_wrapping = "Wrap your final answer in bold by enclosing it with double asterisks."
        italicizing_wrapping = "Wrap your final answer in italics by enclosing it with single asterisks."
        brackets_wrapping = "Wrap your final answer using double square brackets."
        parenthesis_wrapping = "Wrap your final answer using double parentheses."
        placeholder_wrapping = "Wrap your final answer by filling in the placeholder below: 'So the answer is: \{\{placeholder\}\}'"
        quoting_wrapping = "Wrap your final answer using triple quotation marks."

        raw_zs_free_form_result = get_mistral_answer(zero_shot_character_instruct + "\n" + input_data + "\n" + free_form_wrapping)
        zs_free_form_result = extract_character_free_form_answer(raw_zs_free_form_result)
        zs_free_form_cnt += int(zs_free_form_result==correct_key)
        
        raw_zs_bolding_result = get_mistral_answer(zero_shot_character_instruct + "\n" + input_data + "\n" + bolding_wrapping)
        zs_bolding_result = extract_character_bolding_answer(raw_zs_bolding_result)
        zs_bolding_cnt += int(zs_bolding_result==correct_key)
        
        raw_zs_italicizing_result = get_mistral_answer(zero_shot_character_instruct + "\n" + input_data + "\n" + italicizing_wrapping)
        zs_italicizing_result = extract_character_italicizing_answer(raw_zs_italicizing_result)
        zs_italicizing_cnt += int(zs_italicizing_result==correct_key)
        
        raw_zs_brackets_result = get_mistral_answer(zero_shot_character_instruct + "\n" + input_data + "\n" + brackets_wrapping)
        zs_brackets_result = extract_character_brackets_answer(raw_zs_brackets_result)
        zs_brackets_cnt += int(zs_brackets_result==correct_key)
        
        raw_zs_parenthesis_result = get_mistral_answer(zero_shot_character_instruct + "\n" + input_data + "\n" + parenthesis_wrapping)
        zs_parenthesis_result = extract_character_parenthesis_answer(raw_zs_parenthesis_result)
        zs_parenthesis_cnt += int(zs_parenthesis_result==correct_key)
        
        raw_zs_placeholder_result = get_mistral_answer(zero_shot_character_instruct + "\n" + input_data + "\n" + placeholder_wrapping)
        zs_placeholder_result = extract_character_placeholder_answer(raw_zs_placeholder_result)
        zs_placeholder_cnt += int(zs_placeholder_result==correct_key)
        
        raw_zs_quoting_result = get_mistral_answer(zero_shot_character_instruct + "\n" + input_data + "\n" + quoting_wrapping)
        zs_quoting_result = extract_character_quoting_answer(raw_zs_quoting_result)
        zs_quoting_cnt += int(zs_quoting_result==correct_key)
        
        all_cnt += 1
        
        tmp_row = row 
        tmp_row.extend([
            raw_zs_free_form_result, zs_free_form_result, 
            raw_zs_bolding_result, zs_bolding_result, 
            raw_zs_italicizing_result, zs_italicizing_result, 
            raw_zs_brackets_result, zs_brackets_result,
            raw_zs_parenthesis_result, zs_parenthesis_result,
            raw_zs_placeholder_result, zs_placeholder_result,
            raw_zs_quoting_result, zs_quoting_result
        ])
        saved_rows.append(tmp_row)

with open("./src/output/mistral_mcq_bbh_zs_Wrapping.csv", "w"):
    csvwriter = csv.writer(file)
    csvwriter.writerow([
        "topic", "question", "choices_texts", "input_data",
        "correct_key", "correct_text", 
        "raw_zs_free_form_result", "zs_free_form_result", 
        "raw_zs_bolding_result", "zs_bolding_result", 
        "raw_zs_italicizing_result", "zs_italicizing_result", 
        "raw_zs_brackets_result", "zs_brackets_result",
        "raw_zs_parenthesis_result", "zs_parenthesis_result",
        "raw_zs_placeholder_result", "zs_placeholder_result",
        "raw_zs_quoting_result", "zs_quoting_result"
    ])
    csvwriter.writerows(saved_rows)

print(f"zs_free_form_cnt: {zs_free_form_cnt/all_cnt}")
print(f"zs_bolding_cnt: {zs_bolding_cnt/all_cnt}")
print(f"zs_italicizing_cnt: {zs_italicizing_cnt/all_cnt}")
print(f"zs_brackets_cnt: {zs_brackets_cnt/all_cnt}")  
print(f"zs_parenthesis_cnt: {zs_parenthesis_cnt/all_cnt}") 
print(f"zs_placeholder_cnt: {zs_placeholder_cnt/all_cnt}") 
print(f"zs_quoting_cnt: {zs_quoting_cnt/all_cnt}")  
from transformers import AutoModelForCausalLM, AutoTokenizer
from vllm import LLM, SamplingParams
import time
import json
from tqdm import tqdm
import pandas as pd
from random import sample
import csv 
import random
import re
import torch

MODEL_SAVE_PATH = 'YOUR MODEL SAVE PATH'

sampling_params = SamplingParams(temperature=0.1, top_p=0.95, max_tokens = 256)

llm = LLM(model=MODEL_SAVE_PATH, tensor_parallel_size=4)

#########
def get_gemma_7b_answer(prompt):
    output = llm.generate(prompt, sampling_params)
    generated_text = output[0].outputs[0].text
    print(generated_text)
    return generated_text

###### Loading data ######
def extract_character_free_form_answer(answer, cot=False):
    if "<ANSWER>" in answer and "</ANSWER>" in answer:
        answer_content = answer.split("<ANSWER>")[1].split("</ANSWER>")[0].strip()
    else:
        answer_content = " "
    
    # Check if the answer_content is not empty before accessing the first character
    return answer_content[0].strip() if answer_content else " "
    
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
with open("../../../EvalData/mcq_bbh_test_1.csv") as file:
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

        raw_zs_free_form_result = get_gemma_7b_answer(zero_shot_character_instruct + "\n" + input_data + "\n" + free_form_wrapping * 1)
        zs_free_form_result = extract_character_free_form_answer(raw_zs_free_form_result)
        zs_free_form_cnt += int(zs_free_form_result==correct_key)
        
        raw_zs_bolding_result = get_gemma_7b_answer(zero_shot_character_instruct + "\n" + input_data + "\n" + bolding_wrapping * 1)
        zs_bolding_result = extract_character_bolding_answer(raw_zs_bolding_result)
        zs_bolding_cnt += int(zs_bolding_result==correct_key)
        
        raw_zs_italicizing_result = get_gemma_7b_answer(zero_shot_character_instruct + "\n" + input_data + "\n" + italicizing_wrapping *1)
        zs_italicizing_result = extract_character_italicizing_answer(raw_zs_italicizing_result)
        zs_italicizing_cnt += int(zs_italicizing_result==correct_key)
        
        raw_zs_brackets_result = get_gemma_7b_answer(zero_shot_character_instruct + "\n" + input_data + "\n" + brackets_wrapping * 1)
        zs_brackets_result = extract_character_brackets_answer(raw_zs_brackets_result)
        zs_brackets_cnt += int(zs_brackets_result==correct_key)
        
        raw_zs_parenthesis_result = get_gemma_7b_answer(zero_shot_character_instruct + "\n" + input_data + "\n" + parenthesis_wrapping * 1) 
        zs_parenthesis_result = extract_character_parenthesis_answer(raw_zs_parenthesis_result)
        zs_parenthesis_cnt += int(zs_parenthesis_result==correct_key)
        
        raw_zs_placeholder_result = get_gemma_7b_answer(zero_shot_character_instruct + "\n" + input_data + "\n" + placeholder_wrapping * 1)
        zs_placeholder_result = extract_character_placeholder_answer(raw_zs_placeholder_result)
        zs_placeholder_cnt += int(zs_placeholder_result==correct_key)
        
        raw_zs_quoting_result = get_gemma_7b_answer(zero_shot_character_instruct + "\n" + input_data + "\n" + quoting_wrapping * 1)
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

print(f"zs_free_form_cnt: {zs_free_form_cnt/all_cnt}")
print(f"zs_bolding_cnt: {zs_bolding_cnt/all_cnt}")
print(f"zs_italicizing_cnt: {zs_italicizing_cnt/all_cnt}")
print(f"zs_brackets_cnt: {zs_brackets_cnt/all_cnt}")  
print(f"zs_parenthesis_cnt: {zs_parenthesis_cnt/all_cnt}") 
print(f"zs_placeholder_cnt: {zs_placeholder_cnt/all_cnt}") 
print(f"zs_quoting_cnt: {zs_quoting_cnt/all_cnt}")  

with open("../../output/gemma_7B_it_mcq_bbh_zs_Wrapping.csv", "w") as file:
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


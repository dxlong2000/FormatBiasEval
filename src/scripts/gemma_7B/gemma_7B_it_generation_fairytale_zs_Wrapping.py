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

MODEL_SAVE_PATH = '/home/bizon/hai/gemma_7b_save'

sampling_params = SamplingParams(temperature=0.1, top_p=0.95, max_tokens = 256)

llm = LLM(model=MODEL_SAVE_PATH, tensor_parallel_size=4)

#########
def get_gemma_7b_answer(prompt):
    output = llm.generate(prompt, sampling_params)
    generated_text = output[0].outputs[0].text
    print(generated_text)
    return generated_text

###### Loading data ######
def extract_text_free_form_answer(answer, cot=False):
    if "<ANSWER>" in answer and "</ANSWER>" in answer:
        answer = answer.split("<ANSWER>")[1].split("</ANSWER>")[0].strip()
    else:
        answer = " "
    return answer.strip()

def extract_text_bolding_answer(answer, cot=False):
    try:
        pattern = r'\*\*(.*?)\*\*'
        matches = re.findall(pattern, answer)
        return matches[0]
    except:
        return ""


def extract_text_italicizing_answer(input_string):
    try:
        pattern = r'\*(.*?)\*'
        matches = re.findall(pattern, input_string)
        return matches[0]
    except:
        return ""

def extract_text_brackets_answer(input_string):
    try:
        pattern = r'\[\[(.*?)\]\]'
        matches = re.findall(pattern, input_string)
        return matches[0]
    except:
        return ""

def extract_text_parenthesis_answer(input_string):
    try:
        pattern = r'\(\((.*?)\)\)'
        matches = re.findall(pattern, input_string)
        return matches[0]
    except:
        return ""

def extract_text_placeholder_answer(input_text):
    try:
        pattern = r'So the answer is: ([^}]+)'
        matches = re.findall(pattern, input_text)
        return matches[0]
    except:
        return ""

def extract_text_quoting_answer(input_string):
    try:
        pattern = r'\'\'\'(.*?)\'\'\'|\"\"\"(.*?)\"\"\"'
        matches = re.findall(pattern, input_string, re.DOTALL)
        return [match[0] or match[1] for match in matches][0]
    except:
        return ""
    

saved_rows = []
with open("/home/bizon/hai/EvalData/generation_fairytaleqa_200.csv") as file:
    csvreader = csv.reader(file)
    header = next(csvreader) # question,cot_answer,answer,input_data
    for row in tqdm(csvreader):
        input_data = row[list(header).index("input_data")]
        answer = row[list(header).index("answer")]
        
        zero_shot_character_instruct = "Answer the following question."
        
        free_form_wrapping = "Wrap your final answer by <ANSWER> and </ANSWER>."
        bolding_wrapping = "Wrap your final answer in bold by enclosing it with double asterisks."
        italicizing_wrapping = "Wrap your final answer in italics by enclosing it with single asterisks."
        brackets_wrapping = "Wrap your final answer using double square brackets."
        parenthesis_wrapping = "Wrap your final answer using double parentheses."
        placeholder_wrapping = "Wrap your final answer by filling in the placeholder below: 'So the answer is: \{\{placeholder\}\}'"
        quoting_wrapping = "Wrap your final answer using triple quotation marks."

        raw_zs_free_form_result = get_gemma_7b_answer(zero_shot_character_instruct + "\n" + input_data + "\n" + free_form_wrapping)
        zs_free_form_result = extract_text_free_form_answer(raw_zs_free_form_result)
        
        raw_zs_bolding_result = get_gemma_7b_answer(zero_shot_character_instruct + "\n" + input_data + "\n" + bolding_wrapping)
        zs_bolding_result = extract_text_bolding_answer(raw_zs_bolding_result)
        
        raw_zs_italicizing_result = get_gemma_7b_answer(zero_shot_character_instruct + "\n" + input_data + "\n" + italicizing_wrapping)
        zs_italicizing_result = extract_text_italicizing_answer(raw_zs_italicizing_result)
        
        raw_zs_brackets_result = get_gemma_7b_answer(zero_shot_character_instruct + "\n" + input_data + "\n" + brackets_wrapping)
        zs_brackets_result = extract_text_brackets_answer(raw_zs_brackets_result)
        
        raw_zs_parenthesis_result = get_gemma_7b_answer(zero_shot_character_instruct + "\n" + input_data + "\n" + parenthesis_wrapping)
        zs_parenthesis_result = extract_text_parenthesis_answer(raw_zs_parenthesis_result)
        
        raw_zs_placeholder_result = get_gemma_7b_answer(zero_shot_character_instruct + "\n" + input_data + "\n" + placeholder_wrapping)
        zs_placeholder_result = extract_text_placeholder_answer(raw_zs_placeholder_result)
        
        raw_zs_quoting_result = get_gemma_7b_answer(zero_shot_character_instruct + "\n" + input_data + "\n" + quoting_wrapping)
        zs_quoting_result = extract_text_quoting_answer(raw_zs_quoting_result)
            
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

with open("../../output/Gemma_7b_it_generation_fairytale_zs_Wrapping_200_Shot1.csv", "w") as file:
    csvwriter = csv.writer(file)
    csvwriter.writerow([
        "story_name","content","answer","question",
        "gem_id","target","references","local_or_sum","attribute","ex_or_im","input_data",
        "raw_zs_free_form_result", "zs_free_form_result", 
        "raw_zs_bolding_result", "zs_bolding_result", 
        "raw_zs_italicizing_result", "zs_italicizing_result", 
        "raw_zs_brackets_result", "zs_brackets_result",
        "raw_zs_parenthesis_result", "zs_parenthesis_result",
        "raw_zs_placeholder_result", "zs_placeholder_result",
        "raw_zs_quoting_result", "zs_quoting_result"
    ])
    csvwriter.writerows(saved_rows)
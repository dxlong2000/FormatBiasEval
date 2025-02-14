from transformers import AutoModelForCausalLM, AutoTokenizer
import time
import json
from tqdm import tqdm
import pandas as pd
from random import sample
import csv 
import random
import re

import openai
from openai import OpenAI

key = ""
client = OpenAI(api_key = key)

def get_chatgpt_answer(input_prompt, role="You are helpful assistant!", n=1):
    response = client.chat.completions.create(
        model="gpt-3.5-turbo-0125",
        messages=[
            {"role": "system", "content": role},
            {"role": "user", "content": input_prompt},
        ],
        max_tokens=1024,
        top_p=1,
        n=n,
        frequency_penalty=0,
        presence_penalty=0
    )
    if n > 1:
        ans = []
        for idx in range(n): ans.append(response.choices[idx].message.content)
        return ans
    return response.choices[0].message.content

###### Loading data ######
def extract_character_free_form_answer(answer, cot=False):
    try:
        if "<ANSWER>" in answer and "</ANSWER>" in answer:
            answer = answer.split("<ANSWER>")[1].split("</ANSWER>")[0].strip()
        else: 
            answer = " "
        return answer[0].strip()
    except:
        return ""
    
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
with open("./EvalData/generation_fairytaleqa_200.csv") as file:
    csvreader = csv.reader(file)
    header = next(csvreader) # context,question,answer,input_data
    for row in tqdm(csvreader):
        input_data = row[list(header).index("input_data")]
        answer = row[list(header).index("answer")]
        
        zero_shot_character_instruct = "Answer the following question without any explanation."
        
        free_form_wrapping = "Wrap your final answer by <ANSWER> and </ANSWER>."
        bolding_wrapping = "Wrap your final answer in bold by enclosing it with double asterisks."
        italicizing_wrapping = "Wrap your final answer in italics by enclosing it with single asterisks."
        brackets_wrapping = "Wrap your final answer using double square brackets."
        parenthesis_wrapping = "Wrap your final answer using double parentheses."
        placeholder_wrapping = "Wrap your final answer by filling in the placeholder below: 'So the answer is: \{\{placeholder\}\}'"
        quoting_wrapping = "Wrap your final answer using triple quotation marks."

        raw_zs_free_form_result = get_chatgpt_answer(zero_shot_character_instruct + "\n" + input_data + "\n" + free_form_wrapping)
        zs_free_form_result = extract_character_free_form_answer(raw_zs_free_form_result)
        
        raw_zs_bolding_result = get_chatgpt_answer(zero_shot_character_instruct + "\n" + input_data + "\n" + bolding_wrapping)
        zs_bolding_result = extract_character_bolding_answer(raw_zs_bolding_result)
        
        raw_zs_italicizing_result = get_chatgpt_answer(zero_shot_character_instruct + "\n" + input_data + "\n" + italicizing_wrapping)
        zs_italicizing_result = extract_character_italicizing_answer(raw_zs_italicizing_result)
        
        raw_zs_brackets_result = get_chatgpt_answer(zero_shot_character_instruct + "\n" + input_data + "\n" + brackets_wrapping)
        zs_brackets_result = extract_character_brackets_answer(raw_zs_brackets_result)
        
        raw_zs_parenthesis_result = get_chatgpt_answer(zero_shot_character_instruct + "\n" + input_data + "\n" + parenthesis_wrapping)
        zs_parenthesis_result = extract_character_parenthesis_answer(raw_zs_parenthesis_result)
        
        raw_zs_placeholder_result = get_chatgpt_answer(zero_shot_character_instruct + "\n" + input_data + "\n" + placeholder_wrapping)
        zs_placeholder_result = extract_character_placeholder_answer(raw_zs_placeholder_result)
        
        raw_zs_quoting_result = get_chatgpt_answer(zero_shot_character_instruct + "\n" + input_data + "\n" + quoting_wrapping)
        zs_quoting_result = extract_character_quoting_answer(raw_zs_quoting_result)
            
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

with open("./src/output/chatgpt_fairytaleqa_zs_Wrapping_200.csv", "w") as file:
    csvwriter = csv.writer(file)
    csvwriter.writerow([
        "context", "question", "answer", "input_data",
        "raw_zs_free_form_result", "zs_free_form_result", 
        "raw_zs_bolding_result", "zs_bolding_result", 
        "raw_zs_italicizing_result", "zs_italicizing_result", 
        "raw_zs_brackets_result", "zs_brackets_result",
        "raw_zs_parenthesis_result", "zs_parenthesis_result",
        "raw_zs_placeholder_result", "zs_placeholder_result",
        "raw_zs_quoting_result", "zs_quoting_result"
    ])
    csvwriter.writerows(saved_rows)
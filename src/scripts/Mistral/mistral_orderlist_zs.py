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

def average_precision(ranklist):
    """
    Computes the average precision of a ranklist.

    Parameters:
    ranklist (list): A list of binary relevance scores (1 for relevant, 0 for irrelevant) in ranked order.

    Returns:
    float: Average Precision (AP) of the ranklist.
    """
    precision_sum = 0.0
    hits = 0.0
    for i, relevance in enumerate(ranklist):
        if relevance == 1:
            hits += 1
            precision_sum += hits / (i + 1)
    if hits == 0:
        return 0
    return precision_sum / hits

def mean_average_precision(ranklist):
    """
    Computes the mean average precision of a list of ranklists.

    Parameters:
    ranklists (list): A list of ranklists, where each ranklist is a list of binary relevance scores (1 for relevant, 0 for irrelevant) in ranked order.

    Returns:
    float: Mean Average Precision (mAP) of the ranklists.
    """
    ranklists = [ranklist]

    total_ap = 0.0
    num_ranklists = len(ranklists)
    for ranklist in ranklists:
        total_ap += average_precision(ranklist)
    return total_ap / num_ranklists

# Example usage:
# Ranklists: Each list represents the relevance scores (1 for relevant, 0 for irrelevant) of items in a ranked order.
ranklist = [1, 1, 1, 0, 1]

# Compute mAP
mAP = mean_average_precision(ranklist)
print("Mean Average Precision:", mAP)

from posixpath import expandvars
import re
import yaml
import ast

def check_binary_list(answer):
    for ans in answer:
        if ans not in [0, 1]: return False
    return True

def check_python_follow_ranking(answer, cot=False):
    if '```python' in answer and '```' in answer: answer = answer.replace('```python', '').replace('```', '')
    if cot:
        try:
            answer = answer.split("<ANSWER>")[1].split("</ANSWER>")[0].strip()
            answer = ast.literal_eval(answer)
            return check_binary_list(answer)
        except:
            return False
    try:
        answer = ast.literal_eval(answer)
        return check_binary_list(answer)
    except: return False

def extract_python_ranking(answer, cot=False):
    if '```python' in answer and '```' in answer: answer = answer.replace('```python', '').replace('```', '')
    if cot: answer = answer.split("<ANSWER>")[1].split("</ANSWER>")[0].strip()
    return ast.literal_eval(answer)

def check_newline_follow_ranking(answer, cot=False):
    if cot:
        try:
            answer = answer.split("<ANSWER>")[1].split("</ANSWER>")[0].strip().split("\n")
            answer = [int(ele) for ele in answer]
            return check_binary_list(answer)
        except: return False
    try:
        answer = answer.split("\n")
        answer = [int(ele) for ele in answer]
        return check_binary_list(answer)
    except: return False

def extract_newline_ranking(answer, cot=False):
    if cot: answer = answer.split("<ANSWER>")[1].split("</ANSWER>")[0].strip()
    answer = answer.split("\n")
    answer = [int(ele) for ele in answer]
    return answer

def check_bullet_follow_ranking(answer, cot=False):
    if cot:
        try:
            answer = answer.split("<ANSWER>")[1].split("</ANSWER>")[0].strip()
            answer = yaml.safe_load(answer)
            return check_binary_list(answer)
        except: return False
    try:
        answer = yaml.safe_load(answer)
        return check_binary_list(answer)
    except: return False

def extract_bullet_ranking(answer, cot=False):
    if cot: answer = answer.split("<ANSWER>")[1].split("</ANSWER>")[0].strip()
    answer = yaml.safe_load(answer)
    return answer

def check_special_character_follow_ranking(answer, cot=False):
    if cot:
        try:
            answer = answer.split("<ANSWER>")[1].split("</ANSWER>")[0].strip().split("<SEP>")
            answer = [int(ele) for ele in answer]
            return check_binary_list(answer)
        except: return False
    try:
        answer = answer.split("<SEP>")
        answer = [int(ele) for ele in answer]
        return check_binary_list(answer)
    except: return False

def extract_special_character_ranking(answer, cot=False):
    if cot: answer = answer.split("<ANSWER>")[1].split("</ANSWER>")[0].strip()
    answer = answer.split("<SEP>")
    answer = [int(ele) for ele in answer]
    return answer

# Zero-shot performance
from tqdm import tqdm
import csv
import yaml
import ast

python_fi = 0
special_fi = 0
bullet_fi = 0
newline_fi = 0

python_map = 0
special_map = 0
bullet_map = 0
newline_map = 0

all_cnt = 0

saved_data = []
with open("./EvalData/list_scidocs_reranking_200.csv") as file:
    csvreader = csv.reader(file)
    header = next(csvreader)
    for row in tqdm(csvreader):
        query = row[list(header).index("query")]
        positive = eval(row[list(header).index("positive")])
        negative = eval(row[list(header).index("negative")])

        all_samples = positive
        all_samples.extend(negative)

        zs_python_prompt = f"""Given a query, and a list of documents:
Query: {query}
List of documents: {str(all_samples)}

You are required to output a binary list of 1 or 0 where 1 indicates the document brings useful information to the query, and 0 indicates the document does not bring useful information to the query.

Generate your binary list as a Python list without any explanation."""

        zs_bullet_prompt = f"""Given a query, and a list of documents:
Query: {query}
List of documents: {str(all_samples)}

You are required to output a list of 1 or 0 where 1 indicates the document brings useful information to the query, and 0 indicates the document does not bring useful information to the query.

Generate your binary list using bullet points without any explanation."""

        zs_special_prompt = f"""Given a query, and a list of documents:
Query: {query}
List of documents: {str(all_samples)}

You are required to output a list of 1 or 0 where 1 indicates the document brings useful information to the query, and 0 indicates the document does not bring useful information to the query.

Generate your binary list using <SEP> to seperate elements without any explanation."""

        zs_newline_prompt = f"""Given a query, and a list of documents:
Query: {query}
List of documents: {str(all_samples)}

You are required to output a list of 1 or 0 where 1 indicates the document brings useful information to the query, and 0 indicates the document does not bring useful information to the query.

Generate your binary list such that each element is in a new line without any explanation."""

        zs_python_answer = get_mistral_answer(zs_python_prompt)
        zs_special_answer = get_mistral_answer(zs_special_prompt)
        zs_bullet_answer = get_mistral_answer(zs_bullet_prompt)
        zs_newline_answer =  get_mistral_answer(zs_newline_prompt)

        python_follow = check_python_follow_ranking(zs_python_answer)
        if python_follow:
            python_fi += 1
            extract_python = extract_python_ranking(zs_python_answer)
            python_map += mean_average_precision(extract_python)

        special_follow = check_special_character_follow_ranking(zs_special_answer)
        if special_follow:
            special_fi += 1
            extract_special = extract_special_character_ranking(zs_special_answer)
            special_map += mean_average_precision(extract_special)

        bullet_follow = check_bullet_follow_ranking(zs_bullet_answer)
        if bullet_follow:
            bullet_fi += 1
            extract_bullet = extract_bullet_ranking(zs_bullet_answer)
            bullet_map += mean_average_precision(extract_bullet)

        newline_follow = check_newline_follow_ranking(zs_newline_answer)
        if newline_follow:
            newline_fi += 1
            extract_newline = extract_newline_ranking(zs_newline_answer)
            newline_map += mean_average_precision(extract_newline)

        all_cnt += 1
        saved_data.append([query, positive, negative, zs_python_answer, zs_special_answer, zs_bullet_answer, zs_newline_answer])

print(f"python_fi: {python_fi/all_cnt}")
print(f"special_fi: {special_fi/all_cnt}")
print(f"bullet_fi: {bullet_fi/all_cnt}")
print(f"newline_fi: {newline_fi/all_cnt}")
print("===")
print(f"python_map: {python_map/all_cnt}")
print(f"special_map: {special_map/all_cnt}")
print(f"bullet_map: {bullet_map/all_cnt}")
print(f"newline_map: {newline_map/all_cnt}")

import csv

with open("./src/output/mistral_zs_scidocs_15Apr.csv", "w") as file:
    csvwriter = csv.writer(file)
    csvwriter.writerow(["query", "positive", "negative", "zs_python_answer", "zs_special_answer", "zs_bullet_answer", "zs_newline_answer"])
    csvwriter.writerows(saved_data)
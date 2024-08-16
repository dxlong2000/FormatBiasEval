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
    generated_ids = model.generate(model_inputs, max_new_tokens=1500, num_beams=5)
    assistant_message = tokenizer.batch_decode(generated_ids)[0]
    return assistant_message.split("[/INST]")[1].strip().split("</s>")[0]

def compute_f1(list1, list2):
    # Convert lists to sets for efficient intersection and union operations
    set1 = set(list1)
    set2 = set(list2)

    # Calculate True Positives, False Positives, and False Negatives
    true_positives = len(set1.intersection(set2))
    false_positives = len(set2.difference(set1))
    false_negatives = len(set1.difference(set2))

    # Calculate precision, recall, and F1 score
    precision = true_positives / (true_positives + false_positives)
    recall = true_positives / (true_positives + false_negatives)
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) != 0 else 0

    return f1_score

# Example usage
list1 = ["apple", "banana", "orange", "pear"]
list2 = ["banana", "orange", "grape", "kiwi"]

f1_score = compute_f1(list1, list2)
print("F1 Score:", f1_score)

from posixpath import expandvars
import re
import yaml
import ast

def check_python_follow_ranking(answer, cot=False):
    if '```python' in answer and '```' in answer: answer = answer.replace('```python', '').replace('```', '')
    if cot:
        try: answer = answer.split("<ANSWER>")[1].split("</ANSWER>")[0].strip()
        except: return False
    try:
        answer = ast.literal_eval(answer)
        return answer
    except: return False

def extract_python_ranking(answer, cot=False):
    if '```python' in answer and '```' in answer: answer = answer.replace('```python', '').replace('```', '')
    if cot: answer = answer.split("<ANSWER>")[1].split("</ANSWER>")[0].strip()
    return ast.literal_eval(answer)

def check_newline_follow_ranking(answer, cot=False):
    if cot:
        try: answer = answer.split("<ANSWER>")[1].split("</ANSWER>")[0].strip()
        except: return False
    try:
        answer = answer.split("\n")
        return answer
    except: return False

def extract_newline_ranking(answer, cot=False):
    if cot: answer = answer.split("<ANSWER>")[1].split("</ANSWER>")[0].strip()
    answer = answer.split("\n")
    answer = [ans for ans in answer if len(ans) >= 2]
    answer = [ans[1:].strip() if ans[0] == "-" else ans.strip() for ans in answer]
    return answer

def check_bullet_follow_ranking(answer, cot=False):
    if cot:
        try: answer = answer.split("<ANSWER>")[1].split("</ANSWER>")[0].strip()
        except: return False
    try:
        answer = yaml.safe_load(answer)
        return answer
    except: return False

def extract_bullet_ranking(answer, cot=False):
    if cot: answer = answer.split("<ANSWER>")[1].split("</ANSWER>")[0].strip()
    answer = yaml.safe_load(answer)
    return answer

def check_special_character_follow_ranking(answer, cot=False):
    if cot:
        try: answer = answer.split("<ANSWER>")[1].split("</ANSWER>")[0].strip()
        except: return False
    try:
        answer = answer.split("<SEP>")
        return answer
    except: return False

def extract_special_character_ranking(answer, cot=False):
    if cot: answer = answer.split("<ANSWER>")[1].split("</ANSWER>")[0].strip()
    answer = answer.split("<SEP>")
    return answer

# CoT performance
from tqdm import tqdm
import csv
import yaml
import ast

python_fi = 0
special_fi = 0
bullet_fi = 0
newline_fi = 0

python_F1 = 0
special_F1 = 0
bullet_F1 = 0
newline_F1 = 0

all_cnt = 0

saved_data = []
with open("./EvalData/list_keyphrases_SemEval2017_200.csv") as file:
    csvreader = csv.reader(file)
    header = next(csvreader)
    for row in tqdm(csvreader):
        id = row[list(header).index("id")]
        document = row[list(header).index("document")]
        keyphrases = eval(row[list(header).index("keyphrases")])

        wrapping = "Wrap your final list by <ANSWER> and </ANSWER>."

        cot_python_prompt = f"""Extract a list of keyphrases from the following document:
Document: {document}
Generate your list in Python format. Generate your list step-by-step. {wrapping}"""

        cot_bullet_prompt = f"""Extract a list of keyphrases from the following document:
Document: {document}
Generate your list using bullet points. Generate your list step-by-step. {wrapping}"""

        cot_special_prompt = f"""Extract a list of keyphrases from the following document:
Document: {document}
Generate your list using <SEP> to seperate elements. Generate your list step-by-step. {wrapping}"""

        cot_newline_prompt = f"""Extract a list of keyphrases from the following document:
Document: {document}
Generate your list such that each element is in a new line. Generate your list step-by-step. {wrapping}"""

        cot_python_answer = get_mistral_answer(cot_python_prompt)
        cot_special_answer = get_mistral_answer(cot_special_prompt)
        cot_bullet_answer = get_mistral_answer(cot_bullet_prompt)
        cot_newline_answer =  get_mistral_answer(cot_newline_prompt)
        all_cnt += 1
        saved_data.append([id, document, keyphrases, cot_python_answer, cot_special_answer, cot_bullet_answer, cot_newline_answer])

        python_follow = check_python_follow_ranking(cot_python_answer, cot=True)
        if python_follow:
            try:
                python_fi += 1
                extract_python = extract_python_ranking(cot_python_answer, cot=True)
                python_F1 += compute_f1(extract_python, keyphrases)
            except: pass

        special_follow = check_special_character_follow_ranking(cot_special_answer, cot=True)
        if special_follow:
            try:
                special_fi += 1
                extract_special = extract_special_character_ranking(cot_special_answer, cot=True)
                special_F1 += compute_f1(extract_special, keyphrases)
            except: pass

        bullet_follow = check_bullet_follow_ranking(cot_bullet_answer, cot=True)
        if bullet_follow:
            try:
                extract_bullet = extract_bullet_ranking(cot_bullet_answer, cot=True)
                bullet_F1 += compute_f1(extract_bullet, keyphrases)
                bullet_fi += 1
            except: pass

        newline_follow = check_newline_follow_ranking(cot_newline_answer, cot=True)
        if newline_follow:
            try:
                newline_fi += 1
                extract_newline = extract_newline_ranking(cot_newline_answer, cot=True)
                newline_F1 += compute_f1(extract_newline, keyphrases)
            except: pass

print(f"python_fi: {python_fi/all_cnt}")
print(f"special_fi: {special_fi/all_cnt}")
print(f"bullet_fi: {bullet_fi/all_cnt}")
print(f"newline_fi: {newline_fi/all_cnt}")
print("===")
print(f"python_F1: {python_F1/all_cnt}")
print(f"special_F1: {special_F1/all_cnt}")
print(f"bullet_F1: {bullet_F1/all_cnt}")
print(f"newline_F1: {newline_F1/all_cnt}")

import csv

with open("./src/output/mistral_cot_semeval2017_15Apr.csv", "w") as file:
    csvwriter = csv.writer(file)
    csvwriter.writerow(["id", "document", "keyphrases", "cot_python_answer", "cot_special_answer", "cot_bullet_answer", "cot_newline_answer"])
    csvwriter.writerows(saved_data)
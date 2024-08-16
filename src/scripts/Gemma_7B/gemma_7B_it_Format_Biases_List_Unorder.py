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
from posixpath import expandvars
import re
import yaml
import ast

MODEL_SAVE_PATH = 'YOUR MODEL SAVE PATH'

sampling_params = SamplingParams(temperature=0, top_p=0.95, min_tokens = 10, max_tokens = 2000, repetition_penalty = 1.01)

llm = LLM(model=MODEL_SAVE_PATH, tensor_parallel_size=4, dtype = "float16")

#########
def get_gemma_7b_answer(prompt):
    output = llm.generate(prompt, sampling_params)
    generated_text = output[0].outputs[0].text
    print(generated_text)
    return generated_text

def compute_f1(list1, list2):
    # Convert lists to sets for efficient intersection and union operations
    set1 = set(list1)
    set2 = set(list2)

    # Calculate True Positives, False Positives, and False Negatives
    true_positives = len(set1.intersection(set2))
    false_positives = len(set2.difference(set1))
    false_negatives = len(set1.difference(set2))

    # Calculate precision, recall, and F1 score
    precision = true_positives / (true_positives + false_positives + 1e-6)
    recall = true_positives / (true_positives + false_negatives + 1e-6)
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) != 0 else 0

    return f1_score

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

################ ZERO-SHOT PERFORMANCE ###############

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
with open("../../../EvalData/list_keyphrases_SemEval2017_200.csv") as file:
    csvreader = csv.reader(file)
    header = next(csvreader)
    for row in tqdm(csvreader):
        id = row[list(header).index("id")]
        document = row[list(header).index("document")]
        keyphrases = eval(row[list(header).index("keyphrases")])

        zs_python_prompt = f"""Extract a list of keyphrases from the following document:
Document: {document}
Generate your list as a Python list without any explanation."""

        zs_bullet_prompt = f"""Extract a list of keyphrases from the following document:
Document: {document}
Generate your list using bullet points without any explanation."""

        zs_special_prompt = f"""Extract a list of keyphrases from the following document:
Document: {document}
Generate your list using <SEP> to seperate elements without any explanation."""

        zs_newline_prompt = f"""Extract a list of keyphrases from the following document:
Document: {document}
Generate your list such that each element is in a new line without any explanation."""

        zs_python_answer = get_gemma_7b_answer(zs_python_prompt)
        zs_special_answer = get_gemma_7b_answer(zs_special_prompt)
        zs_bullet_answer = get_gemma_7b_answer(zs_bullet_prompt)
        zs_newline_answer =  get_gemma_7b_answer(zs_newline_prompt)

        python_follow = check_python_follow_ranking(zs_python_answer)
        if python_follow:
            python_fi += 1
            extract_python = extract_python_ranking(zs_python_answer)
            python_F1 += compute_f1(extract_python, keyphrases)

        special_follow = check_special_character_follow_ranking(zs_special_answer)
        if special_follow:
            special_fi += 1
            extract_special = extract_special_character_ranking(zs_special_answer)
            special_F1 += compute_f1(extract_special, keyphrases)

        bullet_follow = check_bullet_follow_ranking(zs_bullet_answer)
        if bullet_follow:
            bullet_fi += 1
            extract_bullet = extract_bullet_ranking(zs_bullet_answer)
            bullet_F1 += compute_f1(extract_bullet, keyphrases)

        newline_follow = check_newline_follow_ranking(zs_newline_answer)
        if newline_follow:
            newline_fi += 1
            extract_newline = extract_newline_ranking(zs_newline_answer)
            newline_F1 += compute_f1(extract_newline, keyphrases)

        all_cnt += 1
        saved_data.append([id, document, keyphrases, zs_python_answer, zs_special_answer, zs_bullet_answer, zs_newline_answer])

print(f"python_fi: {python_fi/all_cnt}")
print(f"special_fi: {special_fi/all_cnt}")
print(f"bullet_fi: {bullet_fi/all_cnt}")
print(f"newline_fi: {newline_fi/all_cnt}")
print("===")
print(f"python_F1: {python_F1/all_cnt}")
print(f"special_F1: {special_F1/all_cnt}")
print(f"bullet_F1: {bullet_F1/all_cnt}")
print(f"newline_F1: {newline_F1/all_cnt}")


with open("../../output/Gemma_7b_it_semeval2017_zs.csv", "w") as file:
    csvwriter = csv.writer(file)
    csvwriter.writerow(["id", "document", "keyphrases", "zs_python_answer", "zs_special_answer", "zs_bullet_answer", "zs_newline_answer"])
    csvwriter.writerows(saved_data)



################ CoT PERFORMANCE ###############



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
with open("../../../EvalData/list_keyphrases_SemEval2017_200.csv") as file:
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

        cot_python_answer = get_gemma_7b_answer(cot_python_prompt)
        cot_special_answer = get_gemma_7b_answer(cot_special_prompt)
        cot_bullet_answer = get_gemma_7b_answer(cot_bullet_prompt)
        cot_newline_answer =  get_gemma_7b_answer(cot_newline_prompt)

        # print(cot_python_answer)
        # print(cot_special_answer)
        # print(cot_bullet_answer)
        # print(cot_newline_answer)
        # print("===")

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

        all_cnt += 1
        saved_data.append([id, document, keyphrases, cot_python_answer, cot_special_answer, cot_bullet_answer, cot_newline_answer])

print(f"python_fi: {python_fi/all_cnt}")
print(f"special_fi: {special_fi/all_cnt}")
print(f"bullet_fi: {bullet_fi/all_cnt}")
print(f"newline_fi: {newline_fi/all_cnt}")
print("===")
print(f"python_F1: {python_F1/all_cnt}")
print(f"special_F1: {special_F1/all_cnt}")
print(f"bullet_F1: {bullet_F1/all_cnt}")
print(f"newline_F1: {newline_F1/all_cnt}")

with open("../../output/Gemma_7b_it_semeval2017_cot.csv", "w") as file:
    csvwriter = csv.writer(file)
    csvwriter.writerow(["id", "document", "keyphrases", "cot_python_answer", "cot_special_answer", "cot_bullet_answer", "cot_newline_answer"])
    csvwriter.writerows(saved_data)

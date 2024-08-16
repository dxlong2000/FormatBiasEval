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

MODEL_SAVE_PATH = '/home/bizon/hai/gemma_7b_save'

sampling_params = SamplingParams(temperature=0.8, top_p=0.95, min_tokens = 1, max_tokens = 5000)

llm = LLM(model=MODEL_SAVE_PATH, tensor_parallel_size=4, dtype = "float16")

#########
def get_gemma_7b_answer(prompt):
    output = llm.generate(prompt, sampling_params)
    generated_text = output[0].outputs[0].text
    print(generated_text)
    return generated_text

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

###########  ZERO SHOT PERFORMANCE #################
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
with open("/home/bizon/hai/EvalData/list_scidocs_reranking_200.csv") as file:
    csvreader = csv.reader(file)
    header = next(csvreader)
    for row in tqdm(csvreader):
        query = row[list(header).index("query")]
        positive = eval(row[list(header).index("positive")])
        negative = eval(row[list(header).index("negative")])

        all_samples = positive
        all_samples.extend(negative)

        zs_python_prompt = f"""Given a query, and a list of documents:
Topic: {query}
List of documents: {str(all_samples)}

You are required to transform the list of documents into a binary list of 1 or 0 where 1 indicates the document brings useful information to the topic, and 0 indicates the document does not bring useful information to the topic.

Generate your binary list as a Python list without any explanation."""

        zs_bullet_prompt = f"""Given a topic, and a list of documents:
Topic: {query}
List of documents: {str(all_samples)}

You are required to transform the list of documents into a binary list of 1 or 0 where 1 indicates the document brings useful information to the topic, and 0 indicates the document does not bring useful information to the topic.

Generate your binary list using bullet points without any explanation."""

        zs_special_prompt = f"""Given a topic, and a list of documents:
Topic: {query}
List of documents: {str(all_samples)}

You are required to transform the list of documents into a binary list of 1 or 0 where 1 indicates the document brings useful information to the topic, and 0 indicates the document does not bring useful information to the topic.

Generate your binary list using <SEP> to seperate elements. Do it without any explanation."""

        zs_newline_prompt = f"""Given a topic, and a list of documents:
Topic: {query}
List of documents: {str(all_samples)}

You are required to transform the list of documents into a binary list of 1 or 0 where 1 indicates the document brings useful information to the topic, and 0 indicates the document does not bring useful information to the topic.

Generate your binary list such that each element is in a new line. Do it without any explanation."""

        zs_python_answer = get_gemma_7b_answer(zs_python_prompt)
        zs_special_answer = get_gemma_7b_answer(zs_special_prompt)
        zs_bullet_answer = get_gemma_7b_answer(zs_bullet_prompt)
        zs_newline_answer =  get_gemma_7b_answer(zs_newline_prompt)

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
        saved_data.append([query, all_samples, zs_python_answer, zs_special_answer, zs_bullet_answer, zs_newline_answer])

print(f"python_fi: {python_fi/all_cnt}")
print(f"special_fi: {special_fi/all_cnt}")
print(f"bullet_fi: {bullet_fi/all_cnt}")
print(f"newline_fi: {newline_fi/all_cnt}")
print("===")
print(f"python_map: {python_map/all_cnt}")
print(f"special_map: {special_map/all_cnt}")
print(f"bullet_map: {bullet_map/all_cnt}")
print(f"newline_map: {newline_map/all_cnt}")

with open("../../output/Gemma_7b_it_SciDocsRR_zs_Shot2.csv", "w") as file:
    csvwriter = csv.writer(file)
    csvwriter.writerow(["query", "all_samples", "zs_python_answer", "zs_special_answer", "zs_bullet_answer", "zs_newline_answer"])
    csvwriter.writerows(saved_data)


############ COT PERFORMANCE ####################

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
with open("/home/bizon/hai/EvalData/list_scidocs_reranking_200.csv") as file:
    csvreader = csv.reader(file)
    header = next(csvreader)
    for row in tqdm(csvreader):
        query = row[list(header).index("query")]
        positive = eval(row[list(header).index("positive")])
        negative = eval(row[list(header).index("negative")])

        all_samples = positive
        all_samples.extend(negative)

        wrapping = "Wrap your final list by <ANSWER> and </ANSWER>."

        cot_python_prompt = f"""Given a query, and a list of documents:
Topic: {query}
List of documents: {str(all_samples)}

You are required to transform the list of documents into a binary list of 1 or 0 where 1 indicates the document brings useful information to the topic, and 0 indicates the document does not bring useful information to the topic.

Generate your binary list in Python format [...] step-by-step. {wrapping}"""

        cot_bullet_prompt = f"""Given a query, and a list of documents:
Topic: {query}
List of documents: {str(all_samples)}

You are required to transform the list of documents into a binary list of 1 or 0 where 1 indicates the document brings useful information to the topic, and 0 indicates the document does not bring useful information to the topic.

Generate your binary list using bullet points. Generate your answer step-by-step. {wrapping}"""

        cot_special_prompt = f"""Given a query, and a list of documents:
Topic: {query}
List of documents: {str(all_samples)}

You are required to transform the list of documents into a binary list of 1 or 0 where 1 indicates the document brings useful information to the topic, and 0 indicates the document does not bring useful information to the topic.

Generate your binary list using <SEP> to seperate elements. Generate your answer step-by-step. {wrapping}"""

        cot_newline_prompt = f"""Given a query, and a list of documents:
Topic: {query}
List of documents: {str(all_samples)}

You are required to transform the list of documents into a binary list of 1 or 0 where 1 indicates the document brings useful information to the topic, and 0 indicates the document does not bring useful information to the topic.

Generate your binary list such that each element is in a new line. Generate your answer step-by-step. {wrapping}"""

        cot_python_answer = get_gemma_7b_answer(cot_python_prompt)
        cot_special_answer = get_gemma_7b_answer(cot_special_prompt)
        cot_bullet_answer = get_gemma_7b_answer(cot_bullet_prompt)
        cot_newline_answer =  get_gemma_7b_answer(cot_newline_prompt)

        python_follow = check_python_follow_ranking(cot_python_answer, cot=True)
        if python_follow:
            python_fi += 1
            extract_python = extract_python_ranking(cot_python_answer, cot=True)
            python_map += mean_average_precision(extract_python)

        special_follow = check_special_character_follow_ranking(cot_special_answer, cot=True)
        if special_follow:
            special_fi += 1
            extract_special = extract_special_character_ranking(cot_special_answer, cot=True)
            special_map += mean_average_precision(extract_special)

        bullet_follow = check_bullet_follow_ranking(cot_bullet_answer, cot=True)
        if bullet_follow:
            bullet_fi += 1
            extract_bullet = extract_bullet_ranking(cot_bullet_answer, cot=True)
            bullet_map += mean_average_precision(extract_bullet)

        newline_follow = check_newline_follow_ranking(cot_newline_answer, cot=True)
        if newline_follow:
            newline_fi += 1
            extract_newline = extract_newline_ranking(cot_newline_answer, cot=True)
            newline_map += mean_average_precision(extract_newline)

        all_cnt += 1
        saved_data.append([query, all_samples, cot_python_answer, cot_special_answer, cot_bullet_answer, cot_newline_answer])

print(f"python_fi: {python_fi/all_cnt}")
print(f"special_fi: {special_fi/all_cnt}")
print(f"bullet_fi: {bullet_fi/all_cnt}")
print(f"newline_fi: {newline_fi/all_cnt}")
print("===")
print(f"python_map: {python_map/all_cnt}")
print(f"special_map: {special_map/all_cnt}")
print(f"bullet_map: {bullet_map/all_cnt}")
print(f"newline_map: {newline_map/all_cnt}")

with open("../../output/Gemma_7b_it_SciDocsRR_cot_Shot2.csv", "w") as file:
    csvwriter = csv.writer(file)
    csvwriter.writerow(["query", "all_samples", "cot_python_answer", "cot_special_answer", "cot_bullet_answer", "cot_newline_answer"])
    csvwriter.writerows(saved_data)
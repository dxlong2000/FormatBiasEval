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

# Medium
import csv
import ast
import yaml
from tqdm import tqdm

PYTHON_DICTIONARY_FORMAT = {
    "Task": [...],
    "Method": [...]
}

YAML_FORMAT = """
Task: [...]
Method: [...]
"""


def extract_python_dict(answer, cot=False):
    if '```python' in answer and '```' in answer: answer = answer.replace('```python', '').replace('```', '')
    if cot: answer = answer.split("<ANSWER>")[1].split("</ANSWER>")[0].strip()
    answer = ast.literal_eval(answer)
    return answer


def extract_yaml_dict(answer, cot=False):
    if '```yaml' in answer and '```' in answer: answer = answer.replace('```yaml', '').replace('```', '')
    if cot: answer = answer.split("<ANSWER>")[1].split("</ANSWER>")[0].strip()
    answer = yaml.safe_load(answer)
    return answer


def compute_f1(list1, list2):
    # Convert lists to sets for efficient intersection and union operations
    list1 = [ele.lower() for ele in list1]
    set1 = set(list1)

    list2 = [ele.lower() for ele in list2]
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

all_saved_data = []
fi_python = 0
fi_yaml = 0

f1_python = 0
f1_yaml = 0

METRIC_CATEGORIES = ["Task", "Method"]

all_cnt = 0
with open("./EvalData/dict_SciREX_medium.csv") as file:
    csvreader = csv.reader(file)
    header = next(csvreader)
    for row in tqdm(csvreader):
        input_data = row[list(header).index("input_data")]
        ground_truth = eval(row[list(header).index("answer")])

        # Python dictionary format
        python_dictionary_prompt = f"""Extract the entities reflecting the tasks and methods in the following document:
Document: {input_data}
Your output must be a Python dictionary with the keys are 'Task' and 'Method', and value is a list of task name entities and method name entities:
{str(PYTHON_DICTIONARY_FORMAT)}
"""
        # Csv format
        yaml_prompt = f"""Extract the entities reflecting the tasks and methods in the following document:
Document: {input_data}
Your output must be in YAML format:
{str(YAML_FORMAT)}"""

        python_output = get_mistral_answer(python_dictionary_prompt)
        yaml_output = get_mistral_answer(yaml_prompt)

        try:
            extracted_python = extract_python_dict(python_output)
            tmpf1 = 0
            for cat in METRIC_CATEGORIES: tmpf1 += compute_f1(extracted_python[cat], ground_truth[cat])
            f1_python += tmpf1/len(METRIC_CATEGORIES)
            fi_python += 1
        except: pass

        try:
            extracted_yaml = extract_yaml_dict(yaml_output)
            tmpf1 = 0
            for cat in METRIC_CATEGORIES: tmpf1 += compute_f1(extracted_yaml[cat], ground_truth[cat])
            f1_yaml += tmpf1/len(METRIC_CATEGORIES)
            fi_yaml += 1
        except: pass

        all_cnt += 1

        saved_data = row
        saved_data.extend([python_output, yaml_output])
        all_saved_data.append(saved_data)
        

print(f"fi_python: {fi_python/all_cnt}")
print(f"fi_yaml: {fi_yaml/all_cnt}")
print("===")
print(f"f1_python: {f1_python/all_cnt}")
print(f"f1_yaml: {f1_yaml/all_cnt}")

import csv 

new_header = list(header)
new_header.extend(["python_output", "yaml_output"])

with open("./src/output/mistral_dictionary_medium.csv", "w") as file:
    csvwriter = csv.writer(file)
    csvwriter.writerow(new_header)
    csvwriter.writerows(all_saved_data)
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

sampling_params = SamplingParams(temperature=0.7, top_p=1, min_tokens = 1, max_tokens = 1024)

llm = LLM(model=MODEL_SAVE_PATH, tensor_parallel_size=4, dtype = "float16")

#########
def get_gemma_7b_answer(prompt):
    output = llm.generate(prompt, sampling_params)
    generated_text = output[0].outputs[0].text
    #print(generated_text)
    return generated_text

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

######## Easy #######


PYTHON_DICTIONARY_FORMAT = {
    "Task": [...]
}

YAML_FORMAT = """
Task: [...]
"""

all_saved_data = []
fi_python = 0
fi_yaml = 0

f1_python_task = 0
f1_yaml_task = 0

all_cnt = 0
with open("../../../EvalData/dict_SciREX_easy.csv") as file:
    csvreader = csv.reader(file)
    header = next(csvreader)
    for row in tqdm(csvreader):
        input_data = row[list(header).index("input_data")]
        ground_truth = eval(row[list(header).index("answer")])

        # Python dictionary format
        python_dictionary_prompt = f"""Extract the entities reflecting the tasks in the following document:
Document: {input_data}
Your output must be a Python dictionary with the key is 'Task' and value is a list of task name entities:
{str(PYTHON_DICTIONARY_FORMAT)}
"""
        # Csv format
        yaml_prompt = f"""Extract the entities reflecting the tasks in the following document:
Document: {input_data}
Your output must be a YAML format as following definition:
{str(YAML_FORMAT)}
Fill your entities into []. For example your final answer: Task: ['Label Ambiguity']
"""

        python_output = get_gemma_7b_answer(python_dictionary_prompt)
        yaml_output = get_gemma_7b_answer(yaml_prompt)

        try:
            extracted_python = extract_python_dict(python_output)
            f1_python_task += compute_f1(extracted_python["Task"], ground_truth["Task"])
            fi_python += 1
        except: pass

        try:
            extracted_yaml = extract_yaml_dict(yaml_output)
            f1_yaml_task += compute_f1(extracted_yaml["Task"], ground_truth["Task"])
            fi_yaml += 1
        except: pass

        all_cnt += 1

        saved_data = row
        saved_data.extend([python_output, yaml_output])
        all_saved_data.append(saved_data)

print(f"fi_python_zs_easy: {fi_python/all_cnt}")
print(f"fi_yaml_zs_easy: {fi_yaml/all_cnt}")
print("===")
print(f"f1_python_zs_easy: {f1_python_task/all_cnt}")
print(f"f1_yaml_zs_easy: {f1_yaml_task/all_cnt}")

new_header = list(header)
new_header.extend(["python_output", "yaml_output"])

with open("../../output/dictionary_easy_zs.csv", "w") as file:
    csvwriter = csv.writer(file)
    csvwriter.writerow(new_header)
    csvwriter.writerows(all_saved_data)


################## Easy CoT #########################
PYTHON_DICTIONARY_FORMAT = {
    "Task": [...]
}

YAML_FORMAT = """
Task: [...]
"""

all_saved_data = []
fi_python = 0
fi_yaml = 0

f1_python_task = 0
f1_yaml_task = 0

all_cnt = 0
with open("../../../EvalData/dict_SciREX_easy.csv") as file:
    csvreader = csv.reader(file)
    header = next(csvreader)
    for row in tqdm(csvreader):
        input_data = row[list(header).index("input_data")]
        ground_truth = eval(row[list(header).index("answer")])

        # Python dictionary format
        wrapping_instruction = "Wrap your final answer by <ANSWER> and </ANSWER>."
        python_dictionary_prompt = f"""Extract the entities reflecting the tasks in the following document step-by-step:
Document: {input_data}
Your output must be a Python dictionary with the key is 'Task' and value is a list of task name entities:
{str(PYTHON_DICTIONARY_FORMAT)}
{wrapping_instruction}
"""
        # Csv format
        yaml_prompt = f"""Extract the entities reflecting the tasks in the following document:
Document: {input_data}
Your output must be in YAML format:
{str(YAML_FORMAT)}
{wrapping_instruction}"""

        python_output = get_gemma_7b_answer(python_dictionary_prompt)
        yaml_output = get_gemma_7b_answer(yaml_prompt)

        try:
            extracted_python = extract_python_dict(python_output, cot=True)
            f1_python_task += compute_f1(extracted_python["Task"], ground_truth["Task"])
            fi_python += 1
        except: pass

        try:
            extracted_yaml = extract_yaml_dict(yaml_output, cot=True)
            f1_yaml_task += compute_f1(extracted_yaml["Task"], ground_truth["Task"])
            fi_yaml += 1
        except: pass

        all_cnt += 1

        saved_data = row
        saved_data.extend([python_output, yaml_output])
        all_saved_data.append(saved_data)

print(f"fi_python_cot_easy: {fi_python/all_cnt}")
print(f"fi_yaml_cot_easy: {fi_yaml/all_cnt}")
print("===")
print(f"f1_python_cot_easy: {f1_python_task/all_cnt}")
print(f"f1_yaml_cot_easy: {f1_yaml_task/all_cnt}")

new_header = list(header)
new_header.extend(["python_output", "yaml_output"])

with open("../../output/dictionary_easy_cot.csv", "w") as file:
    csvwriter = csv.writer(file)
    csvwriter.writerow(new_header)
    csvwriter.writerows(all_saved_data)
'''

'''
####################################### Medium #########################################

PYTHON_DICTIONARY_FORMAT = {
    "Task": [...],
    "Method": [...]
}

YAML_FORMAT = """
Task: [...]
Method: [...]
"""

all_saved_data = []
fi_python = 0
fi_yaml = 0

f1_python = 0
f1_yaml = 0

METRIC_CATEGORIES = ["Task", "Method"]

all_cnt = 0

with open("../../../EvalData/dict_SciREX_medium.csv") as file:
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

        python_output = get_gemma_7b_answer(python_dictionary_prompt)
        yaml_output = get_gemma_7b_answer(yaml_prompt)

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


print(f"fi_python_zs_med: {fi_python/all_cnt}")
print(f"fi_yaml_zs_med: {fi_yaml/all_cnt}")
print("===")
print(f"f1_python_zs_med: {f1_python/all_cnt}")
print(f"f1_yaml_zs_med: {f1_yaml/all_cnt}")

new_header = list(header)
new_header.extend(["python_output", "yaml_output"])

with open("../../output/dictionary_medium_zs.csv", "w") as file:
    csvwriter = csv.writer(file)
    csvwriter.writerow(new_header)
    csvwriter.writerows(all_saved_data)





####################################### Medium COT #########################################

all_saved_data = []
fi_python = 0
fi_yaml = 0

f1_python = 0
f1_yaml = 0

METRIC_CATEGORIES = ["Task", "Method"]

all_cnt = 0
with open("../../../EvalData/dict_SciREX_medium.csv") as file:
    csvreader = csv.reader(file)
    header = next(csvreader)
    for row in tqdm(csvreader):
        input_data = row[list(header).index("input_data")]
        ground_truth = eval(row[list(header).index("answer")])

        # Python dictionary format
        wrapping_instruction = "Wrap your final answer by <ANSWER> and </ANSWER>."
        python_dictionary_prompt = f"""Extract the entities reflecting the tasks and methods in the following document step-by-step:
Document: {input_data}
Your output must be a Python dictionary with the keys are 'Task' and 'Method', and value is a list of task name entities and method name entities:
{str(PYTHON_DICTIONARY_FORMAT)}
{wrapping_instruction}
"""
        # Csv format
        yaml_prompt = f"""Extract the entities reflecting the tasks and methods in the following document step-by-step:
Document: {input_data}
Your output must be in YAML format:
{str(YAML_FORMAT)}
{wrapping_instruction}"""

        python_output = get_gemma_7b_answer(python_dictionary_prompt)
        yaml_output = get_gemma_7b_answer(yaml_prompt)

        try:
            extracted_python = extract_python_dict(python_output, cot=True)
            tmpf1 = 0
            for cat in METRIC_CATEGORIES: tmpf1 += compute_f1(extracted_python[cat], ground_truth[cat])
            f1_python += tmpf1/len(METRIC_CATEGORIES)
            fi_python += 1
        except: pass

        try:
            extracted_yaml = extract_yaml_dict(yaml_output, cot=True)
            tmpf1 = 0
            for cat in METRIC_CATEGORIES: tmpf1 += compute_f1(extracted_yaml[cat], ground_truth[cat])
            f1_yaml += tmpf1/len(METRIC_CATEGORIES)
            fi_yaml += 1
        except: pass

        all_cnt += 1

        saved_data = row
        saved_data.extend([python_output, yaml_output])
        all_saved_data.append(saved_data)

print(f"fi_python_cot_med: {fi_python/all_cnt}")
print(f"fi_yaml_cot_med: {fi_yaml/all_cnt}")
print("===")
print(f"f1_python_cot_med: {f1_python/all_cnt}")
print(f"f1_yaml_cot_med: {f1_yaml/all_cnt}")

new_header = list(header)
new_header.extend(["python_output", "yaml_output"])

with open("../../output/dictionary_medium_cot.csv", "w") as file:
    csvwriter = csv.writer(file)
    csvwriter.writerow(new_header)
    csvwriter.writerows(all_saved_data)

########## Hard ###################

PYTHON_DICTIONARY_FORMAT = {
    "Task": [...],
    "Method": [...],
    "Material": [...],
    "Metric": [...]

}

YAML_FORMAT = """
Task: [...]
Method: [...]
Material: [...]
Metric: [...]
"""

all_saved_data = []
fi_python = 0
fi_yaml = 0

f1_python = 0
f1_yaml = 0

METRIC_CATEGORIES = ["Task", "Method", "Material", "Metric"]

all_cnt = 0
with open("../../../EvalData/dict_SciREX_hard.csv") as file:
    csvreader = csv.reader(file)
    header = next(csvreader)
    for row in tqdm(csvreader):
        input_data = row[list(header).index("input_data")]
        ground_truth = eval(row[list(header).index("answer")])

        # Python dictionary format
        python_dictionary_prompt = f"""Extract the entities reflecting the tasks and methods in the following document:
Document: {input_data}
Your output must be a Python dictionary with the keys are 'Task', 'Method', 'Material', 'Metric', and value is a list of task name entities, method name entities, material name entities, metric name entities:
{str(PYTHON_DICTIONARY_FORMAT)}
"""
        # Csv format
        yaml_prompt = f"""Extract the entities reflecting the tasks, methods, materials, metrics in the following document:
Document: {input_data}
Your output must be in YAML format:
{str(YAML_FORMAT)}"""

        python_output = get_gemma_7b_answer(python_dictionary_prompt)
        yaml_output = get_gemma_7b_answer(yaml_prompt)

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

with open("../../output/dictionary_hard_zs.csv", "w") as file:
    csvwriter = csv.writer(file)
    csvwriter.writerow(new_header)
    csvwriter.writerows(all_saved_data)


##################### Hard Cot ############################


all_saved_data = []
fi_python = 0
fi_yaml = 0

f1_python = 0
f1_yaml = 0

METRIC_CATEGORIES = ["Task", "Method", "Material", "Metric"]

all_cnt = 0
with open("../../../EvalData/dict_SciREX_hard.csv") as file:
    csvreader = csv.reader(file)
    header = next(csvreader)
    for row in tqdm(csvreader):
        input_data = row[list(header).index("input_data")]
        ground_truth = eval(row[list(header).index("answer")])

        # Python dictionary format
        wrapping_instruction = "Wrap your final answer by <ANSWER> and </ANSWER>."
        python_dictionary_prompt = f"""Extract the entities reflecting the tasks and methods in the following document:
Document: {input_data}
Your output must be a Python dictionary with the keys are 'Task', 'Method', 'Material', 'Metric', and value is a list of task name entities, method name entities, material name entities, metric name entities:
{str(PYTHON_DICTIONARY_FORMAT)}
{wrapping_instruction}
"""
        # Csv format
        yaml_prompt = f"""Extract the entities reflecting the tasks, methods, materials, metrics in the following document:
Document: {input_data}
Your output must be in YAML format:
{str(YAML_FORMAT)}
{wrapping_instruction}"""

        python_output = get_gemma_7b_answer(python_dictionary_prompt)
        yaml_output = get_gemma_7b_answer(yaml_prompt)

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

with open("../../output/dictionary_hard_cot.csv", "w") as file:
    csvwriter = csv.writer(file)
    csvwriter.writerow(new_header)
    csvwriter.writerows(all_saved_data)
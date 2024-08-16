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

sampling_params = SamplingParams(temperature = 0, min_tokens = 1, max_tokens = 256)

llm = LLM(model=MODEL_SAVE_PATH, tensor_parallel_size=4, dtype = "float16")

#########
def get_gemma_7b_answer(prompt):
    output = llm.generate(prompt, sampling_params)
    generated_text = output[0].outputs[0].text
    return generated_text

input_data = "document : Deep Label Distribution Learning With Label Ambiguity"

PYTHON_DICTIONARY_FORMAT = {
    "Task": [ ]
}

YAML_FORMAT = """
Task: [  ]
"""

# Python dictionary format
python_dictionary_prompt = f"""Extract the entities reflecting the tasks in the following document:
{input_data}
Your output must be a Python dictionary with the key is 'Task' and value is a list of task name entities as following form:
{str(PYTHON_DICTIONARY_FORMAT)}
"""

# Csv format
yaml_prompt = f"""Extract the entities reflecting the tasks in the following document:
{input_data}
Your output must be a YAML format as following definition:
{str(YAML_FORMAT)}
Fill your entities into []. For example: Task: ['Label Ambiguity']
"""

python_output = get_gemma_7b_answer(python_dictionary_prompt)
yaml_output = get_gemma_7b_answer(yaml_prompt)

print(python_output)
print("---------------------")
print(yaml_output)
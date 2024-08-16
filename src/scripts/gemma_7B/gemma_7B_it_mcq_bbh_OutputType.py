from transformers import AutoModelForCausalLM, AutoTokenizer
import time
import json
from tqdm import tqdm
import pandas as pd
from random import sample
import csv 
import random
from vllm import LLM, SamplingParams
import torch

MODEL_SAVE_PATH = 'YOUR MODEL SAVE PATH'

sampling_params = SamplingParams(top_p=1, min_tokens = 1, max_tokens = 1024)

llm = LLM(model=MODEL_SAVE_PATH, tensor_parallel_size=4, dtype = "float16")

#########
def get_gemma_7b_answer(prompt):
    output = llm.generate(prompt, sampling_params)
    generated_text = output[0].outputs[0].text
    print(generated_text)
    return generated_text
    
###### Loading data ######
def extract_character_answer(answer, cot=False):
    if cot:
        if "<ANSWER>" in answer and "</ANSWER>" in answer:
            answer_content = answer.split("<ANSWER>")[1].split("</ANSWER>")[0].strip()
        else:
            answer_content = " "
        
        # Check if the answer_content is not empty before accessing the first character
        return answer_content[0].strip() if answer_content else " "
    else:
        return answer[0].strip() if answer else " "

def extract_textual_value_answer(answer, cot=False):
    if cot:
        if "<ANSWER>" in answer and "</ANSWER>" in answer:
            answer_content = answer.split("<ANSWER>")[1].split("</ANSWER>")[0].strip()
        else:
            answer_content = " "
        
        # Check if the answer_content is not empty before accessing the first character
        return answer_content.strip() if answer_content else " "
    else:
        return answer.strip() if answer else " "

saved_rows = []
with open("../../../EvalData/mcq_bbh_test_1.csv") as file:
    csvreader = csv.reader(file)
    header = next(csvreader) # topic,question,choices_texts,input_data,correct_key,correct_text
    
    zero_shot_character_cnt = 0
    zero_shot_value_cnt = 0
    
    cot_character_cnt = 0
    cot_text_cnt = 0
    
    all_cnt = 0
    for row in tqdm(csvreader):
        input_data = row[list(header).index("input_data")]
        correct_key = row[list(header).index("correct_key")]
        correct_text = row[list(header).index("correct_text")]
        
        #demonstration = "For an example of MCQ answer like A. Yes, define \n - Designated character identifier: A \n - Choice value: Yes \n - Designated character identifier and choice value: A. Yes"

        demonstration = ""
        zero_shot_character_instruct = "Answer the following yes-no question by outputting only the designated character identifier (A or B)."
        zero_shot_text_val_instruct = "Answer the following yes-no question by outputting the textual value of your choice without the character identifier."  
        
        cot_character_instruct = "Answer the following yes-no question step-by-step and finally outputting only the designated character identifier (A or B)."
        cot_text_val_instruct = "Answer the following yes-no question step-by-step and finally outputting the textual value of your choice without the character identifier."  
        cot_wrapping = "Wrap your final answer between <ANSWER> and </ANSWER>."
                
        raw_zs_character_result = get_gemma_7b_answer(demonstration + zero_shot_character_instruct * 5 + "\n" + input_data)
        zero_shot_character_result = extract_character_answer(raw_zs_character_result)
        zero_shot_character_cnt += int(zero_shot_character_result==correct_key)
        
        raw_zs_text_result = get_gemma_7b_answer(demonstration + zero_shot_text_val_instruct *5 + "\n" + input_data)
        zero_shot_text_result = extract_textual_value_answer(raw_zs_text_result)
        zero_shot_value_cnt += int(zero_shot_text_result==correct_text)
        
        raw_cot_character_result = get_gemma_7b_answer(demonstration + cot_character_instruct *5 + "\n" + input_data + "\n" + cot_wrapping)
        cot_character_result = extract_character_answer(raw_cot_character_result, cot=True)
        cot_character_cnt += int(cot_character_result==correct_key)
        
        raw_cot_text_result = get_gemma_7b_answer(demonstration + cot_text_val_instruct *5 + "\n" + input_data + "\n" + cot_wrapping)
        cot_text_result = extract_character_answer(raw_cot_text_result, cot=True)
        cot_text_cnt += int(cot_text_result==correct_text)
        
        all_cnt += 1
        
        tmp_row = row 
        tmp_row.extend([
            raw_zs_character_result, zero_shot_character_result, 
            raw_zs_text_result, zero_shot_text_result, 
            raw_cot_character_result, cot_character_result, 
            raw_cot_text_result, cot_text_result
        ])
        saved_rows.append(tmp_row)

print(f"zero_shot_character_cnt: {zero_shot_character_cnt/all_cnt}")
print(f"zero_shot_value_cnt: {zero_shot_value_cnt/all_cnt}")
print(f"cot_character_cnt: {cot_character_cnt/all_cnt}")
print(f"cot_value_cnt: {cot_text_cnt/all_cnt}")    

with open("../../output/gemma_7b_it_mcq_bbh_OutputType.csv", "w") as file:
    csvwriter = csv.writer(file)
    csvwriter.writerow([
        "topic", "question", "choices_texts", "input_data",
        "correct_key", "correct_text", 
        "raw_zs_character_result", "zs_character_result", 
        "raw_zs_text_result", "zs_text_result", 
        "raw_cot_character_result", "cot_character_result", 
        "raw_cot_text_result", "cot_text_result"
    ])
    csvwriter.writerows(saved_rows)


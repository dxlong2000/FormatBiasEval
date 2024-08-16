import string, re
import pandas as pd 
from tqdm import tqdm
import evaluate
import csv 

def check_mcq_answer(extracted_answer):
    if extracted_answer.lower() in ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']: return True 
    return False
    # return True

def check_free_form_follow(answer, mcq=False):
    if "<ANSWER>" not in answer or "</ANSWER>" not in answer: return False
    if not mcq: return True
    else: 
        try: return check_mcq_answer(answer.split("<ANSWER>")[1].split("</ANSWER>")[0].strip())
        except: return False

def check_bolding(answer, mcq=False):
    try:
        pattern = r'\*\*(.*?)\*\*'
        matches = re.findall(pattern, answer)
        if len(matches) == 1: 
            if not mcq: return True
            else: 
                try: return check_mcq_answer(matches[0].strip())
                except: return False
        else: return False
    except:
        return False

def check_italicizing(answer, mcq=False):
    try:
        pattern = r'\*(.*?)\*'
        matches = re.findall(pattern, answer)
        if len(matches) == 1: 
            if not mcq: return True
            else: 
                try: return check_mcq_answer(matches[0].strip())
                except: return False
        else: return False
    except:
        return False

def check_brackets(answer, mcq=False):
    try:
        pattern = r'\[\[(.*?)\]\]'
        matches = re.findall(pattern, answer)
        
        if len(matches) == 1:
            if not mcq: return True
            else: 
                try: return check_mcq_answer(matches[0].strip())
                except: return False 
        else: return False
    except:
        return False

def check_parentheses(answer, mcq=False):
    try:
        pattern = r'\(\((.*?)\)\)'
        matches = re.findall(pattern, answer)
        if len(matches) == 1:
            if not mcq: return True
            else: 
                try: return check_mcq_answer(matches[0].strip())
                except: return False 
        else: return False
    except:
        return False

def check_placeholder(answer, mcq=False):
    try:
        pattern = r"\{(.*?)\}"
        matches = re.findall(pattern, answer)        
        if len(matches) == 1:
            if not mcq: return True
            else: 
                try: return check_mcq_answer(matches[0].strip())
                except: return False 
        else: return False
    except:
        return False

def check_quoting(answer, mcq=False):
    try:
        pattern = r'\'\'\'(.*?)\'\'\'|\"\"\"(.*?)\"\"\"'
        matches = re.findall(pattern, answer, re.DOTALL)
        if len(matches) > 0:
            return check_mcq_answer(extract_character_quoting_answer(answer))
        else: return False
    except:
        return False

###### Extracting data ######
def extract_character_free_form_answer(answer, cot=False):
    try:
        if "<ANSWER>" in answer and "</ANSWER>" in answer:
            answer = answer.split("<ANSWER>")[1].split("</ANSWER>")[0].strip()
        else: 
            answer = " "
        return answer.strip()
    except:
        return ""
    
def extract_character_bolding_answer(answer, cot=False):
    try:
        pattern = r'\*\*(.*?)\*\*'
        matches = re.findall(pattern, answer)
        return matches[0]
    except:
        return ""

def extract_character_italicizing_answer(input_string):
    try:
        pattern = r'\*(.*?)\*'
        matches = re.findall(pattern, input_string)
        return matches[0]
    except:
        return ""

def extract_character_brackets_answer(input_string):
    try:
        pattern = r'\[\[(.*?)\]\]'
        matches = re.findall(pattern, input_string)
        return matches[0]
    except:
        return ""

def extract_character_parenthesis_answer(input_string):
    try:
        pattern = r'\(\((.*?)\)\)'
        matches = re.findall(pattern, input_string)
        return matches[0]
    except:
        return ""

def extract_character_placeholder_answer(input_text):
    try:
        match = re.search(r"\{(.*?)\}", input_text)
        final_answer = match.group(1)
        return final_answer
    except:
        return ""

def extract_character_quoting_answer(input_string):
    try:
        pattern = r'\'\'\'(.*?)\'\'\'|\"\"\"(.*?)\"\"\"'
        matches = re.findall(pattern, input_string, re.DOTALL)
        return [match[0] or match[1] for match in matches][0]
    except:
        return ""
    
def normalize_text(s):
    """Removing articles and punctuation, and standardizing whitespace are all typical text processing steps."""

    def remove_articles(text):
        regex = re.compile(r"\b(a|an|the)\b", re.UNICODE)
        return re.sub(regex, " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))

def compute_exact_match(prediction, truth):
    return int(normalize_text(prediction) == normalize_text(truth))

def compute_f1(prediction, truth):
    pred_tokens = normalize_text(prediction).split()
    truth_tokens = normalize_text(truth).split()
    
    # if either the prediction or the truth is no-answer then f1 = 1 if they agree, 0 otherwise
    if len(pred_tokens) == 0 or len(truth_tokens) == 0:
        return int(pred_tokens == truth_tokens)
    
    common_tokens = set(pred_tokens) & set(truth_tokens)
    
    # if there are no common tokens then f1 = 0
    if len(common_tokens) == 0:
        return 0
    
    prec = len(common_tokens) / len(pred_tokens)
    rec = len(common_tokens) / len(truth_tokens)
    
    return 2 * (prec * rec) / (prec + rec)

def compute_metric(prediction, truth, metric_mode='f1'):
    if metric_mode == 'f1': return compute_f1(prediction, truth)
    return int(prediction.lower()==truth.lower())
    # try: return int(prediction[0].lower()==truth.lower())
    # except: return 0

model_compute = "zs"
dataset_name = "mmlu"
metric_mode = 'em'
mcq_flag =True
lag = 0

# FILE_PATH = f"/home/long/PreliminaryTesting/FormatBias/src/output/chatgpt_finetuned_mcq_{dataset_name}_{model_compute}_Wrapping.csv"
FILE_PATH = "/home/long/PreliminaryTesting/FormatBias/src/output/chatgpt_replicate_3_mcq_mmlu_zs_Wrapping.csv"

ff_fi = 0
bolding_fi = 0
italicizing_fi = 0
brackets_fi = 0
parenthesis_fi = 0
placeholder_fi = 0
quoting_fi = 0
all_cnt = 0

ff_systematic = 0
bolding_systematic = 0
italicizing_systematic = 0
brackets_systematic = 0
parenthesis_systematic = 0
placeholder_systematic = 0
quoting_systematic = 0

with open(FILE_PATH) as file:
    csvreader = csv.reader(file)
    header = next(csvreader)
    print(header)
    
    ff_scores = []
    bracket_scores = []
    placeholder_scores = []
    
    for row in csvreader:
        answer = row[list(header).index("correct_key")]
        
        raw_zs_free_form_result = row[list(header).index(f"raw_{model_compute}_free_form_result") + lag]
        ff_fi += check_free_form_follow(raw_zs_free_form_result, mcq=mcq_flag)
        ff_sc = compute_metric(
            extract_character_free_form_answer(raw_zs_free_form_result), answer, metric_mode
        )
        ff_systematic += ff_sc
        if check_free_form_follow(raw_zs_free_form_result, mcq=mcq_flag):
            ff_scores.append(ff_sc)
        
        raw_zs_bolding_result = row[list(header).index(f"raw_{model_compute}_bolding_result") + lag]
        bolding_fi += check_bolding(raw_zs_bolding_result, mcq=mcq_flag)
        bolding_systematic += compute_metric(
            extract_character_bolding_answer(raw_zs_bolding_result), answer, metric_mode
        )
        
        raw_zs_italicizing_result = row[list(header).index(f"raw_{model_compute}_italicizing_result") + lag]
        italicizing_fi += check_italicizing(raw_zs_italicizing_result, mcq=mcq_flag)
        italicizing_systematic += compute_metric(
            extract_character_italicizing_answer(raw_zs_italicizing_result), answer, metric_mode
        )
        
        raw_zs_brackets_result = row[list(header).index(f"raw_{model_compute}_brackets_result") + lag]
        brackets_fi += check_brackets(raw_zs_brackets_result, mcq=mcq_flag)
        brackets_sc = compute_metric(
            extract_character_brackets_answer(raw_zs_brackets_result), answer, metric_mode
        )
        brackets_systematic += brackets_sc
        if check_brackets(raw_zs_brackets_result, mcq=mcq_flag):
            bracket_scores.append(brackets_sc)
        
        raw_zs_parenthesis_result = row[list(header).index(f"raw_{model_compute}_parenthesis_result") + lag]
        parenthesis_fi += check_parentheses(raw_zs_parenthesis_result, mcq=mcq_flag)
        parenthesis_systematic += compute_metric(
            extract_character_parenthesis_answer(raw_zs_parenthesis_result), answer, metric_mode
        )
        
        raw_zs_placeholder_result = row[list(header).index(f"raw_{model_compute}_placeholder_result") + lag]
        placeholder_fi += check_placeholder(raw_zs_placeholder_result, mcq=mcq_flag)
        placeholder_sc = compute_metric(
            extract_character_placeholder_answer(raw_zs_placeholder_result), answer, metric_mode
        )
        placeholder_systematic += placeholder_sc
        if check_placeholder(raw_zs_placeholder_result, mcq=mcq_flag):
            placeholder_scores.append(placeholder_sc)
        
        raw_zs_quoting_result = row[list(header).index(f"raw_{model_compute}_quoting_result") + lag]
        quoting_fi += check_quoting(raw_zs_quoting_result, mcq=mcq_flag)
        quoting_systematic += compute_metric(
            extract_character_quoting_answer(raw_zs_quoting_result), answer, metric_mode
        )
        
        all_cnt += 1

import numpy as np
from scipy.stats import t
import math 

def compute_sample_variance(data):
    n = len(data)
    mean = np.mean(data)
    squared_deviations = [(x - mean) ** 2 for x in data]
    sample_variance = sum(squared_deviations) / (n - 1)
    return sample_variance

def compute_estimated_fi(num_FI, list_eval_scores, num_samples=200):
    ####### t-statistics #######
    alpha = 0.05  # 5% significance level
    df = num_FI  # degrees of freedom
    alpha_two_tailed = alpha / 2
    t_statistic = t.ppf(1 - alpha_two_tailed, df)

    #######Compute MOE_FI #######
    MOE = 0.05
    s = math.sqrt(compute_sample_variance(list_eval_scores))
    return num_FI/num_samples > 1/(1 + num_samples * (MOE/(t_statistic * s))**2)
    

print(f"FI ff_fi: {ff_fi/all_cnt}")
print(f"FI bolding_fi: {bolding_fi/all_cnt}")
print(f"FI italicizing_fi: {italicizing_fi/all_cnt}")
print(f"FI brackets_fi: {brackets_fi/all_cnt}")
print(f"FI parenthesis_fi: {parenthesis_fi/all_cnt}")
print(f"FI placeholder_fi: {placeholder_fi/all_cnt}")
print(f"FI quoting_fi: {quoting_fi/all_cnt}")
print("===")
print(f"{metric_mode} ff_systematic: {ff_systematic/all_cnt}")
print(f"{metric_mode} bolding_systematic: {bolding_systematic/all_cnt}")
print(f"{metric_mode} italicizing_systematic: {italicizing_systematic/all_cnt}")
print(f"{metric_mode} brackets_systematic: {brackets_systematic/all_cnt}")
print(f"{metric_mode} parenthesis_systematic: {parenthesis_systematic/all_cnt}")
print(f"{metric_mode} placeholder_systematic: {placeholder_systematic/all_cnt}")
print(f"{metric_mode} quoting_systematic: {quoting_systematic/all_cnt}")
print("===")
print(f"compute_estimated_fi_ff: {str(compute_estimated_fi(ff_fi, ff_scores, all_cnt))}")
print(f"compute_estimated_fi_brackets: {str(compute_estimated_fi(brackets_fi, bracket_scores, all_cnt))}")
print(f"compute_estimated_fi_placeholder: {str(compute_estimated_fi(placeholder_fi, placeholder_scores, all_cnt))}")

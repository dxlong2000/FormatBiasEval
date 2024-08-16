import csv 
from posixpath import expandvars
import yaml
import ast

def compute_f1(list1, list2):
    # Convert lists to sets for efficient intersection and union operations
    list1 = [ele.strip() for ele in list1]
    list2 = [ele.strip() for ele in list2]
    
    list1 = [ele.split(". ")[1].strip() if ". " in ele else ele for ele in list1]
    list2 = [ele.split(". ")[1].strip() if ". " in ele else ele for ele in list2]
    
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

# Zero-shot performance
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

FILE_PATH = "./src/output/mistral_zs_semeval2017_20Apr.csv"
with open(FILE_PATH) as file:
    csvreader = csv.reader(file)
    header = next(csvreader)
    for row in csvreader:
        
        zs_python_answer = row[list(header).index("zs_python_answer")]
        zs_special_answer = row[list(header).index("zs_special_answer")]
        zs_bullet_answer = row[list(header).index("zs_bullet_answer")]
        zs_newline_answer =  row[list(header).index("zs_newline_answer")]
        
        keyphrases = eval(row[list(header).index("keyphrases")])
        
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

print(f"python_fi: {python_fi/all_cnt}")
print(f"special_fi: {special_fi/all_cnt}")
print(f"bullet_fi: {bullet_fi/all_cnt}")
print(f"newline_fi: {newline_fi/all_cnt}")
print("===")
print(f"python_F1: {python_F1/all_cnt}")
print(f"special_F1: {special_F1/all_cnt}")
print(f"bullet_F1: {bullet_F1/all_cnt}")
print(f"newline_F1: {newline_F1/all_cnt}")
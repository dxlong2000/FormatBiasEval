import csv 

cnt = 0 

FILE_PATH = "/home/long/PreliminaryTesting/FormatBias/EvalData/mcq_mmlu_test_27.csv"

with open(FILE_PATH) as file:
    csvreader = csv.reader(file)
    header = next(csvreader)
    for row in csvreader:
        cnt += 1

def compute_variance(data):
    if len(data) == 0:
        return 0
    mean = sum(data) / len(data)
    squared_diffs = [(x - mean) ** 2 for x in data]
    variance = sum(squared_diffs) / len(data)
    return variance

# data = [21.58, 8.09, 10.43, 16.29]
# mistral = [41.66, 30.43, 28.91, 48.78]
# chatgpt = [35.76, 33.73, 40.50, 39.03]

sys = [60.09, 67.88, 55.65, 61.99, 63.71, 30.31, 68.28]
fi = [94.26, 98.67, 89.71, 57.65, 81.03, 9.40, 48.52]
# true = [sys[idx]/fi[idx]*100 for idx in range(len(fi))]

print(sum(fi)/len(fi))

# print(true)
print(compute_variance(sys))
# print(compute_variance(mistral))
# print(compute_variance(chatgpt))
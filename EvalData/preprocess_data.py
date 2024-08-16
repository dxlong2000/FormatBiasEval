# import csv 
# import random

# data = []
# with open("/home/long/PreliminaryTesting/FormatBias/EvalData/generation_rag_test.csv") as file:
#     csvreader = csv.reader(file)
#     header = next(csvreader)
#     for row in csvreader:
#         data.append(row)

# random.shuffle(data)
# with open("/home/long/PreliminaryTesting/FormatBias/EvalData/generation_rag_test_1000.csv", "w") as file:
#     csvwriter = csv.writer(file)
#     csvwriter.writerow(header)
#     csvwriter.writerows(data[:1000])

# print(len(data))
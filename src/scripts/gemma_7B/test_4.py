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

prompt = f''' "<ANSWER>" huhu "</ANSWER>" '''

print(extract_textual_value_answer(prompt))
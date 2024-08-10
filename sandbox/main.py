import json
import pandas as pd

from typing import Dict
from rapidfuzz import fuzz

def read_input() -> Dict[str, str]:
    """Reads the input question from a json file"""

    with open("input.json", "r", encoding="utf-8") as file:
        data = json.load(file)
        data = json.dumps(data, ensure_ascii=False)
    return data




def main():
    question = read_input()

    questions_df = pd.read_excel(io="../data/perguntas_chatbot_v1.xlsx")
    questions_faq = questions_df["perguntas"].tolist()

    results = [fuzz.ratio(question, i) for i in questions_faq]

    questions_faq = dict(zip(questions_faq, results))

    max_key = max(questions_faq, key=questions_faq.get)
    max_value = questions_faq[max_key]


    print(max_key, max_value)



if __name__ == "__main__":
    main()
import json
from typing import Dict, List, Tuple
import pandas as pd
from rapidfuzz import fuzz, process, utils


def read_input() -> Dict[str, str]:
    """Reads the input question from a json file"""

    with open("input.json", "r", encoding="utf-8") as file:
        data = json.load(file)
    return data


def make_json_output(
    data: List[Tuple[str, float, int]], filename: str = "output.json"
) -> None:
    """Converts and sorts the output from rapidfuzz.process"""

    sorted_data = sorted(data, key=lambda x: x[1], reverse=True)

    result = {item[0]: round(item[1], 5) for item in sorted_data}

    with open(filename, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=4)


def main():
    question = read_input()["pergunta"]

    questions_df = pd.read_excel(io="../data/perguntas_chatbot_v1.xlsx")
    questions_faq = questions_df["perguntas"].tolist()

    ratio_results = process.extract(
        question,
        questions_faq,
        scorer=fuzz.ratio,
        limit=None,
        processor=utils.default_process,
    )
    token_set_ratio_results = process.extract(
        question,
        questions_faq,
        scorer=fuzz.token_set_ratio,
        limit=None,
        processor=utils.default_process,
    )

    make_json_output(ratio_results, "output_ratio.json")
    make_json_output(token_set_ratio_results, "output_token_set_ratio.json")


if __name__ == "__main__":
    main()

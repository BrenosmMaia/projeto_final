import json

import pandas as pd
from rapidfuzz import fuzz, process, utils


def read_input() -> dict[str, str]:
    """Reads the input question from a json file"""

    with open("input.json", encoding="utf-8") as file:
        data = json.load(file)
    return data


def make_json_output(
    data: list[tuple[str, float, int]], filename: str = "output.json"
) -> None:
    """Converts and sorts the output from rapidfuzz.process"""

    sorted_data = sorted(data, key=lambda x: x[1], reverse=True)

    result = {item[0]: round(item[1], 4) for item in sorted_data}

    with open(filename, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=4)


def calculate_similarity(question, questions_faq):
    scorers = (
        fuzz.ratio,
        fuzz.partial_ratio,
        fuzz.token_sort_ratio,
        fuzz.token_set_ratio,
        fuzz.token_ratio,
        fuzz.partial_token_sort_ratio,
        fuzz.partial_token_set_ratio,
        fuzz.partial_token_ratio,
        fuzz.WRatio,
        fuzz.QRatio,
    )

    for scorer in scorers:
        scorer_name = scorer.__name__
        results = process.extract(
            question,
            questions_faq,
            scorer=scorer,
            limit=5,
            processor=utils.default_process,
        )
        filename = f"output_{scorer_name}.json"
        make_json_output(results, filename)


def main():
    question = read_input()["pergunta"]

    questions_df = pd.read_excel(
        io="../data/Perguntas_chatbot - 09.10_relacao FAQ perguntas users.xlsx"
    )
    questions_faq = questions_df["PERGUNTA FAQ"].tolist()

    calculate_similarity(question, questions_faq)


if __name__ == "__main__":
    main()

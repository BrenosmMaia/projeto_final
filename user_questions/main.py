import json
from typing import Dict, List, Tuple
import pandas as pd
from rapidfuzz import fuzz, process, utils


def create_dataframe(
    user_questions: List[str], results: List[Dict[str, List[Tuple[str, float, int]]]]
) -> pd.DataFrame:
    """CreateS a DataFrame from the results of the matching process."""

    df_rows = []

    for question, result in zip(user_questions, results):
        row = {"user_question": question}

        for score_name, matches in result.items():
            row[score_name] = tuple(match[0] for match in matches)

        df_rows.append(row)

    return pd.DataFrame(df_rows)


def main():
    user_questions = pd.read_excel(io="../data/mapeamento_de_perguntas_chatbot.xlsx")
    questions_df = pd.read_excel(io="../data/perguntas_chatbot_v1.xlsx")

    user_questions = user_questions["PERGUNTA"].tolist()
    questions_faq = questions_df["perguntas"].tolist()

    scoares = (
        fuzz.ratio,
        fuzz.partial_ratio,
        fuzz.token_set_ratio,
        fuzz.partial_token_set_ratio,
        fuzz.token_sort_ratio,
        fuzz.partial_token_sort_ratio,
        fuzz.token_ratio,
        fuzz.partial_token_ratio,
        fuzz.WRatio,
    )

    results = []

    for i in user_questions:
        results.append(
            {
                score.__name__: process.extract(
                    i,
                    questions_faq,
                    scorer=score,
                    limit=1,
                    processor=utils.default_process,
                )
                for score in scoares
            }
        )

    df = create_dataframe(user_questions, results)
    df.to_csv("output.csv", index=False)


main()

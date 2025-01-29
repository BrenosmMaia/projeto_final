import os
import sys

import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from rapidfuzz import fuzz, process, utils

from utils import calculate_list_scores, fix_excel_table, remove_stopwords


def process_similarity_results(
    wpp_questions: list[str], results: list[dict[str, list[tuple[str, float, int]]]]
) -> pd.DataFrame:
    """
    Process similarity results into a structured DataFrame with dynamic scoring methods,
    handling multiple matches per scoring method
    """
    processed_data = []
    scoring_methods = results[0].keys()

    for question, result in zip(wpp_questions, results, strict=False):
        row_data = {"wpp_question": question}

        for score_name in scoring_methods:
            matches = result[score_name]

            indices = [match[2] + 1 for match in matches]
            scores = [round(match[1], 2) for match in matches]

            row_data[f"{score_name}_question"] = indices
            row_data[f"{score_name}_value"] = scores

        processed_data.append(row_data)

    df = pd.DataFrame(processed_data)

    columns = ["wpp_question"]

    for score_name in scoring_methods:
        columns.extend([f"{score_name}_question", f"{score_name}_value"])

    return df[columns]


def make_output_csv(df: pd.DataFrame, df_faq_users: pd.DataFrame) -> pd.DataFrame:
    """Process output"""

    df = pd.concat(
        [df_faq_users[["n_wpp_questions", "wpp_to_faq_annotation"]].dropna(how="all"), df], axis=1
    )

    df = df.drop("wpp_question", axis=1)
    df = df.query("n_wpp_questions != -1")

    return df


def main():
    df_faq_users = pd.read_excel(
        io="../../data/Perguntas_chatbot - 09.10_relacao FAQ perguntas users.xlsx",
        sheet_name="relacao_clean",
    )

    df_faq_users = fix_excel_table(df_faq_users)
    wpp_questions = remove_stopwords(df_faq_users["wpp_question"].dropna().tolist())
    faq = remove_stopwords(df_faq_users["pergunta_faq"].dropna().tolist())

    scoares = (
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

    results = []

    for i in wpp_questions:
        results.append(
            {
                score.__name__: process.extract(
                    i,
                    faq,
                    scorer=score,
                    limit=1,
                    processor=utils.default_process,
                )
                for score in scoares
            }
        )

    final_df = process_similarity_results(wpp_questions, results)
    final_df = make_output_csv(final_df, df_faq_users)

    final_df.to_csv("anotacao_output.csv", index=False)

    scores = calculate_list_scores(final_df)
    scores.to_csv("scores.csv", index=False)


main()

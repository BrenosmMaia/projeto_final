import os
import sys

import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from rapidfuzz import fuzz, process, utils

from utils import (
    calculate_list_scores,
    fix_excel_table,
    process_similarity_results,
    remove_stopwords,
)


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
        io="../../../data/Perguntas_chatbot - 09.10_relacao FAQ perguntas users.xlsx",
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
                    limit=3,
                    processor=utils.default_process,
                )
                for score in scoares
            }
        )

    metohds_results = process_similarity_results(wpp_questions, results)
    metohds_results = make_output_csv(metohds_results, df_faq_users)
    metohds_results.to_csv("metohds_results.csv", index=False)

    methods_scores = calculate_list_scores(metohds_results)
    methods_scores.to_csv("scores.csv", index=False)


main()

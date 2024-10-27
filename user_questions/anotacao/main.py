from typing import Dict, List, Tuple
import pandas as pd
from rapidfuzz import fuzz, process, utils


def fix_excel_table(df: pd.DataFrame) -> pd.DataFrame:
    """Apply fixes such as renaming columns and changing data types to the given DataFrame."""

    df = df.rename(
        columns={
            df.columns[0]: "n_wpp_questions",
            df.columns[1]: "pergunta_wpp",
            df.columns[3]: "n_faq_question",
            df.columns[5]: "pergunta_faq",
        }
    )

    df["n_wpp_questions"] = df["n_wpp_questions"].fillna(-1).astype(int)

    df = df.drop(df.columns[3], axis=1)

    return df


def process_similarity_results(
    wpp_questions: List[str], results: List[Dict[str, List[Tuple[str, float, int]]]]
) -> pd.DataFrame:
    """
    Process similarity results into a structured DataFrame with dynamic scoring methods
    """
    processed_data = []

    scoring_methods = results[0].keys()

    for question, result in zip(wpp_questions, results):
        row_data = {"pergunta_wpp": question}

        for score_name in scoring_methods:
            matched_text, score, index = result[score_name][0]
            row_data[f"{score_name}_question"] = index + 1
            row_data[f"{score_name}_value"] = round(score, 3)

        processed_data.append(row_data)

    df = pd.DataFrame(processed_data)

    columns = ["pergunta_wpp"]
    for score_name in scoring_methods:
        columns.extend([f"{score_name}_question", f"{score_name}_value"])

    return df[columns]


def main():
    df_faq_users = pd.read_excel(
        io="../../data/Perguntas_chatbot - 09.10_v4.xlsx", sheet_name="ANOTACAO"
    )

    df_faq_users = fix_excel_table(df_faq_users)
    wpp_questions = df_faq_users["pergunta_wpp"].dropna().tolist()
    faq = df_faq_users["pergunta_faq"].dropna().tolist()

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
    final_df["pergunta_wpp"] = range(1, len(final_df) + 1)

    final_df.to_csv("anotacao_output.csv", index=False)


main()

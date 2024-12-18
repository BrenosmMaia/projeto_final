import os
import sys

import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from rapidfuzz import fuzz, process, utils

from utils import fix_excel_table, remove_stopwords


def process_similarity_results(
    wpp_questions: list[str], results: list[dict[str, list[tuple[str, float, int]]]]
) -> pd.DataFrame:
    """
    Process similarity results into a structured DataFrame with dynamic scoring methods,
    handling multiple matches per scoring method
    """
    processed_data = []
    scoring_methods = results[0].keys()

    for question, result in zip(wpp_questions, results):
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

    df["wpp_question"] = range(1, len(df) + 1)
    df = pd.concat(
        [df, df_faq_users["wpp_to_faq_annotation"].dropna(how="all")], axis=1
    )
    last_col = df.pop("wpp_to_faq_annotation")
    df.insert(1, "wpp_to_faq_annotation", last_col)

    return df


def calculate_scores(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate the score of each similarity method."""

    similarity_columns = [
        col for col in df.columns if col.endswith("_question") and col != "wpp_question"
    ]
    results = []

    valid_df = df[
        ~pd.isna(df["wpp_to_faq_annotation"])
        & (df["wpp_to_faq_annotation"].str.lower() != "none")
    ]

    ground_truth_sets = valid_df["wpp_to_faq_annotation"].apply(
        lambda x: set(map(str.strip, str(x).split(";")))
    )

    def parse_prediction(pred):
        try:
            if isinstance(pred, list):
                return [str(p) for p in pred]
        except Exception:
            return [str(pred)]

    for method in similarity_columns:
        predictions = valid_df[method]
        correct_questions = []

        for idx, (pred, gt_set) in enumerate(zip(predictions, ground_truth_sets)):
            pred_list = parse_prediction(pred)
            if any(p in gt_set for p in pred_list):
                correct_questions.append(int(valid_df.iloc[idx]["wpp_question"]))

        method_name = method.replace("_question", "")
        results.append(
            {
                "similarity_method": method_name,
                "score": len(correct_questions),
                "right_questions": correct_questions,
            }
        )

    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values("score", ascending=False)

    return results_df


def main():
    df_faq_users = pd.read_excel(
        io="../../../data/Perguntas_chatbot - 09.10_relacao FAQ perguntas users.xlsx",
        sheet_name="relacao",
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
                    limit=2,
                    processor=utils.default_process,
                )
                for score in scoares
            }
        )

    metohds_results = process_similarity_results(wpp_questions, results)
    metohds_results = make_output_csv(metohds_results, df_faq_users)
    metohds_results.to_csv("metohds_results.csv", index=False)

    methods_scores = calculate_scores(metohds_results)
    methods_scores.to_csv("scores.csv", index=False)

    print(methods_scores)


main()

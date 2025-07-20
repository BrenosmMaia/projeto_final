import os
import sys

import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
import torch

from bert_score import score as bert_scorer
from utils import (
    calculate_list_scores,
    fix_excel_table,
)


def make_output_csv(df_results: pd.DataFrame, df_annotations: pd.DataFrame) -> pd.DataFrame:
    """Merges the similarity results with the original annotations and removes the merge key column"""

    annotation_columns = ["n_wpp_questions", "wpp_question", "wpp_to_faq_annotation"]
    for col in annotation_columns:
        if col not in df_annotations.columns:
            raise ValueError(f"Missing required column in the annotation DataFrame: {col}")

    df_annotations_unique = (
        df_annotations[annotation_columns]
        .dropna(subset=["wpp_question"])
        .drop_duplicates(subset=["wpp_question"])
    )
    df_merged = pd.merge(df_annotations_unique, df_results, on="wpp_question", how="left")

    df_final = df_merged.drop("wpp_question", axis=1)

    return df_final


def process_bert_score_results(
    original_questions: list[str],
    f1_scores_matrix: torch.Tensor,
    limit: int = 3,
) -> pd.DataFrame:
    """Processes the reshaped (N x M) F1 score matrix from BERT Score"""

    processed_data = []
    for i, question in enumerate(original_questions):
        top_scores, top_indices = torch.topk(f1_scores_matrix[i], limit)

        row_data = {
            "wpp_question": question,  # Use the original question as the key for merging
            "bert_score_question": [idx.item() + 1 for idx in top_indices],
            "bert_score_value": [round(score.item(), 4) for score in top_scores],
        }
        processed_data.append(row_data)

    return pd.DataFrame(processed_data)


def main():
    df_faq_users = pd.read_excel(
        io="../../../data/Perguntas_chatbot_clean - 09.10_relacao FAQ perguntas users.xlsx",
        sheet_name="relacao_clean",
    )

    df_faq_users = fix_excel_table(df_faq_users)
    wpp_questions = df_faq_users["wpp_question"].dropna().tolist()
    faq = df_faq_users["pergunta_faq"].dropna().tolist()

    num_wpp = len(wpp_questions)
    num_faq = len(faq)
    cands_batch = [q for q in wpp_questions for _ in range(num_faq)]
    refs_batch = faq * num_wpp

    P, R, F1 = bert_scorer(cands_batch, refs_batch, lang="pt", model_type="bert-base-multilingual-cased")
    f1_matrix = F1.view(num_wpp, num_faq)

    methods_results = process_bert_score_results(wpp_questions, f1_matrix, limit=3)
    methods_results_final = make_output_csv(methods_results, df_faq_users)
    methods_results_final.to_csv("methods_results_bert.csv", index=False)

    methods_scores = calculate_list_scores(methods_results_final)
    methods_scores.to_csv("methods_scores_bert.csv", index=False)


main()

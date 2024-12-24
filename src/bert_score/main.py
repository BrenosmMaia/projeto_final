import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import pandas as pd

from bert_score import score
from utils import calculate_scores, fix_excel_table, remove_stopwords


def calculate_bert_score(
    whatsapp_questions: list[str],
    faq_questions: list,
    lang: str = "pt",
    model_type: str = "bert-base-multilingual-cased",
) -> pd.DataFrame:
    "Calculate bert score between elements of two lists"

    if not whatsapp_questions or not faq_questions:
        return pd.DataFrame()

    results = []

    for i, wq in enumerate(whatsapp_questions):
        _, _, F1 = score(
            [wq] * len(faq_questions),
            faq_questions,
            lang=lang,
            model_type=model_type,
        )

        best_score = F1.max().item()
        best_match_idx = F1.argmax().item()

        results.append(
            {
                "wpp_question": int(i + 1),
                "bert_match_faq_question": int(best_match_idx + 1),
                "score": round(best_score, 4),
            }
        )

    return pd.DataFrame(results)


def main():
    df_faq_users = pd.read_excel(
        io="../../data/Perguntas_chatbot - 09.10_relacao FAQ perguntas users.xlsx",
        sheet_name="relacao",
    )

    df_faq_users = fix_excel_table(df_faq_users)
    wpp_questions = remove_stopwords(df_faq_users["wpp_question"].dropna().tolist())
    faq = remove_stopwords(df_faq_users["pergunta_faq"].dropna().tolist())

    bert_scores = calculate_bert_score(wpp_questions, faq)

    df_final = pd.concat(
        [bert_scores, df_faq_users["wpp_to_faq_annotation"].dropna(how="all")], axis=1
    )

    last_col = df_final.pop("wpp_to_faq_annotation")
    df_final.insert(1, "wpp_to_faq_annotation", last_col)

    df_final.to_csv("bert_scores.csv", index=False)

    bert_scores_evaluation = calculate_scores(df_final)

    bert_scores_evaluation.to_csv("bert_scores_evaluation.csv", index=False)


main()

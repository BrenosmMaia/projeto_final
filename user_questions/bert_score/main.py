import os
import re
import sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import nltk
import pandas as pd
from bert_score import score
from nltk.corpus import stopwords
from utils import calculate_scores, fix_excel_table


def remove_stopwords(strings: list[str]) -> list[str]:
    """Removes Portuguese stopwords, greetings and punctuation (except ?)
    from a list of strings."""

    try:
        stop_words = set(stopwords.words("portuguese"))
    except LookupError:
        nltk.download("stopwords", quiet=True)
        stop_words = set(stopwords.words("portuguese"))

    greetings = {"bom dia", "boa tarde", "boa noite"}

    def clean_text(text: str) -> str:
        text_lower = text.lower()
        for greeting in greetings:
            text_lower = re.sub(f"{greeting}[!.,]?", "", text_lower)
        text_lower = re.sub(r"[^\w\s\?]", "", text_lower)
        return " ".join(
            word
            for word in text_lower.split()
            if word not in stop_words and word.strip()
        ).strip()

    return [clean_text(text) for text in strings]


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

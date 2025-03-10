import re

import nltk
import pandas as pd
from nltk.corpus import stopwords


def fix_excel_table(df: pd.DataFrame) -> pd.DataFrame:
    """Apply fixes such as renaming columns and changing data
    types to the given DataFrame."""

    df = df.rename(
        columns={
            df.columns[0]: "n_wpp_questions",
            df.columns[1]: "wpp_question",
            df.columns[2]: "wpp_to_faq_annotation",
            df.columns[4]: "n_faq_question",
            df.columns[5]: "pergunta_faq",
        }
    )

    df["n_wpp_questions"] = df["n_wpp_questions"].fillna(-1).astype(int)

    df = df.drop(df.columns[3], axis=1)

    return df


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
            word for word in text_lower.split() if word not in stop_words and word.strip()
        ).strip()

    return [clean_text(text) for text in strings]


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


def calculate_scores(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate the score of each similarity method."""

    similarity_columns = [
        col for col in df.columns if col.endswith("_question") and col != "wpp_question"
    ]
    results = []

    valid_df = df[
        ~pd.isna(df["wpp_to_faq_annotation"]) & (df["wpp_to_faq_annotation"].str.lower() != "none")
    ]

    ground_truth_sets = valid_df["wpp_to_faq_annotation"].apply(
        lambda x: set(map(str.strip, str(x).split(";")))
    )

    for method in similarity_columns:
        predictions = valid_df[method].astype(str)
        correct_questions = []

        for idx, (pred, gt_set) in enumerate(zip(predictions, ground_truth_sets, strict=False)):
            if pred in gt_set:
                correct_questions.append(int(valid_df.iloc[idx]["n_wpp_questions"]))

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


def calculate_list_scores(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate the score of each similarity method."""

    similarity_columns = [
        col for col in df.columns if col.endswith("_question") and col != "wpp_question"
    ]
    results = []

    valid_df = df[
        ~pd.isna(df["wpp_to_faq_annotation"]) & (df["wpp_to_faq_annotation"].str.lower() != "none")
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

        for idx, (pred, gt_set) in enumerate(zip(predictions, ground_truth_sets, strict=False)):
            pred_list = parse_prediction(pred)
            if any(p in gt_set for p in pred_list):
                correct_questions.append(int(valid_df.iloc[idx]["n_wpp_questions"]))

        method_name = method.replace("_question", "")
        results.append(
            {
                "similarity_method": method_name,
                "accuracy": round(len(correct_questions) / len(ground_truth_sets), 3),
                "total_questions": len(ground_truth_sets),
                "right_questions": correct_questions,
            }
        )

    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values("accuracy", ascending=False)

    return results_df


def replace_acronyms(text_list: list[str]) -> list[str]:
    """Replace acronyms in a list of strings with their full descriptions"""

    acronym_dict = {
        "ASPLO": "Assessorias Setoriais de Planejamento e Orçamento (ASPLO)",
        "SIPLAG": "Sistema de Inteligência em Planejamento e Gestão (SIPLAG)",
        "REDEPLAN": "Rede de Planejamento (REDEPLAN)",
        "PPA": "Plano Plurianual (PPA)",
        "SUPLAN": "Superintendência de Planejamento (SUPLAN)",
        "SUBPLO": "Subsecretaria de Planejamento e Orçamento (SUBPLO)",
    }

    patterns = {
        acronym: re.compile(r"\b" + re.escape(acronym) + r"\b", re.IGNORECASE)
        for acronym in sorted(acronym_dict, key=len, reverse=True)
    }

    result_list = []
    for text in text_list:
        result = text
        for acronym, pattern in patterns.items():
            result = pattern.sub(acronym_dict[acronym], result, 1)
        result_list.append(result)

    return result_list

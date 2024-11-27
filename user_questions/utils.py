import pandas as pd


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


def calculate_scores(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate the score of each similarity method."""

    similarity_columns = [col for col in df.columns if col.endswith("_question") and col != "wpp_question"]
    results = []

    valid_df = df[
        ~pd.isna(df["wpp_to_faq_annotation"])
        & (df["wpp_to_faq_annotation"].str.lower() != "none")
    ]

    ground_truth_sets = valid_df["wpp_to_faq_annotation"].apply(
        lambda x: set(map(str.strip, str(x).split(";")))
    )

    for method in similarity_columns:
        predictions = valid_df[method].astype(str)
        correct_questions = []

        for idx, (pred, gt_set) in enumerate(zip(predictions, ground_truth_sets)):
            if pred in gt_set:
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

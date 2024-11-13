import pandas as pd

def calculate_scores(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate the score of each similarity method."""

    similarity_columns = [col for col in df.columns if col.endswith('_question')]
    results = []

    valid_df = df[~pd.isna(df['wpp_to_faq_annotation']) & \
                (df['wpp_to_faq_annotation'].str.lower() != 'none')]

    ground_truth_sets = valid_df['wpp_to_faq_annotation'].apply(
        lambda x: set(map(str.strip, str(x).split(';')))
    )

    for method in similarity_columns:
        predictions = valid_df[method].astype(str)
        correct_questions = []

        for idx, (pred, gt_set) in enumerate(zip(predictions, ground_truth_sets)):
            if pred in gt_set:
                correct_questions.append(int(valid_df.iloc[idx]['pergunta_wpp']))

        method_name = method.replace('_question', '')
        results.append({
            'similarity_method': method_name,
            'score': len(correct_questions),
            'right_questions': correct_questions
        })

    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('score', ascending=False)

    return results_df

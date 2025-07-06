import pandas as pd


def load_and_filter_data(file_path: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load the CSV data and filter out rows where
    'A melhor resposta foi útil?' is empty or 'Não se Aplica'.

    Returns:
      - original_data: The full DataFrame loaded from the CSV.
      - filtered_data: The DataFrame with the selected columns and filtered rows.
    """
    original_data = pd.read_csv(file_path)
    relevant_data = original_data[["Qual a melhor resposta?", "A melhor resposta foi útil?"]]
    filtered_data = relevant_data[
        ~relevant_data["A melhor resposta foi útil?"].isin(["", "Não se Aplica"])
    ]
    return original_data, filtered_data


def generate_results_dataframe(
    filtered_data: pd.DataFrame, original_data: pd.DataFrame
) -> pd.DataFrame:
    """
    Generate a DataFrame with the evaluation results for LLM 1 and LLM 2.

    The new definition for "Acurácia" is:

         Acurácia = (Respostas úteis / Total respostas avaliadas) * 100

    A response is considered useful for a given LLM only if:
       - The column "Qual a melhor resposta?" exactly equals the model name (e.g., "LLM 2")
       - The column "A melhor resposta foi útil?" equals "SIM"

    Total respostas avaliadas is computed as all rows where
    "A melhor resposta foi útil?" is either "SIM" or "NÃO", regardless of the model.
    """
    # Calculate total evaluated responses (common denominator)
    evaluated_mask = filtered_data["A melhor resposta foi útil?"].isin(["SIM", "NÃO"])
    total_evaluated = len(filtered_data[evaluated_mask])

    # Count useful responses per model (where each response must also match the model)
    llm1_useful = (
        (filtered_data["Qual a melhor resposta?"] == "LLM 1")
        & (filtered_data["A melhor resposta foi útil?"] == "SIM")
    ).sum()
    llm2_useful = (
        (filtered_data["Qual a melhor resposta?"] == "LLM 2")
        & (filtered_data["A melhor resposta foi útil?"] == "SIM")
    ).sum()

    # Calculate accuracy per model
    llm1_accuracy = round((llm1_useful / total_evaluated) * 100, 3) if total_evaluated > 0 else 0
    llm2_accuracy = round((llm2_useful / total_evaluated) * 100, 3) if total_evaluated > 0 else 0

    # Count total responses from original_data: valid responses for 'Resposta LLM 1' are strings with length >= 5
    total_respostas = (original_data["Resposta LLM 1"].fillna("").str.len() >= 5).sum()

    results = pd.DataFrame(
        {
            "Modelo": ["LLM 1", "LLM 2"],
            "Acurácia": [llm1_accuracy, llm2_accuracy],
            "Respostas úteis": [llm1_useful, llm2_useful],
            "Total respostas avaliadas": [total_evaluated, total_evaluated],
            "Total respostas": [total_respostas, total_respostas],
        }
    )

    return results


def main():
    file_path = "../../../../data/Respostas LLM Para Perguntas Wpp - relacao.csv"
    original_data, filtered_data = load_and_filter_data(file_path)
    results_df = generate_results_dataframe(filtered_data, original_data)
    results_df.to_csv("evaluation.csv", index=False)


if __name__ == "__main__":
    main()

import pandas as pd


def load_and_filter_data(file_path: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load the CSV data and filter out rows where
    'A melhor resposta foi útil?' is either empty or 'Não se Aplica'.

    Returns a tuple containing:
      - original_data: The full DataFrame loaded from CSV.
      - filtered_data: The subset DataFrame with the selected columns and filtered rows.
    """
    original_data = pd.read_csv(file_path)
    # Select the relevant columns
    relevant_data = original_data[["Qual a melhor resposta?", "A melhor resposta foi útil?"]]
    # Filter out rows with undesired responses
    filtered_data = relevant_data[
        ~relevant_data["A melhor resposta foi útil?"].isin(["", "Não se Aplica"])
    ]
    return original_data, filtered_data


def generate_results_dataframe(
    filtered_data: pd.DataFrame, original_data: pd.DataFrame
) -> pd.DataFrame:
    """
    Generate a DataFrame with the evaluation results for LLM 1 and LLM 2.
    """

    is_useful = filtered_data["A melhor resposta foi útil?"].isin(["SIM", "NÃO"])
    useful_responses = filtered_data[is_useful]

    is_llm1 = useful_responses["Qual a melhor resposta?"] == "LLM 1"
    is_llm2 = useful_responses["Qual a melhor resposta?"] == "LLM 2"

    llm1_useful = is_llm1.sum()
    llm2_useful = is_llm2.sum()

    llm1_correct = (is_llm1 & (useful_responses["A melhor resposta foi útil?"] == "SIM")).sum()
    llm2_correct = (is_llm2 & (useful_responses["A melhor resposta foi útil?"] == "SIM")).sum()

    total_evaluated = len(useful_responses)

    llm1_accuracy = round((llm1_correct / total_evaluated) * 100, 3) if total_evaluated > 0 else 0
    llm2_accuracy = round((llm2_correct / total_evaluated) * 100, 3) if total_evaluated > 0 else 0

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

import os
import sys

import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from utils import calculate_scores


def main():
    anotacao_output = pd.read_csv("../anotacao_output.csv")

    columns_to_check = [
        "partial_ratio_question",
        "token_set_ratio_question",
        "partial_token_sort_ratio_question",
        "token_ratio_question",
        "token_sort_ratio_question",
    ]

    anotacao_output["ensemble_question"] = (
        anotacao_output[columns_to_check].mode(axis=1)[0].astype(int)
    )

    scores = calculate_scores(anotacao_output)

    scores.to_csv("scores_ensemble.csv", index=False)
    anotacao_output.to_csv("anotacao_ensemble_output.csv", index=False)


main()

import json
import pandas as pd

from typing import Dict, List, Tuple
from rapidfuzz import fuzz, process, utils


def read_input() -> Dict[str, str]:
    """Reads the input question from a json file"""

    with open("input.json", "r", encoding="utf-8") as file:
        data = json.load(file)
    return data["perguntas"]


def create_match_dataframe(
    data_list: List[Dict[str, List[Tuple[str, int]]]],
    pergunta_base_e_match: Dict[str, str]
) -> pd.DataFrame:
    """Creates a DataFrame with the results of the matching process"""

    def check_match(tuple_list, pergunta_match):
        return any(pergunta_match == item[0] for item in tuple_list)
    
    results = []
    
    for data, (pergunta_base, pergunta_match) in zip(data_list, pergunta_base_e_match):
        result = {
            'pergunta_base': pergunta_base,
            'pergunta_match': pergunta_match
        }
        
        for ratio_func, tuple_list in data.items():
            result[ratio_func] = check_match(tuple_list, pergunta_match)
        
        results.append(result)
    
    df = pd.DataFrame(results)
    
    return df


def main():
    questions_and_expected = read_input()
    questions_df = pd.read_excel(io="../../data/perguntas_chatbot_v1.xlsx")
    questions_faq = questions_df["perguntas"].tolist()

    scoares = (
        fuzz.ratio,
        fuzz.partial_ratio,
        fuzz.token_set_ratio,
        fuzz.partial_token_set_ratio,
        fuzz.token_sort_ratio,
        fuzz.partial_token_sort_ratio,
        fuzz.token_ratio,
        fuzz.partial_token_ratio,
        fuzz.WRatio
    )
    
    results = []
    for i in questions_and_expected:
        results.append({score.__name__: process.extract(i[0], questions_faq, scorer=score, limit=2, processor=utils.default_process) for score in scoares})

    df = create_match_dataframe(results, questions_and_expected)
    df.to_csv('output.csv', index=False)

main()
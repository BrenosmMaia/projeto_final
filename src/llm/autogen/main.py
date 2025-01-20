import ast
import os
import sys

import autogen
import pandas as pd
from llm_config import llama_bedrock

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from utils import fix_excel_table, calculate_list_scores

def parse_response(json_block: str) -> dict:
    json_block = json_block.strip()
    
    if json_block.startswith("```json"):
        json_block = json_block[len("```json"):]

    if json_block.endswith("```"):
        json_block = json_block[:-3]
    
    return  ast.literal_eval(json_block.strip())

def create_assistants() -> tuple[autogen.ConversableAgent, autogen.ConversableAgent]:
    """Create LLM assistants."""

    system_message = """
    Você é um assistente que irá receber uma pergunta de usuário de \
um FAQ. Você deve retornar as 3 perguntas do FAQ que são mais similares ao que o usuário \
quer saber. Use seu conhecimento para entender quais as perguntas do FAQ que melhor\
representem a pergunta do usuário.

Retorne sempre apenas um Json com o formato:
{'pergunta': 'pergunta do FAQ', indice: 'indice da pergunta do FAQ', \
'justicativa': 'justificação da resposta'}

Não retorne nada além do json.
"""

    assistant = autogen.ConversableAgent(
        "assistant",
        llm_config={
            "config_list": llama_bedrock,
        },
        system_message=system_message,
    )

    user_proxy = autogen.ConversableAgent(
        "user_proxy",
        human_input_mode="ALWAYS",
        is_termination_msg=lambda x: x.get("content", "") and "TERMINATE" in x.get("content", ""),
        max_consecutive_auto_reply=1,
    )

    return assistant, user_proxy

def make_output_csv(df: pd.DataFrame, df_faq_users: pd.DataFrame) -> pd.DataFrame:
    """Process output"""

    df = pd.concat(
        [df, df_faq_users["wpp_to_faq_annotation"].dropna(how="all")], axis=1
    )
    last_col = df.pop("wpp_to_faq_annotation")
    df.insert(1, "wpp_to_faq_annotation", last_col)

    return df

def main():
    all_responses = []

    df_faq_users = pd.read_excel(
        io="../../../data/Perguntas_chatbot - 09.10_relacao FAQ perguntas users.xlsx",
        sheet_name="relacao",
    )

    df_faq_users = fix_excel_table(df_faq_users)
    wpp_questions = df_faq_users["wpp_question"].dropna().tolist()
    faq = df_faq_users["pergunta_faq"].dropna().tolist()
    faq_formatted = "\n".join([f"{i}: {q}" for i, q in enumerate(faq)])

    for i in range(len(wpp_questions)):

        assistant, user_proxy = create_assistants()

        message = f"""\
            Dado a pergunta de usuário abaixo, quais são as 3 perguntas do FAQ que são mais \
    similares ao que o usuário quer saber?

            Pergunta do usuário: {wpp_questions[i]}
            Perguntas do FAQ:\n{faq_formatted}

            Retorne sempre apenas um Json com o formato:
        {{'indices: ['indices da pergunta do FAQ'], \
'justicativa': 'justificação da resposta'}}
            """

        chat_result = user_proxy.initiate_chat(
            assistant,
            message=message,
            max_turns=1,
        )

        response = parse_response(chat_result.chat_history[1]["content"])
        response["wpp_question"] = i + 1
        response['llm_question'] = response.pop('indices') 

        all_responses.append(response)

    response = pd.DataFrame(all_responses)
    response['justicativa'] = response.pop('justicativa')
    response = make_output_csv(response, df_faq_users)
    response.to_csv("llm_results.csv", index=False)

    llm_scores = calculate_list_scores(response)
    llm_scores.to_csv("llm_scores.csv", index=False)

main()

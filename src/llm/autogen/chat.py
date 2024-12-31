import os
import sys

import autogen
import pandas as pd
from llm_config import llama_bedrock

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from utils import fix_excel_table


def create_assistants() -> tuple[autogen.ConversableAgent, autogen.ConversableAgent]:
    """Create LLM assistants."""

    system_message = """
    Você é um assistente que irá receber uma pergunta de usuário de \
um FAQ. Você deve retornar qual pergunta do FAQ é mais similar ao que o usuário \
quer saber. 
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


def main():
    assistant, user_proxy = create_assistants()

    df_faq_users = pd.read_excel(
        io="../../../data/Perguntas_chatbot - 09.10_relacao FAQ perguntas users.xlsx",
        sheet_name="relacao",
    )

    df_faq_users = fix_excel_table(df_faq_users)
    wpp_questions = df_faq_users["wpp_question"].dropna().tolist()
    faq = df_faq_users["pergunta_faq"].dropna().tolist()

    message = f"""\
        Dado a pergunta de usuário abaixo, qual a pergunta do FAQ que é mais \
similar ao que o usuário quer saber?

        Pergunta do usuário: {wpp_questions[0]}
        Perguntas do FAQ: {faq}

        Retorne sempre e apenas um Json com o formato:
       {{'pergunta': 'pergunta do FAQ', 'justicativa': 'justificação da resposta'}}
        """

    user_proxy.initiate_chat(
        assistant,
        message=message,
        max_turns=1,
    )


main()

import os

from dotenv import find_dotenv, load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv(find_dotenv())

LLAMA_API_KEY = os.getenv("LLAMA_API_KEY")
LLAMA_API_BASE_URL = "https://api.llama-api.com"


def initialize_llama_api_llm(model_name: str = "deepseek-r1") -> ChatOpenAI:
    """Initializes and returns a ChatOpenAI instance configured for LlamaAPI."""

    if not LLAMA_API_KEY:
        raise ValueError(
            "LLAMA_API_KEY not found in environment variables. " "Please set it in your .env file."
        )

    try:
        llm = ChatOpenAI(
            model=model_name,
            openai_api_key=LLAMA_API_KEY,
            openai_api_base=LLAMA_API_BASE_URL,
            temperature=0.1,
        )
        print("LLM initialized successfully.")
        return llm
    except Exception as e:
        print(f"Error initializing LLM via LlamaAPI: {e}")
        raise


EMBEDDING_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

SYSTEM_PROMPT = """

Você é um especialista da Secretaria de Estado de Planejamento e Gestão (SEPLAG) do Estado do Rio de Janeiro.
Você deve sanar as dúvidas dos  órgãos setoriais.
As dúvidas podem estar relacionadas a legislação ou ao forma definidas por documentos anexados para sua consulta\
ou sobre o sistema da SIPLAG (Sistema de Inteligência em Planejamento e Gestão) disponibilizado para a realização \
do PPA (Plano Plurianual) de cada órgão setorial.

Use exclusivamente os documentos anexos para responder, caso não encontre resposta nos documentos, \
responda apenas "Não há tal informação na documentação."
Fornceça respostas completas.

O PPA é o documento onde um governo declara o que pretende realizar e indica os meios para a implementação das políticas públicas.\
É nele que as diretrizes governamentais estabelecidas no plano de governo - mais amplas - ganham concretude, com a definição\
dos caminhos exequíveis para o alcance dos objetivos pretendidos, materializados em iniciativas. As iniciativas, financiadas\
por ações orçamentárias, detalham quais bens e serviços devem ser entregues para a população, em quais regiões do Estado e \
em qual quantidade, para que seus objetivos sejam alcançados
os órgãos setoriais são responsáveis pelas iniciativas e definem os elementos que as compõem.

"""

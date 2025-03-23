import os

from dotenv import find_dotenv, load_dotenv

from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI
from llama_index.llms.openai_like import OpenAILike

load_dotenv(find_dotenv())

llama = OpenAILike(
    api_base="https://api.llama-api.com",
    api_key=os.getenv("LLAMA_API_KEY"),
    model="llama3.3-70b",
    is_function_calling_model=True,
    is_chat_model=True,
    temperature=0.1,
)

deepseek = OpenAILike(
    api_base="https://api.llama-api.com",
    api_key=os.getenv("LLAMA_API_KEY"),
    model="deepseek-r1",
    is_function_calling_model=True,
    is_chat_model=True,
    temperature=0.1,
)


hugging_face_embedding = HuggingFaceEmbedding(
    model_name="BAAI/bge-small-en-v1.5",
    device="cpu",
)


gpt_4o = OpenAI(model="gpt-4o", api_key=os.getenv("OPENAI_KEY"), temperature=0.1)

openai_embedding = OpenAIEmbedding(model="text-embedding-ada-002", api_key=os.getenv("OPENAI_KEY"))

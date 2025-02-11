from llm_config import llama

from llama_index.core import Settings, SimpleDirectoryReader, VectorStoreIndex
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

embed_model = HuggingFaceEmbedding(
    model_name="BAAI/bge-small-en-v1.5",
    device="cpu",
)

Settings.llm = llama
Settings.embed_model = embed_model
Settings.context_window = 128_000

documents = SimpleDirectoryReader(
    input_files=["pdfs/Manual de Elaboração PPA 24-27.pdf"]
).load_data()

index = VectorStoreIndex.from_documents(documents)

query_engine = index.as_query_engine()

response = query_engine.query("Resuma o PPA")
print(f"Answer: {response}")

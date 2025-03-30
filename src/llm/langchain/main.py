import os
import sys
import traceback


from typing import TypedDict

from langchain.text_splitter import RecursiveCharacterTextSplitter

# LangChain Components
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings

from langgraph.graph import END, StateGraph

# --- Import the LLM initializer ---
from llm_config import SYSTEM_PROMPT, initialize_llama_api_llm, EMBEDDING_MODEL_NAME

EMBEDDING_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

try:
    llm = initialize_llama_api_llm()
except ValueError as e:
    print(f"Fatal Error: Could not initialize LLM. {e}")
    sys.exit(1)  # Exit if LLM setup fails
except Exception as e:
    print(f"Fatal Error during LLM initialization: {e}")
    sys.exit(1)


# --- 1. Document Loading and Processing (Function remains the same) ---
def load_and_process_pdfs(pdf_paths: list[str], embedding_model_name: str):
    """Loads specific PDFs from a list, splits them, creates embeddings, and returns a retriever."""
    all_docs: list[Document] = []
    print("\n--- Loading and Processing PDFs ---")
    print(f"Using embedding model: {embedding_model_name}")

    for pdf_path in pdf_paths:
        if not os.path.exists(pdf_path):
            print(f"Warning: PDF file not found at {pdf_path}. Skipping.")
            continue
        try:
            print(f"Loading: {pdf_path}")
            loader = PyPDFLoader(pdf_path)
            documents = loader.load()
            all_docs.extend(documents)
            print(f"Loaded {len(documents)} pages from {os.path.basename(pdf_path)}.")
        except Exception as e:
            print(f"Error loading {pdf_path}: {e}")

    if not all_docs:
        print("Error: No documents were successfully loaded. Cannot proceed.")
        return None

    print(f"\nTotal documents loaded: {len(all_docs)}")

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    split_docs = text_splitter.split_documents(all_docs)
    print(f"Split into {len(split_docs)} text chunks.")

    if not split_docs:
        print("Error: No text chunks generated after splitting. Cannot create vector store.")
        return None

    print(f"Initializing embedding model: {embedding_model_name}")
    try:
        embeddings = HuggingFaceEmbeddings(model_name=embedding_model_name)
    except Exception as e:
        print(f"Error initializing embedding model: {e}")
        return None

    print("Creating FAISS vector store in memory...")
    try:
        db = FAISS.from_documents(split_docs, embeddings)
        print("Vector store created successfully.")
        return db.as_retriever(search_kwargs={"k": 5})  # Retrieve top 5
    except Exception as e:
        print(f"Error creating FAISS vector store: {e}")
        return None


# --- 3. LangGraph Agent State (Remains the same) ---
class AgentState(TypedDict):
    query: str
    documents: list[str]
    generation: str
    iterations: int


# --- 4. LangGraph Nodes (Remain the same, using the global 'llm' instance) ---
retriever = None  # Will be initialized in main


def retrieve_docs(state: AgentState) -> AgentState:
    """Retrieves documents relevant to the query using the initialized retriever."""
    print("---NODE: RETRIEVE DOCUMENTS---")
    if retriever is None:
        print("Error: Retriever is not initialized.")
        return {"documents": [], "iterations": state.get("iterations", 0) + 1}

    query = state["query"]
    iterations = state.get("iterations", 0)
    print(f"Retrieving documents for query: {query}")
    try:
        retrieved_docs = retriever.invoke(query)
        doc_contents = [doc.page_content for doc in retrieved_docs]
        print(f"Retrieved {len(doc_contents)} documents.")
        return {"documents": doc_contents, "iterations": iterations + 1}
    except Exception as e:
        print(f"Error during document retrieval: {e}")
        return {"documents": [], "iterations": iterations + 1}


def grade_documents(state: AgentState) -> AgentState:
    """Determines whether the retrieved documents are relevant to the query."""
    print("---NODE: GRADE DOCUMENTS---")
    query = state["query"]
    documents = state["documents"]

    if not documents:
        print("No documents retrieved to grade.")
        return {"documents": documents}

    print("Documents found, proceeding to generation.")
    return {"documents": documents}


def generate_answer(state: AgentState) -> AgentState:
    """Generates an answer using the LLM based on the query and retrieved documents."""
    print("---NODE: GENERATE ANSWER---")
    query = state["query"]
    documents = state["documents"]

    if not documents:
        print("No relevant documents found to generate answer.")
        generation = "I couldn't find relevant information in the provided documents to answer your question."
        return {"generation": generation}

    context = "\n\n".join(documents)
    prompt = f"""{SYSTEM_PROMPT}".

Context:
{context}

Question: {query}

Answer:"""

    print("Generating answer with LLM...")
    try:
        # Use the globally initialized llm instance
        response = llm.invoke(prompt)
        generation = response.content
        print(f"LLM Generation: {generation}")
    except Exception as e:
        print(f"Error during LLM generation: {e}")
        generation = "Sorry, I encountered an error while generating the answer."

    return {"generation": generation}


def fallback(state: AgentState) -> AgentState:
    """Fallback node if generation fails or no documents are found after grading."""
    print("---NODE: FALLBACK---")
    if not state.get("documents"):
        generation = "I could not find relevant information in the specified documents to answer your query."
    else:
        generation = "Sorry, I encountered an issue and could not process your request based on the documents."
    return {"generation": generation}


# --- 5. Define Conditional Edges (Remains the same) ---
def decide_to_generate(state: AgentState) -> str:
    """Determines whether to generate an answer or fallback based on graded documents."""
    print("---CONDITIONAL EDGE: DECIDE TO GENERATE---")
    documents = state["documents"]
    iterations = state["iterations"]

    if iterations > 3:
        print("Max iterations reached, going to fallback.")
        return "fallback"

    if documents:
        print("Decision: Relevant documents found -> Generate Answer")
        return "generate"
    else:
        print("Decision: No relevant documents -> Fallback")
        return "fallback"


# --- 6. Build the LangGraph Graph (Remains the same) ---
print("\n--- Building Agent Graph ---")
workflow = StateGraph(AgentState)

workflow.add_node("retrieve", retrieve_docs)
workflow.add_node("grade_documents", grade_documents)
workflow.add_node("generate", generate_answer)
workflow.add_node("fallback", fallback)

workflow.set_entry_point("retrieve")

workflow.add_edge("retrieve", "grade_documents")
workflow.add_conditional_edges(
    "grade_documents",
    decide_to_generate,
    {"generate": "generate", "fallback": "fallback"},
)
workflow.add_edge("generate", END)
workflow.add_edge("fallback", END)

app = workflow.compile()
print("Graph compiled successfully.")

# --- 7. Run the Agent ---
if __name__ == "__main__":
    pdf_file_paths = [
        "../llama_index/pdfs/manual_de_revisao_PPA_2025.pdf",
        "../llama_index/pdfs/manual_de_monitoramento_PPA_2024-2027.pdf",
        "../llama_index/pdfs/manual_de_elaboracao_PPA_24-27.pdf",
        "../llama_index/pdfs/guia_operacional_SIPLAG_PPA_24-27.pdf",
        "../llama_index/pdfs/faq_PPA_SIPLAG.pdf",
    ]

    if not pdf_file_paths:
        print("\nError: No PDF file paths provided in the 'pdf_file_paths' list.")
        sys.exit(1)

    # Initialize the global retriever variable
    retriever = load_and_process_pdfs(pdf_file_paths, EMBEDDING_MODEL_NAME)

    if retriever is None:
        print("\nFailed to initialize the document retriever. Exiting.")
        sys.exit(1)

    print("\n--- Agentic RAG System Ready ---")
    # LLM details are printed during initialization now
    print("Based on documents:")
    for path in pdf_file_paths:
        if os.path.exists(path):
            print(f" - {os.path.basename(path)}")
    print("Type 'exit' to quit.")

    while True:
        user_query = input("\nAsk a question about the PDFs: ")
        if user_query.lower() == "exit":
            break
        if not user_query:
            continue

        initial_state = {"query": user_query, "iterations": 0}

        try:
            final_state = app.invoke(initial_state, {"recursion_limit": 5})
            print("\n--- Final Answer ---")
            print(final_state.get("generation", "No generation found in final state."))
            print("-" * 20)
        except Exception as e:
            print(f"\nAn error occurred during graph execution: {e}")
            traceback.print_exc()
            print("-" * 20)

    print("Exiting.")

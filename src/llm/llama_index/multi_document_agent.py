from pathlib import Path

from llm_config import (
    gpt_4o,
    hugging_face_embedding,
    llama,
    openai_embedding,
)

from llama_index.core import (
    Settings,
    SimpleDirectoryReader,
    SummaryIndex,
    VectorStoreIndex,
)
from llama_index.core.agent import AgentRunner, FunctionCallingAgentWorker, ReActAgentWorker
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.objects import ObjectIndex
from llama_index.core.tools import FunctionTool, QueryEngineTool
from llama_index.core.vector_stores import FilterCondition, MetadataFilters


def initialize_settings(model_type: str = "llama") -> None:
    """Initialize global settings for LLM and embeddings.

    Args:
        model_type: Type of model to use ('llama' or 'openai')
    """
    if model_type == "llama":
        Settings.llm = llama
        Settings.embed_model = hugging_face_embedding
        Settings.context_window = 128_000
    elif model_type == "openai":
        Settings.llm = gpt_4o
        Settings.embed_model = openai_embedding
    else:
        raise ValueError(f"Unsupported model type: {model_type}")


def get_doc_tools(file_path: str, name: str) -> tuple[FunctionTool, QueryEngineTool]:
    """Create vector and summary query tools for a document.

    Args:
        file_path: Path to the document file
        name: Name identifier for the document

    Returns:
        Tuple of (vector query tool, summary query tool)
    """
    # Load and process document
    documents = SimpleDirectoryReader(input_files=[file_path]).load_data()
    splitter = SentenceSplitter(chunk_size=1024)
    nodes = splitter.get_nodes_from_documents(documents)

    # Create vector index
    vector_index = VectorStoreIndex(nodes)

    # Vector query function
    def vector_query(query: str, page_numbers: list[str] | None = None) -> str:
        """Answer questions over specific document pages.

        Args:
            query: Search query
            page_numbers: Optional list of pages to filter
        """
        page_numbers = page_numbers or []
        metadata_dicts = [{"key": "page_label", "value": p} for p in page_numbers]

        query_engine = vector_index.as_query_engine(
            similarity_top_k=2,
            filters=MetadataFilters.from_dicts(metadata_dicts, condition=FilterCondition.OR),
        )
        return str(query_engine.query(query))

    # Create tools
    vector_tool = FunctionTool.from_defaults(name=f"vector_tool_{name}", fn=vector_query)

    summary_index = SummaryIndex(nodes)
    summary_engine = summary_index.as_query_engine(
        response_mode="tree_summarize",
        use_async=True,
    )

    summary_tool = QueryEngineTool.from_defaults(
        name=f"summary_tool_{name}",
        query_engine=summary_engine,
        description=f"Useful for summarization questions about {name}",
    )

    return vector_tool, summary_tool


def create_agent(model_type: str = "llama", papers: list[str] = None) -> AgentRunner:
    """Create an agent with appropriate configuration.

    Args:
        model_type: Type of model to use ('llama' or 'openai')
        papers: List of paper file paths

    Returns:
        Configured AgentRunner instance
    """
    # Process papers and create tools
    paper_to_tools: dict[str, list] = {}
    for paper in papers:
        print(f"Processing {Path(paper).name}")
        vector_tool, summary_tool = get_doc_tools(paper, Path(paper).stem)
        paper_to_tools[paper] = [vector_tool, summary_tool]

    all_tools = [t for paper in papers for t in paper_to_tools[paper]]

    # Create object index for tools
    obj_index = ObjectIndex.from_objects(
        all_tools,
        index_cls=VectorStoreIndex,
    )
    obj_retriever = obj_index.as_retriever(similarity_top_k=3)

    # Select appropriate agent worker
    if model_type == "llama":
        agent_worker = ReActAgentWorker.from_tools(
            tool_retriever=obj_retriever, verbose=True, max_iterations=15
        )
    else:  # openai
        agent_worker = FunctionCallingAgentWorker.from_tools(
            tool_retriever=obj_retriever, verbose=True
        )

    return AgentRunner(agent_worker)


def main():
    """Main execution function."""

    model_type = "openai"  # Options: "llama" | "openai"

    query = "Me fale sobre o PPA"  # Your query here

    papers = [
        "pdfs/manual_de_revisao_ppa_2025.pdf",
        "pdfs/manual_de_monitoramento_2024-2027.pdf",
        "pdfs/manual_de_elaboracao_PPA_24-27.pdf",
        "pdfs/guia_operacional_SIPLAG_PPA_24-27.pdf",
    ]

    # Validate model choice
    valid_models = ["llama", "openai"]
    if model_type not in valid_models:
        raise ValueError(f"Invalid model type. Choose from: {', '.join(valid_models)}")

    # Initialize components
    initialize_settings(model_type)
    agent = create_agent(model_type, papers)

    # Execute query
    response = agent.query(query)
    print(f"\nFinal Response:\n{response}")

    # Save output
    with open("output.txt", "w", encoding="utf-8") as f:
        f.write(str(response))


main()

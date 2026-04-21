import os
from typing import Optional

from llama_index.core import Document
from llama_index.core import VectorStoreIndex, get_response_synthesizer
from llama_index.core import SimpleDirectoryReader
from llama_index.core.bridge.pydantic import Field
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core.indices.vector_store import VectorIndexRetriever
from llama_index.core.llms import ChatMessage, MessageRole
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.postprocessor.sbert_rerank import SentenceTransformerRerank
from llama_index.core.postprocessor.types import BaseNodePostprocessor
from llama_index.core.prompts.base import ChatPromptTemplate
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.schema import NodeWithScore, QueryBundle, TransformComponent
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.vector_stores.qdrant import QdrantVectorStore
from qdrant_client import QdrantClient


PDF_FILENAME = "data/long-term-care-2.pdf"
COLLECTION_NAME = "naive_rag_ll_etl_rag"
QUERY_TEXT = "要怎麼申請長照2.0？"
QDRANT_HOST = "localhost"
QDRANT_PORT = 6333
OLLAMA_MODEL = "qwen3:4b"
OLLAMA_BASE_URL = "http://127.0.0.1:11434"
OLLAMA_REQUEST_TIMEOUT = 120.0
EMBEDDING_MODEL_NAME = "BAAI/bge-base-zh-v1.5"
EMBEDDING_CACHE_FOLDER = os.path.expanduser("~/.cache/huggingface/hub")
CHUNK_SIZE = 512
CHUNK_OVERLAP = 128
SIMILARITY_TOP_K = 6
RERANKER_MODEL_NAME = "Qwen/Qwen3-Reranker-0.6B"
RERANKER_TOP_N = SIMILARITY_TOP_K

SYSTEM_PROMPT = """You are an expert Q&A system that is trusted around the world.
Always answer the query using the provided context information, and not prior knowledge.
Some rules to follow:
1. Never directly reference the given context in your answer.
2. Avoid statements like 'Based on the context, ...' or 'The context information ...' or anything along those lines.
3. Detect the language of the query, and always answer in that language
"""

USER_PROMPT = """Context information is below.
---------------------
{context}
---------------------
Given the context information and not prior knowledge, answer the query.
Query: {query}
Answer:
"""


class MetadataExclusionPostProcessor(BaseNodePostprocessor):
    """Exclude specified metadata keys from LLM synthesis."""

    excluding_metadata_keys: list[str] = Field(
        description="Metadata keys to exclude from generating the prompt."
    )

    def __init__(self, excluding_metadata_keys: list[str]) -> None:
        super().__init__(excluding_metadata_keys=excluding_metadata_keys)

    @classmethod
    def class_name(cls) -> str:
        return "MetadataExclusionPostProcessor"

    def _postprocess_nodes(
            self,
            nodes: list[NodeWithScore],
            query_bundle: Optional[QueryBundle] = None,
    ) -> list[NodeWithScore]:
        for node in nodes:
            node.node.excluded_llm_metadata_keys.extend(
                self.excluding_metadata_keys
            )

        return nodes


def create_qdrant_client() -> QdrantClient:
    return QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)


def create_vector_store(
        db_client: QdrantClient,
        collection_name: str,
) -> QdrantVectorStore:
    return QdrantVectorStore(
        collection_name=collection_name,
        client=db_client,
    )


def load_documents() -> list[Document]:
    return SimpleDirectoryReader(
        input_files=[PDF_FILENAME],
    ).load_data()


def create_embedding_model() -> HuggingFaceEmbedding:
    return HuggingFaceEmbedding(
        model_name=EMBEDDING_MODEL_NAME,
        cache_folder=EMBEDDING_CACHE_FOLDER,
    )


def create_ingestion_transforms() -> list[TransformComponent]:
    node_parser = SentenceSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    embed_model = create_embedding_model()

    return [node_parser, embed_model]


def ingest_documents(
        documents: list[Document],
        vector_store: QdrantVectorStore,
        transformations: list[TransformComponent],
) -> None:
    pipeline = IngestionPipeline(
        transformations=transformations,
        vector_store=vector_store,
    )
    pipeline.run(documents=documents)


def create_retriever(vector_store: QdrantVectorStore) -> VectorIndexRetriever:
    index = VectorStoreIndex.from_vector_store(
        vector_store=vector_store,
        embed_model=create_embedding_model(),
    )

    return VectorIndexRetriever(
        index=index,
        similarity_top_k=SIMILARITY_TOP_K,
    )


def print_retrieved_nodes(
        retriever: VectorIndexRetriever,
        query_text: str,
        title: str,
        reranker: BaseNodePostprocessor | None = None,
) -> None:
    nodes = retriever.retrieve(query_text)
    if reranker is not None:
        nodes = reranker.postprocess_nodes(nodes, query_str=query_text)

    print(title)
    for rank, node_with_score in enumerate(nodes, start=1):
        print(f"\nRank: {rank}")
        print(f"Score: {node_with_score.score}")
        print(f"Metadata: {node_with_score.node.metadata}")
        print("Text:")
        print(node_with_score.node.get_content())


def create_prompt_template() -> ChatPromptTemplate:
    system_prompt = ChatMessage(role=MessageRole.SYSTEM, content=SYSTEM_PROMPT)
    user_prompt = ChatMessage(role=MessageRole.USER, content=USER_PROMPT)

    return ChatPromptTemplate(
        message_templates=[system_prompt, user_prompt],
        template_var_mappings={"context_str": "context", "query_str": "query"},
    )


def create_reranker() -> SentenceTransformerRerank:
    return SentenceTransformerRerank(
        model=RERANKER_MODEL_NAME,
        top_n=RERANKER_TOP_N,
    )


def create_llm() -> Ollama:
    return Ollama(
        model=OLLAMA_MODEL,
        base_url=OLLAMA_BASE_URL,
        request_timeout=OLLAMA_REQUEST_TIMEOUT,
    )


def create_query_engine(
        retriever: VectorIndexRetriever,
        prompt_template: ChatPromptTemplate,
        llm: Ollama,
        reranker: BaseNodePostprocessor | None = None,
) -> RetrieverQueryEngine:
    response_synthesizer = get_response_synthesizer(
        llm=llm,
        text_qa_template=prompt_template,
    )
    node_postprocessors: list[BaseNodePostprocessor] = []
    if reranker is not None:
        node_postprocessors.append(reranker)
    node_postprocessors.append(
        MetadataExclusionPostProcessor(["page_label", "file_path"])
    )

    return RetrieverQueryEngine(
        retriever=retriever,
        node_postprocessors=node_postprocessors,
        response_synthesizer=response_synthesizer,
    )


def delete_collection(
        db_client: QdrantClient,
        collection_name: str,
) -> None:
    if db_client.collection_exists(collection_name):
        db_client.delete_collection(collection_name=collection_name)


def run_etl_and_query() -> None:
    db_client = create_qdrant_client()
    vector_store = create_vector_store(db_client, COLLECTION_NAME)

    try:
        documents = load_documents()
        ingest_documents(
            documents=documents,
            vector_store=vector_store,
            transformations=create_ingestion_transforms(),
        )

        retriever = create_retriever(vector_store)
        prompt_template = create_prompt_template()
        llm = create_llm()

        #
        # Answers without reranker
        #
        print_retrieved_nodes(
            retriever=retriever,
            query_text=QUERY_TEXT,
            title="Retrieved nodes without reranker:",
        )

        query_engine_without_reranker = create_query_engine(
            retriever=retriever,
            prompt_template=prompt_template,
            llm=llm,
        )

        print("\n============Without reranker:============")
        response_without_reranker = query_engine_without_reranker.query(QUERY_TEXT)
        

        #
        # Answers with reranker
        #
        reranker = create_reranker()
        print_retrieved_nodes(
            retriever=retriever,
            query_text=QUERY_TEXT,
            title="\nReranked retrieved nodes:",
            reranker=reranker,
        )

        query_engine_with_reranker = create_query_engine(
            retriever=retriever,
            prompt_template=prompt_template,
            llm=llm,
            reranker=reranker,
        )

        print("\n============With reranker:============")
        response_with_reranker = query_engine_with_reranker.query(QUERY_TEXT)


        print("\n============Answer without reranker:============")
        print(response_without_reranker)
        print("\n============Answer with reranker:============")
        print(response_with_reranker)
    finally:
        delete_collection(db_client, COLLECTION_NAME)


if __name__ == "__main__":
    run_etl_and_query()

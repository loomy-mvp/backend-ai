import logging
from typing import Optional

from backend.services.retrieve import Retriever, RetrieveRequest
from backend.services.reranker import rerank
from backend.utils.ai_workflow_utils.get_config_value import get_config_value
from backend.config.chatbot_config import RETRIEVAL_PIPELINE_CONFIG

logger = logging.getLogger(__name__)

retriever = Retriever()


def retrieve_and_rerank(
    query: str,
    index_name: str,
    namespace: str,
    libraries: list[str],
    sources: Optional[list[str]] = None,
) -> list[dict]:
    """Full retrieve → BM25 → reranking model pipeline.

    Config values (all from ``RETRIEVAL_PIPELINE_CONFIG``):
        - ``retrieval_top_k`` / ``retrieval_similarity_threshold``: initial vector search params
        - ``bm25_top_k``: how many docs BM25 keeps
        - ``reranker_top_k``: final count after the reranking model
        - ``use_reranker``: whether the reranking-model step runs

    Returns:
        A list of document dicts ready for the LLM context.
    """
    top_k = get_config_value(config_set=RETRIEVAL_PIPELINE_CONFIG, key="retrieval_top_k")
    similarity_threshold = get_config_value(config_set=RETRIEVAL_PIPELINE_CONFIG, key="retrieval_similarity_threshold")

    # --- Step 1: Vector retrieval ---
    logger.info(
        "[pipeline] Retrieving up to %d chunks (threshold=%.2f)",
        top_k,
        similarity_threshold,
    )
    retrieve_request = RetrieveRequest(
        query=query,
        index_name=index_name,
        namespace=namespace,
        libraries=libraries,
        top_k=top_k,
        similarity_threshold=similarity_threshold,
        sources=sources,
    )
    retrieval = retriever.retrieve(retrieve_request)
    docs = retrieval.get("results", [])
    logger.info("[pipeline] Retrieved %d chunks from vector store", len(docs))

    if not docs:
        return []

    # --- Steps 2 & 3: BM25 → Reranking model ---
    docs = rerank(query=query, documents=docs)
    logger.info("[pipeline] Final document count after reranking: %d", len(docs))

    return docs

import logging

from backend.utils.ai_workflow_utils.bm25 import BM25
from backend.utils.ai_workflow_utils.reranking_model import RerankingModel
from backend.utils.ai_workflow_utils.get_config_value import get_config_value
from backend.config.chatbot_config import RETRIEVAL_PIPELINE_CONFIG, BM25_CONFIG

logger = logging.getLogger(__name__)

# Initialize BM25 with configurable k1 and b parameters
bm25_k1 = get_config_value(config_set=BM25_CONFIG, key="k1")
bm25_b = get_config_value(config_set=BM25_CONFIG, key="b")
bm25 = BM25(k1=bm25_k1, b=bm25_b)

reranker_provider = get_config_value(config_set=RETRIEVAL_PIPELINE_CONFIG, key="reranker_provider")
reranker_model = get_config_value(config_set=RETRIEVAL_PIPELINE_CONFIG, key="reranker_model")
reranking_model = RerankingModel(provider=reranker_provider, model=reranker_model)


def _deduplicate_documents(documents: list[dict], text_key: str = "chunk_text") -> list[dict]:
    """Remove documents with duplicate chunk_text content.

    Returns the first occurrence of each unique text, preserving order.
    """
    seen_texts = set()
    deduplicated = []

    for doc in documents:
        text = doc.get(text_key, "")
        if text and (text not in seen_texts):
            seen_texts.add(text)
            deduplicated.append(doc)
        elif text:
            logger.debug("[dedup] Skipping duplicate chunk text (length=%d)", len(text))

    if len(deduplicated) < len(documents):
        logger.info("[dedup] Removed %d duplicate chunks from %d", len(documents) - len(deduplicated), len(documents))

    return deduplicated


def rerank(
    query: str,
    documents: list[dict],
    *,
    use_reranker: bool | None = None,
) -> list[dict]:
    """Run the 3-step reranking pipeline: dedup → BM25 → reranking model.

    Args:
        query: The user's search query.
        documents: Candidate documents returned by the retriever.
            Each dict must contain a ``chunk_text`` key.
        use_reranker: Override the config flag for the reranking-model step.
            When *None* (default), the ``USE_RERANKER`` config value is used.

    Returns:
        The top-ranked documents after the full pipeline.
    """
    if not documents:
        return []

    # Step 0 — Deduplication
    logger.info("[rerank] Step 0: Deduplication — removing exact duplicates from %d docs", len(documents))
    documents = _deduplicate_documents(documents)

    if not documents:
        logger.warning("[rerank] No documents remain after deduplication")
        return []

    bm25_top_k = get_config_value(config_set=RETRIEVAL_PIPELINE_CONFIG, key="bm25_top_k")
    reranker_top_k = get_config_value(config_set=RETRIEVAL_PIPELINE_CONFIG, key="reranker_top_k")

    # Step 1 — BM25
    logger.info("[rerank] Step 1: BM25 — reducing %d docs to %d", len(documents), bm25_top_k)
    documents = bm25.rank(query=query, documents=documents, top_k=bm25_top_k)

    # Step 2 — Reranking model (optional)
    should_rerank = use_reranker if use_reranker is not None else get_config_value(
        config_set=RETRIEVAL_PIPELINE_CONFIG, key="use_reranker"
    )

    if should_rerank:
        logger.info(
            "[rerank] Step 2: Reranking model (%s/%s) — reducing %d docs to %d",
            reranker_provider,
            reranker_model,
            len(documents),
            reranker_top_k,
        )
        documents = reranking_model.rerank(query=query, documents=documents, top_k=reranker_top_k)
    else:
        logger.info("[rerank] Step 2 skipped (use_reranker=False); trimming to %d", reranker_top_k)
        documents = documents[:reranker_top_k]

    return documents

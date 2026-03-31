import logging
import os

import cohere
import httpx
from dotenv import load_dotenv

load_dotenv(override=True)

logger = logging.getLogger(__name__)

cohere_api_key = os.getenv("COHERE_API_KEY")
langsearch_api_key = os.getenv("LANGSEARCH_API_KEY")


class RerankingModel:
    """Multi-provider semantic reranking model supporting Cohere and Langsearch."""

    def __init__(self, provider: str = "langsearch", model: str = "langsearch-reranker-v1"):
        """Initialize the reranking model.

        Args:
            provider: Either "langsearch" or "cohere"
            model: Model identifier (e.g., "langsearch-reranker-v1" or "rerank-v3.5")
        """
        if provider not in ("langsearch", "cohere"):
            raise ValueError(f"Unknown reranker provider: {provider}")

        self.provider = provider
        self.model = model

        if provider == "cohere":
            self.client = cohere.ClientV2(cohere_api_key)
        elif provider == "langsearch":
            self.langsearch_url = "https://api.langsearch.com/v1/rerank"

    def rerank(
        self,
        query: str,
        documents: list[dict],
        top_k: int,
        text_key: str = "chunk_text",
    ) -> list[dict]:
        """Rerank *documents* and return the top-k.

        Each document dict must contain a field identified by *text_key*.
        The returned list preserves the original dicts (no mutation).
        """
        if not documents:
            return []

        if self.provider == "cohere":
            return self._rerank_cohere(query, documents, top_k, text_key)
        elif self.provider == "langsearch":
            return self._rerank_langsearch(query, documents, top_k, text_key)

    def _rerank_cohere(
        self,
        query: str,
        documents: list[dict],
        top_k: int,
        text_key: str,
    ) -> list[dict]:
        """Rerank using Cohere's API."""
        texts = [doc.get(text_key, "") for doc in documents]

        response = self.client.rerank(
            model=self.model,
            query=query,
            documents=texts,
            top_n=top_k,
        )

        reranked: list[dict] = []
        for result in response.results:
            doc = documents[result.index]
            reranked.append(doc)

        logger.info(
            "[RerankingModel] Cohere reranked %d documents to %d",
            len(documents),
            len(reranked),
        )
        return reranked

    def _rerank_langsearch(
        self,
        query: str,
        documents: list[dict],
        top_k: int,
        text_key: str,
    ) -> list[dict]:
        """Rerank using Langsearch's API."""
        if not langsearch_api_key:
            raise ValueError("LANGSEARCH_API_KEY environment variable is not set")

        texts = [doc.get(text_key, "") for doc in documents]

        payload = {
            "model": self.model,
            "query": query,
            "top_n": top_k,
            "return_documents": True,
            "documents": texts,
        }

        headers = {
            "Authorization": f"Bearer {langsearch_api_key}",
            "Content-Type": "application/json",
        }

        try:
            response = httpx.post(
                self.langsearch_url,
                json=payload,
                headers=headers,
                timeout=30.0,
            )
            response.raise_for_status()
        except httpx.HTTPError as e:
            logger.error("[RerankingModel] Langsearch HTTP error: %s", e)
            raise

        data = response.json()

        if data.get("code") != 200:
            raise ValueError(f"Langsearch API error: {data.get('msg', 'Unknown error')}")

        reranked: list[dict] = []
        for result in data.get("results", []):
            original_index = result["index"]
            if 0 <= original_index < len(documents):
                doc = documents[original_index]
                reranked.append(doc)

        logger.info(
            "[RerankingModel] Langsearch reranked %d documents to %d",
            len(documents),
            len(reranked),
        )
        return reranked

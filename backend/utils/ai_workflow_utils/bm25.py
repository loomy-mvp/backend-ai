import logging
import re

from rank_bm25 import BM25Okapi

logger = logging.getLogger(__name__)


def _tokenize(text: str) -> list[str]:
    """Lowercase and split on non-alphanumeric characters.
    
    Args:
        text: Text to tokenize.
        
    Returns:
        List of lowercased alphanumeric tokens.
    """
    return re.findall(r"[a-z0-9]+", text.lower())


class BM25:
    """BM25 scorer using the rank-bm25 library.

    BM25 (Best Matching 25) is a probabilistic ranking function that scores documents
    based on query term frequency, inverse document frequency, and document length.
    This implementation ranks a candidate set (≤ a few hundred docs) returned by a 
    vector store, providing complementary lexical matching to semantic similarity.

    Parameters:
        k1 (float): Term frequency saturation parameter. Controls how much additional 
            term frequency beyond the first occurrence contributes to the score.
            - Range: typically 1.0–2.0
            - Low values (< 1.0): diminishing returns on term frequency; emphasizes presence over frequency
            - High values (> 2.0): term frequency gains more weight; favors documents with many query matches
            - Default 1.5: balanced middle ground
            
        b (float): Document length normalization parameter. Controls how much document 
            length affects the score.
            - Range: 0.0–1.0
            - 0.0: no document length normalization; score is independent of document length
            - 1.0: full document length normalization; longer documents penalized proportionally
            - Default 0.75: slight bias toward shorter documents (reduces "verbose" document advantage)
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        """Initialize BM25 scorer with tuning parameters.
        
        Args:
            k1: Term frequency saturation. Higher values increase sensitivity to term frequency.
            b: Document length normalization. Controls penalty for longer documents.
        """
        self.k1 = k1
        self.b = b

    def rank(
        self,
        query: str,
        documents: list[dict],
        top_k: int,
        text_key: str = "chunk_text",
    ) -> list[dict]:
        """Score documents against query using BM25 and return top-k.
        
        BM25 scoring provides lexical relevance ranking, making it effective for:
        - Exact term matching (complements semantic vector search)
        - Keyword-heavy queries
        - Short document sets (reranking pre-filtered results)
        
        Args:
            query: Query string (will be tokenized).
            documents: List of document dicts to score. Each must contain text_key field.
            top_k: Number of top-scoring documents to return.
            text_key: Dict key containing the text to score (default: "chunk_text").
        
        Returns:
            List of top-k documents sorted by BM25 score (descending).
            Original document dicts are preserved unchanged.
        """
        if not documents:
            return []

        query_tokens = _tokenize(query)
        if not query_tokens:
            logger.warning("[BM25] Query produced no tokens; returning documents unchanged")
            return documents[:top_k]

        # Tokenize all documents
        doc_tokens = [_tokenize(doc.get(text_key, "")) for doc in documents]

        # Initialize BM25Okapi with tokenized documents
        bm25 = BM25Okapi(doc_tokens, k1=self.k1, b=self.b)

        # Score all documents against the query
        scores = bm25.get_scores(query_tokens)

        # Get top-k indices sorted by score (descending)
        ranked_indices = sorted(range(len(documents)), key=lambda i: scores[i], reverse=True)
        top_indices = ranked_indices[:top_k]

        logger.info("[BM25] Reduced %d documents to %d", len(documents), len(top_indices))
        return [documents[i] for i in top_indices]

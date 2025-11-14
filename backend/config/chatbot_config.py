# Configuration for chatbot API

CHATBOT_CONFIG = {
    # Explicit values (used if set, otherwise fallback to defaults below)
    "model": 'gpt-5-2025-08-07',
    "provider": 'openai', # e.g., "openai", "anthropic", "google"
    "max_tokens": 20000,
    "top_k": 5  # Number of relevant documents to retrieve from each kb
}

SIMILARITY_THRESHOLD = 0.5  # Minimum similarity score for retrieved documents
CHATBOT_CONFIG["temperature"] = 0.0 if not "gpt-5" in CHATBOT_CONFIG['model'] else 1

EMBEDDING_CONFIG = {
    "model": 'embed-v4.0',  # e.g., "text-embedding-3-small"
    "provider": 'cohere',  # e.g., "openai"
}

RETRIEVAL_JUDGE_CONFIG = {
    "model": 'gpt-5-nano-2025-08-07',
    "provider": 'openai',
    "max_tokens": 1000
}
RETRIEVAL_JUDGE_CONFIG["temperature"] = 0.0 if not "gpt-5" in RETRIEVAL_JUDGE_CONFIG['model'] else 1
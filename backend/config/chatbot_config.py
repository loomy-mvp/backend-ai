CHATBOT_CONFIG = {
    "model": "gemini-3-pro-preview", # 'claude-sonnet-4-5-20250929', # 'gemini-3-pro-preview', # 'gpt-5-2025-08-07', 'gpt-5.1-2025-11-13
    "provider": 'google', # "openai", "anthropic", "google"
    "max_tokens": 2000,
    "top_k": 5  # Number of relevant documents to retrieve from each kb
}

# Provider-specific extra kwargs to enable thinking / reasoning features
PROVIDER_THINKING_KWARGS = {
    "anthropic": {
        "thinking": {
            "type": "enabled",
            "budget_tokens": 2000,
        }
    },
    "openai": {
        "reasoning": {"effort": "medium"},
        "text": {"verbosity": "low"},
    },
}

SIMILARITY_THRESHOLD = 0.25  # Minimum similarity score for retrieved documents
CHATBOT_CONFIG["temperature"] = 0.0 if not "gpt-5" in CHATBOT_CONFIG['model'] else 1

EMBEDDING_CONFIG = {
    "model": 'embed-v4.0',  # e.g., "text-embedding-3-small"
    "provider": 'cohere',  # e.g., "openai"
}

RETRIEVAL_JUDGE_CONFIG = {
    "model": 'gemini-2.5-flash-lite', # gpt-5-nano-2025-08-07, claude-haiku-4-5, gemini-2.5-flash-lite
    "provider": 'google',
    "max_tokens": 1000
}
RETRIEVAL_JUDGE_CONFIG["temperature"] = 0.0 if not "gpt-5" in RETRIEVAL_JUDGE_CONFIG['model'] else 1
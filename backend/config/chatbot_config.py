######## Model configuration ########
CHATBOT_CONFIG = {
    "model": "gpt-5.1-2025-11-13", # 'claude-sonnet-4-5-20250929', # 'gemini-3-pro-preview', # 'gpt-5-2025-08-07', 'gpt-5.1-2025-11-13'
    "provider": 'openai', # "openai", "anthropic", "google"
    "max_tokens": 20000,
    "top_k": 5  # Number of relevant documents to retrieve from each kb
}
CHATBOT_CONFIG["temperature"] = 0.0 if not "gpt-5" in CHATBOT_CONFIG['model'] else 1

######## Tools and reasoning configuration ########
PROVIDER_THINKING_KWARGS = {
    "anthropic": {
        "thinking": {
            "type": "enabled",
            "budget_tokens": 2000,
        }
    },
    "openai": {
        "reasoning": {"effort": "none"},
        "text": {"verbosity": "low"},
    },
}

WEB_SEARCH_KWARG = {
    "openai": {
        "tools": [
            {
                "type": "web_search",
                # "filters": {
                #     "allowed_domains": [
                #         "ntplusfisco.ilsole24ore.com",
                #     ]
                # }
            }
        ],
        "tool_choice": "auto",
        "include": ["web_search_call.action.sources"],
    },
}

######## Retrieval Pipeline Configuration ########
SIMILARITY_THRESHOLD = 0.3  # Minimum similarity score for initial retrieval

EMBEDDING_CONFIG = {
    "model": 'embed-v4.0',  # e.g., "text-embedding-3-small"
    "provider": 'cohere',  # e.g., "openai"
}

RETRIEVAL_PIPELINE_CONFIG = {
    "retrieval_top_k": 100,          # Initial retrieval: how many chunks to fetch from vector store
    "retrieval_similarity_threshold": 0.3,  # Minimum similarity for initial retrieval
    "bm25_top_k": 25,                # BM25 reranking: reduce to this many chunks
    "reranker_top_k": 5,             # Reranking model: final number of chunks to return
    "reranker_provider": "langsearch",  # "langsearch" or "cohere"
    "reranker_model": "langsearch-reranker-v1",  # Model identifier for the selected provider
    "use_reranker": True,            # Set to False to skip the reranking model step
}

# BM25 parameters for lexical reranking of retrieved documents
BM25_CONFIG = {
    "k1": 1.5,
    "b": 0.75,
}

RETRIEVAL_JUDGE_CONFIG = {
    "model": 'gemini-2.5-flash-lite', # gpt-5-nano-2025-08-07, claude-haiku-4-5, gemini-2.5-flash-lite
    "provider": 'google',
    "max_tokens": 100
}
RETRIEVAL_JUDGE_CONFIG["temperature"] = 0.0 if not "gpt-5" in RETRIEVAL_JUDGE_CONFIG['model'] else 1

######## Vision Configuration ########
VISION_CONFIG = {
    "model": 'gemini-2.5-flash-lite',
    "provider": 'google',
    "max_tokens": 1000
}
VISION_CONFIG["temperature"] = 0.0

######## Writer configuration ########
WRITER_CONFIG = {
    "model": 'gpt-5.1-2025-11-13',
    "provider": 'openai',
    "max_tokens": 20000
}
WRITER_CONFIG["temperature"] = 0.0 if not "gpt-5" in WRITER_CONFIG['model'] else 1

# Writer-specific reasoning/verbosity settings
WRITER_PROVIDER_THINKING_KWARGS = {
    "openai": {
        "reasoning": {"effort": "high"},
        "text": {"verbosity": "high"},
    },
}
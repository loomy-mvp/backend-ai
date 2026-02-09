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

######## RAG Configuration ########
SIMILARITY_THRESHOLD = 0.4  # Minimum similarity score for retrieved documents

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
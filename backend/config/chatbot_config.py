# Configuration for chatbot API

CHATBOT_CONFIG = {
    # Explicit values (used if set, otherwise fallback to defaults below)
    "model": 'claude-sonnet-4-20250514',  # e.g., "gpt-5-nano"
    "provider": 'anthropic',  # e.g., "openai"
    "max_tokens": 20000,  # e.g., 1000
    "temperature": 0.0,
    "top_k": 5,  # Number of relevant documents to retrieve


    # Defaults (used if above are not set)
    "default_model": "gpt-5-nano",
    "default_provider": "openai",
    "default_max_tokens": 1000,
    "default_temperature": 0.0,
    "default_top_k": 5,
} 
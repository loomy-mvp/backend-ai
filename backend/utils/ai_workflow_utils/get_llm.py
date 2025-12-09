from backend.config.chatbot_config import CHATBOT_CONFIG
from backend.utils.ai_workflow_utils.get_config_value import get_config_value
from enum import Enum

# Enums for model providers
class ModelProvider(str, Enum):
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GOOGLE = "google"
    
# LLM initialization functions
def get_llm(provider: ModelProvider = None, model: str = None, temperature: float = 0, max_tokens: int = None):
    """Initialize LLM based on provider and model using config, with fallback to defaults."""
    import importlib
    
    # Mapping of providers to their LangChain modules and classes
    provider_mapping = {
        ModelProvider.OPENAI: ("langchain_openai", "ChatOpenAI", "max_tokens"),
        ModelProvider.ANTHROPIC: ("langchain_anthropic", "ChatAnthropic", "max_tokens"),
        ModelProvider.GOOGLE: ("langchain_google_genai", "ChatGoogleGenerativeAI", "max_tokens"),
        # Add more providers as needed
    }
    # Provider-specific logging opt-out parameters to keep requests private by default
    privacy_overrides = {
        ModelProvider.OPENAI: {"default_headers": {"OpenAI-Data-Opt-Out": "true"}},
        ModelProvider.ANTHROPIC: {"metadata": {"anthropic-beta": "do-not-log"}},
        # ModelProvider.GOOGLE: not_available
    }
    
    if provider is None:
        provider = get_config_value(config_set=CHATBOT_CONFIG, key="provider")
    if model is None:
        model = get_config_value(config_set=CHATBOT_CONFIG, key="model")
    if max_tokens is None:
        max_tokens = get_config_value(config_set=CHATBOT_CONFIG, key="max_tokens")

    if isinstance(provider, str):
        try:
            provider = ModelProvider(provider)
        except ValueError as exc:
            raise ValueError(f"Unsupported provider: {provider}") from exc

    if provider not in provider_mapping:
        raise ValueError(f"Unsupported provider: {provider}")
    
    module_name, class_name, max_tokens_param = provider_mapping[provider]
    
    try:
        # Dynamically import the module and get the class
        module = importlib.import_module(module_name)
        llm_class = getattr(module, class_name)
        
        # Prepare kwargs with the correct parameter name for max_tokens
        kwargs = {
            "model": model,
            "temperature": temperature,
            max_tokens_param: max_tokens
        }
        opt_out_kwargs = privacy_overrides.get(provider)
        if opt_out_kwargs:
            kwargs.update(opt_out_kwargs)
        
        return llm_class(**kwargs)
        
    except ImportError as e:
        raise ValueError(f"Failed to import {module_name}: {str(e)}. Make sure the package is installed.")
    except Exception as e:
        raise ValueError(f"Failed to initialize LLM for provider {provider} with model {model}: {str(e)}")
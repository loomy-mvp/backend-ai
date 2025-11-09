from backend.config.chatbot_config import CHATBOT_CONFIG
from backend.utils.get_config_value import get_config_value
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
        ModelProvider.GOOGLE: ("langchain_google_vertexai", "ChatVertexAI", "max_output_tokens"),
        # Add more providers as needed
    }
    
    if provider is None:
        provider = get_config_value(config_set=CHATBOT_CONFIG, key="provider")
    if model is None:
        model = get_config_value(config_set=CHATBOT_CONFIG, key="model")
    if max_tokens is None:
        max_tokens = get_config_value(config_set=CHATBOT_CONFIG, key="max_tokens")

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
        
        return llm_class(**kwargs)
        
    except ImportError as e:
        raise ValueError(f"Failed to import {module_name}: {str(e)}. Make sure the package is installed.")
    except Exception as e:
        raise ValueError(f"Failed to initialize LLM for provider {provider} with model {model}: {str(e)}")
"""
OpenRouter configuration for DSPy models
Handles setup for models accessed via OpenRouter API
"""

import os
import dspy

# Default model configuration
DEFAULT_MODEL = "openai/gpt-5"

def validate_model_name(model_name: str) -> bool:
    """
    Validate that the model name is in the correct format and supported
    
    Args:
        model_name: The model name to validate
        
    Returns:
        bool: True if model is valid, False otherwise
    """
    # List of supported providers and their models
    SUPPORTED_MODELS = {
        "anthropic": ["claude-3-opus", "claude-3-sonnet", "claude-2"],
        "meta-llama": ["llama-3.1-70b", "llama-2-70b"],
        "mistral": ["mistral-large", "mistral-medium"],
        "google": ["gemini-pro", "palm-2"],
        "openai": ["gpt-5"],  # Added GPT-5 support
        "moonshotai": ["kimi-k2"]  # Added Kimi K2 support
    }
    
    try:
        provider, model = model_name.split("/")
        if provider not in SUPPORTED_MODELS:
            print(f"[OpenRouter] Unsupported provider: {provider}")
            return False
        if model not in SUPPORTED_MODELS[provider]:
            print(f"[OpenRouter] Unsupported model: {model} for provider {provider}")
            return False
        return True
    except ValueError:
        print(f"[OpenRouter] Invalid model name format: {model_name}")
        return False

def setup_openrouter_model(model_name: str = DEFAULT_MODEL):
    """
    Setup DSPy to use a model via OpenRouter
    
    Args:
        model_name: The model name in format "provider/model"
                   e.g., "openai/gpt-5", "kimi/k2", "anthropic/claude-3-opus"
                   
    Raises:
        ValueError: If model name is invalid or unsupported
    """
    # Validate model name
    if not validate_model_name(model_name):
        raise ValueError(f"Invalid or unsupported model: {model_name}")
    api_key = os.getenv('OPENROUTER_API_KEY')
    if not api_key:
        raise ValueError("OPENROUTER_API_KEY environment variable not set")
    
    # OpenRouter base URL
    base_url = "https://openrouter.ai/api/v1"
    
    # Configure the model with OpenRouter endpoint
    # Note: For OpenRouter, we need to use the OpenAI-compatible interface
    # but point it to OpenRouter's base URL
    
    # Create LM instance with OpenRouter configuration
    lm = dspy.LM(
        model=model_name,
        api_key=api_key,
        api_base=base_url,
        max_tokens=20000,  # Required for OpenAI's reasoning models
        temperature=1.0    # Required for OpenAI's reasoning models
    )
    
    print(f"[OpenRouter] Configured DSPy with model: {model_name}")
    print(f"[OpenRouter] Using base URL: {base_url}")
    
    return lm

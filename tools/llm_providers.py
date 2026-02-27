"""LLM Provider Factory - Unified interface for multiple LLM providers.

This module provides a factory function to create LLM clients based on the
configured provider. Supports:
- Anthropic API (direct)
- AWS Bedrock (Claude models)
- Cerebras z.ai (GLM 4.7 - ultra-fast inference)

Example usage:
    from tools.llm_providers import get_llm_client
    from config.settings import get_settings

    settings = get_settings()
    llm = get_llm_client(settings)

    # Use with LangChain
    response = await llm.ainvoke("Hello, world!")
"""

import json
import logging
import os
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from langchain_core.language_models.chat_models import BaseChatModel

from config.settings import Settings

logger = logging.getLogger(__name__)


def get_llm_client(
    settings: Settings,
    temperature: float = 0.1,
    max_tokens: int = 4096,
) -> "BaseChatModel":
    """Create an LLM client based on the configured provider.

    Args:
        settings: Application settings with LLM configuration
        temperature: Sampling temperature (0.0-1.0). Lower = more deterministic.
        max_tokens: Maximum tokens in the response.

    Returns:
        A LangChain chat model instance

    Raises:
        ValueError: If the configured provider is not supported or credentials are missing
    """
    provider = settings.llm_provider

    if provider == "anthropic":
        return _create_anthropic_client(settings, temperature, max_tokens)
    elif provider == "bedrock":
        return _create_bedrock_client(settings, temperature, max_tokens)
    elif provider == "cerebras":
        return _create_cerebras_client(settings, temperature, max_tokens)
    else:
        raise ValueError(f"Unsupported LLM provider: {provider}")


def _create_anthropic_client(
    settings: Settings,
    temperature: float,
    max_tokens: int,
) -> "BaseChatModel":
    """Create a direct Anthropic API client."""
    from langchain_anthropic import ChatAnthropic

    if not settings.anthropic_api_key:
        raise ValueError(
            "ANTHROPIC_API_KEY is required when using anthropic provider. "
            "Set it in your .env file or environment."
        )

    # Extract model name from bedrock model ID if it looks like a bedrock ID
    model = settings.bedrock_model_id
    if "anthropic" in model and ":" in model:
        parts = model.split(".")
        if len(parts) >= 3:
            model = parts[2].split(":")[0]
            model = model.rsplit("-v", 1)[0]

    logger.info(f"Creating Anthropic client: model={model}")

    return ChatAnthropic(
        api_key=settings.anthropic_api_key,
        model_name=model,
        temperature=temperature,
        max_tokens=max_tokens,
    )


def _create_bedrock_client(
    settings: Settings,
    temperature: float,
    max_tokens: int,
) -> "BaseChatModel":
    """Create an AWS Bedrock LLM client.

    Uses IAM credentials from the environment or instance profile.
    """
    from langchain_aws import ChatBedrock

    logger.info(f"Creating Bedrock client: model={settings.bedrock_model_id}, region={settings.bedrock_region}")

    return ChatBedrock(
        model_id=settings.bedrock_model_id,
        region_name=settings.bedrock_region,
        model_kwargs={
            "temperature": temperature,
            "max_tokens": max_tokens,
        },
    )


def _create_cerebras_client(
    settings: Settings,
    temperature: float,
    max_tokens: int,
) -> "BaseChatModel":
    """Create a Cerebras z.ai LLM client.

    Cerebras provides OpenAI-compatible API for ultra-fast inference.
    The GLM 4.7 model runs at ~1000 tokens/second on Cerebras hardware.

    API key loading priority:
    1. CEREBRAS_API_KEY environment variable
    2. cerebras_api_key in settings
    """
    from langchain_openai import ChatOpenAI

    api_key = None

    # 1. Check environment variable first
    api_key = os.environ.get("CEREBRAS_API_KEY")
    if api_key:
        logger.info("Using Cerebras API key from CEREBRAS_API_KEY env var")

    # 2. Check settings
    if not api_key and settings.cerebras_api_key:
        api_key = settings.cerebras_api_key
        logger.info("Using Cerebras API key from settings")

    if not api_key:
        raise ValueError(
            "CEREBRAS_API_KEY is required when using cerebras provider. "
            "Set it via:\n"
            "  1. CEREBRAS_API_KEY environment variable\n"
            "  2. cerebras_api_key in settings/.env\n"
            "Get your API key from https://cloud.cerebras.ai/"
        )

    logger.info(f"Creating Cerebras client: model={settings.cerebras_model}, base_url={settings.cerebras_base_url}")

    return ChatOpenAI(
        api_key=api_key,
        base_url=settings.cerebras_base_url,
        model=settings.cerebras_model,
        temperature=temperature,
        max_tokens=max_tokens,
    )


def get_provider_info(settings: Settings) -> dict:
    """Get information about the currently configured LLM provider."""
    provider = settings.llm_provider

    if provider == "anthropic":
        model = settings.bedrock_model_id
        if "anthropic" in model and ":" in model:
            parts = model.split(".")
            if len(parts) >= 3:
                model = parts[2].split(":")[0].rsplit("-v", 1)[0]
        return {
            "provider": "anthropic",
            "model": model,
            "description": "Anthropic API (direct)",
        }
    elif provider == "bedrock":
        return {
            "provider": "bedrock",
            "model": settings.bedrock_model_id,
            "region": settings.bedrock_region,
            "description": "AWS Bedrock (Claude models)",
        }
    elif provider == "cerebras":
        return {
            "provider": "cerebras",
            "model": settings.cerebras_model,
            "base_url": settings.cerebras_base_url,
            "description": "Cerebras z.ai (GLM 4.7 - ultra-fast inference, ~1000 tok/s)",
        }
    else:
        return {
            "provider": provider,
            "description": "Unknown provider",
        }

"""Shared OpenAI-compatible provider configuration."""

from __future__ import annotations

import os
from typing import Optional

from openai import OpenAI


PROVIDER_BASE_URLS = {
    "openai": "https://api.openai.com/v1",
    "openrouter": "https://openrouter.ai/api/v1",
    "vllm": "http://localhost:8000/v1",
}

PROVIDER_API_KEY_ENVS = {
    "openai": "OPENAI_API_KEY",
    "openrouter": "OPENROUTER_API_KEY",
}


def resolve_api_key_env(provider: str, api_key_env: Optional[str] = None) -> Optional[str]:
    """Return the configured key variable, or the provider-specific default."""
    if provider == "vllm":
        return None
    if api_key_env:
        return api_key_env
    try:
        return PROVIDER_API_KEY_ENVS[provider]
    except KeyError as error:
        raise ValueError(
            f"Unknown provider '{provider}'; configure an API key variable and base URL."
        ) from error


def create_openai_compatible_client(
    provider: str,
    *,
    api_key: Optional[str] = None,
    api_key_env: Optional[str] = None,
    base_url: Optional[str] = None,
    timeout: float = 120.0,
) -> OpenAI:
    """Build an OpenAI SDK client for OpenAI, OpenRouter, or local vLLM."""
    resolved_url = base_url or PROVIDER_BASE_URLS.get(provider)
    if not resolved_url:
        raise ValueError(f"Unknown provider '{provider}'; configure a base URL.")

    resolved_key_env = resolve_api_key_env(provider, api_key_env)
    resolved_key = api_key or (
        os.environ.get(resolved_key_env) if resolved_key_env else "EMPTY"
    )
    if not resolved_key:
        raise ValueError(
            f"{resolved_key_env} is not set; it is required by provider '{provider}'."
        )

    headers = {}
    if provider == "openrouter":
        if referer := os.environ.get("OPENROUTER_HTTP_REFERER"):
            headers["HTTP-Referer"] = referer
        if title := os.environ.get("OPENROUTER_APP_NAME"):
            headers["X-OpenRouter-Title"] = title

    return OpenAI(
        base_url=resolved_url,
        api_key=resolved_key,
        default_headers=headers or None,
        max_retries=3,
        timeout=timeout,
    )

from __future__ import annotations

from langchain_core.language_models.chat_models import BaseChatModel

from app.config import Settings


def get_chat_model(settings: Settings) -> BaseChatModel:
    provider = settings.provider
    if provider == "claude":
        from provider.claude import build_claude_model
        return build_claude_model(settings)
    if provider == "gemini":
        from provider.gemini import build_gemini_model
        return build_gemini_model(settings)
    if provider == "openai":
        from provider.openai import build_openai_model
        return build_openai_model(settings)
    if provider == "openrouter":
        from provider.openrouter import build_openrouter_model
        return build_openrouter_model(settings)
    if provider == "ollama":
        from provider.ollama import build_ollama_model
        return build_ollama_model(settings)
    if provider == "custom":
        from provider.custom import build_custom_model
        return build_custom_model(settings)
    raise ValueError(f"Unsupported provider: {provider}")

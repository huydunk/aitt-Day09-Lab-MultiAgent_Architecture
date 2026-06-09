from __future__ import annotations

from langchain_anthropic import ChatAnthropic

from app.config import Settings


def build_claude_model(settings: Settings) -> ChatAnthropic:
    if not settings.anthropic_api_key:
        raise ValueError("ANTHROPIC_API_KEY is required for provider=claude")
    return ChatAnthropic(
        model=settings.model,
        api_key=settings.anthropic_api_key,
        temperature=settings.temperature,
    )

"""Thin LLM interface so the provider is swappable (Anthropic, or any OpenAI-compatible server)."""
from functools import lru_cache
from typing import Protocol

from ..config import get_settings


class LLM(Protocol):
    def complete(self, system: str, user: str, max_tokens: int = 1024) -> str: ...


class AnthropicLLM:
    def __init__(self, model: str, api_key: str | None):
        import anthropic

        self.client = anthropic.Anthropic(api_key=api_key, timeout=60)  # falls back to ANTHROPIC_API_KEY
        self.model = model

    def complete(self, system: str, user: str, max_tokens: int = 1024) -> str:
        resp = self.client.messages.create(
            model=self.model, max_tokens=max_tokens, system=system, messages=[{"role": "user", "content": user}]
        )
        return "".join(b.text for b in resp.content if b.type == "text")


class OpenAICompatLLM:
    def __init__(self, model: str, base_url: str | None, api_key: str | None):
        from openai import OpenAI

        self.client = OpenAI(base_url=base_url, api_key=api_key or "none", timeout=60)
        self.model = model

    def complete(self, system: str, user: str, max_tokens: int = 1024) -> str:
        resp = self.client.chat.completions.create(
            model=self.model,
            max_tokens=max_tokens,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        )
        return resp.choices[0].message.content or ""


@lru_cache
def get_llm() -> LLM:
    s = get_settings()
    if s.llm_provider == "openai":
        return OpenAICompatLLM(s.llm_model, s.openai_base_url, s.openai_api_key)
    return AnthropicLLM(s.llm_model, s.anthropic_api_key)

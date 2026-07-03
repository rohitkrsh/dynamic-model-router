from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional


@dataclass(frozen=True)
class ProviderAdapter:
    name: str
    cost_usd_per_1k_tokens: float
    latency_ms: int
    description: str = ""


class ProviderRegistry:
    """A tiny registry that exposes provider adapters used by the router."""

    def __init__(self, adapters: Optional[Dict[str, ProviderAdapter]] = None) -> None:
        self._adapters = adapters or {}

    def register(self, adapter: ProviderAdapter) -> None:
        self._adapters[adapter.name] = adapter

    def get(self, provider_name: str) -> Optional[ProviderAdapter]:
        return self._adapters.get(provider_name)

    @classmethod
    def default_registry(cls) -> "ProviderRegistry":
        registry = cls()
        registry.register(ProviderAdapter("gpt-4.1", 0.02, 900, "Code-focused provider"))
        registry.register(ProviderAdapter("gpt-4.1-mini", 0.008, 700, "General fallback provider"))
        registry.register(ProviderAdapter("claude-3.5-sonnet", 0.015, 1100, "Reasoning and creative provider"))
        registry.register(ProviderAdapter("llama-3.1-8b", 0.004, 800, "Low-cost summarization provider"))
        registry.register(ProviderAdapter("finance-model", 0.012, 950, "Custom finance example provider"))
        registry.register(ProviderAdapter("fallback-model", 0.01, 750, "Custom fallback provider"))
        return registry

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import List


@dataclass(frozen=True)
class RouterDecision:
    selected_provider: str
    category: str
    reason: str
    confidence: float


class Router:
    """A lightweight router for directing prompts to suitable model families."""

    def __init__(self) -> None:
        self._provider_profiles = {
            "gpt-4.1": {
                "category": "code",
                "keywords": ["debug", "code", "python", "javascript", "typescript", "function", "bug", "error", "refactor"],
            },
            "claude-3.5-sonnet": {
                "category": "reasoning",
                "keywords": ["reason", "plan", "solve", "analyze", "step", "multi-step", "strategy", "design"],
            },
            "claude-3.5-sonnet-creative": {
                "category": "creative",
                "keywords": ["poem", "story", "write", "creative", "essay", "novel", "song"],
            },
            "llama-3.1-8b": {
                "category": "summarization",
                "keywords": ["summarize", "summary", "bullet points", "transcript", "meeting", "notes", "brief"],
            },
        }

    def route(self, request: str) -> RouterDecision:
        normalized = request.lower()

        if self._matches_keywords(normalized, ["poem", "story", "write", "creative", "essay", "novel", "song"]):
            return RouterDecision(
                selected_provider="claude-3.5-sonnet",
                category="creative",
                reason="Creative signals matched, so the request is routed to a creative-capable model.",
                confidence=0.93,
            )

        if self._matches_keywords(normalized, ["debug", "code", "python", "javascript", "typescript", "function", "bug", "error", "refactor"]):
            return RouterDecision(
                selected_provider="gpt-4.1",
                category="code",
                reason="Code signals matched, so the request is routed to a code-focused model.",
                confidence=0.95,
            )

        if self._matches_keywords(normalized, ["reason", "plan", "solve", "analyze", "step", "multi-step", "strategy", "design"]):
            return RouterDecision(
                selected_provider="claude-3.5-sonnet",
                category="reasoning",
                reason="Reasoning signals matched, so the request is routed to a reasoning-focused model.",
                confidence=0.92,
            )

        if self._matches_keywords(normalized, ["summarize", "summary", "bullet points", "transcript", "meeting", "notes", "brief"]):
            return RouterDecision(
                selected_provider="llama-3.1-8b",
                category="summarization",
                reason="Summarization signals matched, so the request is routed to a cheaper summarization model.",
                confidence=0.9,
            )

        return RouterDecision(
            selected_provider="gpt-4.1-mini",
            category="general",
            reason="No strong task signals were detected, so the request uses the general-purpose default model.",
            confidence=0.7,
        )

    def _matches_keywords(self, text: str, keywords: List[str]) -> bool:
        return any(keyword in text for keyword in keywords)

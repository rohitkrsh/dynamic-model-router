from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import math
from typing import Any, Dict, List, Optional


@dataclass(frozen=True)
class RouterDecision:
    selected_provider: str
    category: str
    reason: str
    confidence: float
    estimated_cost_usd: float
    estimated_latency_ms: int
    routing_method: str = "keyword"


class Router:
    """A lightweight but configurable router for directing prompts to suitable model families."""

    def __init__(self, config: Optional[Dict[str, Any]] = None, event_sink: Optional[List[Dict[str, Any]]] = None) -> None:
        self._config = self._merge_config(config)
        self._event_sink = event_sink

    def route(self, request: str) -> RouterDecision:
        normalized = request.lower()
        keyword_scores: List[tuple[str, str, int]] = []

        for category_name, category_config in self._config.get("categories", {}).items():
            weight = int(category_config.get("weight", 1))
            keywords = category_config.get("keywords", [])
            if self._matches_keywords(normalized, keywords):
                keyword_scores.append((category_name, category_config.get("provider", self._config.get("default_provider", "gpt-4.1-mini")), weight))

        if keyword_scores:
            category_name, provider, weight = max(keyword_scores, key=lambda item: item[2])
            confidence = min(0.99, 0.7 + (weight * 0.08))
            decision = RouterDecision(
                selected_provider=provider,
                category=category_name,
                reason=f"Matched {category_name} signals with confidence {confidence:.2f}.",
                confidence=confidence,
                estimated_cost_usd=self._get_cost_estimate(provider),
                estimated_latency_ms=self._get_latency_estimate(provider),
                routing_method="keyword",
            )
        else:
            semantic_decision = self._route_with_semantic_similarity(request)
            if semantic_decision is not None and semantic_decision.confidence >= 0.65:
                decision = semantic_decision
            else:
                decision = RouterDecision(
                    selected_provider=self._config.get("default_provider", "gpt-4.1-mini"),
                    category=self._config.get("default_category", "general"),
                    reason="No strong task signals were detected, so the request uses the general-purpose default model.",
                    confidence=0.65,
                    estimated_cost_usd=self._get_cost_estimate(self._config.get("default_provider", "gpt-4.1-mini")),
                    estimated_latency_ms=self._get_latency_estimate(self._config.get("default_provider", "gpt-4.1-mini")),
                    routing_method="fallback",
                )

        self._emit_event(request, decision)
        return decision

    def _matches_keywords(self, text: str, keywords: List[str]) -> bool:
        return any(keyword in text for keyword in keywords)

    def _route_with_semantic_similarity(self, request: str) -> Optional[RouterDecision]:
        examples = self._config.get("semantic_examples", {})
        if not examples:
            return None

        vocabulary = set()
        for example_texts in examples.values():
            for example in example_texts:
                vocabulary.update(self._tokenize(example))
        vocabulary.update(self._tokenize(request))
        vocabulary_list = sorted(vocabulary)

        query_vector = self._embed_text(request, vocabulary_list)
        best_category = None
        best_score = -1.0
        for category_name, example_texts in examples.items():
            category_vectors = [self._embed_text(example, vocabulary_list) for example in example_texts]
            if not category_vectors:
                continue
            similarity = float(sum(self._cosine_similarity(query_vector, vector) for vector in category_vectors) / len(category_vectors))
            if similarity > best_score:
                best_score = similarity
                best_category = category_name

        if best_category is None:
            return None

        category_config = self._config.get("categories", {}).get(best_category, {})
        provider = category_config.get("provider", self._config.get("default_provider", "gpt-4.1-mini"))
        confidence = min(0.99, 0.6 + (best_score * 0.4))
        return RouterDecision(
            selected_provider=provider,
            category=best_category,
            reason=f"Matched {best_category} semantically with confidence {confidence:.2f}.",
            confidence=confidence,
            estimated_cost_usd=self._get_cost_estimate(provider),
            estimated_latency_ms=self._get_latency_estimate(provider),
            routing_method="semantic",
        )

    def _tokenize(self, text: str) -> List[str]:
        normalized = text.lower().strip()
        if not normalized:
            return []

        stop_words = {"the", "a", "an", "and", "or", "to", "of", "for", "with", "this", "that", "is", "in", "on", "be", "me", "my", "it", "as", "how", "from", "into"}
        return [token for token in normalized.split() if token.isalnum() and token not in stop_words]

    def _embed_text(self, text: str, vocabulary: Optional[List[str]] = None) -> List[float]:
        tokens = self._tokenize(text)
        if not tokens:
            return [0.0] * len(vocabulary or [])

        if vocabulary is None:
            vocabulary = sorted(set(tokens))

        counts = Counter(tokens)
        return [float(counts.get(token, 0)) for token in vocabulary]

    def _cosine_similarity(self, left: List[float], right: List[float]) -> float:
        dot_product = sum(l * r for l, r in zip(left, right))
        left_norm = math.sqrt(sum(value * value for value in left))
        right_norm = math.sqrt(sum(value * value for value in right))
        if left_norm == 0 or right_norm == 0:
            return 0.0
        return dot_product / (left_norm * right_norm)

    def _get_cost_estimate(self, provider: str) -> float:
        provider_costs = self._config.get("provider_metadata", {}).get(provider, {})
        return float(provider_costs.get("cost_usd_per_1k_tokens", 0.01))

    def _get_latency_estimate(self, provider: str) -> int:
        provider_costs = self._config.get("provider_metadata", {}).get(provider, {})
        return int(provider_costs.get("latency_ms", 1200))

    def _emit_event(self, request: str, decision: RouterDecision) -> None:
        if self._event_sink is None:
            return
        self._event_sink.append(
            {
                "request": request,
                "provider": decision.selected_provider,
                "category": decision.category,
                "confidence": decision.confidence,
                "cost_usd": decision.estimated_cost_usd,
                "latency_ms": decision.estimated_latency_ms,
            }
        )

    def _merge_config(self, config: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        default_config = {
            "default_provider": "gpt-4.1-mini",
            "default_category": "general",
            "semantic_examples": {
                "code": ["debug this python function", "fix a failing unit test", "refactor this javascript code"],
                "reasoning": ["solve this multi-step planning problem", "analyze this tradeoff carefully", "design a strategy for the roadmap"],
                "creative": ["write a poem about the ocean", "create a short story about winter", "draft an essay on imagination"],
                "summarization": ["summarize this meeting transcript", "brief me on these notes", "create bullet points from this document"],
            },
            "provider_metadata": {
                "gpt-4.1": {"cost_usd_per_1k_tokens": 0.02, "latency_ms": 900},
                "gpt-4.1-mini": {"cost_usd_per_1k_tokens": 0.008, "latency_ms": 700},
                "claude-3.5-sonnet": {"cost_usd_per_1k_tokens": 0.015, "latency_ms": 1100},
                "llama-3.1-8b": {"cost_usd_per_1k_tokens": 0.004, "latency_ms": 800},
            },
            "categories": {
                "code": {
                    "provider": "gpt-4.1",
                    "keywords": ["debug", "code", "python", "javascript", "typescript", "function", "bug", "error", "refactor"],
                    "weight": 3,
                },
                "reasoning": {
                    "provider": "claude-3.5-sonnet",
                    "keywords": ["reason", "plan", "solve", "analyze", "step", "multi-step", "strategy", "design"],
                    "weight": 3,
                },
                "creative": {
                    "provider": "claude-3.5-sonnet",
                    "keywords": ["poem", "story", "write", "creative", "essay", "novel", "song"],
                    "weight": 3,
                },
                "summarization": {
                    "provider": "llama-3.1-8b",
                    "keywords": ["summarize", "summary", "bullet points", "transcript", "meeting", "notes", "brief"],
                    "weight": 3,
                },
            },
        }

        if not config:
            return default_config

        merged = dict(default_config)
        merged.update(config)
        merged["categories"] = dict(default_config.get("categories", {}))
        for category_name, category_config in config.get("categories", {}).items():
            merged["categories"][category_name] = {**merged["categories"].get(category_name, {}), **category_config}
        merged["semantic_examples"] = dict(default_config.get("semantic_examples", {}))
        for category_name, example_texts in config.get("semantic_examples", {}).items():
            merged["semantic_examples"][category_name] = list(example_texts)
        return merged

import unittest

from dynamic_model_router.router import Router, RouterDecision


class RouterTests(unittest.TestCase):
    def setUp(self) -> None:
        self.router = Router()

    def test_code_requests_route_to_code_model(self) -> None:
        decision = self.router.route("Debug this Python function and explain the bug")

        self.assertEqual(decision.selected_provider, "gpt-4.1")
        self.assertEqual(decision.category, "code")
        self.assertIn("code", decision.reason.lower())

    def test_reasoning_requests_route_to_reasoning_model(self) -> None:
        decision = self.router.route("Solve this multi-step planning problem carefully")

        self.assertEqual(decision.selected_provider, "claude-3.5-sonnet")
        self.assertEqual(decision.category, "reasoning")

    def test_creative_requests_route_to_creative_model(self) -> None:
        decision = self.router.route("Write a poem about a rainy evening in Tokyo")

        self.assertEqual(decision.selected_provider, "claude-3.5-sonnet")
        self.assertEqual(decision.category, "creative")

    def test_summarization_requests_use_cheaper_model(self) -> None:
        decision = self.router.route("Summarize this long meeting transcript in bullet points")

        self.assertEqual(decision.selected_provider, "llama-3.1-8b")
        self.assertEqual(decision.category, "summarization")

    def test_router_returns_fallback_decision_for_unknown_requests(self) -> None:
        decision = self.router.route("Help me with a general question")

        self.assertEqual(decision.selected_provider, "gpt-4.1-mini")
        self.assertEqual(decision.category, "general")
        self.assertIsInstance(decision, RouterDecision)

    def test_router_reports_confidence_for_clear_matches(self) -> None:
        decision = self.router.route("Debug this Python function")

        self.assertGreaterEqual(decision.confidence, 0.75)
        self.assertLessEqual(decision.confidence, 0.99)

    def test_router_accepts_custom_config(self) -> None:
        custom_router = Router(
            config={
                "default_provider": "fallback-model",
                "default_category": "general",
                "categories": {
                    "finance": {
                        "provider": "finance-model",
                        "keywords": ["invoice", "budget", "forecast"],
                        "weight": 2,
                    }
                },
            }
        )

        decision = custom_router.route("Prepare a forecast for next quarter")

        self.assertEqual(decision.selected_provider, "finance-model")
        self.assertEqual(decision.category, "finance")

    def test_router_includes_provider_metadata(self) -> None:
        decision = self.router.route("Debug this Python function")

        self.assertGreater(decision.estimated_cost_usd, 0.0)
        self.assertGreater(decision.estimated_latency_ms, 0)

    def test_semantic_routing_uses_similarity_when_available(self) -> None:
        decision = self.router.route("Explain how to fix a broken test suite")

        self.assertEqual(decision.category, "code")
        self.assertEqual(decision.routing_method, "semantic")

    def test_router_logs_decisions_when_sink_is_provided(self) -> None:
        events = []
        router = Router(event_sink=events)

        router.route("Write a short story")

        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["category"], "creative")
        self.assertEqual(events[0]["provider"], "claude-3.5-sonnet")


if __name__ == "__main__":
    unittest.main()

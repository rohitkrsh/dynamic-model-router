import unittest

from dynamic_model_router.router import Router, RouterDecision


class RouterTests(unittest.TestCase):
    def setUp(self) -> None:
        self.router = Router()

    def test_code_requests_route_to_code_model(self) -> None:
        decision = self.router.route("Debug this Python function and explain the bug")

        self.assertEqual(decision.selected_provider, "gpt-4.1")
        self.assertEqual(decision.category, "code")
        self.assertIn("code signals", decision.reason.lower())

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


if __name__ == "__main__":
    unittest.main()

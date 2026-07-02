import anthropic
import pytest

from model_router.classifier import CLASSIFIER_MODEL, classify, heuristic_tier
from tests.conftest import FakeClient, FakeMessages, make_response, text_block, thinking_block


class TestHeuristicTier:
    def test_high_keyword(self):
        assert heuristic_tier("plan the database migration to postgres") == "high"

    def test_high_on_huge_context(self):
        assert heuristic_tier("summarize this", context_tokens=500_000) == "high"

    def test_mid_keyword(self):
        assert heuristic_tier("debug this flaky test") == "mid"

    def test_mid_on_large_context(self):
        assert heuristic_tier("summarize this", context_tokens=150_000) == "mid"

    def test_default_low(self):
        assert heuristic_tier("what does this function return?") == "low"


class TestClassify:
    def test_uses_classifier_verdict(self):
        messages = FakeMessages(create_response=make_response([text_block("high")]))
        route = classify(FakeClient(messages), "restructure everything")
        assert (route.model, route.effort) == ("claude-fable-5", "xhigh")
        assert route.source == "classifier"
        assert messages.create_calls[0]["model"] == CLASSIFIER_MODEL

    def test_verdict_whitespace_and_case_tolerated(self):
        messages = FakeMessages(create_response=make_response([text_block("  Mid \n")]))
        route = classify(FakeClient(messages), "anything")
        assert (route.model, route.effort) == ("claude-opus-4-8", "high")

    def test_skips_non_text_blocks(self):
        messages = FakeMessages(
            create_response=make_response([thinking_block(), text_block("low")])
        )
        route = classify(FakeClient(messages), "anything")
        assert route.model == "claude-sonnet-5"

    def test_garbage_verdict_falls_back_to_heuristic(self):
        messages = FakeMessages(
            create_response=make_response([text_block("extremely hard")])
        )
        route = classify(FakeClient(messages), "debug this crash")
        assert route.source == "heuristic"
        assert route.model == "claude-opus-4-8"

    def test_empty_content_falls_back_to_heuristic(self):
        messages = FakeMessages(create_response=make_response([]))
        route = classify(FakeClient(messages), "quick question")
        assert route.source == "heuristic"
        assert route.model == "claude-sonnet-5"

    def test_api_error_falls_back_to_heuristic(self):
        messages = FakeMessages(create_error=anthropic.AnthropicError("boom"))
        route = classify(FakeClient(messages), "plan the migration")
        assert route.source == "heuristic"
        assert route.model == "claude-fable-5"

    def test_non_anthropic_errors_propagate(self):
        messages = FakeMessages(create_error=ValueError("bug in caller"))
        with pytest.raises(ValueError):
            classify(FakeClient(messages), "anything")

    def test_long_prompt_truncated_for_classifier(self):
        messages = FakeMessages(create_response=make_response([text_block("low")]))
        classify(FakeClient(messages), "x" * 50_000)
        sent = messages.create_calls[0]["messages"][0]["content"]
        assert len(sent) < 3_000

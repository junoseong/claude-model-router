import pytest

from model_router.classifier import Route
from model_router.router import (
    FABLE_FALLBACK_MODEL,
    FALLBACK_BETA,
    MAX_TOKENS,
    RefusalError,
    build_request,
    execute,
    run,
)
from tests.conftest import FakeClient, FakeMessages, make_response, text_block, thinking_block

FABLE_ROUTE = Route(model="claude-fable-5", effort="xhigh", tier="high", source="classifier")
OPUS_ROUTE = Route(model="claude-opus-4-8", effort="high", tier="mid", source="classifier")
SONNET_ROUTE = Route(model="claude-sonnet-5", effort="medium", tier="low", source="classifier")
HAIKU_ROUTE = Route(model="claude-haiku-4-5", effort=None, tier="trivial", source="classifier")

MSGS = [{"role": "user", "content": "hello"}]


class TestBuildRequest:
    def test_fable_gets_fallbacks_and_no_thinking(self):
        kwargs = build_request(FABLE_ROUTE, MSGS)
        assert kwargs["betas"] == [FALLBACK_BETA]
        assert kwargs["fallbacks"] == [{"model": FABLE_FALLBACK_MODEL}]
        assert "thinking" not in kwargs  # explicit thinking config 400s on Fable
        assert kwargs["output_config"] == {"effort": "xhigh"}
        assert kwargs["max_tokens"] == MAX_TOKENS

    def test_opus_gets_adaptive_thinking_and_no_fallbacks(self):
        kwargs = build_request(OPUS_ROUTE, MSGS)
        assert kwargs["thinking"] == {"type": "adaptive"}
        assert "betas" not in kwargs
        assert "fallbacks" not in kwargs

    def test_sonnet_gets_adaptive_thinking(self):
        kwargs = build_request(SONNET_ROUTE, MSGS)
        assert kwargs["thinking"] == {"type": "adaptive"}

    def test_haiku_gets_no_thinking_no_effort_no_fallbacks(self):
        kwargs = build_request(HAIKU_ROUTE, MSGS)
        assert "thinking" not in kwargs  # adaptive thinking 400s on Haiku 4.5
        assert "output_config" not in kwargs  # effort param 400s on Haiku 4.5
        assert "betas" not in kwargs
        assert "fallbacks" not in kwargs


class TestExecute:
    def test_fable_routes_through_beta_namespace(self):
        beta = FakeMessages(
            stream_response=make_response([text_block("answer")], model="claude-fable-5")
        )
        client = FakeClient(FakeMessages(), beta_messages=beta)
        result = execute(client, FABLE_ROUTE, MSGS)
        assert result.text == "answer"
        assert len(beta.stream_calls) == 1
        assert client.messages.stream_calls == []

    def test_non_fable_uses_regular_namespace(self):
        messages = FakeMessages(
            stream_response=make_response([text_block("ok")], model="claude-opus-4-8")
        )
        client = FakeClient(messages)
        result = execute(client, OPUS_ROUTE, MSGS)
        assert result.text == "ok"
        assert len(messages.stream_calls) == 1

    def test_refusal_raises_with_details(self):
        beta = FakeMessages(
            stream_response=make_response(
                [], stop_reason="refusal", stop_details={"category": "test"}
            )
        )
        client = FakeClient(FakeMessages(), beta_messages=beta)
        with pytest.raises(RefusalError) as exc:
            execute(client, FABLE_ROUTE, MSGS)
        assert exc.value.stop_details == {"category": "test"}

    def test_text_extraction_skips_thinking_blocks(self):
        messages = FakeMessages(
            stream_response=make_response(
                [thinking_block(), text_block("part1 "), text_block("part2")]
            )
        )
        result = execute(FakeClient(messages), OPUS_ROUTE, MSGS)
        assert result.text == "part1 part2"

    def test_served_by_reports_fallback_model(self):
        beta = FakeMessages(
            stream_response=make_response(
                [text_block("rescued")], model="claude-opus-4-8"
            )
        )
        client = FakeClient(FakeMessages(), beta_messages=beta)
        result = execute(client, FABLE_ROUTE, MSGS)
        assert result.route.model == "claude-fable-5"
        assert result.served_by == "claude-opus-4-8"


class TestRun:
    def test_classifies_then_executes(self):
        messages = FakeMessages(
            create_response=make_response([text_block("mid")]),
            stream_response=make_response([text_block("done")], model="claude-opus-4-8"),
        )
        client = FakeClient(messages)
        result = run(client, "debug this")
        assert result.route.model == "claude-opus-4-8"
        assert result.text == "done"
        assert messages.stream_calls[0]["messages"] == [
            {"role": "user", "content": "debug this"}
        ]

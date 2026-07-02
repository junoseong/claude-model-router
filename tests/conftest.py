"""Shared fakes standing in for the Anthropic SDK response surface."""

from __future__ import annotations

from types import SimpleNamespace


class FakeBlock(SimpleNamespace):
    """Content block with .type and optional .text, like SDK blocks."""


def text_block(text: str) -> FakeBlock:
    return FakeBlock(type="text", text=text)


def thinking_block(text: str = "") -> FakeBlock:
    return FakeBlock(type="thinking", thinking=text, text=None)


class FakeResponse(SimpleNamespace):
    pass


def make_response(
    content,
    stop_reason="end_turn",
    model="claude-test",
    stop_details=None,
) -> FakeResponse:
    return FakeResponse(
        content=content,
        stop_reason=stop_reason,
        model=model,
        stop_details=stop_details,
    )


class FakeStream:
    """Context manager mimicking client.messages.stream(...)."""

    def __init__(self, response):
        self._response = response

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def get_final_message(self):
        return self._response


class FakeMessages:
    """Records calls; serves canned responses for create() and stream()."""

    def __init__(self, create_response=None, stream_response=None, create_error=None):
        self.create_response = create_response
        self.stream_response = stream_response
        self.create_error = create_error
        self.create_calls = []
        self.stream_calls = []

    def create(self, **kwargs):
        self.create_calls.append(kwargs)
        if self.create_error is not None:
            raise self.create_error
        return self.create_response

    def stream(self, **kwargs):
        self.stream_calls.append(kwargs)
        return FakeStream(self.stream_response)


class FakeClient:
    def __init__(self, messages: FakeMessages, beta_messages: FakeMessages | None = None):
        self.messages = messages
        self.beta = SimpleNamespace(messages=beta_messages or FakeMessages())

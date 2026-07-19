import urllib.error
from types import SimpleNamespace

import pytest

from ansi_scaler.stages.ollama import OllamaRequestError, request_with_retry


def settings(attempts: int = 4) -> SimpleNamespace:
    return SimpleNamespace(
        retry_attempts=attempts,
        retry_initial_seconds=1.0,
        retry_max_seconds=8.0,
    )


def test_ollama_request_retries_transient_failure_with_backoff() -> None:
    calls = 0
    sleeps = []

    def request(_payload):
        nonlocal calls
        calls += 1
        if calls < 3:
            raise urllib.error.URLError("connection reset")
        return {"message": {"content": "ok"}}

    result = request_with_retry(request, {}, settings(), service="test Ollama", sleep=sleeps.append)

    assert result["message"]["content"] == "ok"
    assert calls == 3
    assert sleeps == [1.0, 2.0]


def test_ollama_request_exhaustion_is_an_item_error() -> None:
    calls = 0

    def request(_payload):
        nonlocal calls
        calls += 1
        raise TimeoutError("timed out")

    with pytest.raises(OllamaRequestError, match="failed after 3 attempts"):
        request_with_retry(request, {}, settings(3), service="test Ollama", sleep=lambda _: None)

    assert calls == 3


def test_ollama_request_does_not_retry_non_transient_http_error() -> None:
    calls = 0

    def request(_payload):
        nonlocal calls
        calls += 1
        raise urllib.error.HTTPError("http://ollama/api/chat", 400, "bad request", {}, None)

    with pytest.raises(OllamaRequestError, match="HTTP 400"):
        request_with_retry(request, {}, settings(), service="test Ollama", sleep=lambda _: None)

    assert calls == 1

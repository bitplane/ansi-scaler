from __future__ import annotations

import json
import logging
import time
import urllib.error
from collections.abc import Callable
from typing import Any, Protocol


LOGGER = logging.getLogger(__name__)
TRANSIENT_HTTP_STATUSES = {408, 429, 500, 502, 503, 504}
RequestFunction = Callable[[dict[str, Any]], dict[str, Any]]


class RetrySettings(Protocol):
    retry_attempts: int
    retry_initial_seconds: float
    retry_max_seconds: float


class OllamaRequestError(RuntimeError):
    """An Ollama request exhausted its retries for one corpus item."""


def request_with_retry(
    request_function: RequestFunction,
    payload: dict[str, Any],
    settings: RetrySettings,
    *,
    service: str,
    sleep: Callable[[float], None] = time.sleep,
) -> dict[str, Any]:
    last_error: Exception | None = None
    for attempt in range(1, settings.retry_attempts + 1):
        try:
            return request_function(payload)
        except urllib.error.HTTPError as error:
            if error.code not in TRANSIENT_HTTP_STATUSES:
                raise OllamaRequestError(f"{service} request failed with HTTP {error.code}: {error.reason}") from error
            last_error = error
        except (OSError, json.JSONDecodeError) as error:
            last_error = error

        if attempt == settings.retry_attempts:
            break
        delay = min(settings.retry_initial_seconds * (2 ** (attempt - 1)), settings.retry_max_seconds)
        LOGGER.warning(
            "%s request failed (%s); retrying in %.1fs (%d/%d)",
            service,
            last_error,
            delay,
            attempt + 1,
            settings.retry_attempts,
        )
        sleep(delay)

    raise OllamaRequestError(
        f"{service} request failed after {settings.retry_attempts} attempts: {last_error}"
    ) from last_error

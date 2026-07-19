from __future__ import annotations

import fcntl
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator


class CorpusBusyError(RuntimeError):
    """Raised when exclusive corpus maintenance cannot start safely."""


@contextmanager
def corpus_lock(data_dir: Path, *, exclusive: bool, blocking: bool = True) -> Iterator[None]:
    """Coordinate corpus readers/writers with exclusive maintenance operations."""
    data_dir.mkdir(parents=True, exist_ok=True)
    path = data_dir / ".corpus.lock"
    with path.open("a+b") as handle:
        operation = fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH
        if not blocking:
            operation |= fcntl.LOCK_NB
        try:
            fcntl.flock(handle.fileno(), operation)
        except BlockingIOError as error:
            raise CorpusBusyError(
                "Corpus is in use by a pipeline stage or review server; stop it and retry garbage collection"
            ) from error
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)

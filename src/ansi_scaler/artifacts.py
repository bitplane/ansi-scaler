from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from typing import Iterator


def artifact_path(root: Path, stage: str, artifact_id: str, suffix: str) -> Path:
    return root / stage / artifact_id[:2] / f"{artifact_id}{suffix}"


@contextmanager
def atomic_destination(destination: Path) -> Iterator[Path]:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp")
    try:
        yield temporary
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)

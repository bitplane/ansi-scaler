import hashlib
import io
import tarfile
from pathlib import Path

import pytest
import zstandard

from ansi_scaler.review.ansi import PyramidCache, ansi_to_runs


def write_archive(path: Path, levels: dict[int, bytes]) -> dict:
    path.parent.mkdir(parents=True, exist_ok=True)
    buffer = io.BytesIO()
    metadata = []
    with tarfile.open(fileobj=buffer, mode="w") as archive:
        for width, data in levels.items():
            member_path = f"levels/{width:03d}.ansi"
            info = tarfile.TarInfo(member_path)
            info.size = len(data)
            archive.addfile(info, io.BytesIO(data))
            metadata.append(
                {
                    "width": width,
                    "rows": data.count(b"\n"),
                    "source_lod": "lod-1",
                    "bytes": len(data),
                    "sha256": hashlib.sha256(data).hexdigest(),
                    "path": member_path,
                }
            )
    path.write_bytes(zstandard.ZstdCompressor().compress(buffer.getvalue()))
    return {
        "id": path.stem,
        "archive_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "pyramid_levels": metadata,
    }


def test_ansi_parser_tracks_combined_foreground_and_background() -> None:
    data = b"\x1b[38;2;195;74;29;48;2;131;55;34m<\xe2\x96\x83 \n\x1b[39;49m&\x1b[38;2;1;2;3mX \x1b[0m\n"

    rendered = ansi_to_runs(data, width=3, rows=2)

    assert rendered == {
        "palette": [[195, 74, 29], [131, 55, 34], [1, 2, 3]],
        "runs": [["<▃ \n", 0, 1], ["&", None, None], ["X ", 2, None], ["\n", None, None]],
    }


def test_ansi_parser_rejects_unsupported_or_invalid_sequences() -> None:
    with pytest.raises(ValueError, match="Unsupported ANSI SGR"):
        ansi_to_runs(b"\x1b[1mbright", width=6, rows=1)
    with pytest.raises(ValueError, match="unsupported escape"):
        ansi_to_runs(b"\x1b[2J", width=0, rows=0)
    with pytest.raises(ValueError, match="out of range"):
        ansi_to_runs(b"\x1b[38;2;999;0;0mX", width=1, rows=1)


def test_ansi_parser_validates_grid_dimensions() -> None:
    with pytest.raises(ValueError, match="row count mismatch"):
        ansi_to_runs(b"aa\n", width=2, rows=2)
    with pytest.raises(ValueError, match="row 2 width mismatch"):
        ansi_to_runs(b"aa\nb\n", width=2, rows=2)


def test_pyramid_cache_validates_levels_and_drops_previous_asset(tmp_path: Path) -> None:
    first_path = tmp_path / "first.tar.zst"
    second_path = tmp_path / "second.tar.zst"
    first = write_archive(first_path, {2: b"aa\n", 3: b"bbb\n"})
    second = write_archive(second_path, {2: b"cccc\n"})
    cache = PyramidCache(byte_budget=8)

    assert cache.level(first, first_path, 3) == b"bbb\n"
    assert cache.level(second, second_path, 2) == b"cccc\n"
    assert cache.byte_size == len(b"cccc\n")
    assert len(cache._items) == 1
    assert next(iter(cache._items))[0] == second["id"]

    second["pyramid_levels"][0]["sha256"] = "0" * 64
    cache.clear()
    with pytest.raises(ValueError, match="hash mismatch"):
        cache.level(second, second_path, 2)

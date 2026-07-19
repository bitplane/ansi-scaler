from __future__ import annotations

import hashlib
import re
import tarfile
import threading
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import zstandard


SGR_PATTERN = re.compile(rb"\x1b\[([0-9;]*)m")
RGB = tuple[int, int, int]


def ansi_to_runs(data: bytes, *, width: int, rows: int) -> dict[str, list[Any]]:
    foreground: RGB | None = None
    background: RGB | None = None
    styled_text: list[tuple[str, RGB | None, RGB | None]] = []
    position = 0
    for match in SGR_PATTERN.finditer(data):
        if match.start() > position:
            styled_text.append((data[position : match.start()].decode("utf-8"), foreground, background))
        raw_parameters = match.group(1)
        parameters = [int(value) if value else 0 for value in raw_parameters.split(b";")] if raw_parameters else [0]
        index = 0
        while index < len(parameters):
            value = parameters[index]
            if value == 0:
                foreground = None
                background = None
                index += 1
            elif value in (38, 48):
                if index + 4 >= len(parameters) or parameters[index + 1] != 2:
                    raise ValueError(f"Unsupported ANSI colour sequence: {match.group(0)!r}")
                colour = tuple(parameters[index + 2 : index + 5])
                if any(component < 0 or component > 255 for component in colour):
                    raise ValueError(f"ANSI RGB component is out of range: {match.group(0)!r}")
                if value == 38:
                    foreground = colour
                else:
                    background = colour
                index += 5
            elif value == 39:
                foreground = None
                index += 1
            elif value == 49:
                background = None
                index += 1
            else:
                raise ValueError(f"Unsupported ANSI SGR parameter {value}: {match.group(0)!r}")
        position = match.end()
    if position < len(data):
        styled_text.append((data[position:].decode("utf-8"), foreground, background))
    if b"\x1b" in SGR_PATTERN.sub(b"", data):
        raise ValueError("ANSI stream contains an unsupported escape sequence")

    text = "".join(item[0] for item in styled_text)
    lines = text.splitlines()
    if len(lines) != rows:
        raise ValueError(f"ANSI row count mismatch: expected {rows}, got {len(lines)}")
    invalid = next(((index, len(line)) for index, line in enumerate(lines, 1) if len(line) != width), None)
    if invalid is not None:
        line_number, actual_width = invalid
        raise ValueError(f"ANSI row {line_number} width mismatch: expected {width}, got {actual_width}")

    palette: list[RGB] = []
    palette_indexes: dict[RGB, int] = {}

    def colour_index(colour: RGB | None) -> int | None:
        if colour is None:
            return None
        if colour not in palette_indexes:
            palette_indexes[colour] = len(palette)
            palette.append(colour)
        return palette_indexes[colour]

    runs: list[list[Any]] = []
    for run_text, run_foreground, run_background in styled_text:
        if not run_text:
            continue
        encoded = [run_text, colour_index(run_foreground), colour_index(run_background)]
        if runs and runs[-1][1:] == encoded[1:]:
            runs[-1][0] += run_text
        else:
            runs.append(encoded)
    return {"palette": [list(colour) for colour in palette], "runs": runs}


@dataclass(frozen=True)
class CachedPyramid:
    levels: dict[int, bytes]
    byte_size: int


class PyramidCache:
    def __init__(self, byte_budget: int = 64 * 1024 * 1024) -> None:
        self.byte_budget = byte_budget
        self.byte_size = 0
        self._items: OrderedDict[tuple[str, str], CachedPyramid] = OrderedDict()
        self._lock = threading.Lock()

    def clear(self) -> None:
        with self._lock:
            self._items.clear()
            self.byte_size = 0

    def level(self, record: dict[str, Any], archive_path: Path, width: int) -> bytes:
        key = (record["id"], record["archive_sha256"])
        with self._lock:
            cached = self._items.get(key)
            if cached is None:
                cached = self._load(record, archive_path)
                self._insert(key, cached)
            else:
                self._items.move_to_end(key)
            try:
                return cached.levels[width]
            except KeyError as error:
                raise KeyError(f"Pyramid {record['id']} has no width {width}") from error

    def _insert(self, key: tuple[str, str], item: CachedPyramid) -> None:
        self._items.clear()
        self.byte_size = 0
        self._items[key] = item
        self.byte_size += item.byte_size

    @staticmethod
    def _load(record: dict[str, Any], archive_path: Path) -> CachedPyramid:
        expected = {level["path"]: level for level in record.get("pyramid_levels", [])}
        if not expected:
            raise ValueError(f"Pyramid record {record['id']} has no declared levels")
        levels: dict[int, bytes] = {}
        with archive_path.open("rb") as compressed:
            decompressor = zstandard.ZstdDecompressor()
            with decompressor.stream_reader(compressed) as stream:
                with tarfile.open(fileobj=stream, mode="r|") as archive:
                    for member in archive:
                        level = expected.get(member.name)
                        if level is None:
                            continue
                        if not member.isfile():
                            raise ValueError(f"Pyramid level is not a regular file: {member.name}")
                        handle = archive.extractfile(member)
                        if handle is None:
                            raise ValueError(f"Could not read pyramid level: {member.name}")
                        data = handle.read()
                        digest = hashlib.sha256(data).hexdigest()
                        if digest != level["sha256"]:
                            raise ValueError(f"Pyramid level hash mismatch: {member.name}")
                        levels[level["width"]] = data
        expected_widths = {level["width"] for level in expected.values()}
        missing = expected_widths - levels.keys()
        if missing:
            raise ValueError(f"Pyramid archive is missing widths: {', '.join(map(str, sorted(missing)))}")
        return CachedPyramid(levels=levels, byte_size=sum(map(len, levels.values())))

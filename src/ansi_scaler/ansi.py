from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np


SGR_PATTERN = re.compile(rb"\x1b\[([0-9;]*)m")


@dataclass(frozen=True)
class AnsiCells:
    glyphs: np.ndarray
    foreground_rgb: np.ndarray
    background_rgb: np.ndarray
    background_present: np.ndarray


def decode_ansi(data: bytes, *, width: int, rows: int) -> AnsiCells:
    foreground: tuple[int, int, int] | None = None
    background: tuple[int, int, int] | None = None
    cells: list[tuple[str, tuple[int, int, int], tuple[int, int, int] | None]] = []
    row_widths: list[int] = []
    current_row_width = 0
    position = 0

    def add_text(raw: bytes) -> None:
        nonlocal current_row_width
        if not raw:
            return
        text = raw.decode("utf-8")
        for character in text:
            if character == "\n":
                row_widths.append(current_row_width)
                current_row_width = 0
                continue
            if character == "\r":
                raise ValueError("ANSI stream contains a carriage return")
            if foreground is None:
                raise ValueError("ANSI cell has no explicit foreground colour")
            cells.append((character, foreground, background))
            current_row_width += 1

    for match in SGR_PATTERN.finditer(data):
        add_text(data[position : match.start()])
        values = [int(value) if value else 0 for value in match.group(1).split(b";")] if match.group(1) else [0]
        index = 0
        while index < len(values):
            value = values[index]
            if value == 0:
                foreground = background = None
                index += 1
            elif value in (38, 48):
                if index + 4 >= len(values) or values[index + 1] != 2:
                    raise ValueError(f"Unsupported ANSI colour sequence: {match.group(0)!r}")
                colour = tuple(values[index + 2 : index + 5])
                if any(component not in range(256) for component in colour):
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
    add_text(data[position:])
    if b"\x1b" in SGR_PATTERN.sub(b"", data):
        raise ValueError("ANSI stream contains an unsupported escape sequence")
    if len(row_widths) != rows:
        raise ValueError(f"ANSI row count mismatch: expected {rows}, got {len(row_widths)}")
    invalid = next(((index, actual) for index, actual in enumerate(row_widths, 1) if actual != width), None)
    if invalid:
        raise ValueError(f"ANSI row {invalid[0]} width mismatch: expected {width}, got {invalid[1]}")
    if len(cells) != width * rows:
        raise ValueError(f"ANSI cell count mismatch: expected {width * rows}, got {len(cells)}")

    return AnsiCells(
        glyphs=np.asarray([ord(cell[0]) for cell in cells], dtype=np.uint32),
        foreground_rgb=np.asarray([cell[1] for cell in cells], dtype=np.uint8),
        background_rgb=np.asarray([cell[2] or (0, 0, 0) for cell in cells], dtype=np.uint8),
        background_present=np.asarray([cell[2] is not None for cell in cells], dtype=np.bool_),
    )


def encode_ansi(cells: AnsiCells, *, width: int, rows: int) -> bytes:
    if len(cells.glyphs) != width * rows:
        raise ValueError("Cell count does not match dimensions")
    output = bytearray()
    foreground: tuple[int, int, int] | None = None
    background: tuple[int, int, int] | None = None
    for index in range(width * rows):
        new_foreground = tuple(int(value) for value in cells.foreground_rgb[index])
        new_background = (
            tuple(int(value) for value in cells.background_rgb[index]) if cells.background_present[index] else None
        )
        parameters: list[str] = []
        if new_foreground != foreground:
            parameters.append("38;2;" + ";".join(map(str, new_foreground)))
            foreground = new_foreground
        if new_background != background:
            parameters.append("49" if new_background is None else "48;2;" + ";".join(map(str, new_background)))
            background = new_background
        if parameters:
            output.extend(f"\x1b[{';'.join(parameters)}m".encode())
        output.extend(chr(int(cells.glyphs[index])).encode("utf-8"))
        if (index + 1) % width == 0 and index + 1 < width * rows:
            output.extend(b"\n")
    output.extend(b"\x1b[0m\n")
    return bytes(output)

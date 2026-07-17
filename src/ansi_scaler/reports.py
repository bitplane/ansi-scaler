from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw


def checkerboard(size: tuple[int, int], square: int = 16) -> Image.Image:
    image = Image.new("RGB", size, "white")
    draw = ImageDraw.Draw(image)
    colors = ((238, 238, 238), (190, 190, 190))
    for y in range(0, size[1], square):
        for x in range(0, size[0], square):
            draw.rectangle(
                (x, y, x + square - 1, y + square - 1),
                fill=colors[((x // square) + (y // square)) % 2],
            )
    return image


def contact_sheet(paths: list[Path], destination: Path, columns: int = 5, tile_size: int = 256) -> None:
    if not paths:
        return
    rows = (len(paths) + columns - 1) // columns
    sheet = Image.new("RGB", (columns * tile_size, rows * tile_size), "white")
    for index, path in enumerate(paths):
        foreground = Image.open(path).convert("RGBA")
        foreground.thumbnail((tile_size, tile_size), Image.Resampling.LANCZOS)
        tile = checkerboard((tile_size, tile_size))
        offset = ((tile_size - foreground.width) // 2, (tile_size - foreground.height) // 2)
        tile.paste(foreground, offset, foreground)
        sheet.paste(tile, ((index % columns) * tile_size, (index // columns) * tile_size))
    destination.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(destination)

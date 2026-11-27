"""Frame rendering: raw 64x64 grid -> upscaled PNG (512x512 at the default scale), plus the
crop (`inspect`) and resample (`read_pixels`) helpers.

The image is the model's only observation: no grid text, no diff, no ruler. Each cell's outline
is drawn in the cell's own colour darkened (75% brightness by default), so cell boundaries are
visible without adding a colour that could be confused with one in the palette (which already
contains several greys).

Cell values are the engine's 0-15 range (`arc_agi.rendering.COLOR_MAP`); out-of-range values are
not defended against.
"""
import string
from typing import Optional

import numpy as np
from PIL import Image

from arc_agi.rendering import COLOR_MAP, hex_to_rgb

_PALETTE_RGB: dict[int, tuple[int, int, int]] = {v: hex_to_rgb(h) for v, h in COLOR_MAP.items()}


def color_for(value: int) -> tuple[int, int, int]:
    """RGB for a raw cell value."""
    return _PALETTE_RGB[int(value)]


def darken(rgb: tuple[int, int, int], brightness: float) -> tuple[int, int, int]:
    return tuple(round(c * brightness) for c in rgb)


def _luts(gridline_brightness: float) -> tuple[np.ndarray, np.ndarray]:
    base = np.array([color_for(v) for v in range(16)], dtype=np.uint8)
    line = np.array([darken(color_for(v), gridline_brightness) for v in range(16)], dtype=np.uint8)
    return base, line


def render_frame(frame: np.ndarray, scale: int = 8, gridline_brightness: float = 0.75) -> Image.Image:
    """Render one raw int grid to an upscaled RGB image with per-cell gridlines.

    Each cell becomes a `scale`x`scale` block of its own colour; the block's last row and last
    column are darkened to `gridline_brightness` of that same colour, so the boundary is visible
    without introducing a colour the palette doesn't already contain. The line consumes a pixel
    *of* the cell rather than sitting between cells, so `image_px = 64 * scale` still holds
    exactly and an image pixel maps to its grid cell by `x // scale`.
    """
    base_lut, line_lut = _luts(gridline_brightness)
    idx = frame.astype(np.intp)
    base_img = base_lut[idx]
    line_img = line_lut[idx]

    out = np.repeat(np.repeat(base_img, scale, axis=0), scale, axis=1)
    lines = np.repeat(np.repeat(line_img, scale, axis=0), scale, axis=1)
    out[scale - 1 :: scale, :] = lines[scale - 1 :: scale, :]
    out[:, scale - 1 :: scale] = lines[:, scale - 1 :: scale]
    return Image.fromarray(out, mode="RGB")


def frame_to_png_bytes(frame: np.ndarray, scale: int = 8, gridline_brightness: float = 0.75) -> bytes:
    import io

    buf = io.BytesIO()
    render_frame(frame, scale=scale, gridline_brightness=gridline_brightness).save(buf, format="PNG")
    return buf.getvalue()


def crop_region(frame: np.ndarray, region: Optional[dict], scale: int = 8,
                 gridline_brightness: float = 0.75) -> Image.Image:
    """Re-render a stored raw frame through the same renderer the model sees live, optionally
    cropped to an image-pixel region -- used by `inspect`. `region=None` is the whole image.

    A crop is magnified back up to fill the full frame size, aspect preserved, nearest-neighbour,
    so a small region stays legible. Region validation caps `width`/`height` at the frame size, so this only ever
    magnifies up or leaves the size unchanged, never downscales.
    """
    img = render_frame(frame, scale=scale, gridline_brightness=gridline_brightness)
    if region is None:
        return img
    x, y, w, h = region["x"], region["y"], region["width"], region["height"]
    cropped = img.crop((x, y, x + w, y + h))
    longest = max(w, h)
    rendered_w = w * img.width // longest
    rendered_h = h * img.width // longest
    if (rendered_w, rendered_h) != cropped.size:
        cropped = cropped.resize((rendered_w, rendered_h), Image.Resampling.NEAREST)
    return cropped


def region_to_png_bytes(frame: np.ndarray, region: Optional[dict], scale: int = 8,
                         gridline_brightness: float = 0.75) -> bytes:
    import io

    buf = io.BytesIO()
    crop_region(frame, region, scale=scale, gridline_brightness=gridline_brightness).save(buf, format="PNG")
    return buf.getvalue()


_SYMBOLS = string.ascii_uppercase  # at most 16 distinct raw values ever occur (COLOR_MAP is 0-15)


def resample_region(frame: np.ndarray, region: Optional[dict], rows: int, columns: int,
                     scale: int = 8, symbol_table: Optional[dict] = None) -> tuple:
    """Sample `rows`x`columns` points evenly across `region` (image pixels; `None` is the whole
    image) from the RAW grid -- exact colours, not anti-aliased pixels -- and return
    `(grid_rows, palette, axes)`.

    `grid_rows` is `rows` strings of `columns` characters; `palette` maps each character to the
    hex colour it stands for -- never the raw engine value, which the harness withholds. Symbols
    are assigned in ascending hex-colour order, so the symbol sequence does not reveal the
    engine's colour numbering. `rows=64, columns=64` over the whole image reproduces `frame`
    exactly, up to that remapping.

    `symbol_table` is an optional `{raw_value: symbol}` dict the caller owns and reuses across
    calls (`env.EnvAdapter.pixel_symbols`, for the life of a run): existing entries are kept and
    only colours not yet in it get a new symbol, so a colour keeps the same letter from call to
    call. If omitted, a fresh table is used for this call only.

    `axes` is `{"row_y": [...], "col_x": [...], "coarse": bool}`: the image-pixel centre of each
    sample row / column (the caller's own coordinate frame, so labelling the output with them
    discloses nothing), and `coarse` -- `True` when the sample points did not land on every source
    cell whose centre falls inside the region, i.e. the read is a summary, not an exact one. This
    is advisory: undersampling is a legitimate technique (a coarse strip to see gross structure),
    it just must not be mistaken for ground truth.
    """
    grid_h, grid_w = frame.shape
    if region is None:
        x0, y0, w, h = 0, 0, grid_w * scale, grid_h * scale
    else:
        x0, y0, w, h = region["x"], region["y"], region["width"], region["height"]

    row_y = [y0 + (r + 0.5) * h / rows for r in range(rows)]
    col_x = [x0 + (c + 0.5) * w / columns for c in range(columns)]

    def _cell_row(py: float) -> int:
        return min(max(int(py // scale), 0), grid_h - 1)

    def _cell_col(px: float) -> int:
        return min(max(int(px // scale), 0), grid_w - 1)

    values = [[int(frame[_cell_row(py), _cell_col(px)]) for px in col_x] for py in row_y]

    distinct = sorted({v for row in values for v in row}, key=lambda v: COLOR_MAP[v])
    table = symbol_table if symbol_table is not None else {}
    used = set(table.values())
    unused_symbols = (s for s in _SYMBOLS if s not in used)
    for v in distinct:
        if v not in table:
            table[v] = next(unused_symbols)
    symbol_of = {v: table[v] for v in distinct}
    grid_rows = ["".join(symbol_of[v] for v in row) for row in values]
    palette = {symbol_of[v]: COLOR_MAP[v] for v in distinct}

    # Full-grid scan for "did the samples touch every cell in the region"; trivial at 64x64.
    sampled = {(_cell_row(py), _cell_col(px)) for py in row_y for px in col_x}
    covered = {
        (cr, cc)
        for cr in range(grid_h) for cc in range(grid_w)
        if x0 <= cc * scale + scale / 2 < x0 + w and y0 <= cr * scale + scale / 2 < y0 + h
    }
    axes = {
        "row_y": [int(round(y)) for y in row_y],
        "col_x": [int(round(x)) for x in col_x],
        "coarse": bool(covered - sampled),
    }
    return grid_rows, palette, axes

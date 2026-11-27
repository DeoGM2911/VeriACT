"""The fixed 16-entry ARC-AGI-3 color palette (raw grid value 0-15 -> hex), needed to render
`read_pixels` sampled grids as actual color swatches instead of raw letters.

Not imported from the SDK (`arc_agi.rendering.COLOR_MAP`) on purpose: this tool is standalone by
design, and this table is a fixed, game-agnostic engine constant, not something that changes per
game or is worth an SDK dependency for. Values match arc-agi==0.9.8.
"""

COLOR_MAP = {
    0: "#FFFFFF",
    1: "#CCCCCC",
    2: "#999999",
    3: "#666666",
    4: "#333333",
    5: "#000000",
    6: "#E53AA3",
    7: "#FF7BCC",
    8: "#F93C31",
    9: "#1E93FF",
    10: "#88D8F1",
    11: "#FFDC00",
    12: "#FF851B",
    13: "#921231",
    14: "#4FCC30",
    15: "#A356D6",
}


def as_js_array() -> str:
    """The table as a JS array literal, indexed by raw value, for embedding in the page."""
    return "[" + ",".join(f'"{COLOR_MAP[i]}"' for i in range(16)) + "]"

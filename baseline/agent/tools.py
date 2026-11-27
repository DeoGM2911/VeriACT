"""Tool schemas and argument validation for the tools offered to the model every turn:

- `play`        -- take one game action; ends the turn.
- `read_notes` / `write_notes` -- read or replace a notes file (`GUIDE.md`, `WORKING.md`).
- `inspect`     -- re-view archived frames (optionally cropped) as images.
- `read_pixels` -- resample a frame region into a rows x columns grid of colour symbols.
- `history`     -- a text summary of past attempts or the per-turn event log.

Each `validate_*_args` turns raw tool-call arguments into a typed action, or raises
`ToolCallError` whose message is sent back to the model so it can retry.

`ACTION6` is named literally: it is the SDK's click action, part of the fixed action vocabulary
shared by every ARC-AGI-3 environment, not a game-specific fact. The model clicks in image pixels;
`validate_play_args` converts them to grid cells by flooring by the image scale.
"""
import dataclasses
from typing import Optional

from baseline.memory.memory import NOTES_FILES


class ToolCallError(Exception):
    """A tool call the model made cannot be executed; the message is sent back to the model."""


@dataclasses.dataclass
class PlayAction:
    action: str
    expectation: str = ""  # required by the schema: the model's prediction of the next frame
    x: Optional[int] = None  # image pixel, as emitted by the model
    y: Optional[int] = None
    col: Optional[int] = None  # grid cell, as sent to the SDK
    row: Optional[int] = None


def play_schema(description: str) -> dict:
    """OpenAI-style function schema for `play`. `description` is the rendered tools/play.md text."""
    return {
        "type": "function",
        "function": {
            "name": "play",
            "description": description,
            "parameters": {
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "description": "One of this turn's available actions.",
                    },
                    "expectation": {
                        "type": "string",
                        "description": "What you expect the next image to look like, specifically enough to be wrong.",
                    },
                    "x": {
                        "type": "integer",
                        "description": "Image pixel x, 0-max_px. Required only for the click-style action.",
                    },
                    "y": {
                        "type": "integer",
                        "description": "Image pixel y, 0-max_px. Required only for the click-style action.",
                    },
                },
                "required": ["action", "expectation"],
            },
        },
    }


def validate_play_args(
    args: dict, available_actions: list, scale: int, max_px: int, state: str = "NOT_FINISHED",
) -> PlayAction:
    """Turn raw tool-call args into a PlayAction, or raise ToolCallError with a model-facing reason."""
    action = args.get("action")
    if not isinstance(action, str):
        raise ToolCallError("`action` is required and must be a string.")
    action = action.upper()
    # While GAME_OVER, arcengine answers any non-RESET action with an empty frame list and zeroed
    # progress counters. Never send one: reject it here so the model is told to RESET instead.
    if state == "GAME_OVER" and action != "RESET":
        raise ToolCallError("The game is over; the only legal action now is RESET.")
    # RESET is always legal: arcengine handles it before consulting available_actions, and never
    # lists it there.
    if action != "RESET" and action not in available_actions:
        raise ToolCallError(
            f"'{action}' is not one of this turn's available actions: {', '.join(available_actions)}."
        )

    expectation = args.get("expectation")
    if not isinstance(expectation, str) or not expectation.strip():
        raise ToolCallError(
            "`expectation` is required: state what you expect the next image to look like."
        )

    x, y = args.get("x"), args.get("y")
    is_complex = action == "ACTION6"

    if is_complex:
        if x is None or y is None:
            raise ToolCallError("ACTION6 requires both `x` and `y`.")
        for name, v in (("x", x), ("y", y)):
            if not isinstance(v, int) or isinstance(v, bool) or not (0 <= v <= max_px):
                raise ToolCallError(f"`{name}` must be an integer 0-{max_px}, got {v!r}.")

        return PlayAction(
            action=action, expectation=expectation, x=x, y=y, col=x // scale, row=y // scale,
        )

    if x is not None or y is not None:
        raise ToolCallError(f"`x`/`y` are only valid for the click-style action, not {action}.")
    return PlayAction(action=action, expectation=expectation)


@dataclasses.dataclass
class ReadNotesAction:
    file: str


@dataclasses.dataclass
class WriteNotesAction:
    file: str
    content: str


def read_notes_schema(description: str) -> dict:
    return {
        "type": "function",
        "function": {
            "name": "read_notes",
            "description": description,
            "parameters": {
                "type": "object",
                "properties": {
                    "file": {
                        "type": "string",
                        "description": f"One of: {', '.join(NOTES_FILES)}.",
                    },
                },
                "required": ["file"],
            },
        },
    }


def write_notes_schema(description: str) -> dict:
    return {
        "type": "function",
        "function": {
            "name": "write_notes",
            "description": description,
            "parameters": {
                "type": "object",
                "properties": {
                    "file": {
                        "type": "string",
                        "description": f"One of: {', '.join(NOTES_FILES)}.",
                    },
                    "content": {
                        "type": "string",
                        "description": "The complete new contents of the file. Replaces it whole.",
                    },
                },
                "required": ["file", "content"],
            },
        },
    }


def _validate_notes_file(args: dict, allowed_files: tuple = NOTES_FILES) -> str:
    file = args.get("file")
    if not isinstance(file, str) or file not in allowed_files:
        raise ToolCallError(f"`file` must be one of {', '.join(allowed_files)}, got {file!r}.")
    return file


def validate_read_notes_args(args: dict) -> ReadNotesAction:
    return ReadNotesAction(file=_validate_notes_file(args))


def validate_write_notes_args(args: dict) -> WriteNotesAction:
    """`write_notes` replaces a whole note file (`GUIDE.md` or `WORKING.md`) with `content`."""
    file = _validate_notes_file(args)
    content = args.get("content")
    if not isinstance(content, str):
        raise ToolCallError("`content` is required and must be a string.")
    return WriteNotesAction(file=file, content=content)


@dataclasses.dataclass
class View:
    label: str
    turn: int
    frame: int = 0
    region: Optional[dict] = None  # {x, y, width, height} in image pixels; None = whole image


@dataclasses.dataclass
class InspectAction:
    question: str
    views: list  # list[View]


@dataclasses.dataclass
class ReadPixelsView(View):
    rows: int = 0
    columns: int = 0


@dataclasses.dataclass
class ReadPixelsAction:
    question: str
    views: list  # list[ReadPixelsView]


_MAX_SAMPLES = 64  # the raw grid is 64x64; finer sampling reads nothing new


def _views_schema(extra_props: dict, extra_required: list) -> dict:
    return {
        "type": "array",
        "items": {
            "type": "object",
            "properties": {
                "label": {"type": "string", "description": "A short name to refer back to this view by."},
                "turn": {"type": "integer", "description": "The turn number this frame is from."},
                "frame": {
                    "type": "integer",
                    "description": "Which frame within that turn's action; 0 is the first one it returned.",
                },
                "region": {
                    "type": "object",
                    "description": "Image-pixel region {x, y, width, height}. Omit for the whole image.",
                    "properties": {
                        "x": {"type": "integer"},
                        "y": {"type": "integer"},
                        "width": {"type": "integer"},
                        "height": {"type": "integer"},
                    },
                },
                **extra_props,
            },
            "required": ["label", "turn"] + extra_required,
            # `question` belongs once, at the call's top level, never per view. Rejecting unknown
            # keys stops a misplaced `question` from being silently dropped.
            "additionalProperties": False,
        },
    }


def inspect_schema(description: str) -> dict:
    return {
        "type": "function",
        "function": {
            "name": "inspect",
            "description": description,
            "parameters": {
                "type": "object",
                "properties": {
                    "question": {
                        "type": "string",
                        "description": "What you are trying to find out, written before you look.",
                    },
                    "views": _views_schema({}, []),
                },
                "required": ["question", "views"],
            },
        },
    }


def read_pixels_schema(description: str) -> dict:
    return {
        "type": "function",
        "function": {
            "name": "read_pixels",
            "description": description,
            "parameters": {
                "type": "object",
                "properties": {
                    "question": {
                        "type": "string",
                        "description": "What you are trying to establish, written before you look.",
                    },
                    "views": _views_schema(
                        {
                            "rows": {"type": "integer", "description": f"Sample rows, 1-{_MAX_SAMPLES}."},
                            "columns": {"type": "integer", "description": f"Sample columns, 1-{_MAX_SAMPLES}."},
                        },
                        ["rows", "columns"],
                    ),
                },
                "required": ["question", "views"],
            },
        },
    }


def _validate_region(region, max_px: int) -> Optional[dict]:
    if region is None:
        return None
    if not isinstance(region, dict):
        raise ToolCallError("`region` must be an object with x, y, width, height.")
    missing = [k for k in ("x", "y", "width", "height") if k not in region]
    if missing:
        raise ToolCallError(f"`region` is missing {', '.join(missing)}.")
    x, y, w, h = region["x"], region["y"], region["width"], region["height"]
    for name, v in (("x", x), ("y", y), ("width", w), ("height", h)):
        if not isinstance(v, int) or isinstance(v, bool):
            raise ToolCallError(f"`region.{name}` must be an integer.")
    if w <= 0 or h <= 0:
        raise ToolCallError("`region.width` and `region.height` must be positive.")
    if not (0 <= x <= max_px) or not (0 <= y <= max_px):
        raise ToolCallError(f"`region.x`/`region.y` must be 0-{max_px}.")
    if x + w - 1 > max_px or y + h - 1 > max_px:
        raise ToolCallError(f"`region` extends past the image edge (0-{max_px}).")
    return {"x": x, "y": y, "width": w, "height": h}


def _validate_view_common(v: dict, max_px: int) -> dict:
    if not isinstance(v, dict):
        raise ToolCallError("each view must be an object.")
    label = v.get("label")
    if not isinstance(label, str) or not label.strip():
        raise ToolCallError("each view needs a non-empty `label`.")
    turn = v.get("turn")
    if not isinstance(turn, int) or isinstance(turn, bool) or turn < 0:
        raise ToolCallError(f"view '{label}': `turn` must be a non-negative integer.")
    frame = v.get("frame", 0)
    if not isinstance(frame, int) or isinstance(frame, bool) or frame < 0:
        raise ToolCallError(f"view '{label}': `frame` must be a non-negative integer.")
    region = _validate_region(v.get("region"), max_px)
    return {"label": label, "turn": turn, "frame": frame, "region": region}


def _validate_views(raw_views, max_views: int) -> list:
    if not isinstance(raw_views, list) or not raw_views:
        raise ToolCallError("`views` is required and must be a non-empty list.")
    if len(raw_views) > max_views:
        raise ToolCallError(f"at most {max_views} views per call, got {len(raw_views)}.")
    return raw_views


def _validate_question(args: dict) -> str:
    question = args.get("question")
    if not isinstance(question, str) or not question.strip():
        # If the model put `question` inside `views[]`, say so specifically rather than reporting
        # it as missing.
        raw_views = args.get("views")
        if isinstance(raw_views, list) and any(
            isinstance(v, dict) and "question" in v for v in raw_views
        ):
            raise ToolCallError(
                "`question` goes once, at the top level of the call -- not inside `views[]`. "
                "Move it up next to `views`, not as a field on any individual view."
            )
        raise ToolCallError("`question` is required: state what you are trying to find out.")
    return question


def validate_inspect_args(args: dict, max_px: int, max_views: int) -> InspectAction:
    question = _validate_question(args)
    raw_views = _validate_views(args.get("views"), max_views)
    views = [View(**_validate_view_common(v, max_px)) for v in raw_views]
    return InspectAction(question=question, views=views)


def validate_read_pixels_args(args: dict, max_px: int, max_views: int,
                               max_samples: int = _MAX_SAMPLES) -> ReadPixelsAction:
    question = _validate_question(args)
    raw_views = _validate_views(args.get("views"), max_views)
    views = []
    for v in raw_views:
        common = _validate_view_common(v, max_px)
        rows, columns = v.get("rows"), v.get("columns")
        for name, val in (("rows", rows), ("columns", columns)):
            if not isinstance(val, int) or isinstance(val, bool) or not (1 <= val <= max_samples):
                raise ToolCallError(f"view '{common['label']}': `{name}` must be an integer 1-{max_samples}.")
        views.append(ReadPixelsView(**common, rows=rows, columns=columns))
    return ReadPixelsAction(question=question, views=views)



@dataclasses.dataclass
class HistoryAction:
    view: str                        # "attempts" | "events"
    from_turn: Optional[int] = None  # events only; None = from the start of the run
    to_turn: Optional[int] = None    # events only; None = through the most recent turn


def history_schema(description: str) -> dict:
    return {
        "type": "function",
        "function": {
            "name": "history",
            "description": description,
            "parameters": {
                "type": "object",
                "properties": {
                    "view": {
                        "type": "string",
                        "enum": ["attempts", "events"],
                        "description": "'attempts' summarizes each reset-to-reset span; 'events' is the raw per-turn log.",
                    },
                    "from_turn": {"type": "integer", "description": "`events` only. 1-indexed, inclusive."},
                    "to_turn": {"type": "integer", "description": "`events` only. 1-indexed, inclusive."},
                },
                "required": ["view"],
            },
        },
    }


def validate_history_args(args: dict, current_turn: int) -> HistoryAction:
    view = args.get("view")
    if view not in ("attempts", "events"):
        raise ToolCallError(f"`view` must be 'attempts' or 'events', got {view!r}.")
    from_turn, to_turn = args.get("from_turn"), args.get("to_turn")
    for name, v in (("from_turn", from_turn), ("to_turn", to_turn)):
        if v is not None and (not isinstance(v, int) or isinstance(v, bool) or not (1 <= v <= current_turn)):
            raise ToolCallError(
                f"`{name}` must be an integer between 1 and {current_turn} (the most recent turn "
                f"recorded so far), got {v!r}."
            )
    if from_turn is not None and to_turn is not None and from_turn > to_turn:
        raise ToolCallError(f"`from_turn` ({from_turn}) must not be after `to_turn` ({to_turn}).")
    return HistoryAction(view=view, from_turn=from_turn, to_turn=to_turn)


_SCHEMA_BUILDERS = {
    "play": play_schema,
    "read_notes": read_notes_schema,
    "write_notes": write_notes_schema,
    "inspect": inspect_schema,
    "read_pixels": read_pixels_schema,
    "history": history_schema,
}


def build_schemas(tool_descs: dict, active_names: list) -> list:
    """One OpenAI-style function schema per active tool, in manifest order."""
    return [_SCHEMA_BUILDERS[name](tool_descs[name]) for name in active_names]

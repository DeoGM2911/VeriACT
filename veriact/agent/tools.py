"""Tool schemas and argument validation for VeriACT.

Tools offered through the manifest (which stage offers which is decided in `loop.py`):

- `observe`         -- terminal call of OBSERVE: `description` (plus `objective_reconfirmed` /
                       `goal_changed` in the TERMINAL OBSERVE variant).
- `verify_and_plan` -- terminal call of VERIFY+PLAN: `verdict`, `evidence`, `observation_check`,
                       `objects_changed`, `mode`, and the conditionally required fields.
- `write_notes`     -- replace a notes file; the terminal call of UPDATE.
- `play`            -- take one game action (terminal call of ACT); clicks also need
                       `target_description`.
- `read_notes`, `inspect`, `read_pixels`, `history` -- free read-only tools.

`confirm_target` is not offered in the main conversation: it is the schema for the separate
click-target confirmation request (`loop._confirm_click_target`).

Each `validate_*_args` turns raw tool-call arguments into a typed result, or raises
`ToolCallError` whose message is sent back to the model so it can retry. Conditionally required
fields are checked for presence, not for truth.

`ACTION6` is named literally: it is the SDK's click action, part of the fixed action vocabulary
shared by every ARC-AGI-3 environment, not a game-specific fact. The model clicks in image pixels;
`validate_play_args` converts them to grid cells by flooring by the image scale.
"""
import dataclasses
from typing import Optional

from veriact.memory.memory import NOTES_FILES


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
    target_description: Optional[str] = None  # click action only: what the click is aimed at


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
                    "target_description": {
                        "type": "string",
                        "description": (
                            "Required only for the click-style action: a short description of the "
                            "object/feature you're clicking on (e.g. 'the blue switch'), used to check "
                            "your click landed in the right place."
                        ),
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
    # arcengine's own perform_action() (base_game.py) special-cases GAME_OVER/WIN ahead of the
    # game's own step(): any non-RESET action short-circuits to a response that omits
    # levels_completed/win_levels (silently defaulting to 0) and returns an empty frame list --
    # never a real game frame. The harness must never send the SDK such an action; reject it here,
    # the same way an action outside available_actions is rejected, so the model gets a corrective
    # message and a chance to RESET instead of the run crashing on the resulting empty frame.
    if state == "GAME_OVER" and action != "RESET":
        raise ToolCallError("The game is over; the only legal action now is RESET.")
    # RESET is a meta action, not one of the game's own moves: arcengine special-cases it ahead
    # of available_actions (base_game.py's step() checks `action_input.id == GameAction.RESET`
    # before consulting the list at all), so the SDK never includes it there and it is always
    # legal regardless of what this turn's available_actions says.
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

        target_description = args.get("target_description")
        if target_description is not None and (
            not isinstance(target_description, str) or not target_description.strip()
        ):
            raise ToolCallError("`target_description`, if given, must be a non-empty string.")
        # Not required here: native mode enforces it in loop._get_play_call, right before the
        # click-target check. Memo mode has no way to supply it and skips the check.

        return PlayAction(
            action=action, expectation=expectation, x=x, y=y, col=x // scale, row=y // scale,
            target_description=target_description,
        )

    if x is not None or y is not None:
        raise ToolCallError(f"`x`/`y` are only valid for the click-style action, not {action}.")
    if args.get("target_description") is not None:
        raise ToolCallError(f"`target_description` is only valid for the click-style action, not {action}.")
    return PlayAction(action=action, expectation=expectation)


@dataclasses.dataclass
class ReadNotesAction:
    file: str


@dataclasses.dataclass
class WriteNotesAction:
    file: str
    content: str
    more_writes_expected: bool = False  # True if another write_notes call is coming this stage
    step_check: Optional[str] = None  # required iff require_step_check (ACT's write_notes)


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


def write_notes_schema(description: str, allowed_files: tuple = NOTES_FILES,
                        require_step_check: bool = False) -> dict:
    """`allowed_files` narrows the schema's own `file` description to match whatever restriction
    `validate_write_notes_args` will actually enforce for this exposure -- see its docstring.
    `require_step_check` adds `step_check` to `required` -- set for ACT's exposure only."""
    required = ["file", "content"]
    if require_step_check:
        required.append("step_check")
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
                        "description": f"One of: {', '.join(allowed_files)}.",
                    },
                    "content": {
                        "type": "string",
                        "description": "The complete new contents of the file. Replaces it whole.",
                    },
                    "more_writes_expected": {
                        "type": "boolean",
                        "description": (
                            "True if you are about to write the other note file too, in this same step. "
                            "False (the default -- you can omit this) if this is your last write for now."
                        ),
                    },
                    "step_check": {
                        "type": "string",
                        "description": (
                            "Which `## Plan` step this marks done, and how the action you actually took "
                            "(and its verdict) matches what that step called for. Required when marking "
                            "a step done; leave it out otherwise."
                        ),
                    },
                },
                "required": required,
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


def validate_write_notes_args(args: dict, *, allowed_files: tuple = NOTES_FILES,
                               require_step_check: bool = False) -> WriteNotesAction:
    """`allowed_files` narrows which file this call may target: ACT passes `("WORKING.md",)`,
    since ACT may only mark a plan step done; VERIFY+PLAN and UPDATE may write either file.

    `require_step_check` (ACT only) requires `step_check`: which plan step this marks done, and how
    the action actually taken matches what that step called for.
    """
    file = _validate_notes_file(args, allowed_files=allowed_files)
    content = args.get("content")
    if not isinstance(content, str):
        raise ToolCallError("`content` is required and must be a string.")
    more_writes_expected = bool(args.get("more_writes_expected", False))
    step_check = args.get("step_check")
    if require_step_check and not (isinstance(step_check, str) and step_check.strip()):
        raise ToolCallError(
            "`step_check` is required: which plan step this marks done, and how your action matches it."
        )
    return WriteNotesAction(
        file=file, content=content, more_writes_expected=more_writes_expected, step_check=step_check,
    )


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
    registry_entry: Optional[str] = None  # named object/feature registry entry to check this crop against


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
                            "registry_entry": {
                                "type": "string",
                                "description": (
                                    "Optional: the name of an entry in your object/feature registry "
                                    "(WORKING.md) this crop is meant to check. Its recorded description "
                                    "is returned alongside the sampled grid."
                                ),
                            },
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
        registry_entry = v.get("registry_entry")
        if registry_entry is not None and (not isinstance(registry_entry, str) or not registry_entry.strip()):
            raise ToolCallError(f"view '{common['label']}': `registry_entry` must be a non-empty string.")
        views.append(ReadPixelsView(**common, rows=rows, columns=columns, registry_entry=registry_entry))
    return ReadPixelsAction(question=question, views=views)


@dataclasses.dataclass
class ObserveResult:
    description: str
    objective_reconfirmed: Optional[str] = None  # TERMINAL OBSERVE only
    goal_changed: Optional[bool] = None  # TERMINAL OBSERVE only


def observe_schema(description: str, terminal: bool = False) -> dict:
    """`terminal=True` builds the TERMINAL OBSERVE variant, used on the turn after a level
    completes: the model is shown the frames before and after the completing action, and must also
    give `objective_reconfirmed` / `goal_changed`."""
    properties = {
        "description": {
            "type": "string",
            "description": (
                "What you currently see, tied to the specific objects/features you're "
                "tracking (or a first read, if this is your first look at this level). "
                "Note anything new, missing, or different from what you had recorded."
            ),
        },
    }
    required = ["description"]
    if terminal:
        properties["description"]["description"] = (
            "What changed between the two frames you were just shown -- the one right before your "
            "last action, and the one right after it. Your strongest evidence for what winning "
            "actually looked like."
        )
        properties["objective_reconfirmed"] = {
            "type": "string",
            "description": "Restate what you now believe the goal is, using this evidence.",
        }
        properties["goal_changed"] = {
            "type": "boolean",
            "description": (
                "True if this evidence changes or sharpens your previous understanding of the "
                "goal, false if it just confirms what you already believed."
            ),
        }
        required += ["objective_reconfirmed", "goal_changed"]
    return {
        "type": "function",
        "function": {
            "name": "observe",
            "description": description,
            "parameters": {"type": "object", "properties": properties, "required": required},
        },
    }


def validate_observe_args(args: dict, *, terminal: bool = False) -> ObserveResult:
    description = args.get("description")
    if not isinstance(description, str) or not description.strip():
        raise ToolCallError("`description` is required: what you currently see.")
    if not terminal:
        return ObserveResult(description=description)

    objective_reconfirmed = args.get("objective_reconfirmed")
    if not isinstance(objective_reconfirmed, str) or not objective_reconfirmed.strip():
        raise ToolCallError(
            "`objective_reconfirmed` is required: restate what you now believe the goal is."
        )
    goal_changed = args.get("goal_changed")
    if not isinstance(goal_changed, bool):
        raise ToolCallError("`goal_changed` is required and must be true or false.")
    return ObserveResult(
        description=description, objective_reconfirmed=objective_reconfirmed, goal_changed=goal_changed,
    )


@dataclasses.dataclass
class ConfirmTargetResult:
    matches: bool
    reason: str
    inconclusive: bool = False  # set only by loop.py's fail-open path, never by the model itself


def confirm_target_schema() -> dict:
    """Schema for the click-target confirmation request. Not registered in `_SCHEMA_BUILDERS` or
    offered in the main conversation; `loop._confirm_click_target` uses it for its own separate
    request."""
    return {
        "type": "function",
        "function": {
            "name": "confirm_target",
            "description": "Report whether the cropped image shows the described target.",
            "parameters": {
                "type": "object",
                "properties": {
                    "matches": {
                        "type": "boolean",
                        "description": "true if the crop shows the described target, false if not.",
                    },
                    "reason": {
                        "type": "string",
                        "description": "One sentence explaining the answer either way.",
                    },
                },
                "required": ["matches", "reason"],
            },
        },
    }


def validate_confirm_target_args(args: dict) -> ConfirmTargetResult:
    matches = args.get("matches")
    if not isinstance(matches, bool):
        raise ToolCallError("`matches` is required and must be true or false.")
    reason = args.get("reason")
    if not isinstance(reason, str) or not reason.strip():
        raise ToolCallError("`reason` is required: one sentence explaining the answer.")
    return ConfirmTargetResult(matches=matches, reason=reason)


@dataclasses.dataclass
class VerifyAndPlanResult:
    verdict: str  # "confirmed" | "contradicted" | "uncertain"
    evidence: str
    mode: str  # "carry_forward" | "replan"
    observation_check: str  # every turn: checks `observe`'s description against WORKING.md
    objects_changed: bool  # every turn: did observation_check find something new/missing/changed
    objective_check: Optional[str] = None  # required iff TERMINAL OBSERVE ran this turn
    post_mortem: Optional[str] = None  # required iff state == "GAME_OVER" this turn
    note_check: Optional[str] = None  # required iff notes were written last turn
    phase: Optional[str] = None  # "explore" | "exploit", required iff mode == "replan"
    plan: Optional[str] = None  # required iff mode == "replan"
    rationale: Optional[str] = None  # required iff mode == "replan"


def verify_and_plan_schema(description: str) -> dict:
    return {
        "type": "function",
        "function": {
            "name": "verify_and_plan",
            "description": description,
            "parameters": {
                "type": "object",
                "properties": {
                    "verdict": {
                        "type": "string",
                        "enum": ["confirmed", "contradicted", "uncertain"],
                        "description": "Does this frame match what you expected after your last action?",
                    },
                    "evidence": {
                        "type": "string",
                        "description": (
                            "What you already reported in `observe` that supports this verdict -- point "
                            "to it, don't re-describe the frame."
                        ),
                    },
                    "observation_check": {
                        "type": "string",
                        "description": (
                            "Check `observe`'s description against what you've already recorded in "
                            "WORKING.md: does it account for every object/feature you're tracking? Note "
                            "anything missing, new, or changed."
                        ),
                    },
                    "objects_changed": {
                        "type": "boolean",
                        "description": (
                            "True if `observation_check` found something worth writing down -- a "
                            "missing, new, or changed object/feature. False if the description matches "
                            "what's already recorded."
                        ),
                    },
                    "objective_check": {
                        "type": "string",
                        "description": (
                            "Required when the turn's terminal `observe` call ran, right after a level "
                            "completed: does your plan still hold given its `objective_reconfirmed`/"
                            "`goal_changed` evidence, or does it need to change? Leave it out otherwise."
                        ),
                    },
                    "post_mortem": {
                        "type": "string",
                        "description": (
                            "Required when the game is over: why you think this happened, and what you "
                            "were attempting when it did."
                        ),
                    },
                    "note_check": {
                        "type": "string",
                        "description": (
                            "Required when you wrote notes last turn: does what you wrote still hold, "
                            "checked against what you see now? Leave it out otherwise."
                        ),
                    },
                    "mode": {
                        "type": "string",
                        "enum": ["carry_forward", "replan"],
                        "description": "carry_forward if the existing plan still holds; replan if it needs revising.",
                    },
                    "phase": {
                        "type": "string",
                        "enum": ["explore", "exploit"],
                        "description": "Required when replanning: still discovering rules, or executing a known solution?",
                    },
                    "plan": {
                        "type": "string",
                        "description": "Required when replanning: the plan itself, specific enough to act on.",
                    },
                    "rationale": {
                        "type": "string",
                        "description": "Required when replanning: why this plan, and why not something already tried without effect.",
                    },
                },
                "required": ["verdict", "evidence", "observation_check", "objects_changed", "mode"],
            },
        },
    }


def validate_verify_and_plan_args(
    args: dict, *, objective_check_required: bool,
    post_mortem_required: bool, note_check_required: bool,
) -> VerifyAndPlanResult:
    verdict = args.get("verdict")
    if verdict not in ("confirmed", "contradicted", "uncertain"):
        raise ToolCallError("`verdict` must be one of confirmed, contradicted, uncertain.")
    evidence = args.get("evidence")
    if not isinstance(evidence, str) or not evidence.strip():
        raise ToolCallError("`evidence` is required: point to what you already observed that supports the verdict.")

    observation_check = args.get("observation_check")
    if not isinstance(observation_check, str) or not observation_check.strip():
        raise ToolCallError(
            "`observation_check` is required: check `observe`'s description against WORKING.md."
        )
    objects_changed = args.get("objects_changed")
    if not isinstance(objects_changed, bool):
        raise ToolCallError("`objects_changed` is required and must be true or false.")

    objective_check = args.get("objective_check")
    if objective_check_required and not (
        isinstance(objective_check, str) and objective_check.strip()
    ):
        raise ToolCallError(
            "The turn's terminal `observe` call ran: `objective_check` is required."
        )

    post_mortem = args.get("post_mortem")
    if post_mortem_required and not (isinstance(post_mortem, str) and post_mortem.strip()):
        raise ToolCallError("The game is over: `post_mortem` is required.")

    note_check = args.get("note_check")
    if note_check_required and not (isinstance(note_check, str) and note_check.strip()):
        raise ToolCallError("You wrote notes last turn: `note_check` is required.")

    mode = args.get("mode")
    if mode not in ("carry_forward", "replan"):
        raise ToolCallError("`mode` must be 'carry_forward' or 'replan'.")

    if mode == "carry_forward":
        return VerifyAndPlanResult(
            verdict=verdict, evidence=evidence, mode=mode,
            observation_check=observation_check, objects_changed=objects_changed,
            objective_check=objective_check,
            post_mortem=post_mortem, note_check=note_check,
        )

    phase = args.get("phase")
    if phase not in ("explore", "exploit"):
        raise ToolCallError("Replanning requires `phase`: 'explore' or 'exploit'.")
    plan = args.get("plan")
    if not isinstance(plan, str) or not plan.strip():
        raise ToolCallError("Replanning requires `plan`: specific enough to act on.")
    rationale = args.get("rationale")
    if not isinstance(rationale, str) or not rationale.strip():
        raise ToolCallError("Replanning requires `rationale`: why this plan, not just what it is.")

    return VerifyAndPlanResult(
        verdict=verdict, evidence=evidence, mode=mode,
        observation_check=observation_check, objects_changed=objects_changed,
        objective_check=objective_check,
        post_mortem=post_mortem, note_check=note_check, phase=phase, plan=plan, rationale=rationale,
    )


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
    "observe": observe_schema,
    "verify_and_plan": verify_and_plan_schema,
    "history": history_schema,
}


def build_schemas(tool_descs: dict, active_names: list) -> list:
    """One OpenAI-style function schema per active tool, in manifest order."""
    return [_SCHEMA_BUILDERS[name](tool_descs[name]) for name in active_names]

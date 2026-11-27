"""Loads `config/default.yaml` (plus an optional override file) into typed config objects.

`max_px` -- the one prompt variable sourced from config (`prompts/manifest.yaml`:
`max_px: {source: config}`) -- is derived as `image.scale * 64 - 1` rather than stored as its own
YAML key, so changing `scale` cannot desynchronise the two.

Unknown YAML keys are dropped silently (see `_filtered`), so a misspelt key is a no-op, not an
error.
"""
import dataclasses
import os
from pathlib import Path
from typing import Any, Optional, Union

import yaml

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent / "config" / "default.yaml"


@dataclasses.dataclass
class ModelConfig:
    provider: str = "openai_compatible"
    base_url: str = "http://localhost:8000/v1"
    model: str = ""
    api_key_env: str = "VLLM_API_KEY"
    max_context_tokens: int = 32768
    max_output_tokens: int = 2048
    temperature: float = 0.7
    supports_images: bool = True
    # "native" (tool calls) or "memo" (a plain-text fallback for models whose tool calling is
    # unreliable; see agent/memo.py).
    tool_mode: str = "native"
    # Optional sampling controls. None means "don't send it", so the server's own default applies.
    # `reasoning_effort` is model-specific; llm.py forwards it via chat_template_kwargs.
    top_p: Optional[float] = None
    reasoning_effort: Optional[str] = None
    # Per-request HTTP timeout and retry count for the openai client. A thinking model can generate
    # up to max_output_tokens in one turn, which on a shared server can take well over the SDK's
    # 600s default. A retry regenerates the reply from scratch.
    request_timeout_seconds: float = 1800.0
    max_retries: int = 10


@dataclasses.dataclass
class ImageConfig:
    scale: int = 8
    gridline_brightness: float = 0.75
    confirm_crop_px: int = 128  # size of the crop used to confirm a click's target


@dataclasses.dataclass
class FramesConfig:
    archive_png: bool = True
    max_inspect_views: int = 7


@dataclasses.dataclass
class RunConfig:
    action_budget_multiplier: int = 5
    # Free tool calls in the ACT stage before `play` is required.
    max_tool_calls_per_turn: int = 12
    # Free tool calls inside one of OBSERVE / VERIFY+PLAN / UPDATE before its terminal call is
    # required.
    max_tool_calls_per_stage: int = 30
    max_protocol_retries: int = 5
    max_turns: int = 300
    max_resets_per_level: int = 10
    max_total_actions: int = 2000
    # Real-time cap on a run, in seconds. None disables it.
    max_wall_clock_seconds: Optional[float] = None
    # memory/stagnation.py's RepetitionTracker: how far back a new repeated pattern can be
    # recognized, and how many repeats before it is reported.
    stagnation_window: int = 6
    stagnation_min_repeats: int = 3


@dataclasses.dataclass
class TraceConfig:
    log_thinking: bool = False  # write the model's thinking-mode output to trace.jsonl
    # Diagnostic: also record the exact, unparsed JSON response body for each turn. Useful for
    # debugging provider-specific response fields; can make trace.jsonl large.
    log_raw_response: bool = False


@dataclasses.dataclass
class PathsConfig:
    # Where the downloaded game environments live. None -> env.py's default,
    # `<repo>/environment_files`.
    environments_dir: Optional[str] = None


@dataclasses.dataclass
class SessionConfig:
    """How the conversation is kept within the model's context window (see memory/session.py).

    The defaults are conservative starting points, not tuned values; adjust them for your model and
    server.
    """

    # High-water mark: once a reply's token count crosses this, a cut is made before the next turn.
    compact_at_frac: float = 0.75  # of model.max_context_tokens
    # Low-water mark a cut trims down to. Treated as an upper bound: session.py clamps it further so
    # that the space left after a cut always fits one worst-case turn (cuts never happen mid-turn).
    target_after_cut_frac: float = 0.55

    # Hard image budgets. `max_images_in_context` must stay below the server's per-request image
    # limit (vLLM's --limit-mm-per-prompt) with room for one whole turn; exceeding it fails the
    # request. `max_images_per_turn` caps what a single turn (observation + `inspect` views) may add.
    max_images_in_context: int = 100
    max_images_per_turn: int = 10

    # Retention windows: the newest `keep_full_turns` turn records stay verbatim, the next
    # `keep_text_turns` keep their text with images replaced by a placeholder, and older ones are
    # folded into the digest.
    keep_full_turns: int = 8
    keep_text_turns: int = 40

    # Digest size before the oldest levels collapse to one summary line each.
    digest_max_chars: int = 6000
    # Per-file cap on the notes copied into the digest message at a cut.
    notes_push_max_chars: int = 3500

    # Token-estimator constants. Cuts are decided before a request is sent, so the harness estimates
    # the size of a message list it has not sent yet. Over-estimating is the safe direction.
    tokens_per_image: int = 1024
    chars_per_token: float = 3.5  # conservative; English prose is nearer 4


@dataclasses.dataclass
class Config:
    model: ModelConfig = dataclasses.field(default_factory=ModelConfig)
    image: ImageConfig = dataclasses.field(default_factory=ImageConfig)
    frames: FramesConfig = dataclasses.field(default_factory=FramesConfig)
    run: RunConfig = dataclasses.field(default_factory=RunConfig)
    trace: TraceConfig = dataclasses.field(default_factory=TraceConfig)
    paths: PathsConfig = dataclasses.field(default_factory=PathsConfig)
    session: SessionConfig = dataclasses.field(default_factory=SessionConfig)

    @property
    def image_px(self) -> int:
        return self.image.scale * 64

    @property
    def max_px(self) -> int:
        return self.image_px - 1

    def api_key(self) -> str:
        return os.environ.get(self.model.api_key_env, "")


def _filtered(cls, data: dict) -> dict:
    return {k: v for k, v in data.items() if k in cls.__dataclass_fields__}


def load_config(path: Optional[Union[str, Path]] = None) -> Config:
    """`config/default.yaml`, with the file at `path` (if given) deep-merged over it."""
    raw: dict[str, Any] = {}
    if DEFAULT_CONFIG_PATH.exists():
        raw = yaml.safe_load(DEFAULT_CONFIG_PATH.read_text(encoding="utf-8")) or {}
    if path is not None:
        override = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        raw = _deep_merge(raw, override)

    model = raw.get("model") or {}
    image = (raw.get("observation") or {}).get("image") or {}
    frames = (raw.get("memory") or {}).get("frames") or {}
    run = raw.get("run") or {}
    trace = raw.get("trace") or {}
    paths = raw.get("paths") or {}
    session = raw.get("session") or {}

    return Config(
        model=ModelConfig(**_filtered(ModelConfig, model)),
        image=ImageConfig(**_filtered(ImageConfig, image)),
        frames=FramesConfig(**_filtered(FramesConfig, frames)),
        run=RunConfig(**_filtered(RunConfig, run)),
        trace=TraceConfig(**_filtered(TraceConfig, trace)),
        paths=PathsConfig(**_filtered(PathsConfig, paths)),
        session=SessionConfig(**_filtered(SessionConfig, session)),
    )


def _deep_merge(base: dict, override: dict) -> dict:
    out = dict(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out

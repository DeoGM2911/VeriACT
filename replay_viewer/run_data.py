"""Parses a harness run directory (`runs/<run_id>/`) into plain dataclasses ready to render.

Reads only on-disk artifacts a run directory already contains -- `trace.jsonl`,
`frames/manifest.json` + the archived PNGs, `GUIDE.md`/`WORKING.md`/`working/level_*.md`,
`run_summary.json` -- never harness Python modules. The schema is the one written by
`<arm>/evaluation/tracelog.py` and `<arm>/agent/env.py` in either harness.
"""
import base64
import io
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from PIL import Image

_PROGRESS_RE = re.compile(r"Progress:\s*(\d+)\s+of\s+(\d+)\s+levels complete")
_READ_PIXELS_PALETTE_RE = re.compile(r"([A-Za-z])=(\d+)")


@dataclass
class ManifestRow:
    ref: str
    frame: int
    label: str
    file: str
    level: int
    turn_in_level: int
    attempt: int
    action: Optional[str]
    state: str


def load_manifest(run_dir: Path) -> dict:
    """`ref -> [ManifestRow, ...]`, sorted by `.frame` ascending. `{}` if no archive exists."""
    path = run_dir / "frames" / "manifest.json"
    if not path.exists():
        return {}
    rows = json.loads(path.read_text(encoding="utf-8"))
    by_ref: dict = {}
    for r in rows:
        by_ref.setdefault(r["ref"], []).append(ManifestRow(**r))
    for ref in by_ref:
        by_ref[ref].sort(key=lambda row: row.frame)
    return by_ref


def _data_uri(path: Path) -> Optional[str]:
    try:
        return "data:image/png;base64," + base64.b64encode(path.read_bytes()).decode("ascii")
    except OSError:
        return None


def settled_data_uri(run_dir: Path, by_ref: dict, ref: str) -> Optional[str]:
    """The frame the model actually saw for `ref` -- the highest-`frame`-index row, i.e.
    `step.frames[-1]` in harness terms. Embeds the archived PNG bytes as-is, no re-encode."""
    rows = by_ref.get(ref)
    if not rows:
        return None
    return _data_uri(run_dir / "frames" / rows[-1].file)


def resolve_view_file(by_ref: dict, turn: int, frame: int) -> Optional[str]:
    rows = by_ref.get(f"T{turn}")
    if not rows:
        return None
    for row in rows:
        if row.frame == frame:
            return row.file
    return None


def crop_data_uri(run_dir: Path, by_ref: dict, turn: int, frame: int, region: Optional[dict]) -> Optional[str]:
    """Reconstructs what an `inspect` view showed the model: the archived frame for
    `(turn, frame)`, cropped to `region` if given. The harness never persisted the images
    `inspect` actually returned (they only existed as a transient follow-up message), but its own
    rendering (`region_to_png_bytes`) is a full re-render then crop at the same scale -- cropping
    the already-archived PNG here reproduces the same view exactly. Returns None (never raises) on
    any failure, so one malformed view doesn't abort the whole build."""
    filename = resolve_view_file(by_ref, turn, frame)
    if filename is None:
        return None
    try:
        img = Image.open(run_dir / "frames" / filename)
        if region:
            box = (region["x"], region["y"], region["x"] + region["width"], region["y"] + region["height"])
            img = img.crop(box)
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")
    except Exception:
        return None


def _parse_read_pixels_result(tc_arguments: dict, result: str) -> list:
    """Pulls the sampled grid + per-call palette back out of a `read_pixels` call's stored
    `result` text. `arguments` gives `rows`/`columns` per view up front, so exactly how many
    lines to consume is known without parsing the Python-dict-repr `region {...}` header text --
    that header line is simply skipped, never parsed."""
    question = tc_arguments.get("question", "")
    prefix = f"read_pixels: {question}\n"
    body = result[len(prefix):] if result.startswith(prefix) else result
    lines = body.split("\n")
    ptr = 0
    out = []
    for v in tc_arguments.get("views", []):
        rows = v.get("rows")
        if not isinstance(rows, int) or ptr >= len(lines):
            break
        ptr += 1  # skip "[label] turn N, frame M[, region {...}], RxC:"
        grid = lines[ptr: ptr + rows]
        ptr += rows
        palette_line = lines[ptr] if ptr < len(lines) else ""
        ptr += 1
        palette = {m.group(1): int(m.group(2)) for m in _READ_PIXELS_PALETTE_RE.finditer(palette_line)}
        out.append({
            "label": v.get("label", ""), "turn": v.get("turn"), "frame": v.get("frame", 0),
            "region": v.get("region"), "rows": rows, "columns": v.get("columns"),
            "grid": grid, "palette": palette,
        })
    return out


@dataclass
class ToolCallRecord:
    name: str
    arguments: dict
    result: str
    views: list = field(default_factory=list)  # name-specific rendering payload; see _enrich_tool_call


def _enrich_tool_call(run_dir: Path, by_ref: dict, tc: dict) -> ToolCallRecord:
    name = tc.get("name", "")
    arguments = tc.get("arguments") or {}
    result = tc.get("result", "")
    views: list = []
    if name == "inspect":
        for v in arguments.get("views", []):
            turn, frame = v.get("turn"), v.get("frame", 0)
            data_uri = crop_data_uri(run_dir, by_ref, turn, frame, v.get("region")) if isinstance(turn, int) else None
            views.append({
                "label": v.get("label", ""), "turn": turn, "frame": frame,
                "region": v.get("region"), "data_uri": data_uri,
            })
    elif name == "read_pixels":
        views = _parse_read_pixels_result(arguments, result)
    return ToolCallRecord(name=name, arguments=arguments, result=result, views=views)


def _turn_usage(action_input: dict) -> dict:
    """`action_input.reasoning` is a JSON-*encoded string* (double-encoded by
    tracelog.py) -- json.loads() it again to
    reach `usage`. `model_output` at the record's top level already has the plain text, so this is
    only needed for the token-count side info."""
    raw = action_input.get("reasoning")
    if not raw:
        return {}
    try:
        return json.loads(raw).get("usage", {})
    except (TypeError, ValueError):
        return {}


def _extract_win_levels(turn_message: str) -> Optional[int]:
    """`win_levels` isn't a structured trace field (see tracelog.py) -- only the rendered
    `turn.md` text carries it ("Progress: N of M levels complete"). Best-effort only."""
    m = _PROGRESS_RE.search(turn_message)
    return int(m.group(2)) if m else None


@dataclass
class Turn:
    index: int                      # 1-based, over non-compaction trace records only
    before_ref: str
    after_ref: str
    game_id: str
    state: str
    levels_completed: int
    win_levels: Optional[int]
    full_reset: bool
    available_actions: list
    action_id: Optional[str]        # None on protocol_failure
    click: Optional[dict]           # {x, y, row, col} or None
    reasoning: str                  # model_output -- the full turn memo
    usage: dict
    expectation: Optional[str]      # [M4] nullable
    reported_change: str            # [M4] "" if absent (older runs, or nothing narrated)
    frame_changed: Optional[bool]   # harness-computed: settled frame differs from last turn's; None on older runs
    tool_calls: list                # list[ToolCallRecord]
    protocol_failure: Optional[str]
    step_in_level: int
    attempt_in_level: int
    thinking: Optional[str] = None
    raw_response: Optional[dict] = None
    before_data_uri: Optional[str] = None
    after_data_uri: Optional[str] = None


@dataclass
class CompactionEvent:
    turn: int
    tokens_before: int
    continuation: str


@dataclass
class NotesBundle:
    guide: Optional[str]
    working: Optional[str]
    working_by_level: dict  # int level -> str content


def load_notes(run_dir: Path) -> NotesBundle:
    def _read(name):
        p = run_dir / name
        return p.read_text(encoding="utf-8") if p.exists() else None

    by_level: dict = {}
    working_dir = run_dir / "working"
    if working_dir.is_dir():
        for p in sorted(working_dir.glob("level_*.md")):
            try:
                level = int(p.stem.split("_", 1)[1])
            except (IndexError, ValueError):
                continue
            by_level[level] = p.read_text(encoding="utf-8")

    return NotesBundle(guide=_read("GUIDE.md"), working=_read("WORKING.md"), working_by_level=by_level)


def load_run_summary(run_dir: Path) -> Optional[dict]:
    p = run_dir / "run_summary.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def parse_trace(run_dir: Path, by_ref: dict) -> list:
    """Returns `list[Turn | CompactionEvent]`, in file order.

    `before_ref`/`after_ref` chain off each record's own `frame_ref` field rather than being
    assumed as `f"T{index}"`/`f"T{index+1}"` by position: on a `protocol_failure` turn no
    `env.act()` happened, so `frame_ref` is unchanged from the previous turn (confirmed against
    env.py's `_finalize` and real trace data) -- position-based numbering would silently point one
    frame ahead on any run containing a protocol failure.
    """
    trace_path = run_dir / "trace.jsonl"
    if not trace_path.exists():
        return []

    timeline: list = []
    prev_after_ref = "T1"
    index = 0
    for line in trace_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        rec = json.loads(line)

        if rec.get("event") == "compaction":
            timeline.append(CompactionEvent(
                turn=rec.get("turn", 0),
                tokens_before=rec.get("tokens_before", 0),
                continuation=rec.get("continuation", ""),
            ))
            continue

        index += 1
        data = rec.get("data", {})
        action_input = data.get("action_input", {})
        turn_message = rec.get("turn_message", "")
        before_ref = prev_after_ref
        after_ref = rec.get("frame_ref", prev_after_ref)

        timeline.append(Turn(
            index=index,
            before_ref=before_ref,
            after_ref=after_ref,
            game_id=data.get("game_id", ""),
            state=data.get("state", ""),
            levels_completed=data.get("levels_completed", 0),
            win_levels=_extract_win_levels(turn_message),
            full_reset=data.get("full_reset", False),
            available_actions=data.get("available_actions", []),
            action_id=action_input.get("id"),
            click=rec.get("click"),
            reasoning=rec.get("model_output", ""),
            usage=_turn_usage(action_input),
            expectation=rec.get("expectation"),
            reported_change=rec.get("reported_change", ""),
            frame_changed=rec.get("frame_changed"),
            tool_calls=[_enrich_tool_call(run_dir, by_ref, tc) for tc in rec.get("tool_calls", [])],
            protocol_failure=rec.get("protocol_failure"),
            step_in_level=rec.get("step_in_level", 0),
            attempt_in_level=rec.get("attempt_in_level", 0),
            thinking=rec.get("thinking"),
            raw_response=rec.get("raw_response"),
            before_data_uri=settled_data_uri(run_dir, by_ref, before_ref),
            after_data_uri=settled_data_uri(run_dir, by_ref, after_ref),
        ))
        prev_after_ref = after_ref

    return timeline


@dataclass
class RunData:
    run_dir: Path
    run_id: str
    timeline: list  # list[Turn | CompactionEvent]
    notes: NotesBundle
    run_summary: Optional[dict]


def load_run(run_dir) -> RunData:
    run_dir = Path(run_dir)
    by_ref = load_manifest(run_dir)
    return RunData(
        run_dir=run_dir,
        run_id=run_dir.name,
        timeline=parse_trace(run_dir, by_ref),
        notes=load_notes(run_dir),
        run_summary=load_run_summary(run_dir),
    )

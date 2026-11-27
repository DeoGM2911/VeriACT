"""The session digest: what a cut leaves behind in place of the turns it evicted.

No model is asked to summarize anything. Every line is assembled mechanically from fields
`tracelog.py` already writes, so the digest can only report things that actually happened, a cut
costs no model call, and the digest for any completed run can be rebuilt offline from its
`trace.jsonl` (`facts_from_trace`).

The digest carries facts only: what was done, what was expected, what was reported. The model's
interpretation reaches the post-cut context by two other routes, both verbatim copies: the notes
files, copied into the digest message at the cut, and the model's most recent change narration
(`last_description`). The caller assembles those and renders everything through
`prompts/digest.md`; this module provides only the per-turn record.
"""
import dataclasses
import json
from pathlib import Path
from typing import Optional

_ELLIPSIS = "…"


@dataclasses.dataclass
class TurnFacts:
    """One turn, reduced to the fields the digest reports. Every one is copied from a trace record;
    none is derived by judgement.

    `level` is the level the turn was *played on*, not the level count after it -- a turn that
    completes a level was still played on the level it completed, and grouping it under the next
    one would misattribute every action it took. `level_completed` carries the transition instead.
    """
    turn: int
    level: int
    action: Optional[str] = None
    x: Optional[int] = None
    y: Optional[int] = None
    expectation: Optional[str] = None
    reported_change: str = ""
    frame_changed: Optional[bool] = None
    state: str = ""
    level_completed: bool = False
    protocol_failure: Optional[str] = None


def facts_from_trace(trace_path) -> list:
    """Rebuild `TurnFacts` from a harness `trace.jsonl`.

    Like `scoring.levels_from_trace`, this works offline from the trace alone, so the digest a run
    sent can be reproduced (and audited) from its own log.

    Non-turn rows (a cut event, which carries `event` instead of `data`) are skipped.
    """
    facts = []
    level = 0
    for line in Path(trace_path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if "data" not in row:
            continue
        data = row["data"]
        click = row.get("click") or {}
        after = data.get("levels_completed", level)
        facts.append(TurnFacts(
            turn=row.get("turn"),
            level=level,  # the level this turn was played on -- see TurnFacts' docstring
            action=(data.get("action_input") or {}).get("id"),
            x=click.get("x"),
            y=click.get("y"),
            expectation=row.get("expectation"),
            reported_change=row.get("reported_change") or "",
            frame_changed=row.get("frame_changed"),
            state=data.get("state", ""),
            level_completed=after != level,
            protocol_failure=row.get("protocol_failure"),
        ))
        level = after
    return facts


def _clip(text: Optional[str], limit: int) -> str:
    text = " ".join((text or "").split())  # collapse newlines: one turn is one line
    if len(text) <= limit:
        return text
    return text[: max(limit - 1, 0)].rstrip() + _ELLIPSIS


def turn_line(fact: TurnFacts, field_chars: int) -> str:
    """One turn, one line. Only fields that carry something are emitted, so a quiet turn stays
    short and an eventful one is self-explanatory without a legend."""
    action = fact.action or "(no action)"
    if fact.x is not None:
        action = f"{action} at ({fact.x},{fact.y})"
    parts = [f"T{fact.turn}: {action}"]
    if fact.expectation:
        parts.append(f"expected {_clip(fact.expectation, field_chars)}")
    if fact.reported_change:
        parts.append(f"then reported {_clip(fact.reported_change, field_chars)}")
    if fact.frame_changed is False:
        parts.append("frame unchanged")
    if fact.state and fact.state not in ("NOT_FINISHED", "NOT_PLAYED"):
        parts.append(fact.state)
    if fact.protocol_failure:
        parts.append(f"protocol failure: {fact.protocol_failure}")
    if fact.level_completed:
        parts.append("**level completed**")
    return "- " + " | ".join(parts)


def _group_by_level(facts: list) -> list:
    """`[(level, [facts])]`, in play order. Not a dict keyed by level: a level can be revisited
    after a `RESET`, and merging those stretches would imply a continuity the run did not have."""
    groups: list = []
    for fact in facts:
        if not groups or groups[-1][0] != fact.level:
            groups.append((fact.level, []))
        groups[-1][1].append(fact)
    return groups


def _level_summary(level: int, facts: list) -> str:
    """The collapsed form of a level, used oldest-first once the digest outgrows its ceiling.
    Counts only -- every count is of records actually present, nothing inferred."""
    resets = sum(1 for f in facts if f.action == "RESET")
    completed = any(f.level_completed for f in facts)
    turns = [f.turn for f in facts if f.turn is not None]
    span = f"T{turns[0]}-T{turns[-1]}" if turns else "no turns"
    bits = [f"{len(facts)} turns ({span})"]
    if resets:
        bits.append(f"{resets} reset(s)")
    bits.append("completed" if completed else "not completed")
    return f"### Level {level} — " + ", ".join(bits)


def render_digest(facts: list, max_chars: int = 6000, field_chars: int = 90) -> str:
    """The per-turn record, grouped by level, bounded by `max_chars`.

    Degradation is mechanical and predictable, never a second summarization prompt: when the digest
    outgrows its ceiling, the **oldest** level collapses to a single summary line, then the next
    oldest, and so on. If every level is collapsed and it still does not fit, the oldest summary
    lines are dropped and a line says so -- the recent past is what a turn is most likely to need,
    so it is what survives.
    """
    if not facts:
        return ""
    groups = _group_by_level(facts)
    collapsed = 0

    def render(n_collapsed: int) -> str:
        blocks = []
        for i, (level, level_facts) in enumerate(groups):
            if i < n_collapsed:
                blocks.append(_level_summary(level, level_facts))
            else:
                lines = [f"### Level {level}"]
                lines += [turn_line(f, field_chars) for f in level_facts]
                blocks.append("\n".join(lines))
        return "\n\n".join(blocks)

    text = render(collapsed)
    while len(text) > max_chars and collapsed < len(groups):
        collapsed += 1
        text = render(collapsed)

    if len(text) > max_chars:
        # Everything is a summary line and it still does not fit: drop from the oldest end.
        lines = text.split("\n\n")
        while len(lines) > 1 and len("\n\n".join(lines)) > max_chars:
            lines.pop(0)
        text = "(earlier levels omitted for length)\n\n" + "\n\n".join(lines)
    return text

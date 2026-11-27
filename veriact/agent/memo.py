"""Parses the `tool_mode: memo` reply format into the same argument shapes `tools.py`
validates: this module recognises the format, `tools.py` still decides whether the result is
legal. Memo mode is a fallback for models whose tool calling is unreliable.

Format:

    <free text: expectation and reasoning>

    ## GUIDE
    <the complete new contents of GUIDE.md>

    ## WORKING
    <the complete new contents of WORKING.md>

    ACTION6 344 456

Both section headers are optional -- omitting one means "leave that file as it is this turn",
mirroring how `write_notes` is voluntary in native mode. The last non-blank line must be exactly
one action, alone on its own line, with two numbers only for the click-style action; anything else
there is a parse failure the caller re-prompts against, exactly like an invalid native tool call.

Only `play` and the notes are supported; there is no text grammar for `inspect` / `read_pixels`.
"""
import dataclasses
import re
from typing import Optional

_ACTION_LINE = re.compile(r"^\s*(RESET|ACTION[1-7])\s*(?:(-?\d+)\s+(-?\d+))?\s*$", re.IGNORECASE)
_SECTION_HEADER = re.compile(r"^\s*##\s*(GUIDE|WORKING)\s*$", re.IGNORECASE)
_EXPECTATION_LINE = re.compile(r"^\s*Expectation:\s*(.*)$", re.IGNORECASE)

_SECTION_FILES = {"GUIDE": "GUIDE.md", "WORKING": "WORKING.md"}


@dataclasses.dataclass
class ParsedMemo:
    reported_change: str  # free text before any section header or the action line
    play_args: Optional[dict]  # None if the reply didn't end in a recognisable action line
    notes: dict  # {"GUIDE.md": content, ...} -- only files this turn's reply actually included


def parse(text: str) -> ParsedMemo:
    lines = (text or "").splitlines()
    while lines and not lines[-1].strip():  # trailing blank lines don't count
        lines.pop()

    action_match = _ACTION_LINE.match(lines[-1]) if lines else None
    body_lines = lines[:-1] if action_match else lines

    notes: dict = {}
    free_text_lines: list = []
    current_section: Optional[str] = None
    section_lines: list = []

    def flush_section():
        if current_section is not None:
            notes[_SECTION_FILES[current_section]] = "\n".join(section_lines).strip()

    for line in body_lines:
        header_match = _SECTION_HEADER.match(line)
        if header_match:
            flush_section()
            current_section = header_match.group(1).upper()
            section_lines = []
            continue
        (free_text_lines if current_section is None else section_lines).append(line)
    flush_section()

    reported_change = "\n".join(free_text_lines).strip()

    play_args = None
    if action_match:
        play_args = {"action": action_match.group(1).upper()}
        if action_match.group(2) is not None:
            play_args["x"] = int(action_match.group(2))
            play_args["y"] = int(action_match.group(3))
        expectation = ""
        for line in free_text_lines:
            m = _EXPECTATION_LINE.match(line)
            if m:
                expectation = m.group(1).strip()
                break
        play_args["expectation"] = expectation

    return ParsedMemo(reported_change=reported_change, play_args=play_args, notes=notes)

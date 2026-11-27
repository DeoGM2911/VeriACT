"""Renders `prompts/` into the strings the model sees.

Two rules, the same ones `scripts/check_prompts.py` validates: templates use plain `{{var}}`
substitution with no logic, and a block (text between blank lines, not a markdown section) whose
variables are *all* unavailable is omitted entirely rather than rendering a hole. A block that
mixes available and unavailable variables is a bug and raises `PromptRenderError` rather than
sending `{{var}}` literally to the model.

Manifest entries carry a `since` version tag; `PromptSet` renders only the entries at or before
the version it was constructed with (see `MILESTONES`).
"""
import re
from pathlib import Path
from typing import Any

import yaml

MILESTONES = ["M1", "M2", "M3", "M4", "M5", "M6", "M7", "M8", "M9", "M10", "M11", "M12", "M13", "M14"]
_VAR_RE = re.compile(r"\{\{(\w+)\}\}")


class PromptRenderError(Exception):
    """A surviving (non-fully-dropped) block referenced a variable with no value."""


def at_or_before(since: str, milestone: str) -> bool:
    return MILESTONES.index(since) <= MILESTONES.index(milestone)


def load_manifest(prompts_dir: Path) -> dict:
    return yaml.safe_load((prompts_dir / "manifest.yaml").read_text(encoding="utf-8"))


def render_template(path: Path, variables: dict) -> str:
    text = path.read_text(encoding="utf-8")
    available = variables.keys()
    kept = []
    for block in re.split(r"\n\s*\n", text):
        names = set(_VAR_RE.findall(block))
        if names and not (names & available):
            continue  # every variable in this block is unavailable -- the whole block drops
        missing = names - available
        if missing:
            raise PromptRenderError(f"{path}: unfilled variable(s) {sorted(missing)}")
        kept.append(_VAR_RE.sub(lambda m: str(variables[m.group(1)]), block))
    return "\n\n".join(kept).strip()


class PromptSet:
    """A loaded `manifest.yaml`, rendering for one fixed version tag."""

    def __init__(self, prompts_dir: Path, milestone: str):
        self.dir = Path(prompts_dir)
        self.milestone = milestone
        self.manifest = load_manifest(self.dir)

    def _active(self, entries: dict) -> dict:
        return {k: v for k, v in entries.items() if at_or_before(v["since"], self.milestone)}

    def system_prompt(self, variables: dict) -> str:
        parts = [p for p in self.manifest["system"] if at_or_before(p["since"], self.milestone)]
        return "\n\n".join(render_template(self.dir / p["file"], variables) for p in parts)

    def turn_message(self, variables: dict) -> str:
        return render_template(self.dir / self.manifest["turn"], variables)

    def digest_message(self, variables: dict) -> str:
        """The single user message a cut leaves in place of the turns it evicted."""
        return render_template(self.dir / self.manifest["digest"]["file"], variables)

    def frame_placeholder(self, variables: dict) -> str:
        """What an evicted image leaves behind inside an older turn record."""
        return render_template(self.dir / self.manifest["frame_placeholder"]["file"], variables)

    # -- stage-specific prompt pieces (see prompts/manifest.yaml) ----------------------------

    def observe_terminal_description(self, variables: dict) -> str:
        return render_template(self.dir / self.manifest["observe_terminal"]["file"], variables)

    def click_confirm_prompt(self, variables: dict) -> str:
        return render_template(self.dir / self.manifest["click_confirm"]["file"], variables)

    def stage_tools_reminder(self, variables: dict) -> str:
        """Rendered fresh right before each stage's own model exchange."""
        return render_template(self.dir / self.manifest["stage_tools"]["file"], variables)

    def memo_system_prompt(self, variables: dict) -> str:
        """`tool_mode: memo` front end: rendered in full, not composed from `since`-gated parts."""
        return render_template(self.dir / self.manifest["memo"]["system"], variables)

    def memo_turn_message(self, variables: dict) -> str:
        return render_template(self.dir / self.manifest["memo"]["turn"], variables)

    def tool_descriptions(self, variables: dict) -> dict:
        return {name: render_template(self.dir / t["file"], variables) for name, t in self._active(self.manifest["tools"]).items()}

    def active_tool_names(self) -> list:
        return list(self._active(self.manifest["tools"]).keys())

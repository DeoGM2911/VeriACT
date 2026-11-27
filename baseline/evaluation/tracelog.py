"""The run log: one JSON record per line in `runs/<id>/trace.jsonl`.

Two kinds of record:

- **Turn records** (have a `data` key), one per turn. `data` follows the SDK's own recording
  layout -- `data.{game_id, guid, state, levels_completed, full_reset, available_actions}` and
  `data.action_input.{id, data.x, data.y, reasoning}` -- where `reasoning` is a JSON-encoded
  *string* `'{"output": "...", "usage": {...}}'`, as in SDK recordings. Harness-specific fields
  sit beside `data`:
  - `turn`, `turn_message`, `model_output`, `tool_calls`, `click`, `frame_ref`,
    `step_in_level`, `attempt_in_level`, `protocol_failure`;
  - `expectation` -- the model's prediction for this turn's action (`play.expectation`);
  - `reported_change` -- the model's narration this turn, written after seeing the previous
    action's result, so it reports on the *previous* turn's expectation. Pair
    `trace[N-1].expectation` with `trace[N].reported_change` to compare prediction with outcome;
    the harness does not judge agreement itself;
  - `frame_changed` -- whether any frame the action returned differs from the one shown before
    it (`None` on a protocol-failure record);
  - `turn_prompt_tokens` / `turn_completion_tokens` -- summed over every model call this turn,
    retries and tool round-trips included (`usage` inside `reasoning` covers only the final
    `play` call);
  - `thinking` / `raw_response` -- only when `trace.log_thinking` / `trace.log_raw_response`
    are on.
- **Cut records** (`"event": "cut"`), written when the context is trimmed; see
  `session.CutRecord` for the fields.

`scoring.levels_from_trace` and `digest.facts_from_trace` both rebuild their inputs from this file
alone.
"""
import json
from pathlib import Path
from typing import Optional


class TraceLog:
    def __init__(self, path, log_thinking: bool = False, log_raw_response: bool = False):
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = open(self._path, "a", encoding="utf-8")
        self._log_thinking = log_thinking
        self._log_raw_response = log_raw_response

    def log_turn(
        self,
        *,
        game_id: str,
        play,  # tools.PlayAction, or None on a protocol failure
        step,  # env.StepResult
        turn: Optional[int] = None,  # the loop's 1-indexed turn number
        frame_changed: Optional[bool] = None,  # stagnation.sequence_changed for this action
        turn_message: str,
        model_output: str,
        tool_calls: list,
        reasoning_text: str,
        reported_change: str = "",  # this turn's narration of what changed; see module docstring
        usage: dict,
        turn_prompt_tokens: int = 0,  # every send() this turn, not just play's; see module docstring
        turn_completion_tokens: int = 0,  # ditto
        thinking: str = "",  # thinking-mode output; only written when log_thinking is on
        raw_response_json: str = "",  # unparsed response body; only written when log_raw_response is on
        protocol_failure: Optional[str] = None,
    ) -> None:
        action_input = {
            "id": play.action if play else None,
            "data": {"x": play.col, "y": play.row} if (play and play.col is not None) else {},
            "reasoning": json.dumps({"output": reasoning_text, "usage": usage}),
        }
        record = {
            "turn": turn,
            "data": {
                "game_id": game_id,
                "guid": step.guid,
                "state": step.state,
                "levels_completed": step.levels_completed,
                "full_reset": step.full_reset,
                "available_actions": step.available_actions,
                "action_input": action_input,
            },
            "turn_message": turn_message,
            "model_output": model_output,
            "tool_calls": tool_calls,
            "expectation": play.expectation if play else None,
            "reported_change": reported_change,  # reports on the *previous* turn's expectation
            "click": (
                {"x": play.x, "y": play.y, "row": play.row, "col": play.col}
                if (play and play.x is not None)
                else None
            ),
            "frame_changed": frame_changed,
            "turn_prompt_tokens": turn_prompt_tokens,
            "turn_completion_tokens": turn_completion_tokens,
            "step_in_level": step.turn_in_level,
            "attempt_in_level": step.attempt,
            "frame_ref": step.ref,
            "protocol_failure": protocol_failure,
        }
        if self._log_thinking:
            record["thinking"] = thinking
        if self._log_raw_response:
            try:
                record["raw_response"] = json.loads(raw_response_json) if raw_response_json else None
            except ValueError:
                record["raw_response"] = raw_response_json  # not JSON; keep it as-is rather than lose it
        self._fh.write(json.dumps(record) + "\n")
        self._fh.flush()

    def log_cut(self, fields: dict) -> None:
        """A context-cut event, not a played turn: it has `event` instead of `data`, so readers
        (`scoring.py`, `digest.py`) skip it. Written between the turn it followed and the next.
        `fields` is a `session.CutRecord` as a dict."""
        record = {"event": "cut"}
        record.update(fields)
        self._fh.write(json.dumps(record) + "\n")
        self._fh.flush()

    def close(self) -> None:
        self._fh.close()


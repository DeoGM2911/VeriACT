"""The turn loop: drives one game from start to finish.

    messages = [system]
    loop:
      messages += user(render("turn.md", progress, available_actions) + current frame image)
      model calls tools freely (read_notes / write_notes / inspect / read_pixels / history)
        until it calls play(...)
      step = env.act(action, reasoning=<turn memo>)
      log the turn to trace.jsonl
      if the context is past its high-water mark (or a level just ended above low water):
          window.cut(...)   # [system, digest, degraded records, recent records] -- no model call
      if _stop_reason(...): stop

Each turn is one model exchange with the same prompt and the same tools; the harness imposes no
stages. The model decides when to look (`inspect` / `read_pixels`), when to read or rewrite its
notes, what it expects an action to do (`play.expectation`), and what changed (its free narration,
logged as `reported_change`).

GAME_OVER does not stop the loop by itself: the model can RESET and keep playing. Some games only
allow resetting up to a point, after which GAME_OVER is permanent; there is no SDK flag for this,
so a RESET that leaves the state still GAME_OVER ends the run (`reset_failed`).

`_stop_reason` also enforces a per-level action budget (`baseline_actions[level] x
action_budget_multiplier`, never shown to the model), `max_resets_per_level`, `max_total_actions`
and `max_wall_clock_seconds`. Every exit path writes `run_summary.json` with a `stop_reason`.

Context management lives in `memory/session.py`. Cuts happen only between turns: when the context
crosses its high-water mark, or early when a level has just ended and the context is already above
the low-water mark (a level boundary is the cheapest point to give up old history).
"""
import json
import logging
import time
from pathlib import Path
from typing import Optional

import openai

from baseline.memory import digest
from baseline.agent.env import EnvAdapter
from baseline.agent.llm import ChatClient, TokenTrackingChatClient, image_content
from baseline.memory.memory import NotesStore
from baseline.perception.observation import frame_to_png_bytes, region_to_png_bytes, resample_region
from baseline.perception.render import PromptSet
from baseline.memory import session
from baseline.memory import stagnation
from baseline.agent import tools
from baseline.evaluation.tracelog import TraceLog

logger = logging.getLogger(__name__)

# Version tag the prompts are rendered for: manifest entries with `since` at or before it are
# active (see perception/render.py). Every current entry is.
MILESTONE = "M13"
_CONTEXT_LIMIT_MARKERS = ("context length", "context_length_exceeded", "maximum context")


class ProtocolFailure(Exception):
    """The model never produced a legal `play` call within the retry budget."""


def run(game_id: str, cfg, chat: ChatClient, run_dir, env: Optional[EnvAdapter] = None) -> None:
    run_dir = Path(run_dir)
    start_time = time.monotonic()
    # Counts prompt/completion tokens across every send() in a turn, retries included.
    chat = TokenTrackingChatClient(chat)
    prompts_dir = Path(__file__).resolve().parent.parent / "prompts"
    prompts = PromptSet(prompts_dir, MILESTONE)

    if env is None:
        env = EnvAdapter(
            game_id, run_dir,
            image_scale=cfg.image.scale,
            gridline_brightness=cfg.image.gridline_brightness,
            environments_dir=cfg.paths.environments_dir,
        )
    trace = TraceLog(
        run_dir / "trace.jsonl",
        log_thinking=cfg.trace.log_thinking,
        log_raw_response=cfg.trace.log_raw_response,
    )
    notes = NotesStore(run_dir)

    system_text = prompts.system_prompt({"max_px": cfg.max_px})
    tool_descs = prompts.tool_descriptions({"max_px": cfg.max_px})
    # All six tools are offered on every turn: play, read_notes, write_notes, inspect, read_pixels,
    # history. `play` ends the turn; the other five cost no game action.
    tool_schemas = tools.build_schemas(tool_descs, prompts.active_tool_names())

    # The window owns the conversation. `window.messages` is a plain flat list that the helpers
    # below append to; only the window restructures it, and only between turns.
    window = session.SessionWindow({"role": "system", "content": system_text}, cfg)
    messages = window.messages
    run_facts: list = []  # one digest.TurnFacts per completed turn, for the digest a cut builds
    last_observed = {"description": None}  # last turn's change narration, carried into the digest
    notes_truncated = {"value": False}  # set by _build_digest, read back onto the cut record

    def _build_digest(turn_numbers: list) -> str:
        """The digest message a cut leaves behind. Every part of it is a verbatim copy of
        something already recorded: the per-turn facts come from the same fields `tracelog.py`
        writes, the notes are read fresh off disk, and `last_description` is the model's own last
        change narration. Nothing is regenerated, and no model is asked anything.
        """
        wanted = set(turn_numbers)
        body = digest.render_digest(
            [f for f in run_facts if f.turn in wanted], max_chars=cfg.session.digest_max_chars,
        )
        cap = cfg.session.notes_push_max_chars
        guide, working = notes.read("GUIDE.md"), notes.read("WORKING.md")
        notes_truncated["value"] = len(guide) > cap or len(working) > cap
        return prompts.digest_message({
            "digest_body": body or "(no turns recorded yet)",
            # Notes are normally pulled with `read_notes`; at a cut they are copied in, because the
            # model has just lost the context it would otherwise rely on.
            "guide_notes": guide[:cap] or "(empty)",
            "working_notes": working[:cap] or "(empty)",
            # Capped like the notes: `_change_report` is reasoning_content + text and is otherwise
            # unbounded on a thinking model.
            "last_description": (last_observed["description"] or "(none recorded)")[:cap],
        })

    def _placeholder_for(placeheld_turn: int) -> str:
        return prompts.frame_placeholder({"placeheld_turn": placeheld_turn})

    turn = 0  # bound before the loop so the crash-record except clause below always has a value,
    total_actions = 0  # even if the exception fires before the for loop's first iteration.
    try:
        step = env.last
        actions_this_level = 0  # cumulative across resets, unlike step.turn_in_level
        resets_this_level = 0
        level_boundary = False  # set at the end of each turn; drives the next turn's level_opening
        for turn in range(1, cfg.run.max_turns + 1):
            chat.reset_turn()  # trace.log_turn below reads the running count as of its own point
                                # in the turn; a protocol-failure exit still gets whatever was
                                # spent up to the point it failed.
            # The conversation is never trimmed mid-turn, so check here, before the turn starts:
            # if the last cut could not bring the context under a sendable size, stop with a
            # recorded reason rather than send a request that will be rejected.
            if window.over_hard_limit():
                _write_run_summary(
                    run_dir, stop_reason="turn_does_not_fit", turn=turn,
                    total_actions=total_actions,
                    images_in_context=window.image_count(),
                    estimated_tokens=window.estimate_tokens(),
                )
                logger.warning("turn %d does not fit after a cut; stopping", turn)
                break

            turn_vars = {
                "turn": turn,
                "obs_image": "[image, not logged as text]",
                "available_actions": ", ".join(step.available_actions),
                "levels_completed": step.levels_completed,
                "win_levels": step.win_levels,
                "state": step.state,
            }
            turn_text = prompts.turn_message(turn_vars)
            # Opened before the observation frame is appended, so that frame counts against this
            # turn's image allowance. `level_opening` marks the first turn of a level (turn 1, or
            # the turn after a level completes); session.py keeps that turn's frame longer.
            window.begin_turn(
                turn, step.levels_completed, level_opening=(level_boundary or turn == 1),
            )
            png = frame_to_png_bytes(step.frames[-1], scale=cfg.image.scale, gridline_brightness=cfg.image.gridline_brightness)
            messages.append(
                {"role": "user", "content": [{"type": "text", "text": turn_text}, image_content(png)]}
            )

            try:
                # One model exchange: non-play tools are dispatched and the model re-called without
                # a reprompt, up to max_tool_calls_per_turn; `play` ends the turn.
                play, reply, tool_call_log = _get_play_call(
                    chat, messages, tool_schemas, step.available_actions, step.state, notes, env, cfg,
                    window=window, run_facts=run_facts,
                )
            except Exception as exc:  # provider error, possibly a context-length rejection
                if _is_context_limit_error(exc):
                    _write_run_summary(
                        run_dir, stop_reason="context_limit", turn=turn,
                        turns_until_context_limit=turn, total_actions=total_actions,
                    )
                    logger.warning("context limit reached at turn %d", turn)
                    break
                raise

            reasoning_text = _turn_memo(reply, play) if reply else ""
            reported_change = _change_report(reply) if reply else ""
            thinking_text = reply.reasoning_content if reply else ""
            raw_response_json = reply.raw_response_json if reply else ""

            if play is None:
                trace.log_turn(
                    game_id=game_id,
                    turn=turn,
                    play=None,
                    step=step,
                    turn_message=turn_text,
                    model_output=reasoning_text,
                    tool_calls=tool_call_log,
                    reasoning_text=reasoning_text,
                    reported_change=reported_change,
                    usage={"output_tokens": 0, "total_tokens": 0},
                    turn_prompt_tokens=chat.prompt_tokens,
                    turn_completion_tokens=chat.completion_tokens,
                    thinking=thinking_text,
                    raw_response_json=raw_response_json,
                    protocol_failure="no_play_call",
                )
                _write_run_summary(
                    run_dir, stop_reason="protocol_failure", turn=turn, total_actions=total_actions,
                )
                raise ProtocolFailure(
                    f"model did not produce a legal `play` call within {cfg.run.max_protocol_retries} retries"
                )

            # The turn's messages are final once a legal `play` is in hand (`env.act()` appends
            # nothing to the conversation), so the turn record is closed here.
            window.end_turn()
            window.calibrate(reply.total_tokens, reply.output_tokens)

            reasoning = _reasoning_payload(reasoning_text, reply.output_tokens, reply.total_tokens)
            prev_levels_completed = step.levels_completed
            pre_action_frame = step.frames[-1]  # for the frame_changed trace field, below
            step = env.act(play.action, x=play.col, y=play.row, reasoning=reasoning)
            total_actions += 1
            actions_this_level += 1
            if play.action == "RESET":
                resets_this_level += 1
            level_boundary = step.levels_completed != prev_levels_completed
            # Checked against every frame this action returned, not just the settled last one, so
            # an animation that changes and reverts still counts as a change. Logged for digest.py;
            # not shown to the model.
            changed = stagnation.sequence_changed(pre_action_frame, step.frames)
            if level_boundary:
                notes.archive_working(prev_levels_completed)  # keep a copy of this level's WORKING.md
                actions_this_level = 0
                resets_this_level = 0

            trace.log_turn(
                game_id=game_id,
                turn=turn,
                frame_changed=changed,
                play=play,
                step=step,
                turn_message=turn_text,
                model_output=reasoning_text,
                tool_calls=tool_call_log,
                reasoning_text=reasoning_text,
                reported_change=reported_change,
                usage=reasoning["usage"],
                turn_prompt_tokens=chat.prompt_tokens,
                turn_completion_tokens=chat.completion_tokens,
                thinking=thinking_text,
                raw_response_json=raw_response_json,
            )

            # -- the session cut (memory/session.py) -------------------------------------------
            # Everything the digest may later need about this turn, captured while it is in hand.
            run_facts.append(digest.TurnFacts(
                turn=turn, level=prev_levels_completed, action=play.action, x=play.x, y=play.y,
                expectation=play.expectation,
                reported_change=reported_change,
                frame_changed=changed, state=step.state, level_completed=level_boundary,
            ))
            if reported_change:
                last_observed["description"] = reported_change

            # A cut happens only at a turn boundary. Two triggers: a budget has crossed its
            # high-water mark, or a level just completed while the context is already above the
            # low-water mark. A due cut is never postponed to wait for a level boundary, since that
            # would eat into the room reserved for the next turn.
            if window.over_high_water() or (level_boundary and window.estimate_tokens() > _cut_floor(cfg)):
                record = window.cut(
                    _build_digest, _placeholder_for, turn=turn, level_boundary=level_boundary,
                )
                record.notes_truncated = notes_truncated["value"]
                trace.log_cut(record.as_dict())

            stop_reason = _stop_reason(
                step, play, total_actions, actions_this_level, resets_this_level,
                env.baseline_actions, time.monotonic() - start_time, cfg,
            )
            if stop_reason:
                _write_run_summary(
                    run_dir, stop_reason=stop_reason, turn=turn, total_actions=total_actions,
                )
                break
        else:
            # Ran out of turns without _stop_reason firing. With the default config (300 turns vs
            # 2000 actions) the turn cap usually binds first.
            _write_run_summary(
                run_dir, stop_reason="turn_cap_exceeded", turn=cfg.run.max_turns,
                total_actions=total_actions,
            )
    except ProtocolFailure:
        # Already wrote its own run_summary.json (stop_reason="protocol_failure"); don't let the
        # handler below overwrite it.
        raise
    except Exception as exc:
        # Safety net: record any unexpected crash (e.g. a provider error other than a context-
        # length rejection) in run_summary.json, then re-raise it.
        _write_run_summary(
            run_dir, stop_reason="unhandled_exception", turn=turn, total_actions=total_actions,
            error=f"{type(exc).__name__}: {exc}",
        )
        raise
    finally:
        env.close()
        trace.close()


def _get_play_call(chat: ChatClient, messages: list, tool_schemas: list, available_actions: list,
                    state: str, notes: NotesStore, env: EnvAdapter, cfg, *, window=None, run_facts=None):
    """Call the model until it produces a legal `play`, or exhaust the retry budget.

    Within one attempt the model may call `read_notes` / `write_notes` / `inspect` /
    `read_pixels` / `history` up to `max_tool_calls_per_turn` times; each is dispatched
    immediately, its result appended, and the model called again with no reprompt. `play` ends the
    turn. No tool call at all, an invalid `play`, or exhausting the non-`play` budget triggers a
    corrective reprompt ("you must call play"), up to `max_protocol_retries` times.

    The click action (`ACTION6`) carries `x`/`y` in image pixels; `tools.validate_play_args`
    converts them to grid cells.

    Returns `(play_action or None, last_reply, tool_call_log)`.
    """
    max_attempts = cfg.run.max_protocol_retries + 1
    last_reply = None
    tool_call_log: list = []
    non_play_calls = 0

    for attempt in range(max_attempts):
        while True:
            reply = chat.send(messages, tool_schemas)
            last_reply = reply
            assistant_msg = {"role": "assistant", "content": reply.text or None}
            if reply.tool_calls:
                assistant_msg["tool_calls"] = [_tool_call_dict(tc) for tc in reply.tool_calls]
            messages.append(assistant_msg)

            play_result = None
            saw_invalid_play = False
            for tc in reply.tool_calls:
                if play_result is not None:
                    # The turn already ended earlier in this same response; every tool_call_id
                    # still needs a reply for the API contract.
                    messages.append({"role": "tool", "tool_call_id": tc.id, "content": "ok (unused: turn already ended)"})
                    continue
                if tc.name == "play":
                    try:
                        play = tools.validate_play_args(
                            tc.arguments, available_actions, cfg.image.scale, cfg.max_px, state,
                        )
                    except tools.ToolCallError as e:
                        messages.append({"role": "tool", "tool_call_id": tc.id, "content": str(e)})
                        saw_invalid_play = True
                        continue
                    messages.append({"role": "tool", "tool_call_id": tc.id, "content": "ok"})
                    play_result = play
                else:
                    non_play_calls += 1
                    result, extra_messages = _dispatch_tool(tc, notes, env, cfg, window=window, run_facts=run_facts)
                    if result.startswith("unknown tool:"):
                        result = f"{result} You must call `play` to end your turn."
                    tool_call_log.append({"name": tc.name, "arguments": tc.arguments, "result": result})
                    messages.append({"role": "tool", "tool_call_id": tc.id, "content": result})
                    messages.extend(extra_messages)

            if play_result is not None:
                return play_result, reply, tool_call_log

            # An invalid `play` counts against the retry budget like no call at all; otherwise a
            # model that keeps sending bad `play` arguments would loop here forever.
            if saw_invalid_play or not reply.tool_calls or non_play_calls >= cfg.run.max_tool_calls_per_turn:
                break  # -> reprompt below, consuming one attempt

        if attempt < max_attempts - 1:
            messages.append({"role": "user", "content": "You must call `play` to end your turn."})

    return None, last_reply, tool_call_log


def _dispatch_tool(tc, notes: NotesStore, env: EnvAdapter, cfg, *, window=None, run_facts=None) -> tuple:
    """Execute a non-`play` tool call. Returns `(tool_message_content, extra_messages)`: most
    tools return only content; `inspect` also returns a follow-up `user` message carrying the
    requested images. Images travel only on `user` messages, never inside a `tool` message, so
    this works with providers that don't support images in tool results.

    `window` enforces the per-turn image allowance (`session.max_images_per_turn`). An `inspect`
    that would exceed it is rejected as a `ToolCallError`, so the model is told how many images
    remain and can retry with fewer views.
    """
    try:
        if tc.name == "read_notes":
            action = tools.validate_read_notes_args(tc.arguments)
            return notes.read(action.file) or "(empty)", []
        if tc.name == "write_notes":
            action = tools.validate_write_notes_args(tc.arguments)
            notes.write(action.file, action.content)
            return "ok", []
        if tc.name == "inspect":
            action = tools.validate_inspect_args(tc.arguments, cfg.max_px, cfg.frames.max_inspect_views)
            if window is not None:
                remaining = window.remaining_turn_images()
                if len(action.views) > remaining:
                    raise tools.ToolCallError(
                        f"that would add {len(action.views)} image(s), but only {remaining} "
                        f"remain in this turn's image allowance. Ask for fewer views, or use "
                        f"`read_pixels`, which returns text rather than images and costs nothing "
                        f"against this allowance."
                    )
            return _run_inspect(action, env, cfg)
        if tc.name == "read_pixels":
            action = tools.validate_read_pixels_args(tc.arguments, cfg.max_px, cfg.frames.max_inspect_views)
            return _run_read_pixels(action, env, cfg)
        if tc.name == "history":
            facts = run_facts or []
            current_turn = facts[-1].turn if facts else 0
            action = tools.validate_history_args(tc.arguments, current_turn=current_turn)
            return _run_history(action, facts)
        return f"unknown tool: {tc.name!r}", []
    except tools.ToolCallError as e:
        return str(e), []


def _resolve_frame(env: EnvAdapter, turn: int, frame: int):
    """Raw frame for `(turn, frame)`, or a `tools.ToolCallError` the model can act on -- an
    unrecognised address is a model mistake (a turn/frame that never happened), not a harness bug.
    """
    try:
        return env.get_frame(turn, frame)
    except (KeyError, IndexError) as e:
        raise tools.ToolCallError(str(e))


def _run_inspect(action: tools.InspectAction, env: EnvAdapter, cfg) -> tuple:
    # The leading text block must start with `session.INSPECT_VIEWS_PREFIX`: that is how
    # session.py tells an `inspect` result apart from a turn's own observation frame (inspect
    # images are dropped first when trimming context).
    content = [{"type": "text", "text": f"{session.INSPECT_VIEWS_PREFIX}{action.question}"}]
    labels = []
    for v in action.views:
        frame = _resolve_frame(env, v.turn, v.frame)
        png = region_to_png_bytes(frame, v.region, scale=cfg.image.scale, gridline_brightness=cfg.image.gridline_brightness)
        region_desc = f", region {v.region}" if v.region else ""
        content.append({"type": "text", "text": f"[{v.label}] turn {v.turn}, frame {v.frame}{region_desc}"})
        content.append(image_content(png))
        labels.append(v.label)
    summary = f"{len(action.views)} view(s) returned below: {', '.join(labels)}."
    return summary, [{"role": "user", "content": content}]


def _run_read_pixels(action: tools.ReadPixelsAction, env: EnvAdapter, cfg) -> tuple:
    lines = [f"read_pixels: {action.question}"]
    for v in action.views:
        frame = _resolve_frame(env, v.turn, v.frame)
        grid, palette, axes = resample_region(frame, v.region, v.rows, v.columns, scale=cfg.image.scale,
                                               symbol_table=env.pixel_symbols)
        region_desc = f", region {v.region}" if v.region else ""
        col_x = axes["col_x"]
        col_desc = f"x={col_x[0]}" if len(col_x) == 1 else (
            f"x {col_x[0]}..{col_x[-1]} (step {(col_x[-1] - col_x[0]) / (len(col_x) - 1):.3g})"
        )
        lines.append(
            f"[{v.label}] turn {v.turn}, frame {v.frame}{region_desc}, {v.rows}x{v.columns}"
            f" -- rows labelled by pixel y, columns span {col_desc}:"
        )
        lines.extend(f"y={y}: {row}" for y, row in zip(axes["row_y"], grid))
        lines.append("palette: " + ", ".join(f"{sym}={val}" for sym, val in palette.items()))
        if axes["coarse"]:
            lines.append(
                "note: this sampling did not land on every cell in the region -- it is a summary "
                "view, not an exact one. Use more rows/columns, or a smaller region, when you need "
                "a precise boundary."
            )
    return "\n".join(lines), []


def _run_history(action: tools.HistoryAction, run_facts: list) -> tuple:
    """Look back over the run so far, from the per-turn facts in `run_facts` (the same ones
    `digest.py` uses at a cut); nothing is inferred. `events` renders each turn with
    `digest.turn_line`, so a turn reads the same here as in a digest. `attempts` summarizes each
    RESET-to-RESET span."""
    if not run_facts:
        return "no turns recorded yet.", []
    if action.view == "events":
        lo = action.from_turn or 1
        hi = action.to_turn or run_facts[-1].turn
        rows = [f for f in run_facts if lo <= f.turn <= hi]
        if not rows:
            return f"no turns recorded in range {lo}-{hi}.", []
        lines = [f"events, turns {lo}-{hi}:"]
        lines += [digest.turn_line(f, field_chars=90) for f in rows]
        return "\n".join(lines), []

    # view == "attempts": one span per RESET-to-RESET/GAME_OVER stretch. A RESET action starts a
    # new attempt; the attempt it closes is summarized by the state/actions seen since the last one.
    attempts: list = []
    start, actions, outcome = run_facts[0].turn, [], None
    for f in run_facts:
        if f.action == "RESET" and actions:
            attempts.append((start, f.turn - 1, actions, outcome))
            start, actions, outcome = f.turn, [], None
        actions.append(f.action)
        outcome = f.state
    attempts.append((start, run_facts[-1].turn, actions, outcome))
    lines = ["attempts:"]
    lines += [
        f"attempt {i + 1}: turns {s}-{e}, outcome={o}, actions={acts}"
        for i, (s, e, acts, o) in enumerate(attempts)
    ]
    return "\n".join(lines), []


def _change_report(reply) -> str:
    """Everything the model said this turn apart from its `play` arguments: its thinking output
    plus its final text. This is written after seeing the result of the previous action, so it is
    the model's report on the previous turn's `expectation`. Pair `trace[N-1].expectation` with
    `trace[N].reported_change` to compare prediction with outcome offline.

    Includes `reasoning_content` because a thinking model often says everything while thinking and
    leaves the final message empty.
    """
    parts = [p for p in (reply.reasoning_content, reply.text) if p]
    return "\n\n".join(parts)


def _turn_memo(reply, play=None) -> str:
    """The turn memo: everything the model emitted this turn, recorded with the action (as the
    SDK's `reasoning` payload) and in the trace's `reasoning` field.

    `play.expectation` comes first, since it is required and so always present; then the change
    report (thinking output plus final text). Thinking output is always included here;
    `cfg.trace.log_thinking` only controls an additional verbatim copy in the trace.
    """
    parts = []
    if play is not None and play.expectation:
        parts.append(f"Expectation: {play.expectation}")
    change_report = _change_report(reply)
    if change_report:
        parts.append(change_report)
    return "\n\n".join(parts)


def _tool_call_dict(tc) -> dict:
    return {
        "id": tc.id,
        "type": "function",
        "function": {"name": tc.name, "arguments": json.dumps(tc.arguments)},
    }


_MAX_REASONING_BYTES = 16 * 1024  # mirrors arcengine.enums.MAX_REASONING_BYTES; the SDK's
                                   # ActionInput validator is still the actual enforcement point,
                                   # this is only what the harness truncates against ahead of it.
_REASONING_SAFETY_MARGIN_BYTES = 64
_TRUNCATION_SUFFIX = "…(truncated)"


def _reasoning_payload(text: str, output_tokens: int, total_tokens: int,
                        max_bytes: int = _MAX_REASONING_BYTES) -> dict:
    """Build the `{"output": ..., "usage": {...}}` dict `env.act(reasoning=...)` sends, truncating
    `text` only as much as needed to keep the dict's JSON-serialized size under `max_bytes`.

    The serialized size, not the raw text's byte length, is what arcengine's `ActionInput`
    validator checks (`MAX_REASONING_BYTES`, 16 KB), and JSON escaping makes the two differ
    unpredictably. An oversized payload is rejected silently: `env.step()` just returns None. So
    this measures the real serialized size and binary-searches the longest prefix (by character, so
    a multi-byte character is never split) that fits.
    """
    usage = {"output_tokens": output_tokens, "total_tokens": total_tokens}
    budget = max_bytes - _REASONING_SAFETY_MARGIN_BYTES

    def serialized_size(output: str) -> int:
        payload = json.dumps({"output": output, "usage": usage}, separators=(",", ":"))
        return len(payload.encode("utf-8"))

    if serialized_size(text) <= budget:
        return {"output": text, "usage": usage}

    # Escaping only ever expands a character's serialized size, never shrinks it, so serialized
    # size is monotonic non-decreasing in prefix length -- binary search is valid.
    lo, hi = 0, len(text)
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if serialized_size(text[:mid] + _TRUNCATION_SUFFIX) <= budget:
            lo = mid
        else:
            hi = mid - 1
    return {"output": text[:lo] + _TRUNCATION_SUFFIX, "usage": usage}


def _is_context_limit_error(exc: Exception) -> bool:
    if isinstance(exc, openai.BadRequestError):
        msg = str(exc).lower()
        return any(marker in msg for marker in _CONTEXT_LIMIT_MARKERS)
    return False


def _cut_floor(cfg) -> int:
    """The token mark above which a level boundary is worth cutting at (the low-water mark). After
    a level ends the map has changed and older frames matter less, so it is the cheapest point to
    give history up. Below this mark there is nothing worth cutting."""
    _, low = session.token_budgets(cfg)
    return low


def _stop_reason(step, play, total_actions: int, actions_this_level: int, resets_this_level: int,
                  baseline_actions: list, elapsed_seconds: float, cfg) -> Optional[str]:
    """Whether the run should stop after this turn, and why. A pure function of values the caller
    already has, so it's testable without running the loop. Checked in this order because more
    than one can be true on the same turn:

    1. `WIN`.
    2. A `RESET` just taken that did not recover the game (see the module docstring) -- no
       further action can help, so it outranks the budgets below.
    3. `max_resets_per_level` -- this level has been retried enough times, whether or not the
       resets themselves are working.
    4. The per-level action budget (`baseline_actions[level] x action_budget_multiplier`) --
       skipped entirely if this environment doesn't publish `baseline_actions` or has fewer
       entries than `step.level`, rather than applying a budget of zero.
    5. `max_total_actions`, the global fallback that exists even for a game with no baseline data.
    6. `max_wall_clock_seconds` -- real time spent. `None` (the default) disables it.
    """
    if step.state == "WIN":
        return "win"
    if play.action == "RESET" and step.state == "GAME_OVER":
        return "reset_failed"
    if play.action == "RESET" and resets_this_level >= cfg.run.max_resets_per_level:
        return "reset_cap_exceeded"
    if baseline_actions and step.level < len(baseline_actions):
        level_budget = baseline_actions[step.level] * cfg.run.action_budget_multiplier
        if actions_this_level >= level_budget:
            return "level_budget_exceeded"
    if total_actions >= cfg.run.max_total_actions:
        return "action_budget_exceeded"
    if cfg.run.max_wall_clock_seconds is not None and elapsed_seconds >= cfg.run.max_wall_clock_seconds:
        return "wall_clock_exceeded"
    return None


def _write_run_summary(run_dir, **fields) -> None:
    (Path(run_dir) / "run_summary.json").write_text(json.dumps(fields, indent=2), encoding="utf-8")

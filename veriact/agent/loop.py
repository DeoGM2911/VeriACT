"""The VeriACT turn loop: OBSERVE -> VERIFY+PLAN -> [UPDATE] -> ACT.

    messages = [system]
    loop:
      messages += user(render("turn.md", progress, available_actions, grounded facts) + frame)
      if the last action completed a level:
          TERMINAL OBSERVE  -- before/after frames of the completing action; until observe(...)
      OBSERVE             -- free read-only tools, until observe(description)
      VERIFY+PLAN         -- free tools, until verify_and_plan(verdict, evidence, mode, plan, ...)
      if update required:
          UPDATE          -- until write_notes(...)
      ACT                 -- until a legal play(action, expectation[, x, y, target_description])
        (a click is first checked against a crop around the target: click-target confirmation)
      step = env.act(action, reasoning=<turn memo>)
      compute grounded evidence for next turn (frame changed? repeated action? click mismatch?)
      log the turn; cut the context if needed; stop if _stop_reason(...)

Each stage is its own model exchange that must end with its own terminal tool call; the
controller enforces the order, so ACT is unreachable until VERIFY+PLAN (and UPDATE, when
required) have completed. Each stage is offered only the tools that make sense there:

- OBSERVE / TERMINAL OBSERVE: `read_notes`, `inspect`, `read_pixels`, `history`, `observe`.
  OBSERVE commits a description of the current frame before the previous action is judged.
  TERMINAL OBSERVE runs first on the turn after a level completes: it sees the frame before and
  after the completing action and must state `objective_reconfirmed` / `goal_changed`.
- VERIFY+PLAN: every tool except `play`. Returns a `verdict` (confirmed / contradicted /
  uncertain) on the previous action's `expectation`, `observation_check` / `objects_changed`
  against `WORKING.md`, a `mode` (carry_forward / replan) and `plan`. `objective_check` is
  required after a level boundary, `post_mortem` after GAME_OVER, and `note_check` after a turn
  that wrote notes.
- UPDATE: only `write_notes`. Required when the verdict is not `confirmed`, the mode is `replan`,
  objects changed, a level just ended, the game is over, or the goal changed.
- ACT: `play`, the read tools, and `write_notes` restricted to `WORKING.md` with a `step_check`.

Grounded evidence shown in the next turn's message: whether the frame changed
("Last action changed the frame."), a repetition note from `stagnation.RepetitionTracker`, a
multi-frame note when an action returned several frames, and a click-mismatch note when a click
could not be confirmed after every retry.

GAME_OVER does not stop the loop by itself: the model can RESET and keep playing. Some games only
allow resetting up to a point, after which GAME_OVER is permanent; there is no SDK flag for this,
so a RESET that leaves the state still GAME_OVER ends the run (`reset_failed`). `_stop_reason`
also enforces a per-level action budget (`baseline_actions[level] x action_budget_multiplier`,
never shown to the model), `max_resets_per_level`, `max_total_actions` and
`max_wall_clock_seconds`. Every exit path writes `run_summary.json` with a `stop_reason`.

Context management lives in `memory/session.py`: cuts make no model call and happen only between
turns -- when the context crosses its high-water mark, or early when a
level has just ended and the context is already above the low-water mark.

`tool_mode: memo` is a plain-text fallback for models without reliable tool calling: the model
writes free text plus a trailing action line (parsed by `memo.py`), notes are pushed into each
turn, and the OBSERVE / VERIFY+PLAN / UPDATE stages are skipped.
"""
import json
import logging
import time
from pathlib import Path
from typing import Optional

import openai

from veriact.memory import digest
from veriact.agent.env import EnvAdapter
from veriact.agent.llm import ChatClient, TokenTrackingChatClient, image_content
from veriact.agent import memo
from veriact.memory.memory import NotesStore
from veriact.perception.observation import confirm_crop_region, frame_to_png_bytes, region_to_png_bytes, resample_region
from veriact.perception.render import PromptSet
from veriact.memory import session
from veriact.memory import stagnation
from veriact.agent import tools
from veriact.evaluation.tracelog import TraceLog

logger = logging.getLogger(__name__)

# Version tag the prompts are rendered for: manifest entries with `since` at or before it are
# active (see perception/render.py). Every current entry is.
MILESTONE = "M14"
_CONTEXT_LIMIT_MARKERS = ("context length", "context_length_exceeded", "maximum context")
_ACT_WRITABLE_FILES = ("WORKING.md",)  # ACT may only mark plan steps done in WORKING.md


class ProtocolFailure(Exception):
    """The model never produced a legal terminal call for a stage within the retry budget."""


def run(game_id: str, cfg, chat: ChatClient, run_dir, env: Optional[EnvAdapter] = None) -> None:
    run_dir = Path(run_dir)
    start_time = time.monotonic()
    # Counts prompt/completion tokens across every send() in a turn: all stages, retries included.
    chat = TokenTrackingChatClient(chat)
    is_memo = cfg.model.tool_mode == "memo"
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

    if is_memo:
        system_text = prompts.memo_system_prompt({"max_px": cfg.max_px})
        tool_schemas = []  # no JSON schemas at all -- there is no tool-calling in memo mode
        update_tool_schemas = []
        observe_tool_schemas = []
        terminal_observe_tool_schemas = []
        act_tool_schemas = []
    else:
        system_text = prompts.system_prompt({"max_px": cfg.max_px})
        tool_descs = prompts.tool_descriptions({"max_px": cfg.max_px})
        tool_schemas = tools.build_schemas(tool_descs, prompts.active_tool_names())
        # Per-stage tool sets (see the module docstring). VERIFY+PLAN uses the full `tool_schemas`.
        # UPDATE: write_notes only.
        update_tool_schemas = [s for s in tool_schemas if s["function"]["name"] == "write_notes"]
        # OBSERVE: read-only tools plus its terminal `observe`.
        observe_tool_schemas = [
            s for s in tool_schemas
            if s["function"]["name"] in ("read_notes", "inspect", "read_pixels", "history", "observe")
        ]
        # TERMINAL OBSERVE: OBSERVE's tools, with an `observe` schema variant that also requires
        # objective_reconfirmed / goal_changed (the function name stays "observe").
        observe_terminal_desc = prompts.observe_terminal_description({"max_px": cfg.max_px})
        terminal_observe_tool_schemas = [
            tools.observe_schema(observe_terminal_desc, terminal=True)
            if s["function"]["name"] == "observe" else s
            for s in observe_tool_schemas
        ]
        # ACT: play, the read tools, and write_notes narrowed to WORKING.md -- for marking a
        # `## Plan` step done right after the action that completed it; broader rewrites are
        # UPDATE's job. `require_step_check=True` makes the model say which plan step this marks
        # done and how the action matches it. tools.validate_write_notes_args enforces the file
        # restriction too, not just the schema.
        act_tool_schemas = [
            tools.write_notes_schema(
                tool_descs["write_notes"], allowed_files=_ACT_WRITABLE_FILES, require_step_check=True,
            )
            if s["function"]["name"] == "write_notes" else s
            for s in tool_schemas
            if s["function"]["name"] in ("play", "read_notes", "inspect", "read_pixels", "history", "write_notes")
        ]

    # The window owns the conversation. `window.messages` is a plain flat list that the stage
    # functions append to; only the window restructures it, and only between turns.
    window = session.SessionWindow({"role": "system", "content": system_text}, cfg)
    messages = window.messages
    run_facts: list = []  # one digest.TurnFacts per completed turn, for the digest a cut builds
    last_observed = {"description": None, "plan": None}  # last OBSERVE / VERIFY+PLAN output, for the digest
    notes_truncated = {"value": False}  # set by _build_digest, read back onto the cut record

    def _build_digest(turn_numbers: list) -> str:
        """The digest message a cut leaves behind. Every part of it is a verbatim copy of
        something already recorded: the per-turn facts come from the same fields `tracelog.py`
        writes, the notes are read fresh off disk, and the two carried-forward lines are the last
        `observe`/`verify_and_plan` values the model itself produced. Nothing is regenerated, and
        no model is asked anything.
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
            "last_description": last_observed["description"] or "(none recorded)",
            "last_plan": last_observed["plan"] or "(none recorded)",
        })

    def _placeholder_for(placeheld_turn: int) -> str:
        return prompts.frame_placeholder({"placeheld_turn": placeheld_turn})

    turn = 0  # bound before the loop so the crash-record except clause below always has a value,
    total_actions = 0  # even if the exception fires before the for loop's first iteration.
    try:
        step = env.last
        actions_this_level = 0  # cumulative across resets, unlike step.turn_in_level
        resets_this_level = 0
        # Set at the end of each turn, read by the *next* turn:
        level_boundary = False  # the last action completed a level -> TERMINAL OBSERVE, objective_check
        wrote_notes_last_turn = False  # UPDATE ran last turn -> VERIFY+PLAN's note_check
        frame_changed_note = None  # grounded "did the frame change" fact for the turn message
        last_action_changed_frame = None  # sharpens frames_note's wording
        click_mismatch_note = None  # the last click was forced through without confirmation
        repetition = stagnation.RepetitionTracker(
            cfg.run.stagnation_window, cfg.run.stagnation_min_repeats,
        )
        for turn in range(1, cfg.run.max_turns + 1):
            chat.reset_turn()  # every trace.log_turn below this reads the running count as of
                                # its own point in the turn; a protocol-failure exit still gets
                                # whatever was spent up to the point it failed.
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
            if frame_changed_note is not None:
                turn_vars["frame_changed_note"] = frame_changed_note
            if click_mismatch_note is not None:
                turn_vars["click_mismatch_note"] = click_mismatch_note
            stagnation_note = repetition.note()
            if stagnation_note is not None:
                turn_vars["stagnation_note"] = stagnation_note
            if is_memo:
                # Memo mode has no read_notes tool, so the notes are included in every turn.
                turn_vars["guide_content"] = notes.read("GUIDE.md") or "(empty)"
                turn_vars["working_content"] = notes.read("WORKING.md") or "(empty)"
                turn_text = prompts.memo_turn_message(turn_vars)
            else:
                if len(step.frames) > 1:
                    # Multi-frame note, only when the action returned several frames (otherwise the
                    # block is dropped by render.py's block-omission rule). When the frame changed
                    # somewhere in the sequence, it suggests inspecting which frame it changed on.
                    if last_action_changed_frame:
                        turn_vars["frames_note"] = (
                            f"This action returned {len(step.frames)} frames, and the frame changed "
                            f"at some point during them; frame 0 is the first, {len(step.frames) - 1} "
                            f"is the one shown above. Consider using `inspect`/`read_pixels` to see "
                            f"which frame the change happened on."
                        )
                    else:
                        turn_vars["frames_note"] = (
                            f"This action returned {len(step.frames)} frames; frame 0 is the first, "
                            f"{len(step.frames) - 1} is the one shown above. `inspect`/`read_pixels` "
                            f"can look at any of them by frame index."
                        )
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

            observe_result = None
            observe_tool_log: list = []
            terminal_observe_result = None
            terminal_observe_tool_log: list = []
            verify_result = None
            verify_tool_log: list = []
            update_tool_log: list = []
            try:
                if not is_memo:
                    if level_boundary:
                        # TERMINAL OBSERVE: shown the frame from just before the level-completing
                        # action (`pre_action_frame`, still bound from last turn) and the frame
                        # from just after it. When a level completes, arcengine returns the
                        # completed state as the second-to-last frame and the next level's opening
                        # frame last; fall back to the only frame if just one was returned.
                        objective_frame = step.frames[-2] if len(step.frames) > 1 else step.frames[-1]
                        before_png = frame_to_png_bytes(
                            pre_action_frame, scale=cfg.image.scale,
                            gridline_brightness=cfg.image.gridline_brightness,
                        )
                        after_png = frame_to_png_bytes(
                            objective_frame, scale=cfg.image.scale,
                            gridline_brightness=cfg.image.gridline_brightness,
                        )
                        messages.append({"role": "user", "content": [
                            {"type": "text", "text": "Before your last action:"},
                            image_content(before_png),
                            {"type": "text", "text": "After your last action:"},
                            image_content(after_png),
                        ]})
                        messages.append(_stage_tools_message(prompts, terminal_observe_tool_schemas))
                        terminal_observe_result, terminal_observe_reply, terminal_observe_tool_log = _get_observe_call(
                            chat, messages, terminal_observe_tool_schemas, notes, env, cfg,
                            terminal=True, window=window, run_facts=run_facts,
                        )
                        if terminal_observe_result is None:
                            reasoning_text = _turn_memo(terminal_observe_reply, None) if terminal_observe_reply else ""
                            reported_change = _change_report(terminal_observe_reply) if terminal_observe_reply else ""
                            trace.log_turn(
                                game_id=game_id,
                                turn=turn,
                                play=None,
                                step=step,
                                turn_message=turn_text,
                                model_output=reasoning_text,
                                tool_calls=terminal_observe_tool_log,
                                reasoning_text=reasoning_text,
                                reported_change=reported_change,
                                usage={"output_tokens": 0, "total_tokens": 0},
                                turn_prompt_tokens=chat.prompt_tokens,
                                turn_completion_tokens=chat.completion_tokens,
                                thinking=terminal_observe_reply.reasoning_content if terminal_observe_reply else "",
                                raw_response_json=terminal_observe_reply.raw_response_json if terminal_observe_reply else "",
                                protocol_failure="no_terminal_observe_call",
                            )
                            _write_run_summary(
                                run_dir, stop_reason="protocol_failure", turn=turn, total_actions=total_actions,
                            )
                            raise ProtocolFailure(
                                f"model did not produce a legal terminal `observe` call within "
                                f"{cfg.run.max_protocol_retries} retries"
                            )

                    # OBSERVE runs every turn, before VERIFY+PLAN, so the observation is committed
                    # before the previous action is judged.
                    messages.append(_stage_tools_message(prompts, observe_tool_schemas))
                    observe_result, observe_reply, observe_tool_log = _get_observe_call(
                        chat, messages, observe_tool_schemas, notes, env, cfg, window=window,
                        run_facts=run_facts,
                    )
                    if observe_result is None:
                        reasoning_text = _turn_memo(observe_reply, None) if observe_reply else ""
                        reported_change = _change_report(observe_reply) if observe_reply else ""
                        trace.log_turn(
                            game_id=game_id,
                            turn=turn,
                            play=None,
                            step=step,
                            turn_message=turn_text,
                            model_output=reasoning_text,
                            tool_calls=observe_tool_log,
                            reasoning_text=reasoning_text,
                            reported_change=reported_change,
                            usage={"output_tokens": 0, "total_tokens": 0},
                            turn_prompt_tokens=chat.prompt_tokens,
                            turn_completion_tokens=chat.completion_tokens,
                            thinking=observe_reply.reasoning_content if observe_reply else "",
                            raw_response_json=observe_reply.raw_response_json if observe_reply else "",
                            protocol_failure="no_observe_call",
                        )
                        _write_run_summary(
                            run_dir, stop_reason="protocol_failure", turn=turn, total_actions=total_actions,
                        )
                        raise ProtocolFailure(
                            f"model did not produce a legal `observe` call within "
                            f"{cfg.run.max_protocol_retries} retries"
                        )

                    # VERIFY+PLAN must complete before ACT is reachable.
                    objective_check_required = level_boundary
                    post_mortem_required = step.state == "GAME_OVER"
                    note_check_required = wrote_notes_last_turn
                    messages.append(_stage_tools_message(prompts, tool_schemas))
                    verify_result, verify_reply, verify_tool_log = _get_verify_and_plan_call(
                        chat, messages, tool_schemas, notes, env, cfg,
                        objective_check_required=objective_check_required,
                        post_mortem_required=post_mortem_required,
                        note_check_required=note_check_required, window=window,
                        run_facts=run_facts,
                    )
                    if verify_result is None:
                        reasoning_text = _turn_memo(verify_reply, None) if verify_reply else ""
                        reported_change = _change_report(verify_reply) if verify_reply else ""
                        trace.log_turn(
                            game_id=game_id,
                            turn=turn,
                            play=None,
                            step=step,
                            turn_message=turn_text,
                            model_output=reasoning_text,
                            tool_calls=terminal_observe_tool_log + observe_tool_log + verify_tool_log,
                            reasoning_text=reasoning_text,
                            reported_change=reported_change,
                            usage={"output_tokens": 0, "total_tokens": 0},
                            turn_prompt_tokens=chat.prompt_tokens,
                            turn_completion_tokens=chat.completion_tokens,
                            thinking=verify_reply.reasoning_content if verify_reply else "",
                            raw_response_json=verify_reply.raw_response_json if verify_reply else "",
                            protocol_failure="no_verify_and_plan_call",
                            observe=observe_result,
                            terminal_observe=terminal_observe_result,
                        )
                        _write_run_summary(
                            run_dir, stop_reason="protocol_failure", turn=turn, total_actions=total_actions,
                        )
                        raise ProtocolFailure(
                            f"model did not produce a legal `verify_and_plan` call within "
                            f"{cfg.run.max_protocol_retries} retries"
                        )

                    # UPDATE runs only when VERIFY+PLAN's result (or a level boundary / game over)
                    # says the notes may be stale; otherwise no model call is made.
                    update_required = (
                        verify_result.verdict != "confirmed"
                        or level_boundary
                        or post_mortem_required
                        or verify_result.mode == "replan"
                        or verify_result.objects_changed
                        # Currently implied by level_boundary (TERMINAL OBSERVE only runs then);
                        # kept explicit for clarity.
                        or (terminal_observe_result is not None and terminal_observe_result.goal_changed)
                    )
                    update_happened = False
                    if update_required:
                        messages.append(_stage_tools_message(prompts, update_tool_schemas))
                        update_ok, update_reply, update_tool_log = _get_update_call(
                            chat, messages, update_tool_schemas, notes, env, cfg, window=window,
                        )
                        if not update_ok:
                            reasoning_text = _turn_memo(update_reply, None) if update_reply else ""
                            reported_change = _change_report(update_reply) if update_reply else ""
                            trace.log_turn(
                                game_id=game_id,
                                turn=turn,
                                play=None,
                                step=step,
                                turn_message=turn_text,
                                model_output=reasoning_text,
                                tool_calls=terminal_observe_tool_log + observe_tool_log + verify_tool_log + update_tool_log,
                                reasoning_text=reasoning_text,
                                reported_change=reported_change,
                                usage={"output_tokens": 0, "total_tokens": 0},
                                turn_prompt_tokens=chat.prompt_tokens,
                                turn_completion_tokens=chat.completion_tokens,
                                thinking=update_reply.reasoning_content if update_reply else "",
                                raw_response_json=update_reply.raw_response_json if update_reply else "",
                                protocol_failure="no_update_call",
                                verify_and_plan=verify_result,
                                observe=observe_result,
                                terminal_observe=terminal_observe_result,
                            )
                            _write_run_summary(
                                run_dir, stop_reason="protocol_failure", turn=turn, total_actions=total_actions,
                            )
                            raise ProtocolFailure(
                                f"model did not produce a legal `write_notes` call to finish "
                                f"UPDATE within {cfg.run.max_protocol_retries} retries"
                            )
                        update_happened = True

                click_mismatch = None  # set by _get_play_call when a click is forced through unconfirmed
                if is_memo:
                    play, reply, tool_call_log, reported_change = _get_play_call_memo(
                        chat, messages, step.available_actions, step.state, notes, cfg
                    )
                else:
                    messages.append(_stage_tools_message(prompts, act_tool_schemas))
                    play, reply, tool_call_log, click_mismatch = _get_play_call(
                        chat, messages, act_tool_schemas, step.available_actions, step.state, notes, env, cfg,
                        prompts, window=window, run_facts=run_facts,
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

            tool_call_log = (
                terminal_observe_tool_log + observe_tool_log + verify_tool_log + update_tool_log + tool_call_log
            )  # whole turn, all stages

            if is_memo:
                # The whole memo-mode reply is the turn memo, as written.
                reasoning_text = (reply.text or reply.reasoning_content or "") if reply else ""
            else:
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
                    verify_and_plan=verify_result,
                    observe=observe_result,
                    terminal_observe=terminal_observe_result,
                )
                _write_run_summary(
                    run_dir, stop_reason="protocol_failure", turn=turn, total_actions=total_actions,
                )
                raise ProtocolFailure(
                    f"model did not produce a legal `play` call within {cfg.run.max_protocol_retries} retries"
                )

            # The turn's messages are final once a legal `play` is in hand (`env.act()` appends
            # nothing to the conversation), so the turn record is closed here. On a protocol
            # failure the function has already raised and the turn never completes.
            window.end_turn()
            window.calibrate(reply.total_tokens, reply.output_tokens)

            reasoning = _reasoning_payload(reasoning_text, reply.output_tokens, reply.total_tokens)
            prev_levels_completed = step.levels_completed
            # The frame the model was just shown. Also read next turn as TERMINAL OBSERVE's
            # "before" image if this action completes a level.
            pre_action_frame = step.frames[-1]
            step = env.act(play.action, x=play.col, y=play.row, reasoning=reasoning)
            total_actions += 1
            actions_this_level += 1
            if play.action == "RESET":
                resets_this_level += 1
            level_boundary = step.levels_completed != prev_levels_completed
            wrote_notes_last_turn = update_happened if not is_memo else False
            # Checked against every frame this action returned, not just the settled last one, so
            # an animation that changes and reverts still counts as a change.
            changed = stagnation.sequence_changed(pre_action_frame, step.frames)
            frame_changed_note = (
                "Last action changed the frame." if changed else "Last action did not change the frame."
            )
            last_action_changed_frame = changed  # sharpens *next* turn's frames_note, same step
            click_mismatch_note = (
                f"Your last action clicked (x={play.x}, y={play.y}) intending "
                f"'{play.target_description}', but the click-target check could not confirm that "
                f"target was there after every retry: {click_mismatch} Check the resulting frame "
                f"to see what you actually affected."
            ) if click_mismatch else None
            repetition.record((play.action, play.col, play.row), changed)
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
                verify_and_plan=verify_result,
                observe=observe_result,
                terminal_observe=terminal_observe_result,
            )

            # -- the session cut (memory/session.py) -------------------------------------------
            # Everything the digest may later need about this turn, captured while it is in hand.
            run_facts.append(digest.TurnFacts(
                turn=turn, level=prev_levels_completed, action=play.action, x=play.x, y=play.y,
                target_description=play.target_description, expectation=play.expectation,
                reported_change=reported_change,
                verdict=verify_result.verdict if verify_result else None,
                frame_changed=changed, state=step.state, level_completed=level_boundary,
            ))
            if observe_result is not None:
                last_observed["description"] = observe_result.description
            if verify_result is not None and verify_result.plan:
                last_observed["plan"] = verify_result.plan

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


def _get_observe_call(chat: ChatClient, messages: list, observe_tool_schemas: list,
                       notes: NotesStore, env: EnvAdapter, cfg, *, terminal: bool = False,
                       window=None, run_facts=None):
    """Run the OBSERVE stage: call the model until it produces a legal `observe`, or exhaust the
    retry budget. Returns `(ObserveResult or None, last_reply, tool_call_log)`.

    Within one attempt the model may call the read-only tools in `observe_tool_schemas`
    (`read_notes`, `inspect`, `read_pixels`, `history`) up to `max_tool_calls_per_stage` times;
    `observe` ends the stage. No tool call at all, an invalid `observe`, or exhausting the budget
    triggers a corrective reprompt, up to `max_protocol_retries` times.

    `terminal=True` is TERMINAL OBSERVE: same protocol and tools, but the `observe` schema also
    requires `objective_reconfirmed` / `goal_changed`. On a level-boundary turn `run()` calls this
    twice: first with `terminal=True`, then the routine OBSERVE.
    """
    max_attempts = cfg.run.max_protocol_retries + 1
    last_reply = None
    tool_call_log: list = []
    non_terminal_calls = 0

    for attempt in range(max_attempts):
        while True:
            reply = chat.send(messages, observe_tool_schemas)
            last_reply = reply
            assistant_msg = {"role": "assistant", "content": reply.text or None}
            if reply.tool_calls:
                assistant_msg["tool_calls"] = [_tool_call_dict(tc) for tc in reply.tool_calls]
            messages.append(assistant_msg)

            observe_result = None
            saw_invalid_observe = False
            for tc in reply.tool_calls:
                if observe_result is not None:
                    messages.append({"role": "tool", "tool_call_id": tc.id, "content": "ok (unused: turn already ended)"})
                    continue
                if tc.name == "observe":
                    try:
                        observe_result = tools.validate_observe_args(tc.arguments, terminal=terminal)
                    except tools.ToolCallError as e:
                        messages.append({"role": "tool", "tool_call_id": tc.id, "content": str(e)})
                        saw_invalid_observe = True
                        continue
                    messages.append({"role": "tool", "tool_call_id": tc.id, "content": "ok"})
                else:
                    non_terminal_calls += 1
                    result, extra_messages = _dispatch_tool(tc, notes, env, cfg, window=window, run_facts=run_facts)
                    if result.startswith("unknown tool:"):
                        # Remind the model of the stage's terminal call immediately, not only
                        # after the attempt's tool-call budget is used up.
                        result = f"{result} You must call `observe` to continue your turn."
                    tool_call_log.append({"name": tc.name, "arguments": tc.arguments, "result": result})
                    messages.append({"role": "tool", "tool_call_id": tc.id, "content": result})
                    messages.extend(extra_messages)

            if observe_result is not None:
                return observe_result, reply, tool_call_log

            if saw_invalid_observe or not reply.tool_calls or non_terminal_calls >= cfg.run.max_tool_calls_per_stage:
                break  # -> reprompt below, consuming one attempt

        if attempt < max_attempts - 1:
            messages.append({"role": "user", "content": "You must call `observe` to continue your turn."})

    return None, last_reply, tool_call_log


def _get_verify_and_plan_call(chat: ChatClient, messages: list, tool_schemas: list,
                               notes: NotesStore, env: EnvAdapter, cfg, *,
                               objective_check_required: bool,
                               post_mortem_required: bool, note_check_required: bool,
                               window=None, run_facts=None):
    """Run the VERIFY+PLAN stage: call the model until it produces a legal `verify_and_plan`, or
    exhaust the retry budget. Returns `(VerifyAndPlanResult or None, last_reply, tool_call_log)`.

    Within one attempt the model may call `read_notes` / `write_notes` / `inspect` /
    `read_pixels` / `history` up to `max_tool_calls_per_stage` times; `verify_and_plan` ends the
    stage. No tool call at all, an invalid `verify_and_plan`, or exhausting the budget triggers a
    corrective reprompt, up to `max_protocol_retries` times. A `play` call here is rejected as an
    unknown tool: ACT is not reachable until VERIFY+PLAN completes.

    The `*_required` flags make the matching `verify_and_plan` field mandatory this turn.
    """
    max_attempts = cfg.run.max_protocol_retries + 1
    last_reply = None
    tool_call_log: list = []
    non_terminal_calls = 0

    for attempt in range(max_attempts):
        while True:
            reply = chat.send(messages, tool_schemas)
            last_reply = reply
            assistant_msg = {"role": "assistant", "content": reply.text or None}
            if reply.tool_calls:
                assistant_msg["tool_calls"] = [_tool_call_dict(tc) for tc in reply.tool_calls]
            messages.append(assistant_msg)

            verify_result = None
            saw_invalid_verify = False
            for tc in reply.tool_calls:
                if verify_result is not None:
                    messages.append({"role": "tool", "tool_call_id": tc.id, "content": "ok (unused: turn already ended)"})
                    continue
                if tc.name == "verify_and_plan":
                    try:
                        verify_result = tools.validate_verify_and_plan_args(
                            tc.arguments,
                            objective_check_required=objective_check_required,
                            post_mortem_required=post_mortem_required,
                            note_check_required=note_check_required,
                        )
                    except tools.ToolCallError as e:
                        messages.append({"role": "tool", "tool_call_id": tc.id, "content": str(e)})
                        saw_invalid_verify = True
                        continue
                    messages.append({"role": "tool", "tool_call_id": tc.id, "content": "ok"})
                else:
                    non_terminal_calls += 1
                    result, extra_messages = _dispatch_tool(tc, notes, env, cfg, window=window, run_facts=run_facts)
                    if result.startswith("unknown tool:"):
                        # Same immediate reminder as in _get_observe_call.
                        result = f"{result} You must call `verify_and_plan` to continue your turn."
                    tool_call_log.append({"name": tc.name, "arguments": tc.arguments, "result": result})
                    messages.append({"role": "tool", "tool_call_id": tc.id, "content": result})
                    messages.extend(extra_messages)

            if verify_result is not None:
                return verify_result, reply, tool_call_log

            if saw_invalid_verify or not reply.tool_calls or non_terminal_calls >= cfg.run.max_tool_calls_per_stage:
                break  # -> reprompt below, consuming one attempt

        if attempt < max_attempts - 1:
            messages.append({"role": "user", "content": "You must call `verify_and_plan` to continue your turn."})

    return None, last_reply, tool_call_log


def _get_update_call(chat: ChatClient, messages: list, update_tool_schemas: list,
                      notes: NotesStore, env: EnvAdapter, cfg, *, window=None):
    """Run the UPDATE stage: call the model until it produces a legal `write_notes` with
    `more_writes_expected: False`, or exhaust the retry budget. Returns `(ok, last_reply,
    tool_call_log)`. Only entered when `run()` decides UPDATE is required; only `write_notes` is
    offered.

    Each `write_notes` call is written immediately; the stage ends at the first one with
    `more_writes_expected == False`, so the model can write `GUIDE.md` and then `WORKING.md` in one
    stage. Any other tool call falls through to `_dispatch_tool`. An invalid `write_notes`, or
    exhausting the budget without an ending write, triggers a corrective reprompt, up to
    `max_protocol_retries` times.
    """
    max_attempts = cfg.run.max_protocol_retries + 1
    last_reply = None
    tool_call_log: list = []
    non_terminal_calls = 0

    for attempt in range(max_attempts):
        while True:
            reply = chat.send(messages, update_tool_schemas)
            last_reply = reply
            assistant_msg = {"role": "assistant", "content": reply.text or None}
            if reply.tool_calls:
                assistant_msg["tool_calls"] = [_tool_call_dict(tc) for tc in reply.tool_calls]
            messages.append(assistant_msg)

            stage_done = False
            saw_invalid_write = False
            for tc in reply.tool_calls:
                if stage_done:
                    messages.append({"role": "tool", "tool_call_id": tc.id, "content": "ok (unused: turn already ended)"})
                    continue
                if tc.name == "write_notes":
                    try:
                        write = tools.validate_write_notes_args(tc.arguments)
                    except tools.ToolCallError as e:
                        messages.append({"role": "tool", "tool_call_id": tc.id, "content": str(e)})
                        saw_invalid_write = True
                        continue
                    notes.write(write.file, write.content)
                    tool_call_log.append({"name": "write_notes", "arguments": tc.arguments, "result": "ok"})
                    messages.append({"role": "tool", "tool_call_id": tc.id, "content": "ok"})
                    if not write.more_writes_expected:
                        stage_done = True
                else:
                    non_terminal_calls += 1
                    result, extra_messages = _dispatch_tool(tc, notes, env, cfg, window=window)
                    if result.startswith("unknown tool:"):
                        # Same immediate reminder as in _get_observe_call.
                        result = (
                            f"{result} You must call `write_notes` at least once to finish "
                            f"updating your notes."
                        )
                    tool_call_log.append({"name": tc.name, "arguments": tc.arguments, "result": result})
                    messages.append({"role": "tool", "tool_call_id": tc.id, "content": result})
                    messages.extend(extra_messages)

            if stage_done:
                return True, reply, tool_call_log

            if saw_invalid_write or not reply.tool_calls or non_terminal_calls >= cfg.run.max_tool_calls_per_stage:
                break  # -> reprompt below, consuming one attempt

        if attempt < max_attempts - 1:
            messages.append({
                "role": "user",
                "content": "You must call `write_notes` at least once to finish updating your notes.",
            })

    return False, last_reply, tool_call_log


def _confirm_click_target(chat: ChatClient, env: EnvAdapter, prompts: PromptSet, cfg, play) -> "tools.ConfirmTargetResult":
    """Click-target confirmation: check that the click at `(play.x, play.y)` lands on what the
    model said it was clicking on (`play.target_description`).

    Crops the current frame around the click and asks the same model, in a separate throwaway
    request (no system prompt, no history -- just the crop and one question), whether the target
    is there. The crop never joins the main conversation, so it costs no context.

    Fails open: if the probe raises, or returns no legal `confirm_target` call, the click is
    treated as confirmed (with a `reason` saying verification was inconclusive). A wrong click that
    slips through is still caught by next turn's OBSERVE / VERIFY+PLAN against its `expectation`.
    """
    region = confirm_crop_region(play.x, play.y, cfg.image.confirm_crop_px, cfg.max_px)
    png = region_to_png_bytes(
        env.last.frames[-1], region, scale=cfg.image.scale, gridline_brightness=cfg.image.gridline_brightness,
    )
    question = prompts.click_confirm_prompt({"target_description": play.target_description})
    probe = [{"role": "user", "content": [{"type": "text", "text": question}, image_content(png)]}]
    try:
        reply = chat.send(probe, [tools.confirm_target_schema()])
        for tc in reply.tool_calls:
            if tc.name == "confirm_target":
                return tools.validate_confirm_target_args(tc.arguments)
    except Exception as exc:  # noqa: BLE001 -- deliberately broad: any failure here fails open
        return tools.ConfirmTargetResult(
            matches=True, inconclusive=True,
            reason=f"verification inconclusive ({type(exc).__name__}: {exc}), proceeding",
        )
    return tools.ConfirmTargetResult(
        matches=True, inconclusive=True,
        reason="verification inconclusive (no confirm_target call in the reply), proceeding",
    )


def _get_play_call(chat: ChatClient, messages: list, act_tool_schemas: list, available_actions: list,
                    state: str, notes: NotesStore, env: EnvAdapter, cfg, prompts: PromptSet, *,
                    window=None, run_facts=None):
    """Run the ACT stage: call the model until it produces a legal `play`, or exhaust the retry
    budget. Returns `(play or None, last_reply, tool_call_log, click_mismatch)`.

    Within one attempt the model may call `read_notes` / `inspect` / `read_pixels` / `history`, and
    `write_notes` restricted to `WORKING.md` with a required `step_check` (for marking a `## Plan`
    step done), up to `max_tool_calls_per_turn` times. `_dispatch_tool` enforces both restrictions,
    not just the schema. `play` ends the turn. No tool call at all, an invalid `play`, or exhausting
    the budget triggers a corrective reprompt, up to `max_protocol_retries` times.

    The click action (`ACTION6`) also requires `target_description`. Each click is checked by
    `_confirm_click_target`; if the target is not confirmed, the call is rejected like any invalid
    `play` and the model retries with corrected coordinates or description. On the *final* attempt
    only, a still-unconfirmed click is forced through instead, so a run does not end over a missed
    click; `click_mismatch` is then the probe's `reason` (else None), and `run()` turns it into the
    next turn's `click_mismatch_note`.
    """
    max_attempts = cfg.run.max_protocol_retries + 1
    last_reply = None
    tool_call_log: list = []
    non_play_calls = 0
    click_mismatch = None  # set only if a click is forced through on the final attempt

    for attempt in range(max_attempts):
        while True:
            reply = chat.send(messages, act_tool_schemas)
            last_reply = reply
            assistant_msg = {"role": "assistant", "content": reply.text or None}
            if reply.tool_calls:
                assistant_msg["tool_calls"] = [_tool_call_dict(tc) for tc in reply.tool_calls]
            messages.append(assistant_msg)

            play_result = None
            saw_invalid_play = False
            for tc in reply.tool_calls:
                if play_result is not None:
                    # The turn already ended (or already produced a valid play) earlier in this
                    # same response; every tool_call_id still needs a reply for the API contract.
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
                    if play.action == "ACTION6" and not play.target_description:
                        # Enforced here rather than in tools.validate_play_args: only native mode's
                        # click-target confirmation needs this field.
                        messages.append({
                            "role": "tool", "tool_call_id": tc.id,
                            "content": (
                                "`target_description` is required for the click-style action: a "
                                "short description of the object/feature you're clicking on."
                            ),
                        })
                        saw_invalid_play = True
                        continue
                    if play.action == "ACTION6":
                        confirmation = _confirm_click_target(chat, env, prompts, cfg, play)
                        # Fail open on the final attempt only, so a click that never passes the
                        # check does not end the run; earlier attempts reject normally so the
                        # model can correct its coordinates first.
                        forced = not confirmation.matches and attempt == max_attempts - 1
                        if forced:
                            click_confirm_result = f"forced (retries exhausted): {confirmation.reason}"
                        elif not confirmation.matches:
                            click_confirm_result = f"rejected: {confirmation.reason}"
                        elif confirmation.inconclusive:
                            click_confirm_result = f"inconclusive (proceeding): {confirmation.reason}"
                        else:
                            click_confirm_result = "confirmed"
                        tool_call_log.append({
                            "name": "click_confirm",
                            "arguments": {"x": play.x, "y": play.y, "target_description": play.target_description},
                            "result": click_confirm_result,
                            **({"forced_after_retries": True} if forced else {}),
                        })
                        if not confirmation.matches and not forced:
                            messages.append({
                                "role": "tool", "tool_call_id": tc.id,
                                "content": (
                                    f"target check failed: {confirmation.reason}. Try again with "
                                    f"corrected coordinates and/or a corrected description."
                                ),
                            })
                            saw_invalid_play = True
                            continue
                        if forced:
                            # Shown to the model next turn as `click_mismatch_note`.
                            click_mismatch = confirmation.reason
                            messages.append({
                                "role": "tool", "tool_call_id": tc.id,
                                "content": (
                                    f"ok (forced through: retries exhausted, target not confirmed "
                                    f"-- {confirmation.reason})"
                                ),
                            })
                            play_result = play
                            continue
                    messages.append({"role": "tool", "tool_call_id": tc.id, "content": "ok"})
                    play_result = play
                else:
                    non_play_calls += 1
                    result, extra_messages = _dispatch_tool(
                        tc, notes, env, cfg, allowed_write_files=_ACT_WRITABLE_FILES,
                        require_step_check=True, window=window, run_facts=run_facts,
                    )
                    if result.startswith("unknown tool:"):
                        # Same immediate reminder as in the other stages.
                        result = f"{result} You must call `play` to end your turn."
                    tool_call_log.append({"name": tc.name, "arguments": tc.arguments, "result": result})
                    messages.append({"role": "tool", "tool_call_id": tc.id, "content": result})
                    messages.extend(extra_messages)

            if play_result is not None:
                return play_result, reply, tool_call_log, click_mismatch

            # An invalid `play` counts against the retry budget like no call at all; otherwise a
            # model that keeps sending bad `play` arguments would loop here forever.
            if saw_invalid_play or not reply.tool_calls or non_play_calls >= cfg.run.max_tool_calls_per_turn:
                break  # -> reprompt below, consuming one attempt

        if attempt < max_attempts - 1:
            messages.append({"role": "user", "content": "You must call `play` to end your turn."})

    return None, last_reply, tool_call_log, None


def _get_play_call_memo(chat: ChatClient, messages: list, available_actions: list,
                         state: str, notes: NotesStore, cfg):
    """`tool_mode: memo` equivalent of `_get_play_call`. One model call per attempt with no tool
    schemas: the reply is a markdown memo that `memo.parse()` turns into the same argument shapes
    `tools.py` validates. A reply without a legal trailing action line gets a corrective reprompt,
    with the same retry budget as native mode.

    Returns `(play, reply, tool_call_log, reported_change)`; `reported_change` comes from
    `memo.parse()`.
    """
    max_attempts = cfg.run.max_protocol_retries + 1
    last_reply = None
    tool_call_log: list = []
    reported_change = ""

    for attempt in range(max_attempts):
        reply = chat.send(messages, [])
        last_reply = reply
        raw_text = reply.text or reply.reasoning_content or ""
        messages.append({"role": "assistant", "content": raw_text})

        parsed = memo.parse(raw_text)
        reported_change = parsed.reported_change

        play, error = None, None
        if parsed.play_args is None:
            error = "Your reply must end with exactly one legal action, alone on its own line."
        else:
            try:
                play = tools.validate_play_args(parsed.play_args, available_actions, cfg.image.scale, cfg.max_px, state)
            except tools.ToolCallError as e:
                error = str(e)

        if play is not None:
            for file, content in parsed.notes.items():
                write = tools.validate_write_notes_args({"file": file, "content": content})
                notes.write(write.file, write.content)
                tool_call_log.append({"name": "write_notes", "arguments": {"file": file}, "result": "ok"})
            return play, reply, tool_call_log, reported_change

        if attempt < max_attempts - 1:
            messages.append({"role": "user", "content": error})

    return None, last_reply, tool_call_log, reported_change


def _dispatch_tool(tc, notes: NotesStore, env: EnvAdapter, cfg, *, allowed_write_files: tuple = None,
                    require_step_check: bool = False, window=None, run_facts=None) -> tuple:
    """Execute a non-`play` tool call. Returns `(tool_message_content, extra_messages)`: most
    tools return only content; `inspect` also returns a follow-up `user` message carrying the
    requested images. Images travel only on `user` messages, never inside a `tool` message, so
    this works with providers that don't support images in tool results.

    `allowed_write_files` / `require_step_check` enforce ACT's restricted `write_notes` (only
    `WORKING.md`, `step_check` required); other stages pass neither. A model that ignores the
    offered schema is still rejected here.

    `window` enforces the per-turn image allowance (`session.max_images_per_turn`) across all
    stages. An `inspect` that would exceed it is rejected as a `ToolCallError`, so the model is
    told how many images remain and can retry with fewer views.
    """
    try:
        if tc.name == "read_notes":
            action = tools.validate_read_notes_args(tc.arguments)
            return notes.read(action.file) or "(empty)", []
        if tc.name == "write_notes":
            kwargs = {}
            if allowed_write_files is not None:
                kwargs["allowed_files"] = allowed_write_files
            if require_step_check:
                kwargs["require_step_check"] = True
            action = tools.validate_write_notes_args(tc.arguments, **kwargs)
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
            return _run_read_pixels(action, notes, env, cfg)
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


def _run_read_pixels(action: tools.ReadPixelsAction, notes: NotesStore, env: EnvAdapter, cfg) -> tuple:
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
        if v.registry_entry:
            lines.append(f"registry entry '{v.registry_entry}': {_lookup_registry_entry(notes, v.registry_entry)}")
    return "\n".join(lines), []


def _lookup_registry_entry(notes: NotesStore, name: str) -> str:
    """Best-effort lookup for `read_pixels`'s optional `registry_entry`: the lines of `WORKING.md`
    that mention `name` (case-insensitive). No particular notes structure is assumed."""
    working = notes.read("WORKING.md")
    if not working:
        return f"(no entry found for '{name}' -- WORKING.md is empty)"
    matches = [line for line in working.splitlines() if name.lower() in line.lower()]
    if not matches:
        return f"(no entry found for '{name}' in WORKING.md)"
    return "; ".join(matches)


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
    """Everything the model said in a reply apart from its tool arguments: its thinking output
    plus its final text. For the ACT reply this is logged as `reported_change`.

    Includes `reasoning_content` because a thinking model often says everything while thinking and
    leaves the final message empty.
    """
    parts = [p for p in (reply.reasoning_content, reply.text) if p]
    return "\n\n".join(parts)


def _turn_memo(reply, play=None) -> str:
    """The turn memo: what the model emitted in a reply, recorded with the action (as the SDK's
    `reasoning` payload) and in the trace's `reasoning` field.

    `play.expectation` comes first when given, since it is required and so always present; then
    the reply's thinking output and final text. Thinking output is always included here;
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


def _stage_tools_message(prompts, schemas: list) -> dict:
    """A user message, sent at the start of each stage, naming exactly the tools callable in that
    stage. Built from the same schema list handed to the stage's `chat.send()`, so it can only
    name tools actually offered."""
    names = ", ".join(s["function"]["name"] for s in schemas)
    return {"role": "user", "content": prompts.stage_tools_reminder({"stage_tools": names})}


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

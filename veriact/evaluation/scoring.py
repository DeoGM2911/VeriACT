"""Scoring, replicating the SDK's own formula (`arc_agi.scorecard.EnvironmentScoreCalculator`'s
`add_level` / `to_score`) so results are comparable with official ARC-AGI-3 scores.

Per level (`add_level`): a completed level scores `min(((baseline_actions / actions_taken) ** 2)
* 100, 115)`; beating the baseline can score above 100, capped at 115. An incomplete level always
scores 0, regardless of how many actions were spent on it.

Per environment (`to_score`): a weighted average of level scores, weighted by 1-indexed level
number (level 1 weight 1, level 2 weight 2, ...) -- later levels count for more, on the theory
that they are harder. The result is additionally capped at `(completed weight / total weight) *
100`, so a few over-100 levels early on cannot make a mostly-incomplete run look better than a
mostly-complete one.

This module only computes the score; it does not run anything. `levels_from_trace` reconstructs
the `(completed, actions_taken)` per level that `arc_agi.scorecard` gets from its own live
bookkeeping (`Card.actions_by_level`) from a harness `trace.jsonl` instead, since the harness
runs offline and does not use the SDK's scorecard system.
"""
import json
from pathlib import Path


def level_score(completed: bool, actions_taken: int, baseline_actions: int) -> float:
    if not completed or actions_taken <= 0:
        return 0.0
    return min(((baseline_actions / actions_taken) ** 2) * 100, 115.0)


def environment_score(levels: list) -> float:
    """`levels`: one dict per level, in level order (index 0 = level 1), each with `completed`,
    `actions_taken`, `baseline_actions`. Returns 0-100."""
    if not levels:
        return 0.0
    total_score = 0.0
    total_weight = 0
    completed_weight = 0
    for i, lvl in enumerate(levels):
        weight = i + 1  # 1-indexed level number, matching arc_agi.scorecard
        score = level_score(lvl["completed"], lvl["actions_taken"], lvl["baseline_actions"])
        total_score += score * weight
        total_weight += weight
        if score > 0:
            completed_weight += weight
    if total_weight == 0:
        return 0.0
    score = total_score / total_weight
    max_score = completed_weight / total_weight * 100
    return min(score, max_score)


def levels_from_trace(trace_path, baseline_actions: list) -> list:
    """Reconstruct per-level `(completed, actions_taken)` from a harness trace.jsonl, in the same
    shape `environment_score` consumes. Mirrors `arc_agi.scorecard`'s own reconstruction: actions
    are attributed to the level being played when they were taken (levels_completed at the start
    of that turn), a level is "completed" once levels_completed passes it (or the run ends in
    WIN, which completes whatever level was in progress), and any level never reached at all gets
    zero actions and no completion -- present in the list so it still dilutes the environment
    score, exactly like a game the model never got to.
    """
    num_levels = len(baseline_actions)
    levels: list = []
    actions_since_last_completion = 0
    levels_completed_before = 0
    won = False

    for row in _turn_rows(trace_path):
        actions_since_last_completion += 1
        lc = row["data"]["levels_completed"]
        while lc > levels_completed_before and len(levels) < num_levels:
            levels.append({
                "completed": True,
                "actions_taken": actions_since_last_completion,
                "baseline_actions": baseline_actions[len(levels)],
            })
            actions_since_last_completion = 0
            levels_completed_before += 1
        if row["data"]["state"] == "WIN":
            won = True

    if won and len(levels) < num_levels:
        levels.append({
            "completed": True,
            "actions_taken": actions_since_last_completion,
            "baseline_actions": baseline_actions[len(levels)],
        })
        actions_since_last_completion = 0

    if len(levels) < num_levels:
        # The level in progress when the run ended -- whatever it spent, it didn't finish.
        levels.append({
            "completed": False,
            "actions_taken": actions_since_last_completion,
            "baseline_actions": baseline_actions[len(levels)],
        })
    while len(levels) < num_levels:  # levels never even reached
        levels.append({"completed": False, "actions_taken": 0, "baseline_actions": baseline_actions[len(levels)]})

    return levels


def _turn_rows(trace_path) -> list:
    rows = []
    for line in Path(trace_path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if "data" in row:  # skips cut events, which carry "event" instead
            rows.append(row)
    return rows

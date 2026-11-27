"""Frame-change and repetition checks computed by the harness.

`sequence_changed` decides whether an action changed the frame; the result is stated in the next
turn's message ("Last action changed the frame.") and logged as `frame_changed`.

`RepetitionTracker` detects a repeated action or an alternating pair and states it as a fact in
the turn message, so the model does not have to notice repetition from its own visual read.
"""
from typing import Optional

import numpy as np


def frame_changed(before: np.ndarray, after: np.ndarray) -> bool:
    return not np.array_equal(before, after)


def sequence_changed(before: np.ndarray, frames: list) -> bool:
    """Whether any frame an action returned differs from the frame shown before it -- not just the
    settled last one. An action can return several frames (an animation, a chain reaction, a level-
    end sequence); comparing only the endpoints misses a change that happens and then reverts within
    that same sequence (e.g. something moves and moves back before the frame settles)."""
    return any(frame_changed(before, f) for f in frames)


class RepetitionTracker:
    """A short window of recent `(action_identity, changed)` pairs; flags a repeated action, or an
    alternating pair, once it has held for `min_repeats` or more. `window` / `min_repeats` come
    from config and are the same for every game.

    `changed` does not gate or reset the streak: some games redraw an unrelated element (a timer,
    a step counter) into every frame, so `changed` can be `True` on every turn of a stuck loop, and
    no game-agnostic rule can tell that apart from real progress. So `changed` is reported
    alongside the streak and the model judges whether the changes are real.

    `window` only bounds how far back a *new* pattern can be recognized; once a pattern is locked
    in, its length keeps growing for as long as it holds.
    """

    def __init__(self, window: int, min_repeats: int):
        self._window = window
        self._min_repeats = min_repeats
        self._history: list = []  # [(action_identity, changed), ...], most recent last
        self._streak_period: Optional[int] = None
        self._streak_pattern: Optional[list] = None  # identities, most-recent-first
        self._streak_length: int = 0
        self._streak_changed_recently: bool = False

    def record(self, action_identity, changed: bool) -> None:
        self._history.append((action_identity, changed))
        self._history = self._history[-self._window:]
        self._extend_or_reset_streak(action_identity, changed)

    def _extend_or_reset_streak(self, action_identity, changed: bool) -> None:
        if self._streak_period is not None:
            expected = self._streak_pattern[self._streak_length % self._streak_period]
            if action_identity == expected:
                self._streak_length += 1
                self._streak_changed_recently = self._streak_changed_recently or changed
                return
        self._streak_period = None
        self._streak_pattern = None
        self._streak_length = 0
        self._streak_changed_recently = False
        for period in (1, 2):
            pattern = self._detect_period(period)
            if pattern is not None:
                needed = period * self._min_repeats
                self._streak_period = period
                self._streak_pattern = pattern
                self._streak_length = needed
                self._streak_changed_recently = any(c for _, c in self._history[-needed:])
                return

    def _detect_period(self, period: int) -> Optional[list]:
        """`period` identities, most-recent-first, if the last `period * min_repeats` entries of
        `_history` are exactly that pattern repeating; `None` otherwise."""
        needed = period * self._min_repeats
        if len(self._history) < needed:
            return None
        pattern_from_end = [self._history[-1 - k][0] for k in range(period)]
        for k in range(needed):
            identity, _ = self._history[-1 - k]
            if identity != pattern_from_end[k % period]:
                return None
        return pattern_from_end

    def note(self) -> Optional[str]:
        """A fact string once the live streak reaches `min_repeats`, else `None`. Checked shortest
        period first, so a period-1 repeat is reported as such rather than also matching as a
        degenerate period-2 alternation."""
        if self._streak_period is None:
            return None
        names = [_describe(identity) for identity in reversed(self._streak_pattern)]
        caveat = (
            " The frame has reportedly changed during that stretch -- check whether that's real "
            "progress or something incidental (e.g. a timer, counter, or animation) rather than "
            "an actual change in the puzzle."
            if self._streak_changed_recently else
            " No visible change has been reported during that stretch."
        )
        if self._streak_period == 1:
            return f"{names[0]} has been repeated {self._streak_length} times in a row.{caveat}"
        return (
            f"{names[0]} and {names[1]} have been alternating for the last {self._streak_length} "
            f"actions.{caveat}"
        )


def _describe(action_identity) -> str:
    action, col, row = action_identity
    if col is None and row is None:
        return action
    return f"{action} ({col}, {row})"

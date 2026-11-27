"""Environment adapter: wraps the ARC-AGI-3 SDK, stores every frame, and writes the PNG archive.

Wraps `arc_agi.Arcade().make(game_id)` in offline mode. Every step is recorded under a frame ref
`T{turn}` from a run-global counter that never resets (a level-local numbering would collide when
a level is restarted). Each step's raw grids go into an in-memory store (flushed to
`frame_store.npz` on `close()`) and are rendered to `runs/<id>/frames/*.png`, indexed by
`frames/manifest.json`. `inspect` / `read_pixels` look frames up here by `(turn, frame)`.
"""
import dataclasses
import json
from pathlib import Path
from typing import Any, Optional

import numpy as np

import arc_agi
from arc_agi import OperationMode
from arcengine import GameAction

from baseline.perception.observation import frame_to_png_bytes

# Default location of the downloaded games: `<repo>/environment_files`, resolved from this file
# rather than the SDK's CWD-relative default so it works from any working directory. Override with
# config `paths.environments_dir` or `--environments-dir`.
_DEFAULT_ENVIRONMENTS_DIR = str(Path(__file__).resolve().parents[2] / "environment_files")


def list_available_games(environments_dir: Optional[str] = None) -> list:
    """Every game id `EnvAdapter`/`arc.make` can resolve locally: one per subdirectory of
    `environments_dir` (each holds `<game_code>/<hash>/`), sorted for a stable order. Used by
    `eval.py --all-games`.
    """
    environments_dir = environments_dir or _DEFAULT_ENVIRONMENTS_DIR
    return sorted(p.name for p in Path(environments_dir).iterdir() if p.is_dir())


def _visible_actions(state: str, raw_names: list) -> list:
    """What `available_actions` should show the model this turn. While `GAME_OVER`, `RESET` is
    the only action that does anything (arcengine freezes the game until it) but the SDK's own
    `action_space` doesn't reflect that -- so this overrides the raw list to `["RESET"]` rather
    than offering actions that will just be rejected."""
    if state == "GAME_OVER":
        return ["RESET"]
    return raw_names


@dataclasses.dataclass
class StepResult:
    ref: str  # "T{turn}", the run-global opaque handle for this step
    frames: list  # every raw frame this step returned (list[np.ndarray]); frames[-1] is settled
    state: str  # GameState name: NOT_PLAYED | NOT_FINISHED | WIN | GAME_OVER
    levels_completed: int
    win_levels: int
    available_actions: list  # action names, e.g. "ACTION1", "ACTION6"
    guid: Optional[str]
    full_reset: bool
    action: Optional[str]  # the action name that produced this frame; None for the initial reset
    level: int  # levels_completed as of the start of this step
    turn_in_level: int
    attempt: int


class EnvAdapter:
    def __init__(self, game_id: str, run_dir, image_scale: int = 8, gridline_brightness: float = 0.75,
                 environments_dir: Optional[str] = None):
        # Always offline: games are loaded from `environments_dir`, never fetched (the competition
        # evaluator has no internet access).
        environments_dir = environments_dir or _DEFAULT_ENVIRONMENTS_DIR
        self._arc = arc_agi.Arcade(operation_mode=OperationMode.OFFLINE, environments_dir=environments_dir)
        env = self._arc.make(game_id, render_mode="")
        if env is None:
            raise RuntimeError(
                f"unknown game id: {game_id!r} (scanned environments_dir={environments_dir!r})"
            )
        self._env = env

        self._run_dir = Path(run_dir)
        self._frames_dir = self._run_dir / "frames"
        self._frames_dir.mkdir(parents=True, exist_ok=True)
        self._manifest_path = self._frames_dir / "manifest.json"
        self._manifest: list = []
        self._store: dict = {}

        self._image_scale = image_scale
        self._gridline_brightness = gridline_brightness

        self._turn = 1  # ref numbering starts at T1 so it matches loop.py's 1-indexed turn display
        self._levels_completed = 0
        self._turn_in_level = 0
        self._attempt = 1
        self._frame_counts: dict = {}  # turn (int) -> number of frames archived
        # {raw_value: symbol}, persistent for the run so a colour keeps its read_pixels letter
        # across calls instead of being relabelled every time.
        self._pixel_symbols: dict = {}

        # LocalEnvironmentWrapper.__init__ already performed the initial reset.
        self._last: StepResult = self._finalize(self._env.observation_space, action=None)

    @property
    def last(self) -> StepResult:
        return self._last

    @property
    def pixel_symbols(self) -> dict:
        """The run-persistent `{raw_value: symbol}` table `read_pixels` reuses."""
        return self._pixel_symbols

    @property
    def baseline_actions(self) -> list:
        """Per-level reference action counts from the environment's own metadata -- used for the
        per-level action budget and scoring only, never shown to the model. Empty if this environment
        doesn't publish one (`EnvironmentInfo.baseline_actions` is `Optional`)."""
        return list(self._env.environment_info.baseline_actions or [])

    def reset(self, reasoning: Optional[dict] = None) -> StepResult:
        resp = self._env.reset()
        if resp is None:
            raise RuntimeError("env.reset() returned None")
        return self._finalize(resp, action="RESET")

    def act(self, action_name: str, x: Optional[int] = None, y: Optional[int] = None,
            reasoning: Optional[dict] = None) -> StepResult:
        action_name = action_name.upper()
        if action_name == "RESET":
            return self.reset(reasoning=reasoning)

        action = GameAction.from_name(action_name)
        data = {"x": x, "y": y} if action.is_complex() else None
        resp = self._env.step(action, data=data, reasoning=reasoning)
        if resp is None:
            raise RuntimeError(f"env.step({action_name}) returned None")
        return self._finalize(resp, action=action_name)

    def close(self) -> None:
        """Flush the raw frame store to `frame_store.npz`."""
        if self._store:
            np.savez_compressed(self._run_dir / "frame_store.npz", **self._store)

    # -- frame archive lookup (used by the inspect / read_pixels / history tools) ---

    def frame_count(self, turn: int) -> int:
        """Number of frames (settled + animation sub-frames) archived for `turn`."""
        if turn not in self._frame_counts:
            raise KeyError(f"no frame archived for turn {turn}")
        return self._frame_counts[turn]

    def get_frame(self, turn: int, frame: int) -> np.ndarray:
        """The raw int grid for `(turn, frame)` -- the same addressing the archive manifest uses."""
        return self._store[self._frame_key(turn, frame)]

    def frame_filename(self, turn: int, frame: int) -> str:
        """The archived PNG's filename for `(turn, frame)`, relative to `runs/<id>/frames/`."""
        return f"{self._frame_key(turn, frame)}.png"

    def _frame_key(self, turn: int, frame: int) -> str:
        count = self.frame_count(turn)  # raises KeyError for an unknown turn
        if not (0 <= frame < count):
            raise IndexError(f"turn {turn} has {count} frame(s); frame {frame} is out of range")
        return f"T{turn}" if count == 1 else f"T{turn}_{frame}"

    # -- internals ---------------------------------------------------------

    def _wrap(self, resp, action: Optional[str]) -> StepResult:
        state = resp.state.name if hasattr(resp.state, "name") else str(resp.state)
        available = [GameAction.from_id(a).name for a in resp.available_actions]
        return StepResult(
            ref=f"T{self._turn}",
            frames=list(resp.frame),
            state=state,
            levels_completed=resp.levels_completed,
            win_levels=resp.win_levels,
            available_actions=_visible_actions(state, available),
            guid=resp.guid,
            full_reset=resp.full_reset,
            action=action,
            level=self._levels_completed,
            turn_in_level=self._turn_in_level,
            attempt=self._attempt,
        )

    def _finalize(self, resp, action: Optional[str]) -> StepResult:
        result = self._wrap(resp, action)
        self._archive(result)
        self._last = result
        self._turn += 1

        if result.full_reset:
            self._attempt, self._turn_in_level = 1, 0
        elif action == "RESET":
            self._attempt += 1
            self._turn_in_level = 0
        elif result.levels_completed != self._levels_completed:
            self._levels_completed = result.levels_completed
            self._turn_in_level = 0
            self._attempt = 1
        else:
            self._turn_in_level += 1

        return result

    def _archive(self, result: StepResult) -> None:
        label = f"LVL {result.level} TURN {result.turn_in_level}"
        n = len(result.frames)
        self._frame_counts[int(result.ref[1:])] = n
        for idx, frame in enumerate(result.frames):
            key = result.ref if n == 1 else f"{result.ref}_{idx}"
            self._store[key] = frame.astype(np.int8)
            filename = f"{key}.png"
            png = frame_to_png_bytes(frame, scale=self._image_scale, gridline_brightness=self._gridline_brightness)
            (self._frames_dir / filename).write_bytes(png)
            self._manifest.append(
                {
                    "ref": result.ref,
                    "frame": idx,
                    "label": label,
                    "file": filename,
                    "level": result.level,
                    "turn_in_level": result.turn_in_level,
                    "attempt": result.attempt,
                    "action": result.action,
                    "state": result.state,
                }
            )
        self._manifest_path.write_text(json.dumps(self._manifest, indent=2), encoding="utf-8")

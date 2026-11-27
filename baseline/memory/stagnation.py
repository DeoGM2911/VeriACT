"""Frame-change checks computed by the harness.

The result is logged to the trace as `frame_changed` (and appears in the digest as "frame
unchanged"); it is never shown to the model as a separate hint.
"""
import numpy as np


def frame_changed(before: np.ndarray, after: np.ndarray) -> bool:
    return not np.array_equal(before, after)


def sequence_changed(before: np.ndarray, frames: list) -> bool:
    """Whether any frame an action returned differs from the frame shown before it -- not just the
    settled last one. An action can return several frames (an animation, a chain reaction, a level-
    end sequence); comparing only the endpoints misses a change that happens and then reverts within
    that same sequence (e.g. something moves and moves back before the frame settles)."""
    return any(frame_changed(before, f) for f in frames)

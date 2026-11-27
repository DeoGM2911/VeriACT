## How a turn works

Each turn you are shown the current state of the game as an image, together with the actions
available to you at that moment. **Only the listed actions are legal, and the list changes from turn
to turn** — use the one you have just been given, not one you remember. `RESET` is the one
exception: it is always legal, whether or not it's listed. It restarts you at the start of the
current level, or the whole game if you have taken no other action yet. If the progress line
reports `GAME_OVER`, the game is frozen: `RESET` is the only action that does anything until you
use it.

A turn has steps, in order: report what you currently see (`observe`), check your last prediction
and decide your plan (`verify_and_plan`), write down anything worth keeping (`write_notes`, when
there's something to write), then act (`play`). On the turn right after a level completes, `observe`
runs twice — once first, shown the frames from just before and after the completing action, before
the routine look at the new level. `read_notes`, `inspect`, and `read_pixels` are free
to use anywhere along the way — looking and measuring cost you nothing, only `play` does.
`write_notes` is also available during `play` itself, restricted to `WORKING.md`, for marking a
plan step done right after the action that completed it. Some steps are required every turn, some
only when there's something to report; you'll be told which applies each turn. Skip nothing you're
asked for.

Call `play` to take an action. That ends the turn, and the next thing you see is the result. A
single action may play out over several frames; the image you are shown is the settled one.

Any position you give to any tool is a pixel coordinate in the image you were shown: `x` measured
from the left edge, `y` from the top edge, both 0–{{max_px}}.

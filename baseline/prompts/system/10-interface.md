## How a turn works

Each turn you are shown the current state of the game as an image, together with the actions
available to you at that moment. **Only the listed actions are legal, and the list changes from turn
to turn** — use the one you have just been given, not one you remember. `RESET` is the one
exception: it is always legal, whether or not it's listed. It restarts you at the start of the
current level, or the whole game if you have taken no other action yet. If the progress line
reports `GAME_OVER`, the game is frozen: `RESET` is the only action that does anything until you
use it.

Call `play` to take an action. That ends the turn, and the next thing you see is the result. A
single action may play out over several frames; the image you are shown is the settled one.

**Before each `play`, state what you expect the next image to show** — specifically enough to be
wrong. **After you see a result, state what actually changed**, expected or not. `play` takes an
`expectation` field for the first; the second is just a sentence or two at the start of your next
turn.

`read_notes`, `write_notes`, `inspect`, and `read_pixels` are free to use, in any order, as often
as useful — looking, measuring, and note-keeping cost you nothing. Only `play` advances the game.

Any position you give to any tool is a pixel coordinate in the image you were shown: `x` measured
from the left edge, `y` from the top edge, both 0–{{max_px}}.

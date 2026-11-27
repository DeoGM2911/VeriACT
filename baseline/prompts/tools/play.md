Take one game action. This is the only tool that advances the game, and the only one that costs
anything. It ends the turn.

`action` must be one of the actions listed as available **this turn**. That list changes; an action
offered earlier may not be offered now. `RESET` is always legal too, listed or not.

`x` and `y` are required for the click-style action and must be omitted for the others. They are
pixel coordinates in the image you were shown — `x` from the left edge, `y` from the top edge, both
0–{{max_px}}.

`expectation` is required: state what you expect the next image to look like. Being specific enough
to be wrong is the point — a prediction that cannot fail teaches you nothing when it holds.

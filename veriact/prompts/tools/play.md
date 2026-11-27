Take one game action. This is the only tool that advances the game, and the only one that costs
anything.

`action` must be one of the actions listed as available **this turn**. That list changes; an action
offered earlier may not be offered now. `RESET` is always legal too, listed or not.

`x` and `y` are required for the click-style action and must be omitted for the others. They are
pixel coordinates in the image you were shown — `x` from the left edge, `y` from the top edge, both
0–{{max_px}}.

`expectation` is required: state what you expect the next image to look like. Being specific enough
to be wrong is the point: a prediction that cannot fail teaches you nothing when it holds.

For the click-style action specifically, `target_description` is also required: a short description
of the object/feature you're clicking on (e.g. "the blue switch"). The harness checks a close-up
crop of where you pointed against that description before the action is taken; if it doesn't match,
this call is rejected and you can try again with corrected coordinates and/or a corrected
description, the same as any other invalid `play` call.

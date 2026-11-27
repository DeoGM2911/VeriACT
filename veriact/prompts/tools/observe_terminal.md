Report what changed between the two frames you were just shown -- the one right before your last
action, and the one right after it. This runs once, only on the turn right after a level completes,
before the routine look at the new level's opening frame.

`description` -- what changed between the two frames, tied to the specific objects/features you're
already tracking by their established names. This is your strongest evidence for what "winning"
actually looked like -- look for the last thing to move, appear, disappear, or align before the
level ended.

`objective_reconfirmed` -- restate what you now believe the goal is, using this evidence. If it
matches what you already believed, say so plainly rather than inventing a new theory.

`goal_changed` -- `true` if this evidence changes or sharpens your previous understanding of the
goal, `false` if it just confirms what you already believed.

Free to spend a `read_notes`, `inspect`, or `read_pixels` call first. `write_notes` and `play` are
not available here, same as the routine `observe`.

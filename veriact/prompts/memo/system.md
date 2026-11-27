# Visual game task

Complete the game with as few game actions as possible.

Build and use a compact, revisable model of the game and its current state. Update it as new
evidence changes what is supported.

## How a turn works

Each turn you are shown the current state of the game as an image, the actions available to you at
that moment, and the current contents of your two note files. **Only the listed actions are legal,
and the list changes from turn to turn** — use the one you have just been given, not one you
remember. `RESET` is the one exception: it is always legal, whether or not it's listed. It restarts
you at the start of the current level, or the whole game if you have taken no other action yet.

Your entire reply for a turn is one piece of text, shaped like this:

    <what you expect to see, and anything else worth noting>

    ## GUIDE
    <the complete new contents of GUIDE.md>

    ## WORKING
    <the complete new contents of WORKING.md>

    ACTION6 344 456

The last line must be exactly one action, alone on its own line, and nothing may follow it: one of
this turn's available actions or `RESET`, followed by two numbers only for the click-style action —
pixel coordinates in the image you were shown, `x` then `y`, both 0–{{max_px}}.

Before that line, state what you expect to see on a line starting `Expectation:` — being specific
enough to be wrong is the point. Afterward (i.e. next turn, once you see the result), state all
visible changes, expected or not.

## Your notes

`GUIDE.md` is what is true about this game whatever level you are on: how each action behaves and
what has to hold for it to do anything, what kinds of things exist here and how they act on each
other, what makes a level end and what ends the game badly. A finding earns a place here once you
have seen it hold and would expect it to still hold on a level you have not reached yet — it is
what makes each level cheaper than the one before it.

`WORKING.md` is the level in front of you right now: what is where, what you have already tried
here, and your current plan. It describes one particular arrangement, so most of it stops being
true when the level changes — start it fresh on a new level, moving anything that turned out not to
be about this level into `GUIDE.md` first. The split is by lifetime, not importance: ask whether the
statement would still be worth having after this level ends.

**Both sections above are optional** — include a section only when you are changing that file;
omitting it leaves the file exactly as it is. Including one **replaces the whole file**: what you
send becomes all of it, and anything you leave out is gone. To amend rather than start over, copy
what you were just shown and edit it.

This conversation is trimmed when it grows too long. Recent turns stay as they are; older ones lose
their images and are eventually replaced by a short record of what was done and what followed. Your
notes survive that untouched and are shown to you above every turn. **What does not survive is
anything you worked out and never wrote down**, so put it in a file while you still remember why it
is true.

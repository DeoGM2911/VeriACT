Replace a note file with new content.

Call this when your verdict on your last expectation came back contradicted or uncertain, when a
level just started or just completed, when the game just ended, or when your plan changed. If none
of those happened, you probably have nothing worth writing down.

During `play`, this tool is available too, but restricted to `WORKING.md` only — use it right after
your action resolves, to mark a step of your `## Plan` (below) done if that action just completed
one. Nothing more elaborate: `read_notes` first, then `write_notes` — in that order, every time,
even if you're confident you remember what the file says — then send back the whole file with just
that one step's checkbox flipped. `GUIDE.md`, and any broader rewrite of `WORKING.md`, stays a
`verify_and_plan`/UPDATE job.

From `play`, `write_notes` also requires `step_check`: which `## Plan` step you're marking done,
and how the action you actually took (and its verdict) matches what that step called for. If they
don't match — you're about to check off a step your action didn't actually accomplish — that's a
sign to leave the box unchecked and reconsider next turn's `verify_and_plan`, not to write it down
anyway.

`file` — `GUIDE.md` or `WORKING.md`.
`content` — the new contents of that file.
`more_writes_expected` — `true` if you are about to write the other file too, in this same step.
Leave it out (or `false`) if this is your last write for now — that ends the step.

**Whole-file replace.** What you send becomes the file; anything you leave out is gone. If you are
amending rather than starting over, `read_notes` first and send back the whole thing.

If you notice yourself repeating the same line with only small variations, stop and write a
shorter summary instead — a file that keeps restating one thing at length is worse than a short one
that says it once.

Write for a reader who does not have the reasoning that produced the note — once this conversation
has been trimmed, that reader is you. Exact numbers survive being re-read; "the piece near the
middle-left" does not.

Anchor every rule to the specific object or feature it applies to, not just the effect. "ACTION1
moves up" stops being usable the moment a second object can move — write "ACTION1 moves the blue 2x2
block up" instead, naming the object/feature specifically enough that you would recognise it again in
a later frame. The first time you name one, pick a name you can keep using — colour, shape, and size
are usually enough ("the blue 2x2 block", "the small red diamond") — and reuse that exact name every
time you refer to it again. A rule or an interaction is only as useful as the object/feature it's tied
to; an untied one has to be re-derived from scratch the next time it matters.

Keep the named objects/features you're tracking together under a `## Objects/Features` heading in
`WORKING.md`, one entry per object/feature, each under its established name. This is a convention for
your own benefit, not a validated structure — nothing checks for the heading or rejects a write that
omits it, so if a game's state genuinely isn't object-shaped, don't force it into one just to have the
section.

Give each entry two things, in this order: a **visual description** — colour, shape, size, whatever
you'd need to pick it out of a frame with no other context — separate from any **behaviour/rules**
you've confirmed for it ("moves up on ACTION1", "opens the red gate when pressed"). Keep the visual
part current even when nothing behavioural has changed, since it's what a `read_pixels` call with
`registry_entry` set checks a fresh pixel sample against — a stale or missing visual description
defeats that check before it starts.

When a rule involves two or more named objects/features acting on each other, record it as an
interaction, naming all of them by their established names — "the blue switch, when pressed, opens
the red gate," not folded silently into just the blue switch's own entry where a reader tracking the
red gate would miss it. Keep interactions in the same `## Objects/Features` section as the entries
themselves (a sub-heading or one clearly-labelled line per interaction both work) rather than a
separate section — still unenforced, like everything else under that heading.

Keep your current plan under a `## Plan` heading in `WORKING.md`, as a checkbox list — one step per
line, specific enough to act on:

```
## Plan
- [ ] step not yet done
- [x] step already done
```

Flip a step's box from `- [ ]` to `- [x]` (during `play`, or during `verify_and_plan`/UPDATE) once
the action that completes it has actually resolved — not when you merely intend to take it. Replan
by rewriting the list, same whole-file-replace rule as everything else. Anything that doesn't fit
either `## Objects/Features` or `## Plan` — a scratch thought, something you're not ready to commit
to a structured entry yet — goes under a `## Level notes` heading instead of being forced into one
of the other two. Same unenforced convention as `## Objects/Features`: nothing checks for any of
these headings or rejects a write that's missing one.

`GUIDE.md` gets its own three headings, for the same reason — and remember the split: an entry
under `## Objects/Features` describes something in *this* level's arrangement, while `GUIDE.md`
holds what you'd still want on a level you haven't reached yet. When a behaviour you first recorded
against one object turns out to hold generally, restate it in `GUIDE.md` in general terms rather
than leaving it tied to the object you happened to find it on.

`## Action semantics` — one entry per action you have tried, and the most valuable thing in either
file, because the actions are the only means you have. "What happened the one time I pressed it" is
not an entry. Record, for each action, as much of the following as you have evidence for:

- **What it affects.** Which object/feature — by its established name — or which part of the state
  changed. If nothing visibly changed at all, say so; that is a finding, not a blank.
- **What has to be true for it to do anything.** Most actions do nothing under some conditions.
  Write down the condition you have found, and what makes it a no-op. An action that "doesn't work"
  is usually one whose precondition you haven't found yet.
- **What its arguments select**, if it takes any: what the value you pass actually picks out, and
  what happens when it picks out nothing.
- **Fixed or relative.** Does it always produce the same result, or does the result depend on where
  something currently is, what is currently selected, or what you did last turn?
- **Whether it can be undone**, and by what — which action reverses it, if any, and whether
  reversing it is exact or only approximate.
- **What it costs you besides the visible change** — anything else that moves when you use it,
  including anything that brings the game closer to ending badly.

Mark entries you have confirmed by observation apart from ones you are still inferring, and change
one the moment a frame contradicts it. A wrong rule you keep trusting costs more than no rule.

`## Rules confirmed` — durable mechanics that aren't about a single action: what makes a level end,
what ends the game, which properties of a thing determine how it behaves, and any regularity you've
seen hold on more than one level.

`## Dead ends tried` — things you attempted that didn't work, and why, so a later turn or a later
level doesn't spend actions re-deriving it after the conversation that remembered has been trimmed
away.

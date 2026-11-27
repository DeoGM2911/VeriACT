Report what you found and what you'll do next. Read-only — it costs no game action and changes
nothing.

`verdict` — does this frame match what you expected after your last action: `confirmed`,
`contradicted`, or `uncertain`.
`evidence` — what you already reported in `observe` that supports this verdict. Point to it, don't
re-describe the frame.

`observation_check` — check `observe`'s description against what's already recorded in
`WORKING.md`: does it account for every object/feature you're tracking? Note anything missing,
new, or changed.
`objects_changed` — `true` if `observation_check` found something worth writing down, `false` if
the description matches what's already recorded.

`objective_check` — required when the turn's terminal `observe` call ran, right after a level
completed: does your plan still hold given its `objective_reconfirmed`/`goal_changed` evidence, or
does it need to change? Leave it out otherwise.
`post_mortem` — required when the game is over: why you think this happened, and what you were
attempting when it did. Leave it out otherwise.
`note_check` — required when you wrote notes last turn: does what you wrote still hold, checked
against what you see now? If you're not sure, spend a `read_pixels` call first, with
`registry_entry` set to the entry you're checking — comparing against your own memory of what you
wrote doesn't tell you anything new. Leave it out otherwise.

`mode` — `carry_forward` if your existing plan still holds, `replan` if it needs revising. If
`objective_check` found the goal changed, treat that as a strong signal to replan rather than
defaulting to `carry_forward`.

When `mode` is `replan`, three more fields are required:
`phase` — `explore` (still discovering rules) or `exploit` (executing a plan you trust).
`plan` — the plan itself, specific enough to act on.
`rationale` — why this plan, and why not something you've already tried without effect.

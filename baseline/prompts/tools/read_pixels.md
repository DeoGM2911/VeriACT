Use this whenever "somewhere around X" isn't precise enough to act on.

Sample exact colours out of a region and get them back as a grid of symbols. Read-only — it costs no
game action and changes nothing.

This reads the colours underneath the picture, not the picture itself — the thin darker boundary
lines drawn over the image are not part of what comes back. Where two things overlap, a sample can
only return whichever one is in front.

You choose the region **and how finely to sample it**. `rows` × `columns` samples are taken evenly
across the region and returned as `rows` strings of `columns` characters, together with a `palette`
mapping each character to the colour it stands for. Each row is labelled with the image-pixel `y`
of its samples, and the column range gives their `x` spacing — so a symbol's place in the grid
maps straight back to a coordinate you can act on.

If the reply says the sampling **did not land on every cell in the region**, you asked for fewer
`rows`/`columns` than the region has detail: what came back is a summary, and a boundary read off
it can be a cell or two out. That is fine for seeing gross structure; sample a smaller region, or
raise `rows`/`columns`, before you commit to an exact position.

**The palette is built for that one call**, from the colours actually present in what you sampled,
in a fixed but arbitrary order. A character means nothing outside the call it came from — do not
carry symbols between calls, do not read anything into their order, and do not assume they
correspond to any numbering the game itself uses.

Choosing the sampling resolution is the whole point of this tool:

- a few rows by a few columns over one object reads that object exactly
- a single row spanning the board, sampled finely, reveals the spacing of whatever repeats along it
- a large square sampling of the whole image gives you the board as a symbol map you can index

This is how you turn "somewhere around the middle-left" into a coordinate you can act on.

`question` — what you are trying to establish, written before you look. **This goes once, next to
`views` — never inside one of the views themselves.**

`views` — as for `inspect` (`label`, `turn`, `frame`, `region`), plus `rows` and `columns`.

For example, sampling one region at 8×8 to check a specific object:

```
{"question": "is the switch still the blue 2x2 block I recorded, or has it changed colour?",
 "views": [{"label": "switch", "turn": 12, "region": {"x": 200, "y": 88, "width": 16, "height": 16},
            "rows": 8, "columns": 8}]}
```

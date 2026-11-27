Call this when the single frame you were just shown doesn't settle whether your last expectation
held — e.g. you need to check whether something moved, rotated, spawned, or disappeared, which
means comparing against an earlier turn, not just looking harder at this one.

Look again at images you have already been sent: an earlier turn, an intermediate frame from inside
a single action, or a magnified region of either. Read-only — it costs no game action and changes
nothing.

`question` — what you are trying to find out, written before you look. The actual question ("did the
marker on the player change when it turned?"), not a label ("checking the player"). Committing to
what you are looking for is most of the value of the call. **This goes once, next to `views` —
never inside one of the views themselves.**

`views` — one or more things to bring back, each of:

- `label` — a short name you will use when referring to it afterwards
- `turn` — which turn's frames
- `frame` — which frame within that turn's action; `0` is the first one it returned
- `region` — `{x, y, width, height}` in image pixels; the whole image is a legitimate region

Ask for several views in one call whenever you want to compare them. Two frames side by side answer
questions that neither answers alone.

For example, one question, two views to compare:

```
{"question": "did the marker on the player change when it turned?",
 "views": [{"label": "before", "turn": 4, "region": {"x": 100, "y": 100, "width": 40, "height": 40}},
           {"label": "after",  "turn": 5, "region": {"x": 100, "y": 100, "width": 40, "height": 40}}]}
```

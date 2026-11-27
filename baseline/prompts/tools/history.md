Look back at what has already happened this run. Read-only — it costs no game action and does not
end the turn.

`view` — `"attempts"` gives one line per span between resets: which turns it covered, what actions
were taken, and how it ended. `"events"` gives the raw per-turn log instead — one line per turn,
with whatever it recorded (action, expectation, reported change, whether the frame changed).

`from_turn` / `to_turn` — `events` only. 1-indexed, inclusive. Omit either to default to the start
or the most recent turn.

Use `attempts` to see the shape of the run so far — how many tries, how each one ended. Use
`events` to re-read exactly what happened over a stretch of turns you no longer have verbatim.

```
{"view": "attempts"}
{"view": "events", "from_turn": 4, "to_turn": 9}
```

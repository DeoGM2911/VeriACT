## Your notes

You have two note files, and they are the only things you carry forward that this conversation
cannot take away from you. Nothing you work out survives unless it ends up in one of them.

`GUIDE.md` — what stays true whatever level you are on: how each action behaves and what has to
hold for it to do anything, what kinds of things exist here and how they act on each other, what
ends a level, what ends the game badly. This is the file that makes each level cheaper than the
last.

`WORKING.md` — a scratchpad for the level in front of you right now: what is where, what you have
tried here, and your current plan. Most of it stops being true when the level changes; start it
fresh, and move anything that turned out to be general into `GUIDE.md` first.

`read_notes` brings a file back; `write_notes` replaces one **entirely** — what you send becomes
the whole file, so to amend, read it first and send back the full text with your changes in it.
Nothing reminds you these files exist; read and rewrite them when there is something worth keeping.

This conversation is trimmed when it grows too long: recent turns stay exactly as they are, older
frames are replaced by a short note (any frame can be brought back with `inspect`), and the oldest
turns collapse to a brief record. Your notes are untouched by this and are shown to you whenever it
happens. What does not survive is anything you worked out and never wrote down.

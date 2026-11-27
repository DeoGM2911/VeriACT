# Replay viewer

`replay_viewer/` turns one run directory into a single self-contained HTML file that you can open in
any browser, with no server and no internet connection. For every turn it shows:

- the frame the model saw;
- the model's reasoning and its stated expectation;
- the action taken (with the click point marked for `ACTION6`);
- the tool calls of that turn. `inspect` views are reconstructed as the cropped images the model
  was shown, and `read_pixels` results are drawn as colour grids;
- the agent's notes files.

The viewer reads only the files a run writes to disk (`trace.jsonl`, `frames/manifest.json` and
the frame PNGs, the notes files and `run_summary.json`). It does not import either harness, so it
works on runs from both `baseline/` and `veriact/`.

## Creating a replay

1. Play a game (see the main [README](README.md#running)). Each run writes a directory such as
   `runs/ar25-af8a3032/`:

   ```bash
   python -m veriact.agent.run --game ar25
   ```

2. Install the viewer's only dependency, Pillow (already included if you installed
   `requirements.txt`):

   ```bash
   pip install -r replay_viewer/requirements.txt
   ```

3. Build the replay from that run directory:

   ```bash
   python replay_viewer/build_replay.py runs/ar25-af8a3032
   ```

   This writes `runs/ar25-af8a3032_replay.html` next to the run directory and prints its size and
   the number of embedded frames.

4. Open the HTML file in a browser. Click a turn in the timeline, or use **←**/**→** to step through
   turns and **Home**/**End** to jump to the first or last turn.

To build replays for every run of a batch:

```bash
for d in runs/veriact/*/; do python replay_viewer/build_replay.py "$d"; done
```

## Options

```
python replay_viewer/build_replay.py <run_dir> [-o OUTPUT.html] [--title TITLE]
                                     [--include-raw-response] [--warn-frame-threshold N]
```

| Option | Effect |
|---|---|
| `-o`, `--output` | Write the HTML here instead of `<run_dir>_replay.html`. |
| `--title` | Page title. Defaults to the run id. |
| `--include-raw-response` | Embed the raw model response for each turn. This only works if the run was made with `trace.log_raw_response: true`. Useful for debugging; it can make the file large. |
| `--warn-frame-threshold N` | Print a warning when more than `N` distinct frames would be embedded (default 5000). Frames are embedded as base64 PNGs, so very long runs produce large files. |

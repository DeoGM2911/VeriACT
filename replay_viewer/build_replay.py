"""Turns one harness run directory (`runs/<run_id>/`) into a single self-contained HTML replay
file: frames, per-turn reasoning/expectation, actions, and tool activity (`inspect`/`read_pixels`
get an actual visual reconstruction, not just raw text), navigable turn by turn.

Usage:
  python build_replay.py <run_dir> [-o output.html] [--title STR]
                          [--include-raw-response] [--warn-frame-threshold N]

Standalone: reads only the on-disk artifacts a run directory already contains
(trace.jsonl, frames/manifest.json + PNGs, notes files) -- see run_data.py. No harness import.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))  # allow `python build_replay.py` from anywhere

from render_html import build_html, estimate_payload
from run_data import load_run


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("run_dir", help="path to a harness run directory, e.g. runs/ar25-af8a3032")
    parser.add_argument("-o", "--output", default=None, help="output .html path (default: <run_dir>_replay.html, next to run_dir)")
    parser.add_argument("--title", default=None, help="page title override (default: the run id)")
    parser.add_argument("--include-raw-response", action="store_true",
                         help="embed the raw provider response per turn, if the run captured it (diagnostic, can be large)")
    parser.add_argument("--warn-frame-threshold", type=int, default=5000,
                         help="print a size warning if more than this many distinct frames would be embedded (default: 5000)")
    args = parser.parse_args(argv)

    run_dir = Path(args.run_dir)
    if not run_dir.is_dir():
        parser.error(f"not a directory: {run_dir}")
    if not (run_dir / "trace.jsonl").exists():
        parser.error(f"no trace.jsonl found in {run_dir} -- is this a harness run directory?")

    run_data = load_run(run_dir)

    frame_count, approx_bytes = estimate_payload(run_data)
    if frame_count > args.warn_frame_threshold:
        print(
            f"warning: {frame_count} distinct frames will be embedded (~{approx_bytes / 1_000_000:.1f}MB "
            f"base64) -- exceeds --warn-frame-threshold={args.warn_frame_threshold}. "
            f"The output file may be large.",
            file=sys.stderr,
        )

    html_text = build_html(run_data, title=args.title, include_raw_response=args.include_raw_response)

    output = Path(args.output) if args.output else run_dir.parent / f"{run_dir.name}_replay.html"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(html_text, encoding="utf-8")

    print(f"wrote {output} ({output.stat().st_size / 1_000_000:.2f}MB, {frame_count} embedded frames)")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

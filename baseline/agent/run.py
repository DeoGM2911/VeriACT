"""CLI entrypoint: play one game.

    python -m baseline.agent.run --game <game_id> [--config override.yaml] [--max-turns N]

The harness contains no game-specific code; the game id is only ever a runtime argument. Each run
writes to `<runs-dir>/<game>-<random id>/`.
"""
import argparse
import logging
import sys
import uuid
from pathlib import Path

from baseline.agent.config import load_config
from baseline.agent import llm
from baseline.agent import loop


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Play one ARC-AGI-3 game.")
    parser.add_argument("--game", required=True, help="game id, e.g. ls20 or ls20-9607627b")
    parser.add_argument("--config", default=None, help="path to a config override YAML")
    parser.add_argument("--max-turns", type=int, default=None)
    parser.add_argument("--runs-dir", default="runs")
    parser.add_argument(
        "--environments-dir", default=None,
        help="override paths.environments_dir; where the downloaded games live",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    cfg = load_config(args.config)
    if args.max_turns is not None:
        cfg.run.max_turns = args.max_turns
    if args.environments_dir is not None:
        cfg.paths.environments_dir = args.environments_dir

    run_id = f"{args.game}-{uuid.uuid4().hex[:8]}"
    run_dir = Path(args.runs_dir) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    chat = llm.build_client(cfg)
    loop.run(args.game, cfg, chat, run_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())

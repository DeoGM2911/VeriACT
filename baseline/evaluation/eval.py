"""CLI entrypoint: run a batch of games and score them.

    python -m baseline.evaluation.eval --games <id1>,<id2> [--config override.yaml]
    python -m baseline.evaluation.eval --all-games

Each game is played through the same `loop.run()` path `run.py` uses, then scored from its
`trace.jsonl` with `scoring.py`. Prints a table and writes `<runs-dir>/eval_summary.json`.
"""
import argparse
import json
import logging
import sys
import uuid
from pathlib import Path

from baseline.agent.config import load_config
from baseline.agent.env import EnvAdapter, list_available_games
from baseline.agent import llm
from baseline.agent import loop
from baseline.evaluation import scoring

logger = logging.getLogger(__name__)


def evaluate_game(game_id: str, cfg, chat: llm.ChatClient, runs_dir, env=None) -> dict:
    """Runs one game to completion (or a budget/failure) and scores it. Never raises -- a failed
    game is recorded in the result and the batch continues, since one bad game (a config typo, a
    dropped connection) should not lose results already collected for the others.

    `env`, like `loop.run`'s own parameter of the same name, is normally left `None` (a real
    `EnvAdapter` is built here); tests inject a fake one for a fast, deterministic, offline run.
    """
    run_id = f"{game_id}-{uuid.uuid4().hex[:8]}"
    run_dir = Path(runs_dir) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    error = None
    baseline_actions: list = []
    try:
        if env is None:
            env = EnvAdapter(
                game_id, run_dir,
                image_scale=cfg.image.scale,
                gridline_brightness=cfg.image.gridline_brightness,
                environments_dir=cfg.paths.environments_dir,
            )
        baseline_actions = env.baseline_actions
        loop.run(game_id, cfg, chat, run_dir, env=env)
    except Exception as exc:  # noqa: BLE001 -- a batch runner must not let one game kill the rest
        error = str(exc)
        logger.warning("%s: %s", game_id, error)

    trace_path = run_dir / "trace.jsonl"
    levels = scoring.levels_from_trace(trace_path, baseline_actions) if (baseline_actions and trace_path.exists()) else []
    score = scoring.environment_score(levels) if levels else None

    return {
        "game_id": game_id,
        "run_id": run_id,
        "run_dir": str(run_dir),
        "score": score,
        "levels_completed": sum(1 for lvl in levels if lvl["completed"]),
        "total_levels": len(levels),
        "actions_used": sum(lvl["actions_taken"] for lvl in levels),
        "baseline_actions_total": sum(baseline_actions) if baseline_actions else None,
        "error": error,
    }


def evaluate(games: list, cfg, chat: llm.ChatClient, runs_dir, env_factory=None) -> dict:
    """`env_factory(game_id) -> EnvAdapter`, if given, replaces the real `EnvAdapter` for every
    game in the batch -- how tests exercise multi-game batching offline. Every game in `games`
    runs under the same `cfg`, so run-level parameters (max turns, action budgets, ...) are
    configured once for the whole batch, not per game."""
    results = [
        evaluate_game(g, cfg, chat, runs_dir, env=env_factory(g) if env_factory else None)
        for g in games
    ]
    scored = [r["score"] for r in results if r["score"] is not None]
    aggregate = sum(scored) / len(scored) if scored else None
    return {"games": results, "aggregate_score": aggregate}


def format_report(summary: dict) -> str:
    lines = [f"{'game':<10} {'levels':>10} {'actions':>10} {'baseline':>10} {'score':>8}"]
    for r in summary["games"]:
        levels = f"{r['levels_completed']}/{r['total_levels']}"
        score = f"{r['score']:.1f}" if r["score"] is not None else "n/a"
        baseline = r["baseline_actions_total"] if r["baseline_actions_total"] is not None else "n/a"
        lines.append(f"{r['game_id']:<10} {levels:>10} {r['actions_used']:>10} {str(baseline):>10} {score:>8}")
        if r["error"]:
            lines.append(f"  ! {r['error']}")
    agg = f"{summary['aggregate_score']:.2f}" if summary["aggregate_score"] is not None else "n/a"
    lines.append(f"\naggregate score: {agg}")
    return "\n".join(lines)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Batch-run and score ARC-AGI-3 games.")
    games_group = parser.add_mutually_exclusive_group(required=True)
    games_group.add_argument("--games", default=None, help="comma-separated game ids, e.g. ls20,vc33")
    games_group.add_argument(
        "--all-games", action="store_true",
        help="run every game found under environments_dir instead of listing ids by hand",
    )
    parser.add_argument("--config", default=None, help="path to a config override YAML")
    parser.add_argument("--max-turns", type=int, default=None)
    parser.add_argument("--runs-dir", default="runs")
    parser.add_argument(
        "--environments-dir", default=None,
        help="override paths.environments_dir; where the downloaded games live",
    )
    parser.add_argument("--out", default=None, help="eval summary JSON path (default: <runs-dir>/eval_summary.json)")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    cfg = load_config(args.config)
    if args.max_turns is not None:
        cfg.run.max_turns = args.max_turns
    if args.environments_dir is not None:
        cfg.paths.environments_dir = args.environments_dir

    if args.all_games:
        games = list_available_games(cfg.paths.environments_dir)
    else:
        games = [g.strip() for g in args.games.split(",") if g.strip()]
    chat = llm.build_client(cfg)

    summary = evaluate(games, cfg, chat, args.runs_dir)

    out_path = Path(args.out) if args.out else Path(args.runs_dir) / "eval_summary.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(format_report(summary))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

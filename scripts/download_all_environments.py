"""Download every ARC-AGI-3 environment the API key exposes.

Mirrors arc_agi.base.Arcade._download_game (metadata + source GETs, same on-disk
layout) without going through make(), so it does not create a scorecard, does not
import 23 unseen game modules, and does not rewrite metadata for games already
downloaded.
"""

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import requests

import arc_agi

ROOT = Path(__file__).resolve().parent.parent
ENV_DIR_NAME = "environment_files"
ENV_ROOT = ROOT / ENV_DIR_NAME

arc = arc_agi.Arcade(environments_dir=str(ENV_ROOT))
if not arc.arc_api_key:
    sys.exit("No ARC_API_KEY found")

BASE = arc.arc_base_url
HEADERS = {"X-Api-Key": arc.arc_api_key, "Accept": "application/json"}

envs = arc.get_environments()
print(f"API exposes {len(envs)} environment(s)\n")

ok, skipped, failed = [], [], []

for env in sorted(envs, key=lambda e: e.game_id):
    full_id = env.game_id
    base_id, _, version = full_id.partition("-")
    env_dir = ENV_ROOT / base_id / version

    try:
        # class_name drives the source filename; same fallback as the SDK.
        meta_resp = requests.get(f"{BASE}/api/games/{full_id}", headers=HEADERS, timeout=30)
        meta_resp.raise_for_status()
        metadata = meta_resp.json()

        class_name = metadata.get("class_name") or (
            base_id[0].upper() + base_id[1:] if base_id else "Game"
        )
        source_file = env_dir / f"{class_name.lower()}.py"

        if source_file.exists() and source_file.stat().st_size > 0:
            skipped.append((full_id, "already present"))
            print(f"  skip  {full_id:<16} already downloaded")
            continue

        src_resp = requests.get(f"{BASE}/api/games/{full_id}/source", headers=HEADERS, timeout=60)
        src_resp.raise_for_status()
        source_code = src_resp.text

        if not source_code.strip():
            raise ValueError("empty source body")
        if "ARCBaseGame" not in source_code:
            raise ValueError(
                f"source does not look like a game module "
                f"(first 120 chars: {source_code[:120]!r})"
            )

        env_dir.mkdir(parents=True, exist_ok=True)
        metadata["tags"] = metadata.get("tags", [])
        metadata["baseline_actions"] = metadata.get("baseline_actions", [])
        # Relative form, matching the two environments already on disk.
        metadata["local_dir"] = str(Path(ENV_DIR_NAME) / base_id / version)
        metadata["date_downloaded"] = datetime.now(timezone.utc).isoformat()

        (env_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        source_file.write_text(source_code, encoding="utf-8")

        ok.append((full_id, len(source_code)))
        print(f"  ok    {full_id:<16} {metadata.get('tags')} -> {source_file.name} ({len(source_code):,} bytes)")

    except Exception as e:
        failed.append((full_id, repr(e)))
        print(f"  FAIL  {full_id:<16} {e!r}")

print(f"\ndownloaded={len(ok)} skipped={len(skipped)} failed={len(failed)}")
for gid, err in failed:
    print(f"  {gid}: {err}")

"""Validate each harness's prompt set (`<harness>/prompts/`) against its `manifest.yaml`.

    python scripts/check_prompts.py [baseline|veriact]     # default: both

Checks, in order:

1. every {{variable}} used in a prompt is declared in the manifest
2. every declared variable is used somewhere
3. every file the manifest references exists
4. at the arm's version tag, the prompt renders with nothing unfilled
5. model-facing text in system/, turn.md and the other always-sent prompts does not disclose the
   grid structure (size, cell size, colour numbering), and no prompt names a specific game

Check 5 is the mechanical half of the task-agnosticism rule. tools/ is exempt from the
grid-structure part -- read_pixels.md cannot explain sampling without saying "cell" -- and is
reviewed by hand instead.

VeriACT's `memo:` front end (`tool_mode: memo`) has no `since` tags; it is always checked in full.
"""

import glob
import os
import re
import sys

import yaml

REPO = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
# The version tag each harness renders its prompts for (`MILESTONE` in <harness>/agent/loop.py).
ARMS = {"baseline": "M13", "veriact": "M14"}
ROOT = None  # set per arm by check_arm()
MILESTONES = ["M1", "M2", "M3", "M4", "M5", "M6", "M7", "M8", "M9", "M10", "M11", "M12", "M13", "M14"]
GRID_DISCLOSURE = [r"\b64\b", r"\b8\s*px\b", r"\bcells?\b", r"\b0\s*[-–]\s*15\b"]
FIXTURES = [r"\bls20\b", r"\bvc33\b"]


def rel(path):
    return os.path.relpath(path, ROOT).replace("\\", "/")


def vars_in(path):
    with open(path, encoding="utf-8") as fh:
        return set(re.findall(r"\{\{(\w+)\}\}", fh.read()))


def load():
    with open(os.path.join(ROOT, "manifest.yaml"), encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def at_or_before(since, milestone):
    return MILESTONES.index(since) <= MILESTONES.index(milestone)


def check_arm(arm):
    global ROOT
    ROOT = os.path.join(REPO, arm, "prompts")
    milestone = ARMS[arm]
    print("== %s (%s) ==" % (arm, milestone))
    m = load()
    failures = []

    used = {}
    for path in glob.glob(os.path.join(ROOT, "**", "*.md"), recursive=True):
        for v in vars_in(path):
            used.setdefault(v, set()).add(rel(path))

    declared = set(m["variables"])
    undeclared = {v: sorted(f) for v, f in used.items() if v not in declared}
    unused = sorted(declared - set(used))
    print("UNDECLARED variables:", undeclared or "none")
    print("DECLARED but unused :", unused or "none")
    if undeclared:
        failures.append("undeclared variables")
    if unused:
        failures.append("unused variables")

    referenced = [p["file"] for p in m["system"]] + [m["turn"]]
    for key in ("digest", "frame_placeholder"):
        if key in m:
            referenced.append(m[key]["file"])
    if "observe_terminal" in m:
        referenced.append(m["observe_terminal"]["file"])
    if "click_confirm" in m:
        referenced.append(m["click_confirm"]["file"])
    if "memo" in m:
        referenced += [m["memo"]["system"], m["memo"]["turn"]]
    missing = [f for f in referenced if not os.path.exists(os.path.join(ROOT, f))]
    print("MISSING files       :", sorted(missing) or "none")
    if missing:
        failures.append("missing files")

    undrafted = sorted(
        t["file"] for t in m["tools"].values() if not os.path.exists(os.path.join(ROOT, t["file"]))
    )
    missing_now = [
        t["file"]
        for t in m["tools"].values()
        if at_or_before(t["since"], milestone) and not os.path.exists(os.path.join(ROOT, t["file"]))
    ]
    print("unwritten tool files:", undrafted or "none", "(gated to a later version)")
    if missing_now:
        print("MISSING tool files needed at %s: %s" % (milestone, missing_now))
        failures.append("missing tool files")

    # 4. render check, honouring the block-dropping rule: a block whose variables are all
    #    unavailable is omitted, so it is not a gap. A block that mixes available and
    #    unavailable variables cannot drop, and its unavailable ones render as holes.
    have = {v for v, c in m["variables"].items() if at_or_before(c["since"], milestone)}
    parts = [p["file"] for p in m["system"] if at_or_before(p["since"], milestone)] + [m["turn"]]
    parts += [t["file"] for t in m["tools"].values() if at_or_before(t["since"], milestone)]
    for key in ("digest", "frame_placeholder"):
        if key in m and at_or_before(m[key]["since"], milestone):
            parts.append(m[key]["file"])
    if "observe_terminal" in m and at_or_before(m["observe_terminal"]["since"], milestone):
        parts.append(m["observe_terminal"]["file"])
    if "click_confirm" in m and at_or_before(m["click_confirm"]["since"], milestone):
        parts.append(m["click_confirm"]["file"])
    if "memo" in m and at_or_before("M5", milestone):
        parts += [m["memo"]["system"], m["memo"]["turn"]]
    gap = set()
    for f in parts:
        path = os.path.join(ROOT, f)
        if not os.path.exists(path):
            continue
        for block in re.split(r"\n\s*\n", open(path, encoding="utf-8").read()):
            block_vars = set(re.findall(r"\{\{(\w+)\}\}", block))
            if not block_vars or not (block_vars & have):
                continue          # no variables, or the whole block drops
            gap |= block_vars - have
    gap = sorted(gap)
    print("%s render, unfilled  : %s" % (milestone, gap or "none"))
    if gap:
        failures.append("%s render gap" % milestone)

    # 5. disclosure and fixture names
    disclosure_files = [p["file"] for p in m["system"]] + [m["turn"]]
    for key in ("digest", "frame_placeholder"):
        if key in m:
            disclosure_files.append(m[key]["file"])
    if "click_confirm" in m:
        disclosure_files.append(m["click_confirm"]["file"])
    if "memo" in m:
        disclosure_files += [m["memo"]["system"], m["memo"]["turn"]]
    leaks = []
    for f in disclosure_files:
        path = os.path.join(ROOT, f)
        if not os.path.exists(path):
            continue
        text = open(path, encoding="utf-8").read()
        for pat in GRID_DISCLOSURE:
            if re.search(pat, text, re.I):
                leaks.append("%s: %s" % (rel(path), pat))
    print("grid disclosure     :", leaks or "none")
    if leaks:
        failures.append("grid disclosure")

    named = []
    for path in glob.glob(os.path.join(ROOT, "**", "*.md"), recursive=True):
        text = open(path, encoding="utf-8").read()
        for pat in FIXTURES:
            if re.search(pat, text, re.I):
                named.append("%s: %s" % (rel(path), pat))
    print("fixture ids         :", named or "none")
    if named:
        failures.append("fixture ids in prompts")

    print()
    if failures:
        print("FAIL:", ", ".join(failures))
        return 1
    print("OK")
    print()
    return 0


def main():
    arms = sys.argv[1:] or list(ARMS)
    unknown = [a for a in arms if a not in ARMS]
    if unknown:
        print("unknown arm(s): %s; expected %s" % (", ".join(unknown), ", ".join(ARMS)))
        return 2
    return max(check_arm(arm) for arm in arms)


if __name__ == "__main__":
    sys.exit(main())

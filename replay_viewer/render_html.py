"""Builds the single self-contained HTML replay file. Pure string assembly, no disk I/O -- all
data comes in already-loaded from `run_data.py`. Three parts in the emitted document: a small
static shell, one embedded JSON blob of precomputed per-turn data, and a vanilla-JS controller
(no imports, no CDN) that renders the filmstrip/detail-panel from that blob lazily.
"""
import html
import json

from arc_colors import as_js_array
from run_data import CompactionEvent, RunData, Turn


def estimate_payload(run_data: RunData) -> tuple:
    """(frame_count, approx_bytes) of everything that will be base64-embedded -- used for the
    CLI's --warn-frame-threshold, computed to match what `to_json_dict`/`_turn_to_dict` actually
    embeds. Deliberately NOT deduplicated by content or by ref: each turn's `before`/`after` is
    embedded independently (e.g. turn N's `after` and turn N+1's `before` are the same underlying
    frame but are two separate embeds in the JSON blob, since turns are meant to be independently
    lazy-renderable in the viewer) -- counting distinct byte strings here would under-report the
    real output size."""
    frame_count = 0
    approx_bytes = 0
    for entry in run_data.timeline:
        if not isinstance(entry, Turn):
            continue
        for uri in (entry.before_data_uri, entry.after_data_uri):
            if uri:
                frame_count += 1
                approx_bytes += len(uri)
        for tc in entry.tool_calls:
            for v in tc.views:
                uri = v.get("data_uri")
                if uri:
                    frame_count += 1
                    approx_bytes += len(uri)
    return frame_count, approx_bytes


def _turn_to_dict(t: Turn, include_raw_response: bool) -> dict:
    d = {
        "type": "turn",
        "index": t.index,
        "beforeRef": t.before_ref,
        "afterRef": t.after_ref,
        "state": t.state,
        "levelsCompleted": t.levels_completed,
        "winLevels": t.win_levels,
        "fullReset": t.full_reset,
        "availableActions": t.available_actions,
        "actionId": t.action_id,
        "click": t.click,
        "reasoning": t.reasoning,
        "usage": t.usage,
        "expectation": t.expectation,
        "reportedChange": t.reported_change,
        "frameChanged": t.frame_changed,
        "protocolFailure": t.protocol_failure,
        "stepInLevel": t.step_in_level,
        "attemptInLevel": t.attempt_in_level,
        "thinking": t.thinking,
        "beforeDataUri": t.before_data_uri,
        "afterDataUri": t.after_data_uri,
        "toolCalls": [
            {
                "name": tc.name,
                "question": tc.arguments.get("question"),
                "file": tc.arguments.get("file"),
                "result": tc.result,
                "views": [
                    {
                        "label": v.get("label"),
                        "turn": v.get("turn"),
                        "frame": v.get("frame"),
                        "region": v.get("region"),
                        "dataUri": v.get("data_uri"),
                        "rows": v.get("rows"),
                        "columns": v.get("columns"),
                        "grid": v.get("grid"),
                        "palette": v.get("palette"),
                    }
                    for v in tc.views
                ],
            }
            for tc in t.tool_calls
        ],
    }
    if include_raw_response:
        d["rawResponse"] = t.raw_response
    return d


def to_json_dict(run_data: RunData, include_raw_response: bool = False) -> dict:
    timeline = []
    for entry in run_data.timeline:
        if isinstance(entry, CompactionEvent):
            timeline.append({
                "type": "compaction",
                "turn": entry.turn,
                "tokensBefore": entry.tokens_before,
                "continuation": entry.continuation,
            })
        else:
            timeline.append(_turn_to_dict(entry, include_raw_response))

    last_turn = next((e for e in reversed(run_data.timeline) if isinstance(e, Turn)), None)
    all_turns = [e for e in run_data.timeline if isinstance(e, Turn)]
    # Header progress uses the max levels_completed seen across the run, not just the last
    # record's: a non-RESET action sent while GAME_OVER returns a response whose
    # levels_completed defaults to 0, so the last record can understate real progress. Progress
    # never decreases within a run, so the max is the correct "best reached".
    max_levels_completed = max((t.levels_completed for t in all_turns), default=0)

    return {
        "runId": run_data.run_id,
        "gameId": last_turn.game_id if last_turn else "",
        "finalState": last_turn.state if last_turn else "",
        "levelsCompleted": max_levels_completed,
        "winLevels": last_turn.win_levels if last_turn else None,
        "totalTurns": sum(1 for e in run_data.timeline if isinstance(e, Turn)),
        "runSummary": run_data.run_summary,
        "notes": {
            "guide": run_data.notes.guide,
            "working": run_data.notes.working,
            "byLevel": {str(k): v for k, v in sorted(run_data.notes.working_by_level.items())},
        },
        "colorMap": None,  # filled in as a JS literal separately, not JSON (see build_html)
        "timeline": timeline,
    }


def build_html(run_data: RunData, title: str = None, include_raw_response: bool = False) -> str:
    data = to_json_dict(run_data, include_raw_response=include_raw_response)
    data.pop("colorMap", None)
    data_json = json.dumps(data, separators=(",", ":")).replace("<", "\\u003c")
    page_title = html.escape(title or run_data.run_id)

    return (
        _HTML_HEAD.replace("__TITLE__", page_title)
        + _HTML_BODY_SHELL
        + f'<script id="run-data" type="application/json">{data_json}</script>\n'
        + f"<script>const ARC_COLORS = {as_js_array()};\n" + _JS_CONTROLLER + "</script>\n"
        + "</body>\n</html>\n"
    )


_HTML_HEAD = """<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>__TITLE__ -- replay</title>
<style>
""" + """
:root { color-scheme: dark; }
* { box-sizing: border-box; }
body {
  margin: 0; background: #16181d; color: #d7dae0;
  font: 14px/1.5 -apple-system, Segoe UI, Helvetica, Arial, sans-serif;
}
code, pre, .mono { font: 13px/1.4 ui-monospace, SFMono-Regular, Consolas, monospace; }
a { color: #6ab0f3; }
header.top {
  display: flex; align-items: center; gap: 14px; padding: 10px 16px;
  background: #1c1f26; border-bottom: 1px solid #2a2e37; position: sticky; top: 0; z-index: 5;
  flex-wrap: wrap;
}
header.top h1 { font-size: 15px; margin: 0; font-weight: 600; }
header.top .path { color: #8b909c; font-size: 12px; }
.badge {
  display: inline-block; padding: 2px 9px; border-radius: 999px; font-size: 12px; font-weight: 600;
  background: #333844; color: #cfd3db;
}
.badge.WIN { background: #16532a; color: #7ee399; }
.badge.GAME_OVER { background: #55201f; color: #ff9c93; }
.badge.NOT_FINISHED, .badge.NOT_PLAYED { background: #333844; color: #cfd3db; }
.progress { color: #a9adb6; font-size: 12px; }
#jump-form { margin-left: auto; display: flex; gap: 6px; align-items: center; }
#jump-form input {
  width: 64px; background: #12141a; border: 1px solid #333844; color: #d7dae0; border-radius: 4px;
  padding: 3px 6px;
}
button, #jump-form button {
  background: #2a2e37; border: 1px solid #3a3f4c; color: #d7dae0; border-radius: 4px;
  padding: 3px 9px; cursor: pointer; font-size: 12px;
}
button:hover { background: #333844; }
details.drawer { border-bottom: 1px solid #2a2e37; background: #191c22; }
details.drawer > summary {
  cursor: pointer; padding: 8px 16px; font-weight: 600; font-size: 13px; color: #b9bec9;
  list-style: none;
}
details.drawer > summary::-webkit-details-marker { display: none; }
details.drawer > summary::before { content: "\\25b8  "; }
details.drawer[open] > summary::before { content: "\\25be  "; }
.drawer-body { padding: 0 16px 14px; display: flex; gap: 18px; flex-wrap: wrap; }
.note-col { flex: 1 1 260px; min-width: 220px; }
.note-col h3 { font-size: 12px; text-transform: uppercase; color: #8b909c; margin: 6px 0; }
.note-col pre {
  white-space: pre-wrap; word-break: break-word; background: #12141a; border: 1px solid #262a33;
  border-radius: 6px; padding: 10px; max-height: 260px; overflow: auto; margin: 0;
}
.summary-box {
  margin: 10px 16px; padding: 8px 12px; background: #191c22; border: 1px solid #262a33;
  border-radius: 6px; font-size: 12px; color: #b9bec9; display: flex; gap: 16px; flex-wrap: wrap;
}
.summary-box b { color: #d7dae0; }
#filmstrip {
  display: flex; gap: 4px; overflow-x: auto; padding: 10px 16px; background: #14161b;
  border-bottom: 1px solid #2a2e37;
}
.thumb {
  flex: 0 0 auto; width: 54px; cursor: pointer; text-align: center; border-radius: 5px;
  padding: 3px; border: 2px solid transparent; background: #1c1f26;
}
.thumb img { width: 48px; height: 48px; object-fit: cover; border-radius: 3px; display: block; image-rendering: pixelated; }
.thumb .n { font-size: 10px; color: #8b909c; margin-top: 2px; }
.thumb.selected { border-color: #6ab0f3; }
.thumb.protocol-failure { border-color: #d9534f; }
.thumb.level-start { box-shadow: inset 3px 0 0 #4fcc30; }
.thumb.reset-turn .n { color: #ffb45c; }
.thumb.no-op { opacity: .5; }
.thumb.no-op .n::after { content: " \\2205"; color: #d9a441; }
.noop-flag {
  display: inline-block; margin-left: 8px; padding: 2px 8px; border-radius: 5px;
  background: #2d2a20; color: #d9a441; font-size: 11.5px; font-weight: 600;
}
.compact-marker {
  flex: 0 0 auto; width: 4px; align-self: stretch; background: repeating-linear-gradient(
    180deg, #6ab0f3, #6ab0f3 3px, transparent 3px, transparent 6px); border-radius: 2px;
  cursor: pointer;
}
main { padding: 16px; max-width: 1180px; margin: 0 auto; }
.frames-row { display: flex; gap: 16px; flex-wrap: wrap; margin-bottom: 14px; }
.frame-box { position: relative; flex: 0 0 auto; }
.frame-box img { width: 320px; height: 320px; object-fit: contain; border-radius: 6px; border: 1px solid #262a33; image-rendering: pixelated; background: #000; }
.frame-box .cap { font-size: 11px; color: #8b909c; margin-top: 4px; }
.click-marker {
  position: absolute; width: 12px; height: 12px; margin: -6px 0 0 -6px; border-radius: 50%;
  background: rgba(255,90,90,0.85); border: 2px solid #fff; pointer-events: none;
}
.pills { display: flex; gap: 8px; flex-wrap: wrap; margin: 10px 0; }
.pill { padding: 6px 10px; border-radius: 6px; font-size: 12.5px; max-width: 560px; }
.pill .k { display: block; font-size: 10px; text-transform: uppercase; letter-spacing: .03em; opacity: .75; margin-bottom: 2px; }
.pill.expect { background: #1c2a3d; color: #a9cdfa; }
.pill.change { background: #2a2418; color: #e8c98a; }
.action-chip { display: inline-block; padding: 2px 8px; border-radius: 5px; background: #263826; color: #8fe08f; font-weight: 600; font-size: 12.5px; }
.action-chip.reset { background: #3d2a1c; color: #ffb45c; }
.avail { font-size: 11px; color: #757b87; margin-left: 8px; }
.protocol-banner {
  background: #3a1c1c; border: 1px solid #6b2a2a; color: #ff9c93; border-radius: 6px;
  padding: 8px 12px; margin: 10px 0; font-size: 13px;
}
.section-label {
  font-size: 11px; text-transform: uppercase; letter-spacing: .04em; color: #8b909c;
  margin: 14px 0 4px;
}
.thinking-box, .reasoning-box {
  white-space: pre-wrap; word-break: break-word; border-radius: 6px; padding: 12px; margin: 0 0 6px;
}
.thinking-box {
  background: #1a1a22; border: 1px solid #322f45; border-left: 3px solid #8577dd; color: #b9b4d6;
  font-style: italic;
}
.reasoning-box {
  background: #191c22; border: 1px solid #262a33;
}
details.tool-call {
  background: #191c22; border: 1px solid #262a33; border-radius: 6px; margin: 8px 0;
}
details.tool-call > summary {
  cursor: pointer; padding: 8px 12px; font-size: 12.5px; list-style: none; display: flex; gap: 8px; align-items: center;
}
details.tool-call > summary::-webkit-details-marker { display: none; }
.tool-name {
  font-weight: 700; text-transform: uppercase; font-size: 10.5px; letter-spacing: .04em;
  padding: 1px 7px; border-radius: 4px; background: #2a2e37; color: #9fe0ff;
}
.tool-body { padding: 0 12px 12px; }
.tool-body pre { white-space: pre-wrap; word-break: break-word; background: #12141a; border: 1px solid #262a33; border-radius: 5px; padding: 8px; max-height: 300px; overflow: auto; }
.views { display: flex; gap: 12px; flex-wrap: wrap; margin-top: 8px; }
.view-card { flex: 0 0 auto; }
.view-card img { max-width: 180px; max-height: 180px; border-radius: 5px; border: 1px solid #262a33; image-rendering: pixelated; background: #000; }
.view-card canvas { border-radius: 5px; border: 1px solid #262a33; image-rendering: pixelated; cursor: crosshair; }
.view-card .cap { font-size: 10.5px; color: #8b909c; max-width: 200px; margin-top: 3px; }
.pixel-tip { font-size: 10.5px; color: #cfd3db; margin-top: 3px; min-height: 13px; font-family: ui-monospace, monospace; }
details.aux { margin: 10px 0; }
details.aux > summary { cursor: pointer; font-size: 12px; color: #8b909c; }
details.aux pre { white-space: pre-wrap; word-break: break-word; background: #12141a; border: 1px solid #262a33; border-radius: 5px; padding: 8px; max-height: 320px; overflow: auto; }
.empty { color: #757b87; padding: 40px; text-align: center; }
</style>
</head>
"""

_HTML_BODY_SHELL = """<body>
<header class="top">
  <h1 id="hdr-title"></h1>
  <span id="hdr-state" class="badge"></span>
  <span id="hdr-progress" class="progress"></span>
  <span class="path" id="hdr-path"></span>
  <form id="jump-form">
    <input id="jump-input" type="number" min="1" placeholder="turn #">
    <button type="submit">Go</button>
    <button type="button" id="prev-btn">&larr;</button>
    <button type="button" id="next-btn">&rarr;</button>
  </form>
</header>
<details class="drawer" id="notes-drawer">
  <summary>Notes</summary>
  <div class="drawer-body" id="notes-body"></div>
</details>
<div class="summary-box" id="summary-box" style="display:none"></div>
<div id="filmstrip"></div>
<main id="detail"></main>
"""

_JS_CONTROLLER = r"""
const RUN = JSON.parse(document.getElementById('run-data').textContent);
const TIMELINE = RUN.timeline;

document.getElementById('hdr-title').textContent = RUN.gameId + ' -- ' + RUN.runId;
const stateBadge = document.getElementById('hdr-state');
stateBadge.textContent = RUN.finalState;
stateBadge.className = 'badge ' + RUN.finalState;
document.getElementById('hdr-progress').textContent =
  RUN.levelsCompleted + (RUN.winLevels != null ? ' of ' + RUN.winLevels : '') + ' levels complete · ' +
  RUN.totalTurns + ' turns';
document.getElementById('hdr-path').textContent = RUN.runId;

if (RUN.runSummary) {
  const box = document.getElementById('summary-box');
  box.style.display = 'flex';
  box.innerHTML = '';
  for (const [k, v] of Object.entries(RUN.runSummary)) {
    const span = document.createElement('span');
    const b = document.createElement('b'); b.textContent = k + ': ';
    span.appendChild(b);
    span.appendChild(document.createTextNode(String(v)));
    box.appendChild(span);
  }
}

function renderNotes() {
  const body = document.getElementById('notes-body');
  body.innerHTML = '';
  const col = (title, text) => {
    const d = document.createElement('div'); d.className = 'note-col';
    const h = document.createElement('h3'); h.textContent = title; d.appendChild(h);
    const pre = document.createElement('pre'); pre.textContent = text || '(empty)'; d.appendChild(pre);
    return d;
  };
  body.appendChild(col('GUIDE.md', RUN.notes.guide));
  body.appendChild(col('WORKING.md (final)', RUN.notes.working));
  const levels = Object.keys(RUN.notes.byLevel);
  if (levels.length) {
    for (const lvl of levels) {
      body.appendChild(col('WORKING.md @ end of level ' + lvl, RUN.notes.byLevel[lvl]));
    }
  }
}
renderNotes();

const filmstrip = document.getElementById('filmstrip');
const thumbEls = {};
let prevLevels = null;
TIMELINE.forEach((entry, i) => {
  if (entry.type === 'compaction') {
    const m = document.createElement('div');
    m.className = 'compact-marker';
    m.title = 'context compacted after turn ' + entry.turn;
    m.addEventListener('click', () => select(i));
    filmstrip.appendChild(m);
    return;
  }
  const t = document.createElement('div');
  t.className = 'thumb';
  if (entry.protocolFailure) t.classList.add('protocol-failure');
  // ">" not "!==": a corrupted trailing record from a known harness bug (see comment in
  // render_html.py's to_json_dict) can report levelsCompleted dropping back to 0 even though
  // no real level-start happened -- only an increase is a genuine level transition.
  if (prevLevels !== null && entry.levelsCompleted > prevLevels) t.classList.add('level-start');
  prevLevels = entry.levelsCompleted;
  if (entry.actionId === 'RESET') t.classList.add('reset-turn');
  if (entry.frameChanged === false && entry.actionId && !entry.protocolFailure) t.classList.add('no-op');
  const img = document.createElement('img');
  img.src = entry.afterDataUri || entry.beforeDataUri || '';
  img.loading = 'lazy';
  const n = document.createElement('div'); n.className = 'n'; n.textContent = '#' + entry.index;
  t.appendChild(img); t.appendChild(n);
  t.addEventListener('click', () => select(i));
  filmstrip.appendChild(t);
  thumbEls[i] = t;
});

const detailCache = {};
let currentIndex = null;

function actionChip(entry) {
  if (!entry.actionId) return null;
  const chip = document.createElement('span');
  chip.className = 'action-chip' + (entry.actionId === 'RESET' ? ' reset' : '');
  let text = entry.actionId;
  if (entry.click) text += ' (' + entry.click.x + ',' + entry.click.y + ')';
  chip.textContent = text;
  return chip;
}

function pill(kind, label, text) {
  const p = document.createElement('div');
  p.className = 'pill ' + kind;
  const k = document.createElement('span'); k.className = 'k'; k.textContent = label;
  p.appendChild(k);
  p.appendChild(document.createTextNode(text));
  return p;
}

function frameBox(dataUri, caption, click) {
  const box = document.createElement('div'); box.className = 'frame-box';
  const img = document.createElement('img');
  img.src = dataUri || '';
  if (!dataUri) img.alt = '(frame unavailable)';
  box.appendChild(img);
  if (click) {
    const marker = document.createElement('div');
    marker.className = 'click-marker';
    marker.style.left = (click.x / 512 * 100) + '%';
    marker.style.top = (click.y / 512 * 100) + '%';
    box.appendChild(marker);
  }
  const cap = document.createElement('div'); cap.className = 'cap'; cap.textContent = caption;
  box.appendChild(cap);
  return box;
}

function drawPixelGrid(canvas, view) {
  const rows = view.rows, cols = view.columns;
  const cell = Math.max(3, Math.min(20, Math.floor(360 / Math.max(rows, cols))));
  canvas.width = cols * cell; canvas.height = rows * cell;
  const ctx = canvas.getContext('2d');
  for (let r = 0; r < rows; r++) {
    const line = view.grid[r] || '';
    for (let c = 0; c < cols; c++) {
      const sym = line[c];
      const raw = view.palette[sym];
      ctx.fillStyle = (raw != null && ARC_COLORS[raw]) ? ARC_COLORS[raw] : '#222';
      ctx.fillRect(c * cell, r * cell, cell, cell);
    }
  }
  const tip = canvas.parentElement.querySelector('.pixel-tip');
  canvas.addEventListener('mousemove', (ev) => {
    const rect = canvas.getBoundingClientRect();
    const c = Math.min(cols - 1, Math.max(0, Math.floor((ev.clientX - rect.left) / cell)));
    const r = Math.min(rows - 1, Math.max(0, Math.floor((ev.clientY - rect.top) / cell)));
    const sym = (view.grid[r] || '')[c];
    const raw = view.palette[sym];
    const region = view.region;
    let px, py;
    if (region) {
      px = Math.round(region.x + (c + 0.5) * region.width / cols);
      py = Math.round(region.y + (r + 0.5) * region.height / rows);
    } else {
      px = Math.round((c + 0.5) * 512 / cols);
      py = Math.round((r + 0.5) * 512 / rows);
    }
    tip.textContent = sym + '  raw=' + raw + '  row ' + r + ' col ' + c +
      '  ~pixel (' + px + ',' + py + ')  ~grid (' + Math.floor(px/8) + ',' + Math.floor(py/8) + ')';
  });
  canvas.addEventListener('mouseleave', () => { tip.textContent = ''; });
}

function toolCallCard(tc) {
  const det = document.createElement('details'); det.className = 'tool-call';
  const sum = document.createElement('summary');
  const name = document.createElement('span'); name.className = 'tool-name'; name.textContent = tc.name;
  sum.appendChild(name);
  const label = tc.question || tc.file || '';
  sum.appendChild(document.createTextNode(label));
  det.appendChild(sum);

  const body = document.createElement('div'); body.className = 'tool-body';
  let built = false;
  det.addEventListener('toggle', () => {
    if (!det.open || built) return;
    built = true;
    if (tc.name === 'inspect' && tc.views.length) {
      const views = document.createElement('div'); views.className = 'views';
      for (const v of tc.views) {
        const card = document.createElement('div'); card.className = 'view-card';
        const img = document.createElement('img'); img.src = v.dataUri || '';
        if (!v.dataUri) img.alt = '(view unavailable)';
        card.appendChild(img);
        const cap = document.createElement('div'); cap.className = 'cap';
        cap.textContent = v.label + ' -- turn ' + v.turn + ', frame ' + v.frame;
        card.appendChild(cap);
        views.appendChild(card);
      }
      body.appendChild(views);
    } else if (tc.name === 'read_pixels' && tc.views.length) {
      const views = document.createElement('div'); views.className = 'views';
      for (const v of tc.views) {
        const card = document.createElement('div'); card.className = 'view-card';
        const canvas = document.createElement('canvas');
        card.appendChild(canvas);
        const cap = document.createElement('div'); cap.className = 'cap';
        cap.textContent = v.label + ' -- turn ' + v.turn + ', ' + v.rows + 'x' + v.columns;
        card.appendChild(cap);
        const tip = document.createElement('div'); tip.className = 'pixel-tip';
        card.appendChild(tip);
        views.appendChild(card);
        drawPixelGrid(canvas, v);
      }
      body.appendChild(views);
    } else {
      const pre = document.createElement('pre'); pre.textContent = tc.result || '(no result)';
      body.appendChild(pre);
    }
  });
  det.appendChild(body);
  return det;
}

function auxSection(label, text) {
  const det = document.createElement('details'); det.className = 'aux';
  const sum = document.createElement('summary'); sum.textContent = label;
  det.appendChild(sum);
  const pre = document.createElement('pre'); pre.textContent = text;
  det.appendChild(pre);
  return det;
}

function renderCompaction(entry) {
  const el = document.createElement('div');
  const h = document.createElement('h2'); h.textContent = 'Context compacted after turn ' + entry.turn;
  el.appendChild(h);
  const p = document.createElement('p'); p.className = 'progress';
  p.textContent = 'tokens before compaction: ' + entry.tokensBefore;
  el.appendChild(p);
  el.appendChild(auxSection('Continuation state sent to the model', entry.continuation));
  return el;
}

function renderTurn(entry) {
  const el = document.createElement('div');

  const h = document.createElement('h2');
  h.textContent = 'Turn ' + entry.index;
  const chip = actionChip(entry);
  if (chip) { h.appendChild(document.createTextNode('  ')); h.appendChild(chip); }
  if (entry.frameChanged === false && entry.actionId && !entry.protocolFailure) {
    const f = document.createElement('span'); f.className = 'noop-flag';
    f.textContent = 'frame unchanged';
    h.appendChild(f);
  }
  const avail = document.createElement('span'); avail.className = 'avail';
  avail.textContent = 'available: ' + (entry.availableActions || []).join(', ');
  h.appendChild(avail);
  el.appendChild(h);

  if (entry.protocolFailure) {
    const banner = document.createElement('div'); banner.className = 'protocol-banner';
    banner.textContent = 'Protocol failure: ' + entry.protocolFailure + ' -- no action was taken this turn.';
    el.appendChild(banner);
  }

  const frames = document.createElement('div'); frames.className = 'frames-row';
  frames.appendChild(frameBox(entry.beforeDataUri, 'before (turn ' + entry.beforeRef + ')', entry.click));
  frames.appendChild(frameBox(entry.afterDataUri, 'after (turn ' + entry.afterRef + ')', null));
  el.appendChild(frames);

  const pills = document.createElement('div'); pills.className = 'pills';
  if (entry.expectation) pills.appendChild(pill('expect', 'expectation (this turn)', entry.expectation));
  if (entry.reportedChange) pills.appendChild(pill('change', 'reported change (of previous turn)', entry.reportedChange));
  if (pills.children.length) el.appendChild(pills);

  const sectionLabel = (text) => {
    const h = document.createElement('div'); h.className = 'section-label'; h.textContent = text;
    return h;
  };

  if (entry.thinking) {
    el.appendChild(sectionLabel('Thinking'));
    const box = document.createElement('div'); box.className = 'thinking-box';
    box.textContent = entry.thinking;
    el.appendChild(box);
  }

  if (entry.reasoning) {
    el.appendChild(sectionLabel('Output'));
    const box = document.createElement('div'); box.className = 'reasoning-box';
    box.textContent = entry.reasoning;
    el.appendChild(box);
  }

  for (const tc of entry.toolCalls) el.appendChild(toolCallCard(tc));

  if (entry.rawResponse) el.appendChild(auxSection('Raw provider response', JSON.stringify(entry.rawResponse, null, 2)));

  return el;
}

function select(i) {
  if (currentIndex !== null && thumbEls[currentIndex]) thumbEls[currentIndex].classList.remove('selected');
  currentIndex = i;
  if (thumbEls[i]) {
    thumbEls[i].classList.add('selected');
    thumbEls[i].scrollIntoView({ block: 'nearest', inline: 'center' });
  }
  const detail = document.getElementById('detail');
  if (!detailCache[i]) {
    const entry = TIMELINE[i];
    detailCache[i] = entry.type === 'compaction' ? renderCompaction(entry) : renderTurn(entry);
  }
  detail.innerHTML = '';
  detail.appendChild(detailCache[i]);
}

function firstTurnIndex() { return TIMELINE.findIndex(e => e.type === 'turn'); }
function stepTurn(delta) {
  if (currentIndex === null) { select(firstTurnIndex()); return; }
  let i = currentIndex;
  do { i += delta; } while (i >= 0 && i < TIMELINE.length && false);
  i = Math.max(0, Math.min(TIMELINE.length - 1, i));
  select(i);
}

document.getElementById('prev-btn').addEventListener('click', () => stepTurn(-1));
document.getElementById('next-btn').addEventListener('click', () => stepTurn(1));
document.getElementById('jump-form').addEventListener('submit', (ev) => {
  ev.preventDefault();
  const n = parseInt(document.getElementById('jump-input').value, 10);
  const idx = TIMELINE.findIndex(e => e.type === 'turn' && e.index === n);
  if (idx >= 0) select(idx);
});
document.addEventListener('keydown', (ev) => {
  if (document.activeElement && document.activeElement.tagName === 'INPUT') return;
  if (ev.key === 'ArrowLeft') stepTurn(-1);
  else if (ev.key === 'ArrowRight') stepTurn(1);
  else if (ev.key === 'Home') select(firstTurnIndex());
  else if (ev.key === 'End') select(TIMELINE.length - 1);
});

if (TIMELINE.length) {
  select(firstTurnIndex());
} else {
  document.getElementById('detail').innerHTML = '<div class="empty">This run has no turns to show.</div>';
}
"""

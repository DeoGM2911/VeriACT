"""Context-window management: how the conversation grows, stays within budget, and degrades.

The conversation is one list, conceptually four layers, always in this order:

    SYSTEM   the system prompt              never changes for the whole run
    DIGEST   one user message               (re)written only at a cut
    WINDOW   completed turn records         appended at the back, evicted from the front
    LIVE     the turn being played now      full fidelity, always

`SessionWindow.messages` is a plain flat list; `loop.py`'s helpers append to it and know nothing
about layers. This module records where each turn's messages start and end (`TurnSpan`) and, at a
cut, rebuilds the list in place (`messages[:] = ...`) so no caller is left holding a stale
reference. Nothing but `cut()` rewrites the list, and `loop.py` only calls it between turns, so the
conversation never changes underneath a turn in progress.

A cut makes no model call. It walks a retention ladder until the budgets are under their low-water
marks:

1. drop the images that `inspect` calls pulled in (the model can request them again);
2. demote older records to text: same messages, images replaced by a placeholder that says how to
   retrieve the frame (every frame stays archived by `env.py`);
3. fold the oldest records into the digest (see `digest.py`).

The newest `keep_full_turns` records are never touched.

Two rules worth knowing:

- **Message dicts are never mutated in place.** A degraded record is built from new dicts.
- **Images are counted exactly, never estimated**, since servers enforce a hard per-request image
  limit. Only tokens are estimated, conservatively, and calibrated against reported usage.
"""
import dataclasses
from typing import Optional

GRADE_FULL = "A"  # verbatim, every image present
GRADE_TEXT = "B"  # same structure, frames replaced by a placeholder

# Tokens a single turn's *text* is assumed to add, reserved (with the per-turn image allowance)
# below the high-water mark so the turn after a cut always fits. A safety reserve rather than a
# tuning knob, so it is a module constant.
TURN_TEXT_RESERVE_TOKENS = 6000


# The text block `loop.py::_run_inspect` puts in front of the views it returns. Defined here
# because this module uses it to tell an `inspect` result apart from a turn's own observation
# frame (inspect views are dropped first).
INSPECT_VIEWS_PREFIX = "inspect: "


def is_inspect_views_message(msg) -> bool:
    """True for the follow-up `user` message carrying an `inspect` call's views."""
    content = msg.get("content")
    if not isinstance(content, list) or not content:
        return False
    first = content[0]
    return first.get("type") == "text" and (first.get("text") or "").startswith(INSPECT_VIEWS_PREFIX)


def to_text_grade(messages, placeholder_text: str, *, keep_observation_image: bool = False,
                   inspect_views_only: bool = False) -> list:
    """Grade B: the same record with its images replaced by a text placeholder.

    Structurally identical -- same messages, roles, order and `tool_call_id` pairings -- so the
    request stays valid and the model keeps all of its own text. Only the images go; every frame
    remains retrievable from `env.py`'s archive by turn number.

    Returns new dicts and never mutates the ones it was given. Messages with no image are passed
    through by reference.

    `inspect_views_only`: replace only the images an `inspect` call pulled in (ladder step 1),
    leaving the turn's own observation frame.

    `keep_observation_image`: keep the turn's observation frame. Used for the first turn of a
    level, whose frame is the "before" state later changes on that level are judged against; it is
    given up only when the record is folded into the digest.
    """
    out = []
    kept_observation = False
    for msg in messages:
        content = msg.get("content")
        if not isinstance(content, list) or not any(b.get("type") == "image_url" for b in content):
            out.append(msg)
            continue
        is_views = is_inspect_views_message(msg)
        if inspect_views_only and not is_views:
            out.append(msg)
            continue
        new_content = []
        for block in content:
            if block.get("type") != "image_url":
                new_content.append(block)
            elif keep_observation_image and not kept_observation and not is_views:
                kept_observation = True
                new_content.append(block)
            else:
                new_content.append({"type": "text", "text": placeholder_text})
        out.append({**msg, "content": new_content})
    return out


def count_images(messages) -> int:
    """Exact count of image blocks in a message list. Multimodal content is a list of blocks on
    `content`; a plain string `content` carries no image."""
    total = 0
    for msg in messages:
        content = msg.get("content")
        if isinstance(content, list):
            total += sum(1 for block in content if block.get("type") == "image_url")
    return total


def estimate_tokens(messages, tokens_per_image: int, chars_per_token: float) -> int:
    """A conservative, tokenizer-free estimate of what a message list costs to send.

    Deliberately not a real tokenizer: importing one would couple the harness to a specific served
    checkpoint, and the number only has to be good enough to schedule a cut before a hard limit is
    hit. `SessionWindow.calibrate` corrects it against reported usage as the run proceeds.

    Tool calls on an assistant message are counted too -- they are real tokens in the request.
    """
    chars = 0
    images = 0
    for msg in messages:
        content = msg.get("content")
        if isinstance(content, list):
            for block in content:
                if block.get("type") == "image_url":
                    images += 1
                else:
                    chars += len(block.get("text") or "")
        elif content:
            chars += len(content)
        for tc in msg.get("tool_calls") or []:
            chars += len(str(tc))
    return int(chars / chars_per_token) + images * tokens_per_image


@dataclasses.dataclass
class TurnSpan:
    """One completed turn's slice of `SessionWindow.messages` -- the unit that eviction and
    degradation operate on, so a turn's messages are always kept or dropped together. `start`/`end`
    are indices into the flat list and are recomputed on every rebuild.

    `level_opening` marks the first turn of a level (from the SDK's `levels_completed` only); its
    observation frame survives demotion to text.
    """
    turn: int
    level: int
    start: int
    end: int
    grade: str = GRADE_FULL
    level_opening: bool = False
    inspect_images_dropped: bool = False


@dataclasses.dataclass
class CutRecord:
    """What one cut did, logged to the trace: which turn records changed grade, and the image and
    token counts before and after."""
    turn: int
    level_boundary: bool
    images_before: int
    images_after: int
    tokens_before: int
    tokens_after: int
    inspect_images_dropped: list = dataclasses.field(default_factory=list)
    demoted_to_text: list = dataclasses.field(default_factory=list)
    digested: list = dataclasses.field(default_factory=list)
    records_kept: int = 0
    notes_truncated: bool = False
    reached_low_water: bool = True

    def as_dict(self) -> dict:
        return dataclasses.asdict(self)


class SessionWindow:
    """Owns the conversation. See the module docstring for the layer model and why `messages` is
    handed out as a plain list."""

    def __init__(self, system_message: dict, cfg):
        self.messages = [system_message]
        self._cfg = cfg
        self._spans: list = []
        self._open: Optional[TurnSpan] = None
        self._digested_turns: list = []
        # Running correction on the raw estimate, from reported usage. Never below 1.0: the
        # estimator may over-price a request but must never under-price one, since under-pricing
        # is what lets a hard limit arrive unannounced.
        self._calibration = 1.0

    # -- turn boundaries -------------------------------------------------------------------

    def begin_turn(self, turn: int, level: int, level_opening: bool = False) -> None:
        """Open a span at the current end of the list. Called *before* the turn's own user message
        is appended, so the observation frame counts against the turn's image allowance."""
        self._open = TurnSpan(
            turn=turn, level=level, start=len(self.messages), end=len(self.messages),
            level_opening=level_opening,
        )

    def end_turn(self) -> None:
        """Close the open span and add it to the window. A no-op if no turn is open, so an exception
        path that never reached `begin_turn` cannot corrupt the bookkeeping."""
        if self._open is None:
            return
        self._open.end = len(self.messages)
        self._spans.append(self._open)
        self._open = None

    # -- budgets ---------------------------------------------------------------------------

    def image_count(self) -> int:
        return count_images(self.messages)

    def images_this_turn(self) -> int:
        """Images the in-flight turn has added so far. Computed from the list rather than tracked
        incrementally -- there is no counter to forget to update on a retry path."""
        if self._open is None:
            return 0
        return count_images(self.messages[self._open.start:])

    def remaining_turn_images(self) -> int:
        return max(self._cfg.session.max_images_per_turn - self.images_this_turn(), 0)

    def estimate_tokens(self) -> int:
        raw = estimate_tokens(
            self.messages, self._cfg.session.tokens_per_image, self._cfg.session.chars_per_token,
        )
        return int(raw * self._calibration)

    def calibrate(self, total_tokens: int, output_tokens: int) -> None:
        """Correct the estimator against a reply's real usage.

        `total_tokens - output_tokens` is the prompt half, which is what this window actually
        priced. The factor is a running *maximum* rather than an average: over-estimating costs an
        earlier cut, under-estimating costs a rejected request, so the asymmetry is deliberate.
        """
        reported_prompt = (total_tokens or 0) - (output_tokens or 0)
        if reported_prompt <= 0:
            return
        raw = estimate_tokens(
            self.messages, self._cfg.session.tokens_per_image, self._cfg.session.chars_per_token,
        )
        if raw <= 0:
            return
        self._calibration = max(self._calibration, reported_prompt / raw)

    @property
    def calibration(self) -> float:
        return self._calibration

    @property
    def spans(self) -> list:
        """Completed turn records, oldest first. Read-only by convention."""
        return list(self._spans)

    @property
    def digested_turns(self) -> list:
        """Every turn the digest now stands in for, oldest first and cumulative across cuts."""
        return list(self._digested_turns)

    def over_high_water(self) -> bool:
        """Whether either context budget has crossed its high-water mark, i.e. a cut is due.
        Checked between turns; never acted on mid-turn."""
        high_tokens, _ = token_budgets(self._cfg)
        high_images, _ = image_budgets(self._cfg)
        return self.estimate_tokens() > high_tokens or self.image_count() > high_images

    def over_hard_limit(self) -> bool:
        """Whether the conversation is *already* past a budget that the provider itself enforces.
        Unlike `over_high_water` this is not a scheduling signal: since the conversation is never
        trimmed mid-turn, `loop.py` reports it and stops (`turn_does_not_fit`)."""
        high_tokens, _ = token_budgets(self._cfg)
        high_images, _ = image_budgets(self._cfg)
        return (
            self.estimate_tokens() > self._cfg.model.max_context_tokens
            or self.image_count() > self._cfg.session.max_images_in_context
        )

    # -- the cut ---------------------------------------------------------------------------

    def cut(self, digest_builder, placeholder_for, *, turn: int = 0,
            level_boundary: bool = False, notes_truncated: bool = False) -> CutRecord:
        """Apply the retention ladder until both context budgets are under the low-water mark, then
        rebuild the conversation around a fresh digest. No model call is made.

        `digest_builder(turn_numbers) -> str` renders the digest message for every turn the digest
        now stands in for, cumulatively across cuts. `placeholder_for(turn) -> str` renders what an
        evicted image leaves behind. Both come from `prompts/`, so no model-facing wording lives
        here.

        The ladder is best-effort: if the eligible set is exhausted and the budgets are still not
        met, it stops rather than eating into the verbatim window, and says so on the returned
        record. `keep_full_turns` is a floor nothing may cross, not merely a preference.
        """
        cfg = self._cfg
        _, low_tokens = token_budgets(cfg)
        _, low_images = image_budgets(cfg)
        images_before, tokens_before = self.image_count(), self.estimate_tokens()

        system = self.messages[0]
        tail_start = self._spans[-1].end if self._spans else 1
        tail = list(self.messages[tail_start:])  # nothing at a turn boundary; defensive
        parts = [list(self.messages[s.start:s.end]) for s in self._spans]
        spans = [dataclasses.replace(s) for s in self._spans]

        # Price the digest at its *ceiling* rather than its current size: the ladder has to decide
        # how much to give up before the digest it is making room for exists.
        digest_reserve = int(
            (cfg.session.digest_max_chars + 2 * cfg.session.notes_push_max_chars)
            / cfg.session.chars_per_token
        )

        def fits(dropped: int) -> bool:
            candidate = [system] + [m for part in parts[dropped:] for m in part] + tail
            tokens = int(
                estimate_tokens(candidate, cfg.session.tokens_per_image, cfg.session.chars_per_token)
                * self._calibration
            ) + digest_reserve
            return tokens <= low_tokens and count_images(candidate) <= low_images

        eligible = max(len(parts) - cfg.session.keep_full_turns, 0)
        record = CutRecord(
            turn=turn, level_boundary=level_boundary, images_before=images_before,
            images_after=images_before, tokens_before=tokens_before, tokens_after=tokens_before,
            notes_truncated=notes_truncated,
        )

        # Ladder step 1: give up `inspect` views (the model can request them again).
        for i in range(eligible):
            if fits(0):
                break
            if spans[i].inspect_images_dropped:
                continue
            parts[i] = to_text_grade(
                parts[i], placeholder_for(spans[i].turn), inspect_views_only=True,
            )
            spans[i].inspect_images_dropped = True
            record.inspect_images_dropped.append(spans[i].turn)

        # Ladder step 2: FULL -> TEXT. Keeps every word, gives up the frames.
        for i in range(eligible):
            if fits(0):
                break
            if spans[i].grade != GRADE_FULL:
                continue
            parts[i] = to_text_grade(
                parts[i], placeholder_for(spans[i].turn),
                keep_observation_image=spans[i].level_opening,
            )
            spans[i].grade = GRADE_TEXT
            record.demoted_to_text.append(spans[i].turn)

        # Ladder step 3: TEXT -> DIGESTED. Anything older than the whole window goes regardless of
        # budget -- `keep_text_turns` is a cap on how long the window may be, not only a lever --
        # and budget pressure digests further, down to the `keep_full_turns` floor.
        dropped = max(len(parts) - (cfg.session.keep_full_turns + cfg.session.keep_text_turns), 0)
        while dropped < eligible and not fits(dropped):
            dropped += 1
        record.digested = [s.turn for s in spans[:dropped]]

        self._digested_turns = self._digested_turns + record.digested
        digest_message = {"role": "user", "content": digest_builder(self._digested_turns)}

        rebuilt = [system, digest_message]
        index = len(rebuilt)
        kept_spans = []
        for span, part in zip(spans[dropped:], parts[dropped:]):
            span.start, span.end = index, index + len(part)
            index += len(part)
            rebuilt.extend(part)
            kept_spans.append(span)
        rebuilt.extend(tail)

        self.messages[:] = rebuilt  # same list object: no caller is left holding a stale reference
        self._spans = kept_spans

        record.images_after = self.image_count()
        record.tokens_after = self.estimate_tokens()
        record.records_kept = len(kept_spans)
        record.reached_low_water = (
            record.tokens_after <= low_tokens and record.images_after <= low_images
        )
        return record


def token_budgets(cfg) -> tuple:
    """`(high_water, low_water)` in tokens.

    The low-water mark is clamped to `high_water - worst-case turn cost`: cuts never happen
    mid-turn, so what a cut leaves behind must have room for a whole turn on top of it.
    `target_after_cut_frac` can lower the mark (making cuts deeper and rarer) but never raise it
    above that ceiling.
    """
    high = int(cfg.session.compact_at_frac * cfg.model.max_context_tokens)
    reserve = cfg.session.max_images_per_turn * cfg.session.tokens_per_image + TURN_TEXT_RESERVE_TOKENS
    requested = int(cfg.session.target_after_cut_frac * cfg.model.max_context_tokens)
    return high, max(min(requested, high - reserve), 0)


def image_budgets(cfg) -> tuple:
    """`(high_water, low_water)` in images, with the same one-turn reserve as `token_budgets`. The
    reserve is exact: a turn cannot add more than `max_images_per_turn` images, because an
    `inspect` call that would exceed it is rejected."""
    high = cfg.session.max_images_in_context
    requested = int(cfg.session.target_after_cut_frac * high)
    return high, max(min(requested, high - cfg.session.max_images_per_turn), 0)

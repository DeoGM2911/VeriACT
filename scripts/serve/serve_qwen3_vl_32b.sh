#!/usr/bin/env bash
# Hosts Qwen/Qwen3-VL-32B-Instruct behind an OpenAI-compatible vLLM server, for
# config/default.yaml's `model.base_url` to point at.
#
# One of two model-serving scripts (the other is ./serve_gemma4_31b.sh, for
# google/gemma-4-31B-it) -- switch config/default.yaml's `model:` block to match whichever one
# you actually start.
#
# Values here (model id, --max-model-len, port) must stay in sync with
# config/default.yaml's `model:` block -- nothing enforces that automatically.
#
# This replaces an earlier version of this script that served the plain (non-VL) `Qwen/Qwen3-32B`
# -- a text-only dense model. It cannot be used with this harness at all: the observation this
# harness sends every turn is an image and nothing else (harness/CLAUDE.md), and vLLM rejects an
# image-carrying request against that model outright ("... is not a multimodal model"). Confirmed
# against a live run before switching. `Qwen3-VL-32B-Instruct` is the vision-capable sibling in
# the same size class.
#
# The **-Instruct** checkpoint is used deliberately, not **-Thinking**: as of vLLM's own tracker,
# `--reasoning-parser qwen3` support for Qwen3-VL's thinking variants was still an open feature
# request (https://github.com/vllm-project/vllm/issues/26239) -- the <think> block leaks into
# `content` instead of landing in `reasoning_content`. This project has already been burned once
# by a silently-wrong reasoning field (see harness/CLAUDE.md's M1 paragraph, the
# `reasoning_content` vs `reasoning` bug against gemma4) and isn't repeating that against a
# tracker issue that says the parser doesn't work yet. -Instruct emits no <think> tags at all, so
# no --reasoning-parser or `enable_thinking` chat-template kwarg is passed here. Revisit if that
# tracker issue closes and thinking mode is wanted.
#
# Needs vllm>=0.11.0 (Qwen3-VL support landed there) and, per the model card, may need
# transformers built from source (`pip install git+https://github.com/huggingface/transformers`)
# if your pinned transformers predates upstream Qwen3-VL support -- check `vllm serve` errors
# about an unrecognized model type first.
#
# Usage:
#   NUM_GPUS=4 ./scripts/serve_qwen3_vl_32b.sh
#
# Env vars (all optional):
#   NUM_GPUS        tensor-parallel degree, i.e. number of GPUs to shard the model across.
#                   Default 4. Qwen3-VL-32B's dense text backbone has 64 attention heads / 8 KV
#                   heads, so 1/2/4/8 all divide cleanly; other values may still work via vLLM's
#                   KV-head replication but aren't as clean a fit -- confirm the server actually
#                   starts before trusting one.
#   PORT            default 8000, matching config/default.yaml's base_url.
#   MAX_MODEL_LEN   default 131072. Qwen3-VL-32B's *native* context is 256K, well above this, so
#                   -- unlike the old text-only script -- no YaRN/rope-scaling is needed at any
#                   value up to that native length.
#   GPU_MEM_UTIL    default 0.90 -- fraction of each GPU's memory vLLM may claim for weights + KV
#                   cache. Lower this if you're sharing the GPU with anything else.
#   MM_IMAGE_LIMIT  default 128 -- max images vLLM will accept in a single request
#                   (--limit-mm-per-prompt image=...). This harness's `session.keep_last_n_images`
#                   defaults to `all` (config/default.yaml) and nothing currently reads it as
#                   anything but that literal string, so a single turn's request can carry every
#                   image shown so far this compaction window, not just the current turn's, and
#                   click-tagged games add a second image per turn on top of that (the confirm-crop
#                   exchange). Raising this number only delays the crash -- confirmed against a full
#                   25-game batch (docs/batch-findings-vista-comparison-v1.md Bucket 1): 8/25 games
#                   hit the old default of 64 well under `max_turns`, several with zero images spent
#                   on inspect/read_pixels. The actual fix is bounded image retention in the
#                   harness itself (an integer `keep_last_n_images` with real eviction logic, which
#                   does not exist yet), not a bigger number here.
#   VLLM_API_KEY    if set, required by clients as a bearer token and matches
#                   config/default.yaml's model.api_key_env. If unset, the server is unauthenticated
#                   -- fine for localhost-only use, not for anything reachable off-box.
#
# What this enables:
#   --tool-call-parser hermes --enable-auto-tool-choice
#       function/tool calling, which loop.py needs for the `play` tool -- Qwen3-VL's chat
#       template emits the same Hermes-style <tool_call> blocks the plain Qwen3 models do. (Not
#       "qwen3_coder" -- that parser is for the Qwen3-Coder / newer MoE variants' tool-call
#       format, not this model's.)
#   --limit-mm-per-prompt '{"image": '"$MM_IMAGE_LIMIT"'}'
#       per vLLM's Qwen3-VL deployment notes, this bounds how many images one request may carry.
#       Takes a JSON object, not vLLM's older key=value form (that form errors as
#       "cannot be converted to <function loads>" on current vLLM).
set -euo pipefail

MODEL="Qwen/Qwen3-VL-32B-Instruct"
NUM_GPUS="${NUM_GPUS:-4}"
PORT="${PORT:-8000}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-131072}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.90}"
MM_IMAGE_LIMIT="${MM_IMAGE_LIMIT:-128}"

if ! python -c "import vllm" >/dev/null 2>&1; then
    echo "vllm not found in this environment; installing it..." >&2
    pip install "vllm>=0.11.0"
fi

API_KEY_ARGS=()
if [[ -n "${VLLM_API_KEY:-}" ]]; then
    API_KEY_ARGS=(--api-key "$VLLM_API_KEY")
fi

exec vllm serve "$MODEL" \
    --tensor-parallel-size "$NUM_GPUS" \
    --max-model-len "$MAX_MODEL_LEN" \
    --gpu-memory-utilization "$GPU_MEM_UTIL" \
    --port "$PORT" \
    --enable-auto-tool-choice \
    --tool-call-parser hermes \
    --limit-mm-per-prompt "{\"image\": ${MM_IMAGE_LIMIT}}" \
    "${API_KEY_ARGS[@]}"

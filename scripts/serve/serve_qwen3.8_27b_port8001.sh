#!/usr/bin/env bash
# Second instance of serve_qwen3.8_27b.sh, defaulting to PORT=8001 so two servers can share one machine.
# Pick GPUs yourself, e.g. CUDA_VISIBLE_DEVICES=2,3 NUM_GPUS=2 ./scripts/serve_qwen3.8_27b_port8001.sh -- and
# use --config config/qwen3.8_27b_port8001.yaml with eval.py/run.py. Otherwise identical to serve_qwen3.8_27b.sh.
# Hosts Qwen/Qwen3.8-27B behind an OpenAI-compatible vLLM server, for config/default.yaml's
# `model.base_url` to point at.
#
# One of three model-serving scripts (the others: ./serve_gemma4_31b.sh for
# google/gemma-4-31B-it, and ./serve_qwen3_vl_32b.sh for the older Qwen/Qwen3-VL-32B-Instruct) --
# switch config/default.yaml's `model:` block to match whichever one you actually start.
#
# Values here (model id, --max-model-len, port) must stay in sync with config/default.yaml's
# `model:` block -- nothing enforces that automatically.
#
# Why Qwen3.8-27B over the older Qwen3-VL-32B-Instruct this replaces as the default:
# native vision-language, dense 27B, natively 262,144 tokens of context (double the old
# checkpoint's 131,072 configured max, and without any YaRN quality trade-off -- see
# MAX_MODEL_LEN below), and vLLM's own recipe (recipes.vllm.ai/Qwen/Qwen3.8-27B) documents a
# working `--reasoning-parser qwen3` for its thinking mode -- unlike Qwen3-VL's -Thinking
# variant, which was blocked on an open vLLM tracker issue (vllm-project/vllm#26239) when the
# prior script's -Instruct choice was made (see that script's own header). Thinking mode is ON
# by default here, matching the model card, and its output lands in `reasoning_content` via
# --reasoning-parser -- llm.py already checks both `reasoning_content` and `reasoning` field
# names (see harness/CLAUDE.md's M1 paragraph on the gemma4 field-name bug), so either way it's
# captured, not silently dropped.
#
# Needs vllm>=0.17.0 (the version floor the Qwen3.8-27B recipe documents for `--reasoning-parser
# qwen3` + `--tool-call-parser qwen3_coder` support). Confirm `vllm serve` starts cleanly before
# trusting an older pinned version.
#
# Usage:
#   NUM_GPUS=4 ./scripts/serve_qwen3.8_27b.sh
#
# Env vars (all optional):
#   NUM_GPUS        tensor-parallel degree, i.e. number of GPUs to shard the model across.
#                   Default 4, matching serve_qwen3_vl_32b.sh. Qwen3.8-27B's gated-attention
#                   component has 24 Q heads / 4 KV heads (its gated-deltanet layers, the
#                   majority, don't constrain TP the same way) -- 1/2/4 all divide the 4 KV heads
#                   cleanly; confirm the server actually starts before trusting any other value.
#   PORT            default 8001 (second server on the same box; serve_qwen3.8_27b.sh uses 8000). Match config/qwen3.8_27b_port8001.yaml.
#   MAX_MODEL_LEN   default 262144 -- Qwen3.8-27B's full native context, no rope-scaling
#                   required. This is deliberately NOT pushed to the model's YaRN-extended 1M:
#                   the model card's own note is that "all the notable open-source frameworks
#                   implement static YaRN, which means the scaling factor remains constant
#                   regardless of input length, potentially impacting performance on shorter
#                   texts" -- and this harness's turns are nowhere near 1M tokens (session.
#                   compact_at_frac keeps it well under 262144 already), so YaRN would trade away
#                   quality on every ordinary turn for headroom this harness doesn't use. Set
#                   LONG_CONTEXT=1 below if a run genuinely needs beyond 262144.
#   LONG_CONTEXT    if set to any non-empty value, extends context past the native 262144 via the
#                   model card's documented YaRN hf-overrides (factor 4.0, i.e. up to ~1,048,576)
#                   instead of the plain --max-model-len path. MAX_MODEL_LEN still sets the cap
#                   when this is on. Off by default -- see MAX_MODEL_LEN's note on why.
#   GPU_MEM_UTIL    default 0.90 -- fraction of each GPU's memory vLLM may claim for weights + KV
#                   cache. Lower this if you're sharing the GPU with anything else.
#   KV_CACHE_DTYPE  default fp8 -- halves KV-cache memory per token versus the model's native
#                   bf16 weights, which is what actually buys headroom for a long context budget
#                   at a fixed GPU_MEM_UTIL (this is what "prefer max context" cashes out to in
#                   practice, not just a bigger --max-model-len number that OOMs on the first long
#                   run). Independent of the checkpoint's own weight dtype. Set to "auto" to
#                   disable and use the weights' native dtype for KV cache instead.
#   MM_IMAGE_LIMIT  default 128 -- max images vLLM will accept in a single request
#                   (--limit-mm-per-prompt image=...). Must stay above config/default.yaml's
#                   session.max_images_in_context (100) with headroom for one turn's own images --
#                   see that config file's comment on why raising this number alone isn't a fix
#                   for unbounded image growth.
#   MAX_NUM_SEQS    default 4 -- caps how many requests vLLM sizes its post-load memory-profiling
#                   step for (--max-num-seqs). Left unset, vLLM defaults to a high concurrency
#                   figure (256 in current versions) and profiles for the *worst case across all
#                   of them simultaneously* -- at this script's MAX_MODEL_LEN and MM_IMAGE_LIMIT,
#                   that means sizing as if up to 256 requests could each carry ~262K tokens and
#                   128 full-resolution images at once. That single profiling step is what hung
#                   (not merely slowed) an actual serving box hard enough that even SSH into it
#                   timed out -- almost certainly host RAM exhaustion from the image-preprocessing
#                   buffers that sizing implies, not GPU VRAM (which had headroom to spare).
#                   loop.py only ever drives one conversation with the server at a time, so there
#                   is no throughput reason to pay for high concurrency here; 4 leaves a little
#                   slack (e.g. a stray retry) without reintroducing the blow-up. Raise this only
#                   if something in this harness starts actually issuing concurrent requests.
#   ENFORCE_EAGER   if set to any non-empty value, passes --enforce-eager, which disables
#                   torch.compile AND CUDA-graph capture. That phase runs between weight loading
#                   and the API server binding its port, and -- critically -- it runs on the CPU
#                   and *host* RAM, not VRAM. A cold TorchInductor compile of this model's 64
#                   layers across two structurally different layer types (Gated DeltaNet and
#                   Gated Attention) is heavy enough to have taken a whole serving box
#                   unresponsive (SSH included) while its GPUs sat nearly idle, with plenty of
#                   VRAM free. The two older scripts don't hit this because their models had
#                   already been served on that box and their compile artifacts were cached in
#                   ~/.cache/vllm/; a model new to a machine pays the full cost. Costs meaningful
#                   steady-state throughput, so this is a diagnostic and a fallback, not a
#                   default -- once a cold start succeeds once, the cache makes later starts
#                   cheap and this can come back off.
#   VLLM_API_KEY    if set, required by clients as a bearer token and matches
#                   config/default.yaml's model.api_key_env. If unset, the server is unauthenticated
#                   -- fine for localhost-only use, not for anything reachable off-box.
#   MEM_LIMIT_PCT   default 90 -- host RAM percent-used threshold. A background watchdog polls
#                   /proc/meminfo every MEM_CHECK_INTERVAL seconds and sends SIGTERM (then SIGKILL
#                   if it doesn't exit) to the whole vLLM process group the instant used memory
#                   reaches this. Exists because a cold-start ENFORCE_EAGER=0 compile/CUDA-graph
#                   capture on this model has taken a serving box unresponsive -- SSH included --
#                   while its GPUs sat nearly idle (see ENFORCE_EAGER above); this stops vLLM
#                   itself before host memory pressure gets that far, rather than relying on the
#                   OOM killer to pick the right process, or on someone watching a terminal.
#                   Set to 100 (or anything unreachable) to disable.
#   MEM_CHECK_INTERVAL default 5 -- seconds between watchdog checks. Lower this if you'd rather
#                   catch a fast climb sooner at the cost of more frequent /proc/meminfo reads
#                   (negligible cost either way).
#
# What this enables:
#   --reasoning-parser qwen3
#       thinking-mode output (the model's <think>...</think> block) parsed into
#       `message.reasoning_content` rather than leaking into `message.content` -- per
#       recipes.vllm.ai/Qwen/Qwen3.8-27B, confirmed working (unlike Qwen3-VL's -Thinking variant
#       at the time serve_qwen3_vl_32b.sh was written). Thinking is ON by default per the model
#       card; config/default.yaml's model.reasoning_effort controls depth (this vLLM build's own
#       enum -- none/low/medium/high, not the model card's xhigh/medium/low) and
#       is sent as a top-level request field, not a server flag.
#   --tool-call-parser qwen3_coder --enable-auto-tool-choice
#       function/tool calling, which loop.py needs for the `play` tool. Per the same recipe,
#       Qwen3.8-27B's tool-call format is parsed by "qwen3_coder", not the "hermes" parser the
#       older Qwen3-VL script used.
#   --limit-mm-per-prompt '{"image": '"$MM_IMAGE_LIMIT"'}'
#       per vLLM's Qwen deployment notes, this bounds how many images one request may carry.
#       Takes a JSON object, not vLLM's older key=value form.
#   --max-num-seqs "$MAX_NUM_SEQS"
#       see MAX_NUM_SEQS above -- keeps the memory-profiling step sized for this harness's actual
#       (single-conversation) concurrency instead of vLLM's high-throughput default.
set -euo pipefail
set -m  # job control on, so the backgrounded vLLM server gets its own process group -- see
        # below, this is what lets the watchdog kill it (and its TP worker subprocesses) as a unit.

MODEL="Qwen/Qwen3.8-27B"
NUM_GPUS="${NUM_GPUS:-4}"
PORT="${PORT:-8001}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-262144}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.90}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-4}"
KV_CACHE_DTYPE="${KV_CACHE_DTYPE:-fp8}"
MM_IMAGE_LIMIT="${MM_IMAGE_LIMIT:-128}"
MEM_LIMIT_PCT="${MEM_LIMIT_PCT:-90}"
MEM_CHECK_INTERVAL="${MEM_CHECK_INTERVAL:-5}"

if ! python -c "import vllm" >/dev/null 2>&1; then
    echo "vllm not found in this environment; installing it..." >&2
    pip install "vllm>=0.17.0"
fi

API_KEY_ARGS=()
if [[ -n "${VLLM_API_KEY:-}" ]]; then
    API_KEY_ARGS=(--api-key "$VLLM_API_KEY")
fi

EAGER_ARGS=()
if [[ -n "${ENFORCE_EAGER:-}" ]]; then
    EAGER_ARGS=(--enforce-eager)
fi

HF_OVERRIDES_ARGS=()
if [[ -n "${LONG_CONTEXT:-}" ]]; then
    # Static YaRN, per the model card's own recipe -- factor 4.0 extends the native 262,144 up to
    # ~1,048,576. See MAX_MODEL_LEN's comment above for why this isn't the default.
    export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
    HF_OVERRIDES_ARGS=(--hf-overrides '{"text_config": {"rope_parameters": {"mrope_interleaved": true, "mrope_section": [11, 11, 10], "rope_type": "yarn", "rope_theta": 10000000, "partial_rotary_factor": 0.25, "factor": 4.0, "original_max_position_embeddings": 262144}}}')
fi

# Not `exec`'d: the watchdog below needs the shell to stay alive alongside vLLM so it can act on
# what it observes, rather than handing the process image away entirely.
vllm serve "$MODEL" \
    --tensor-parallel-size "$NUM_GPUS" \
    --max-model-len "$MAX_MODEL_LEN" \
    --gpu-memory-utilization "$GPU_MEM_UTIL" \
    --kv-cache-dtype "$KV_CACHE_DTYPE" \
    --max-num-seqs "$MAX_NUM_SEQS" \
    --port "$PORT" \
    --reasoning-parser qwen3 \
    --enable-auto-tool-choice \
    --tool-call-parser qwen3_coder \
    --limit-mm-per-prompt "{\"image\": ${MM_IMAGE_LIMIT}}" \
    "${EAGER_ARGS[@]}" \
    "${HF_OVERRIDES_ARGS[@]}" \
    "${API_KEY_ARGS[@]}" &
VLLM_PID=$!

# Host-memory watchdog (MEM_LIMIT_PCT above). Reads /proc/meminfo directly rather than shelling
# out to `free` -- MemAvailable already accounts for reclaimable cache/buffers the way `free`'s
# own "available" column does, so this is the same number without depending on `free`'s output
# format being stable across distros.
cleanup() {
    if kill -0 "-$VLLM_PID" 2>/dev/null || kill -0 "$VLLM_PID" 2>/dev/null; then
        kill -TERM "-$VLLM_PID" 2>/dev/null || kill -TERM "$VLLM_PID" 2>/dev/null || true
    fi
}
trap cleanup EXIT INT TERM

echo "vLLM started (pid $VLLM_PID); watchdog will stop it if host memory use reaches ${MEM_LIMIT_PCT}% (checking every ${MEM_CHECK_INTERVAL}s)." >&2

while kill -0 "$VLLM_PID" 2>/dev/null; do
    sleep "$MEM_CHECK_INTERVAL"
    kill -0 "$VLLM_PID" 2>/dev/null || break

    mem_total=""
    mem_avail=""
    read -r mem_total mem_avail < <(awk '
        /^MemTotal:/     { total = $2 }
        /^MemAvailable:/ { avail = $2 }
        END              { print total, avail }
    ' /proc/meminfo 2>/dev/null)

    if [[ -z "$mem_total" || -z "$mem_avail" || "$mem_total" -eq 0 ]]; then
        continue  # /proc/meminfo unreadable this cycle -- don't act on missing data
    fi

    used_pct=$(( (mem_total - mem_avail) * 100 / mem_total ))

    if (( used_pct >= MEM_LIMIT_PCT )); then
        echo "Host memory at ${used_pct}% (limit ${MEM_LIMIT_PCT}%) -- stopping vLLM (pid $VLLM_PID) before the box locks up." >&2
        kill -TERM "-$VLLM_PID" 2>/dev/null || kill -TERM "$VLLM_PID" 2>/dev/null || true
        sleep 5
        if kill -0 "$VLLM_PID" 2>/dev/null; then
            kill -KILL "-$VLLM_PID" 2>/dev/null || kill -KILL "$VLLM_PID" 2>/dev/null || true
        fi
        exit 1
    fi
done

wait "$VLLM_PID"

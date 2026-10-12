#!/usr/bin/env bash
# Boot order is load-bearing.
#
# vLLM profiles free VRAM at startup and sizes its KV pool from what it sees. Start the
# other processes first and it silently takes a smaller pool -- same config, worse
# throughput, no warning. So the reasoning core goes first with an explicit fraction, the
# document reader takes its share next, and the small models fill the remainder.
set -euo pipefail

# Component mapping: a COMPONENT_* variable you pass wins, then /app/components.env if you
# mount one, then the defaults baked into the image (components.env.example).
load_components() {
    local key value
    while IFS='=' read -r key value || [[ -n "${key}" ]]; do
        value="${value%$'\r'}"
        [[ "${key}" =~ ^COMPONENT_[A-Z_]+$ && -n "${value}" ]] || continue
        [[ -n "${!key:-}" ]] || export "${key}=${value}"
    done < "$1"
}
[[ -f /app/components.env ]] && load_components /app/components.env
load_components /app/components.env.default
COMPONENT_NAMES="$(env | grep '^COMPONENT_' | grep -v '^COMPONENT_NAMES=' | cut -d= -f2- | paste -sd, -)"
export COMPONENT_NAMES

# Some secret stores expose the token in lowercase or under the older name.
export HF_TOKEN="${HF_TOKEN:-${HUGGING_FACE_HUB_TOKEN:-${hf_token:-}}}"
export HUGGING_FACE_HUB_TOKEN="${HF_TOKEN}"
if [[ -z "${HF_TOKEN}" ]]; then
    echo "FATAL: HF_TOKEN is unset. The diarization and guard models are gated on Hugging" >&2
    echo "       Face; accept their terms and pass -e HF_TOKEN=hf_..." >&2
    exit 1
fi

# The services start many threads; a finite stack keeps that from exhausting memory.
ulimit -s 65536 || true

BRAIN_MODEL="${BRAIN_MODEL:-${COMPONENT_BRAIN}}"
SERVED_NAME="${BRAIN_SERVED_NAME:-interfaze-lite}"
SPEC_TOKENS="${BRAIN_SPEC_TOKENS:-3}"

PIDS=()
term() { echo "shutting down"; kill "${PIDS[@]}" 2>/dev/null || true; wait || true; }
trap term SIGTERM SIGINT

# wait_for NAME URL PID BUDGET_S
wait_for() {
    local started=$SECONDS
    until curl -fsS "$2" >/dev/null 2>&1; do
        if ! kill -0 "$3" 2>/dev/null; then echo "FATAL: $1 exited during startup" >&2; exit 1; fi
        if (( SECONDS - started > $4 )); then echo "FATAL: $1 not healthy after $4 s" >&2; exit 1; fi
        sleep 5
    done
    echo "      $1 ready in $(( SECONDS - started )) s"
}

echo "[1/5] reasoning core on :8001"
BRAIN_ARGS=(
    serve "${BRAIN_MODEL}" --host 127.0.0.1 --port 8001
    --served-model-name "${SERVED_NAME}"
    --max-model-len "${MAX_MODEL_LEN:-131072}"
    # The hybrid attention layers allocate one state block per concurrent sequence; vLLM's
    # default of 1024 does not fit next to the other models and CUDA graph capture aborts.
    --max-num-seqs "${MAX_NUM_SEQS:-6}"
    # Below 0.90 on purpose: the document reader and the perception models share the card.
    --gpu-memory-utilization "${GPU_FRACTION:-0.56}"
    # Not fp8: this FP8 checkpoint ships no attention scaling factors, and fp8 KV made it
    # return about 4x the boxes it should on an H100. Opt in only after re-measuring.
    --kv-cache-dtype "${KV_CACHE_DTYPE:-auto}"
    --enable-prefix-caching
    --enable-auto-tool-choice --tool-call-parser "${TOOL_CALL_PARSER:-qwen3_coder}"
    --trust-remote-code
)
[[ "${REASONING_PARSER:-qwen3}" != "off" ]] && BRAIN_ARGS+=(--reasoning-parser "${REASONING_PARSER:-qwen3}")
(( SPEC_TOKENS > 0 )) && BRAIN_ARGS+=(--speculative-config "{\"method\":\"mtp\",\"num_speculative_tokens\":${SPEC_TOKENS}}")
vllm "${BRAIN_ARGS[@]}" &
PIDS+=($!); wait_for "reasoning core" http://127.0.0.1:8001/health "$!" 3600

echo "[2/5] document reader on :8004"
vllm serve "${COMPONENT_OCR_VLM}" --host 127.0.0.1 --port 8004 \
    --served-model-name "${OCR_SERVED_NAME:-document-reader}" \
    --max-model-len "${OCR_MAX_MODEL_LEN:-32768}" \
    --max-num-seqs 16 \
    --gpu-memory-utilization "${OCR_GPU_FRACTION:-0.20}" \
    --enable-prefix-caching \
    --limit-mm-per-prompt '{"image": 1}' \
    --chat-template-content-format string \
    --trust-remote-code &
PIDS+=($!); wait_for "document reader" http://127.0.0.1:8004/health "$!" 3600

echo "[3/5] perception on :8002"
uvicorn interfaze_lite.services.perception:app --host 127.0.0.1 --port 8002 &
PIDS+=($!); wait_for "perception" http://127.0.0.1:8002/health "$!" 1800

echo "[4/5] diarization on :8003"
uvicorn interfaze_lite.services.diarize:app --host 127.0.0.1 --port 8003 &
PIDS+=($!); wait_for "diarization" http://127.0.0.1:8003/health "$!" 1800

echo "[5/5] interfaze-1-lite on :8000"
uvicorn interfaze_lite.app:app --host 0.0.0.0 --port 8000 &
PIDS+=($!); wait_for "orchestrator" http://127.0.0.1:8000/health "$!" 300
echo "ready: POST http://localhost:8000/v1/chat/completions"

# If any process dies, stop the container so its restart policy brings the stack back
# together, rather than serving with a piece missing.
set +e
wait -n "${PIDS[@]}"
echo "a service exited (status $?); stopping the container" >&2
term
exit 1

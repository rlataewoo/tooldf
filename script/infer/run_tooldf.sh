#!/usr/bin/env bash
# Single-GPU eval.
#
# Usage:
#   bash script/infer/run_tooldf.sh [SPLIT_NAME]
#
# Env knobs:
#   PYTHON        python interpreter to use (default: `python` from the active
#                 env). Activate your conda env first, or set PYTHON to its
#                 interpreter. swift must be installed in whichever env is used.
#   GPU           CUDA device id (default: 0)
#   ENGINE        pt | vllm  (default: vllm)
#   MAX_RETRIES   retry attempts on failure (default: 2).
#                 Resume is automatic — the python script counts valid lines
#                 in the existing output file and appends from there.
#   ENFORCE_EAGER 1 to pass --enforce_eager to vLLM (default: 0).
#                 Disables CUDA-graph capture: small throughput hit, eliminates
#                 a known RAM/VRAM spike during warmup.
#   SAMPLES_PER_RUN  samples per batch before recycling the python process
#                 (default: 400). Each batch ends with process exit, which
#                 releases vLLM prefix-cache / MM-cache / glibc arenas to the
#                 OS so RSS doesn't grow unbounded. Resume picks up the next
#                 batch automatically. Lower this if RSS still grows within a
#                 batch; raise it to amortize model-load overhead.
#
# Output:
#   ${OUT_BASE}   (num_shards=1, so no .shard{ID} suffix and no merge step)
set -euo pipefail

SPLIT="${1:-c2}"
PYTHON="${PYTHON:-python}"
GPU="${GPU:-0}"
ENGINE="${ENGINE:-vllm}"
MAX_RETRIES="${MAX_RETRIES:-2}"
ENFORCE_EAGER="${ENFORCE_EAGER:-0}"
SAMPLES_PER_RUN="${SAMPLES_PER_RUN:-400}"

# Cap thread fan-out. Without this the process spawns OMP/BLAS threads = CPU
# count, which multiplies glibc malloc arenas and inflates peak host RAM.
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-4}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export MALLOC_ARENA_MAX="${MALLOC_ARENA_MAX:-2}"

if ! command -v "${PYTHON}" >/dev/null 2>&1; then
    echo "[1gpu] ERROR: python interpreter not found: ${PYTHON}" >&2
    echo "[1gpu]   activate your env or set PYTHON=/path/to/python" >&2
    exit 1
fi

# vllm 0.23.0 here is a CUDA-13 wheel (vllm._C links libcudart.so.13) while
# torch is cu128. Both runtimes are installed; only the cu12 one is on the
# loader path by default, so vllm._C fails to load. Add nvidia/cu13/lib.
SITE_PACKAGES="$("${PYTHON}" -c 'import site; print(site.getsitepackages()[0])')"
CU13_LIB="${SITE_PACKAGES}/nvidia/cu13/lib"
if [[ -d "${CU13_LIB}" ]]; then
    export LD_LIBRARY_PATH="${CU13_LIB}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
fi

# These GPUs are Blackwell (sm120). FlashInfer JIT-compiles its sampling kernel
# and refuses sm12.x unless nvcc >= 12.9; this env's toolkit is 12.8, so engine
# startup dies with "FlashInfer requires GPUs with sm75 or higher". Fall back to
# vLLM's native sampler. Eval decodes greedily (temperature=0), so no behavior
# change -- the sampler only masks logits.
export VLLM_USE_FLASHINFER_SAMPLER="${VLLM_USE_FLASHINFER_SAMPLER:-0}"

MODEL_PATH="pretrained/qwen2.5-omni-3b-tooldf-sft/v0-20260427-161534/checkpoint-10047-merged"
DATASET_PATH="label/tooldf_sft/${SPLIT}_eval_sft.jsonl"
OUT_BASE="${OUT_BASE:-result/tooldf_sft_${SPLIT}.jsonl}"
LOG_DIR="${LOG_DIR:-logs/tooldf_eval}"
LOG="${LOG_DIR}/${SPLIT}.gpu${GPU}.log"
mkdir -p "$LOG_DIR" "$(dirname "${OUT_BASE}")"

EAGER_FLAG=""
if [[ "${ENFORCE_EAGER}" == "1" ]]; then
    EAGER_FLAG="--enforce_eager"
fi

echo "[1gpu] split=${SPLIT}  gpu=${GPU}  engine=${ENGINE}"
echo "[1gpu] retries=${MAX_RETRIES}  enforce_eager=${ENFORCE_EAGER}"
echo "[1gpu] samples_per_run=${SAMPLES_PER_RUN} (process recycle interval)"
echo "[1gpu] threads: OMP=${OMP_NUM_THREADS} MKL=${MKL_NUM_THREADS} MALLOC_ARENA_MAX=${MALLOC_ARENA_MAX}"
echo "[1gpu] log: ${LOG}"

AASIST_HOST="${AASIST_HOST:-127.0.0.1}"
AASIST_PORTS=(8021 8022 8023 8024 8025 8026 8027 8028)
DOWN_PORTS=()
for PORT in "${AASIST_PORTS[@]}"; do
    if ! curl -fsS --max-time 3 "http://${AASIST_HOST}:${PORT}/health" >/dev/null 2>&1; then
        DOWN_PORTS+=("${PORT}")
    fi
done
if [[ ${#DOWN_PORTS[@]} -gt 0 ]]; then
    echo "[1gpu] ERROR: AASIST API not fully open on ${AASIST_HOST}." >&2
    echo "[1gpu]   unreachable ports: ${DOWN_PORTS[*]}" >&2
    echo "[1gpu]   start them with: bash script/infer/run_aasist_api.sh" >&2
    exit 1
fi
echo "[1gpu] AASIST API healthy on ports: ${AASIST_PORTS[*]}"

# shard_id=0 -> plugin/tools/separator.py resolves demucs to PORT_BASE + 0.
DEMUCS_HOST="${DEMUCS_HOST:-127.0.0.1}"
DEMUCS_PORT_BASE="${DEMUCS_PORT_BASE:-8001}"
if ! curl -fsS --max-time 3 "http://${DEMUCS_HOST}:${DEMUCS_PORT_BASE}/health" >/dev/null 2>&1; then
    echo "[1gpu] ERROR: demucs API not reachable at ${DEMUCS_HOST}:${DEMUCS_PORT_BASE}." >&2
    echo "[1gpu]   start it with: bash script/infer/run_demucs_api.sh" >&2
    exit 1
fi
echo "[1gpu] demucs API healthy on port ${DEMUCS_PORT_BASE}"

# Run one batch (--max_new SAMPLES_PER_RUN) with retry. The python script
# resumes from the last valid line of the output file, so retries are safe.
run_one_batch() {
    local attempt=0
    while : ; do
        attempt=$((attempt + 1))
        if CUDA_VISIBLE_DEVICES=${GPU} "${PYTHON}" script/infer/run_tooldf.py \
            --engine ${ENGINE} \
            --model_path ${MODEL_PATH} \
            --dataset_path ${DATASET_PATH} \
            --output_path ${OUT_BASE} \
            --gpu_memory_utilization 0.75 \
            --max_model_len 8192 \
            --max_new ${SAMPLES_PER_RUN} \
            ${EAGER_FLAG} \
            >> "${LOG}" 2>&1 ; then
            return 0
        fi
        echo "[1gpu] batch FAILED on attempt ${attempt}" >&2
        if [[ ${attempt} -gt ${MAX_RETRIES} ]]; then
            return 1
        fi
        sleep $((30 * attempt))
    done
}

# Loop batches until no further progress. Each batch ends with python exit ->
# vLLM/MM caches and glibc arenas released to the OS, so RSS resets between
# batches instead of growing unbounded.
: >> "${LOG}"
batch=0
while : ; do
    batch=$((batch + 1))
    before=0
    [[ -f "${OUT_BASE}" ]] && before=$(wc -l < "${OUT_BASE}")

    if ! run_one_batch ; then
        echo "[1gpu] gave up on batch ${batch}; check ${LOG}" >&2
        exit 1
    fi

    after=0
    [[ -f "${OUT_BASE}" ]] && after=$(wc -l < "${OUT_BASE}")
    delta=$((after - before))
    echo "[1gpu] batch ${batch}: ${before} -> ${after} (+${delta})"

    # No progress = python exited via the `[resume] already complete` path
    # (or an unrecoverable issue we already retried). Either way: stop.
    if [[ ${delta} -le 0 ]]; then
        echo "[1gpu] done after ${batch} batch(es)"
        break
    fi
    # Less than a full batch landed = dataset tail; one more pass would just
    # re-load the model for 0 work.
    if [[ ${delta} -lt ${SAMPLES_PER_RUN} ]]; then
        echo "[1gpu] reached dataset tail after batch ${batch}"
        break
    fi
done

echo "[1gpu] done. lines=$(wc -l < ${OUT_BASE})  file=${OUT_BASE}"

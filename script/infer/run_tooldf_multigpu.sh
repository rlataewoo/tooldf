#!/usr/bin/env bash
# Multi-GPU sharded eval (data-parallel).
#
# Usage:
#   GPUS="0 1 2 3 4 5 6 7" bash script/infer/run_tooldf_multigpu.sh [SPLIT_NAME]
#
# Env knobs:
#   PYTHON        python interpreter to use (default: `python` from the active
#                 env). Activate your conda env first, or set PYTHON to its
#                 interpreter. swift must be installed in whichever env is used.
#   GPUS          space-separated CUDA device ids (default: "1", one shard)
#   ENGINE        pt | vllm  (default: vllm)
#   STAGGER_SEC   seconds to sleep between shard launches (default: 45).
#                 Spreads model-load / CUDA-graph RAM peaks so the host OOM
#                 killer doesn't pick off shards during startup.
#   MAX_RETRIES   per-shard retry attempts on failure (default: 2).
#                 Resume is automatic — the python script counts valid lines
#                 in the existing shard file and appends from there.
#   ENFORCE_EAGER 1 to pass --enforce_eager to vLLM (default: 1).
#                 Disables CUDA-graph capture: small throughput hit, eliminates
#                 a known RAM/VRAM spike during warmup.
#   SAMPLES_PER_RUN  per-shard batch size before recycling the python process
#                 (default: 200). Each batch ends with process exit, which
#                 releases vLLM prefix-cache / MM-cache / glibc arenas to the
#                 OS so RSS doesn't grow unbounded. Resume picks up the next
#                 batch automatically. Lower this if RSS still grows within a
#                 batch; raise it to amortize model-load overhead.
#
# Output:
#   - per-shard:  ${OUT_BASE%.jsonl}.shard{ID}.jsonl
#   - merged:     ${OUT_BASE}
set -euo pipefail

SPLIT="${1:-c2}"
PYTHON="${PYTHON:-python}"
GPUS="${GPUS:-1}"
ENGINE="${ENGINE:-vllm}"
STAGGER_SEC="${STAGGER_SEC:-45}"
MAX_RETRIES="${MAX_RETRIES:-2}"
ENFORCE_EAGER="${ENFORCE_EAGER:-0}"
SAMPLES_PER_RUN="${SAMPLES_PER_RUN:-400}"

# Cap per-process thread fan-out. Without this each shard spawns OMP/BLAS
# threads = CPU count, which multiplies glibc malloc arenas and pushes peak
# host RAM well past what 8 concurrent vLLM processes can survive.
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-4}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export MALLOC_ARENA_MAX="${MALLOC_ARENA_MAX:-2}"

if ! command -v "${PYTHON}" >/dev/null 2>&1; then
    echo "[multigpu] ERROR: python interpreter not found: ${PYTHON}" >&2
    echo "[multigpu]   activate your env or set PYTHON=/path/to/python" >&2
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

MODEL_PATH="output/qwen2.5-omni-3b-tooldf-sft/v0-20260427-161534/checkpoint-10047-merged"
DATASET_PATH="label/tooldf_sft/${SPLIT}_eval_sft.jsonl"
OUT_BASE="result/tooldf_sft_${SPLIT}.jsonl"
LOG_DIR="${LOG_DIR:-logs/tooldf_eval}"

GPU_ARR=($GPUS)
NUM_SHARDS=${#GPU_ARR[@]}
mkdir -p "$LOG_DIR"

EAGER_FLAG=""
if [[ "${ENFORCE_EAGER}" == "1" ]]; then
    EAGER_FLAG="--enforce_eager"
fi

echo "[multigpu] split=${SPLIT}  shards=${NUM_SHARDS}  gpus=(${GPUS})  engine=${ENGINE}"
echo "[multigpu] stagger=${STAGGER_SEC}s  retries=${MAX_RETRIES}  enforce_eager=${ENFORCE_EAGER}"
echo "[multigpu] samples_per_run=${SAMPLES_PER_RUN} (process recycle interval)"
echo "[multigpu] threads: OMP=${OMP_NUM_THREADS} MKL=${MKL_NUM_THREADS} MALLOC_ARENA_MAX=${MALLOC_ARENA_MAX}"

AASIST_HOST="${AASIST_HOST:-127.0.0.1}"
AASIST_PORTS=(8021 8022 8023 8024 8025 8026 8027 8028)
DOWN_PORTS=()
for PORT in "${AASIST_PORTS[@]}"; do
    if ! curl -fsS --max-time 3 "http://${AASIST_HOST}:${PORT}/health" >/dev/null 2>&1; then
        DOWN_PORTS+=("${PORT}")
    fi
done
if [[ ${#DOWN_PORTS[@]} -gt 0 ]]; then
    echo "[multigpu] ERROR: AASIST API not fully open on ${AASIST_HOST}." >&2
    echo "[multigpu]   unreachable ports: ${DOWN_PORTS[*]}" >&2
    echo "[multigpu]   start them with: bash script/infer/run_aasist_api.sh" >&2
    exit 1
fi
echo "[multigpu] AASIST API healthy on ports: ${AASIST_PORTS[*]}"

# Run one batch (--max_new SAMPLES_PER_RUN) with retry. The python script
# resumes from the last valid line of the shard file, so retries are safe.
run_one_batch() {
    local IDX="$1" GPU="$2" LOG="$3"
    local attempt=0
    while : ; do
        attempt=$((attempt + 1))
        if CUDA_VISIBLE_DEVICES=${GPU} "${PYTHON}" script/infer/run_tooldf.py \
            --engine ${ENGINE} \
            --model_path ${MODEL_PATH} \
            --dataset_path ${DATASET_PATH} \
            --output_path ${OUT_BASE} \
            --num_shards ${NUM_SHARDS} \
            --shard_id ${IDX} \
            --gpu_memory_utilization 0.75 \
            --max_model_len 8192 \
            --max_new ${SAMPLES_PER_RUN} \
            ${EAGER_FLAG} \
            >> "${LOG}" 2>&1 ; then
            return 0
        fi
        echo "[multigpu] shard ${IDX} (GPU ${GPU}) batch FAILED on attempt ${attempt}" >&2
        if [[ ${attempt} -gt ${MAX_RETRIES} ]]; then
            return 1
        fi
        # Back off; let neighbors free RAM/VRAM before retry.
        sleep $((30 * attempt))
    done
}

# Loop batches until the shard makes no further progress. Each batch ends
# with python exit -> vLLM/MM caches and glibc arenas released to the OS,
# so per-process RSS resets between batches instead of growing unbounded.
run_shard() {
    local IDX="$1" GPU="$2" LOG="$3"
    local SHARD_FILE="${OUT_BASE%.jsonl}.shard${IDX}.jsonl"
    local batch=0
    while : ; do
        batch=$((batch + 1))
        local before=0
        [[ -f "${SHARD_FILE}" ]] && before=$(wc -l < "${SHARD_FILE}")

        if ! run_one_batch "${IDX}" "${GPU}" "${LOG}" ; then
            echo "[multigpu] shard ${IDX} (GPU ${GPU}) gave up on batch ${batch}" >&2
            return 1
        fi

        local after=0
        [[ -f "${SHARD_FILE}" ]] && after=$(wc -l < "${SHARD_FILE}")
        local delta=$((after - before))
        echo "[multigpu] shard ${IDX} (GPU ${GPU}) batch ${batch}: ${before} -> ${after} (+${delta})"

        # No progress = python exited via the `[resume] already complete` path
        # (or an unrecoverable issue we already retried). Either way: stop.
        if [[ ${delta} -le 0 ]]; then
            echo "[multigpu] shard ${IDX} (GPU ${GPU}) done after ${batch} batch(es)"
            return 0
        fi
        # Less than a full batch landed = dataset tail; one more pass would
        # just re-load the model for 0 work.
        if [[ ${delta} -lt ${SAMPLES_PER_RUN} ]]; then
            echo "[multigpu] shard ${IDX} (GPU ${GPU}) reached dataset tail after batch ${batch}"
            return 0
        fi
    done
}

PIDS=()
SHARD_FOR_PID=()
GPU_FOR_PID=()
for IDX in "${!GPU_ARR[@]}"; do
    GPU="${GPU_ARR[$IDX]}"
    LOG="${LOG_DIR}/${SPLIT}.shard${IDX}.gpu${GPU}.log"
    echo "[multigpu]  shard ${IDX} -> GPU ${GPU}  (log: ${LOG})"
    : >> "${LOG}"
    run_shard "${IDX}" "${GPU}" "${LOG}" &
    PID=$!
    PIDS+=($PID)
    SHARD_FOR_PID[$PID]=$IDX
    GPU_FOR_PID[$PID]=$GPU
    # Stagger launches so 8 vLLM startups don't peak RAM at the same instant.
    if [[ ${IDX} -lt $((NUM_SHARDS - 1)) && ${STAGGER_SEC} -gt 0 ]]; then
        sleep "${STAGGER_SEC}"
    fi
done

# Wait for all shards; collect failures (don't abort early — let survivors finish).
FAIL=0
for PID in "${PIDS[@]}"; do
    if ! wait "${PID}"; then
        echo "[multigpu] shard ${SHARD_FOR_PID[$PID]} (GPU ${GPU_FOR_PID[$PID]}, pid=${PID}) gave up after retries" >&2
        FAIL=1
    fi
done
if [[ ${FAIL} -ne 0 ]]; then
    echo "[multigpu] one or more shards failed; check ${LOG_DIR}/" >&2
    exit 1
fi

# Merge per-shard outputs into a single file (preserves shard order).
echo "[multigpu] merging shards -> ${OUT_BASE}"
> "${OUT_BASE}"
for IDX in "${!GPU_ARR[@]}"; do
    SHARD_FILE="${OUT_BASE%.jsonl}.shard${IDX}.jsonl"
    cat "${SHARD_FILE}" >> "${OUT_BASE}"
done
echo "[multigpu] done. lines=$(wc -l < ${OUT_BASE})  file=${OUT_BASE}"

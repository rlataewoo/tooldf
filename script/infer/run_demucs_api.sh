#!/usr/bin/env bash
# Launch the Demucs source-separation FastAPI server(s).
#
# Usage:
#   bash script/infer/run_demucs_api.sh start [--port N] [--gpu N] [--num-gpus N]
#   bash script/infer/run_demucs_api.sh stop  [--port N]
#
# demucs usually lives in its own conda env; point PYTHON at that interpreter:
#   PYTHON=~/miniconda3/envs/demucs/bin/python bash script/infer/run_demucs_api.sh start --port 8005 --gpu 3
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
LOG_DIR="${LOG_DIR:-${REPO_ROOT}/logs/demucs}"
PID_DIR="${PID_DIR:-${REPO_ROOT}/logs/demucs}"
mkdir -p "${LOG_DIR}" "${PID_DIR}"

PYTHON="${PYTHON:-python}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8001}"          # first port; --num-gpus N spans PORT..PORT+N-1
MODEL="${MODEL:-htdemucs}"
NUM_GPUS="${NUM_GPUS:-1}"     # >1 -> one child server per GPU on consecutive ports
GPU_OFFSET="${GPU_OFFSET:-0}" # first GPU index

CMD="${1:-start}"
shift || true
while [[ $# -gt 0 ]]; do
    case "$1" in
        --port)     PORT="$2";       shift 2 ;;
        --gpu)      GPU_OFFSET="$2";  shift 2 ;;
        --num-gpus) NUM_GPUS="$2";    shift 2 ;;
        *) echo "unknown option: $1" >&2; exit 1 ;;
    esac
done

# Key pid/log by port so instances on different ports don't clobber each other.
PID_FILE="${PID_DIR}/demucs_${PORT}.pid"
LOG_FILE="${LOG_DIR}/demucs_${PORT}.log"

start() {
    if [[ -f "${PID_FILE}" ]] && kill -0 "$(cat "${PID_FILE}")" 2>/dev/null; then
        echo "[demucs_server] already running on port ${PORT} (pid $(cat "${PID_FILE}"))"
        return
    fi
    if (( NUM_GPUS > 1 )); then
        echo "[demucs_server] starting ${MODEL} on GPUs ${GPU_OFFSET}..$((GPU_OFFSET + NUM_GPUS - 1)), ports ${PORT}..$((PORT + NUM_GPUS - 1))"
    else
        echo "[demucs_server] starting ${MODEL} on GPU ${GPU_OFFSET}, port ${PORT}"
    fi
    nohup "${PYTHON}" "${REPO_ROOT}/script/infer/run_demucs_api.py" \
        --host       "${HOST}" \
        --port       "${PORT}" \
        --model      "${MODEL}" \
        --num-gpus   "${NUM_GPUS}" \
        --gpu-offset "${GPU_OFFSET}" \
        >"${LOG_FILE}" 2>&1 &
    echo $! >"${PID_FILE}"
    echo "[demucs_server]   pid $(cat "${PID_FILE}")  log ${LOG_FILE}"
}

stop() {
    if [[ ! -f "${PID_FILE}" ]]; then
        echo "[demucs_server] not running on port ${PORT} (no pid file)"
        return
    fi
    local pid; pid="$(cat "${PID_FILE}")"
    if kill -0 "${pid}" 2>/dev/null; then
        echo "[demucs_server] stopping port ${PORT} (pid ${pid})"
        kill "${pid}" || true
    fi
    rm -f "${PID_FILE}"
}

case "${CMD}" in
    start) start ;;
    stop)  stop ;;
    *)
        echo "usage: $0 [start|stop] [--port N] [--gpu N] [--num-gpus N]" >&2
        exit 1
        ;;
esac

#!/usr/bin/env bash
# Launch all eight XLSR-AASIST detector FastAPI servers (general + separator-aware),
# one per GPU. Each detector has its own --threshold (decision cut on p(real)).
#
# Layout:
#   General        (non-Aug):  speech/singing/music/sound  ->  GPUs 0..3, ports 8021..8024
#   SeparatorAware (Aug):      speech/singing/music/sound  ->  GPUs 4..7, ports 8025..8028
#
# Usage:
#   bash script/infer/run_aasist_api.sh        # start
#   bash script/infer/run_aasist_api.sh stop   # kill all
#
# Override anything via env vars before invoking, e.g.:
#   XLSR_PATH=/path/to/xlsr2_300m.pt \
#   SPEECH_GEN_CKPT=/path/to/speech.pth      SPEECH_GEN_THRESHOLD=0.5  \
#   SPEECH_SA_CKPT=/path/to/speech_aug.pth   SPEECH_SA_THRESHOLD=0.55  \
#   bash script/infer/run_aasist_api.sh
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
LOG_DIR="${LOG_DIR:-${REPO_ROOT}/logs/aasist_detectors}"
PID_DIR="${PID_DIR:-${REPO_ROOT}/logs/aasist_detectors}"
mkdir -p "${LOG_DIR}" "${PID_DIR}"

# Shared XLSR-2 300M base checkpoint (used by all servers).
XLSR_PATH="${XLSR_PATH:-pretrained/xlsr2_300m.pt}"

# ---- General (non-Aug) detector checkpoints ----
SPEECH_GEN_CKPT="${SPEECH_GEN_CKPT:-pretrained/XLSR_AASIST_Speech_VL/best_model.pth}"
SINGING_GEN_CKPT="${SINGING_GEN_CKPT:-pretrained/XLSR_AASIST_Singing_VL/best_model.pth}"
MUSIC_GEN_CKPT="${MUSIC_GEN_CKPT:-pretrained/XLSR_AASIST_Music_VL/best_model.pth}"
SOUND_GEN_CKPT="${SOUND_GEN_CKPT:-pretrained/XLSR_AASIST_Sound_VL/best_model.pth}"

# ---- Separator-Aware (Aug) detector checkpoints ----
SPEECH_SA_CKPT="${SPEECH_SA_CKPT:-pretrained/XLSR_AASIST_Speech_VL_Aug/best_model.pth}"
SINGING_SA_CKPT="${SINGING_SA_CKPT:-pretrained/XLSR_AASIST_Singing_VL_Aug/best_model.pth}"
MUSIC_SA_CKPT="${MUSIC_SA_CKPT:-pretrained/XLSR_AASIST_Music_VL_Aug/best_model.pth}"
SOUND_SA_CKPT="${SOUND_SA_CKPT:-pretrained/XLSR_AASIST_Sound_VL_Aug/best_model.pth}"

# ---- Thresholds (TODO: fill in) ----
SPEECH_GEN_THRESHOLD="${SPEECH_GEN_THRESHOLD:-0.9825}"
SINGING_GEN_THRESHOLD="${SINGING_GEN_THRESHOLD:-0.1500}"
MUSIC_GEN_THRESHOLD="${MUSIC_GEN_THRESHOLD:-0.0300}"
SOUND_GEN_THRESHOLD="${SOUND_GEN_THRESHOLD:-0.3479}"

SPEECH_SA_THRESHOLD="${SPEECH_SA_THRESHOLD:-0.0507}"
SINGING_SA_THRESHOLD="${SINGING_SA_THRESHOLD:-0.3296}"
MUSIC_SA_THRESHOLD="${MUSIC_SA_THRESHOLD:-0.2350}"
SOUND_SA_THRESHOLD="${SOUND_SA_THRESHOLD:-0.3781}"

# ---- Ports ----
SPEECH_GEN_PORT="${SPEECH_GEN_PORT:-8021}"
SINGING_GEN_PORT="${SINGING_GEN_PORT:-8022}"
MUSIC_GEN_PORT="${MUSIC_GEN_PORT:-8023}"
SOUND_GEN_PORT="${SOUND_GEN_PORT:-8024}"

SPEECH_SA_PORT="${SPEECH_SA_PORT:-8025}"
SINGING_SA_PORT="${SINGING_SA_PORT:-8026}"
MUSIC_SA_PORT="${MUSIC_SA_PORT:-8027}"
SOUND_SA_PORT="${SOUND_SA_PORT:-8028}"

# ---- GPUs ----
SPEECH_GEN_GPU="${SPEECH_GEN_GPU:-0}"
SINGING_GEN_GPU="${SINGING_GEN_GPU:-1}"
MUSIC_GEN_GPU="${MUSIC_GEN_GPU:-2}"
SOUND_GEN_GPU="${SOUND_GEN_GPU:-3}"

SPEECH_SA_GPU="${SPEECH_SA_GPU:-4}"
SINGING_SA_GPU="${SINGING_SA_GPU:-5}"
MUSIC_SA_GPU="${MUSIC_SA_GPU:-6}"
SOUND_SA_GPU="${SOUND_SA_GPU:-7}"

ALL_TAGS=(speech_gen singing_gen music_gen sound_gen speech_sa singing_sa music_sa sound_sa)

start_one() {
    local tag="$1" detector="$2" gpu="$3" port="$4" ckpt="$5" threshold="$6"
    local pid_file="${PID_DIR}/${tag}.pid"
    local log_file="${LOG_DIR}/${tag}.log"
    if [[ -z "${threshold}" ]]; then
        echo "[run_aasist_detector_servers] SKIP ${tag}: empty threshold (set it at the top of this script)"
        return
    fi
    if [[ -f "${pid_file}" ]] && kill -0 "$(cat "${pid_file}")" 2>/dev/null; then
        echo "[run_aasist_detector_servers] ${tag} already running (pid $(cat "${pid_file}"))"
        return
    fi
    echo "[run_aasist_detector_servers] starting ${tag} (${detector}) on GPU ${gpu}, port ${port}, threshold ${threshold}"
    CUDA_VISIBLE_DEVICES="${gpu}" nohup \
        python "${REPO_ROOT}/script/infer/run_aasist_api.py" \
            --detector_type "${detector}" \
            --xlsr_path     "${XLSR_PATH}" \
            --ckpt_path     "${ckpt}" \
            --threshold     "${threshold}" \
            --port          "${port}" \
        >"${log_file}" 2>&1 &
    echo $! >"${pid_file}"
    echo "[run_aasist_detector_servers]   pid $(cat "${pid_file}")  log ${log_file}"
}

stop_all() {
    for tag in "${ALL_TAGS[@]}"; do
        local pid_file="${PID_DIR}/${tag}.pid"
        if [[ -f "${pid_file}" ]]; then
            local pid; pid="$(cat "${pid_file}")"
            if kill -0 "${pid}" 2>/dev/null; then
                echo "[run_aasist_detector_servers] stopping ${tag} (pid ${pid})"
                kill "${pid}" || true
            fi
            rm -f "${pid_file}"
        fi
    done
}

case "${1:-start}" in
    start)
        # General (non-Aug)
        start_one speech_gen  speech  "${SPEECH_GEN_GPU}"  "${SPEECH_GEN_PORT}"  "${SPEECH_GEN_CKPT}"  "${SPEECH_GEN_THRESHOLD}"
        start_one singing_gen singing "${SINGING_GEN_GPU}" "${SINGING_GEN_PORT}" "${SINGING_GEN_CKPT}" "${SINGING_GEN_THRESHOLD}"
        start_one music_gen   music   "${MUSIC_GEN_GPU}"   "${MUSIC_GEN_PORT}"   "${MUSIC_GEN_CKPT}"   "${MUSIC_GEN_THRESHOLD}"
        start_one sound_gen   sound   "${SOUND_GEN_GPU}"   "${SOUND_GEN_PORT}"   "${SOUND_GEN_CKPT}"   "${SOUND_GEN_THRESHOLD}"
        # Separator-Aware (Aug)
        start_one speech_sa   speech  "${SPEECH_SA_GPU}"   "${SPEECH_SA_PORT}"   "${SPEECH_SA_CKPT}"   "${SPEECH_SA_THRESHOLD}"
        start_one singing_sa  singing "${SINGING_SA_GPU}"  "${SINGING_SA_PORT}"  "${SINGING_SA_CKPT}"  "${SINGING_SA_THRESHOLD}"
        start_one music_sa    music   "${MUSIC_SA_GPU}"    "${MUSIC_SA_PORT}"    "${MUSIC_SA_CKPT}"    "${MUSIC_SA_THRESHOLD}"
        start_one sound_sa    sound   "${SOUND_SA_GPU}"    "${SOUND_SA_PORT}"    "${SOUND_SA_CKPT}"    "${SOUND_SA_THRESHOLD}"
        echo "[run_aasist_detector_servers] all launched. tail -f ${LOG_DIR}/*.log"
        ;;
    stop)
        stop_all
        ;;
    *)
        echo "usage: $0 [start|stop]" >&2
        exit 1
        ;;
esac

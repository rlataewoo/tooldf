#!/usr/bin/env bash
# Run the composition and ToolDF-SFT stages (2 -> 3) over every split and category.
#
# Prerequisite: label/audio_caption/ must already be populated, either by running
# stage 1 (script/data/run_audiocaption.sh, needs a captioning API) or by
# downloading the caption files. Stage 1 is not driven from here.
#
# Usage:
#   bash script/data/run.sh                       # all stages, all splits
#   bash script/data/run.sh --stages 3            # Tool-SFT formatting only
#   bash script/data/run.sh --splits dev          # one split only
#   bash script/data/run.sh --categories speech,music
#   bash script/data/run.sh --dry-run             # print commands, run nothing
#
# Run from the project root.

set -euo pipefail

STAGES="2,3"
SPLITS="train,dev,eval"
CATEGORIES="speech,singing,music,sound"
DRY_RUN=0

while [ $# -gt 0 ]; do
    case "$1" in
        --stages)     STAGES="$2"; shift 2 ;;
        --splits)     SPLITS="$2"; shift 2 ;;
        --categories) CATEGORIES="$2"; shift 2 ;;
        --dry-run)    DRY_RUN=1; shift ;;
        -h|--help)    sed -n '2,15p' "$0"; exit 0 ;;
        *) echo "unknown option: $1" >&2; exit 1 ;;
    esac
done

HERE="$(dirname "$0")"
IFS=',' read -r -a STAGE_LIST <<< "${STAGES}"
IFS=',' read -r -a SPLIT_LIST <<< "${SPLITS}"
IFS=',' read -r -a CATEGORY_LIST <<< "${CATEGORIES}"

has_stage() {
    local want="$1" s
    for s in "${STAGE_LIST[@]}"; do [ "$s" = "$want" ] && return 0; done
    return 1
}

run() {
    echo "+ $*"
    [ "${DRY_RUN}" -eq 1 ] || "$@"
}

for SPLIT in "${SPLIT_LIST[@]}"; do
    echo "=============== split: ${SPLIT} ==============="

    # Stage 2: compose C1/C2/C3. Needs all four categories captioned.
    if has_stage 2; then
        for C in c1 c2 c3; do
            run bash "${HERE}/run_compose_${C}.sh" "${SPLIT}"
        done
    fi

    # Stage 3: format Tool-SFT samples for the public datasets and for C1/C2/C3.
    if has_stage 3; then
        for CATEGORY in "${CATEGORY_LIST[@]}"; do
            run bash "${HERE}/run_prepare_sft_single.sh" "${SPLIT}" "${CATEGORY}"
        done
        for C in c1 c2 c3; do
            run bash "${HERE}/run_prepare_sft_${C}.sh" "${SPLIT}"
        done
    fi
done

echo "=============== done ==============="

# category: speech | singing | music | sound
# split: train | dev | eval

SPLIT="${1:-eval}"
CATEGORY="${2:-speech}"

# VAD adds voice_start_time / voice_end_time; only the vocal categories need it.
case "${CATEGORY}" in
    speech)  DATASET=19la;          VAD="--vad" ;;
    singing) DATASET=ctrsvdd;       VAD="--vad" ;;
    music)   DATASET=fakemusiccaps; VAD="" ;;
    sound)   DATASET=ESDD;          VAD="" ;;
    *) echo "unknown category: ${CATEGORY} (speech | singing | music | sound)" >&2; exit 1 ;;
esac

python preprocess/stage1/generate_audio_caption.py \
    --input  label/binary_sft/${DATASET}_${SPLIT}_sft.jsonl \
    --output label/audio_caption/${DATASET}_${SPLIT}.jsonl \
    --category ${CATEGORY} \
    --api-url http://10.10.0.58:8901/v1/chat/completions \
    --temperature 0.3 \
    --workers 4 \
    ${VAD}

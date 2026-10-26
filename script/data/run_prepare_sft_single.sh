# category: speech | singing | music | sound
# split: train | dev | eval

SPLIT="${1:-eval}"
CATEGORY="${2:-speech}"

case "${CATEGORY}" in
    speech)  DATASET=19la ;;
    singing) DATASET=ctrsvdd ;;
    music)   DATASET=fakemusiccaps ;;
    sound)   DATASET=ESDD ;;
    *) echo "unknown category: ${CATEGORY} (speech | singing | music | sound)" >&2; exit 1 ;;
esac

python preprocess/stage3/prepare_tooldf_sft_single.py \
    --category ${CATEGORY} \
    --input    label/audio_caption/${DATASET}_${SPLIT}.jsonl \
    --output   label/tooldf_sft/${DATASET}_${SPLIT}_sft.jsonl \
    --seed     42

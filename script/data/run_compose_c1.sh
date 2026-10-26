# split: train | dev | eval

SPLIT="${1:-eval}"

python preprocess/stage2/compose_c1.py \
    --speech-input    label/audio_caption/19la_${SPLIT}.jsonl \
    --singing-input   label/audio_caption/ctrsvdd_${SPLIT}.jsonl \
    --output-jsonl    label/audio_caption/c1_${SPLIT}.jsonl \
    --output-audio-dir dataset/MAADD/c1/${SPLIT} \
    --seed            42

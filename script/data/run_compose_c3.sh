# split: train | dev | eval

SPLIT="${1:-eval}"

python preprocess/stage2/compose_c3.py \
    --speech-input    label/audio_caption/19la_${SPLIT}.jsonl \
    --singing-input   label/audio_caption/ctrsvdd_${SPLIT}.jsonl \
    --bg-music-input  label/audio_caption/fakemusiccaps_${SPLIT}.jsonl \
    --bg-sound-input  label/audio_caption/ESDD_${SPLIT}.jsonl \
    --output-jsonl    label/audio_caption/c3_${SPLIT}.jsonl \
    --output-audio-dir dataset/MAADD/c3/${SPLIT} \
    --target-sr       16000 \
    --snr-min         0.0 \
    --snr-max         15.0 \
    --seed            42

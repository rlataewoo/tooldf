SPLIT="${1:-eval}"

python preprocess/stage3/prepare_tooldf_sft_c3.py \
    --input  label/audio_caption/c3_${SPLIT}.jsonl \
    --output label/tooldf_sft/c3_${SPLIT}_sft.jsonl \
    --seed   42

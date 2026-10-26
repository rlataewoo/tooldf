SPLIT="${1:-eval}"

python preprocess/stage3/prepare_tooldf_sft_c2.py \
    --input  label/audio_caption/c2_${SPLIT}.jsonl \
    --output label/tooldf_sft/c2_${SPLIT}_sft.jsonl \
    --seed   42

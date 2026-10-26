SPLIT="${1:-eval}"

python preprocess/stage3/prepare_tooldf_sft_c1.py \
    --input  label/audio_caption/c1_${SPLIT}.jsonl \
    --output label/tooldf_sft/c1_${SPLIT}_sft.jsonl \
    --seed   42

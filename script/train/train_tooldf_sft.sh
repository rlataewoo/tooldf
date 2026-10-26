MASTER_PORT=29200 NPROC_PER_NODE=2 CUDA_VISIBLE_DEVICES=0,1 swift sft \
        --model /mnt/Qwen2.5-Omni-3B \
        --model_kwargs '{"enable_audio_output": false}' \
        --output_dir output/qwen2.5-omni-3b-tooldf-sft \
        --dataset label/tooldf_sft/19la_train_sft.jsonl \
                  label/tooldf_sft/ctrsvdd_train_sft.jsonl \
                  label/tooldf_sft/fakemusiccaps_train_sft.jsonl \
                  label/tooldf_sft/ESDD_train_sft.jsonl \
                  label/tooldf_sft/c1_train_sft.jsonl \
                  label/tooldf_sft/c2_train_sft.jsonl \
                  label/tooldf_sft/c3_train_sft.jsonl \
        --val_dataset label/tooldf_sft/19la_dev_sft.jsonl \
                      label/tooldf_sft/ctrsvdd_dev_sft.jsonl \
                      label/tooldf_sft/fakemusiccaps_dev_sft.jsonl \
                      label/tooldf_sft/ESDD_dev_sft.jsonl \
                      label/tooldf_sft/c1_dev_sft.jsonl \
                      label/tooldf_sft/c2_dev_sft.jsonl \
                      label/tooldf_sft/c3_dev_sft.jsonl \
        --system prompt/ToolDF.txt \
        --agent_template hermes \
        --gradient_checkpointing_kwargs '{"use_reentrant": false}' \
        --torch_dtype bfloat16 \
        --freeze_llm false \
        --freeze_aligner false \
        --freeze_vit false \
        --max_length 4096 \
        --dataloader_num_workers 8 \
        --target_modules all-linear \
        --save_strategy epoch \
        --logging_steps 100 \
        --per_device_train_batch_size 4 \
        --per_device_eval_batch_size 4 \
        --gradient_accumulation_steps 4 \
        --save_total_limit 10 \
        --lazy_tokenize true \
        --weight_decay 0.1 \
        --learning_rate 1e-5 \
        --tuner_type lora \
        --warmup_ratio 0.01 \
        --num_train_epochs 3 \
        --lora_rank 64 \
        --lora_alpha 16 \
        --deepspeed zero2

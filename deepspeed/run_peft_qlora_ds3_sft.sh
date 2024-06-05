accelerate launch --config_file "configs/deepspeed_config_z3_qlora.yaml"  train-sft.py \
--seed 100 \
--model_name_or_path "/data/models/Llama-3-Ko-8B-dare-ties" \
--add_special_tokens False \
--append_concat_token False \
--max_seq_len 2048 \
--num_train_epochs 1 \
--logging_steps 100 \
--save_steps 100 \
--log_level "info" \
--logging_strategy "steps" \
--save_strategy "steps" \
--bf16 True \
--packing True \
--learning_rate 2e-8 \
--lr_scheduler_type "cosine" \
--weight_decay 1e-4 \
--warmup_ratio 0.0 \
--max_grad_norm 1.0 \
--output_dir "llama3-Ko-ASKBIZ-130K-ds3-dare-ties" \
--per_device_train_batch_size 2 \
--gradient_accumulation_steps 8 \
--gradient_checkpointing True \
--use_reentrant True \
--use_flash_attn True \
--use_peft_lora True \
--lora_r 16 \
--lora_alpha 16 \
--lora_dropout 0.01 \
--lora_target_modules "all-linear" \
--use_4bit_quantization True \
--use_nested_quant True \
--bnb_4bit_compute_dtype "bfloat16" \
--bnb_4bit_quant_storage_dtype "bfloat16"
#!/bin/bash
# QQP - RoBERTa Base with GP-LoRA (Gauge-Projected LoRA)
export num_gpus=8
export CUBLAS_WORKSPACE_CONFIG=":16:8"
export PYTHONHASHSEED=0
export output_dir="./qqp_gplora"
python -m torch.distributed.launch --nproc_per_node=$num_gpus \
examples/text-classification/run_glue.py \
--model_name_or_path roberta-base \
--task_name qqp \
--do_train \
--do_eval \
--max_seq_length 512 \
--per_device_train_batch_size 16 \
--learning_rate 5e-4 \
--num_train_epochs 30 \
--output_dir $output_dir/model \
--overwrite_output_dir \
--logging_steps 10 \
--logging_dir $output_dir/log \
--evaluation_strategy epoch \
--save_strategy epoch \
--warmup_ratio 0.06 \
--apply_lora \
--lora_r 8 \
--lora_alpha 16 \
--seed 0 \
--weight_decay 0.1 \
--gp_lora \
--gp_mu auto \
--gp_eps 1e-4


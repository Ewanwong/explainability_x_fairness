#!/usr/bin/env bash

python -m model_training.conventional_finetuning \
    --model_name_or_path "distilbert/distilbert-base-uncased" \
    --dataset_name="lighteval/civil_comments_helm" \
    --num_labels=2 \
    --output_dir "/scratch/yifwang/new_fairness_x_explainability/models/baseline_distilbert_civil" \
    --batch_size=32 \
    --max_seq_length=512 \
    --learning_rate=2e-5 \
    --warmup_steps_or_ratio=0.1 \
    --num_train_epochs=3 \
    --early_stopping_patience=-1 \
    --eval_steps=1000 \
    --save_steps=1000 \
    --split_ratio="8, 2" \
    --seed=42 \
    --eval_metric="accuracy" \

python -m model_training.conventional_finetuning \
    --model_name_or_path "bert-base-uncased" \
    --dataset_name="lighteval/civil_comments_helm" \
    --num_labels=2 \
    --output_dir "/scratch/yifwang/new_fairness_x_explainability/models/baseline_bert_civil" \
    --batch_size=32 \
    --max_seq_length=512 \
    --learning_rate=2e-5 \
    --warmup_steps_or_ratio=0.1 \
    --num_train_epochs=3 \
    --early_stopping_patience=-1 \
    --eval_steps=1000 \
    --save_steps=1000 \
    --split_ratio="8, 2" \
    --seed=42 \
    --eval_metric="accuracy" \

python -m model_training.conventional_finetuning \
    --model_name_or_path "FacebookAI/roberta-base" \
    --dataset_name="lighteval/civil_comments_helm" \
    --num_labels=2 \
    --output_dir "/scratch/yifwang/new_fairness_x_explainability/models/baseline_roberta_civil" \
    --batch_size=32 \
    --max_seq_length=512 \
    --learning_rate=2e-5 \
    --warmup_steps_or_ratio=0.1 \
    --num_train_epochs=3 \
    --early_stopping_patience=-1 \
    --eval_steps=1000 \
    --save_steps=1000 \
    --split_ratio="8, 2" \
    --seed=42 \
    --eval_metric="accuracy" \
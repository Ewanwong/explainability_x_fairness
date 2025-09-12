#!/usr/bin/env bash

python -m model_selection.debiasing_preprocessing \
    --model_name_or_path "bert-base-uncased" \
    --dataset_name="lighteval/civil_comments_helm" \
    --num_labels=2 \
    --output_dir "/scratch/yifwang/new_fairness_x_explainability/new_debiased_models/bert_civil_religion" \
    --batch_size=8 \
    --max_seq_length=512 \
    --learning_rate=2e-5 \
    --warmup_steps_or_ratio=0.1 \
    --num_train_epochs=5 \
    --early_stopping_patience=-1 \
    --eval_steps=1000 \
    --save_steps=1000 \
    --split_ratio="8, 2" \
    --seed=42 \
    --eval_metric="accuracy" \
    --bias_type='religion' \
    --debiasing_method='group_balance' \
    --total_num_examples=6300\

python -m model_selection.debiasing_preprocessing \
    --model_name_or_path "bert-base-uncased" \
    --dataset_name="lighteval/civil_comments_helm" \
    --num_labels=2 \
    --output_dir "/scratch/yifwang/new_fairness_x_explainability/new_debiased_models/bert_civil_religion" \
    --batch_size=8 \
    --max_seq_length=512 \
    --learning_rate=2e-5 \
    --warmup_steps_or_ratio=0.1 \
    --num_train_epochs=5 \
    --early_stopping_patience=-1 \
    --eval_steps=1000 \
    --save_steps=1000 \
    --split_ratio="8, 2" \
    --seed=42 \
    --eval_metric="accuracy" \
    --bias_type='religion' \
    --debiasing_method='group_class_balance' \
    --total_num_examples=6300 \

python -m model_selection.debiasing_preprocessing \
    --model_name_or_path "bert-base-uncased" \
    --dataset_name="lighteval/civil_comments_helm" \
    --num_labels=2 \
    --output_dir "/scratch/yifwang/new_fairness_x_explainability/new_debiased_models/bert_civil_religion" \
    --batch_size=8 \
    --max_seq_length=512 \
    --learning_rate=2e-5 \
    --warmup_steps_or_ratio=0.1 \
    --num_train_epochs=5 \
    --early_stopping_patience=-1 \
    --eval_steps=1000 \
    --save_steps=1000 \
    --split_ratio="8, 2" \
    --seed=42 \
    --eval_metric="accuracy" \
    --bias_type='religion' \
    --debiasing_method='cda' \
    --total_num_examples=6300 \

python -m model_selection.debiasing_preprocessing \
    --model_name_or_path "bert-base-uncased" \
    --dataset_name="lighteval/civil_comments_helm" \
    --num_labels=2 \
    --output_dir "/scratch/yifwang/new_fairness_x_explainability/new_debiased_models/bert_civil_religion" \
    --batch_size=8 \
    --max_seq_length=512 \
    --learning_rate=2e-5 \
    --warmup_steps_or_ratio=0.1 \
    --num_train_epochs=5 \
    --early_stopping_patience=-1 \
    --eval_steps=1000 \
    --save_steps=1000 \
    --split_ratio="8, 2" \
    --seed=42 \
    --eval_metric="accuracy" \
    --bias_type='religion' \
    --debiasing_method='no_debiasing' \
    --total_num_examples=6300 \



python -m model_selection.debiasing_preprocessing \
    --model_name_or_path "FacebookAI/roberta-base" \
    --dataset_name="lighteval/civil_comments_helm" \
    --num_labels=2 \
    --output_dir "/scratch/yifwang/new_fairness_x_explainability/new_debiased_models/roberta_civil_religion" \
    --batch_size=8 \
    --max_seq_length=512 \
    --learning_rate=2e-5 \
    --warmup_steps_or_ratio=0.1 \
    --num_train_epochs=5 \
    --early_stopping_patience=-1 \
    --eval_steps=1000 \
    --save_steps=1000 \
    --split_ratio="8, 2" \
    --seed=42 \
    --eval_metric="accuracy" \
    --bias_type='religion' \
    --debiasing_method='group_balance' \
    --total_num_examples=6300\

python -m model_selection.debiasing_preprocessing \
    --model_name_or_path "FacebookAI/roberta-base" \
    --dataset_name="lighteval/civil_comments_helm" \
    --num_labels=2 \
    --output_dir "/scratch/yifwang/new_fairness_x_explainability/new_debiased_models/roberta_civil_religion" \
    --batch_size=8 \
    --max_seq_length=512 \
    --learning_rate=2e-5 \
    --warmup_steps_or_ratio=0.1 \
    --num_train_epochs=5 \
    --early_stopping_patience=-1 \
    --eval_steps=1000 \
    --save_steps=1000 \
    --split_ratio="8, 2" \
    --seed=42 \
    --eval_metric="accuracy" \
    --bias_type='religion' \
    --debiasing_method='group_class_balance' \
    --total_num_examples=6300 \

python -m model_selection.debiasing_preprocessing \
    --model_name_or_path "FacebookAI/roberta-base" \
    --dataset_name="lighteval/civil_comments_helm" \
    --num_labels=2 \
    --output_dir "/scratch/yifwang/new_fairness_x_explainability/new_debiased_models/roberta_civil_religion" \
    --batch_size=8 \
    --max_seq_length=512 \
    --learning_rate=2e-5 \
    --warmup_steps_or_ratio=0.1 \
    --num_train_epochs=5 \
    --early_stopping_patience=-1 \
    --eval_steps=1000 \
    --save_steps=1000 \
    --split_ratio="8, 2" \
    --seed=42 \
    --eval_metric="accuracy" \
    --bias_type='religion' \
    --debiasing_method='cda' \
    --total_num_examples=6300 \

python -m model_selection.debiasing_preprocessing \
    --model_name_or_path "FacebookAI/roberta-base" \
    --dataset_name="lighteval/civil_comments_helm" \
    --num_labels=2 \
    --output_dir "/scratch/yifwang/new_fairness_x_explainability/new_debiased_models/roberta_civil_religion" \
    --batch_size=8 \
    --max_seq_length=512 \
    --learning_rate=2e-5 \
    --warmup_steps_or_ratio=0.1 \
    --num_train_epochs=5 \
    --early_stopping_patience=-1 \
    --eval_steps=1000 \
    --save_steps=1000 \
    --split_ratio="8, 2" \
    --seed=42 \
    --eval_metric="accuracy" \
    --bias_type='religion' \
    --debiasing_method='no_debiasing' \
    --total_num_examples=6300 \

python -m model_selection.debiasing_preprocessing \
    --model_name_or_path "distilbert/distilbert-base-uncased" \
    --dataset_name="lighteval/civil_comments_helm" \
    --num_labels=2 \
    --output_dir "/scratch/yifwang/new_fairness_x_explainability/new_debiased_models/distilbert_civil_religion" \
    --batch_size=8 \
    --max_seq_length=512 \
    --learning_rate=2e-5 \
    --warmup_steps_or_ratio=0.1 \
    --num_train_epochs=5 \
    --early_stopping_patience=-1 \
    --eval_steps=1000 \
    --save_steps=1000 \
    --split_ratio="8, 2" \
    --seed=42 \
    --eval_metric="accuracy" \
    --bias_type='religion' \
    --debiasing_method='group_balance' \
    --total_num_examples=6300\

python -m model_selection.debiasing_preprocessing \
    --model_name_or_path "distilbert/distilbert-base-uncased" \
    --dataset_name="lighteval/civil_comments_helm" \
    --num_labels=2 \
    --output_dir "/scratch/yifwang/new_fairness_x_explainability/new_debiased_models/distilbert_civil_religion" \
    --batch_size=8 \
    --max_seq_length=512 \
    --learning_rate=2e-5 \
    --warmup_steps_or_ratio=0.1 \
    --num_train_epochs=5 \
    --early_stopping_patience=-1 \
    --eval_steps=1000 \
    --save_steps=1000 \
    --split_ratio="8, 2" \
    --seed=42 \
    --eval_metric="accuracy" \
    --bias_type='religion' \
    --debiasing_method='group_class_balance' \
    --total_num_examples=6300 \

python -m model_selection.debiasing_preprocessing \
    --model_name_or_path "distilbert/distilbert-base-uncased" \
    --dataset_name="lighteval/civil_comments_helm" \
    --num_labels=2 \
    --output_dir "/scratch/yifwang/new_fairness_x_explainability/new_debiased_models/distilbert_civil_religion" \
    --batch_size=8 \
    --max_seq_length=512 \
    --learning_rate=2e-5 \
    --warmup_steps_or_ratio=0.1 \
    --num_train_epochs=5 \
    --early_stopping_patience=-1 \
    --eval_steps=1000 \
    --save_steps=1000 \
    --split_ratio="8, 2" \
    --seed=42 \
    --eval_metric="accuracy" \
    --bias_type='religion' \
    --debiasing_method='cda' \
    --total_num_examples=6300 \

python -m model_selection.debiasing_preprocessing \
    --model_name_or_path "distilbert/distilbert-base-uncased" \
    --dataset_name="lighteval/civil_comments_helm" \
    --num_labels=2 \
    --output_dir "/scratch/yifwang/new_fairness_x_explainability/new_debiased_models/distilbert_civil_religion" \
    --batch_size=8 \
    --max_seq_length=512 \
    --learning_rate=2e-5 \
    --warmup_steps_or_ratio=0.1 \
    --num_train_epochs=5 \
    --early_stopping_patience=-1 \
    --eval_steps=1000 \
    --save_steps=1000 \
    --split_ratio="8, 2" \
    --seed=42 \
    --eval_metric="accuracy" \
    --bias_type='religion' \
    --debiasing_method='no_debiasing' \
    --total_num_examples=6300 \

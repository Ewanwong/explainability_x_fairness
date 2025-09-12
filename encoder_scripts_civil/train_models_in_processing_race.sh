#!/bin/bash
python -m model_selection.debiasing_inprocessing \
    --model_name_or_path="bert-base-uncased" \
    --dataset_name="lighteval/civil_comments_helm" \
    --num_labels=2 \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models/bert_civil_race" \
    --batch_size=8 \
    --max_seq_length=512 \
    --learning_rate=2e-05 \
    --warmup_steps_or_ratio=0.1 \
    --num_train_epochs=5 \
    --early_stopping_patience=-1 \
    --eval_steps=1000 \
    --save_steps=1000 \
    --split_ratio="8, 2" \
    --seed=42 \
    --eval_metric="accuracy" \
    --bias_type="race" \
    --debiasing_method="causal_debias" \
    --total_num_examples=8000\
    --entropy_weight=0.1 \
    --causal_debias_weight=0.5 

python -m model_selection.debiasing_inprocessing \
    --model_name_or_path="FacebookAI/roberta-base" \
    --dataset_name="lighteval/civil_comments_helm" \
    --num_labels=2 \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models/roberta_civil_race" \
    --batch_size=8 \
    --max_seq_length=512 \
    --learning_rate=2e-05 \
    --warmup_steps_or_ratio=0.1 \
    --num_train_epochs=5 \
    --early_stopping_patience=-1 \
    --eval_steps=1000 \
    --save_steps=1000 \
    --split_ratio="8, 2" \
    --seed=42 \
    --eval_metric="accuracy" \
    --bias_type="race" \
    --debiasing_method="causal_debias" \
    --total_num_examples=8000\
    --entropy_weight=0.1 \
    --causal_debias_weight=0.5 

python -m model_selection.debiasing_inprocessing \
    --model_name_or_path="distilbert/distilbert-base-uncased" \
    --dataset_name="lighteval/civil_comments_helm" \
    --num_labels=2 \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models/distilbert_civil_race" \
    --batch_size=8 \
    --max_seq_length=512 \
    --learning_rate=2e-05 \
    --warmup_steps_or_ratio=0.1 \
    --num_train_epochs=5 \
    --early_stopping_patience=-1 \
    --eval_steps=1000 \
    --save_steps=1000 \
    --split_ratio="8, 2" \
    --seed=42 \
    --eval_metric="accuracy" \
    --bias_type="race" \
    --debiasing_method="causal_debias" \
    --total_num_examples=8000\
    --entropy_weight=0.1 \
    --causal_debias_weight=0.5 


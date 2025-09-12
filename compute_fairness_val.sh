#!/usr/bin/env bash


python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/models/baseline_bert_civil" \
    --batch_size=32 \
    --max_seq_length=512 \
    --num_examples=500 \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/results/baseline_bert_civil_val_500_gender" \
    --bias_type="gender" \

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/models/baseline_roberta_civil" \
    --batch_size=32 \
    --max_seq_length=512 \
    --num_examples=500 \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/results/baseline_roberta_civil_val_500_gender" \
    --bias_type="gender" \

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/models/baseline_distilbert_civil" \
    --batch_size=32 \
    --max_seq_length=512 \
    --num_examples=500 \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/results/baseline_distilbert_civil_val_500_gender" \
    --bias_type="gender" \

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/models/baseline_bert_civil" \
    --batch_size=32 \
    --max_seq_length=512 \
    --num_examples=500 \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/results/baseline_bert_civil_val_500_race" \
    --bias_type="race" \

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/models/baseline_roberta_civil" \
    --batch_size=32 \
    --max_seq_length=512 \
    --num_examples=500 \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/results/baseline_roberta_civil_val_500_race" \
    --bias_type="race" \

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/models/baseline_distilbert_civil" \
    --batch_size=32 \
    --max_seq_length=512 \
    --num_examples=500 \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/results/baseline_distilbert_civil_val_500_race" \
    --bias_type="race" \

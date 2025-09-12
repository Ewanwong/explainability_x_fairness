#!/usr/bin/env bash

# python -m fairness_evaluation.compute_fairness_results \
#     --dataset_name="lighteval/civil_comments_helm" \
#     --split="test" \
#     --split_ratio="1, 1" \
#     --model_dir="meta-llama/Llama-3.1-8B-Instruct" \
#     --num_examples=2000 \
#     --output_dir="/scratch/yifwang/new_fairness_x_explainability/results/llama_8b_civil_2000_gender" \
#     --bias_type="gender" \

# python -m fairness_evaluation.compute_fairness_results \
#     --dataset_name="lighteval/civil_comments_helm" \
#     --split="test" \
#     --split_ratio="1, 1" \
#     --model_dir="Qwen/Qwen2.5-7B-Instruct" \
#     --num_examples=2000 \
#     --output_dir="/scratch/yifwang/new_fairness_x_explainability/results/qwen_7b_civil_2000_gender" \
#     --bias_type="gender" \

# python -m fairness_evaluation.compute_fairness_results \
#     --dataset_name="lighteval/civil_comments_helm" \
#     --split="test" \
#     --split_ratio="1, 1" \
#     --model_dir="meta-llama/Llama-3.1-8B-Instruct" \
#     --num_examples=2000 \
#     --output_dir="/scratch/yifwang/new_fairness_x_explainability/results/llama_8b_civil_2000_race" \
#     --bias_type="race" \

# python -m fairness_evaluation.compute_fairness_results \
#     --dataset_name="lighteval/civil_comments_helm" \
#     --split="test" \
#     --split_ratio="1, 1" \
#     --model_dir="Qwen/Qwen2.5-7B-Instruct" \
#     --num_examples=2000 \
#     --output_dir="/scratch/yifwang/new_fairness_x_explainability/results/qwen_7b_civil_2000_race" \
#     --bias_type="race" \

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="test" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/models/baseline_bert_civil" \
    --batch_size=8 \
    --max_seq_length=512 \
    --num_examples=2000 \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/results/baseline_bert_civil_2000_gender" \
    --bias_type="gender" \

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="test" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/models/baseline_roberta_civil" \
    --batch_size=8 \
    --max_seq_length=512 \
    --num_examples=2000 \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/results/baseline_roberta_civil_2000_gender" \
    --bias_type="gender" \

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="test" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/models/baseline_distilbert_civil" \
    --batch_size=8 \
    --max_seq_length=512 \
    --num_examples=2000 \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/results/baseline_distilbert_civil_2000_gender" \
    --bias_type="gender" \

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="test" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/models/baseline_bert_civil" \
    --batch_size=8 \
    --max_seq_length=512 \
    --num_examples=2000 \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/results/baseline_bert_civil_2000_race" \
    --bias_type="race" \

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="test" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/models/baseline_roberta_civil" \
    --batch_size=8 \
    --max_seq_length=512 \
    --num_examples=2000 \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/results/baseline_roberta_civil_2000_race" \
    --bias_type="race" \

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="test" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/models/baseline_distilbert_civil" \
    --batch_size=8 \
    --max_seq_length=512 \
    --num_examples=2000 \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/results/baseline_distilbert_civil_2000_race" \
    --bias_type="race" \

#!/bin/bash

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="test" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/explanation_debiased_models/bert_civil_race/Saliency/L2_0.01" \
    --batch_size=64 \
    --max_seq_length=512 \
    --num_examples=2000 \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/bias_mitigation_results_civil/bert_civil_race_race_test_2000/Saliency/L2_0.01" \
    --bias_type="race"

# python -m bias_mitigation.bias_mitigation \
#     --model_name_or_path="bert-base-uncased" \
#     --dataset_name="lighteval/civil_comments_helm" \
#     --num_labels=2 \
#     --output_dir="/scratch/yifwang/new_fairness_x_explainability/explanation_debiased_models/bert_civil_race" \
#     --batch_size=8 \
#     --max_seq_length=512 \
#     --learning_rate=2e-05 \
#     --warmup_steps_or_ratio=0.1 \
#     --num_train_epochs=5 \
#     --early_stopping_patience=-1 \
#     --eval_steps=1000 \
#     --save_steps=1000 \
#     --split_ratio="8, 2" \
#     --seed=42 \
#     --eval_metric="accuracy" \
#     --bias_type="race" \
#     --total_num_examples=8000 \
#     --explanation_method="Saliency" \
#     --aggregation="L2" \
#     --alpha=0.01 \

python -m bias_mitigation.bias_mitigation \
    --model_name_or_path="bert-base-uncased" \
    --dataset_name="lighteval/civil_comments_helm" \
    --num_labels=2 \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/explanation_debiased_models/bert_civil_race" \
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
    --total_num_examples=8000 \
    --explanation_method="Saliency" \
    --aggregation="L1" \
    --alpha=0.01 \

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="test" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/explanation_debiased_models/bert_civil_race/Saliency/L1_0.01" \
    --batch_size=64 \
    --max_seq_length=512 \
    --num_examples=2000 \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/bias_mitigation_results_civil/bert_civil_race_race_test_2000/Saliency/L1_0.01" \
    --bias_type="race"
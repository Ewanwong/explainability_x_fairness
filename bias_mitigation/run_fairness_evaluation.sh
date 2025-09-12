#!/usr/bin/env bash

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
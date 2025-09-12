#!/bin/bash

bash bias_mitigation_scripts_civil/compute_fairness_race.sh
bash bias_mitigation_scripts_civil/compute_fairness_gender.sh
bash bias_mitigation_scripts_civil/compute_fairness_religion.sh

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="test" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/fairness_x_explainability/explanation_debiased_models_civil/roberta_civil_gender_2/Saliency/L1_1.0" \
    --batch_size=64 \
    --max_seq_length=512 \
    --num_examples=2000 \
    --output_dir="/scratch/yifwang/fairness_x_explainability/bias_mitigation_results_civil/roberta_civil_gender_gender_test_2000_2/Saliency/L1_1.0" \
    --bias_type="gender" 
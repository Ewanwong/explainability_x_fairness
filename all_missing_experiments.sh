#!/bin/bash

python -m bias_mitigation.bias_mitigation \
    --model_name_or_path="FacebookAI/roberta-base" \
    --dataset_name="lighteval/civil_comments_helm" \
    --num_labels=2 \
    --output_dir="/scratch/yifwang/fairness_x_explainability/explanation_debiased_models_civil/roberta_civil_gender_2" \
    --batch_size=8 \
    --gradient_accumulation_steps=1 \
    --max_seq_length=512 \
    --learning_rate=2e-05 \
    --warmup_steps_or_ratio=0.1 \
    --num_train_epochs=5 \
    --early_stopping_patience=-1 \
    --eval_steps=1000 \
    --save_steps=1000 \
    --split_ratio="8, 2" \
    --seed=2 \
    --eval_metric="accuracy" \
    --bias_type="gender" \
    --total_num_examples=8000\
    --explanation_method="Saliency" \
    --aggregation="L1" \
    --alpha=1.0 \
    --n_steps=20 

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="test" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/fairness_x_explainability/explanation_debiased_models_civil/roberta_civil_gender_2/Saliency/L1_1.0" \
    --batch_size=4 \
    --max_seq_length=512 \
    --num_examples=2000 \
    --output_dir="/scratch/yifwang/fairness_x_explainability/bias_mitigation_results_civil/roberta_civil_gender_gender_test_2000_2/Saliency/L1_1.0" \
    --bias_type="gender" 

#!/usr/bin/env bash

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/debiased_models/bert_civil_race/no_debiasing" \
    --batch_size=32 \
    --max_seq_length=512 \
    --num_examples=500 \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/model_selection_results/bert_civil_no_debiasing_val_race_500" \
    --bias_type="race" \

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/debiased_models/bert_civil_race/group_balance" \
    --batch_size=32 \
    --max_seq_length=512 \
    --num_examples=500 \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/model_selection_results/bert_civil_group_balance_val_race_500" \
    --bias_type="race" \

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/debiased_models/bert_civil_race/group_class_balance" \
    --batch_size=32 \
    --max_seq_length=512 \
    --num_examples=500 \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/model_selection_results/bert_civil_group_class_balance_val_race_500" \
    --bias_type="race" \

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/debiased_models/bert_civil_race/cda" \
    --batch_size=32 \
    --max_seq_length=512 \
    --num_examples=500 \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/model_selection_results/bert_civil_cda_val_race_500" \
    --bias_type="race" \

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/debiased_models/bert_civil_gender/no_debiasing" \
    --batch_size=32 \
    --max_seq_length=512 \
    --num_examples=500 \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/model_selection_results/bert_civil_no_debiasing_val_gender_500" \
    --bias_type="gender" \

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/debiased_models/bert_civil_gender/group_balance" \
    --batch_size=32 \
    --max_seq_length=512 \
    --num_examples=500 \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/model_selection_results/bert_civil_group_balance_val_gender_500" \
    --bias_type="gender" \

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/debiased_models/bert_civil_gender/group_class_balance" \
    --batch_size=32 \
    --max_seq_length=512 \
    --num_examples=500 \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/model_selection_results/bert_civil_group_class_balance_val_gender_500" \
    --bias_type="gender" \

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/debiased_models/bert_civil_gender/cda" \
    --batch_size=32 \
    --max_seq_length=512 \
    --num_examples=500 \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/model_selection_results/bert_civil_cda_val_gender_500" \
    --bias_type="gender" \

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/debiased_models/roberta_civil_race/no_debiasing" \
    --batch_size=32 \
    --max_seq_length=512 \
    --num_examples=500 \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/model_selection_results/roberta_civil_no_debiasing_val_race_500" \
    --bias_type="race" \

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/debiased_models/roberta_civil_race/group_balance" \
    --batch_size=32 \
    --max_seq_length=512 \
    --num_examples=500 \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/model_selection_results/roberta_civil_group_balance_val_race_500" \
    --bias_type="race" \

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/debiased_models/roberta_civil_race/group_class_balance" \
    --batch_size=32 \
    --max_seq_length=512 \
    --num_examples=500 \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/model_selection_results/roberta_civil_group_class_balance_val_race_500" \
    --bias_type="race" \

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/debiased_models/roberta_civil_race/cda" \
    --batch_size=32 \
    --max_seq_length=512 \
    --num_examples=500 \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/model_selection_results/roberta_civil_cda_val_race_500" \
    --bias_type="race" \

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/debiased_models/roberta_civil_gender/no_debiasing" \
    --batch_size=32 \
    --max_seq_length=512 \
    --num_examples=500 \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/model_selection_results/roberta_civil_no_debiasing_val_gender_500" \
    --bias_type="gender" \

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/debiased_models/roberta_civil_gender/group_balance" \
    --batch_size=32 \
    --max_seq_length=512 \
    --num_examples=500 \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/model_selection_results/roberta_civil_group_balance_val_gender_500" \
    --bias_type="gender" \

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/debiased_models/roberta_civil_gender/group_class_balance" \
    --batch_size=32 \
    --max_seq_length=512 \
    --num_examples=500 \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/model_selection_results/roberta_civil_group_class_balance_val_gender_500" \
    --bias_type="gender" \

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/debiased_models/roberta_civil_gender/cda" \
    --batch_size=32 \
    --max_seq_length=512 \
    --num_examples=500 \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/model_selection_results/roberta_civil_cda_val_gender_500" \
    --bias_type="gender" \

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/debiased_models/distilbert_civil_race/no_debiasing" \
    --batch_size=32 \
    --max_seq_length=512 \
    --num_examples=500 \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/model_selection_results/distilbert_civil_no_debiasing_val_race_500" \
    --bias_type="race" \

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/debiased_models/distilbert_civil_race/group_balance" \
    --batch_size=32 \
    --max_seq_length=512 \
    --num_examples=500 \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/model_selection_results/distilbert_civil_group_balance_val_race_500" \
    --bias_type="race" \

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/debiased_models/distilbert_civil_race/group_class_balance" \
    --batch_size=32 \
    --max_seq_length=512 \
    --num_examples=500 \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/model_selection_results/distilbert_civil_group_class_balance_val_race_500" \
    --bias_type="race" \

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/debiased_models/distilbert_civil_race/cda" \
    --batch_size=32 \
    --max_seq_length=512 \
    --num_examples=500 \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/model_selection_results/distilbert_civil_cda_val_race_500" \
    --bias_type="race" \

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/debiased_models/distilbert_civil_gender/no_debiasing" \
    --batch_size=32 \
    --max_seq_length=512 \
    --num_examples=500 \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/model_selection_results/distilbert_civil_no_debiasing_val_gender_500" \
    --bias_type="gender" \

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/debiased_models/distilbert_civil_gender/group_balance" \
    --batch_size=32 \
    --max_seq_length=512 \
    --num_examples=500 \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/model_selection_results/distilbert_civil_group_balance_val_gender_500" \
    --bias_type="gender" \

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/debiased_models/distilbert_civil_gender/group_class_balance" \
    --batch_size=32 \
    --max_seq_length=512 \
    --num_examples=500 \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/model_selection_results/distilbert_civil_group_class_balance_val_gender_500" \
    --bias_type="gender" \

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/debiased_models/distilbert_civil_gender/cda" \
    --batch_size=32 \
    --max_seq_length=512 \
    --num_examples=500 \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/model_selection_results/distilbert_civil_cda_val_gender_500" \
    --bias_type="gender" \
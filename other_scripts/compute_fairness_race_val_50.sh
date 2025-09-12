#!/bin/bash
python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models/bert_civil_race/no_debiasing" \
    --batch_size=64 \
    --max_seq_length=512 \
    --num_examples=50 \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/bert_civil_race_race_val_50/no_debiasing" \
    --bias_type="race"

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models/bert_civil_all/no_debiasing" \
    --batch_size=64 \
    --max_seq_length=512 \
    --num_examples=50 \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/bert_civil_all_race_val_50/no_debiasing" \
    --bias_type="race"

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models/bert_civil_race/group_balance" \
    --batch_size=64 \
    --max_seq_length=512 \
    --num_examples=50 \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/bert_civil_race_race_val_50/group_balance" \
    --bias_type="race"

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models/bert_civil_all/group_balance" \
    --batch_size=64 \
    --max_seq_length=512 \
    --num_examples=50 \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/bert_civil_all_race_val_50/group_balance" \
    --bias_type="race"

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models/bert_civil_race/group_class_balance" \
    --batch_size=64 \
    --max_seq_length=512 \
    --num_examples=50 \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/bert_civil_race_race_val_50/group_class_balance" \
    --bias_type="race"

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models/bert_civil_all/group_class_balance" \
    --batch_size=64 \
    --max_seq_length=512 \
    --num_examples=50 \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/bert_civil_all_race_val_50/group_class_balance" \
    --bias_type="race"

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models/bert_civil_race/cda" \
    --batch_size=64 \
    --max_seq_length=512 \
    --num_examples=50 \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/bert_civil_race_race_val_50/cda" \
    --bias_type="race"

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models/bert_civil_all/cda" \
    --batch_size=64 \
    --max_seq_length=512 \
    --num_examples=50 \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/bert_civil_all_race_val_50/cda" \
    --bias_type="race"

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models/roberta_civil_race/no_debiasing" \
    --batch_size=64 \
    --max_seq_length=512 \
    --num_examples=50 \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/roberta_civil_race_race_val_50/no_debiasing" \
    --bias_type="race"

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models/roberta_civil_all/no_debiasing" \
    --batch_size=64 \
    --max_seq_length=512 \
    --num_examples=50 \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/roberta_civil_all_race_val_50/no_debiasing" \
    --bias_type="race"

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models/roberta_civil_race/group_balance" \
    --batch_size=64 \
    --max_seq_length=512 \
    --num_examples=50 \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/roberta_civil_race_race_val_50/group_balance" \
    --bias_type="race"

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models/roberta_civil_all/group_balance" \
    --batch_size=64 \
    --max_seq_length=512 \
    --num_examples=50 \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/roberta_civil_all_race_val_50/group_balance" \
    --bias_type="race"

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models/roberta_civil_race/group_class_balance" \
    --batch_size=64 \
    --max_seq_length=512 \
    --num_examples=50 \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/roberta_civil_race_race_val_50/group_class_balance" \
    --bias_type="race"

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models/roberta_civil_all/group_class_balance" \
    --batch_size=64 \
    --max_seq_length=512 \
    --num_examples=50 \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/roberta_civil_all_race_val_50/group_class_balance" \
    --bias_type="race"

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models/roberta_civil_race/cda" \
    --batch_size=64 \
    --max_seq_length=512 \
    --num_examples=50 \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/roberta_civil_race_race_val_50/cda" \
    --bias_type="race"

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models/roberta_civil_all/cda" \
    --batch_size=64 \
    --max_seq_length=512 \
    --num_examples=50 \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/roberta_civil_all_race_val_50/cda" \
    --bias_type="race"

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models/distilbert_civil_race/no_debiasing" \
    --batch_size=64 \
    --max_seq_length=512 \
    --num_examples=50 \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/distilbert_civil_race_race_val_50/no_debiasing" \
    --bias_type="race"

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models/distilbert_civil_all/no_debiasing" \
    --batch_size=64 \
    --max_seq_length=512 \
    --num_examples=50 \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/distilbert_civil_all_race_val_50/no_debiasing" \
    --bias_type="race"

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models/distilbert_civil_race/group_balance" \
    --batch_size=64 \
    --max_seq_length=512 \
    --num_examples=50 \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/distilbert_civil_race_race_val_50/group_balance" \
    --bias_type="race"

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models/distilbert_civil_all/group_balance" \
    --batch_size=64 \
    --max_seq_length=512 \
    --num_examples=50 \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/distilbert_civil_all_race_val_50/group_balance" \
    --bias_type="race"

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models/distilbert_civil_race/group_class_balance" \
    --batch_size=64 \
    --max_seq_length=512 \
    --num_examples=50 \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/distilbert_civil_race_race_val_50/group_class_balance" \
    --bias_type="race"

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models/distilbert_civil_all/group_class_balance" \
    --batch_size=64 \
    --max_seq_length=512 \
    --num_examples=50 \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/distilbert_civil_all_race_val_50/group_class_balance" \
    --bias_type="race"

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models/distilbert_civil_race/cda" \
    --batch_size=64 \
    --max_seq_length=512 \
    --num_examples=50 \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/distilbert_civil_race_race_val_50/cda" \
    --bias_type="race"

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models/distilbert_civil_all/cda" \
    --batch_size=64 \
    --max_seq_length=512 \
    --num_examples=50 \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/distilbert_civil_all_race_val_50/cda" \
    --bias_type="race"


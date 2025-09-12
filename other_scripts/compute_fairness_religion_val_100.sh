#!/bin/bash
python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models/bert_civil_religion/no_debiasing" \
    --batch_size=64 \
    --max_seq_length=512 \
    --num_examples=100 \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/bert_civil_religion_religion_val_100/no_debiasing" \
    --bias_type="religion"

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models/bert_civil_all/no_debiasing" \
    --batch_size=64 \
    --max_seq_length=512 \
    --num_examples=100 \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/bert_civil_all_religion_val_100/no_debiasing" \
    --bias_type="religion"

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models/bert_civil_religion/group_balance" \
    --batch_size=64 \
    --max_seq_length=512 \
    --num_examples=100 \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/bert_civil_religion_religion_val_100/group_balance" \
    --bias_type="religion"

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models/bert_civil_all/group_balance" \
    --batch_size=64 \
    --max_seq_length=512 \
    --num_examples=100 \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/bert_civil_all_religion_val_100/group_balance" \
    --bias_type="religion"

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models/bert_civil_religion/group_class_balance" \
    --batch_size=64 \
    --max_seq_length=512 \
    --num_examples=100 \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/bert_civil_religion_religion_val_100/group_class_balance" \
    --bias_type="religion"

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models/bert_civil_all/group_class_balance" \
    --batch_size=64 \
    --max_seq_length=512 \
    --num_examples=100 \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/bert_civil_all_religion_val_100/group_class_balance" \
    --bias_type="religion"

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models/bert_civil_religion/cda" \
    --batch_size=64 \
    --max_seq_length=512 \
    --num_examples=100 \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/bert_civil_religion_religion_val_100/cda" \
    --bias_type="religion"

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models/bert_civil_all/cda" \
    --batch_size=64 \
    --max_seq_length=512 \
    --num_examples=100 \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/bert_civil_all_religion_val_100/cda" \
    --bias_type="religion"

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models/roberta_civil_religion/no_debiasing" \
    --batch_size=64 \
    --max_seq_length=512 \
    --num_examples=100 \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/roberta_civil_religion_religion_val_100/no_debiasing" \
    --bias_type="religion"

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models/roberta_civil_all/no_debiasing" \
    --batch_size=64 \
    --max_seq_length=512 \
    --num_examples=100 \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/roberta_civil_all_religion_val_100/no_debiasing" \
    --bias_type="religion"

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models/roberta_civil_religion/group_balance" \
    --batch_size=64 \
    --max_seq_length=512 \
    --num_examples=100 \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/roberta_civil_religion_religion_val_100/group_balance" \
    --bias_type="religion"

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models/roberta_civil_all/group_balance" \
    --batch_size=64 \
    --max_seq_length=512 \
    --num_examples=100 \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/roberta_civil_all_religion_val_100/group_balance" \
    --bias_type="religion"

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models/roberta_civil_religion/group_class_balance" \
    --batch_size=64 \
    --max_seq_length=512 \
    --num_examples=100 \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/roberta_civil_religion_religion_val_100/group_class_balance" \
    --bias_type="religion"

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models/roberta_civil_all/group_class_balance" \
    --batch_size=64 \
    --max_seq_length=512 \
    --num_examples=100 \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/roberta_civil_all_religion_val_100/group_class_balance" \
    --bias_type="religion"

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models/roberta_civil_religion/cda" \
    --batch_size=64 \
    --max_seq_length=512 \
    --num_examples=100 \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/roberta_civil_religion_religion_val_100/cda" \
    --bias_type="religion"

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models/roberta_civil_all/cda" \
    --batch_size=64 \
    --max_seq_length=512 \
    --num_examples=100 \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/roberta_civil_all_religion_val_100/cda" \
    --bias_type="religion"

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models/distilbert_civil_religion/no_debiasing" \
    --batch_size=64 \
    --max_seq_length=512 \
    --num_examples=100 \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/distilbert_civil_religion_religion_val_100/no_debiasing" \
    --bias_type="religion"

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models/distilbert_civil_all/no_debiasing" \
    --batch_size=64 \
    --max_seq_length=512 \
    --num_examples=100 \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/distilbert_civil_all_religion_val_100/no_debiasing" \
    --bias_type="religion"

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models/distilbert_civil_religion/group_balance" \
    --batch_size=64 \
    --max_seq_length=512 \
    --num_examples=100 \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/distilbert_civil_religion_religion_val_100/group_balance" \
    --bias_type="religion"

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models/distilbert_civil_all/group_balance" \
    --batch_size=64 \
    --max_seq_length=512 \
    --num_examples=100 \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/distilbert_civil_all_religion_val_100/group_balance" \
    --bias_type="religion"

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models/distilbert_civil_religion/group_class_balance" \
    --batch_size=64 \
    --max_seq_length=512 \
    --num_examples=100 \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/distilbert_civil_religion_religion_val_100/group_class_balance" \
    --bias_type="religion"

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models/distilbert_civil_all/group_class_balance" \
    --batch_size=64 \
    --max_seq_length=512 \
    --num_examples=100 \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/distilbert_civil_all_religion_val_100/group_class_balance" \
    --bias_type="religion"

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models/distilbert_civil_religion/cda" \
    --batch_size=64 \
    --max_seq_length=512 \
    --num_examples=100 \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/distilbert_civil_religion_religion_val_100/cda" \
    --bias_type="religion"

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models/distilbert_civil_all/cda" \
    --batch_size=64 \
    --max_seq_length=512 \
    --num_examples=100 \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/distilbert_civil_all_religion_val_100/cda" \
    --bias_type="religion"


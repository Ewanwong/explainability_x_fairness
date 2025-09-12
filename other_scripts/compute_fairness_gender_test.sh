#!/bin/bash
python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="test" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models/bert_civil_all/dropout" \
    --batch_size=64 \
    --max_seq_length=512 \
    --num_examples=2000 \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/bert_civil_all_gender_test_2000/dropout" \
    --bias_type="gender"

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="test" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models/bert_civil_all/attention_entropy" \
    --batch_size=64 \
    --max_seq_length=512 \
    --num_examples=2000 \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/bert_civil_all_gender_test_2000/attention_entropy" \
    --bias_type="gender"

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="test" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models/bert_civil_all/causal_debias" \
    --batch_size=64 \
    --max_seq_length=512 \
    --num_examples=2000 \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/bert_civil_all_gender_test_2000/causal_debias" \
    --bias_type="gender"

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="test" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models/roberta_civil_all/dropout" \
    --batch_size=64 \
    --max_seq_length=512 \
    --num_examples=2000 \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/roberta_civil_all_gender_test_2000/dropout" \
    --bias_type="gender"

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="test" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models/roberta_civil_all/attention_entropy" \
    --batch_size=64 \
    --max_seq_length=512 \
    --num_examples=2000 \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/roberta_civil_all_gender_test_2000/attention_entropy" \
    --bias_type="gender"

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="test" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models/roberta_civil_all/causal_debias" \
    --batch_size=64 \
    --max_seq_length=512 \
    --num_examples=2000 \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/roberta_civil_all_gender_test_2000/causal_debias" \
    --bias_type="gender"

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="test" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models/distilbert_civil_all/dropout" \
    --batch_size=64 \
    --max_seq_length=512 \
    --num_examples=2000 \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/distilbert_civil_all_gender_test_2000/dropout" \
    --bias_type="gender"

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="test" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models/distilbert_civil_all/attention_entropy" \
    --batch_size=64 \
    --max_seq_length=512 \
    --num_examples=2000 \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/distilbert_civil_all_gender_test_2000/attention_entropy" \
    --bias_type="gender"

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="test" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models/distilbert_civil_all/causal_debias" \
    --batch_size=64 \
    --max_seq_length=512 \
    --num_examples=2000 \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/distilbert_civil_all_gender_test_2000/causal_debias" \
    --bias_type="gender"


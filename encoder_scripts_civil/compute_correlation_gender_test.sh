#!/bin/bash

python -m bias_correlation.compute_bias_correlation \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results_civil/bert_civil_gender_gender_test_2000/no_debiasing" \
    --split="test" \
    --bias_type="gender" \
    --methods="Occlusion"

python -m visualization.visualize_bias_correlation \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results_civil/bert_civil_gender_gender_test_2000/no_debiasing" \
    --split="test" \
    --bias_type="gender" \
    --methods="Occlusion"

python -m bias_correlation.compute_bias_correlation \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results_civil/bert_civil_all_gender_test_2000/no_debiasing" \
    --split="test" \
    --bias_type="gender" \
    --methods="Occlusion"

python -m visualization.visualize_bias_correlation \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results_civil/bert_civil_all_gender_test_2000/no_debiasing" \
    --split="test" \
    --bias_type="gender" \
    --methods="Occlusion"

python -m bias_correlation.compute_bias_correlation \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results_civil/bert_civil_gender_gender_test_2000/group_balance" \
    --split="test" \
    --bias_type="gender" \
    --methods="Occlusion"

python -m visualization.visualize_bias_correlation \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results_civil/bert_civil_gender_gender_test_2000/group_balance" \
    --split="test" \
    --bias_type="gender" \
    --methods="Occlusion"

python -m bias_correlation.compute_bias_correlation \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results_civil/bert_civil_all_gender_test_2000/group_balance" \
    --split="test" \
    --bias_type="gender" \
    --methods="Occlusion"

python -m visualization.visualize_bias_correlation \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results_civil/bert_civil_all_gender_test_2000/group_balance" \
    --split="test" \
    --bias_type="gender" \
    --methods="Occlusion"

python -m bias_correlation.compute_bias_correlation \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results_civil/bert_civil_gender_gender_test_2000/group_class_balance" \
    --split="test" \
    --bias_type="gender" \
    --methods="Occlusion"

python -m visualization.visualize_bias_correlation \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results_civil/bert_civil_gender_gender_test_2000/group_class_balance" \
    --split="test" \
    --bias_type="gender" \
    --methods="Occlusion"

python -m bias_correlation.compute_bias_correlation \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results_civil/bert_civil_all_gender_test_2000/group_class_balance" \
    --split="test" \
    --bias_type="gender" \
    --methods="Occlusion"

python -m visualization.visualize_bias_correlation \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results_civil/bert_civil_all_gender_test_2000/group_class_balance" \
    --split="test" \
    --bias_type="gender" \
    --methods="Occlusion"

python -m bias_correlation.compute_bias_correlation \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results_civil/bert_civil_gender_gender_test_2000/cda" \
    --split="test" \
    --bias_type="gender" \
    --methods="Occlusion"

python -m visualization.visualize_bias_correlation \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results_civil/bert_civil_gender_gender_test_2000/cda" \
    --split="test" \
    --bias_type="gender" \
    --methods="Occlusion"

python -m bias_correlation.compute_bias_correlation \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results_civil/bert_civil_all_gender_test_2000/cda" \
    --split="test" \
    --bias_type="gender" \
    --methods="Occlusion"

python -m visualization.visualize_bias_correlation \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results_civil/bert_civil_all_gender_test_2000/cda" \
    --split="test" \
    --bias_type="gender" \
    --methods="Occlusion"

python -m bias_correlation.compute_bias_correlation \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results_civil/bert_civil_gender_gender_test_2000/dropout" \
    --split="test" \
    --bias_type="gender" \
    --methods="Occlusion"

python -m visualization.visualize_bias_correlation \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results_civil/bert_civil_gender_gender_test_2000/dropout" \
    --split="test" \
    --bias_type="gender" \
    --methods="Occlusion"

python -m bias_correlation.compute_bias_correlation \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results_civil/bert_civil_all_gender_test_2000/dropout" \
    --split="test" \
    --bias_type="gender" \
    --methods="Occlusion"

python -m visualization.visualize_bias_correlation \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results_civil/bert_civil_all_gender_test_2000/dropout" \
    --split="test" \
    --bias_type="gender" \
    --methods="Occlusion"

python -m bias_correlation.compute_bias_correlation \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results_civil/bert_civil_gender_gender_test_2000/attention_entropy" \
    --split="test" \
    --bias_type="gender" \
    --methods="Occlusion"

python -m visualization.visualize_bias_correlation \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results_civil/bert_civil_gender_gender_test_2000/attention_entropy" \
    --split="test" \
    --bias_type="gender" \
    --methods="Occlusion"

python -m bias_correlation.compute_bias_correlation \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results_civil/bert_civil_all_gender_test_2000/attention_entropy" \
    --split="test" \
    --bias_type="gender" \
    --methods="Occlusion"

python -m visualization.visualize_bias_correlation \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results_civil/bert_civil_all_gender_test_2000/attention_entropy" \
    --split="test" \
    --bias_type="gender" \
    --methods="Occlusion"

python -m bias_correlation.compute_bias_correlation \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results_civil/bert_civil_gender_gender_test_2000/causal_debias" \
    --split="test" \
    --bias_type="gender" \
    --methods="Occlusion"

python -m visualization.visualize_bias_correlation \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results_civil/bert_civil_gender_gender_test_2000/causal_debias" \
    --split="test" \
    --bias_type="gender" \
    --methods="Occlusion"

python -m bias_correlation.compute_bias_correlation \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results_civil/bert_civil_all_gender_test_2000/causal_debias" \
    --split="test" \
    --bias_type="gender" \
    --methods="Occlusion"

python -m visualization.visualize_bias_correlation \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results_civil/bert_civil_all_gender_test_2000/causal_debias" \
    --split="test" \
    --bias_type="gender" \
    --methods="Occlusion"

python -m bias_correlation.compute_bias_correlation \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results_civil/roberta_civil_gender_gender_test_2000/no_debiasing" \
    --split="test" \
    --bias_type="gender" \
    --methods="Occlusion"

python -m visualization.visualize_bias_correlation \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results_civil/roberta_civil_gender_gender_test_2000/no_debiasing" \
    --split="test" \
    --bias_type="gender" \
    --methods="Occlusion"

python -m bias_correlation.compute_bias_correlation \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results_civil/roberta_civil_all_gender_test_2000/no_debiasing" \
    --split="test" \
    --bias_type="gender" \
    --methods="Occlusion"

python -m visualization.visualize_bias_correlation \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results_civil/roberta_civil_all_gender_test_2000/no_debiasing" \
    --split="test" \
    --bias_type="gender" \
    --methods="Occlusion"

python -m bias_correlation.compute_bias_correlation \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results_civil/roberta_civil_gender_gender_test_2000/group_balance" \
    --split="test" \
    --bias_type="gender" \
    --methods="Occlusion"

python -m visualization.visualize_bias_correlation \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results_civil/roberta_civil_gender_gender_test_2000/group_balance" \
    --split="test" \
    --bias_type="gender" \
    --methods="Occlusion"

python -m bias_correlation.compute_bias_correlation \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results_civil/roberta_civil_all_gender_test_2000/group_balance" \
    --split="test" \
    --bias_type="gender" \
    --methods="Occlusion"

python -m visualization.visualize_bias_correlation \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results_civil/roberta_civil_all_gender_test_2000/group_balance" \
    --split="test" \
    --bias_type="gender" \
    --methods="Occlusion"

python -m bias_correlation.compute_bias_correlation \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results_civil/roberta_civil_gender_gender_test_2000/group_class_balance" \
    --split="test" \
    --bias_type="gender" \
    --methods="Occlusion"

python -m visualization.visualize_bias_correlation \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results_civil/roberta_civil_gender_gender_test_2000/group_class_balance" \
    --split="test" \
    --bias_type="gender" \
    --methods="Occlusion"

python -m bias_correlation.compute_bias_correlation \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results_civil/roberta_civil_all_gender_test_2000/group_class_balance" \
    --split="test" \
    --bias_type="gender" \
    --methods="Occlusion"

python -m visualization.visualize_bias_correlation \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results_civil/roberta_civil_all_gender_test_2000/group_class_balance" \
    --split="test" \
    --bias_type="gender" \
    --methods="Occlusion"

python -m bias_correlation.compute_bias_correlation \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results_civil/roberta_civil_gender_gender_test_2000/cda" \
    --split="test" \
    --bias_type="gender" \
    --methods="Occlusion"

python -m visualization.visualize_bias_correlation \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results_civil/roberta_civil_gender_gender_test_2000/cda" \
    --split="test" \
    --bias_type="gender" \
    --methods="Occlusion"

python -m bias_correlation.compute_bias_correlation \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results_civil/roberta_civil_all_gender_test_2000/cda" \
    --split="test" \
    --bias_type="gender" \
    --methods="Occlusion"

python -m visualization.visualize_bias_correlation \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results_civil/roberta_civil_all_gender_test_2000/cda" \
    --split="test" \
    --bias_type="gender" \
    --methods="Occlusion"

python -m bias_correlation.compute_bias_correlation \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results_civil/roberta_civil_gender_gender_test_2000/dropout" \
    --split="test" \
    --bias_type="gender" \
    --methods="Occlusion"

python -m visualization.visualize_bias_correlation \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results_civil/roberta_civil_gender_gender_test_2000/dropout" \
    --split="test" \
    --bias_type="gender" \
    --methods="Occlusion"

python -m bias_correlation.compute_bias_correlation \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results_civil/roberta_civil_all_gender_test_2000/dropout" \
    --split="test" \
    --bias_type="gender" \
    --methods="Occlusion"

python -m visualization.visualize_bias_correlation \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results_civil/roberta_civil_all_gender_test_2000/dropout" \
    --split="test" \
    --bias_type="gender" \
    --methods="Occlusion"

python -m bias_correlation.compute_bias_correlation \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results_civil/roberta_civil_gender_gender_test_2000/attention_entropy" \
    --split="test" \
    --bias_type="gender" \
    --methods="Occlusion"

python -m visualization.visualize_bias_correlation \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results_civil/roberta_civil_gender_gender_test_2000/attention_entropy" \
    --split="test" \
    --bias_type="gender" \
    --methods="Occlusion"

python -m bias_correlation.compute_bias_correlation \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results_civil/roberta_civil_all_gender_test_2000/attention_entropy" \
    --split="test" \
    --bias_type="gender" \
    --methods="Occlusion"

python -m visualization.visualize_bias_correlation \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results_civil/roberta_civil_all_gender_test_2000/attention_entropy" \
    --split="test" \
    --bias_type="gender" \
    --methods="Occlusion"

python -m bias_correlation.compute_bias_correlation \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results_civil/roberta_civil_gender_gender_test_2000/causal_debias" \
    --split="test" \
    --bias_type="gender" \
    --methods="Occlusion"

python -m visualization.visualize_bias_correlation \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results_civil/roberta_civil_gender_gender_test_2000/causal_debias" \
    --split="test" \
    --bias_type="gender" \
    --methods="Occlusion"

python -m bias_correlation.compute_bias_correlation \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results_civil/roberta_civil_all_gender_test_2000/causal_debias" \
    --split="test" \
    --bias_type="gender" \
    --methods="Occlusion"

python -m visualization.visualize_bias_correlation \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results_civil/roberta_civil_all_gender_test_2000/causal_debias" \
    --split="test" \
    --bias_type="gender" \
    --methods="Occlusion"

python -m bias_correlation.compute_bias_correlation \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results_civil/distilbert_civil_gender_gender_test_2000/no_debiasing" \
    --split="test" \
    --bias_type="gender" \
    --methods="Occlusion"

python -m visualization.visualize_bias_correlation \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results_civil/distilbert_civil_gender_gender_test_2000/no_debiasing" \
    --split="test" \
    --bias_type="gender" \
    --methods="Occlusion"

python -m bias_correlation.compute_bias_correlation \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results_civil/distilbert_civil_all_gender_test_2000/no_debiasing" \
    --split="test" \
    --bias_type="gender" \
    --methods="Occlusion"

python -m visualization.visualize_bias_correlation \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results_civil/distilbert_civil_all_gender_test_2000/no_debiasing" \
    --split="test" \
    --bias_type="gender" \
    --methods="Occlusion"

python -m bias_correlation.compute_bias_correlation \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results_civil/distilbert_civil_gender_gender_test_2000/group_balance" \
    --split="test" \
    --bias_type="gender" \
    --methods="Occlusion"

python -m visualization.visualize_bias_correlation \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results_civil/distilbert_civil_gender_gender_test_2000/group_balance" \
    --split="test" \
    --bias_type="gender" \
    --methods="Occlusion"

python -m bias_correlation.compute_bias_correlation \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results_civil/distilbert_civil_all_gender_test_2000/group_balance" \
    --split="test" \
    --bias_type="gender" \
    --methods="Occlusion"

python -m visualization.visualize_bias_correlation \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results_civil/distilbert_civil_all_gender_test_2000/group_balance" \
    --split="test" \
    --bias_type="gender" \
    --methods="Occlusion"

python -m bias_correlation.compute_bias_correlation \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results_civil/distilbert_civil_gender_gender_test_2000/group_class_balance" \
    --split="test" \
    --bias_type="gender" \
    --methods="Occlusion"

python -m visualization.visualize_bias_correlation \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results_civil/distilbert_civil_gender_gender_test_2000/group_class_balance" \
    --split="test" \
    --bias_type="gender" \
    --methods="Occlusion"

python -m bias_correlation.compute_bias_correlation \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results_civil/distilbert_civil_all_gender_test_2000/group_class_balance" \
    --split="test" \
    --bias_type="gender" \
    --methods="Occlusion"

python -m visualization.visualize_bias_correlation \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results_civil/distilbert_civil_all_gender_test_2000/group_class_balance" \
    --split="test" \
    --bias_type="gender" \
    --methods="Occlusion"

python -m bias_correlation.compute_bias_correlation \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results_civil/distilbert_civil_gender_gender_test_2000/cda" \
    --split="test" \
    --bias_type="gender" \
    --methods="Occlusion"

python -m visualization.visualize_bias_correlation \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results_civil/distilbert_civil_gender_gender_test_2000/cda" \
    --split="test" \
    --bias_type="gender" \
    --methods="Occlusion"

python -m bias_correlation.compute_bias_correlation \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results_civil/distilbert_civil_all_gender_test_2000/cda" \
    --split="test" \
    --bias_type="gender" \
    --methods="Occlusion"

python -m visualization.visualize_bias_correlation \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results_civil/distilbert_civil_all_gender_test_2000/cda" \
    --split="test" \
    --bias_type="gender" \
    --methods="Occlusion"

python -m bias_correlation.compute_bias_correlation \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results_civil/distilbert_civil_gender_gender_test_2000/dropout" \
    --split="test" \
    --bias_type="gender" \
    --methods="Occlusion"

python -m visualization.visualize_bias_correlation \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results_civil/distilbert_civil_gender_gender_test_2000/dropout" \
    --split="test" \
    --bias_type="gender" \
    --methods="Occlusion"

python -m bias_correlation.compute_bias_correlation \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results_civil/distilbert_civil_all_gender_test_2000/dropout" \
    --split="test" \
    --bias_type="gender" \
    --methods="Occlusion"

python -m visualization.visualize_bias_correlation \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results_civil/distilbert_civil_all_gender_test_2000/dropout" \
    --split="test" \
    --bias_type="gender" \
    --methods="Occlusion"

python -m bias_correlation.compute_bias_correlation \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results_civil/distilbert_civil_gender_gender_test_2000/attention_entropy" \
    --split="test" \
    --bias_type="gender" \
    --methods="Occlusion"

python -m visualization.visualize_bias_correlation \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results_civil/distilbert_civil_gender_gender_test_2000/attention_entropy" \
    --split="test" \
    --bias_type="gender" \
    --methods="Occlusion"

python -m bias_correlation.compute_bias_correlation \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results_civil/distilbert_civil_all_gender_test_2000/attention_entropy" \
    --split="test" \
    --bias_type="gender" \
    --methods="Occlusion"

python -m visualization.visualize_bias_correlation \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results_civil/distilbert_civil_all_gender_test_2000/attention_entropy" \
    --split="test" \
    --bias_type="gender" \
    --methods="Occlusion"

python -m bias_correlation.compute_bias_correlation \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results_civil/distilbert_civil_gender_gender_test_2000/causal_debias" \
    --split="test" \
    --bias_type="gender" \
    --methods="Occlusion"

python -m visualization.visualize_bias_correlation \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results_civil/distilbert_civil_gender_gender_test_2000/causal_debias" \
    --split="test" \
    --bias_type="gender" \
    --methods="Occlusion"

python -m bias_correlation.compute_bias_correlation \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results_civil/distilbert_civil_all_gender_test_2000/causal_debias" \
    --split="test" \
    --bias_type="gender" \
    --methods="Occlusion"

python -m visualization.visualize_bias_correlation \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results_civil/distilbert_civil_all_gender_test_2000/causal_debias" \
    --split="test" \
    --bias_type="gender" \
    --methods="Occlusion"


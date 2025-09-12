#!/bin/bash

python -m sensitive_attribution_extraction.extract_sensitive_token_attribution \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results_civil/bert_civil_religion_religion_test_1000/no_debiasing" \
    --split="test" \
    --bias_type="religion" \
    --methods="Occlusion"

python -m model_selection.compute_reliance_statistics \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results_civil/bert_civil_religion_religion_test_1000/no_debiasing" \
    --split="test" \
    --bias_type="religion" \
    --methods="Occlusion"

python -m sensitive_attribution_extraction.extract_sensitive_token_attribution \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results_civil/bert_civil_all_religion_test_1000/no_debiasing" \
    --split="test" \
    --bias_type="religion" \
    --methods="Occlusion"

python -m model_selection.compute_reliance_statistics \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results_civil/bert_civil_all_religion_test_1000/no_debiasing" \
    --split="test" \
    --bias_type="religion" \
    --methods="Occlusion"

python -m sensitive_attribution_extraction.extract_sensitive_token_attribution \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results_civil/bert_civil_religion_religion_test_1000/group_balance" \
    --split="test" \
    --bias_type="religion" \
    --methods="Occlusion"

python -m model_selection.compute_reliance_statistics \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results_civil/bert_civil_religion_religion_test_1000/group_balance" \
    --split="test" \
    --bias_type="religion" \
    --methods="Occlusion"

python -m sensitive_attribution_extraction.extract_sensitive_token_attribution \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results_civil/bert_civil_all_religion_test_1000/group_balance" \
    --split="test" \
    --bias_type="religion" \
    --methods="Occlusion"

python -m model_selection.compute_reliance_statistics \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results_civil/bert_civil_all_religion_test_1000/group_balance" \
    --split="test" \
    --bias_type="religion" \
    --methods="Occlusion"

python -m sensitive_attribution_extraction.extract_sensitive_token_attribution \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results_civil/bert_civil_religion_religion_test_1000/group_class_balance" \
    --split="test" \
    --bias_type="religion" \
    --methods="Occlusion"

python -m model_selection.compute_reliance_statistics \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results_civil/bert_civil_religion_religion_test_1000/group_class_balance" \
    --split="test" \
    --bias_type="religion" \
    --methods="Occlusion"

python -m sensitive_attribution_extraction.extract_sensitive_token_attribution \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results_civil/bert_civil_all_religion_test_1000/group_class_balance" \
    --split="test" \
    --bias_type="religion" \
    --methods="Occlusion"

python -m model_selection.compute_reliance_statistics \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results_civil/bert_civil_all_religion_test_1000/group_class_balance" \
    --split="test" \
    --bias_type="religion" \
    --methods="Occlusion"

python -m sensitive_attribution_extraction.extract_sensitive_token_attribution \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results_civil/bert_civil_religion_religion_test_1000/cda" \
    --split="test" \
    --bias_type="religion" \
    --methods="Occlusion"

python -m model_selection.compute_reliance_statistics \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results_civil/bert_civil_religion_religion_test_1000/cda" \
    --split="test" \
    --bias_type="religion" \
    --methods="Occlusion"

python -m sensitive_attribution_extraction.extract_sensitive_token_attribution \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results_civil/bert_civil_all_religion_test_1000/cda" \
    --split="test" \
    --bias_type="religion" \
    --methods="Occlusion"

python -m model_selection.compute_reliance_statistics \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results_civil/bert_civil_all_religion_test_1000/cda" \
    --split="test" \
    --bias_type="religion" \
    --methods="Occlusion"

python -m sensitive_attribution_extraction.extract_sensitive_token_attribution \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results_civil/bert_civil_religion_religion_test_1000/dropout" \
    --split="test" \
    --bias_type="religion" \
    --methods="Occlusion"

python -m model_selection.compute_reliance_statistics \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results_civil/bert_civil_religion_religion_test_1000/dropout" \
    --split="test" \
    --bias_type="religion" \
    --methods="Occlusion"

python -m sensitive_attribution_extraction.extract_sensitive_token_attribution \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results_civil/bert_civil_all_religion_test_1000/dropout" \
    --split="test" \
    --bias_type="religion" \
    --methods="Occlusion"

python -m model_selection.compute_reliance_statistics \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results_civil/bert_civil_all_religion_test_1000/dropout" \
    --split="test" \
    --bias_type="religion" \
    --methods="Occlusion"

python -m sensitive_attribution_extraction.extract_sensitive_token_attribution \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results_civil/bert_civil_religion_religion_test_1000/attention_entropy" \
    --split="test" \
    --bias_type="religion" \
    --methods="Occlusion"

python -m model_selection.compute_reliance_statistics \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results_civil/bert_civil_religion_religion_test_1000/attention_entropy" \
    --split="test" \
    --bias_type="religion" \
    --methods="Occlusion"

python -m sensitive_attribution_extraction.extract_sensitive_token_attribution \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results_civil/bert_civil_all_religion_test_1000/attention_entropy" \
    --split="test" \
    --bias_type="religion" \
    --methods="Occlusion"

python -m model_selection.compute_reliance_statistics \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results_civil/bert_civil_all_religion_test_1000/attention_entropy" \
    --split="test" \
    --bias_type="religion" \
    --methods="Occlusion"

python -m sensitive_attribution_extraction.extract_sensitive_token_attribution \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results_civil/bert_civil_religion_religion_test_1000/causal_debias" \
    --split="test" \
    --bias_type="religion" \
    --methods="Occlusion"

python -m model_selection.compute_reliance_statistics \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results_civil/bert_civil_religion_religion_test_1000/causal_debias" \
    --split="test" \
    --bias_type="religion" \
    --methods="Occlusion"

python -m sensitive_attribution_extraction.extract_sensitive_token_attribution \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results_civil/bert_civil_all_religion_test_1000/causal_debias" \
    --split="test" \
    --bias_type="religion" \
    --methods="Occlusion"

python -m model_selection.compute_reliance_statistics \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results_civil/bert_civil_all_religion_test_1000/causal_debias" \
    --split="test" \
    --bias_type="religion" \
    --methods="Occlusion"

python -m sensitive_attribution_extraction.extract_sensitive_token_attribution \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results_civil/roberta_civil_religion_religion_test_1000/no_debiasing" \
    --split="test" \
    --bias_type="religion" \
    --methods="Occlusion"

python -m model_selection.compute_reliance_statistics \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results_civil/roberta_civil_religion_religion_test_1000/no_debiasing" \
    --split="test" \
    --bias_type="religion" \
    --methods="Occlusion"

python -m sensitive_attribution_extraction.extract_sensitive_token_attribution \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results_civil/roberta_civil_all_religion_test_1000/no_debiasing" \
    --split="test" \
    --bias_type="religion" \
    --methods="Occlusion"

python -m model_selection.compute_reliance_statistics \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results_civil/roberta_civil_all_religion_test_1000/no_debiasing" \
    --split="test" \
    --bias_type="religion" \
    --methods="Occlusion"

python -m sensitive_attribution_extraction.extract_sensitive_token_attribution \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results_civil/roberta_civil_religion_religion_test_1000/group_balance" \
    --split="test" \
    --bias_type="religion" \
    --methods="Occlusion"

python -m model_selection.compute_reliance_statistics \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results_civil/roberta_civil_religion_religion_test_1000/group_balance" \
    --split="test" \
    --bias_type="religion" \
    --methods="Occlusion"

python -m sensitive_attribution_extraction.extract_sensitive_token_attribution \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results_civil/roberta_civil_all_religion_test_1000/group_balance" \
    --split="test" \
    --bias_type="religion" \
    --methods="Occlusion"

python -m model_selection.compute_reliance_statistics \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results_civil/roberta_civil_all_religion_test_1000/group_balance" \
    --split="test" \
    --bias_type="religion" \
    --methods="Occlusion"

python -m sensitive_attribution_extraction.extract_sensitive_token_attribution \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results_civil/roberta_civil_religion_religion_test_1000/group_class_balance" \
    --split="test" \
    --bias_type="religion" \
    --methods="Occlusion"

python -m model_selection.compute_reliance_statistics \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results_civil/roberta_civil_religion_religion_test_1000/group_class_balance" \
    --split="test" \
    --bias_type="religion" \
    --methods="Occlusion"

python -m sensitive_attribution_extraction.extract_sensitive_token_attribution \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results_civil/roberta_civil_all_religion_test_1000/group_class_balance" \
    --split="test" \
    --bias_type="religion" \
    --methods="Occlusion"

python -m model_selection.compute_reliance_statistics \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results_civil/roberta_civil_all_religion_test_1000/group_class_balance" \
    --split="test" \
    --bias_type="religion" \
    --methods="Occlusion"

python -m sensitive_attribution_extraction.extract_sensitive_token_attribution \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results_civil/roberta_civil_religion_religion_test_1000/cda" \
    --split="test" \
    --bias_type="religion" \
    --methods="Occlusion"

python -m model_selection.compute_reliance_statistics \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results_civil/roberta_civil_religion_religion_test_1000/cda" \
    --split="test" \
    --bias_type="religion" \
    --methods="Occlusion"

python -m sensitive_attribution_extraction.extract_sensitive_token_attribution \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results_civil/roberta_civil_all_religion_test_1000/cda" \
    --split="test" \
    --bias_type="religion" \
    --methods="Occlusion"

python -m model_selection.compute_reliance_statistics \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results_civil/roberta_civil_all_religion_test_1000/cda" \
    --split="test" \
    --bias_type="religion" \
    --methods="Occlusion"

python -m sensitive_attribution_extraction.extract_sensitive_token_attribution \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results_civil/roberta_civil_religion_religion_test_1000/dropout" \
    --split="test" \
    --bias_type="religion" \
    --methods="Occlusion"

python -m model_selection.compute_reliance_statistics \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results_civil/roberta_civil_religion_religion_test_1000/dropout" \
    --split="test" \
    --bias_type="religion" \
    --methods="Occlusion"

python -m sensitive_attribution_extraction.extract_sensitive_token_attribution \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results_civil/roberta_civil_all_religion_test_1000/dropout" \
    --split="test" \
    --bias_type="religion" \
    --methods="Occlusion"

python -m model_selection.compute_reliance_statistics \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results_civil/roberta_civil_all_religion_test_1000/dropout" \
    --split="test" \
    --bias_type="religion" \
    --methods="Occlusion"

python -m sensitive_attribution_extraction.extract_sensitive_token_attribution \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results_civil/roberta_civil_religion_religion_test_1000/attention_entropy" \
    --split="test" \
    --bias_type="religion" \
    --methods="Occlusion"

python -m model_selection.compute_reliance_statistics \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results_civil/roberta_civil_religion_religion_test_1000/attention_entropy" \
    --split="test" \
    --bias_type="religion" \
    --methods="Occlusion"

python -m sensitive_attribution_extraction.extract_sensitive_token_attribution \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results_civil/roberta_civil_all_religion_test_1000/attention_entropy" \
    --split="test" \
    --bias_type="religion" \
    --methods="Occlusion"

python -m model_selection.compute_reliance_statistics \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results_civil/roberta_civil_all_religion_test_1000/attention_entropy" \
    --split="test" \
    --bias_type="religion" \
    --methods="Occlusion"

python -m sensitive_attribution_extraction.extract_sensitive_token_attribution \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results_civil/roberta_civil_religion_religion_test_1000/causal_debias" \
    --split="test" \
    --bias_type="religion" \
    --methods="Occlusion"

python -m model_selection.compute_reliance_statistics \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results_civil/roberta_civil_religion_religion_test_1000/causal_debias" \
    --split="test" \
    --bias_type="religion" \
    --methods="Occlusion"

python -m sensitive_attribution_extraction.extract_sensitive_token_attribution \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results_civil/roberta_civil_all_religion_test_1000/causal_debias" \
    --split="test" \
    --bias_type="religion" \
    --methods="Occlusion"

python -m model_selection.compute_reliance_statistics \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results_civil/roberta_civil_all_religion_test_1000/causal_debias" \
    --split="test" \
    --bias_type="religion" \
    --methods="Occlusion"

python -m sensitive_attribution_extraction.extract_sensitive_token_attribution \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results_civil/distilbert_civil_religion_religion_test_1000/no_debiasing" \
    --split="test" \
    --bias_type="religion" \
    --methods="Occlusion"

python -m model_selection.compute_reliance_statistics \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results_civil/distilbert_civil_religion_religion_test_1000/no_debiasing" \
    --split="test" \
    --bias_type="religion" \
    --methods="Occlusion"

python -m sensitive_attribution_extraction.extract_sensitive_token_attribution \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results_civil/distilbert_civil_all_religion_test_1000/no_debiasing" \
    --split="test" \
    --bias_type="religion" \
    --methods="Occlusion"

python -m model_selection.compute_reliance_statistics \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results_civil/distilbert_civil_all_religion_test_1000/no_debiasing" \
    --split="test" \
    --bias_type="religion" \
    --methods="Occlusion"

python -m sensitive_attribution_extraction.extract_sensitive_token_attribution \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results_civil/distilbert_civil_religion_religion_test_1000/group_balance" \
    --split="test" \
    --bias_type="religion" \
    --methods="Occlusion"

python -m model_selection.compute_reliance_statistics \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results_civil/distilbert_civil_religion_religion_test_1000/group_balance" \
    --split="test" \
    --bias_type="religion" \
    --methods="Occlusion"

python -m sensitive_attribution_extraction.extract_sensitive_token_attribution \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results_civil/distilbert_civil_all_religion_test_1000/group_balance" \
    --split="test" \
    --bias_type="religion" \
    --methods="Occlusion"

python -m model_selection.compute_reliance_statistics \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results_civil/distilbert_civil_all_religion_test_1000/group_balance" \
    --split="test" \
    --bias_type="religion" \
    --methods="Occlusion"

python -m sensitive_attribution_extraction.extract_sensitive_token_attribution \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results_civil/distilbert_civil_religion_religion_test_1000/group_class_balance" \
    --split="test" \
    --bias_type="religion" \
    --methods="Occlusion"

python -m model_selection.compute_reliance_statistics \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results_civil/distilbert_civil_religion_religion_test_1000/group_class_balance" \
    --split="test" \
    --bias_type="religion" \
    --methods="Occlusion"

python -m sensitive_attribution_extraction.extract_sensitive_token_attribution \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results_civil/distilbert_civil_all_religion_test_1000/group_class_balance" \
    --split="test" \
    --bias_type="religion" \
    --methods="Occlusion"

python -m model_selection.compute_reliance_statistics \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results_civil/distilbert_civil_all_religion_test_1000/group_class_balance" \
    --split="test" \
    --bias_type="religion" \
    --methods="Occlusion"

python -m sensitive_attribution_extraction.extract_sensitive_token_attribution \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results_civil/distilbert_civil_religion_religion_test_1000/cda" \
    --split="test" \
    --bias_type="religion" \
    --methods="Occlusion"

python -m model_selection.compute_reliance_statistics \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results_civil/distilbert_civil_religion_religion_test_1000/cda" \
    --split="test" \
    --bias_type="religion" \
    --methods="Occlusion"

python -m sensitive_attribution_extraction.extract_sensitive_token_attribution \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results_civil/distilbert_civil_all_religion_test_1000/cda" \
    --split="test" \
    --bias_type="religion" \
    --methods="Occlusion"

python -m model_selection.compute_reliance_statistics \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results_civil/distilbert_civil_all_religion_test_1000/cda" \
    --split="test" \
    --bias_type="religion" \
    --methods="Occlusion"

python -m sensitive_attribution_extraction.extract_sensitive_token_attribution \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results_civil/distilbert_civil_religion_religion_test_1000/dropout" \
    --split="test" \
    --bias_type="religion" \
    --methods="Occlusion"

python -m model_selection.compute_reliance_statistics \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results_civil/distilbert_civil_religion_religion_test_1000/dropout" \
    --split="test" \
    --bias_type="religion" \
    --methods="Occlusion"

python -m sensitive_attribution_extraction.extract_sensitive_token_attribution \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results_civil/distilbert_civil_all_religion_test_1000/dropout" \
    --split="test" \
    --bias_type="religion" \
    --methods="Occlusion"

python -m model_selection.compute_reliance_statistics \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results_civil/distilbert_civil_all_religion_test_1000/dropout" \
    --split="test" \
    --bias_type="religion" \
    --methods="Occlusion"

python -m sensitive_attribution_extraction.extract_sensitive_token_attribution \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results_civil/distilbert_civil_religion_religion_test_1000/attention_entropy" \
    --split="test" \
    --bias_type="religion" \
    --methods="Occlusion"

python -m model_selection.compute_reliance_statistics \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results_civil/distilbert_civil_religion_religion_test_1000/attention_entropy" \
    --split="test" \
    --bias_type="religion" \
    --methods="Occlusion"

python -m sensitive_attribution_extraction.extract_sensitive_token_attribution \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results_civil/distilbert_civil_all_religion_test_1000/attention_entropy" \
    --split="test" \
    --bias_type="religion" \
    --methods="Occlusion"

python -m model_selection.compute_reliance_statistics \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results_civil/distilbert_civil_all_religion_test_1000/attention_entropy" \
    --split="test" \
    --bias_type="religion" \
    --methods="Occlusion"

python -m sensitive_attribution_extraction.extract_sensitive_token_attribution \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results_civil/distilbert_civil_religion_religion_test_1000/causal_debias" \
    --split="test" \
    --bias_type="religion" \
    --methods="Occlusion"

python -m model_selection.compute_reliance_statistics \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results_civil/distilbert_civil_religion_religion_test_1000/causal_debias" \
    --split="test" \
    --bias_type="religion" \
    --methods="Occlusion"

python -m sensitive_attribution_extraction.extract_sensitive_token_attribution \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results_civil/distilbert_civil_all_religion_test_1000/causal_debias" \
    --split="test" \
    --bias_type="religion" \
    --methods="Occlusion"

python -m model_selection.compute_reliance_statistics \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results_civil/distilbert_civil_all_religion_test_1000/causal_debias" \
    --split="test" \
    --bias_type="religion" \
    --methods="Occlusion"


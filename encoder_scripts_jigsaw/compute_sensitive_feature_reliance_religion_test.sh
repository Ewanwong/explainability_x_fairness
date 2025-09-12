#!/bin/bash

python -m sensitive_attribution_extraction.extract_sensitive_token_attribution \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results_jigsaw/bert_jigsaw_religion_religion_test_200/no_debiasing" \
    --split="test" \
    --bias_type="religion" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m model_selection.compute_reliance_statistics \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results_jigsaw/bert_jigsaw_religion_religion_test_200/no_debiasing" \
    --split="test" \
    --bias_type="religion" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m sensitive_attribution_extraction.extract_sensitive_token_attribution \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results_jigsaw/bert_jigsaw_all_religion_test_200/no_debiasing" \
    --split="test" \
    --bias_type="religion" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m model_selection.compute_reliance_statistics \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results_jigsaw/bert_jigsaw_all_religion_test_200/no_debiasing" \
    --split="test" \
    --bias_type="religion" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m sensitive_attribution_extraction.extract_sensitive_token_attribution \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results_jigsaw/bert_jigsaw_religion_religion_test_200/group_balance" \
    --split="test" \
    --bias_type="religion" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m model_selection.compute_reliance_statistics \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results_jigsaw/bert_jigsaw_religion_religion_test_200/group_balance" \
    --split="test" \
    --bias_type="religion" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m sensitive_attribution_extraction.extract_sensitive_token_attribution \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results_jigsaw/bert_jigsaw_all_religion_test_200/group_balance" \
    --split="test" \
    --bias_type="religion" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m model_selection.compute_reliance_statistics \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results_jigsaw/bert_jigsaw_all_religion_test_200/group_balance" \
    --split="test" \
    --bias_type="religion" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m sensitive_attribution_extraction.extract_sensitive_token_attribution \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results_jigsaw/bert_jigsaw_religion_religion_test_200/group_class_balance" \
    --split="test" \
    --bias_type="religion" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m model_selection.compute_reliance_statistics \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results_jigsaw/bert_jigsaw_religion_religion_test_200/group_class_balance" \
    --split="test" \
    --bias_type="religion" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m sensitive_attribution_extraction.extract_sensitive_token_attribution \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results_jigsaw/bert_jigsaw_all_religion_test_200/group_class_balance" \
    --split="test" \
    --bias_type="religion" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m model_selection.compute_reliance_statistics \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results_jigsaw/bert_jigsaw_all_religion_test_200/group_class_balance" \
    --split="test" \
    --bias_type="religion" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m sensitive_attribution_extraction.extract_sensitive_token_attribution \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results_jigsaw/bert_jigsaw_religion_religion_test_200/cda" \
    --split="test" \
    --bias_type="religion" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m model_selection.compute_reliance_statistics \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results_jigsaw/bert_jigsaw_religion_religion_test_200/cda" \
    --split="test" \
    --bias_type="religion" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m sensitive_attribution_extraction.extract_sensitive_token_attribution \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results_jigsaw/bert_jigsaw_all_religion_test_200/cda" \
    --split="test" \
    --bias_type="religion" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m model_selection.compute_reliance_statistics \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results_jigsaw/bert_jigsaw_all_religion_test_200/cda" \
    --split="test" \
    --bias_type="religion" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m sensitive_attribution_extraction.extract_sensitive_token_attribution \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results_jigsaw/bert_jigsaw_religion_religion_test_200/dropout" \
    --split="test" \
    --bias_type="religion" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m model_selection.compute_reliance_statistics \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results_jigsaw/bert_jigsaw_religion_religion_test_200/dropout" \
    --split="test" \
    --bias_type="religion" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m sensitive_attribution_extraction.extract_sensitive_token_attribution \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results_jigsaw/bert_jigsaw_all_religion_test_200/dropout" \
    --split="test" \
    --bias_type="religion" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m model_selection.compute_reliance_statistics \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results_jigsaw/bert_jigsaw_all_religion_test_200/dropout" \
    --split="test" \
    --bias_type="religion" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m sensitive_attribution_extraction.extract_sensitive_token_attribution \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results_jigsaw/bert_jigsaw_religion_religion_test_200/attention_entropy" \
    --split="test" \
    --bias_type="religion" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m model_selection.compute_reliance_statistics \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results_jigsaw/bert_jigsaw_religion_religion_test_200/attention_entropy" \
    --split="test" \
    --bias_type="religion" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m sensitive_attribution_extraction.extract_sensitive_token_attribution \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results_jigsaw/bert_jigsaw_all_religion_test_200/attention_entropy" \
    --split="test" \
    --bias_type="religion" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m model_selection.compute_reliance_statistics \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results_jigsaw/bert_jigsaw_all_religion_test_200/attention_entropy" \
    --split="test" \
    --bias_type="religion" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m sensitive_attribution_extraction.extract_sensitive_token_attribution \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results_jigsaw/bert_jigsaw_religion_religion_test_200/causal_debias" \
    --split="test" \
    --bias_type="religion" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m model_selection.compute_reliance_statistics \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results_jigsaw/bert_jigsaw_religion_religion_test_200/causal_debias" \
    --split="test" \
    --bias_type="religion" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m sensitive_attribution_extraction.extract_sensitive_token_attribution \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results_jigsaw/bert_jigsaw_all_religion_test_200/causal_debias" \
    --split="test" \
    --bias_type="religion" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m model_selection.compute_reliance_statistics \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results_jigsaw/bert_jigsaw_all_religion_test_200/causal_debias" \
    --split="test" \
    --bias_type="religion" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m sensitive_attribution_extraction.extract_sensitive_token_attribution \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results_jigsaw/roberta_jigsaw_religion_religion_test_200/no_debiasing" \
    --split="test" \
    --bias_type="religion" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m model_selection.compute_reliance_statistics \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results_jigsaw/roberta_jigsaw_religion_religion_test_200/no_debiasing" \
    --split="test" \
    --bias_type="religion" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m sensitive_attribution_extraction.extract_sensitive_token_attribution \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results_jigsaw/roberta_jigsaw_all_religion_test_200/no_debiasing" \
    --split="test" \
    --bias_type="religion" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m model_selection.compute_reliance_statistics \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results_jigsaw/roberta_jigsaw_all_religion_test_200/no_debiasing" \
    --split="test" \
    --bias_type="religion" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m sensitive_attribution_extraction.extract_sensitive_token_attribution \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results_jigsaw/roberta_jigsaw_religion_religion_test_200/group_balance" \
    --split="test" \
    --bias_type="religion" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m model_selection.compute_reliance_statistics \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results_jigsaw/roberta_jigsaw_religion_religion_test_200/group_balance" \
    --split="test" \
    --bias_type="religion" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m sensitive_attribution_extraction.extract_sensitive_token_attribution \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results_jigsaw/roberta_jigsaw_all_religion_test_200/group_balance" \
    --split="test" \
    --bias_type="religion" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m model_selection.compute_reliance_statistics \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results_jigsaw/roberta_jigsaw_all_religion_test_200/group_balance" \
    --split="test" \
    --bias_type="religion" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m sensitive_attribution_extraction.extract_sensitive_token_attribution \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results_jigsaw/roberta_jigsaw_religion_religion_test_200/group_class_balance" \
    --split="test" \
    --bias_type="religion" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m model_selection.compute_reliance_statistics \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results_jigsaw/roberta_jigsaw_religion_religion_test_200/group_class_balance" \
    --split="test" \
    --bias_type="religion" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m sensitive_attribution_extraction.extract_sensitive_token_attribution \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results_jigsaw/roberta_jigsaw_all_religion_test_200/group_class_balance" \
    --split="test" \
    --bias_type="religion" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m model_selection.compute_reliance_statistics \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results_jigsaw/roberta_jigsaw_all_religion_test_200/group_class_balance" \
    --split="test" \
    --bias_type="religion" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m sensitive_attribution_extraction.extract_sensitive_token_attribution \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results_jigsaw/roberta_jigsaw_religion_religion_test_200/cda" \
    --split="test" \
    --bias_type="religion" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m model_selection.compute_reliance_statistics \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results_jigsaw/roberta_jigsaw_religion_religion_test_200/cda" \
    --split="test" \
    --bias_type="religion" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m sensitive_attribution_extraction.extract_sensitive_token_attribution \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results_jigsaw/roberta_jigsaw_all_religion_test_200/cda" \
    --split="test" \
    --bias_type="religion" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m model_selection.compute_reliance_statistics \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results_jigsaw/roberta_jigsaw_all_religion_test_200/cda" \
    --split="test" \
    --bias_type="religion" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m sensitive_attribution_extraction.extract_sensitive_token_attribution \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results_jigsaw/roberta_jigsaw_religion_religion_test_200/dropout" \
    --split="test" \
    --bias_type="religion" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m model_selection.compute_reliance_statistics \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results_jigsaw/roberta_jigsaw_religion_religion_test_200/dropout" \
    --split="test" \
    --bias_type="religion" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m sensitive_attribution_extraction.extract_sensitive_token_attribution \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results_jigsaw/roberta_jigsaw_all_religion_test_200/dropout" \
    --split="test" \
    --bias_type="religion" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m model_selection.compute_reliance_statistics \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results_jigsaw/roberta_jigsaw_all_religion_test_200/dropout" \
    --split="test" \
    --bias_type="religion" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m sensitive_attribution_extraction.extract_sensitive_token_attribution \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results_jigsaw/roberta_jigsaw_religion_religion_test_200/attention_entropy" \
    --split="test" \
    --bias_type="religion" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m model_selection.compute_reliance_statistics \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results_jigsaw/roberta_jigsaw_religion_religion_test_200/attention_entropy" \
    --split="test" \
    --bias_type="religion" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m sensitive_attribution_extraction.extract_sensitive_token_attribution \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results_jigsaw/roberta_jigsaw_all_religion_test_200/attention_entropy" \
    --split="test" \
    --bias_type="religion" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m model_selection.compute_reliance_statistics \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results_jigsaw/roberta_jigsaw_all_religion_test_200/attention_entropy" \
    --split="test" \
    --bias_type="religion" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m sensitive_attribution_extraction.extract_sensitive_token_attribution \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results_jigsaw/roberta_jigsaw_religion_religion_test_200/causal_debias" \
    --split="test" \
    --bias_type="religion" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m model_selection.compute_reliance_statistics \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results_jigsaw/roberta_jigsaw_religion_religion_test_200/causal_debias" \
    --split="test" \
    --bias_type="religion" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m sensitive_attribution_extraction.extract_sensitive_token_attribution \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results_jigsaw/roberta_jigsaw_all_religion_test_200/causal_debias" \
    --split="test" \
    --bias_type="religion" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m model_selection.compute_reliance_statistics \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results_jigsaw/roberta_jigsaw_all_religion_test_200/causal_debias" \
    --split="test" \
    --bias_type="religion" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m sensitive_attribution_extraction.extract_sensitive_token_attribution \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results_jigsaw/distilbert_jigsaw_religion_religion_test_200/no_debiasing" \
    --split="test" \
    --bias_type="religion" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m model_selection.compute_reliance_statistics \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results_jigsaw/distilbert_jigsaw_religion_religion_test_200/no_debiasing" \
    --split="test" \
    --bias_type="religion" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m sensitive_attribution_extraction.extract_sensitive_token_attribution \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results_jigsaw/distilbert_jigsaw_all_religion_test_200/no_debiasing" \
    --split="test" \
    --bias_type="religion" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m model_selection.compute_reliance_statistics \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results_jigsaw/distilbert_jigsaw_all_religion_test_200/no_debiasing" \
    --split="test" \
    --bias_type="religion" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m sensitive_attribution_extraction.extract_sensitive_token_attribution \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results_jigsaw/distilbert_jigsaw_religion_religion_test_200/group_balance" \
    --split="test" \
    --bias_type="religion" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m model_selection.compute_reliance_statistics \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results_jigsaw/distilbert_jigsaw_religion_religion_test_200/group_balance" \
    --split="test" \
    --bias_type="religion" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m sensitive_attribution_extraction.extract_sensitive_token_attribution \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results_jigsaw/distilbert_jigsaw_all_religion_test_200/group_balance" \
    --split="test" \
    --bias_type="religion" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m model_selection.compute_reliance_statistics \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results_jigsaw/distilbert_jigsaw_all_religion_test_200/group_balance" \
    --split="test" \
    --bias_type="religion" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m sensitive_attribution_extraction.extract_sensitive_token_attribution \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results_jigsaw/distilbert_jigsaw_religion_religion_test_200/group_class_balance" \
    --split="test" \
    --bias_type="religion" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m model_selection.compute_reliance_statistics \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results_jigsaw/distilbert_jigsaw_religion_religion_test_200/group_class_balance" \
    --split="test" \
    --bias_type="religion" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m sensitive_attribution_extraction.extract_sensitive_token_attribution \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results_jigsaw/distilbert_jigsaw_all_religion_test_200/group_class_balance" \
    --split="test" \
    --bias_type="religion" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m model_selection.compute_reliance_statistics \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results_jigsaw/distilbert_jigsaw_all_religion_test_200/group_class_balance" \
    --split="test" \
    --bias_type="religion" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m sensitive_attribution_extraction.extract_sensitive_token_attribution \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results_jigsaw/distilbert_jigsaw_religion_religion_test_200/cda" \
    --split="test" \
    --bias_type="religion" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m model_selection.compute_reliance_statistics \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results_jigsaw/distilbert_jigsaw_religion_religion_test_200/cda" \
    --split="test" \
    --bias_type="religion" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m sensitive_attribution_extraction.extract_sensitive_token_attribution \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results_jigsaw/distilbert_jigsaw_all_religion_test_200/cda" \
    --split="test" \
    --bias_type="religion" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m model_selection.compute_reliance_statistics \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results_jigsaw/distilbert_jigsaw_all_religion_test_200/cda" \
    --split="test" \
    --bias_type="religion" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m sensitive_attribution_extraction.extract_sensitive_token_attribution \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results_jigsaw/distilbert_jigsaw_religion_religion_test_200/dropout" \
    --split="test" \
    --bias_type="religion" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m model_selection.compute_reliance_statistics \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results_jigsaw/distilbert_jigsaw_religion_religion_test_200/dropout" \
    --split="test" \
    --bias_type="religion" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m sensitive_attribution_extraction.extract_sensitive_token_attribution \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results_jigsaw/distilbert_jigsaw_all_religion_test_200/dropout" \
    --split="test" \
    --bias_type="religion" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m model_selection.compute_reliance_statistics \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results_jigsaw/distilbert_jigsaw_all_religion_test_200/dropout" \
    --split="test" \
    --bias_type="religion" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m sensitive_attribution_extraction.extract_sensitive_token_attribution \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results_jigsaw/distilbert_jigsaw_religion_religion_test_200/attention_entropy" \
    --split="test" \
    --bias_type="religion" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m model_selection.compute_reliance_statistics \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results_jigsaw/distilbert_jigsaw_religion_religion_test_200/attention_entropy" \
    --split="test" \
    --bias_type="religion" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m sensitive_attribution_extraction.extract_sensitive_token_attribution \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results_jigsaw/distilbert_jigsaw_all_religion_test_200/attention_entropy" \
    --split="test" \
    --bias_type="religion" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m model_selection.compute_reliance_statistics \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results_jigsaw/distilbert_jigsaw_all_religion_test_200/attention_entropy" \
    --split="test" \
    --bias_type="religion" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m sensitive_attribution_extraction.extract_sensitive_token_attribution \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results_jigsaw/distilbert_jigsaw_religion_religion_test_200/causal_debias" \
    --split="test" \
    --bias_type="religion" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m model_selection.compute_reliance_statistics \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results_jigsaw/distilbert_jigsaw_religion_religion_test_200/causal_debias" \
    --split="test" \
    --bias_type="religion" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m sensitive_attribution_extraction.extract_sensitive_token_attribution \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results_jigsaw/distilbert_jigsaw_all_religion_test_200/causal_debias" \
    --split="test" \
    --bias_type="religion" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m model_selection.compute_reliance_statistics \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results_jigsaw/distilbert_jigsaw_all_religion_test_200/causal_debias" \
    --split="test" \
    --bias_type="religion" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"


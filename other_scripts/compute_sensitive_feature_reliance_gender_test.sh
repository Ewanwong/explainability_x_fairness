#!/bin/bash
python -m sensitive_attribution_extraction.extract_sensitive_token_attribution \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/bert_civil_all_gender_test_2000/dropout" \
    --split="test" \
    --bias_type="gender" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m model_selection.compute_reliance_statistics \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/bert_civil_all_gender_test_2000/dropout" \
    --split="test" \
    --bias_type="gender" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m sensitive_attribution_extraction.extract_sensitive_token_attribution \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/bert_civil_all_gender_test_2000/attention_entropy" \
    --split="test" \
    --bias_type="gender" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m model_selection.compute_reliance_statistics \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/bert_civil_all_gender_test_2000/attention_entropy" \
    --split="test" \
    --bias_type="gender" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m sensitive_attribution_extraction.extract_sensitive_token_attribution \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/bert_civil_all_gender_test_2000/causal_debias" \
    --split="test" \
    --bias_type="gender" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m model_selection.compute_reliance_statistics \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/bert_civil_all_gender_test_2000/causal_debias" \
    --split="test" \
    --bias_type="gender" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m sensitive_attribution_extraction.extract_sensitive_token_attribution \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/roberta_civil_all_gender_test_2000/dropout" \
    --split="test" \
    --bias_type="gender" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m model_selection.compute_reliance_statistics \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/roberta_civil_all_gender_test_2000/dropout" \
    --split="test" \
    --bias_type="gender" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m sensitive_attribution_extraction.extract_sensitive_token_attribution \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/roberta_civil_all_gender_test_2000/attention_entropy" \
    --split="test" \
    --bias_type="gender" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m model_selection.compute_reliance_statistics \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/roberta_civil_all_gender_test_2000/attention_entropy" \
    --split="test" \
    --bias_type="gender" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m sensitive_attribution_extraction.extract_sensitive_token_attribution \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/roberta_civil_all_gender_test_2000/causal_debias" \
    --split="test" \
    --bias_type="gender" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m model_selection.compute_reliance_statistics \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/roberta_civil_all_gender_test_2000/causal_debias" \
    --split="test" \
    --bias_type="gender" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m sensitive_attribution_extraction.extract_sensitive_token_attribution \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/distilbert_civil_all_gender_test_2000/dropout" \
    --split="test" \
    --bias_type="gender" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m model_selection.compute_reliance_statistics \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/distilbert_civil_all_gender_test_2000/dropout" \
    --split="test" \
    --bias_type="gender" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m sensitive_attribution_extraction.extract_sensitive_token_attribution \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/distilbert_civil_all_gender_test_2000/attention_entropy" \
    --split="test" \
    --bias_type="gender" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m model_selection.compute_reliance_statistics \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/distilbert_civil_all_gender_test_2000/attention_entropy" \
    --split="test" \
    --bias_type="gender" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m sensitive_attribution_extraction.extract_sensitive_token_attribution \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/distilbert_civil_all_gender_test_2000/causal_debias" \
    --split="test" \
    --bias_type="gender" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m model_selection.compute_reliance_statistics \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/distilbert_civil_all_gender_test_2000/causal_debias" \
    --split="test" \
    --bias_type="gender" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"


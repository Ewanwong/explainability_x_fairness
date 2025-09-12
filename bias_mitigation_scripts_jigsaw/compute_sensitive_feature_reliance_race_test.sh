#!/bin/bash

python -m sensitive_attribution_extraction.extract_sensitive_token_attribution \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/bias_mitigation_results_civil/bert_civil_race_race_test_2000/Saliency/L1_1.0" \
    --split="test" \
    --bias_type="race" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m model_selection.compute_reliance_statistics \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/bias_mitigation_results_civil/bert_civil_race_race_test_2000/Saliency/L1_1.0" \
    --split="test" \
    --bias_type="race" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m sensitive_attribution_extraction.extract_sensitive_token_attribution \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/bias_mitigation_results_civil/bert_civil_race_race_test_2000/Saliency/L1_0.1" \
    --split="test" \
    --bias_type="race" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m model_selection.compute_reliance_statistics \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/bias_mitigation_results_civil/bert_civil_race_race_test_2000/Saliency/L1_0.1" \
    --split="test" \
    --bias_type="race" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m sensitive_attribution_extraction.extract_sensitive_token_attribution \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/bias_mitigation_results_civil/bert_civil_race_race_test_2000/Saliency/L1_0.01" \
    --split="test" \
    --bias_type="race" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m model_selection.compute_reliance_statistics \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/bias_mitigation_results_civil/bert_civil_race_race_test_2000/Saliency/L1_0.01" \
    --split="test" \
    --bias_type="race" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m sensitive_attribution_extraction.extract_sensitive_token_attribution \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/bias_mitigation_results_civil/bert_civil_race_race_test_2000/Saliency/L1_0.001" \
    --split="test" \
    --bias_type="race" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m model_selection.compute_reliance_statistics \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/bias_mitigation_results_civil/bert_civil_race_race_test_2000/Saliency/L1_0.001" \
    --split="test" \
    --bias_type="race" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m sensitive_attribution_extraction.extract_sensitive_token_attribution \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/bias_mitigation_results_civil/bert_civil_race_race_test_2000/Saliency/L1_0.0001" \
    --split="test" \
    --bias_type="race" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m model_selection.compute_reliance_statistics \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/bias_mitigation_results_civil/bert_civil_race_race_test_2000/Saliency/L1_0.0001" \
    --split="test" \
    --bias_type="race" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m sensitive_attribution_extraction.extract_sensitive_token_attribution \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/bias_mitigation_results_civil/bert_civil_race_race_test_2000/Saliency/L2_1.0" \
    --split="test" \
    --bias_type="race" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m model_selection.compute_reliance_statistics \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/bias_mitigation_results_civil/bert_civil_race_race_test_2000/Saliency/L2_1.0" \
    --split="test" \
    --bias_type="race" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m sensitive_attribution_extraction.extract_sensitive_token_attribution \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/bias_mitigation_results_civil/bert_civil_race_race_test_2000/Saliency/L2_0.1" \
    --split="test" \
    --bias_type="race" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m model_selection.compute_reliance_statistics \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/bias_mitigation_results_civil/bert_civil_race_race_test_2000/Saliency/L2_0.1" \
    --split="test" \
    --bias_type="race" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m sensitive_attribution_extraction.extract_sensitive_token_attribution \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/bias_mitigation_results_civil/bert_civil_race_race_test_2000/Saliency/L2_0.01" \
    --split="test" \
    --bias_type="race" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m model_selection.compute_reliance_statistics \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/bias_mitigation_results_civil/bert_civil_race_race_test_2000/Saliency/L2_0.01" \
    --split="test" \
    --bias_type="race" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m sensitive_attribution_extraction.extract_sensitive_token_attribution \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/bias_mitigation_results_civil/bert_civil_race_race_test_2000/Saliency/L2_0.001" \
    --split="test" \
    --bias_type="race" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m model_selection.compute_reliance_statistics \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/bias_mitigation_results_civil/bert_civil_race_race_test_2000/Saliency/L2_0.001" \
    --split="test" \
    --bias_type="race" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m sensitive_attribution_extraction.extract_sensitive_token_attribution \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/bias_mitigation_results_civil/bert_civil_race_race_test_2000/Saliency/L2_0.0001" \
    --split="test" \
    --bias_type="race" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m model_selection.compute_reliance_statistics \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/bias_mitigation_results_civil/bert_civil_race_race_test_2000/Saliency/L2_0.0001" \
    --split="test" \
    --bias_type="race" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m sensitive_attribution_extraction.extract_sensitive_token_attribution \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/bias_mitigation_results_civil/bert_civil_race_race_test_2000/InputXGradient/L1_1.0" \
    --split="test" \
    --bias_type="race" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m model_selection.compute_reliance_statistics \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/bias_mitigation_results_civil/bert_civil_race_race_test_2000/InputXGradient/L1_1.0" \
    --split="test" \
    --bias_type="race" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m sensitive_attribution_extraction.extract_sensitive_token_attribution \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/bias_mitigation_results_civil/bert_civil_race_race_test_2000/InputXGradient/L1_0.1" \
    --split="test" \
    --bias_type="race" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m model_selection.compute_reliance_statistics \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/bias_mitigation_results_civil/bert_civil_race_race_test_2000/InputXGradient/L1_0.1" \
    --split="test" \
    --bias_type="race" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m sensitive_attribution_extraction.extract_sensitive_token_attribution \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/bias_mitigation_results_civil/bert_civil_race_race_test_2000/InputXGradient/L1_0.01" \
    --split="test" \
    --bias_type="race" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m model_selection.compute_reliance_statistics \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/bias_mitigation_results_civil/bert_civil_race_race_test_2000/InputXGradient/L1_0.01" \
    --split="test" \
    --bias_type="race" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m sensitive_attribution_extraction.extract_sensitive_token_attribution \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/bias_mitigation_results_civil/bert_civil_race_race_test_2000/InputXGradient/L1_0.001" \
    --split="test" \
    --bias_type="race" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m model_selection.compute_reliance_statistics \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/bias_mitigation_results_civil/bert_civil_race_race_test_2000/InputXGradient/L1_0.001" \
    --split="test" \
    --bias_type="race" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m sensitive_attribution_extraction.extract_sensitive_token_attribution \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/bias_mitigation_results_civil/bert_civil_race_race_test_2000/InputXGradient/L1_0.0001" \
    --split="test" \
    --bias_type="race" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m model_selection.compute_reliance_statistics \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/bias_mitigation_results_civil/bert_civil_race_race_test_2000/InputXGradient/L1_0.0001" \
    --split="test" \
    --bias_type="race" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m sensitive_attribution_extraction.extract_sensitive_token_attribution \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/bias_mitigation_results_civil/bert_civil_race_race_test_2000/InputXGradient/L2_1.0" \
    --split="test" \
    --bias_type="race" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m model_selection.compute_reliance_statistics \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/bias_mitigation_results_civil/bert_civil_race_race_test_2000/InputXGradient/L2_1.0" \
    --split="test" \
    --bias_type="race" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m sensitive_attribution_extraction.extract_sensitive_token_attribution \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/bias_mitigation_results_civil/bert_civil_race_race_test_2000/InputXGradient/L2_0.1" \
    --split="test" \
    --bias_type="race" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m model_selection.compute_reliance_statistics \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/bias_mitigation_results_civil/bert_civil_race_race_test_2000/InputXGradient/L2_0.1" \
    --split="test" \
    --bias_type="race" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m sensitive_attribution_extraction.extract_sensitive_token_attribution \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/bias_mitigation_results_civil/bert_civil_race_race_test_2000/InputXGradient/L2_0.01" \
    --split="test" \
    --bias_type="race" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m model_selection.compute_reliance_statistics \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/bias_mitigation_results_civil/bert_civil_race_race_test_2000/InputXGradient/L2_0.01" \
    --split="test" \
    --bias_type="race" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m sensitive_attribution_extraction.extract_sensitive_token_attribution \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/bias_mitigation_results_civil/bert_civil_race_race_test_2000/InputXGradient/L2_0.001" \
    --split="test" \
    --bias_type="race" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m model_selection.compute_reliance_statistics \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/bias_mitigation_results_civil/bert_civil_race_race_test_2000/InputXGradient/L2_0.001" \
    --split="test" \
    --bias_type="race" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m sensitive_attribution_extraction.extract_sensitive_token_attribution \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/bias_mitigation_results_civil/bert_civil_race_race_test_2000/InputXGradient/L2_0.0001" \
    --split="test" \
    --bias_type="race" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m model_selection.compute_reliance_statistics \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/bias_mitigation_results_civil/bert_civil_race_race_test_2000/InputXGradient/L2_0.0001" \
    --split="test" \
    --bias_type="race" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m sensitive_attribution_extraction.extract_sensitive_token_attribution \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/bias_mitigation_results_civil/bert_civil_race_race_test_2000/IntegratedGradients/L1_1.0" \
    --split="test" \
    --bias_type="race" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m model_selection.compute_reliance_statistics \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/bias_mitigation_results_civil/bert_civil_race_race_test_2000/IntegratedGradients/L1_1.0" \
    --split="test" \
    --bias_type="race" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m sensitive_attribution_extraction.extract_sensitive_token_attribution \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/bias_mitigation_results_civil/bert_civil_race_race_test_2000/IntegratedGradients/L1_0.1" \
    --split="test" \
    --bias_type="race" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m model_selection.compute_reliance_statistics \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/bias_mitigation_results_civil/bert_civil_race_race_test_2000/IntegratedGradients/L1_0.1" \
    --split="test" \
    --bias_type="race" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m sensitive_attribution_extraction.extract_sensitive_token_attribution \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/bias_mitigation_results_civil/bert_civil_race_race_test_2000/IntegratedGradients/L1_0.01" \
    --split="test" \
    --bias_type="race" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m model_selection.compute_reliance_statistics \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/bias_mitigation_results_civil/bert_civil_race_race_test_2000/IntegratedGradients/L1_0.01" \
    --split="test" \
    --bias_type="race" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m sensitive_attribution_extraction.extract_sensitive_token_attribution \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/bias_mitigation_results_civil/bert_civil_race_race_test_2000/IntegratedGradients/L1_0.001" \
    --split="test" \
    --bias_type="race" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m model_selection.compute_reliance_statistics \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/bias_mitigation_results_civil/bert_civil_race_race_test_2000/IntegratedGradients/L1_0.001" \
    --split="test" \
    --bias_type="race" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m sensitive_attribution_extraction.extract_sensitive_token_attribution \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/bias_mitigation_results_civil/bert_civil_race_race_test_2000/IntegratedGradients/L1_0.0001" \
    --split="test" \
    --bias_type="race" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m model_selection.compute_reliance_statistics \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/bias_mitigation_results_civil/bert_civil_race_race_test_2000/IntegratedGradients/L1_0.0001" \
    --split="test" \
    --bias_type="race" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m sensitive_attribution_extraction.extract_sensitive_token_attribution \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/bias_mitigation_results_civil/bert_civil_race_race_test_2000/IntegratedGradients/L2_1.0" \
    --split="test" \
    --bias_type="race" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m model_selection.compute_reliance_statistics \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/bias_mitigation_results_civil/bert_civil_race_race_test_2000/IntegratedGradients/L2_1.0" \
    --split="test" \
    --bias_type="race" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m sensitive_attribution_extraction.extract_sensitive_token_attribution \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/bias_mitigation_results_civil/bert_civil_race_race_test_2000/IntegratedGradients/L2_0.1" \
    --split="test" \
    --bias_type="race" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m model_selection.compute_reliance_statistics \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/bias_mitigation_results_civil/bert_civil_race_race_test_2000/IntegratedGradients/L2_0.1" \
    --split="test" \
    --bias_type="race" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m sensitive_attribution_extraction.extract_sensitive_token_attribution \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/bias_mitigation_results_civil/bert_civil_race_race_test_2000/IntegratedGradients/L2_0.01" \
    --split="test" \
    --bias_type="race" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m model_selection.compute_reliance_statistics \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/bias_mitigation_results_civil/bert_civil_race_race_test_2000/IntegratedGradients/L2_0.01" \
    --split="test" \
    --bias_type="race" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m sensitive_attribution_extraction.extract_sensitive_token_attribution \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/bias_mitigation_results_civil/bert_civil_race_race_test_2000/IntegratedGradients/L2_0.001" \
    --split="test" \
    --bias_type="race" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m model_selection.compute_reliance_statistics \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/bias_mitigation_results_civil/bert_civil_race_race_test_2000/IntegratedGradients/L2_0.001" \
    --split="test" \
    --bias_type="race" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m sensitive_attribution_extraction.extract_sensitive_token_attribution \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/bias_mitigation_results_civil/bert_civil_race_race_test_2000/IntegratedGradients/L2_0.0001" \
    --split="test" \
    --bias_type="race" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m model_selection.compute_reliance_statistics \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/bias_mitigation_results_civil/bert_civil_race_race_test_2000/IntegratedGradients/L2_0.0001" \
    --split="test" \
    --bias_type="race" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m sensitive_attribution_extraction.extract_sensitive_token_attribution \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/bias_mitigation_results_civil/bert_civil_race_race_test_2000/raw_attention/L1_1.0" \
    --split="test" \
    --bias_type="race" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m model_selection.compute_reliance_statistics \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/bias_mitigation_results_civil/bert_civil_race_race_test_2000/raw_attention/L1_1.0" \
    --split="test" \
    --bias_type="race" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m sensitive_attribution_extraction.extract_sensitive_token_attribution \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/bias_mitigation_results_civil/bert_civil_race_race_test_2000/raw_attention/L1_0.1" \
    --split="test" \
    --bias_type="race" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m model_selection.compute_reliance_statistics \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/bias_mitigation_results_civil/bert_civil_race_race_test_2000/raw_attention/L1_0.1" \
    --split="test" \
    --bias_type="race" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m sensitive_attribution_extraction.extract_sensitive_token_attribution \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/bias_mitigation_results_civil/bert_civil_race_race_test_2000/raw_attention/L1_0.01" \
    --split="test" \
    --bias_type="race" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m model_selection.compute_reliance_statistics \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/bias_mitigation_results_civil/bert_civil_race_race_test_2000/raw_attention/L1_0.01" \
    --split="test" \
    --bias_type="race" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m sensitive_attribution_extraction.extract_sensitive_token_attribution \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/bias_mitigation_results_civil/bert_civil_race_race_test_2000/raw_attention/L1_0.001" \
    --split="test" \
    --bias_type="race" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m model_selection.compute_reliance_statistics \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/bias_mitigation_results_civil/bert_civil_race_race_test_2000/raw_attention/L1_0.001" \
    --split="test" \
    --bias_type="race" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m sensitive_attribution_extraction.extract_sensitive_token_attribution \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/bias_mitigation_results_civil/bert_civil_race_race_test_2000/raw_attention/L1_0.0001" \
    --split="test" \
    --bias_type="race" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m model_selection.compute_reliance_statistics \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/bias_mitigation_results_civil/bert_civil_race_race_test_2000/raw_attention/L1_0.0001" \
    --split="test" \
    --bias_type="race" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m sensitive_attribution_extraction.extract_sensitive_token_attribution \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/bias_mitigation_results_civil/bert_civil_race_race_test_2000/raw_attention/L2_1.0" \
    --split="test" \
    --bias_type="race" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m model_selection.compute_reliance_statistics \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/bias_mitigation_results_civil/bert_civil_race_race_test_2000/raw_attention/L2_1.0" \
    --split="test" \
    --bias_type="race" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m sensitive_attribution_extraction.extract_sensitive_token_attribution \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/bias_mitigation_results_civil/bert_civil_race_race_test_2000/raw_attention/L2_0.1" \
    --split="test" \
    --bias_type="race" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m model_selection.compute_reliance_statistics \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/bias_mitigation_results_civil/bert_civil_race_race_test_2000/raw_attention/L2_0.1" \
    --split="test" \
    --bias_type="race" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m sensitive_attribution_extraction.extract_sensitive_token_attribution \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/bias_mitigation_results_civil/bert_civil_race_race_test_2000/raw_attention/L2_0.01" \
    --split="test" \
    --bias_type="race" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m model_selection.compute_reliance_statistics \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/bias_mitigation_results_civil/bert_civil_race_race_test_2000/raw_attention/L2_0.01" \
    --split="test" \
    --bias_type="race" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m sensitive_attribution_extraction.extract_sensitive_token_attribution \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/bias_mitigation_results_civil/bert_civil_race_race_test_2000/raw_attention/L2_0.001" \
    --split="test" \
    --bias_type="race" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m model_selection.compute_reliance_statistics \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/bias_mitigation_results_civil/bert_civil_race_race_test_2000/raw_attention/L2_0.001" \
    --split="test" \
    --bias_type="race" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m sensitive_attribution_extraction.extract_sensitive_token_attribution \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/bias_mitigation_results_civil/bert_civil_race_race_test_2000/raw_attention/L2_0.0001" \
    --split="test" \
    --bias_type="race" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m model_selection.compute_reliance_statistics \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/bias_mitigation_results_civil/bert_civil_race_race_test_2000/raw_attention/L2_0.0001" \
    --split="test" \
    --bias_type="race" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m sensitive_attribution_extraction.extract_sensitive_token_attribution \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/bias_mitigation_results_civil/bert_civil_race_race_test_2000/attention_flow/L1_1.0" \
    --split="test" \
    --bias_type="race" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m model_selection.compute_reliance_statistics \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/bias_mitigation_results_civil/bert_civil_race_race_test_2000/attention_flow/L1_1.0" \
    --split="test" \
    --bias_type="race" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m sensitive_attribution_extraction.extract_sensitive_token_attribution \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/bias_mitigation_results_civil/bert_civil_race_race_test_2000/attention_flow/L1_0.1" \
    --split="test" \
    --bias_type="race" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m model_selection.compute_reliance_statistics \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/bias_mitigation_results_civil/bert_civil_race_race_test_2000/attention_flow/L1_0.1" \
    --split="test" \
    --bias_type="race" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m sensitive_attribution_extraction.extract_sensitive_token_attribution \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/bias_mitigation_results_civil/bert_civil_race_race_test_2000/attention_flow/L1_0.01" \
    --split="test" \
    --bias_type="race" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m model_selection.compute_reliance_statistics \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/bias_mitigation_results_civil/bert_civil_race_race_test_2000/attention_flow/L1_0.01" \
    --split="test" \
    --bias_type="race" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m sensitive_attribution_extraction.extract_sensitive_token_attribution \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/bias_mitigation_results_civil/bert_civil_race_race_test_2000/attention_flow/L1_0.001" \
    --split="test" \
    --bias_type="race" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m model_selection.compute_reliance_statistics \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/bias_mitigation_results_civil/bert_civil_race_race_test_2000/attention_flow/L1_0.001" \
    --split="test" \
    --bias_type="race" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m sensitive_attribution_extraction.extract_sensitive_token_attribution \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/bias_mitigation_results_civil/bert_civil_race_race_test_2000/attention_flow/L1_0.0001" \
    --split="test" \
    --bias_type="race" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m model_selection.compute_reliance_statistics \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/bias_mitigation_results_civil/bert_civil_race_race_test_2000/attention_flow/L1_0.0001" \
    --split="test" \
    --bias_type="race" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m sensitive_attribution_extraction.extract_sensitive_token_attribution \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/bias_mitigation_results_civil/bert_civil_race_race_test_2000/attention_flow/L2_1.0" \
    --split="test" \
    --bias_type="race" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m model_selection.compute_reliance_statistics \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/bias_mitigation_results_civil/bert_civil_race_race_test_2000/attention_flow/L2_1.0" \
    --split="test" \
    --bias_type="race" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m sensitive_attribution_extraction.extract_sensitive_token_attribution \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/bias_mitigation_results_civil/bert_civil_race_race_test_2000/attention_flow/L2_0.1" \
    --split="test" \
    --bias_type="race" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m model_selection.compute_reliance_statistics \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/bias_mitigation_results_civil/bert_civil_race_race_test_2000/attention_flow/L2_0.1" \
    --split="test" \
    --bias_type="race" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m sensitive_attribution_extraction.extract_sensitive_token_attribution \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/bias_mitigation_results_civil/bert_civil_race_race_test_2000/attention_flow/L2_0.01" \
    --split="test" \
    --bias_type="race" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m model_selection.compute_reliance_statistics \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/bias_mitigation_results_civil/bert_civil_race_race_test_2000/attention_flow/L2_0.01" \
    --split="test" \
    --bias_type="race" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m sensitive_attribution_extraction.extract_sensitive_token_attribution \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/bias_mitigation_results_civil/bert_civil_race_race_test_2000/attention_flow/L2_0.001" \
    --split="test" \
    --bias_type="race" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m model_selection.compute_reliance_statistics \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/bias_mitigation_results_civil/bert_civil_race_race_test_2000/attention_flow/L2_0.001" \
    --split="test" \
    --bias_type="race" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m sensitive_attribution_extraction.extract_sensitive_token_attribution \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/bias_mitigation_results_civil/bert_civil_race_race_test_2000/attention_flow/L2_0.0001" \
    --split="test" \
    --bias_type="race" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m model_selection.compute_reliance_statistics \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/bias_mitigation_results_civil/bert_civil_race_race_test_2000/attention_flow/L2_0.0001" \
    --split="test" \
    --bias_type="race" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m sensitive_attribution_extraction.extract_sensitive_token_attribution \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/bias_mitigation_results_civil/bert_civil_race_race_test_2000/attention_rollout/L1_1.0" \
    --split="test" \
    --bias_type="race" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m model_selection.compute_reliance_statistics \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/bias_mitigation_results_civil/bert_civil_race_race_test_2000/attention_rollout/L1_1.0" \
    --split="test" \
    --bias_type="race" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m sensitive_attribution_extraction.extract_sensitive_token_attribution \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/bias_mitigation_results_civil/bert_civil_race_race_test_2000/attention_rollout/L1_0.1" \
    --split="test" \
    --bias_type="race" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m model_selection.compute_reliance_statistics \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/bias_mitigation_results_civil/bert_civil_race_race_test_2000/attention_rollout/L1_0.1" \
    --split="test" \
    --bias_type="race" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m sensitive_attribution_extraction.extract_sensitive_token_attribution \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/bias_mitigation_results_civil/bert_civil_race_race_test_2000/attention_rollout/L1_0.01" \
    --split="test" \
    --bias_type="race" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m model_selection.compute_reliance_statistics \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/bias_mitigation_results_civil/bert_civil_race_race_test_2000/attention_rollout/L1_0.01" \
    --split="test" \
    --bias_type="race" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m sensitive_attribution_extraction.extract_sensitive_token_attribution \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/bias_mitigation_results_civil/bert_civil_race_race_test_2000/attention_rollout/L1_0.001" \
    --split="test" \
    --bias_type="race" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m model_selection.compute_reliance_statistics \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/bias_mitigation_results_civil/bert_civil_race_race_test_2000/attention_rollout/L1_0.001" \
    --split="test" \
    --bias_type="race" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m sensitive_attribution_extraction.extract_sensitive_token_attribution \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/bias_mitigation_results_civil/bert_civil_race_race_test_2000/attention_rollout/L1_0.0001" \
    --split="test" \
    --bias_type="race" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m model_selection.compute_reliance_statistics \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/bias_mitigation_results_civil/bert_civil_race_race_test_2000/attention_rollout/L1_0.0001" \
    --split="test" \
    --bias_type="race" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m sensitive_attribution_extraction.extract_sensitive_token_attribution \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/bias_mitigation_results_civil/bert_civil_race_race_test_2000/attention_rollout/L2_1.0" \
    --split="test" \
    --bias_type="race" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m model_selection.compute_reliance_statistics \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/bias_mitigation_results_civil/bert_civil_race_race_test_2000/attention_rollout/L2_1.0" \
    --split="test" \
    --bias_type="race" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m sensitive_attribution_extraction.extract_sensitive_token_attribution \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/bias_mitigation_results_civil/bert_civil_race_race_test_2000/attention_rollout/L2_0.1" \
    --split="test" \
    --bias_type="race" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m model_selection.compute_reliance_statistics \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/bias_mitigation_results_civil/bert_civil_race_race_test_2000/attention_rollout/L2_0.1" \
    --split="test" \
    --bias_type="race" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m sensitive_attribution_extraction.extract_sensitive_token_attribution \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/bias_mitigation_results_civil/bert_civil_race_race_test_2000/attention_rollout/L2_0.01" \
    --split="test" \
    --bias_type="race" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m model_selection.compute_reliance_statistics \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/bias_mitigation_results_civil/bert_civil_race_race_test_2000/attention_rollout/L2_0.01" \
    --split="test" \
    --bias_type="race" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m sensitive_attribution_extraction.extract_sensitive_token_attribution \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/bias_mitigation_results_civil/bert_civil_race_race_test_2000/attention_rollout/L2_0.001" \
    --split="test" \
    --bias_type="race" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m model_selection.compute_reliance_statistics \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/bias_mitigation_results_civil/bert_civil_race_race_test_2000/attention_rollout/L2_0.001" \
    --split="test" \
    --bias_type="race" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m sensitive_attribution_extraction.extract_sensitive_token_attribution \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/bias_mitigation_results_civil/bert_civil_race_race_test_2000/attention_rollout/L2_0.0001" \
    --split="test" \
    --bias_type="race" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m model_selection.compute_reliance_statistics \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/bias_mitigation_results_civil/bert_civil_race_race_test_2000/attention_rollout/L2_0.0001" \
    --split="test" \
    --bias_type="race" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m sensitive_attribution_extraction.extract_sensitive_token_attribution \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/bias_mitigation_results_civil/bert_civil_race_race_test_2000/Occlusion/L1_1.0" \
    --split="test" \
    --bias_type="race" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m model_selection.compute_reliance_statistics \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/bias_mitigation_results_civil/bert_civil_race_race_test_2000/Occlusion/L1_1.0" \
    --split="test" \
    --bias_type="race" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m sensitive_attribution_extraction.extract_sensitive_token_attribution \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/bias_mitigation_results_civil/bert_civil_race_race_test_2000/Occlusion/L1_0.1" \
    --split="test" \
    --bias_type="race" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m model_selection.compute_reliance_statistics \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/bias_mitigation_results_civil/bert_civil_race_race_test_2000/Occlusion/L1_0.1" \
    --split="test" \
    --bias_type="race" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m sensitive_attribution_extraction.extract_sensitive_token_attribution \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/bias_mitigation_results_civil/bert_civil_race_race_test_2000/Occlusion/L1_0.01" \
    --split="test" \
    --bias_type="race" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m model_selection.compute_reliance_statistics \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/bias_mitigation_results_civil/bert_civil_race_race_test_2000/Occlusion/L1_0.01" \
    --split="test" \
    --bias_type="race" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m sensitive_attribution_extraction.extract_sensitive_token_attribution \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/bias_mitigation_results_civil/bert_civil_race_race_test_2000/Occlusion/L1_0.001" \
    --split="test" \
    --bias_type="race" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m model_selection.compute_reliance_statistics \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/bias_mitigation_results_civil/bert_civil_race_race_test_2000/Occlusion/L1_0.001" \
    --split="test" \
    --bias_type="race" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m sensitive_attribution_extraction.extract_sensitive_token_attribution \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/bias_mitigation_results_civil/bert_civil_race_race_test_2000/Occlusion/L1_0.0001" \
    --split="test" \
    --bias_type="race" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m model_selection.compute_reliance_statistics \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/bias_mitigation_results_civil/bert_civil_race_race_test_2000/Occlusion/L1_0.0001" \
    --split="test" \
    --bias_type="race" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m sensitive_attribution_extraction.extract_sensitive_token_attribution \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/bias_mitigation_results_civil/bert_civil_race_race_test_2000/Occlusion/L2_1.0" \
    --split="test" \
    --bias_type="race" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m model_selection.compute_reliance_statistics \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/bias_mitigation_results_civil/bert_civil_race_race_test_2000/Occlusion/L2_1.0" \
    --split="test" \
    --bias_type="race" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m sensitive_attribution_extraction.extract_sensitive_token_attribution \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/bias_mitigation_results_civil/bert_civil_race_race_test_2000/Occlusion/L2_0.1" \
    --split="test" \
    --bias_type="race" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m model_selection.compute_reliance_statistics \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/bias_mitigation_results_civil/bert_civil_race_race_test_2000/Occlusion/L2_0.1" \
    --split="test" \
    --bias_type="race" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m sensitive_attribution_extraction.extract_sensitive_token_attribution \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/bias_mitigation_results_civil/bert_civil_race_race_test_2000/Occlusion/L2_0.01" \
    --split="test" \
    --bias_type="race" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m model_selection.compute_reliance_statistics \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/bias_mitigation_results_civil/bert_civil_race_race_test_2000/Occlusion/L2_0.01" \
    --split="test" \
    --bias_type="race" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m sensitive_attribution_extraction.extract_sensitive_token_attribution \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/bias_mitigation_results_civil/bert_civil_race_race_test_2000/Occlusion/L2_0.001" \
    --split="test" \
    --bias_type="race" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m model_selection.compute_reliance_statistics \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/bias_mitigation_results_civil/bert_civil_race_race_test_2000/Occlusion/L2_0.001" \
    --split="test" \
    --bias_type="race" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m sensitive_attribution_extraction.extract_sensitive_token_attribution \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/bias_mitigation_results_civil/bert_civil_race_race_test_2000/Occlusion/L2_0.0001" \
    --split="test" \
    --bias_type="race" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m model_selection.compute_reliance_statistics \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/bias_mitigation_results_civil/bert_civil_race_race_test_2000/Occlusion/L2_0.0001" \
    --split="test" \
    --bias_type="race" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"


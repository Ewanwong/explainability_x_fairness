#!/bin/bash
python -m sensitive_attribution_extraction.extract_sensitive_token_attribution \
    --results_dir="/scratch/yifwang/fairness_x_explainability/decoder_results_civil/llama_3b_civil_race_test_2000/few_shot" \
    --split="test" \
    --bias_type="race" \
    --methods="IntegratedGradients"

python -m model_selection.compute_reliance_statistics \
    --results_dir="/scratch/yifwang/fairness_x_explainability/decoder_results_civil/llama_3b_civil_race_test_2000/few_shot" \
    --split="test" \
    --bias_type="race" \
    --methods="IntegratedGradients"

python -m sensitive_attribution_extraction.extract_sensitive_token_attribution \
    --results_dir="/scratch/yifwang/fairness_x_explainability/decoder_results_civil/llama_3b_civil_race_test_2000/fairness_imagination" \
    --split="test" \
    --bias_type="race" \
    --methods="IntegratedGradients"

python -m model_selection.compute_reliance_statistics \
    --results_dir="/scratch/yifwang/fairness_x_explainability/decoder_results_civil/llama_3b_civil_race_test_2000/fairness_imagination" \
    --split="test" \
    --bias_type="race" \
    --methods="IntegratedGradients"

python -m sensitive_attribution_extraction.extract_sensitive_token_attribution \
    --results_dir="/scratch/yifwang/fairness_x_explainability/decoder_results_civil/llama_3b_civil_race_test_2000/fairness_instruction" \
    --split="test" \
    --bias_type="race" \
    --methods="IntegratedGradients"

python -m model_selection.compute_reliance_statistics \
    --results_dir="/scratch/yifwang/fairness_x_explainability/decoder_results_civil/llama_3b_civil_race_test_2000/fairness_instruction" \
    --split="test" \
    --bias_type="race" \
    --methods="IntegratedGradients"

python -m sensitive_attribution_extraction.extract_sensitive_token_attribution \
    --results_dir="/scratch/yifwang/fairness_x_explainability/decoder_results_civil/qwen3_4b_civil_race_test_2000/few_shot" \
    --split="test" \
    --bias_type="race" \
    --methods="IntegratedGradients"

python -m model_selection.compute_reliance_statistics \
    --results_dir="/scratch/yifwang/fairness_x_explainability/decoder_results_civil/qwen3_4b_civil_race_test_2000/few_shot" \
    --split="test" \
    --bias_type="race" \
    --methods="IntegratedGradients"

python -m sensitive_attribution_extraction.extract_sensitive_token_attribution \
    --results_dir="/scratch/yifwang/fairness_x_explainability/decoder_results_civil/qwen3_4b_civil_race_test_2000/fairness_imagination" \
    --split="test" \
    --bias_type="race" \
    --methods="IntegratedGradients"

python -m model_selection.compute_reliance_statistics \
    --results_dir="/scratch/yifwang/fairness_x_explainability/decoder_results_civil/qwen3_4b_civil_race_test_2000/fairness_imagination" \
    --split="test" \
    --bias_type="race" \
    --methods="IntegratedGradients"

python -m sensitive_attribution_extraction.extract_sensitive_token_attribution \
    --results_dir="/scratch/yifwang/fairness_x_explainability/decoder_results_civil/qwen3_4b_civil_race_test_2000/fairness_instruction" \
    --split="test" \
    --bias_type="race" \
    --methods="IntegratedGradients"

python -m model_selection.compute_reliance_statistics \
    --results_dir="/scratch/yifwang/fairness_x_explainability/decoder_results_civil/qwen3_4b_civil_race_test_2000/fairness_instruction" \
    --split="test" \
    --bias_type="race" \
    --methods="IntegratedGradients"


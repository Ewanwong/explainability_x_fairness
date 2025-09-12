#!/bin/bash
python -m bias_correlation.compute_bias_correlation \
    --results_dir="/scratch/yifwang/fairness_x_explainability/decoder_results_civil/llama_3b_civil_race_test_2000/few_shot" \
    --split="test" \
    --bias_type="race" \
    --methods="IntegratedGradients"

python -m visualization.visualize_bias_correlation \
    --results_dir="/scratch/yifwang/fairness_x_explainability/decoder_results_civil/llama_3b_civil_race_test_2000/few_shot" \
    --split="test" \
    --bias_type="race" \
    --methods="IntegratedGradients"

python -m bias_correlation.compute_bias_correlation \
    --results_dir="/scratch/yifwang/fairness_x_explainability/decoder_results_civil/llama_3b_civil_race_test_2000/fairness_imagination" \
    --split="test" \
    --bias_type="race" \
    --methods="IntegratedGradients"

python -m visualization.visualize_bias_correlation \
    --results_dir="/scratch/yifwang/fairness_x_explainability/decoder_results_civil/llama_3b_civil_race_test_2000/fairness_imagination" \
    --split="test" \
    --bias_type="race" \
    --methods="IntegratedGradients"

python -m bias_correlation.compute_bias_correlation \
    --results_dir="/scratch/yifwang/fairness_x_explainability/decoder_results_civil/llama_3b_civil_race_test_2000/fairness_instruction" \
    --split="test" \
    --bias_type="race" \
    --methods="IntegratedGradients"

python -m visualization.visualize_bias_correlation \
    --results_dir="/scratch/yifwang/fairness_x_explainability/decoder_results_civil/llama_3b_civil_race_test_2000/fairness_instruction" \
    --split="test" \
    --bias_type="race" \
    --methods="IntegratedGradients"

python -m bias_correlation.compute_bias_correlation \
    --results_dir="/scratch/yifwang/fairness_x_explainability/decoder_results_civil/qwen3_4b_civil_race_test_2000/few_shot" \
    --split="test" \
    --bias_type="race" \
    --methods="IntegratedGradients"

python -m visualization.visualize_bias_correlation \
    --results_dir="/scratch/yifwang/fairness_x_explainability/decoder_results_civil/qwen3_4b_civil_race_test_2000/few_shot" \
    --split="test" \
    --bias_type="race" \
    --methods="IntegratedGradients"

python -m bias_correlation.compute_bias_correlation \
    --results_dir="/scratch/yifwang/fairness_x_explainability/decoder_results_civil/qwen3_4b_civil_race_test_2000/fairness_imagination" \
    --split="test" \
    --bias_type="race" \
    --methods="IntegratedGradients"

python -m visualization.visualize_bias_correlation \
    --results_dir="/scratch/yifwang/fairness_x_explainability/decoder_results_civil/qwen3_4b_civil_race_test_2000/fairness_imagination" \
    --split="test" \
    --bias_type="race" \
    --methods="IntegratedGradients"

python -m bias_correlation.compute_bias_correlation \
    --results_dir="/scratch/yifwang/fairness_x_explainability/decoder_results_civil/qwen3_4b_civil_race_test_2000/fairness_instruction" \
    --split="test" \
    --bias_type="race" \
    --methods="IntegratedGradients"

python -m visualization.visualize_bias_correlation \
    --results_dir="/scratch/yifwang/fairness_x_explainability/decoder_results_civil/qwen3_4b_civil_race_test_2000/fairness_instruction" \
    --split="test" \
    --bias_type="race" \
    --methods="IntegratedGradients"


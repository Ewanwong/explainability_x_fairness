#!/bin/bash
python -m bias_correlation.compute_bias_correlation \
    --results_dir="/scratch/yifwang/fairness_x_explainability/decoder_results_jigsaw/llama_3b_jigsaw_gender_test_800/few_shot" \
    --split="test" \
    --bias_type="gender" \
    --methods="IntegratedGradients"

python -m visualization.visualize_bias_correlation \
    --results_dir="/scratch/yifwang/fairness_x_explainability/decoder_results_jigsaw/llama_3b_jigsaw_gender_test_800/few_shot" \
    --split="test" \
    --bias_type="gender" \
    --methods="IntegratedGradients"

python -m bias_correlation.compute_bias_correlation \
    --results_dir="/scratch/yifwang/fairness_x_explainability/decoder_results_jigsaw/llama_3b_jigsaw_gender_test_800/fairness_imagination" \
    --split="test" \
    --bias_type="gender" \
    --methods="IntegratedGradients"

python -m visualization.visualize_bias_correlation \
    --results_dir="/scratch/yifwang/fairness_x_explainability/decoder_results_jigsaw/llama_3b_jigsaw_gender_test_800/fairness_imagination" \
    --split="test" \
    --bias_type="gender" \
    --methods="IntegratedGradients"

python -m bias_correlation.compute_bias_correlation \
    --results_dir="/scratch/yifwang/fairness_x_explainability/decoder_results_jigsaw/llama_3b_jigsaw_gender_test_800/fairness_instruction" \
    --split="test" \
    --bias_type="gender" \
    --methods="IntegratedGradients"

python -m visualization.visualize_bias_correlation \
    --results_dir="/scratch/yifwang/fairness_x_explainability/decoder_results_jigsaw/llama_3b_jigsaw_gender_test_800/fairness_instruction" \
    --split="test" \
    --bias_type="gender" \
    --methods="IntegratedGradients"

python -m bias_correlation.compute_bias_correlation \
    --results_dir="/scratch/yifwang/fairness_x_explainability/decoder_results_jigsaw/qwen3_4b_jigsaw_gender_test_800/few_shot" \
    --split="test" \
    --bias_type="gender" \
    --methods="IntegratedGradients"

python -m visualization.visualize_bias_correlation \
    --results_dir="/scratch/yifwang/fairness_x_explainability/decoder_results_jigsaw/qwen3_4b_jigsaw_gender_test_800/few_shot" \
    --split="test" \
    --bias_type="gender" \
    --methods="IntegratedGradients"

python -m bias_correlation.compute_bias_correlation \
    --results_dir="/scratch/yifwang/fairness_x_explainability/decoder_results_jigsaw/qwen3_4b_jigsaw_gender_test_800/fairness_imagination" \
    --split="test" \
    --bias_type="gender" \
    --methods="IntegratedGradients"

python -m visualization.visualize_bias_correlation \
    --results_dir="/scratch/yifwang/fairness_x_explainability/decoder_results_jigsaw/qwen3_4b_jigsaw_gender_test_800/fairness_imagination" \
    --split="test" \
    --bias_type="gender" \
    --methods="IntegratedGradients"

python -m bias_correlation.compute_bias_correlation \
    --results_dir="/scratch/yifwang/fairness_x_explainability/decoder_results_jigsaw/qwen3_4b_jigsaw_gender_test_800/fairness_instruction" \
    --split="test" \
    --bias_type="gender" \
    --methods="IntegratedGradients"

python -m visualization.visualize_bias_correlation \
    --results_dir="/scratch/yifwang/fairness_x_explainability/decoder_results_jigsaw/qwen3_4b_jigsaw_gender_test_800/fairness_instruction" \
    --split="test" \
    --bias_type="gender" \
    --methods="IntegratedGradients"


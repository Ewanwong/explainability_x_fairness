#!/bin/bash

python -m model_selection.compute_reliance_statistics \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/decoder_results/llama_3b_civil_gender_test_2000/zero_shot" \
    --split="test" \
    --bias_type="gender" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m model_selection.compute_reliance_statistics \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/decoder_results/llama_3b_civil_religion_test_1000/zero_shot" \
    --split="test" \
    --bias_type="religion" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m bias_correlation.compute_bias_correlation \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/decoder_results/llama_3b_civil_gender_test_2000/zero_shot" \
    --split="test" \
    --bias_type="gender" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m visualization.visualize_bias_correlation \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/decoder_results/llama_3b_civil_gender_test_2000/zero_shot" \
    --split="test" \
    --bias_type="gender" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m bias_correlation.compute_bias_correlation \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/decoder_results/llama_3b_civil_religion_test_1000/zero_shot" \
    --split="test" \
    --bias_type="religion" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m visualization.visualize_bias_correlation \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/decoder_results/llama_3b_civil_religion_test_1000/zero_shot" \
    --split="test" \
    --bias_type="religion" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"
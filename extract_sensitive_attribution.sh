#!/usr/bin/env bash
#CUDA_VISIBLE_DEVICES=0

#--methods "Attention, Saliency, DeepLift, InputXGradient, IntegratedGradients, Occlusion, KernelShap" \

python -m sensitive_attribution_extraction.extract_sensitive_token_attribution \
    --explanation_dir /scratch/yifwang/new_fairness_x_explainability/results/llama_3b_civil_2000_race \
    --split test \
    --bias_type race \
    --methods "Attention, Saliency, DeepLift, InputXGradient, IntegratedGradients, Occlusion, KernelShap" \

python -m sensitive_attribution_extraction.extract_sensitive_token_attribution \
    --explanation_dir /scratch/yifwang/new_fairness_x_explainability/results/llama_8b_civil_2000_race \
    --split test \
    --bias_type race \
    --methods "Attention, Saliency, DeepLift, InputXGradient, IntegratedGradients, Occlusion, KernelShap" \

python -m sensitive_attribution_extraction.extract_sensitive_token_attribution \
    --explanation_dir /scratch/yifwang/new_fairness_x_explainability/results/qwen_3b_civil_2000_race \
    --split test \
    --bias_type race \
    --methods "Attention, Saliency, DeepLift, InputXGradient, IntegratedGradients, Occlusion, KernelShap" \

python -m sensitive_attribution_extraction.extract_sensitive_token_attribution \
    --explanation_dir /scratch/yifwang/new_fairness_x_explainability/results/qwen_7b_civil_2000_race \
    --split test \
    --bias_type race \
    --methods "Attention, Saliency, DeepLift, InputXGradient, IntegratedGradients, Occlusion, KernelShap" \

python -m sensitive_attribution_extraction.extract_sensitive_token_attribution \
    --explanation_dir /scratch/yifwang/new_fairness_x_explainability/results/llama_3b_civil_2000_gender \
    --split test \
    --bias_type gender \
    --methods "Attention, Saliency, DeepLift, InputXGradient, IntegratedGradients, Occlusion, KernelShap" \

python -m sensitive_attribution_extraction.extract_sensitive_token_attribution \
    --explanation_dir /scratch/yifwang/new_fairness_x_explainability/results/llama_8b_civil_2000_gender \
    --split test \
    --bias_type gender \
    --methods "Attention, Saliency, DeepLift, InputXGradient, IntegratedGradients, Occlusion, KernelShap" \

python -m sensitive_attribution_extraction.extract_sensitive_token_attribution \
    --explanation_dir /scratch/yifwang/new_fairness_x_explainability/results/qwen_3b_civil_2000_gender \
    --split test \
    --bias_type gender \
    --methods "Attention, Saliency, DeepLift, InputXGradient, IntegratedGradients, Occlusion, KernelShap" \

python -m sensitive_attribution_extraction.extract_sensitive_token_attribution \
    --explanation_dir /scratch/yifwang/new_fairness_x_explainability/results/qwen_7b_civil_2000_gender \
    --split test \
    --bias_type gender \
    --methods "Attention, Saliency, DeepLift, InputXGradient, IntegratedGradients, Occlusion, KernelShap" \
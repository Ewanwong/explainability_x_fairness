#!/usr/bin/env bash
#CUDA_VISIBLE_DEVICES=0

#--methods "Attention, Saliency, DeepLift, InputXGradient, IntegratedGradients, Occlusion, KernelShap" \

# python -m bias_correlation.compute_bias_correlation \
#     --explanation_dir /scratch/yifwang/new_fairness_x_explainability/results/baseline_bert_civil_2000_gender \
#     --split test \
#     --methods "Attention, Saliency, DeepLift, InputXGradient, IntegratedGradients, Occlusion, KernelShap" \
#     --bias_type gender \


# python -m bias_correlation.compute_bias_correlation \
#     --explanation_dir /scratch/yifwang/new_fairness_x_explainability/results/baseline_roberta_civil_2000_gender \
#     --split test \
#     --methods "Attention, Saliency, DeepLift, InputXGradient, IntegratedGradients, Occlusion, KernelShap" \
#     --bias_type gender \


# python -m bias_correlation.compute_bias_correlation \
#     --explanation_dir /scratch/yifwang/new_fairness_x_explainability/results/baseline_distilbert_civil_2000_gender \
#     --split test \
#     --methods "Attention, Saliency, DeepLift, InputXGradient, IntegratedGradients, Occlusion, KernelShap" \
#     --bias_type gender \

# python -m bias_correlation.compute_bias_correlation \
#     --explanation_dir /scratch/yifwang/new_fairness_x_explainability/results/baseline_bert_civil_2000_race \
#     --split test \
#     --methods "Attention, Saliency, DeepLift, InputXGradient, IntegratedGradients, Occlusion, KernelShap" \
#     --bias_type race \


# python -m bias_correlation.compute_bias_correlation \
#     --explanation_dir /scratch/yifwang/new_fairness_x_explainability/results/baseline_roberta_civil_2000_race \
#     --split test \
#     --methods "Attention, Saliency, DeepLift, InputXGradient, IntegratedGradients, Occlusion, KernelShap" \
#     --bias_type race \


# python -m bias_correlation.compute_bias_correlation \
#     --explanation_dir /scratch/yifwang/new_fairness_x_explainability/results/baseline_distilbert_civil_2000_race \
#     --split test \
#     --methods "Attention, Saliency, DeepLift, InputXGradient, IntegratedGradients, Occlusion, KernelShap" \
#     --bias_type race \

python -m bias_correlation.compute_bias_correlation \
    --explanation_dir /scratch/yifwang/new_fairness_x_explainability/model_selection_results/bert_civil_cda_test_race \
    --split test \
    --methods "Attention, Saliency, DeepLift, InputXGradient, IntegratedGradients, Occlusion, KernelShap" \
    --bias_type race \

python -m bias_correlation.compute_bias_correlation \
    --explanation_dir /scratch/yifwang/new_fairness_x_explainability/model_selection_results/bert_civil_group_balance_test_race \
    --split test \
    --methods "Attention, Saliency, DeepLift, InputXGradient, IntegratedGradients, Occlusion, KernelShap" \
    --bias_type race \

python -m bias_correlation.compute_bias_correlation \
    --explanation_dir /scratch/yifwang/new_fairness_x_explainability/model_selection_results/bert_civil_group_class_balance_test_race \
    --split test \
    --methods "Attention, Saliency, DeepLift, InputXGradient, IntegratedGradients, Occlusion, KernelShap" \
    --bias_type race \

python -m bias_correlation.compute_bias_correlation \
    --explanation_dir /scratch/yifwang/new_fairness_x_explainability/model_selection_results/bert_civil_no_debiasing_test_race \
    --split test \
    --methods "Attention, Saliency, DeepLift, InputXGradient, IntegratedGradients, Occlusion, KernelShap" \
    --bias_type race \
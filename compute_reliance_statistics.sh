#!/usr/bin/env bash

#--methods "Attention, Saliency, DeepLift, InputXGradient, IntegratedGradients, Occlusion, KernelShap" \


python -m model_selection.compute_reliance_statistics \
    --explanation_dir /scratch/yifwang/new_fairness_x_explainability/results/baseline_bert_civil_2000_gender \
    --split test \
    --methods "Attention, Saliency, DeepLift, InputXGradient, IntegratedGradients, Occlusion, KernelShap" \
    --bias_type gender \


python -m model_selection.compute_reliance_statistics \
    --explanation_dir /scratch/yifwang/new_fairness_x_explainability/results/baseline_roberta_civil_2000_gender \
    --split test \
    --methods "Attention, Saliency, DeepLift, InputXGradient, IntegratedGradients, Occlusion, KernelShap" \
    --bias_type gender \

python -m model_selection.compute_reliance_statistics \
    --explanation_dir /scratch/yifwang/new_fairness_x_explainability/results/baseline_distilbert_civil_2000_gender \
    --split test \
    --methods "Attention, Saliency, DeepLift, InputXGradient, IntegratedGradients, Occlusion, KernelShap" \
    --bias_type gender \

python -m model_selection.compute_reliance_statistics \
    --explanation_dir /scratch/yifwang/new_fairness_x_explainability/results/baseline_bert_civil_2000_race \
    --split test \
    --methods "Attention, Saliency, DeepLift, InputXGradient, IntegratedGradients, Occlusion, KernelShap" \
    --bias_type race \

python -m model_selection.compute_reliance_statistics \
    --explanation_dir /scratch/yifwang/new_fairness_x_explainability/results/baseline_roberta_civil_2000_race \
    --split test \
    --methods "Attention, Saliency, DeepLift, InputXGradient, IntegratedGradients, Occlusion, KernelShap" \
    --bias_type race \

python -m model_selection.compute_reliance_statistics \
    --explanation_dir /scratch/yifwang/new_fairness_x_explainability/results/baseline_distilbert_civil_2000_race \
    --split test \
    --methods "Attention, Saliency, DeepLift, InputXGradient, IntegratedGradients, Occlusion, KernelShap" \
    --bias_type race \
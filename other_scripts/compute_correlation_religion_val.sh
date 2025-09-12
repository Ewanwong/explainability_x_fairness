#!/bin/bash
python -m bias_correlation.compute_bias_correlation \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/bert_civil_all_religion_val_200/dropout" \
    --split="val" \
    --bias_type="religion" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m visualization.visualize_bias_correlation \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/bert_civil_all_religion_val_200/dropout" \
    --split="val" \
    --bias_type="religion" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m bias_correlation.compute_bias_correlation \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/bert_civil_all_religion_val_200/attention_entropy" \
    --split="val" \
    --bias_type="religion" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m visualization.visualize_bias_correlation \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/bert_civil_all_religion_val_200/attention_entropy" \
    --split="val" \
    --bias_type="religion" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m bias_correlation.compute_bias_correlation \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/bert_civil_all_religion_val_200/causal_debias" \
    --split="val" \
    --bias_type="religion" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m visualization.visualize_bias_correlation \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/bert_civil_all_religion_val_200/causal_debias" \
    --split="val" \
    --bias_type="religion" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m bias_correlation.compute_bias_correlation \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/roberta_civil_all_religion_val_200/dropout" \
    --split="val" \
    --bias_type="religion" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m visualization.visualize_bias_correlation \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/roberta_civil_all_religion_val_200/dropout" \
    --split="val" \
    --bias_type="religion" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m bias_correlation.compute_bias_correlation \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/roberta_civil_all_religion_val_200/attention_entropy" \
    --split="val" \
    --bias_type="religion" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m visualization.visualize_bias_correlation \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/roberta_civil_all_religion_val_200/attention_entropy" \
    --split="val" \
    --bias_type="religion" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m bias_correlation.compute_bias_correlation \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/roberta_civil_all_religion_val_200/causal_debias" \
    --split="val" \
    --bias_type="religion" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m visualization.visualize_bias_correlation \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/roberta_civil_all_religion_val_200/causal_debias" \
    --split="val" \
    --bias_type="religion" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m bias_correlation.compute_bias_correlation \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/distilbert_civil_all_religion_val_200/dropout" \
    --split="val" \
    --bias_type="religion" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m visualization.visualize_bias_correlation \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/distilbert_civil_all_religion_val_200/dropout" \
    --split="val" \
    --bias_type="religion" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m bias_correlation.compute_bias_correlation \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/distilbert_civil_all_religion_val_200/attention_entropy" \
    --split="val" \
    --bias_type="religion" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m visualization.visualize_bias_correlation \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/distilbert_civil_all_religion_val_200/attention_entropy" \
    --split="val" \
    --bias_type="religion" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m bias_correlation.compute_bias_correlation \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/distilbert_civil_all_religion_val_200/causal_debias" \
    --split="val" \
    --bias_type="religion" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m visualization.visualize_bias_correlation \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/distilbert_civil_all_religion_val_200/causal_debias" \
    --split="val" \
    --bias_type="religion" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"


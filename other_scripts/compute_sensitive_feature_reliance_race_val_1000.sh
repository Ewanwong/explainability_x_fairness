#!/bin/bash
python -m sensitive_attribution_extraction.extract_sensitive_token_attribution \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/bert_civil_race_race_val_1000/no_debiasing" \
    --split="val" \
    --bias_type="race" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m model_selection.compute_reliance_statistics \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/bert_civil_race_race_val_1000/no_debiasing" \
    --split="val" \
    --bias_type="race" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m sensitive_attribution_extraction.extract_sensitive_token_attribution \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/bert_civil_all_race_val_1000/no_debiasing" \
    --split="val" \
    --bias_type="race" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m model_selection.compute_reliance_statistics \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/bert_civil_all_race_val_1000/no_debiasing" \
    --split="val" \
    --bias_type="race" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m sensitive_attribution_extraction.extract_sensitive_token_attribution \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/bert_civil_race_race_val_1000/group_balance" \
    --split="val" \
    --bias_type="race" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m model_selection.compute_reliance_statistics \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/bert_civil_race_race_val_1000/group_balance" \
    --split="val" \
    --bias_type="race" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m sensitive_attribution_extraction.extract_sensitive_token_attribution \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/bert_civil_all_race_val_1000/group_balance" \
    --split="val" \
    --bias_type="race" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m model_selection.compute_reliance_statistics \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/bert_civil_all_race_val_1000/group_balance" \
    --split="val" \
    --bias_type="race" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m sensitive_attribution_extraction.extract_sensitive_token_attribution \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/bert_civil_race_race_val_1000/group_class_balance" \
    --split="val" \
    --bias_type="race" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m model_selection.compute_reliance_statistics \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/bert_civil_race_race_val_1000/group_class_balance" \
    --split="val" \
    --bias_type="race" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m sensitive_attribution_extraction.extract_sensitive_token_attribution \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/bert_civil_all_race_val_1000/group_class_balance" \
    --split="val" \
    --bias_type="race" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m model_selection.compute_reliance_statistics \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/bert_civil_all_race_val_1000/group_class_balance" \
    --split="val" \
    --bias_type="race" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m sensitive_attribution_extraction.extract_sensitive_token_attribution \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/bert_civil_race_race_val_1000/cda" \
    --split="val" \
    --bias_type="race" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m model_selection.compute_reliance_statistics \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/bert_civil_race_race_val_1000/cda" \
    --split="val" \
    --bias_type="race" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m sensitive_attribution_extraction.extract_sensitive_token_attribution \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/bert_civil_all_race_val_1000/cda" \
    --split="val" \
    --bias_type="race" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m model_selection.compute_reliance_statistics \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/bert_civil_all_race_val_1000/cda" \
    --split="val" \
    --bias_type="race" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m sensitive_attribution_extraction.extract_sensitive_token_attribution \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/roberta_civil_race_race_val_1000/no_debiasing" \
    --split="val" \
    --bias_type="race" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m model_selection.compute_reliance_statistics \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/roberta_civil_race_race_val_1000/no_debiasing" \
    --split="val" \
    --bias_type="race" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m sensitive_attribution_extraction.extract_sensitive_token_attribution \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/roberta_civil_all_race_val_1000/no_debiasing" \
    --split="val" \
    --bias_type="race" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m model_selection.compute_reliance_statistics \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/roberta_civil_all_race_val_1000/no_debiasing" \
    --split="val" \
    --bias_type="race" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m sensitive_attribution_extraction.extract_sensitive_token_attribution \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/roberta_civil_race_race_val_1000/group_balance" \
    --split="val" \
    --bias_type="race" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m model_selection.compute_reliance_statistics \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/roberta_civil_race_race_val_1000/group_balance" \
    --split="val" \
    --bias_type="race" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m sensitive_attribution_extraction.extract_sensitive_token_attribution \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/roberta_civil_all_race_val_1000/group_balance" \
    --split="val" \
    --bias_type="race" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m model_selection.compute_reliance_statistics \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/roberta_civil_all_race_val_1000/group_balance" \
    --split="val" \
    --bias_type="race" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m sensitive_attribution_extraction.extract_sensitive_token_attribution \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/roberta_civil_race_race_val_1000/group_class_balance" \
    --split="val" \
    --bias_type="race" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m model_selection.compute_reliance_statistics \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/roberta_civil_race_race_val_1000/group_class_balance" \
    --split="val" \
    --bias_type="race" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m sensitive_attribution_extraction.extract_sensitive_token_attribution \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/roberta_civil_all_race_val_1000/group_class_balance" \
    --split="val" \
    --bias_type="race" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m model_selection.compute_reliance_statistics \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/roberta_civil_all_race_val_1000/group_class_balance" \
    --split="val" \
    --bias_type="race" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m sensitive_attribution_extraction.extract_sensitive_token_attribution \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/roberta_civil_race_race_val_1000/cda" \
    --split="val" \
    --bias_type="race" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m model_selection.compute_reliance_statistics \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/roberta_civil_race_race_val_1000/cda" \
    --split="val" \
    --bias_type="race" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m sensitive_attribution_extraction.extract_sensitive_token_attribution \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/roberta_civil_all_race_val_1000/cda" \
    --split="val" \
    --bias_type="race" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m model_selection.compute_reliance_statistics \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/roberta_civil_all_race_val_1000/cda" \
    --split="val" \
    --bias_type="race" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m sensitive_attribution_extraction.extract_sensitive_token_attribution \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/distilbert_civil_race_race_val_1000/no_debiasing" \
    --split="val" \
    --bias_type="race" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m model_selection.compute_reliance_statistics \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/distilbert_civil_race_race_val_1000/no_debiasing" \
    --split="val" \
    --bias_type="race" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m sensitive_attribution_extraction.extract_sensitive_token_attribution \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/distilbert_civil_all_race_val_1000/no_debiasing" \
    --split="val" \
    --bias_type="race" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m model_selection.compute_reliance_statistics \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/distilbert_civil_all_race_val_1000/no_debiasing" \
    --split="val" \
    --bias_type="race" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m sensitive_attribution_extraction.extract_sensitive_token_attribution \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/distilbert_civil_race_race_val_1000/group_balance" \
    --split="val" \
    --bias_type="race" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m model_selection.compute_reliance_statistics \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/distilbert_civil_race_race_val_1000/group_balance" \
    --split="val" \
    --bias_type="race" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m sensitive_attribution_extraction.extract_sensitive_token_attribution \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/distilbert_civil_all_race_val_1000/group_balance" \
    --split="val" \
    --bias_type="race" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m model_selection.compute_reliance_statistics \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/distilbert_civil_all_race_val_1000/group_balance" \
    --split="val" \
    --bias_type="race" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m sensitive_attribution_extraction.extract_sensitive_token_attribution \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/distilbert_civil_race_race_val_1000/group_class_balance" \
    --split="val" \
    --bias_type="race" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m model_selection.compute_reliance_statistics \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/distilbert_civil_race_race_val_1000/group_class_balance" \
    --split="val" \
    --bias_type="race" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m sensitive_attribution_extraction.extract_sensitive_token_attribution \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/distilbert_civil_all_race_val_1000/group_class_balance" \
    --split="val" \
    --bias_type="race" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m model_selection.compute_reliance_statistics \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/distilbert_civil_all_race_val_1000/group_class_balance" \
    --split="val" \
    --bias_type="race" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m sensitive_attribution_extraction.extract_sensitive_token_attribution \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/distilbert_civil_race_race_val_1000/cda" \
    --split="val" \
    --bias_type="race" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m model_selection.compute_reliance_statistics \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/distilbert_civil_race_race_val_1000/cda" \
    --split="val" \
    --bias_type="race" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m sensitive_attribution_extraction.extract_sensitive_token_attribution \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/distilbert_civil_all_race_val_1000/cda" \
    --split="val" \
    --bias_type="race" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m model_selection.compute_reliance_statistics \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/distilbert_civil_all_race_val_1000/cda" \
    --split="val" \
    --bias_type="race" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"


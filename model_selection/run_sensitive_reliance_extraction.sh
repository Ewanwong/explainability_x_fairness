#!/usr/bin/env bash

#--methods "Attention, Saliency, DeepLift, InputXGradient, IntegratedGradients, Occlusion, KernelShap" \

python -m sensitive_attribution_extraction.extract_sensitive_token_attribution \
    --explanation_dir /scratch/yifwang/new_fairness_x_explainability/model_selection_results/bert_civil_no_debiasing_val_500_race \
    --split test \
    --bias_type race \
    --methods "Attention, Saliency, DeepLift, InputXGradient, IntegratedGradients, Occlusion, KernelShap" \


python -m sensitive_attribution_extraction.extract_sensitive_token_attribution \
    --explanation_dir /scratch/yifwang/new_fairness_x_explainability/model_selection_results/bert_civil_group_balance_val_500_race \
    --split test \
    --bias_type race \
    --methods "Attention, Saliency, DeepLift, InputXGradient, IntegratedGradients, Occlusion, KernelShap" \


python -m sensitive_attribution_extraction.extract_sensitive_token_attribution \
    --explanation_dir /scratch/yifwang/new_fairness_x_explainability/model_selection_results/bert_civil_group_class_balance_val_500_race \
    --split test \
    --bias_type race \
    --methods "Attention, Saliency, DeepLift, InputXGradient, IntegratedGradients, Occlusion, KernelShap" \


python -m sensitive_attribution_extraction.extract_sensitive_token_attribution \
    --explanation_dir /scratch/yifwang/new_fairness_x_explainability/model_selection_results/bert_civil_cda_val_500_race \
    --split test \
    --bias_type race \
    --methods "Attention, Saliency, DeepLift, InputXGradient, IntegratedGradients, Occlusion, KernelShap" \

###################################################################################

python -m model_selection.compute_reliance_statistics \
    --explanation_dir /scratch/yifwang/new_fairness_x_explainability/model_selection_results/bert_civil_no_debiasing_val_500_race \
    --split test \
    --bias_type race \
    --methods "Attention, Saliency, DeepLift, InputXGradient, IntegratedGradients, Occlusion, KernelShap" \
    

python -m model_selection.compute_reliance_statistics \
    --explanation_dir /scratch/yifwang/new_fairness_x_explainability/model_selection_results/bert_civil_group_balance_val_500_race \
    --split test \
    --bias_type race \
    --methods "Attention, Saliency, DeepLift, InputXGradient, IntegratedGradients, Occlusion, KernelShap" \


python -m model_selection.compute_reliance_statistics \
    --explanation_dir /scratch/yifwang/new_fairness_x_explainability/model_selection_results/bert_civil_group_class_balance_val_500_race \
    --split test \
    --bias_type race \
    --methods "Attention, Saliency, DeepLift, InputXGradient, IntegratedGradients, Occlusion, KernelShap" \


python -m model_selection.compute_reliance_statistics \
    --explanation_dir /scratch/yifwang/new_fairness_x_explainability/model_selection_results/bert_civil_cda_val_500_race \
    --split test \
    --bias_type race \
    --methods "Attention, Saliency, DeepLift, InputXGradient, IntegratedGradients, Occlusion, KernelShap" \


python -m sensitive_attribution_extraction.extract_sensitive_token_attribution \
    --explanation_dir /scratch/yifwang/new_fairness_x_explainability/model_selection_results/bert_civil_no_debiasing_val_500_gender \
    --split test \
    --bias_type gender \
    --methods "Attention, Saliency, DeepLift, InputXGradient, IntegratedGradients, Occlusion, KernelShap" \


python -m sensitive_attribution_extraction.extract_sensitive_token_attribution \
    --explanation_dir /scratch/yifwang/new_fairness_x_explainability/model_selection_results/bert_civil_group_balance_val_500_gender \
    --split test \
    --bias_type gender \
    --methods "Attention, Saliency, DeepLift, InputXGradient, IntegratedGradients, Occlusion, KernelShap" \


python -m sensitive_attribution_extraction.extract_sensitive_token_attribution \
    --explanation_dir /scratch/yifwang/new_fairness_x_explainability/model_selection_results/bert_civil_group_class_balance_val_500_gender \
    --split test \
    --bias_type gender \
    --methods "Attention, Saliency, DeepLift, InputXGradient, IntegratedGradients, Occlusion, KernelShap" \


python -m sensitive_attribution_extraction.extract_sensitive_token_attribution \
    --explanation_dir /scratch/yifwang/new_fairness_x_explainability/model_selection_results/bert_civil_cda_val_500_gender \
    --split test \
    --bias_type gender \
    --methods "Attention, Saliency, DeepLift, InputXGradient, IntegratedGradients, Occlusion, KernelShap" \

###################################################################################

python -m model_selection.compute_reliance_statistics \
    --explanation_dir /scratch/yifwang/new_fairness_x_explainability/model_selection_results/bert_civil_no_debiasing_val_500_gender \
    --split test \
    --bias_type gender \
    --methods "Attention, Saliency, DeepLift, InputXGradient, IntegratedGradients, Occlusion, KernelShap" \
    

python -m model_selection.compute_reliance_statistics \
    --explanation_dir /scratch/yifwang/new_fairness_x_explainability/model_selection_results/bert_civil_group_balance_val_500_gender \
    --split test \
    --bias_type gender \
    --methods "Attention, Saliency, DeepLift, InputXGradient, IntegratedGradients, Occlusion, KernelShap" \


python -m model_selection.compute_reliance_statistics \
    --explanation_dir /scratch/yifwang/new_fairness_x_explainability/model_selection_results/bert_civil_group_class_balance_val_500_gender \
    --split test \
    --bias_type gender \
    --methods "Attention, Saliency, DeepLift, InputXGradient, IntegratedGradients, Occlusion, KernelShap" \


python -m model_selection.compute_reliance_statistics \
    --explanation_dir /scratch/yifwang/new_fairness_x_explainability/model_selection_results/bert_civil_cda_val_500_gender \
    --split test \
    --bias_type gender \
    --methods "Attention, Saliency, DeepLift, InputXGradient, IntegratedGradients, Occlusion, KernelShap" \



python -m sensitive_attribution_extraction.extract_sensitive_token_attribution \
    --explanation_dir /scratch/yifwang/new_fairness_x_explainability/model_selection_results/roberta_civil_no_debiasing_val_500_race \
    --split test \
    --bias_type race \
    --methods "Attention, Saliency, DeepLift, InputXGradient, IntegratedGradients, Occlusion, KernelShap" \


python -m sensitive_attribution_extraction.extract_sensitive_token_attribution \
    --explanation_dir /scratch/yifwang/new_fairness_x_explainability/model_selection_results/roberta_civil_group_balance_val_500_race \
    --split test \
    --bias_type race \
    --methods "Attention, Saliency, DeepLift, InputXGradient, IntegratedGradients, Occlusion, KernelShap" \


python -m sensitive_attribution_extraction.extract_sensitive_token_attribution \
    --explanation_dir /scratch/yifwang/new_fairness_x_explainability/model_selection_results/roberta_civil_group_class_balance_val_500_race \
    --split test \
    --bias_type race \
    --methods "Attention, Saliency, DeepLift, InputXGradient, IntegratedGradients, Occlusion, KernelShap" \


python -m sensitive_attribution_extraction.extract_sensitive_token_attribution \
    --explanation_dir /scratch/yifwang/new_fairness_x_explainability/model_selection_results/roberta_civil_cda_val_500_race \
    --split test \
    --bias_type race \
    --methods "Attention, Saliency, DeepLift, InputXGradient, IntegratedGradients, Occlusion, KernelShap" \

###################################################################################

python -m model_selection.compute_reliance_statistics \
    --explanation_dir /scratch/yifwang/new_fairness_x_explainability/model_selection_results/roberta_civil_no_debiasing_val_500_race \
    --split test \
    --bias_type race \
    --methods "Attention, Saliency, DeepLift, InputXGradient, IntegratedGradients, Occlusion, KernelShap" \
    

python -m model_selection.compute_reliance_statistics \
    --explanation_dir /scratch/yifwang/new_fairness_x_explainability/model_selection_results/roberta_civil_group_balance_val_500_race \
    --split test \
    --bias_type race \
    --methods "Attention, Saliency, DeepLift, InputXGradient, IntegratedGradients, Occlusion, KernelShap" \


python -m model_selection.compute_reliance_statistics \
    --explanation_dir /scratch/yifwang/new_fairness_x_explainability/model_selection_results/roberta_civil_group_class_balance_val_500_race \
    --split test \
    --bias_type race \
    --methods "Attention, Saliency, DeepLift, InputXGradient, IntegratedGradients, Occlusion, KernelShap" \


python -m model_selection.compute_reliance_statistics \
    --explanation_dir /scratch/yifwang/new_fairness_x_explainability/model_selection_results/roberta_civil_cda_val_500_race \
    --split test \
    --bias_type race \
    --methods "Attention, Saliency, DeepLift, InputXGradient, IntegratedGradients, Occlusion, KernelShap" \


python -m sensitive_attribution_extraction.extract_sensitive_token_attribution \
    --explanation_dir /scratch/yifwang/new_fairness_x_explainability/model_selection_results/roberta_civil_no_debiasing_val_500_gender \
    --split test \
    --bias_type gender \
    --methods "Attention, Saliency, DeepLift, InputXGradient, IntegratedGradients, Occlusion, KernelShap" \


python -m sensitive_attribution_extraction.extract_sensitive_token_attribution \
    --explanation_dir /scratch/yifwang/new_fairness_x_explainability/model_selection_results/roberta_civil_group_balance_val_500_gender \
    --split test \
    --bias_type gender \
    --methods "Attention, Saliency, DeepLift, InputXGradient, IntegratedGradients, Occlusion, KernelShap" \


python -m sensitive_attribution_extraction.extract_sensitive_token_attribution \
    --explanation_dir /scratch/yifwang/new_fairness_x_explainability/model_selection_results/roberta_civil_group_class_balance_val_500_gender \
    --split test \
    --bias_type gender \
    --methods "Attention, Saliency, DeepLift, InputXGradient, IntegratedGradients, Occlusion, KernelShap" \


python -m sensitive_attribution_extraction.extract_sensitive_token_attribution \
    --explanation_dir /scratch/yifwang/new_fairness_x_explainability/model_selection_results/roberta_civil_cda_val_500_gender \
    --split test \
    --bias_type gender \
    --methods "Attention, Saliency, DeepLift, InputXGradient, IntegratedGradients, Occlusion, KernelShap" \

###################################################################################

python -m model_selection.compute_reliance_statistics \
    --explanation_dir /scratch/yifwang/new_fairness_x_explainability/model_selection_results/roberta_civil_no_debiasing_val_500_gender \
    --split test \
    --bias_type gender \
    --methods "Attention, Saliency, DeepLift, InputXGradient, IntegratedGradients, Occlusion, KernelShap" \
    

python -m model_selection.compute_reliance_statistics \
    --explanation_dir /scratch/yifwang/new_fairness_x_explainability/model_selection_results/roberta_civil_group_balance_val_500_gender \
    --split test \
    --bias_type gender \
    --methods "Attention, Saliency, DeepLift, InputXGradient, IntegratedGradients, Occlusion, KernelShap" \


python -m model_selection.compute_reliance_statistics \
    --explanation_dir /scratch/yifwang/new_fairness_x_explainability/model_selection_results/roberta_civil_group_class_balance_val_500_gender \
    --split test \
    --bias_type gender \
    --methods "Attention, Saliency, DeepLift, InputXGradient, IntegratedGradients, Occlusion, KernelShap" \


python -m model_selection.compute_reliance_statistics \
    --explanation_dir /scratch/yifwang/new_fairness_x_explainability/model_selection_results/roberta_civil_cda_val_500_gender \
    --split test \
    --bias_type gender \
    --methods "Attention, Saliency, DeepLift, InputXGradient, IntegratedGradients, Occlusion, KernelShap" \

python -m sensitive_attribution_extraction.extract_sensitive_token_attribution \
    --explanation_dir /scratch/yifwang/new_fairness_x_explainability/model_selection_results/distilbert_civil_no_debiasing_val_500_race \
    --split test \
    --bias_type race \
    --methods "Attention, Saliency, DeepLift, InputXGradient, IntegratedGradients, Occlusion, KernelShap" \


python -m sensitive_attribution_extraction.extract_sensitive_token_attribution \
    --explanation_dir /scratch/yifwang/new_fairness_x_explainability/model_selection_results/distilbert_civil_group_balance_val_500_race \
    --split test \
    --bias_type race \
    --methods "Attention, Saliency, DeepLift, InputXGradient, IntegratedGradients, Occlusion, KernelShap" \


python -m sensitive_attribution_extraction.extract_sensitive_token_attribution \
    --explanation_dir /scratch/yifwang/new_fairness_x_explainability/model_selection_results/distilbert_civil_group_class_balance_val_500_race \
    --split test \
    --bias_type race \
    --methods "Attention, Saliency, DeepLift, InputXGradient, IntegratedGradients, Occlusion, KernelShap" \


python -m sensitive_attribution_extraction.extract_sensitive_token_attribution \
    --explanation_dir /scratch/yifwang/new_fairness_x_explainability/model_selection_results/distilbert_civil_cda_val_500_race \
    --split test \
    --bias_type race \
    --methods "Attention, Saliency, DeepLift, InputXGradient, IntegratedGradients, Occlusion, KernelShap" \

###################################################################################

python -m model_selection.compute_reliance_statistics \
    --explanation_dir /scratch/yifwang/new_fairness_x_explainability/model_selection_results/distilbert_civil_no_debiasing_val_500_race \
    --split test \
    --bias_type race \
    --methods "Attention, Saliency, DeepLift, InputXGradient, IntegratedGradients, Occlusion, KernelShap" \
    

python -m model_selection.compute_reliance_statistics \
    --explanation_dir /scratch/yifwang/new_fairness_x_explainability/model_selection_results/distilbert_civil_group_balance_val_500_race \
    --split test \
    --bias_type race \
    --methods "Attention, Saliency, DeepLift, InputXGradient, IntegratedGradients, Occlusion, KernelShap" \


python -m model_selection.compute_reliance_statistics \
    --explanation_dir /scratch/yifwang/new_fairness_x_explainability/model_selection_results/distilbert_civil_group_class_balance_val_500_race \
    --split test \
    --bias_type race \
    --methods "Attention, Saliency, DeepLift, InputXGradient, IntegratedGradients, Occlusion, KernelShap" \


python -m model_selection.compute_reliance_statistics \
    --explanation_dir /scratch/yifwang/new_fairness_x_explainability/model_selection_results/distilbert_civil_cda_val_500_race \
    --split test \
    --bias_type race \
    --methods "Attention, Saliency, DeepLift, InputXGradient, IntegratedGradients, Occlusion, KernelShap" \


python -m sensitive_attribution_extraction.extract_sensitive_token_attribution \
    --explanation_dir /scratch/yifwang/new_fairness_x_explainability/model_selection_results/distilbert_civil_no_debiasing_val_500_gender \
    --split test \
    --bias_type gender \
    --methods "Attention, Saliency, DeepLift, InputXGradient, IntegratedGradients, Occlusion, KernelShap" \


python -m sensitive_attribution_extraction.extract_sensitive_token_attribution \
    --explanation_dir /scratch/yifwang/new_fairness_x_explainability/model_selection_results/distilbert_civil_group_balance_val_500_gender \
    --split test \
    --bias_type gender \
    --methods "Attention, Saliency, DeepLift, InputXGradient, IntegratedGradients, Occlusion, KernelShap" \


python -m sensitive_attribution_extraction.extract_sensitive_token_attribution \
    --explanation_dir /scratch/yifwang/new_fairness_x_explainability/model_selection_results/distilbert_civil_group_class_balance_val_500_gender \
    --split test \
    --bias_type gender \
    --methods "Attention, Saliency, DeepLift, InputXGradient, IntegratedGradients, Occlusion, KernelShap" \


python -m sensitive_attribution_extraction.extract_sensitive_token_attribution \
    --explanation_dir /scratch/yifwang/new_fairness_x_explainability/model_selection_results/distilbert_civil_cda_val_500_gender \
    --split test \
    --bias_type gender \
    --methods "Attention, Saliency, DeepLift, InputXGradient, IntegratedGradients, Occlusion, KernelShap" \

###################################################################################

python -m model_selection.compute_reliance_statistics \
    --explanation_dir /scratch/yifwang/new_fairness_x_explainability/model_selection_results/distilbert_civil_no_debiasing_val_500_gender \
    --split test \
    --bias_type gender \
    --methods "Attention, Saliency, DeepLift, InputXGradient, IntegratedGradients, Occlusion, KernelShap" \
    

python -m model_selection.compute_reliance_statistics \
    --explanation_dir /scratch/yifwang/new_fairness_x_explainability/model_selection_results/distilbert_civil_group_balance_val_500_gender \
    --split test \
    --bias_type gender \
    --methods "Attention, Saliency, DeepLift, InputXGradient, IntegratedGradients, Occlusion, KernelShap" \


python -m model_selection.compute_reliance_statistics \
    --explanation_dir /scratch/yifwang/new_fairness_x_explainability/model_selection_results/distilbert_civil_group_class_balance_val_500_gender \
    --split test \
    --bias_type gender \
    --methods "Attention, Saliency, DeepLift, InputXGradient, IntegratedGradients, Occlusion, KernelShap" \


python -m model_selection.compute_reliance_statistics \
    --explanation_dir /scratch/yifwang/new_fairness_x_explainability/model_selection_results/distilbert_civil_cda_val_500_gender \
    --split test \
    --bias_type gender \
    --methods "Attention, Saliency, DeepLift, InputXGradient, IntegratedGradients, Occlusion, KernelShap" \



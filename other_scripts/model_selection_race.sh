#!/bin/bash
python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models/bert_civil_race/no_debiasing" \
    --batch_size=32 \
    --max_seq_length=512 \
    --num_examples=50 \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/bert_civil_race_race_val_50/no_debiasing" \
    --bias_type="race"

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models/bert_civil_all/no_debiasing" \
    --batch_size=32 \
    --max_seq_length=512 \
    --num_examples=50 \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/bert_civil_all_race_val_50/no_debiasing" \
    --bias_type="race"

python -m explanation_generation.gen_explanation_encoder \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models/bert_civil_race/no_debiasing" \
    --num_labels=2 \
    --batch_size=32 \
    --max_length=512 \
    --num_examples=50 \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/bert_civil_race_race_val_50/no_debiasing" \
    --baseline="pad" \
    --bias_type="race" \
    --seed=42 \
    --only_predicted_class

python -m explanation_generation.gen_explanation_encoder \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models/bert_civil_all/no_debiasing" \
    --num_labels=2 \
    --batch_size=32 \
    --max_length=512 \
    --num_examples=50 \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/bert_civil_all_race_val_50/no_debiasing" \
    --baseline="pad" \
    --bias_type="race" \
    --seed=42 \
    --only_predicted_class

python -m sensitive_attribution_extraction.extract_sensitive_token_attribution \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/bert_civil_race_race_val_50/no_debiasing" \
    --split="val" \
    --bias_type="race" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m model_selection.compute_reliance_statistics \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/bert_civil_race_race_val_50/no_debiasing" \
    --split="val" \
    --bias_type="race" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m sensitive_attribution_extraction.extract_sensitive_token_attribution \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/bert_civil_all_race_val_50/no_debiasing" \
    --split="val" \
    --bias_type="race" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m model_selection.compute_reliance_statistics \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/bert_civil_all_race_val_50/no_debiasing" \
    --split="val" \
    --bias_type="race" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models/bert_civil_race/group_balance" \
    --batch_size=32 \
    --max_seq_length=512 \
    --num_examples=50 \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/bert_civil_race_race_val_50/group_balance" \
    --bias_type="race"

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models/bert_civil_all/group_balance" \
    --batch_size=32 \
    --max_seq_length=512 \
    --num_examples=50 \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/bert_civil_all_race_val_50/group_balance" \
    --bias_type="race"

python -m explanation_generation.gen_explanation_encoder \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models/bert_civil_race/group_balance" \
    --num_labels=2 \
    --batch_size=32 \
    --max_length=512 \
    --num_examples=50 \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/bert_civil_race_race_val_50/group_balance" \
    --baseline="pad" \
    --bias_type="race" \
    --seed=42 \
    --only_predicted_class

python -m explanation_generation.gen_explanation_encoder \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models/bert_civil_all/group_balance" \
    --num_labels=2 \
    --batch_size=32 \
    --max_length=512 \
    --num_examples=50 \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/bert_civil_all_race_val_50/group_balance" \
    --baseline="pad" \
    --bias_type="race" \
    --seed=42 \
    --only_predicted_class

python -m sensitive_attribution_extraction.extract_sensitive_token_attribution \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/bert_civil_race_race_val_50/group_balance" \
    --split="val" \
    --bias_type="race" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m model_selection.compute_reliance_statistics \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/bert_civil_race_race_val_50/group_balance" \
    --split="val" \
    --bias_type="race" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m sensitive_attribution_extraction.extract_sensitive_token_attribution \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/bert_civil_all_race_val_50/group_balance" \
    --split="val" \
    --bias_type="race" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m model_selection.compute_reliance_statistics \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/bert_civil_all_race_val_50/group_balance" \
    --split="val" \
    --bias_type="race" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models/bert_civil_race/group_class_balance" \
    --batch_size=32 \
    --max_seq_length=512 \
    --num_examples=50 \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/bert_civil_race_race_val_50/group_class_balance" \
    --bias_type="race"

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models/bert_civil_all/group_class_balance" \
    --batch_size=32 \
    --max_seq_length=512 \
    --num_examples=50 \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/bert_civil_all_race_val_50/group_class_balance" \
    --bias_type="race"

python -m explanation_generation.gen_explanation_encoder \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models/bert_civil_race/group_class_balance" \
    --num_labels=2 \
    --batch_size=32 \
    --max_length=512 \
    --num_examples=50 \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/bert_civil_race_race_val_50/group_class_balance" \
    --baseline="pad" \
    --bias_type="race" \
    --seed=42 \
    --only_predicted_class

python -m explanation_generation.gen_explanation_encoder \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models/bert_civil_all/group_class_balance" \
    --num_labels=2 \
    --batch_size=32 \
    --max_length=512 \
    --num_examples=50 \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/bert_civil_all_race_val_50/group_class_balance" \
    --baseline="pad" \
    --bias_type="race" \
    --seed=42 \
    --only_predicted_class

python -m sensitive_attribution_extraction.extract_sensitive_token_attribution \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/bert_civil_race_race_val_50/group_class_balance" \
    --split="val" \
    --bias_type="race" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m model_selection.compute_reliance_statistics \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/bert_civil_race_race_val_50/group_class_balance" \
    --split="val" \
    --bias_type="race" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m sensitive_attribution_extraction.extract_sensitive_token_attribution \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/bert_civil_all_race_val_50/group_class_balance" \
    --split="val" \
    --bias_type="race" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m model_selection.compute_reliance_statistics \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/bert_civil_all_race_val_50/group_class_balance" \
    --split="val" \
    --bias_type="race" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models/bert_civil_race/cda" \
    --batch_size=32 \
    --max_seq_length=512 \
    --num_examples=50 \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/bert_civil_race_race_val_50/cda" \
    --bias_type="race"

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models/bert_civil_all/cda" \
    --batch_size=32 \
    --max_seq_length=512 \
    --num_examples=50 \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/bert_civil_all_race_val_50/cda" \
    --bias_type="race"

python -m explanation_generation.gen_explanation_encoder \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models/bert_civil_race/cda" \
    --num_labels=2 \
    --batch_size=32 \
    --max_length=512 \
    --num_examples=50 \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/bert_civil_race_race_val_50/cda" \
    --baseline="pad" \
    --bias_type="race" \
    --seed=42 \
    --only_predicted_class

python -m explanation_generation.gen_explanation_encoder \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models/bert_civil_all/cda" \
    --num_labels=2 \
    --batch_size=32 \
    --max_length=512 \
    --num_examples=50 \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/bert_civil_all_race_val_50/cda" \
    --baseline="pad" \
    --bias_type="race" \
    --seed=42 \
    --only_predicted_class

python -m sensitive_attribution_extraction.extract_sensitive_token_attribution \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/bert_civil_race_race_val_50/cda" \
    --split="val" \
    --bias_type="race" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m model_selection.compute_reliance_statistics \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/bert_civil_race_race_val_50/cda" \
    --split="val" \
    --bias_type="race" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m sensitive_attribution_extraction.extract_sensitive_token_attribution \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/bert_civil_all_race_val_50/cda" \
    --split="val" \
    --bias_type="race" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m model_selection.compute_reliance_statistics \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/bert_civil_all_race_val_50/cda" \
    --split="val" \
    --bias_type="race" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models/bert_civil_race/dropout" \
    --batch_size=32 \
    --max_seq_length=512 \
    --num_examples=50 \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/bert_civil_race_race_val_50/dropout" \
    --bias_type="race"

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models/bert_civil_all/dropout" \
    --batch_size=32 \
    --max_seq_length=512 \
    --num_examples=50 \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/bert_civil_all_race_val_50/dropout" \
    --bias_type="race"

python -m explanation_generation.gen_explanation_encoder \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models/bert_civil_race/dropout" \
    --num_labels=2 \
    --batch_size=32 \
    --max_length=512 \
    --num_examples=50 \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/bert_civil_race_race_val_50/dropout" \
    --baseline="pad" \
    --bias_type="race" \
    --seed=42 \
    --only_predicted_class

python -m explanation_generation.gen_explanation_encoder \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models/bert_civil_all/dropout" \
    --num_labels=2 \
    --batch_size=32 \
    --max_length=512 \
    --num_examples=50 \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/bert_civil_all_race_val_50/dropout" \
    --baseline="pad" \
    --bias_type="race" \
    --seed=42 \
    --only_predicted_class

python -m sensitive_attribution_extraction.extract_sensitive_token_attribution \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/bert_civil_race_race_val_50/dropout" \
    --split="val" \
    --bias_type="race" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m model_selection.compute_reliance_statistics \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/bert_civil_race_race_val_50/dropout" \
    --split="val" \
    --bias_type="race" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m sensitive_attribution_extraction.extract_sensitive_token_attribution \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/bert_civil_all_race_val_50/dropout" \
    --split="val" \
    --bias_type="race" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m model_selection.compute_reliance_statistics \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/bert_civil_all_race_val_50/dropout" \
    --split="val" \
    --bias_type="race" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models/bert_civil_race/attention_entropy" \
    --batch_size=32 \
    --max_seq_length=512 \
    --num_examples=50 \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/bert_civil_race_race_val_50/attention_entropy" \
    --bias_type="race"

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models/bert_civil_all/attention_entropy" \
    --batch_size=32 \
    --max_seq_length=512 \
    --num_examples=50 \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/bert_civil_all_race_val_50/attention_entropy" \
    --bias_type="race"

python -m explanation_generation.gen_explanation_encoder \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models/bert_civil_race/attention_entropy" \
    --num_labels=2 \
    --batch_size=32 \
    --max_length=512 \
    --num_examples=50 \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/bert_civil_race_race_val_50/attention_entropy" \
    --baseline="pad" \
    --bias_type="race" \
    --seed=42 \
    --only_predicted_class

python -m explanation_generation.gen_explanation_encoder \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models/bert_civil_all/attention_entropy" \
    --num_labels=2 \
    --batch_size=32 \
    --max_length=512 \
    --num_examples=50 \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/bert_civil_all_race_val_50/attention_entropy" \
    --baseline="pad" \
    --bias_type="race" \
    --seed=42 \
    --only_predicted_class

python -m sensitive_attribution_extraction.extract_sensitive_token_attribution \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/bert_civil_race_race_val_50/attention_entropy" \
    --split="val" \
    --bias_type="race" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m model_selection.compute_reliance_statistics \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/bert_civil_race_race_val_50/attention_entropy" \
    --split="val" \
    --bias_type="race" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m sensitive_attribution_extraction.extract_sensitive_token_attribution \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/bert_civil_all_race_val_50/attention_entropy" \
    --split="val" \
    --bias_type="race" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m model_selection.compute_reliance_statistics \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/bert_civil_all_race_val_50/attention_entropy" \
    --split="val" \
    --bias_type="race" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models/bert_civil_race/causal_debias" \
    --batch_size=32 \
    --max_seq_length=512 \
    --num_examples=50 \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/bert_civil_race_race_val_50/causal_debias" \
    --bias_type="race"

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models/bert_civil_all/causal_debias" \
    --batch_size=32 \
    --max_seq_length=512 \
    --num_examples=50 \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/bert_civil_all_race_val_50/causal_debias" \
    --bias_type="race"

python -m explanation_generation.gen_explanation_encoder \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models/bert_civil_race/causal_debias" \
    --num_labels=2 \
    --batch_size=32 \
    --max_length=512 \
    --num_examples=50 \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/bert_civil_race_race_val_50/causal_debias" \
    --baseline="pad" \
    --bias_type="race" \
    --seed=42 \
    --only_predicted_class

python -m explanation_generation.gen_explanation_encoder \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models/bert_civil_all/causal_debias" \
    --num_labels=2 \
    --batch_size=32 \
    --max_length=512 \
    --num_examples=50 \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/bert_civil_all_race_val_50/causal_debias" \
    --baseline="pad" \
    --bias_type="race" \
    --seed=42 \
    --only_predicted_class

python -m sensitive_attribution_extraction.extract_sensitive_token_attribution \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/bert_civil_race_race_val_50/causal_debias" \
    --split="val" \
    --bias_type="race" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m model_selection.compute_reliance_statistics \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/bert_civil_race_race_val_50/causal_debias" \
    --split="val" \
    --bias_type="race" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m sensitive_attribution_extraction.extract_sensitive_token_attribution \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/bert_civil_all_race_val_50/causal_debias" \
    --split="val" \
    --bias_type="race" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m model_selection.compute_reliance_statistics \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/bert_civil_all_race_val_50/causal_debias" \
    --split="val" \
    --bias_type="race" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m model_selection.compute_model_selection_correlation \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results" \
    --val_results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --bias_type="race" \
    --model_type="bert" \
    --num_test_examples=2000 \
    --num_val_examples=50 \
    --test_seed=-1 \
    --val_seed=-1

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models/roberta_civil_race/no_debiasing" \
    --batch_size=32 \
    --max_seq_length=512 \
    --num_examples=50 \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/roberta_civil_race_race_val_50/no_debiasing" \
    --bias_type="race"

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models/roberta_civil_all/no_debiasing" \
    --batch_size=32 \
    --max_seq_length=512 \
    --num_examples=50 \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/roberta_civil_all_race_val_50/no_debiasing" \
    --bias_type="race"

python -m explanation_generation.gen_explanation_encoder \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models/roberta_civil_race/no_debiasing" \
    --num_labels=2 \
    --batch_size=32 \
    --max_length=512 \
    --num_examples=50 \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/roberta_civil_race_race_val_50/no_debiasing" \
    --baseline="pad" \
    --bias_type="race" \
    --seed=42 \
    --only_predicted_class

python -m explanation_generation.gen_explanation_encoder \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models/roberta_civil_all/no_debiasing" \
    --num_labels=2 \
    --batch_size=32 \
    --max_length=512 \
    --num_examples=50 \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/roberta_civil_all_race_val_50/no_debiasing" \
    --baseline="pad" \
    --bias_type="race" \
    --seed=42 \
    --only_predicted_class

python -m sensitive_attribution_extraction.extract_sensitive_token_attribution \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/roberta_civil_race_race_val_50/no_debiasing" \
    --split="val" \
    --bias_type="race" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m model_selection.compute_reliance_statistics \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/roberta_civil_race_race_val_50/no_debiasing" \
    --split="val" \
    --bias_type="race" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m sensitive_attribution_extraction.extract_sensitive_token_attribution \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/roberta_civil_all_race_val_50/no_debiasing" \
    --split="val" \
    --bias_type="race" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m model_selection.compute_reliance_statistics \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/roberta_civil_all_race_val_50/no_debiasing" \
    --split="val" \
    --bias_type="race" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models/roberta_civil_race/group_balance" \
    --batch_size=32 \
    --max_seq_length=512 \
    --num_examples=50 \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/roberta_civil_race_race_val_50/group_balance" \
    --bias_type="race"

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models/roberta_civil_all/group_balance" \
    --batch_size=32 \
    --max_seq_length=512 \
    --num_examples=50 \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/roberta_civil_all_race_val_50/group_balance" \
    --bias_type="race"

python -m explanation_generation.gen_explanation_encoder \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models/roberta_civil_race/group_balance" \
    --num_labels=2 \
    --batch_size=32 \
    --max_length=512 \
    --num_examples=50 \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/roberta_civil_race_race_val_50/group_balance" \
    --baseline="pad" \
    --bias_type="race" \
    --seed=42 \
    --only_predicted_class

python -m explanation_generation.gen_explanation_encoder \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models/roberta_civil_all/group_balance" \
    --num_labels=2 \
    --batch_size=32 \
    --max_length=512 \
    --num_examples=50 \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/roberta_civil_all_race_val_50/group_balance" \
    --baseline="pad" \
    --bias_type="race" \
    --seed=42 \
    --only_predicted_class

python -m sensitive_attribution_extraction.extract_sensitive_token_attribution \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/roberta_civil_race_race_val_50/group_balance" \
    --split="val" \
    --bias_type="race" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m model_selection.compute_reliance_statistics \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/roberta_civil_race_race_val_50/group_balance" \
    --split="val" \
    --bias_type="race" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m sensitive_attribution_extraction.extract_sensitive_token_attribution \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/roberta_civil_all_race_val_50/group_balance" \
    --split="val" \
    --bias_type="race" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m model_selection.compute_reliance_statistics \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/roberta_civil_all_race_val_50/group_balance" \
    --split="val" \
    --bias_type="race" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models/roberta_civil_race/group_class_balance" \
    --batch_size=32 \
    --max_seq_length=512 \
    --num_examples=50 \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/roberta_civil_race_race_val_50/group_class_balance" \
    --bias_type="race"

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models/roberta_civil_all/group_class_balance" \
    --batch_size=32 \
    --max_seq_length=512 \
    --num_examples=50 \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/roberta_civil_all_race_val_50/group_class_balance" \
    --bias_type="race"

python -m explanation_generation.gen_explanation_encoder \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models/roberta_civil_race/group_class_balance" \
    --num_labels=2 \
    --batch_size=32 \
    --max_length=512 \
    --num_examples=50 \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/roberta_civil_race_race_val_50/group_class_balance" \
    --baseline="pad" \
    --bias_type="race" \
    --seed=42 \
    --only_predicted_class

python -m explanation_generation.gen_explanation_encoder \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models/roberta_civil_all/group_class_balance" \
    --num_labels=2 \
    --batch_size=32 \
    --max_length=512 \
    --num_examples=50 \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/roberta_civil_all_race_val_50/group_class_balance" \
    --baseline="pad" \
    --bias_type="race" \
    --seed=42 \
    --only_predicted_class

python -m sensitive_attribution_extraction.extract_sensitive_token_attribution \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/roberta_civil_race_race_val_50/group_class_balance" \
    --split="val" \
    --bias_type="race" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m model_selection.compute_reliance_statistics \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/roberta_civil_race_race_val_50/group_class_balance" \
    --split="val" \
    --bias_type="race" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m sensitive_attribution_extraction.extract_sensitive_token_attribution \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/roberta_civil_all_race_val_50/group_class_balance" \
    --split="val" \
    --bias_type="race" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m model_selection.compute_reliance_statistics \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/roberta_civil_all_race_val_50/group_class_balance" \
    --split="val" \
    --bias_type="race" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models/roberta_civil_race/cda" \
    --batch_size=32 \
    --max_seq_length=512 \
    --num_examples=50 \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/roberta_civil_race_race_val_50/cda" \
    --bias_type="race"

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models/roberta_civil_all/cda" \
    --batch_size=32 \
    --max_seq_length=512 \
    --num_examples=50 \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/roberta_civil_all_race_val_50/cda" \
    --bias_type="race"

python -m explanation_generation.gen_explanation_encoder \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models/roberta_civil_race/cda" \
    --num_labels=2 \
    --batch_size=32 \
    --max_length=512 \
    --num_examples=50 \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/roberta_civil_race_race_val_50/cda" \
    --baseline="pad" \
    --bias_type="race" \
    --seed=42 \
    --only_predicted_class

python -m explanation_generation.gen_explanation_encoder \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models/roberta_civil_all/cda" \
    --num_labels=2 \
    --batch_size=32 \
    --max_length=512 \
    --num_examples=50 \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/roberta_civil_all_race_val_50/cda" \
    --baseline="pad" \
    --bias_type="race" \
    --seed=42 \
    --only_predicted_class

python -m sensitive_attribution_extraction.extract_sensitive_token_attribution \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/roberta_civil_race_race_val_50/cda" \
    --split="val" \
    --bias_type="race" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m model_selection.compute_reliance_statistics \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/roberta_civil_race_race_val_50/cda" \
    --split="val" \
    --bias_type="race" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m sensitive_attribution_extraction.extract_sensitive_token_attribution \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/roberta_civil_all_race_val_50/cda" \
    --split="val" \
    --bias_type="race" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m model_selection.compute_reliance_statistics \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/roberta_civil_all_race_val_50/cda" \
    --split="val" \
    --bias_type="race" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models/roberta_civil_race/dropout" \
    --batch_size=32 \
    --max_seq_length=512 \
    --num_examples=50 \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/roberta_civil_race_race_val_50/dropout" \
    --bias_type="race"

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models/roberta_civil_all/dropout" \
    --batch_size=32 \
    --max_seq_length=512 \
    --num_examples=50 \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/roberta_civil_all_race_val_50/dropout" \
    --bias_type="race"

python -m explanation_generation.gen_explanation_encoder \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models/roberta_civil_race/dropout" \
    --num_labels=2 \
    --batch_size=32 \
    --max_length=512 \
    --num_examples=50 \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/roberta_civil_race_race_val_50/dropout" \
    --baseline="pad" \
    --bias_type="race" \
    --seed=42 \
    --only_predicted_class

python -m explanation_generation.gen_explanation_encoder \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models/roberta_civil_all/dropout" \
    --num_labels=2 \
    --batch_size=32 \
    --max_length=512 \
    --num_examples=50 \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/roberta_civil_all_race_val_50/dropout" \
    --baseline="pad" \
    --bias_type="race" \
    --seed=42 \
    --only_predicted_class

python -m sensitive_attribution_extraction.extract_sensitive_token_attribution \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/roberta_civil_race_race_val_50/dropout" \
    --split="val" \
    --bias_type="race" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m model_selection.compute_reliance_statistics \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/roberta_civil_race_race_val_50/dropout" \
    --split="val" \
    --bias_type="race" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m sensitive_attribution_extraction.extract_sensitive_token_attribution \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/roberta_civil_all_race_val_50/dropout" \
    --split="val" \
    --bias_type="race" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m model_selection.compute_reliance_statistics \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/roberta_civil_all_race_val_50/dropout" \
    --split="val" \
    --bias_type="race" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models/roberta_civil_race/attention_entropy" \
    --batch_size=32 \
    --max_seq_length=512 \
    --num_examples=50 \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/roberta_civil_race_race_val_50/attention_entropy" \
    --bias_type="race"

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models/roberta_civil_all/attention_entropy" \
    --batch_size=32 \
    --max_seq_length=512 \
    --num_examples=50 \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/roberta_civil_all_race_val_50/attention_entropy" \
    --bias_type="race"

python -m explanation_generation.gen_explanation_encoder \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models/roberta_civil_race/attention_entropy" \
    --num_labels=2 \
    --batch_size=32 \
    --max_length=512 \
    --num_examples=50 \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/roberta_civil_race_race_val_50/attention_entropy" \
    --baseline="pad" \
    --bias_type="race" \
    --seed=42 \
    --only_predicted_class

python -m explanation_generation.gen_explanation_encoder \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models/roberta_civil_all/attention_entropy" \
    --num_labels=2 \
    --batch_size=32 \
    --max_length=512 \
    --num_examples=50 \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/roberta_civil_all_race_val_50/attention_entropy" \
    --baseline="pad" \
    --bias_type="race" \
    --seed=42 \
    --only_predicted_class

python -m sensitive_attribution_extraction.extract_sensitive_token_attribution \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/roberta_civil_race_race_val_50/attention_entropy" \
    --split="val" \
    --bias_type="race" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m model_selection.compute_reliance_statistics \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/roberta_civil_race_race_val_50/attention_entropy" \
    --split="val" \
    --bias_type="race" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m sensitive_attribution_extraction.extract_sensitive_token_attribution \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/roberta_civil_all_race_val_50/attention_entropy" \
    --split="val" \
    --bias_type="race" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m model_selection.compute_reliance_statistics \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/roberta_civil_all_race_val_50/attention_entropy" \
    --split="val" \
    --bias_type="race" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models/roberta_civil_race/causal_debias" \
    --batch_size=32 \
    --max_seq_length=512 \
    --num_examples=50 \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/roberta_civil_race_race_val_50/causal_debias" \
    --bias_type="race"

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models/roberta_civil_all/causal_debias" \
    --batch_size=32 \
    --max_seq_length=512 \
    --num_examples=50 \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/roberta_civil_all_race_val_50/causal_debias" \
    --bias_type="race"

python -m explanation_generation.gen_explanation_encoder \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models/roberta_civil_race/causal_debias" \
    --num_labels=2 \
    --batch_size=32 \
    --max_length=512 \
    --num_examples=50 \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/roberta_civil_race_race_val_50/causal_debias" \
    --baseline="pad" \
    --bias_type="race" \
    --seed=42 \
    --only_predicted_class

python -m explanation_generation.gen_explanation_encoder \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models/roberta_civil_all/causal_debias" \
    --num_labels=2 \
    --batch_size=32 \
    --max_length=512 \
    --num_examples=50 \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/roberta_civil_all_race_val_50/causal_debias" \
    --baseline="pad" \
    --bias_type="race" \
    --seed=42 \
    --only_predicted_class

python -m sensitive_attribution_extraction.extract_sensitive_token_attribution \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/roberta_civil_race_race_val_50/causal_debias" \
    --split="val" \
    --bias_type="race" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m model_selection.compute_reliance_statistics \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/roberta_civil_race_race_val_50/causal_debias" \
    --split="val" \
    --bias_type="race" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m sensitive_attribution_extraction.extract_sensitive_token_attribution \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/roberta_civil_all_race_val_50/causal_debias" \
    --split="val" \
    --bias_type="race" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m model_selection.compute_reliance_statistics \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/roberta_civil_all_race_val_50/causal_debias" \
    --split="val" \
    --bias_type="race" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m model_selection.compute_model_selection_correlation \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results" \
    --val_results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --bias_type="race" \
    --model_type="roberta" \
    --num_test_examples=2000 \
    --num_val_examples=50 \
    --test_seed=-1 \
    --val_seed=-1

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models/distilbert_civil_race/no_debiasing" \
    --batch_size=32 \
    --max_seq_length=512 \
    --num_examples=50 \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/distilbert_civil_race_race_val_50/no_debiasing" \
    --bias_type="race"

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models/distilbert_civil_all/no_debiasing" \
    --batch_size=32 \
    --max_seq_length=512 \
    --num_examples=50 \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/distilbert_civil_all_race_val_50/no_debiasing" \
    --bias_type="race"

python -m explanation_generation.gen_explanation_encoder \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models/distilbert_civil_race/no_debiasing" \
    --num_labels=2 \
    --batch_size=32 \
    --max_length=512 \
    --num_examples=50 \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/distilbert_civil_race_race_val_50/no_debiasing" \
    --baseline="pad" \
    --bias_type="race" \
    --seed=42 \
    --only_predicted_class

python -m explanation_generation.gen_explanation_encoder \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models/distilbert_civil_all/no_debiasing" \
    --num_labels=2 \
    --batch_size=32 \
    --max_length=512 \
    --num_examples=50 \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/distilbert_civil_all_race_val_50/no_debiasing" \
    --baseline="pad" \
    --bias_type="race" \
    --seed=42 \
    --only_predicted_class

python -m sensitive_attribution_extraction.extract_sensitive_token_attribution \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/distilbert_civil_race_race_val_50/no_debiasing" \
    --split="val" \
    --bias_type="race" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m model_selection.compute_reliance_statistics \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/distilbert_civil_race_race_val_50/no_debiasing" \
    --split="val" \
    --bias_type="race" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m sensitive_attribution_extraction.extract_sensitive_token_attribution \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/distilbert_civil_all_race_val_50/no_debiasing" \
    --split="val" \
    --bias_type="race" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m model_selection.compute_reliance_statistics \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/distilbert_civil_all_race_val_50/no_debiasing" \
    --split="val" \
    --bias_type="race" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models/distilbert_civil_race/group_balance" \
    --batch_size=32 \
    --max_seq_length=512 \
    --num_examples=50 \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/distilbert_civil_race_race_val_50/group_balance" \
    --bias_type="race"

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models/distilbert_civil_all/group_balance" \
    --batch_size=32 \
    --max_seq_length=512 \
    --num_examples=50 \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/distilbert_civil_all_race_val_50/group_balance" \
    --bias_type="race"

python -m explanation_generation.gen_explanation_encoder \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models/distilbert_civil_race/group_balance" \
    --num_labels=2 \
    --batch_size=32 \
    --max_length=512 \
    --num_examples=50 \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/distilbert_civil_race_race_val_50/group_balance" \
    --baseline="pad" \
    --bias_type="race" \
    --seed=42 \
    --only_predicted_class

python -m explanation_generation.gen_explanation_encoder \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models/distilbert_civil_all/group_balance" \
    --num_labels=2 \
    --batch_size=32 \
    --max_length=512 \
    --num_examples=50 \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/distilbert_civil_all_race_val_50/group_balance" \
    --baseline="pad" \
    --bias_type="race" \
    --seed=42 \
    --only_predicted_class

python -m sensitive_attribution_extraction.extract_sensitive_token_attribution \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/distilbert_civil_race_race_val_50/group_balance" \
    --split="val" \
    --bias_type="race" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m model_selection.compute_reliance_statistics \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/distilbert_civil_race_race_val_50/group_balance" \
    --split="val" \
    --bias_type="race" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m sensitive_attribution_extraction.extract_sensitive_token_attribution \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/distilbert_civil_all_race_val_50/group_balance" \
    --split="val" \
    --bias_type="race" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m model_selection.compute_reliance_statistics \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/distilbert_civil_all_race_val_50/group_balance" \
    --split="val" \
    --bias_type="race" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models/distilbert_civil_race/group_class_balance" \
    --batch_size=32 \
    --max_seq_length=512 \
    --num_examples=50 \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/distilbert_civil_race_race_val_50/group_class_balance" \
    --bias_type="race"

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models/distilbert_civil_all/group_class_balance" \
    --batch_size=32 \
    --max_seq_length=512 \
    --num_examples=50 \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/distilbert_civil_all_race_val_50/group_class_balance" \
    --bias_type="race"

python -m explanation_generation.gen_explanation_encoder \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models/distilbert_civil_race/group_class_balance" \
    --num_labels=2 \
    --batch_size=32 \
    --max_length=512 \
    --num_examples=50 \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/distilbert_civil_race_race_val_50/group_class_balance" \
    --baseline="pad" \
    --bias_type="race" \
    --seed=42 \
    --only_predicted_class

python -m explanation_generation.gen_explanation_encoder \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models/distilbert_civil_all/group_class_balance" \
    --num_labels=2 \
    --batch_size=32 \
    --max_length=512 \
    --num_examples=50 \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/distilbert_civil_all_race_val_50/group_class_balance" \
    --baseline="pad" \
    --bias_type="race" \
    --seed=42 \
    --only_predicted_class

python -m sensitive_attribution_extraction.extract_sensitive_token_attribution \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/distilbert_civil_race_race_val_50/group_class_balance" \
    --split="val" \
    --bias_type="race" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m model_selection.compute_reliance_statistics \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/distilbert_civil_race_race_val_50/group_class_balance" \
    --split="val" \
    --bias_type="race" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m sensitive_attribution_extraction.extract_sensitive_token_attribution \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/distilbert_civil_all_race_val_50/group_class_balance" \
    --split="val" \
    --bias_type="race" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m model_selection.compute_reliance_statistics \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/distilbert_civil_all_race_val_50/group_class_balance" \
    --split="val" \
    --bias_type="race" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models/distilbert_civil_race/cda" \
    --batch_size=32 \
    --max_seq_length=512 \
    --num_examples=50 \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/distilbert_civil_race_race_val_50/cda" \
    --bias_type="race"

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models/distilbert_civil_all/cda" \
    --batch_size=32 \
    --max_seq_length=512 \
    --num_examples=50 \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/distilbert_civil_all_race_val_50/cda" \
    --bias_type="race"

python -m explanation_generation.gen_explanation_encoder \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models/distilbert_civil_race/cda" \
    --num_labels=2 \
    --batch_size=32 \
    --max_length=512 \
    --num_examples=50 \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/distilbert_civil_race_race_val_50/cda" \
    --baseline="pad" \
    --bias_type="race" \
    --seed=42 \
    --only_predicted_class

python -m explanation_generation.gen_explanation_encoder \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models/distilbert_civil_all/cda" \
    --num_labels=2 \
    --batch_size=32 \
    --max_length=512 \
    --num_examples=50 \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/distilbert_civil_all_race_val_50/cda" \
    --baseline="pad" \
    --bias_type="race" \
    --seed=42 \
    --only_predicted_class

python -m sensitive_attribution_extraction.extract_sensitive_token_attribution \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/distilbert_civil_race_race_val_50/cda" \
    --split="val" \
    --bias_type="race" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m model_selection.compute_reliance_statistics \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/distilbert_civil_race_race_val_50/cda" \
    --split="val" \
    --bias_type="race" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m sensitive_attribution_extraction.extract_sensitive_token_attribution \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/distilbert_civil_all_race_val_50/cda" \
    --split="val" \
    --bias_type="race" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m model_selection.compute_reliance_statistics \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/distilbert_civil_all_race_val_50/cda" \
    --split="val" \
    --bias_type="race" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models/distilbert_civil_race/dropout" \
    --batch_size=32 \
    --max_seq_length=512 \
    --num_examples=50 \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/distilbert_civil_race_race_val_50/dropout" \
    --bias_type="race"

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models/distilbert_civil_all/dropout" \
    --batch_size=32 \
    --max_seq_length=512 \
    --num_examples=50 \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/distilbert_civil_all_race_val_50/dropout" \
    --bias_type="race"

python -m explanation_generation.gen_explanation_encoder \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models/distilbert_civil_race/dropout" \
    --num_labels=2 \
    --batch_size=32 \
    --max_length=512 \
    --num_examples=50 \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/distilbert_civil_race_race_val_50/dropout" \
    --baseline="pad" \
    --bias_type="race" \
    --seed=42 \
    --only_predicted_class

python -m explanation_generation.gen_explanation_encoder \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models/distilbert_civil_all/dropout" \
    --num_labels=2 \
    --batch_size=32 \
    --max_length=512 \
    --num_examples=50 \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/distilbert_civil_all_race_val_50/dropout" \
    --baseline="pad" \
    --bias_type="race" \
    --seed=42 \
    --only_predicted_class

python -m sensitive_attribution_extraction.extract_sensitive_token_attribution \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/distilbert_civil_race_race_val_50/dropout" \
    --split="val" \
    --bias_type="race" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m model_selection.compute_reliance_statistics \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/distilbert_civil_race_race_val_50/dropout" \
    --split="val" \
    --bias_type="race" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m sensitive_attribution_extraction.extract_sensitive_token_attribution \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/distilbert_civil_all_race_val_50/dropout" \
    --split="val" \
    --bias_type="race" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m model_selection.compute_reliance_statistics \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/distilbert_civil_all_race_val_50/dropout" \
    --split="val" \
    --bias_type="race" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models/distilbert_civil_race/attention_entropy" \
    --batch_size=32 \
    --max_seq_length=512 \
    --num_examples=50 \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/distilbert_civil_race_race_val_50/attention_entropy" \
    --bias_type="race"

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models/distilbert_civil_all/attention_entropy" \
    --batch_size=32 \
    --max_seq_length=512 \
    --num_examples=50 \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/distilbert_civil_all_race_val_50/attention_entropy" \
    --bias_type="race"

python -m explanation_generation.gen_explanation_encoder \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models/distilbert_civil_race/attention_entropy" \
    --num_labels=2 \
    --batch_size=32 \
    --max_length=512 \
    --num_examples=50 \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/distilbert_civil_race_race_val_50/attention_entropy" \
    --baseline="pad" \
    --bias_type="race" \
    --seed=42 \
    --only_predicted_class

python -m explanation_generation.gen_explanation_encoder \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models/distilbert_civil_all/attention_entropy" \
    --num_labels=2 \
    --batch_size=32 \
    --max_length=512 \
    --num_examples=50 \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/distilbert_civil_all_race_val_50/attention_entropy" \
    --baseline="pad" \
    --bias_type="race" \
    --seed=42 \
    --only_predicted_class

python -m sensitive_attribution_extraction.extract_sensitive_token_attribution \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/distilbert_civil_race_race_val_50/attention_entropy" \
    --split="val" \
    --bias_type="race" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m model_selection.compute_reliance_statistics \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/distilbert_civil_race_race_val_50/attention_entropy" \
    --split="val" \
    --bias_type="race" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m sensitive_attribution_extraction.extract_sensitive_token_attribution \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/distilbert_civil_all_race_val_50/attention_entropy" \
    --split="val" \
    --bias_type="race" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m model_selection.compute_reliance_statistics \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/distilbert_civil_all_race_val_50/attention_entropy" \
    --split="val" \
    --bias_type="race" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models/distilbert_civil_race/causal_debias" \
    --batch_size=32 \
    --max_seq_length=512 \
    --num_examples=50 \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/distilbert_civil_race_race_val_50/causal_debias" \
    --bias_type="race"

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models/distilbert_civil_all/causal_debias" \
    --batch_size=32 \
    --max_seq_length=512 \
    --num_examples=50 \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/distilbert_civil_all_race_val_50/causal_debias" \
    --bias_type="race"

python -m explanation_generation.gen_explanation_encoder \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models/distilbert_civil_race/causal_debias" \
    --num_labels=2 \
    --batch_size=32 \
    --max_length=512 \
    --num_examples=50 \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/distilbert_civil_race_race_val_50/causal_debias" \
    --baseline="pad" \
    --bias_type="race" \
    --seed=42 \
    --only_predicted_class

python -m explanation_generation.gen_explanation_encoder \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models/distilbert_civil_all/causal_debias" \
    --num_labels=2 \
    --batch_size=32 \
    --max_length=512 \
    --num_examples=50 \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/distilbert_civil_all_race_val_50/causal_debias" \
    --baseline="pad" \
    --bias_type="race" \
    --seed=42 \
    --only_predicted_class

python -m sensitive_attribution_extraction.extract_sensitive_token_attribution \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/distilbert_civil_race_race_val_50/causal_debias" \
    --split="val" \
    --bias_type="race" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m model_selection.compute_reliance_statistics \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/distilbert_civil_race_race_val_50/causal_debias" \
    --split="val" \
    --bias_type="race" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m sensitive_attribution_extraction.extract_sensitive_token_attribution \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/distilbert_civil_all_race_val_50/causal_debias" \
    --split="val" \
    --bias_type="race" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m model_selection.compute_reliance_statistics \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/distilbert_civil_all_race_val_50/causal_debias" \
    --split="val" \
    --bias_type="race" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m model_selection.compute_model_selection_correlation \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results" \
    --val_results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --bias_type="race" \
    --model_type="distilbert" \
    --num_test_examples=2000 \
    --num_val_examples=50 \
    --test_seed=-1 \
    --val_seed=-1

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models/bert_civil_race/no_debiasing" \
    --batch_size=32 \
    --max_seq_length=512 \
    --num_examples=100 \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/bert_civil_race_race_val_100/no_debiasing" \
    --bias_type="race"

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models/bert_civil_all/no_debiasing" \
    --batch_size=32 \
    --max_seq_length=512 \
    --num_examples=100 \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/bert_civil_all_race_val_100/no_debiasing" \
    --bias_type="race"

python -m explanation_generation.gen_explanation_encoder \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models/bert_civil_race/no_debiasing" \
    --num_labels=2 \
    --batch_size=32 \
    --max_length=512 \
    --num_examples=100 \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/bert_civil_race_race_val_100/no_debiasing" \
    --baseline="pad" \
    --bias_type="race" \
    --seed=42 \
    --only_predicted_class

python -m explanation_generation.gen_explanation_encoder \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models/bert_civil_all/no_debiasing" \
    --num_labels=2 \
    --batch_size=32 \
    --max_length=512 \
    --num_examples=100 \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/bert_civil_all_race_val_100/no_debiasing" \
    --baseline="pad" \
    --bias_type="race" \
    --seed=42 \
    --only_predicted_class

python -m sensitive_attribution_extraction.extract_sensitive_token_attribution \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/bert_civil_race_race_val_100/no_debiasing" \
    --split="val" \
    --bias_type="race" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m model_selection.compute_reliance_statistics \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/bert_civil_race_race_val_100/no_debiasing" \
    --split="val" \
    --bias_type="race" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m sensitive_attribution_extraction.extract_sensitive_token_attribution \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/bert_civil_all_race_val_100/no_debiasing" \
    --split="val" \
    --bias_type="race" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m model_selection.compute_reliance_statistics \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/bert_civil_all_race_val_100/no_debiasing" \
    --split="val" \
    --bias_type="race" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models/bert_civil_race/group_balance" \
    --batch_size=32 \
    --max_seq_length=512 \
    --num_examples=100 \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/bert_civil_race_race_val_100/group_balance" \
    --bias_type="race"

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models/bert_civil_all/group_balance" \
    --batch_size=32 \
    --max_seq_length=512 \
    --num_examples=100 \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/bert_civil_all_race_val_100/group_balance" \
    --bias_type="race"

python -m explanation_generation.gen_explanation_encoder \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models/bert_civil_race/group_balance" \
    --num_labels=2 \
    --batch_size=32 \
    --max_length=512 \
    --num_examples=100 \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/bert_civil_race_race_val_100/group_balance" \
    --baseline="pad" \
    --bias_type="race" \
    --seed=42 \
    --only_predicted_class

python -m explanation_generation.gen_explanation_encoder \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models/bert_civil_all/group_balance" \
    --num_labels=2 \
    --batch_size=32 \
    --max_length=512 \
    --num_examples=100 \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/bert_civil_all_race_val_100/group_balance" \
    --baseline="pad" \
    --bias_type="race" \
    --seed=42 \
    --only_predicted_class

python -m sensitive_attribution_extraction.extract_sensitive_token_attribution \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/bert_civil_race_race_val_100/group_balance" \
    --split="val" \
    --bias_type="race" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m model_selection.compute_reliance_statistics \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/bert_civil_race_race_val_100/group_balance" \
    --split="val" \
    --bias_type="race" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m sensitive_attribution_extraction.extract_sensitive_token_attribution \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/bert_civil_all_race_val_100/group_balance" \
    --split="val" \
    --bias_type="race" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m model_selection.compute_reliance_statistics \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/bert_civil_all_race_val_100/group_balance" \
    --split="val" \
    --bias_type="race" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models/bert_civil_race/group_class_balance" \
    --batch_size=32 \
    --max_seq_length=512 \
    --num_examples=100 \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/bert_civil_race_race_val_100/group_class_balance" \
    --bias_type="race"

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models/bert_civil_all/group_class_balance" \
    --batch_size=32 \
    --max_seq_length=512 \
    --num_examples=100 \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/bert_civil_all_race_val_100/group_class_balance" \
    --bias_type="race"

python -m explanation_generation.gen_explanation_encoder \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models/bert_civil_race/group_class_balance" \
    --num_labels=2 \
    --batch_size=32 \
    --max_length=512 \
    --num_examples=100 \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/bert_civil_race_race_val_100/group_class_balance" \
    --baseline="pad" \
    --bias_type="race" \
    --seed=42 \
    --only_predicted_class

python -m explanation_generation.gen_explanation_encoder \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models/bert_civil_all/group_class_balance" \
    --num_labels=2 \
    --batch_size=32 \
    --max_length=512 \
    --num_examples=100 \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/bert_civil_all_race_val_100/group_class_balance" \
    --baseline="pad" \
    --bias_type="race" \
    --seed=42 \
    --only_predicted_class

python -m sensitive_attribution_extraction.extract_sensitive_token_attribution \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/bert_civil_race_race_val_100/group_class_balance" \
    --split="val" \
    --bias_type="race" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m model_selection.compute_reliance_statistics \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/bert_civil_race_race_val_100/group_class_balance" \
    --split="val" \
    --bias_type="race" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m sensitive_attribution_extraction.extract_sensitive_token_attribution \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/bert_civil_all_race_val_100/group_class_balance" \
    --split="val" \
    --bias_type="race" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m model_selection.compute_reliance_statistics \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/bert_civil_all_race_val_100/group_class_balance" \
    --split="val" \
    --bias_type="race" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models/bert_civil_race/cda" \
    --batch_size=32 \
    --max_seq_length=512 \
    --num_examples=100 \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/bert_civil_race_race_val_100/cda" \
    --bias_type="race"

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models/bert_civil_all/cda" \
    --batch_size=32 \
    --max_seq_length=512 \
    --num_examples=100 \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/bert_civil_all_race_val_100/cda" \
    --bias_type="race"

python -m explanation_generation.gen_explanation_encoder \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models/bert_civil_race/cda" \
    --num_labels=2 \
    --batch_size=32 \
    --max_length=512 \
    --num_examples=100 \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/bert_civil_race_race_val_100/cda" \
    --baseline="pad" \
    --bias_type="race" \
    --seed=42 \
    --only_predicted_class

python -m explanation_generation.gen_explanation_encoder \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models/bert_civil_all/cda" \
    --num_labels=2 \
    --batch_size=32 \
    --max_length=512 \
    --num_examples=100 \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/bert_civil_all_race_val_100/cda" \
    --baseline="pad" \
    --bias_type="race" \
    --seed=42 \
    --only_predicted_class

python -m sensitive_attribution_extraction.extract_sensitive_token_attribution \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/bert_civil_race_race_val_100/cda" \
    --split="val" \
    --bias_type="race" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m model_selection.compute_reliance_statistics \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/bert_civil_race_race_val_100/cda" \
    --split="val" \
    --bias_type="race" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m sensitive_attribution_extraction.extract_sensitive_token_attribution \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/bert_civil_all_race_val_100/cda" \
    --split="val" \
    --bias_type="race" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m model_selection.compute_reliance_statistics \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/bert_civil_all_race_val_100/cda" \
    --split="val" \
    --bias_type="race" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models/bert_civil_race/dropout" \
    --batch_size=32 \
    --max_seq_length=512 \
    --num_examples=100 \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/bert_civil_race_race_val_100/dropout" \
    --bias_type="race"

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models/bert_civil_all/dropout" \
    --batch_size=32 \
    --max_seq_length=512 \
    --num_examples=100 \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/bert_civil_all_race_val_100/dropout" \
    --bias_type="race"

python -m explanation_generation.gen_explanation_encoder \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models/bert_civil_race/dropout" \
    --num_labels=2 \
    --batch_size=32 \
    --max_length=512 \
    --num_examples=100 \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/bert_civil_race_race_val_100/dropout" \
    --baseline="pad" \
    --bias_type="race" \
    --seed=42 \
    --only_predicted_class

python -m explanation_generation.gen_explanation_encoder \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models/bert_civil_all/dropout" \
    --num_labels=2 \
    --batch_size=32 \
    --max_length=512 \
    --num_examples=100 \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/bert_civil_all_race_val_100/dropout" \
    --baseline="pad" \
    --bias_type="race" \
    --seed=42 \
    --only_predicted_class

python -m sensitive_attribution_extraction.extract_sensitive_token_attribution \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/bert_civil_race_race_val_100/dropout" \
    --split="val" \
    --bias_type="race" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m model_selection.compute_reliance_statistics \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/bert_civil_race_race_val_100/dropout" \
    --split="val" \
    --bias_type="race" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m sensitive_attribution_extraction.extract_sensitive_token_attribution \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/bert_civil_all_race_val_100/dropout" \
    --split="val" \
    --bias_type="race" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m model_selection.compute_reliance_statistics \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/bert_civil_all_race_val_100/dropout" \
    --split="val" \
    --bias_type="race" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models/bert_civil_race/attention_entropy" \
    --batch_size=32 \
    --max_seq_length=512 \
    --num_examples=100 \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/bert_civil_race_race_val_100/attention_entropy" \
    --bias_type="race"

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models/bert_civil_all/attention_entropy" \
    --batch_size=32 \
    --max_seq_length=512 \
    --num_examples=100 \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/bert_civil_all_race_val_100/attention_entropy" \
    --bias_type="race"

python -m explanation_generation.gen_explanation_encoder \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models/bert_civil_race/attention_entropy" \
    --num_labels=2 \
    --batch_size=32 \
    --max_length=512 \
    --num_examples=100 \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/bert_civil_race_race_val_100/attention_entropy" \
    --baseline="pad" \
    --bias_type="race" \
    --seed=42 \
    --only_predicted_class

python -m explanation_generation.gen_explanation_encoder \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models/bert_civil_all/attention_entropy" \
    --num_labels=2 \
    --batch_size=32 \
    --max_length=512 \
    --num_examples=100 \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/bert_civil_all_race_val_100/attention_entropy" \
    --baseline="pad" \
    --bias_type="race" \
    --seed=42 \
    --only_predicted_class

python -m sensitive_attribution_extraction.extract_sensitive_token_attribution \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/bert_civil_race_race_val_100/attention_entropy" \
    --split="val" \
    --bias_type="race" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m model_selection.compute_reliance_statistics \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/bert_civil_race_race_val_100/attention_entropy" \
    --split="val" \
    --bias_type="race" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m sensitive_attribution_extraction.extract_sensitive_token_attribution \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/bert_civil_all_race_val_100/attention_entropy" \
    --split="val" \
    --bias_type="race" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m model_selection.compute_reliance_statistics \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/bert_civil_all_race_val_100/attention_entropy" \
    --split="val" \
    --bias_type="race" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models/bert_civil_race/causal_debias" \
    --batch_size=32 \
    --max_seq_length=512 \
    --num_examples=100 \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/bert_civil_race_race_val_100/causal_debias" \
    --bias_type="race"

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models/bert_civil_all/causal_debias" \
    --batch_size=32 \
    --max_seq_length=512 \
    --num_examples=100 \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/bert_civil_all_race_val_100/causal_debias" \
    --bias_type="race"

python -m explanation_generation.gen_explanation_encoder \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models/bert_civil_race/causal_debias" \
    --num_labels=2 \
    --batch_size=32 \
    --max_length=512 \
    --num_examples=100 \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/bert_civil_race_race_val_100/causal_debias" \
    --baseline="pad" \
    --bias_type="race" \
    --seed=42 \
    --only_predicted_class

python -m explanation_generation.gen_explanation_encoder \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models/bert_civil_all/causal_debias" \
    --num_labels=2 \
    --batch_size=32 \
    --max_length=512 \
    --num_examples=100 \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/bert_civil_all_race_val_100/causal_debias" \
    --baseline="pad" \
    --bias_type="race" \
    --seed=42 \
    --only_predicted_class

python -m sensitive_attribution_extraction.extract_sensitive_token_attribution \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/bert_civil_race_race_val_100/causal_debias" \
    --split="val" \
    --bias_type="race" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m model_selection.compute_reliance_statistics \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/bert_civil_race_race_val_100/causal_debias" \
    --split="val" \
    --bias_type="race" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m sensitive_attribution_extraction.extract_sensitive_token_attribution \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/bert_civil_all_race_val_100/causal_debias" \
    --split="val" \
    --bias_type="race" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m model_selection.compute_reliance_statistics \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/bert_civil_all_race_val_100/causal_debias" \
    --split="val" \
    --bias_type="race" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m model_selection.compute_model_selection_correlation \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results" \
    --val_results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --bias_type="race" \
    --model_type="bert" \
    --num_test_examples=2000 \
    --num_val_examples=100 \
    --test_seed=-1 \
    --val_seed=-1

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models/roberta_civil_race/no_debiasing" \
    --batch_size=32 \
    --max_seq_length=512 \
    --num_examples=100 \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/roberta_civil_race_race_val_100/no_debiasing" \
    --bias_type="race"

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models/roberta_civil_all/no_debiasing" \
    --batch_size=32 \
    --max_seq_length=512 \
    --num_examples=100 \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/roberta_civil_all_race_val_100/no_debiasing" \
    --bias_type="race"

python -m explanation_generation.gen_explanation_encoder \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models/roberta_civil_race/no_debiasing" \
    --num_labels=2 \
    --batch_size=32 \
    --max_length=512 \
    --num_examples=100 \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/roberta_civil_race_race_val_100/no_debiasing" \
    --baseline="pad" \
    --bias_type="race" \
    --seed=42 \
    --only_predicted_class

python -m explanation_generation.gen_explanation_encoder \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models/roberta_civil_all/no_debiasing" \
    --num_labels=2 \
    --batch_size=32 \
    --max_length=512 \
    --num_examples=100 \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/roberta_civil_all_race_val_100/no_debiasing" \
    --baseline="pad" \
    --bias_type="race" \
    --seed=42 \
    --only_predicted_class

python -m sensitive_attribution_extraction.extract_sensitive_token_attribution \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/roberta_civil_race_race_val_100/no_debiasing" \
    --split="val" \
    --bias_type="race" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m model_selection.compute_reliance_statistics \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/roberta_civil_race_race_val_100/no_debiasing" \
    --split="val" \
    --bias_type="race" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m sensitive_attribution_extraction.extract_sensitive_token_attribution \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/roberta_civil_all_race_val_100/no_debiasing" \
    --split="val" \
    --bias_type="race" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m model_selection.compute_reliance_statistics \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/roberta_civil_all_race_val_100/no_debiasing" \
    --split="val" \
    --bias_type="race" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models/roberta_civil_race/group_balance" \
    --batch_size=32 \
    --max_seq_length=512 \
    --num_examples=100 \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/roberta_civil_race_race_val_100/group_balance" \
    --bias_type="race"

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models/roberta_civil_all/group_balance" \
    --batch_size=32 \
    --max_seq_length=512 \
    --num_examples=100 \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/roberta_civil_all_race_val_100/group_balance" \
    --bias_type="race"

python -m explanation_generation.gen_explanation_encoder \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models/roberta_civil_race/group_balance" \
    --num_labels=2 \
    --batch_size=32 \
    --max_length=512 \
    --num_examples=100 \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/roberta_civil_race_race_val_100/group_balance" \
    --baseline="pad" \
    --bias_type="race" \
    --seed=42 \
    --only_predicted_class

python -m explanation_generation.gen_explanation_encoder \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models/roberta_civil_all/group_balance" \
    --num_labels=2 \
    --batch_size=32 \
    --max_length=512 \
    --num_examples=100 \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/roberta_civil_all_race_val_100/group_balance" \
    --baseline="pad" \
    --bias_type="race" \
    --seed=42 \
    --only_predicted_class

python -m sensitive_attribution_extraction.extract_sensitive_token_attribution \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/roberta_civil_race_race_val_100/group_balance" \
    --split="val" \
    --bias_type="race" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m model_selection.compute_reliance_statistics \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/roberta_civil_race_race_val_100/group_balance" \
    --split="val" \
    --bias_type="race" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m sensitive_attribution_extraction.extract_sensitive_token_attribution \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/roberta_civil_all_race_val_100/group_balance" \
    --split="val" \
    --bias_type="race" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m model_selection.compute_reliance_statistics \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/roberta_civil_all_race_val_100/group_balance" \
    --split="val" \
    --bias_type="race" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models/roberta_civil_race/group_class_balance" \
    --batch_size=32 \
    --max_seq_length=512 \
    --num_examples=100 \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/roberta_civil_race_race_val_100/group_class_balance" \
    --bias_type="race"

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models/roberta_civil_all/group_class_balance" \
    --batch_size=32 \
    --max_seq_length=512 \
    --num_examples=100 \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/roberta_civil_all_race_val_100/group_class_balance" \
    --bias_type="race"

python -m explanation_generation.gen_explanation_encoder \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models/roberta_civil_race/group_class_balance" \
    --num_labels=2 \
    --batch_size=32 \
    --max_length=512 \
    --num_examples=100 \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/roberta_civil_race_race_val_100/group_class_balance" \
    --baseline="pad" \
    --bias_type="race" \
    --seed=42 \
    --only_predicted_class

python -m explanation_generation.gen_explanation_encoder \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models/roberta_civil_all/group_class_balance" \
    --num_labels=2 \
    --batch_size=32 \
    --max_length=512 \
    --num_examples=100 \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/roberta_civil_all_race_val_100/group_class_balance" \
    --baseline="pad" \
    --bias_type="race" \
    --seed=42 \
    --only_predicted_class

python -m sensitive_attribution_extraction.extract_sensitive_token_attribution \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/roberta_civil_race_race_val_100/group_class_balance" \
    --split="val" \
    --bias_type="race" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m model_selection.compute_reliance_statistics \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/roberta_civil_race_race_val_100/group_class_balance" \
    --split="val" \
    --bias_type="race" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m sensitive_attribution_extraction.extract_sensitive_token_attribution \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/roberta_civil_all_race_val_100/group_class_balance" \
    --split="val" \
    --bias_type="race" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m model_selection.compute_reliance_statistics \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/roberta_civil_all_race_val_100/group_class_balance" \
    --split="val" \
    --bias_type="race" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models/roberta_civil_race/cda" \
    --batch_size=32 \
    --max_seq_length=512 \
    --num_examples=100 \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/roberta_civil_race_race_val_100/cda" \
    --bias_type="race"

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models/roberta_civil_all/cda" \
    --batch_size=32 \
    --max_seq_length=512 \
    --num_examples=100 \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/roberta_civil_all_race_val_100/cda" \
    --bias_type="race"

python -m explanation_generation.gen_explanation_encoder \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models/roberta_civil_race/cda" \
    --num_labels=2 \
    --batch_size=32 \
    --max_length=512 \
    --num_examples=100 \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/roberta_civil_race_race_val_100/cda" \
    --baseline="pad" \
    --bias_type="race" \
    --seed=42 \
    --only_predicted_class

python -m explanation_generation.gen_explanation_encoder \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models/roberta_civil_all/cda" \
    --num_labels=2 \
    --batch_size=32 \
    --max_length=512 \
    --num_examples=100 \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/roberta_civil_all_race_val_100/cda" \
    --baseline="pad" \
    --bias_type="race" \
    --seed=42 \
    --only_predicted_class

python -m sensitive_attribution_extraction.extract_sensitive_token_attribution \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/roberta_civil_race_race_val_100/cda" \
    --split="val" \
    --bias_type="race" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m model_selection.compute_reliance_statistics \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/roberta_civil_race_race_val_100/cda" \
    --split="val" \
    --bias_type="race" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m sensitive_attribution_extraction.extract_sensitive_token_attribution \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/roberta_civil_all_race_val_100/cda" \
    --split="val" \
    --bias_type="race" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m model_selection.compute_reliance_statistics \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/roberta_civil_all_race_val_100/cda" \
    --split="val" \
    --bias_type="race" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models/roberta_civil_race/dropout" \
    --batch_size=32 \
    --max_seq_length=512 \
    --num_examples=100 \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/roberta_civil_race_race_val_100/dropout" \
    --bias_type="race"

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models/roberta_civil_all/dropout" \
    --batch_size=32 \
    --max_seq_length=512 \
    --num_examples=100 \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/roberta_civil_all_race_val_100/dropout" \
    --bias_type="race"

python -m explanation_generation.gen_explanation_encoder \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models/roberta_civil_race/dropout" \
    --num_labels=2 \
    --batch_size=32 \
    --max_length=512 \
    --num_examples=100 \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/roberta_civil_race_race_val_100/dropout" \
    --baseline="pad" \
    --bias_type="race" \
    --seed=42 \
    --only_predicted_class

python -m explanation_generation.gen_explanation_encoder \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models/roberta_civil_all/dropout" \
    --num_labels=2 \
    --batch_size=32 \
    --max_length=512 \
    --num_examples=100 \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/roberta_civil_all_race_val_100/dropout" \
    --baseline="pad" \
    --bias_type="race" \
    --seed=42 \
    --only_predicted_class

python -m sensitive_attribution_extraction.extract_sensitive_token_attribution \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/roberta_civil_race_race_val_100/dropout" \
    --split="val" \
    --bias_type="race" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m model_selection.compute_reliance_statistics \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/roberta_civil_race_race_val_100/dropout" \
    --split="val" \
    --bias_type="race" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m sensitive_attribution_extraction.extract_sensitive_token_attribution \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/roberta_civil_all_race_val_100/dropout" \
    --split="val" \
    --bias_type="race" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m model_selection.compute_reliance_statistics \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/roberta_civil_all_race_val_100/dropout" \
    --split="val" \
    --bias_type="race" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models/roberta_civil_race/attention_entropy" \
    --batch_size=32 \
    --max_seq_length=512 \
    --num_examples=100 \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/roberta_civil_race_race_val_100/attention_entropy" \
    --bias_type="race"

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models/roberta_civil_all/attention_entropy" \
    --batch_size=32 \
    --max_seq_length=512 \
    --num_examples=100 \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/roberta_civil_all_race_val_100/attention_entropy" \
    --bias_type="race"

python -m explanation_generation.gen_explanation_encoder \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models/roberta_civil_race/attention_entropy" \
    --num_labels=2 \
    --batch_size=32 \
    --max_length=512 \
    --num_examples=100 \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/roberta_civil_race_race_val_100/attention_entropy" \
    --baseline="pad" \
    --bias_type="race" \
    --seed=42 \
    --only_predicted_class

python -m explanation_generation.gen_explanation_encoder \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models/roberta_civil_all/attention_entropy" \
    --num_labels=2 \
    --batch_size=32 \
    --max_length=512 \
    --num_examples=100 \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/roberta_civil_all_race_val_100/attention_entropy" \
    --baseline="pad" \
    --bias_type="race" \
    --seed=42 \
    --only_predicted_class

python -m sensitive_attribution_extraction.extract_sensitive_token_attribution \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/roberta_civil_race_race_val_100/attention_entropy" \
    --split="val" \
    --bias_type="race" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m model_selection.compute_reliance_statistics \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/roberta_civil_race_race_val_100/attention_entropy" \
    --split="val" \
    --bias_type="race" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m sensitive_attribution_extraction.extract_sensitive_token_attribution \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/roberta_civil_all_race_val_100/attention_entropy" \
    --split="val" \
    --bias_type="race" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m model_selection.compute_reliance_statistics \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/roberta_civil_all_race_val_100/attention_entropy" \
    --split="val" \
    --bias_type="race" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models/roberta_civil_race/causal_debias" \
    --batch_size=32 \
    --max_seq_length=512 \
    --num_examples=100 \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/roberta_civil_race_race_val_100/causal_debias" \
    --bias_type="race"

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models/roberta_civil_all/causal_debias" \
    --batch_size=32 \
    --max_seq_length=512 \
    --num_examples=100 \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/roberta_civil_all_race_val_100/causal_debias" \
    --bias_type="race"

python -m explanation_generation.gen_explanation_encoder \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models/roberta_civil_race/causal_debias" \
    --num_labels=2 \
    --batch_size=32 \
    --max_length=512 \
    --num_examples=100 \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/roberta_civil_race_race_val_100/causal_debias" \
    --baseline="pad" \
    --bias_type="race" \
    --seed=42 \
    --only_predicted_class

python -m explanation_generation.gen_explanation_encoder \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models/roberta_civil_all/causal_debias" \
    --num_labels=2 \
    --batch_size=32 \
    --max_length=512 \
    --num_examples=100 \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/roberta_civil_all_race_val_100/causal_debias" \
    --baseline="pad" \
    --bias_type="race" \
    --seed=42 \
    --only_predicted_class

python -m sensitive_attribution_extraction.extract_sensitive_token_attribution \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/roberta_civil_race_race_val_100/causal_debias" \
    --split="val" \
    --bias_type="race" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m model_selection.compute_reliance_statistics \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/roberta_civil_race_race_val_100/causal_debias" \
    --split="val" \
    --bias_type="race" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m sensitive_attribution_extraction.extract_sensitive_token_attribution \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/roberta_civil_all_race_val_100/causal_debias" \
    --split="val" \
    --bias_type="race" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m model_selection.compute_reliance_statistics \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/roberta_civil_all_race_val_100/causal_debias" \
    --split="val" \
    --bias_type="race" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m model_selection.compute_model_selection_correlation \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results" \
    --val_results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --bias_type="race" \
    --model_type="roberta" \
    --num_test_examples=2000 \
    --num_val_examples=100 \
    --test_seed=-1 \
    --val_seed=-1

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models/distilbert_civil_race/no_debiasing" \
    --batch_size=32 \
    --max_seq_length=512 \
    --num_examples=100 \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/distilbert_civil_race_race_val_100/no_debiasing" \
    --bias_type="race"

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models/distilbert_civil_all/no_debiasing" \
    --batch_size=32 \
    --max_seq_length=512 \
    --num_examples=100 \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/distilbert_civil_all_race_val_100/no_debiasing" \
    --bias_type="race"

python -m explanation_generation.gen_explanation_encoder \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models/distilbert_civil_race/no_debiasing" \
    --num_labels=2 \
    --batch_size=32 \
    --max_length=512 \
    --num_examples=100 \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/distilbert_civil_race_race_val_100/no_debiasing" \
    --baseline="pad" \
    --bias_type="race" \
    --seed=42 \
    --only_predicted_class

python -m explanation_generation.gen_explanation_encoder \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models/distilbert_civil_all/no_debiasing" \
    --num_labels=2 \
    --batch_size=32 \
    --max_length=512 \
    --num_examples=100 \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/distilbert_civil_all_race_val_100/no_debiasing" \
    --baseline="pad" \
    --bias_type="race" \
    --seed=42 \
    --only_predicted_class

python -m sensitive_attribution_extraction.extract_sensitive_token_attribution \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/distilbert_civil_race_race_val_100/no_debiasing" \
    --split="val" \
    --bias_type="race" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m model_selection.compute_reliance_statistics \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/distilbert_civil_race_race_val_100/no_debiasing" \
    --split="val" \
    --bias_type="race" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m sensitive_attribution_extraction.extract_sensitive_token_attribution \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/distilbert_civil_all_race_val_100/no_debiasing" \
    --split="val" \
    --bias_type="race" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m model_selection.compute_reliance_statistics \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/distilbert_civil_all_race_val_100/no_debiasing" \
    --split="val" \
    --bias_type="race" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models/distilbert_civil_race/group_balance" \
    --batch_size=32 \
    --max_seq_length=512 \
    --num_examples=100 \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/distilbert_civil_race_race_val_100/group_balance" \
    --bias_type="race"

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models/distilbert_civil_all/group_balance" \
    --batch_size=32 \
    --max_seq_length=512 \
    --num_examples=100 \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/distilbert_civil_all_race_val_100/group_balance" \
    --bias_type="race"

python -m explanation_generation.gen_explanation_encoder \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models/distilbert_civil_race/group_balance" \
    --num_labels=2 \
    --batch_size=32 \
    --max_length=512 \
    --num_examples=100 \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/distilbert_civil_race_race_val_100/group_balance" \
    --baseline="pad" \
    --bias_type="race" \
    --seed=42 \
    --only_predicted_class

python -m explanation_generation.gen_explanation_encoder \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models/distilbert_civil_all/group_balance" \
    --num_labels=2 \
    --batch_size=32 \
    --max_length=512 \
    --num_examples=100 \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/distilbert_civil_all_race_val_100/group_balance" \
    --baseline="pad" \
    --bias_type="race" \
    --seed=42 \
    --only_predicted_class

python -m sensitive_attribution_extraction.extract_sensitive_token_attribution \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/distilbert_civil_race_race_val_100/group_balance" \
    --split="val" \
    --bias_type="race" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m model_selection.compute_reliance_statistics \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/distilbert_civil_race_race_val_100/group_balance" \
    --split="val" \
    --bias_type="race" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m sensitive_attribution_extraction.extract_sensitive_token_attribution \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/distilbert_civil_all_race_val_100/group_balance" \
    --split="val" \
    --bias_type="race" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m model_selection.compute_reliance_statistics \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/distilbert_civil_all_race_val_100/group_balance" \
    --split="val" \
    --bias_type="race" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models/distilbert_civil_race/group_class_balance" \
    --batch_size=32 \
    --max_seq_length=512 \
    --num_examples=100 \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/distilbert_civil_race_race_val_100/group_class_balance" \
    --bias_type="race"

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models/distilbert_civil_all/group_class_balance" \
    --batch_size=32 \
    --max_seq_length=512 \
    --num_examples=100 \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/distilbert_civil_all_race_val_100/group_class_balance" \
    --bias_type="race"

python -m explanation_generation.gen_explanation_encoder \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models/distilbert_civil_race/group_class_balance" \
    --num_labels=2 \
    --batch_size=32 \
    --max_length=512 \
    --num_examples=100 \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/distilbert_civil_race_race_val_100/group_class_balance" \
    --baseline="pad" \
    --bias_type="race" \
    --seed=42 \
    --only_predicted_class

python -m explanation_generation.gen_explanation_encoder \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models/distilbert_civil_all/group_class_balance" \
    --num_labels=2 \
    --batch_size=32 \
    --max_length=512 \
    --num_examples=100 \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/distilbert_civil_all_race_val_100/group_class_balance" \
    --baseline="pad" \
    --bias_type="race" \
    --seed=42 \
    --only_predicted_class

python -m sensitive_attribution_extraction.extract_sensitive_token_attribution \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/distilbert_civil_race_race_val_100/group_class_balance" \
    --split="val" \
    --bias_type="race" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m model_selection.compute_reliance_statistics \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/distilbert_civil_race_race_val_100/group_class_balance" \
    --split="val" \
    --bias_type="race" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m sensitive_attribution_extraction.extract_sensitive_token_attribution \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/distilbert_civil_all_race_val_100/group_class_balance" \
    --split="val" \
    --bias_type="race" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m model_selection.compute_reliance_statistics \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/distilbert_civil_all_race_val_100/group_class_balance" \
    --split="val" \
    --bias_type="race" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models/distilbert_civil_race/cda" \
    --batch_size=32 \
    --max_seq_length=512 \
    --num_examples=100 \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/distilbert_civil_race_race_val_100/cda" \
    --bias_type="race"

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models/distilbert_civil_all/cda" \
    --batch_size=32 \
    --max_seq_length=512 \
    --num_examples=100 \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/distilbert_civil_all_race_val_100/cda" \
    --bias_type="race"

python -m explanation_generation.gen_explanation_encoder \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models/distilbert_civil_race/cda" \
    --num_labels=2 \
    --batch_size=32 \
    --max_length=512 \
    --num_examples=100 \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/distilbert_civil_race_race_val_100/cda" \
    --baseline="pad" \
    --bias_type="race" \
    --seed=42 \
    --only_predicted_class

python -m explanation_generation.gen_explanation_encoder \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models/distilbert_civil_all/cda" \
    --num_labels=2 \
    --batch_size=32 \
    --max_length=512 \
    --num_examples=100 \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/distilbert_civil_all_race_val_100/cda" \
    --baseline="pad" \
    --bias_type="race" \
    --seed=42 \
    --only_predicted_class

python -m sensitive_attribution_extraction.extract_sensitive_token_attribution \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/distilbert_civil_race_race_val_100/cda" \
    --split="val" \
    --bias_type="race" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m model_selection.compute_reliance_statistics \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/distilbert_civil_race_race_val_100/cda" \
    --split="val" \
    --bias_type="race" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m sensitive_attribution_extraction.extract_sensitive_token_attribution \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/distilbert_civil_all_race_val_100/cda" \
    --split="val" \
    --bias_type="race" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m model_selection.compute_reliance_statistics \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/distilbert_civil_all_race_val_100/cda" \
    --split="val" \
    --bias_type="race" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models/distilbert_civil_race/dropout" \
    --batch_size=32 \
    --max_seq_length=512 \
    --num_examples=100 \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/distilbert_civil_race_race_val_100/dropout" \
    --bias_type="race"

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models/distilbert_civil_all/dropout" \
    --batch_size=32 \
    --max_seq_length=512 \
    --num_examples=100 \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/distilbert_civil_all_race_val_100/dropout" \
    --bias_type="race"

python -m explanation_generation.gen_explanation_encoder \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models/distilbert_civil_race/dropout" \
    --num_labels=2 \
    --batch_size=32 \
    --max_length=512 \
    --num_examples=100 \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/distilbert_civil_race_race_val_100/dropout" \
    --baseline="pad" \
    --bias_type="race" \
    --seed=42 \
    --only_predicted_class

python -m explanation_generation.gen_explanation_encoder \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models/distilbert_civil_all/dropout" \
    --num_labels=2 \
    --batch_size=32 \
    --max_length=512 \
    --num_examples=100 \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/distilbert_civil_all_race_val_100/dropout" \
    --baseline="pad" \
    --bias_type="race" \
    --seed=42 \
    --only_predicted_class

python -m sensitive_attribution_extraction.extract_sensitive_token_attribution \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/distilbert_civil_race_race_val_100/dropout" \
    --split="val" \
    --bias_type="race" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m model_selection.compute_reliance_statistics \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/distilbert_civil_race_race_val_100/dropout" \
    --split="val" \
    --bias_type="race" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m sensitive_attribution_extraction.extract_sensitive_token_attribution \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/distilbert_civil_all_race_val_100/dropout" \
    --split="val" \
    --bias_type="race" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m model_selection.compute_reliance_statistics \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/distilbert_civil_all_race_val_100/dropout" \
    --split="val" \
    --bias_type="race" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models/distilbert_civil_race/attention_entropy" \
    --batch_size=32 \
    --max_seq_length=512 \
    --num_examples=100 \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/distilbert_civil_race_race_val_100/attention_entropy" \
    --bias_type="race"

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models/distilbert_civil_all/attention_entropy" \
    --batch_size=32 \
    --max_seq_length=512 \
    --num_examples=100 \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/distilbert_civil_all_race_val_100/attention_entropy" \
    --bias_type="race"

python -m explanation_generation.gen_explanation_encoder \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models/distilbert_civil_race/attention_entropy" \
    --num_labels=2 \
    --batch_size=32 \
    --max_length=512 \
    --num_examples=100 \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/distilbert_civil_race_race_val_100/attention_entropy" \
    --baseline="pad" \
    --bias_type="race" \
    --seed=42 \
    --only_predicted_class

python -m explanation_generation.gen_explanation_encoder \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models/distilbert_civil_all/attention_entropy" \
    --num_labels=2 \
    --batch_size=32 \
    --max_length=512 \
    --num_examples=100 \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/distilbert_civil_all_race_val_100/attention_entropy" \
    --baseline="pad" \
    --bias_type="race" \
    --seed=42 \
    --only_predicted_class

python -m sensitive_attribution_extraction.extract_sensitive_token_attribution \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/distilbert_civil_race_race_val_100/attention_entropy" \
    --split="val" \
    --bias_type="race" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m model_selection.compute_reliance_statistics \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/distilbert_civil_race_race_val_100/attention_entropy" \
    --split="val" \
    --bias_type="race" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m sensitive_attribution_extraction.extract_sensitive_token_attribution \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/distilbert_civil_all_race_val_100/attention_entropy" \
    --split="val" \
    --bias_type="race" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m model_selection.compute_reliance_statistics \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/distilbert_civil_all_race_val_100/attention_entropy" \
    --split="val" \
    --bias_type="race" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models/distilbert_civil_race/causal_debias" \
    --batch_size=32 \
    --max_seq_length=512 \
    --num_examples=100 \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/distilbert_civil_race_race_val_100/causal_debias" \
    --bias_type="race"

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models/distilbert_civil_all/causal_debias" \
    --batch_size=32 \
    --max_seq_length=512 \
    --num_examples=100 \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/distilbert_civil_all_race_val_100/causal_debias" \
    --bias_type="race"

python -m explanation_generation.gen_explanation_encoder \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models/distilbert_civil_race/causal_debias" \
    --num_labels=2 \
    --batch_size=32 \
    --max_length=512 \
    --num_examples=100 \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/distilbert_civil_race_race_val_100/causal_debias" \
    --baseline="pad" \
    --bias_type="race" \
    --seed=42 \
    --only_predicted_class

python -m explanation_generation.gen_explanation_encoder \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models/distilbert_civil_all/causal_debias" \
    --num_labels=2 \
    --batch_size=32 \
    --max_length=512 \
    --num_examples=100 \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/distilbert_civil_all_race_val_100/causal_debias" \
    --baseline="pad" \
    --bias_type="race" \
    --seed=42 \
    --only_predicted_class

python -m sensitive_attribution_extraction.extract_sensitive_token_attribution \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/distilbert_civil_race_race_val_100/causal_debias" \
    --split="val" \
    --bias_type="race" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m model_selection.compute_reliance_statistics \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/distilbert_civil_race_race_val_100/causal_debias" \
    --split="val" \
    --bias_type="race" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m sensitive_attribution_extraction.extract_sensitive_token_attribution \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/distilbert_civil_all_race_val_100/causal_debias" \
    --split="val" \
    --bias_type="race" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m model_selection.compute_reliance_statistics \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/distilbert_civil_all_race_val_100/causal_debias" \
    --split="val" \
    --bias_type="race" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m model_selection.compute_model_selection_correlation \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results" \
    --val_results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --bias_type="race" \
    --model_type="distilbert" \
    --num_test_examples=2000 \
    --num_val_examples=100 \
    --test_seed=-1 \
    --val_seed=-1

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models/bert_civil_race/no_debiasing" \
    --batch_size=32 \
    --max_seq_length=512 \
    --num_examples=200 \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/bert_civil_race_race_val_200/no_debiasing" \
    --bias_type="race"

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models/bert_civil_all/no_debiasing" \
    --batch_size=32 \
    --max_seq_length=512 \
    --num_examples=200 \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/bert_civil_all_race_val_200/no_debiasing" \
    --bias_type="race"

python -m explanation_generation.gen_explanation_encoder \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models/bert_civil_race/no_debiasing" \
    --num_labels=2 \
    --batch_size=32 \
    --max_length=512 \
    --num_examples=200 \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/bert_civil_race_race_val_200/no_debiasing" \
    --baseline="pad" \
    --bias_type="race" \
    --seed=42 \
    --only_predicted_class

python -m explanation_generation.gen_explanation_encoder \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models/bert_civil_all/no_debiasing" \
    --num_labels=2 \
    --batch_size=32 \
    --max_length=512 \
    --num_examples=200 \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/bert_civil_all_race_val_200/no_debiasing" \
    --baseline="pad" \
    --bias_type="race" \
    --seed=42 \
    --only_predicted_class

python -m sensitive_attribution_extraction.extract_sensitive_token_attribution \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/bert_civil_race_race_val_200/no_debiasing" \
    --split="val" \
    --bias_type="race" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m model_selection.compute_reliance_statistics \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/bert_civil_race_race_val_200/no_debiasing" \
    --split="val" \
    --bias_type="race" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m sensitive_attribution_extraction.extract_sensitive_token_attribution \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/bert_civil_all_race_val_200/no_debiasing" \
    --split="val" \
    --bias_type="race" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m model_selection.compute_reliance_statistics \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/bert_civil_all_race_val_200/no_debiasing" \
    --split="val" \
    --bias_type="race" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models/bert_civil_race/group_balance" \
    --batch_size=32 \
    --max_seq_length=512 \
    --num_examples=200 \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/bert_civil_race_race_val_200/group_balance" \
    --bias_type="race"

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models/bert_civil_all/group_balance" \
    --batch_size=32 \
    --max_seq_length=512 \
    --num_examples=200 \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/bert_civil_all_race_val_200/group_balance" \
    --bias_type="race"

python -m explanation_generation.gen_explanation_encoder \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models/bert_civil_race/group_balance" \
    --num_labels=2 \
    --batch_size=32 \
    --max_length=512 \
    --num_examples=200 \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/bert_civil_race_race_val_200/group_balance" \
    --baseline="pad" \
    --bias_type="race" \
    --seed=42 \
    --only_predicted_class

python -m explanation_generation.gen_explanation_encoder \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models/bert_civil_all/group_balance" \
    --num_labels=2 \
    --batch_size=32 \
    --max_length=512 \
    --num_examples=200 \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/bert_civil_all_race_val_200/group_balance" \
    --baseline="pad" \
    --bias_type="race" \
    --seed=42 \
    --only_predicted_class

python -m sensitive_attribution_extraction.extract_sensitive_token_attribution \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/bert_civil_race_race_val_200/group_balance" \
    --split="val" \
    --bias_type="race" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m model_selection.compute_reliance_statistics \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/bert_civil_race_race_val_200/group_balance" \
    --split="val" \
    --bias_type="race" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m sensitive_attribution_extraction.extract_sensitive_token_attribution \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/bert_civil_all_race_val_200/group_balance" \
    --split="val" \
    --bias_type="race" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m model_selection.compute_reliance_statistics \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/bert_civil_all_race_val_200/group_balance" \
    --split="val" \
    --bias_type="race" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models/bert_civil_race/group_class_balance" \
    --batch_size=32 \
    --max_seq_length=512 \
    --num_examples=200 \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/bert_civil_race_race_val_200/group_class_balance" \
    --bias_type="race"

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models/bert_civil_all/group_class_balance" \
    --batch_size=32 \
    --max_seq_length=512 \
    --num_examples=200 \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/bert_civil_all_race_val_200/group_class_balance" \
    --bias_type="race"

python -m explanation_generation.gen_explanation_encoder \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models/bert_civil_race/group_class_balance" \
    --num_labels=2 \
    --batch_size=32 \
    --max_length=512 \
    --num_examples=200 \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/bert_civil_race_race_val_200/group_class_balance" \
    --baseline="pad" \
    --bias_type="race" \
    --seed=42 \
    --only_predicted_class

python -m explanation_generation.gen_explanation_encoder \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models/bert_civil_all/group_class_balance" \
    --num_labels=2 \
    --batch_size=32 \
    --max_length=512 \
    --num_examples=200 \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/bert_civil_all_race_val_200/group_class_balance" \
    --baseline="pad" \
    --bias_type="race" \
    --seed=42 \
    --only_predicted_class

python -m sensitive_attribution_extraction.extract_sensitive_token_attribution \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/bert_civil_race_race_val_200/group_class_balance" \
    --split="val" \
    --bias_type="race" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m model_selection.compute_reliance_statistics \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/bert_civil_race_race_val_200/group_class_balance" \
    --split="val" \
    --bias_type="race" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m sensitive_attribution_extraction.extract_sensitive_token_attribution \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/bert_civil_all_race_val_200/group_class_balance" \
    --split="val" \
    --bias_type="race" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m model_selection.compute_reliance_statistics \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/bert_civil_all_race_val_200/group_class_balance" \
    --split="val" \
    --bias_type="race" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models/bert_civil_race/cda" \
    --batch_size=32 \
    --max_seq_length=512 \
    --num_examples=200 \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/bert_civil_race_race_val_200/cda" \
    --bias_type="race"

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models/bert_civil_all/cda" \
    --batch_size=32 \
    --max_seq_length=512 \
    --num_examples=200 \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/bert_civil_all_race_val_200/cda" \
    --bias_type="race"

python -m explanation_generation.gen_explanation_encoder \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models/bert_civil_race/cda" \
    --num_labels=2 \
    --batch_size=32 \
    --max_length=512 \
    --num_examples=200 \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/bert_civil_race_race_val_200/cda" \
    --baseline="pad" \
    --bias_type="race" \
    --seed=42 \
    --only_predicted_class

python -m explanation_generation.gen_explanation_encoder \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models/bert_civil_all/cda" \
    --num_labels=2 \
    --batch_size=32 \
    --max_length=512 \
    --num_examples=200 \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/bert_civil_all_race_val_200/cda" \
    --baseline="pad" \
    --bias_type="race" \
    --seed=42 \
    --only_predicted_class

python -m sensitive_attribution_extraction.extract_sensitive_token_attribution \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/bert_civil_race_race_val_200/cda" \
    --split="val" \
    --bias_type="race" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m model_selection.compute_reliance_statistics \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/bert_civil_race_race_val_200/cda" \
    --split="val" \
    --bias_type="race" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m sensitive_attribution_extraction.extract_sensitive_token_attribution \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/bert_civil_all_race_val_200/cda" \
    --split="val" \
    --bias_type="race" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m model_selection.compute_reliance_statistics \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/bert_civil_all_race_val_200/cda" \
    --split="val" \
    --bias_type="race" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models/bert_civil_race/dropout" \
    --batch_size=32 \
    --max_seq_length=512 \
    --num_examples=200 \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/bert_civil_race_race_val_200/dropout" \
    --bias_type="race"

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models/bert_civil_all/dropout" \
    --batch_size=32 \
    --max_seq_length=512 \
    --num_examples=200 \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/bert_civil_all_race_val_200/dropout" \
    --bias_type="race"

python -m explanation_generation.gen_explanation_encoder \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models/bert_civil_race/dropout" \
    --num_labels=2 \
    --batch_size=32 \
    --max_length=512 \
    --num_examples=200 \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/bert_civil_race_race_val_200/dropout" \
    --baseline="pad" \
    --bias_type="race" \
    --seed=42 \
    --only_predicted_class

python -m explanation_generation.gen_explanation_encoder \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models/bert_civil_all/dropout" \
    --num_labels=2 \
    --batch_size=32 \
    --max_length=512 \
    --num_examples=200 \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/bert_civil_all_race_val_200/dropout" \
    --baseline="pad" \
    --bias_type="race" \
    --seed=42 \
    --only_predicted_class

python -m sensitive_attribution_extraction.extract_sensitive_token_attribution \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/bert_civil_race_race_val_200/dropout" \
    --split="val" \
    --bias_type="race" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m model_selection.compute_reliance_statistics \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/bert_civil_race_race_val_200/dropout" \
    --split="val" \
    --bias_type="race" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m sensitive_attribution_extraction.extract_sensitive_token_attribution \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/bert_civil_all_race_val_200/dropout" \
    --split="val" \
    --bias_type="race" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m model_selection.compute_reliance_statistics \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/bert_civil_all_race_val_200/dropout" \
    --split="val" \
    --bias_type="race" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models/bert_civil_race/attention_entropy" \
    --batch_size=32 \
    --max_seq_length=512 \
    --num_examples=200 \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/bert_civil_race_race_val_200/attention_entropy" \
    --bias_type="race"

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models/bert_civil_all/attention_entropy" \
    --batch_size=32 \
    --max_seq_length=512 \
    --num_examples=200 \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/bert_civil_all_race_val_200/attention_entropy" \
    --bias_type="race"

python -m explanation_generation.gen_explanation_encoder \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models/bert_civil_race/attention_entropy" \
    --num_labels=2 \
    --batch_size=32 \
    --max_length=512 \
    --num_examples=200 \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/bert_civil_race_race_val_200/attention_entropy" \
    --baseline="pad" \
    --bias_type="race" \
    --seed=42 \
    --only_predicted_class

python -m explanation_generation.gen_explanation_encoder \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models/bert_civil_all/attention_entropy" \
    --num_labels=2 \
    --batch_size=32 \
    --max_length=512 \
    --num_examples=200 \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/bert_civil_all_race_val_200/attention_entropy" \
    --baseline="pad" \
    --bias_type="race" \
    --seed=42 \
    --only_predicted_class

python -m sensitive_attribution_extraction.extract_sensitive_token_attribution \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/bert_civil_race_race_val_200/attention_entropy" \
    --split="val" \
    --bias_type="race" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m model_selection.compute_reliance_statistics \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/bert_civil_race_race_val_200/attention_entropy" \
    --split="val" \
    --bias_type="race" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m sensitive_attribution_extraction.extract_sensitive_token_attribution \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/bert_civil_all_race_val_200/attention_entropy" \
    --split="val" \
    --bias_type="race" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m model_selection.compute_reliance_statistics \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/bert_civil_all_race_val_200/attention_entropy" \
    --split="val" \
    --bias_type="race" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models/bert_civil_race/causal_debias" \
    --batch_size=32 \
    --max_seq_length=512 \
    --num_examples=200 \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/bert_civil_race_race_val_200/causal_debias" \
    --bias_type="race"

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models/bert_civil_all/causal_debias" \
    --batch_size=32 \
    --max_seq_length=512 \
    --num_examples=200 \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/bert_civil_all_race_val_200/causal_debias" \
    --bias_type="race"

python -m explanation_generation.gen_explanation_encoder \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models/bert_civil_race/causal_debias" \
    --num_labels=2 \
    --batch_size=32 \
    --max_length=512 \
    --num_examples=200 \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/bert_civil_race_race_val_200/causal_debias" \
    --baseline="pad" \
    --bias_type="race" \
    --seed=42 \
    --only_predicted_class

python -m explanation_generation.gen_explanation_encoder \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models/bert_civil_all/causal_debias" \
    --num_labels=2 \
    --batch_size=32 \
    --max_length=512 \
    --num_examples=200 \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/bert_civil_all_race_val_200/causal_debias" \
    --baseline="pad" \
    --bias_type="race" \
    --seed=42 \
    --only_predicted_class

python -m sensitive_attribution_extraction.extract_sensitive_token_attribution \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/bert_civil_race_race_val_200/causal_debias" \
    --split="val" \
    --bias_type="race" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m model_selection.compute_reliance_statistics \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/bert_civil_race_race_val_200/causal_debias" \
    --split="val" \
    --bias_type="race" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m sensitive_attribution_extraction.extract_sensitive_token_attribution \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/bert_civil_all_race_val_200/causal_debias" \
    --split="val" \
    --bias_type="race" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m model_selection.compute_reliance_statistics \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/bert_civil_all_race_val_200/causal_debias" \
    --split="val" \
    --bias_type="race" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m model_selection.compute_model_selection_correlation \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results" \
    --val_results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --bias_type="race" \
    --model_type="bert" \
    --num_test_examples=2000 \
    --num_val_examples=200 \
    --test_seed=-1 \
    --val_seed=-1

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models/roberta_civil_race/no_debiasing" \
    --batch_size=32 \
    --max_seq_length=512 \
    --num_examples=200 \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/roberta_civil_race_race_val_200/no_debiasing" \
    --bias_type="race"

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models/roberta_civil_all/no_debiasing" \
    --batch_size=32 \
    --max_seq_length=512 \
    --num_examples=200 \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/roberta_civil_all_race_val_200/no_debiasing" \
    --bias_type="race"

python -m explanation_generation.gen_explanation_encoder \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models/roberta_civil_race/no_debiasing" \
    --num_labels=2 \
    --batch_size=32 \
    --max_length=512 \
    --num_examples=200 \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/roberta_civil_race_race_val_200/no_debiasing" \
    --baseline="pad" \
    --bias_type="race" \
    --seed=42 \
    --only_predicted_class

python -m explanation_generation.gen_explanation_encoder \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models/roberta_civil_all/no_debiasing" \
    --num_labels=2 \
    --batch_size=32 \
    --max_length=512 \
    --num_examples=200 \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/roberta_civil_all_race_val_200/no_debiasing" \
    --baseline="pad" \
    --bias_type="race" \
    --seed=42 \
    --only_predicted_class

python -m sensitive_attribution_extraction.extract_sensitive_token_attribution \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/roberta_civil_race_race_val_200/no_debiasing" \
    --split="val" \
    --bias_type="race" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m model_selection.compute_reliance_statistics \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/roberta_civil_race_race_val_200/no_debiasing" \
    --split="val" \
    --bias_type="race" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m sensitive_attribution_extraction.extract_sensitive_token_attribution \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/roberta_civil_all_race_val_200/no_debiasing" \
    --split="val" \
    --bias_type="race" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m model_selection.compute_reliance_statistics \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/roberta_civil_all_race_val_200/no_debiasing" \
    --split="val" \
    --bias_type="race" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models/roberta_civil_race/group_balance" \
    --batch_size=32 \
    --max_seq_length=512 \
    --num_examples=200 \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/roberta_civil_race_race_val_200/group_balance" \
    --bias_type="race"

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models/roberta_civil_all/group_balance" \
    --batch_size=32 \
    --max_seq_length=512 \
    --num_examples=200 \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/roberta_civil_all_race_val_200/group_balance" \
    --bias_type="race"

python -m explanation_generation.gen_explanation_encoder \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models/roberta_civil_race/group_balance" \
    --num_labels=2 \
    --batch_size=32 \
    --max_length=512 \
    --num_examples=200 \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/roberta_civil_race_race_val_200/group_balance" \
    --baseline="pad" \
    --bias_type="race" \
    --seed=42 \
    --only_predicted_class

python -m explanation_generation.gen_explanation_encoder \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models/roberta_civil_all/group_balance" \
    --num_labels=2 \
    --batch_size=32 \
    --max_length=512 \
    --num_examples=200 \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/roberta_civil_all_race_val_200/group_balance" \
    --baseline="pad" \
    --bias_type="race" \
    --seed=42 \
    --only_predicted_class

python -m sensitive_attribution_extraction.extract_sensitive_token_attribution \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/roberta_civil_race_race_val_200/group_balance" \
    --split="val" \
    --bias_type="race" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m model_selection.compute_reliance_statistics \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/roberta_civil_race_race_val_200/group_balance" \
    --split="val" \
    --bias_type="race" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m sensitive_attribution_extraction.extract_sensitive_token_attribution \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/roberta_civil_all_race_val_200/group_balance" \
    --split="val" \
    --bias_type="race" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m model_selection.compute_reliance_statistics \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/roberta_civil_all_race_val_200/group_balance" \
    --split="val" \
    --bias_type="race" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models/roberta_civil_race/group_class_balance" \
    --batch_size=32 \
    --max_seq_length=512 \
    --num_examples=200 \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/roberta_civil_race_race_val_200/group_class_balance" \
    --bias_type="race"

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models/roberta_civil_all/group_class_balance" \
    --batch_size=32 \
    --max_seq_length=512 \
    --num_examples=200 \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/roberta_civil_all_race_val_200/group_class_balance" \
    --bias_type="race"

python -m explanation_generation.gen_explanation_encoder \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models/roberta_civil_race/group_class_balance" \
    --num_labels=2 \
    --batch_size=32 \
    --max_length=512 \
    --num_examples=200 \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/roberta_civil_race_race_val_200/group_class_balance" \
    --baseline="pad" \
    --bias_type="race" \
    --seed=42 \
    --only_predicted_class

python -m explanation_generation.gen_explanation_encoder \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models/roberta_civil_all/group_class_balance" \
    --num_labels=2 \
    --batch_size=32 \
    --max_length=512 \
    --num_examples=200 \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/roberta_civil_all_race_val_200/group_class_balance" \
    --baseline="pad" \
    --bias_type="race" \
    --seed=42 \
    --only_predicted_class

python -m sensitive_attribution_extraction.extract_sensitive_token_attribution \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/roberta_civil_race_race_val_200/group_class_balance" \
    --split="val" \
    --bias_type="race" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m model_selection.compute_reliance_statistics \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/roberta_civil_race_race_val_200/group_class_balance" \
    --split="val" \
    --bias_type="race" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m sensitive_attribution_extraction.extract_sensitive_token_attribution \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/roberta_civil_all_race_val_200/group_class_balance" \
    --split="val" \
    --bias_type="race" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m model_selection.compute_reliance_statistics \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/roberta_civil_all_race_val_200/group_class_balance" \
    --split="val" \
    --bias_type="race" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models/roberta_civil_race/cda" \
    --batch_size=32 \
    --max_seq_length=512 \
    --num_examples=200 \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/roberta_civil_race_race_val_200/cda" \
    --bias_type="race"

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models/roberta_civil_all/cda" \
    --batch_size=32 \
    --max_seq_length=512 \
    --num_examples=200 \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/roberta_civil_all_race_val_200/cda" \
    --bias_type="race"

python -m explanation_generation.gen_explanation_encoder \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models/roberta_civil_race/cda" \
    --num_labels=2 \
    --batch_size=32 \
    --max_length=512 \
    --num_examples=200 \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/roberta_civil_race_race_val_200/cda" \
    --baseline="pad" \
    --bias_type="race" \
    --seed=42 \
    --only_predicted_class

python -m explanation_generation.gen_explanation_encoder \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models/roberta_civil_all/cda" \
    --num_labels=2 \
    --batch_size=32 \
    --max_length=512 \
    --num_examples=200 \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/roberta_civil_all_race_val_200/cda" \
    --baseline="pad" \
    --bias_type="race" \
    --seed=42 \
    --only_predicted_class

python -m sensitive_attribution_extraction.extract_sensitive_token_attribution \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/roberta_civil_race_race_val_200/cda" \
    --split="val" \
    --bias_type="race" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m model_selection.compute_reliance_statistics \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/roberta_civil_race_race_val_200/cda" \
    --split="val" \
    --bias_type="race" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m sensitive_attribution_extraction.extract_sensitive_token_attribution \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/roberta_civil_all_race_val_200/cda" \
    --split="val" \
    --bias_type="race" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m model_selection.compute_reliance_statistics \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/roberta_civil_all_race_val_200/cda" \
    --split="val" \
    --bias_type="race" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models/roberta_civil_race/dropout" \
    --batch_size=32 \
    --max_seq_length=512 \
    --num_examples=200 \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/roberta_civil_race_race_val_200/dropout" \
    --bias_type="race"

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models/roberta_civil_all/dropout" \
    --batch_size=32 \
    --max_seq_length=512 \
    --num_examples=200 \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/roberta_civil_all_race_val_200/dropout" \
    --bias_type="race"

python -m explanation_generation.gen_explanation_encoder \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models/roberta_civil_race/dropout" \
    --num_labels=2 \
    --batch_size=32 \
    --max_length=512 \
    --num_examples=200 \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/roberta_civil_race_race_val_200/dropout" \
    --baseline="pad" \
    --bias_type="race" \
    --seed=42 \
    --only_predicted_class

python -m explanation_generation.gen_explanation_encoder \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models/roberta_civil_all/dropout" \
    --num_labels=2 \
    --batch_size=32 \
    --max_length=512 \
    --num_examples=200 \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/roberta_civil_all_race_val_200/dropout" \
    --baseline="pad" \
    --bias_type="race" \
    --seed=42 \
    --only_predicted_class

python -m sensitive_attribution_extraction.extract_sensitive_token_attribution \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/roberta_civil_race_race_val_200/dropout" \
    --split="val" \
    --bias_type="race" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m model_selection.compute_reliance_statistics \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/roberta_civil_race_race_val_200/dropout" \
    --split="val" \
    --bias_type="race" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m sensitive_attribution_extraction.extract_sensitive_token_attribution \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/roberta_civil_all_race_val_200/dropout" \
    --split="val" \
    --bias_type="race" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m model_selection.compute_reliance_statistics \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/roberta_civil_all_race_val_200/dropout" \
    --split="val" \
    --bias_type="race" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models/roberta_civil_race/attention_entropy" \
    --batch_size=32 \
    --max_seq_length=512 \
    --num_examples=200 \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/roberta_civil_race_race_val_200/attention_entropy" \
    --bias_type="race"

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models/roberta_civil_all/attention_entropy" \
    --batch_size=32 \
    --max_seq_length=512 \
    --num_examples=200 \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/roberta_civil_all_race_val_200/attention_entropy" \
    --bias_type="race"

python -m explanation_generation.gen_explanation_encoder \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models/roberta_civil_race/attention_entropy" \
    --num_labels=2 \
    --batch_size=32 \
    --max_length=512 \
    --num_examples=200 \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/roberta_civil_race_race_val_200/attention_entropy" \
    --baseline="pad" \
    --bias_type="race" \
    --seed=42 \
    --only_predicted_class

python -m explanation_generation.gen_explanation_encoder \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models/roberta_civil_all/attention_entropy" \
    --num_labels=2 \
    --batch_size=32 \
    --max_length=512 \
    --num_examples=200 \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/roberta_civil_all_race_val_200/attention_entropy" \
    --baseline="pad" \
    --bias_type="race" \
    --seed=42 \
    --only_predicted_class

python -m sensitive_attribution_extraction.extract_sensitive_token_attribution \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/roberta_civil_race_race_val_200/attention_entropy" \
    --split="val" \
    --bias_type="race" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m model_selection.compute_reliance_statistics \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/roberta_civil_race_race_val_200/attention_entropy" \
    --split="val" \
    --bias_type="race" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m sensitive_attribution_extraction.extract_sensitive_token_attribution \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/roberta_civil_all_race_val_200/attention_entropy" \
    --split="val" \
    --bias_type="race" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m model_selection.compute_reliance_statistics \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/roberta_civil_all_race_val_200/attention_entropy" \
    --split="val" \
    --bias_type="race" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models/roberta_civil_race/causal_debias" \
    --batch_size=32 \
    --max_seq_length=512 \
    --num_examples=200 \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/roberta_civil_race_race_val_200/causal_debias" \
    --bias_type="race"

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models/roberta_civil_all/causal_debias" \
    --batch_size=32 \
    --max_seq_length=512 \
    --num_examples=200 \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/roberta_civil_all_race_val_200/causal_debias" \
    --bias_type="race"

python -m explanation_generation.gen_explanation_encoder \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models/roberta_civil_race/causal_debias" \
    --num_labels=2 \
    --batch_size=32 \
    --max_length=512 \
    --num_examples=200 \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/roberta_civil_race_race_val_200/causal_debias" \
    --baseline="pad" \
    --bias_type="race" \
    --seed=42 \
    --only_predicted_class

python -m explanation_generation.gen_explanation_encoder \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models/roberta_civil_all/causal_debias" \
    --num_labels=2 \
    --batch_size=32 \
    --max_length=512 \
    --num_examples=200 \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/roberta_civil_all_race_val_200/causal_debias" \
    --baseline="pad" \
    --bias_type="race" \
    --seed=42 \
    --only_predicted_class

python -m sensitive_attribution_extraction.extract_sensitive_token_attribution \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/roberta_civil_race_race_val_200/causal_debias" \
    --split="val" \
    --bias_type="race" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m model_selection.compute_reliance_statistics \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/roberta_civil_race_race_val_200/causal_debias" \
    --split="val" \
    --bias_type="race" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m sensitive_attribution_extraction.extract_sensitive_token_attribution \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/roberta_civil_all_race_val_200/causal_debias" \
    --split="val" \
    --bias_type="race" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m model_selection.compute_reliance_statistics \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/roberta_civil_all_race_val_200/causal_debias" \
    --split="val" \
    --bias_type="race" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m model_selection.compute_model_selection_correlation \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results" \
    --val_results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --bias_type="race" \
    --model_type="roberta" \
    --num_test_examples=2000 \
    --num_val_examples=200 \
    --test_seed=-1 \
    --val_seed=-1

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models/distilbert_civil_race/no_debiasing" \
    --batch_size=32 \
    --max_seq_length=512 \
    --num_examples=200 \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/distilbert_civil_race_race_val_200/no_debiasing" \
    --bias_type="race"

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models/distilbert_civil_all/no_debiasing" \
    --batch_size=32 \
    --max_seq_length=512 \
    --num_examples=200 \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/distilbert_civil_all_race_val_200/no_debiasing" \
    --bias_type="race"

python -m explanation_generation.gen_explanation_encoder \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models/distilbert_civil_race/no_debiasing" \
    --num_labels=2 \
    --batch_size=32 \
    --max_length=512 \
    --num_examples=200 \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/distilbert_civil_race_race_val_200/no_debiasing" \
    --baseline="pad" \
    --bias_type="race" \
    --seed=42 \
    --only_predicted_class

python -m explanation_generation.gen_explanation_encoder \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models/distilbert_civil_all/no_debiasing" \
    --num_labels=2 \
    --batch_size=32 \
    --max_length=512 \
    --num_examples=200 \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/distilbert_civil_all_race_val_200/no_debiasing" \
    --baseline="pad" \
    --bias_type="race" \
    --seed=42 \
    --only_predicted_class

python -m sensitive_attribution_extraction.extract_sensitive_token_attribution \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/distilbert_civil_race_race_val_200/no_debiasing" \
    --split="val" \
    --bias_type="race" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m model_selection.compute_reliance_statistics \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/distilbert_civil_race_race_val_200/no_debiasing" \
    --split="val" \
    --bias_type="race" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m sensitive_attribution_extraction.extract_sensitive_token_attribution \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/distilbert_civil_all_race_val_200/no_debiasing" \
    --split="val" \
    --bias_type="race" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m model_selection.compute_reliance_statistics \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/distilbert_civil_all_race_val_200/no_debiasing" \
    --split="val" \
    --bias_type="race" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models/distilbert_civil_race/group_balance" \
    --batch_size=32 \
    --max_seq_length=512 \
    --num_examples=200 \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/distilbert_civil_race_race_val_200/group_balance" \
    --bias_type="race"

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models/distilbert_civil_all/group_balance" \
    --batch_size=32 \
    --max_seq_length=512 \
    --num_examples=200 \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/distilbert_civil_all_race_val_200/group_balance" \
    --bias_type="race"

python -m explanation_generation.gen_explanation_encoder \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models/distilbert_civil_race/group_balance" \
    --num_labels=2 \
    --batch_size=32 \
    --max_length=512 \
    --num_examples=200 \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/distilbert_civil_race_race_val_200/group_balance" \
    --baseline="pad" \
    --bias_type="race" \
    --seed=42 \
    --only_predicted_class

python -m explanation_generation.gen_explanation_encoder \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models/distilbert_civil_all/group_balance" \
    --num_labels=2 \
    --batch_size=32 \
    --max_length=512 \
    --num_examples=200 \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/distilbert_civil_all_race_val_200/group_balance" \
    --baseline="pad" \
    --bias_type="race" \
    --seed=42 \
    --only_predicted_class

python -m sensitive_attribution_extraction.extract_sensitive_token_attribution \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/distilbert_civil_race_race_val_200/group_balance" \
    --split="val" \
    --bias_type="race" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m model_selection.compute_reliance_statistics \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/distilbert_civil_race_race_val_200/group_balance" \
    --split="val" \
    --bias_type="race" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m sensitive_attribution_extraction.extract_sensitive_token_attribution \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/distilbert_civil_all_race_val_200/group_balance" \
    --split="val" \
    --bias_type="race" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m model_selection.compute_reliance_statistics \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/distilbert_civil_all_race_val_200/group_balance" \
    --split="val" \
    --bias_type="race" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models/distilbert_civil_race/group_class_balance" \
    --batch_size=32 \
    --max_seq_length=512 \
    --num_examples=200 \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/distilbert_civil_race_race_val_200/group_class_balance" \
    --bias_type="race"

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models/distilbert_civil_all/group_class_balance" \
    --batch_size=32 \
    --max_seq_length=512 \
    --num_examples=200 \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/distilbert_civil_all_race_val_200/group_class_balance" \
    --bias_type="race"

python -m explanation_generation.gen_explanation_encoder \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models/distilbert_civil_race/group_class_balance" \
    --num_labels=2 \
    --batch_size=32 \
    --max_length=512 \
    --num_examples=200 \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/distilbert_civil_race_race_val_200/group_class_balance" \
    --baseline="pad" \
    --bias_type="race" \
    --seed=42 \
    --only_predicted_class

python -m explanation_generation.gen_explanation_encoder \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models/distilbert_civil_all/group_class_balance" \
    --num_labels=2 \
    --batch_size=32 \
    --max_length=512 \
    --num_examples=200 \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/distilbert_civil_all_race_val_200/group_class_balance" \
    --baseline="pad" \
    --bias_type="race" \
    --seed=42 \
    --only_predicted_class

python -m sensitive_attribution_extraction.extract_sensitive_token_attribution \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/distilbert_civil_race_race_val_200/group_class_balance" \
    --split="val" \
    --bias_type="race" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m model_selection.compute_reliance_statistics \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/distilbert_civil_race_race_val_200/group_class_balance" \
    --split="val" \
    --bias_type="race" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m sensitive_attribution_extraction.extract_sensitive_token_attribution \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/distilbert_civil_all_race_val_200/group_class_balance" \
    --split="val" \
    --bias_type="race" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m model_selection.compute_reliance_statistics \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/distilbert_civil_all_race_val_200/group_class_balance" \
    --split="val" \
    --bias_type="race" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models/distilbert_civil_race/cda" \
    --batch_size=32 \
    --max_seq_length=512 \
    --num_examples=200 \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/distilbert_civil_race_race_val_200/cda" \
    --bias_type="race"

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models/distilbert_civil_all/cda" \
    --batch_size=32 \
    --max_seq_length=512 \
    --num_examples=200 \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/distilbert_civil_all_race_val_200/cda" \
    --bias_type="race"

python -m explanation_generation.gen_explanation_encoder \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models/distilbert_civil_race/cda" \
    --num_labels=2 \
    --batch_size=32 \
    --max_length=512 \
    --num_examples=200 \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/distilbert_civil_race_race_val_200/cda" \
    --baseline="pad" \
    --bias_type="race" \
    --seed=42 \
    --only_predicted_class

python -m explanation_generation.gen_explanation_encoder \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models/distilbert_civil_all/cda" \
    --num_labels=2 \
    --batch_size=32 \
    --max_length=512 \
    --num_examples=200 \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/distilbert_civil_all_race_val_200/cda" \
    --baseline="pad" \
    --bias_type="race" \
    --seed=42 \
    --only_predicted_class

python -m sensitive_attribution_extraction.extract_sensitive_token_attribution \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/distilbert_civil_race_race_val_200/cda" \
    --split="val" \
    --bias_type="race" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m model_selection.compute_reliance_statistics \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/distilbert_civil_race_race_val_200/cda" \
    --split="val" \
    --bias_type="race" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m sensitive_attribution_extraction.extract_sensitive_token_attribution \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/distilbert_civil_all_race_val_200/cda" \
    --split="val" \
    --bias_type="race" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m model_selection.compute_reliance_statistics \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/distilbert_civil_all_race_val_200/cda" \
    --split="val" \
    --bias_type="race" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models/distilbert_civil_race/dropout" \
    --batch_size=32 \
    --max_seq_length=512 \
    --num_examples=200 \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/distilbert_civil_race_race_val_200/dropout" \
    --bias_type="race"

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models/distilbert_civil_all/dropout" \
    --batch_size=32 \
    --max_seq_length=512 \
    --num_examples=200 \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/distilbert_civil_all_race_val_200/dropout" \
    --bias_type="race"

python -m explanation_generation.gen_explanation_encoder \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models/distilbert_civil_race/dropout" \
    --num_labels=2 \
    --batch_size=32 \
    --max_length=512 \
    --num_examples=200 \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/distilbert_civil_race_race_val_200/dropout" \
    --baseline="pad" \
    --bias_type="race" \
    --seed=42 \
    --only_predicted_class

python -m explanation_generation.gen_explanation_encoder \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models/distilbert_civil_all/dropout" \
    --num_labels=2 \
    --batch_size=32 \
    --max_length=512 \
    --num_examples=200 \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/distilbert_civil_all_race_val_200/dropout" \
    --baseline="pad" \
    --bias_type="race" \
    --seed=42 \
    --only_predicted_class

python -m sensitive_attribution_extraction.extract_sensitive_token_attribution \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/distilbert_civil_race_race_val_200/dropout" \
    --split="val" \
    --bias_type="race" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m model_selection.compute_reliance_statistics \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/distilbert_civil_race_race_val_200/dropout" \
    --split="val" \
    --bias_type="race" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m sensitive_attribution_extraction.extract_sensitive_token_attribution \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/distilbert_civil_all_race_val_200/dropout" \
    --split="val" \
    --bias_type="race" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m model_selection.compute_reliance_statistics \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/distilbert_civil_all_race_val_200/dropout" \
    --split="val" \
    --bias_type="race" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models/distilbert_civil_race/attention_entropy" \
    --batch_size=32 \
    --max_seq_length=512 \
    --num_examples=200 \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/distilbert_civil_race_race_val_200/attention_entropy" \
    --bias_type="race"

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models/distilbert_civil_all/attention_entropy" \
    --batch_size=32 \
    --max_seq_length=512 \
    --num_examples=200 \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/distilbert_civil_all_race_val_200/attention_entropy" \
    --bias_type="race"

python -m explanation_generation.gen_explanation_encoder \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models/distilbert_civil_race/attention_entropy" \
    --num_labels=2 \
    --batch_size=32 \
    --max_length=512 \
    --num_examples=200 \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/distilbert_civil_race_race_val_200/attention_entropy" \
    --baseline="pad" \
    --bias_type="race" \
    --seed=42 \
    --only_predicted_class

python -m explanation_generation.gen_explanation_encoder \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models/distilbert_civil_all/attention_entropy" \
    --num_labels=2 \
    --batch_size=32 \
    --max_length=512 \
    --num_examples=200 \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/distilbert_civil_all_race_val_200/attention_entropy" \
    --baseline="pad" \
    --bias_type="race" \
    --seed=42 \
    --only_predicted_class

python -m sensitive_attribution_extraction.extract_sensitive_token_attribution \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/distilbert_civil_race_race_val_200/attention_entropy" \
    --split="val" \
    --bias_type="race" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m model_selection.compute_reliance_statistics \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/distilbert_civil_race_race_val_200/attention_entropy" \
    --split="val" \
    --bias_type="race" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m sensitive_attribution_extraction.extract_sensitive_token_attribution \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/distilbert_civil_all_race_val_200/attention_entropy" \
    --split="val" \
    --bias_type="race" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m model_selection.compute_reliance_statistics \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/distilbert_civil_all_race_val_200/attention_entropy" \
    --split="val" \
    --bias_type="race" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models/distilbert_civil_race/causal_debias" \
    --batch_size=32 \
    --max_seq_length=512 \
    --num_examples=200 \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/distilbert_civil_race_race_val_200/causal_debias" \
    --bias_type="race"

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models/distilbert_civil_all/causal_debias" \
    --batch_size=32 \
    --max_seq_length=512 \
    --num_examples=200 \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/distilbert_civil_all_race_val_200/causal_debias" \
    --bias_type="race"

python -m explanation_generation.gen_explanation_encoder \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models/distilbert_civil_race/causal_debias" \
    --num_labels=2 \
    --batch_size=32 \
    --max_length=512 \
    --num_examples=200 \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/distilbert_civil_race_race_val_200/causal_debias" \
    --baseline="pad" \
    --bias_type="race" \
    --seed=42 \
    --only_predicted_class

python -m explanation_generation.gen_explanation_encoder \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models/distilbert_civil_all/causal_debias" \
    --num_labels=2 \
    --batch_size=32 \
    --max_length=512 \
    --num_examples=200 \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/distilbert_civil_all_race_val_200/causal_debias" \
    --baseline="pad" \
    --bias_type="race" \
    --seed=42 \
    --only_predicted_class

python -m sensitive_attribution_extraction.extract_sensitive_token_attribution \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/distilbert_civil_race_race_val_200/causal_debias" \
    --split="val" \
    --bias_type="race" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m model_selection.compute_reliance_statistics \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/distilbert_civil_race_race_val_200/causal_debias" \
    --split="val" \
    --bias_type="race" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m sensitive_attribution_extraction.extract_sensitive_token_attribution \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/distilbert_civil_all_race_val_200/causal_debias" \
    --split="val" \
    --bias_type="race" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m model_selection.compute_reliance_statistics \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/distilbert_civil_all_race_val_200/causal_debias" \
    --split="val" \
    --bias_type="race" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m model_selection.compute_model_selection_correlation \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results" \
    --val_results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --bias_type="race" \
    --model_type="distilbert" \
    --num_test_examples=2000 \
    --num_val_examples=200 \
    --test_seed=-1 \
    --val_seed=-1

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models/bert_civil_race/no_debiasing" \
    --batch_size=32 \
    --max_seq_length=512 \
    --num_examples=500 \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/bert_civil_race_race_val_500/no_debiasing" \
    --bias_type="race"

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models/bert_civil_all/no_debiasing" \
    --batch_size=32 \
    --max_seq_length=512 \
    --num_examples=500 \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/bert_civil_all_race_val_500/no_debiasing" \
    --bias_type="race"

python -m explanation_generation.gen_explanation_encoder \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models/bert_civil_race/no_debiasing" \
    --num_labels=2 \
    --batch_size=32 \
    --max_length=512 \
    --num_examples=500 \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/bert_civil_race_race_val_500/no_debiasing" \
    --baseline="pad" \
    --bias_type="race" \
    --seed=42 \
    --only_predicted_class

python -m explanation_generation.gen_explanation_encoder \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models/bert_civil_all/no_debiasing" \
    --num_labels=2 \
    --batch_size=32 \
    --max_length=512 \
    --num_examples=500 \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/bert_civil_all_race_val_500/no_debiasing" \
    --baseline="pad" \
    --bias_type="race" \
    --seed=42 \
    --only_predicted_class

python -m sensitive_attribution_extraction.extract_sensitive_token_attribution \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/bert_civil_race_race_val_500/no_debiasing" \
    --split="val" \
    --bias_type="race" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m model_selection.compute_reliance_statistics \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/bert_civil_race_race_val_500/no_debiasing" \
    --split="val" \
    --bias_type="race" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m sensitive_attribution_extraction.extract_sensitive_token_attribution \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/bert_civil_all_race_val_500/no_debiasing" \
    --split="val" \
    --bias_type="race" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m model_selection.compute_reliance_statistics \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/bert_civil_all_race_val_500/no_debiasing" \
    --split="val" \
    --bias_type="race" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models/bert_civil_race/group_balance" \
    --batch_size=32 \
    --max_seq_length=512 \
    --num_examples=500 \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/bert_civil_race_race_val_500/group_balance" \
    --bias_type="race"

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models/bert_civil_all/group_balance" \
    --batch_size=32 \
    --max_seq_length=512 \
    --num_examples=500 \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/bert_civil_all_race_val_500/group_balance" \
    --bias_type="race"

python -m explanation_generation.gen_explanation_encoder \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models/bert_civil_race/group_balance" \
    --num_labels=2 \
    --batch_size=32 \
    --max_length=512 \
    --num_examples=500 \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/bert_civil_race_race_val_500/group_balance" \
    --baseline="pad" \
    --bias_type="race" \
    --seed=42 \
    --only_predicted_class

python -m explanation_generation.gen_explanation_encoder \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models/bert_civil_all/group_balance" \
    --num_labels=2 \
    --batch_size=32 \
    --max_length=512 \
    --num_examples=500 \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/bert_civil_all_race_val_500/group_balance" \
    --baseline="pad" \
    --bias_type="race" \
    --seed=42 \
    --only_predicted_class

python -m sensitive_attribution_extraction.extract_sensitive_token_attribution \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/bert_civil_race_race_val_500/group_balance" \
    --split="val" \
    --bias_type="race" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m model_selection.compute_reliance_statistics \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/bert_civil_race_race_val_500/group_balance" \
    --split="val" \
    --bias_type="race" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m sensitive_attribution_extraction.extract_sensitive_token_attribution \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/bert_civil_all_race_val_500/group_balance" \
    --split="val" \
    --bias_type="race" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m model_selection.compute_reliance_statistics \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/bert_civil_all_race_val_500/group_balance" \
    --split="val" \
    --bias_type="race" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models/bert_civil_race/group_class_balance" \
    --batch_size=32 \
    --max_seq_length=512 \
    --num_examples=500 \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/bert_civil_race_race_val_500/group_class_balance" \
    --bias_type="race"

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models/bert_civil_all/group_class_balance" \
    --batch_size=32 \
    --max_seq_length=512 \
    --num_examples=500 \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/bert_civil_all_race_val_500/group_class_balance" \
    --bias_type="race"

python -m explanation_generation.gen_explanation_encoder \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models/bert_civil_race/group_class_balance" \
    --num_labels=2 \
    --batch_size=32 \
    --max_length=512 \
    --num_examples=500 \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/bert_civil_race_race_val_500/group_class_balance" \
    --baseline="pad" \
    --bias_type="race" \
    --seed=42 \
    --only_predicted_class

python -m explanation_generation.gen_explanation_encoder \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models/bert_civil_all/group_class_balance" \
    --num_labels=2 \
    --batch_size=32 \
    --max_length=512 \
    --num_examples=500 \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/bert_civil_all_race_val_500/group_class_balance" \
    --baseline="pad" \
    --bias_type="race" \
    --seed=42 \
    --only_predicted_class

python -m sensitive_attribution_extraction.extract_sensitive_token_attribution \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/bert_civil_race_race_val_500/group_class_balance" \
    --split="val" \
    --bias_type="race" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m model_selection.compute_reliance_statistics \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/bert_civil_race_race_val_500/group_class_balance" \
    --split="val" \
    --bias_type="race" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m sensitive_attribution_extraction.extract_sensitive_token_attribution \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/bert_civil_all_race_val_500/group_class_balance" \
    --split="val" \
    --bias_type="race" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m model_selection.compute_reliance_statistics \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/bert_civil_all_race_val_500/group_class_balance" \
    --split="val" \
    --bias_type="race" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models/bert_civil_race/cda" \
    --batch_size=32 \
    --max_seq_length=512 \
    --num_examples=500 \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/bert_civil_race_race_val_500/cda" \
    --bias_type="race"

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models/bert_civil_all/cda" \
    --batch_size=32 \
    --max_seq_length=512 \
    --num_examples=500 \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/bert_civil_all_race_val_500/cda" \
    --bias_type="race"

python -m explanation_generation.gen_explanation_encoder \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models/bert_civil_race/cda" \
    --num_labels=2 \
    --batch_size=32 \
    --max_length=512 \
    --num_examples=500 \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/bert_civil_race_race_val_500/cda" \
    --baseline="pad" \
    --bias_type="race" \
    --seed=42 \
    --only_predicted_class

python -m explanation_generation.gen_explanation_encoder \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models/bert_civil_all/cda" \
    --num_labels=2 \
    --batch_size=32 \
    --max_length=512 \
    --num_examples=500 \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/bert_civil_all_race_val_500/cda" \
    --baseline="pad" \
    --bias_type="race" \
    --seed=42 \
    --only_predicted_class

python -m sensitive_attribution_extraction.extract_sensitive_token_attribution \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/bert_civil_race_race_val_500/cda" \
    --split="val" \
    --bias_type="race" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m model_selection.compute_reliance_statistics \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/bert_civil_race_race_val_500/cda" \
    --split="val" \
    --bias_type="race" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m sensitive_attribution_extraction.extract_sensitive_token_attribution \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/bert_civil_all_race_val_500/cda" \
    --split="val" \
    --bias_type="race" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m model_selection.compute_reliance_statistics \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/bert_civil_all_race_val_500/cda" \
    --split="val" \
    --bias_type="race" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models/bert_civil_race/dropout" \
    --batch_size=32 \
    --max_seq_length=512 \
    --num_examples=500 \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/bert_civil_race_race_val_500/dropout" \
    --bias_type="race"

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models/bert_civil_all/dropout" \
    --batch_size=32 \
    --max_seq_length=512 \
    --num_examples=500 \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/bert_civil_all_race_val_500/dropout" \
    --bias_type="race"

python -m explanation_generation.gen_explanation_encoder \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models/bert_civil_race/dropout" \
    --num_labels=2 \
    --batch_size=32 \
    --max_length=512 \
    --num_examples=500 \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/bert_civil_race_race_val_500/dropout" \
    --baseline="pad" \
    --bias_type="race" \
    --seed=42 \
    --only_predicted_class

python -m explanation_generation.gen_explanation_encoder \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models/bert_civil_all/dropout" \
    --num_labels=2 \
    --batch_size=32 \
    --max_length=512 \
    --num_examples=500 \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/bert_civil_all_race_val_500/dropout" \
    --baseline="pad" \
    --bias_type="race" \
    --seed=42 \
    --only_predicted_class

python -m sensitive_attribution_extraction.extract_sensitive_token_attribution \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/bert_civil_race_race_val_500/dropout" \
    --split="val" \
    --bias_type="race" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m model_selection.compute_reliance_statistics \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/bert_civil_race_race_val_500/dropout" \
    --split="val" \
    --bias_type="race" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m sensitive_attribution_extraction.extract_sensitive_token_attribution \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/bert_civil_all_race_val_500/dropout" \
    --split="val" \
    --bias_type="race" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m model_selection.compute_reliance_statistics \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/bert_civil_all_race_val_500/dropout" \
    --split="val" \
    --bias_type="race" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models/bert_civil_race/attention_entropy" \
    --batch_size=32 \
    --max_seq_length=512 \
    --num_examples=500 \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/bert_civil_race_race_val_500/attention_entropy" \
    --bias_type="race"

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models/bert_civil_all/attention_entropy" \
    --batch_size=32 \
    --max_seq_length=512 \
    --num_examples=500 \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/bert_civil_all_race_val_500/attention_entropy" \
    --bias_type="race"

python -m explanation_generation.gen_explanation_encoder \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models/bert_civil_race/attention_entropy" \
    --num_labels=2 \
    --batch_size=32 \
    --max_length=512 \
    --num_examples=500 \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/bert_civil_race_race_val_500/attention_entropy" \
    --baseline="pad" \
    --bias_type="race" \
    --seed=42 \
    --only_predicted_class

python -m explanation_generation.gen_explanation_encoder \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models/bert_civil_all/attention_entropy" \
    --num_labels=2 \
    --batch_size=32 \
    --max_length=512 \
    --num_examples=500 \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/bert_civil_all_race_val_500/attention_entropy" \
    --baseline="pad" \
    --bias_type="race" \
    --seed=42 \
    --only_predicted_class

python -m sensitive_attribution_extraction.extract_sensitive_token_attribution \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/bert_civil_race_race_val_500/attention_entropy" \
    --split="val" \
    --bias_type="race" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m model_selection.compute_reliance_statistics \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/bert_civil_race_race_val_500/attention_entropy" \
    --split="val" \
    --bias_type="race" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m sensitive_attribution_extraction.extract_sensitive_token_attribution \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/bert_civil_all_race_val_500/attention_entropy" \
    --split="val" \
    --bias_type="race" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m model_selection.compute_reliance_statistics \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/bert_civil_all_race_val_500/attention_entropy" \
    --split="val" \
    --bias_type="race" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models/bert_civil_race/causal_debias" \
    --batch_size=32 \
    --max_seq_length=512 \
    --num_examples=500 \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/bert_civil_race_race_val_500/causal_debias" \
    --bias_type="race"

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models/bert_civil_all/causal_debias" \
    --batch_size=32 \
    --max_seq_length=512 \
    --num_examples=500 \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/bert_civil_all_race_val_500/causal_debias" \
    --bias_type="race"

python -m explanation_generation.gen_explanation_encoder \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models/bert_civil_race/causal_debias" \
    --num_labels=2 \
    --batch_size=32 \
    --max_length=512 \
    --num_examples=500 \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/bert_civil_race_race_val_500/causal_debias" \
    --baseline="pad" \
    --bias_type="race" \
    --seed=42 \
    --only_predicted_class

python -m explanation_generation.gen_explanation_encoder \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models/bert_civil_all/causal_debias" \
    --num_labels=2 \
    --batch_size=32 \
    --max_length=512 \
    --num_examples=500 \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/bert_civil_all_race_val_500/causal_debias" \
    --baseline="pad" \
    --bias_type="race" \
    --seed=42 \
    --only_predicted_class

python -m sensitive_attribution_extraction.extract_sensitive_token_attribution \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/bert_civil_race_race_val_500/causal_debias" \
    --split="val" \
    --bias_type="race" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m model_selection.compute_reliance_statistics \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/bert_civil_race_race_val_500/causal_debias" \
    --split="val" \
    --bias_type="race" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m sensitive_attribution_extraction.extract_sensitive_token_attribution \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/bert_civil_all_race_val_500/causal_debias" \
    --split="val" \
    --bias_type="race" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m model_selection.compute_reliance_statistics \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/bert_civil_all_race_val_500/causal_debias" \
    --split="val" \
    --bias_type="race" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m model_selection.compute_model_selection_correlation \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results" \
    --val_results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --bias_type="race" \
    --model_type="bert" \
    --num_test_examples=2000 \
    --num_val_examples=500 \
    --test_seed=-1 \
    --val_seed=-1

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models/roberta_civil_race/no_debiasing" \
    --batch_size=32 \
    --max_seq_length=512 \
    --num_examples=500 \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/roberta_civil_race_race_val_500/no_debiasing" \
    --bias_type="race"

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models/roberta_civil_all/no_debiasing" \
    --batch_size=32 \
    --max_seq_length=512 \
    --num_examples=500 \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/roberta_civil_all_race_val_500/no_debiasing" \
    --bias_type="race"

python -m explanation_generation.gen_explanation_encoder \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models/roberta_civil_race/no_debiasing" \
    --num_labels=2 \
    --batch_size=32 \
    --max_length=512 \
    --num_examples=500 \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/roberta_civil_race_race_val_500/no_debiasing" \
    --baseline="pad" \
    --bias_type="race" \
    --seed=42 \
    --only_predicted_class

python -m explanation_generation.gen_explanation_encoder \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models/roberta_civil_all/no_debiasing" \
    --num_labels=2 \
    --batch_size=32 \
    --max_length=512 \
    --num_examples=500 \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/roberta_civil_all_race_val_500/no_debiasing" \
    --baseline="pad" \
    --bias_type="race" \
    --seed=42 \
    --only_predicted_class

python -m sensitive_attribution_extraction.extract_sensitive_token_attribution \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/roberta_civil_race_race_val_500/no_debiasing" \
    --split="val" \
    --bias_type="race" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m model_selection.compute_reliance_statistics \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/roberta_civil_race_race_val_500/no_debiasing" \
    --split="val" \
    --bias_type="race" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m sensitive_attribution_extraction.extract_sensitive_token_attribution \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/roberta_civil_all_race_val_500/no_debiasing" \
    --split="val" \
    --bias_type="race" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m model_selection.compute_reliance_statistics \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/roberta_civil_all_race_val_500/no_debiasing" \
    --split="val" \
    --bias_type="race" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models/roberta_civil_race/group_balance" \
    --batch_size=32 \
    --max_seq_length=512 \
    --num_examples=500 \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/roberta_civil_race_race_val_500/group_balance" \
    --bias_type="race"

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models/roberta_civil_all/group_balance" \
    --batch_size=32 \
    --max_seq_length=512 \
    --num_examples=500 \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/roberta_civil_all_race_val_500/group_balance" \
    --bias_type="race"

python -m explanation_generation.gen_explanation_encoder \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models/roberta_civil_race/group_balance" \
    --num_labels=2 \
    --batch_size=32 \
    --max_length=512 \
    --num_examples=500 \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/roberta_civil_race_race_val_500/group_balance" \
    --baseline="pad" \
    --bias_type="race" \
    --seed=42 \
    --only_predicted_class

python -m explanation_generation.gen_explanation_encoder \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models/roberta_civil_all/group_balance" \
    --num_labels=2 \
    --batch_size=32 \
    --max_length=512 \
    --num_examples=500 \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/roberta_civil_all_race_val_500/group_balance" \
    --baseline="pad" \
    --bias_type="race" \
    --seed=42 \
    --only_predicted_class

python -m sensitive_attribution_extraction.extract_sensitive_token_attribution \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/roberta_civil_race_race_val_500/group_balance" \
    --split="val" \
    --bias_type="race" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m model_selection.compute_reliance_statistics \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/roberta_civil_race_race_val_500/group_balance" \
    --split="val" \
    --bias_type="race" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m sensitive_attribution_extraction.extract_sensitive_token_attribution \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/roberta_civil_all_race_val_500/group_balance" \
    --split="val" \
    --bias_type="race" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m model_selection.compute_reliance_statistics \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/roberta_civil_all_race_val_500/group_balance" \
    --split="val" \
    --bias_type="race" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models/roberta_civil_race/group_class_balance" \
    --batch_size=32 \
    --max_seq_length=512 \
    --num_examples=500 \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/roberta_civil_race_race_val_500/group_class_balance" \
    --bias_type="race"

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models/roberta_civil_all/group_class_balance" \
    --batch_size=32 \
    --max_seq_length=512 \
    --num_examples=500 \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/roberta_civil_all_race_val_500/group_class_balance" \
    --bias_type="race"

python -m explanation_generation.gen_explanation_encoder \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models/roberta_civil_race/group_class_balance" \
    --num_labels=2 \
    --batch_size=32 \
    --max_length=512 \
    --num_examples=500 \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/roberta_civil_race_race_val_500/group_class_balance" \
    --baseline="pad" \
    --bias_type="race" \
    --seed=42 \
    --only_predicted_class

python -m explanation_generation.gen_explanation_encoder \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models/roberta_civil_all/group_class_balance" \
    --num_labels=2 \
    --batch_size=32 \
    --max_length=512 \
    --num_examples=500 \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/roberta_civil_all_race_val_500/group_class_balance" \
    --baseline="pad" \
    --bias_type="race" \
    --seed=42 \
    --only_predicted_class

python -m sensitive_attribution_extraction.extract_sensitive_token_attribution \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/roberta_civil_race_race_val_500/group_class_balance" \
    --split="val" \
    --bias_type="race" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m model_selection.compute_reliance_statistics \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/roberta_civil_race_race_val_500/group_class_balance" \
    --split="val" \
    --bias_type="race" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m sensitive_attribution_extraction.extract_sensitive_token_attribution \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/roberta_civil_all_race_val_500/group_class_balance" \
    --split="val" \
    --bias_type="race" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m model_selection.compute_reliance_statistics \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/roberta_civil_all_race_val_500/group_class_balance" \
    --split="val" \
    --bias_type="race" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models/roberta_civil_race/cda" \
    --batch_size=32 \
    --max_seq_length=512 \
    --num_examples=500 \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/roberta_civil_race_race_val_500/cda" \
    --bias_type="race"

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models/roberta_civil_all/cda" \
    --batch_size=32 \
    --max_seq_length=512 \
    --num_examples=500 \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/roberta_civil_all_race_val_500/cda" \
    --bias_type="race"

python -m explanation_generation.gen_explanation_encoder \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models/roberta_civil_race/cda" \
    --num_labels=2 \
    --batch_size=32 \
    --max_length=512 \
    --num_examples=500 \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/roberta_civil_race_race_val_500/cda" \
    --baseline="pad" \
    --bias_type="race" \
    --seed=42 \
    --only_predicted_class

python -m explanation_generation.gen_explanation_encoder \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models/roberta_civil_all/cda" \
    --num_labels=2 \
    --batch_size=32 \
    --max_length=512 \
    --num_examples=500 \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/roberta_civil_all_race_val_500/cda" \
    --baseline="pad" \
    --bias_type="race" \
    --seed=42 \
    --only_predicted_class

python -m sensitive_attribution_extraction.extract_sensitive_token_attribution \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/roberta_civil_race_race_val_500/cda" \
    --split="val" \
    --bias_type="race" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m model_selection.compute_reliance_statistics \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/roberta_civil_race_race_val_500/cda" \
    --split="val" \
    --bias_type="race" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m sensitive_attribution_extraction.extract_sensitive_token_attribution \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/roberta_civil_all_race_val_500/cda" \
    --split="val" \
    --bias_type="race" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m model_selection.compute_reliance_statistics \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/roberta_civil_all_race_val_500/cda" \
    --split="val" \
    --bias_type="race" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models/roberta_civil_race/dropout" \
    --batch_size=32 \
    --max_seq_length=512 \
    --num_examples=500 \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/roberta_civil_race_race_val_500/dropout" \
    --bias_type="race"

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models/roberta_civil_all/dropout" \
    --batch_size=32 \
    --max_seq_length=512 \
    --num_examples=500 \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/roberta_civil_all_race_val_500/dropout" \
    --bias_type="race"

python -m explanation_generation.gen_explanation_encoder \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models/roberta_civil_race/dropout" \
    --num_labels=2 \
    --batch_size=32 \
    --max_length=512 \
    --num_examples=500 \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/roberta_civil_race_race_val_500/dropout" \
    --baseline="pad" \
    --bias_type="race" \
    --seed=42 \
    --only_predicted_class

python -m explanation_generation.gen_explanation_encoder \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models/roberta_civil_all/dropout" \
    --num_labels=2 \
    --batch_size=32 \
    --max_length=512 \
    --num_examples=500 \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/roberta_civil_all_race_val_500/dropout" \
    --baseline="pad" \
    --bias_type="race" \
    --seed=42 \
    --only_predicted_class

python -m sensitive_attribution_extraction.extract_sensitive_token_attribution \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/roberta_civil_race_race_val_500/dropout" \
    --split="val" \
    --bias_type="race" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m model_selection.compute_reliance_statistics \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/roberta_civil_race_race_val_500/dropout" \
    --split="val" \
    --bias_type="race" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m sensitive_attribution_extraction.extract_sensitive_token_attribution \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/roberta_civil_all_race_val_500/dropout" \
    --split="val" \
    --bias_type="race" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m model_selection.compute_reliance_statistics \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/roberta_civil_all_race_val_500/dropout" \
    --split="val" \
    --bias_type="race" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models/roberta_civil_race/attention_entropy" \
    --batch_size=32 \
    --max_seq_length=512 \
    --num_examples=500 \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/roberta_civil_race_race_val_500/attention_entropy" \
    --bias_type="race"

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models/roberta_civil_all/attention_entropy" \
    --batch_size=32 \
    --max_seq_length=512 \
    --num_examples=500 \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/roberta_civil_all_race_val_500/attention_entropy" \
    --bias_type="race"

python -m explanation_generation.gen_explanation_encoder \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models/roberta_civil_race/attention_entropy" \
    --num_labels=2 \
    --batch_size=32 \
    --max_length=512 \
    --num_examples=500 \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/roberta_civil_race_race_val_500/attention_entropy" \
    --baseline="pad" \
    --bias_type="race" \
    --seed=42 \
    --only_predicted_class

python -m explanation_generation.gen_explanation_encoder \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models/roberta_civil_all/attention_entropy" \
    --num_labels=2 \
    --batch_size=32 \
    --max_length=512 \
    --num_examples=500 \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/roberta_civil_all_race_val_500/attention_entropy" \
    --baseline="pad" \
    --bias_type="race" \
    --seed=42 \
    --only_predicted_class

python -m sensitive_attribution_extraction.extract_sensitive_token_attribution \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/roberta_civil_race_race_val_500/attention_entropy" \
    --split="val" \
    --bias_type="race" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m model_selection.compute_reliance_statistics \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/roberta_civil_race_race_val_500/attention_entropy" \
    --split="val" \
    --bias_type="race" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m sensitive_attribution_extraction.extract_sensitive_token_attribution \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/roberta_civil_all_race_val_500/attention_entropy" \
    --split="val" \
    --bias_type="race" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m model_selection.compute_reliance_statistics \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/roberta_civil_all_race_val_500/attention_entropy" \
    --split="val" \
    --bias_type="race" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models/roberta_civil_race/causal_debias" \
    --batch_size=32 \
    --max_seq_length=512 \
    --num_examples=500 \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/roberta_civil_race_race_val_500/causal_debias" \
    --bias_type="race"

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models/roberta_civil_all/causal_debias" \
    --batch_size=32 \
    --max_seq_length=512 \
    --num_examples=500 \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/roberta_civil_all_race_val_500/causal_debias" \
    --bias_type="race"

python -m explanation_generation.gen_explanation_encoder \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models/roberta_civil_race/causal_debias" \
    --num_labels=2 \
    --batch_size=32 \
    --max_length=512 \
    --num_examples=500 \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/roberta_civil_race_race_val_500/causal_debias" \
    --baseline="pad" \
    --bias_type="race" \
    --seed=42 \
    --only_predicted_class

python -m explanation_generation.gen_explanation_encoder \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models/roberta_civil_all/causal_debias" \
    --num_labels=2 \
    --batch_size=32 \
    --max_length=512 \
    --num_examples=500 \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/roberta_civil_all_race_val_500/causal_debias" \
    --baseline="pad" \
    --bias_type="race" \
    --seed=42 \
    --only_predicted_class

python -m sensitive_attribution_extraction.extract_sensitive_token_attribution \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/roberta_civil_race_race_val_500/causal_debias" \
    --split="val" \
    --bias_type="race" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m model_selection.compute_reliance_statistics \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/roberta_civil_race_race_val_500/causal_debias" \
    --split="val" \
    --bias_type="race" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m sensitive_attribution_extraction.extract_sensitive_token_attribution \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/roberta_civil_all_race_val_500/causal_debias" \
    --split="val" \
    --bias_type="race" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m model_selection.compute_reliance_statistics \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/roberta_civil_all_race_val_500/causal_debias" \
    --split="val" \
    --bias_type="race" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m model_selection.compute_model_selection_correlation \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results" \
    --val_results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --bias_type="race" \
    --model_type="roberta" \
    --num_test_examples=2000 \
    --num_val_examples=500 \
    --test_seed=-1 \
    --val_seed=-1

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models/distilbert_civil_race/no_debiasing" \
    --batch_size=32 \
    --max_seq_length=512 \
    --num_examples=500 \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/distilbert_civil_race_race_val_500/no_debiasing" \
    --bias_type="race"

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models/distilbert_civil_all/no_debiasing" \
    --batch_size=32 \
    --max_seq_length=512 \
    --num_examples=500 \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/distilbert_civil_all_race_val_500/no_debiasing" \
    --bias_type="race"

python -m explanation_generation.gen_explanation_encoder \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models/distilbert_civil_race/no_debiasing" \
    --num_labels=2 \
    --batch_size=32 \
    --max_length=512 \
    --num_examples=500 \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/distilbert_civil_race_race_val_500/no_debiasing" \
    --baseline="pad" \
    --bias_type="race" \
    --seed=42 \
    --only_predicted_class

python -m explanation_generation.gen_explanation_encoder \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models/distilbert_civil_all/no_debiasing" \
    --num_labels=2 \
    --batch_size=32 \
    --max_length=512 \
    --num_examples=500 \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/distilbert_civil_all_race_val_500/no_debiasing" \
    --baseline="pad" \
    --bias_type="race" \
    --seed=42 \
    --only_predicted_class

python -m sensitive_attribution_extraction.extract_sensitive_token_attribution \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/distilbert_civil_race_race_val_500/no_debiasing" \
    --split="val" \
    --bias_type="race" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m model_selection.compute_reliance_statistics \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/distilbert_civil_race_race_val_500/no_debiasing" \
    --split="val" \
    --bias_type="race" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m sensitive_attribution_extraction.extract_sensitive_token_attribution \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/distilbert_civil_all_race_val_500/no_debiasing" \
    --split="val" \
    --bias_type="race" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m model_selection.compute_reliance_statistics \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/distilbert_civil_all_race_val_500/no_debiasing" \
    --split="val" \
    --bias_type="race" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models/distilbert_civil_race/group_balance" \
    --batch_size=32 \
    --max_seq_length=512 \
    --num_examples=500 \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/distilbert_civil_race_race_val_500/group_balance" \
    --bias_type="race"

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models/distilbert_civil_all/group_balance" \
    --batch_size=32 \
    --max_seq_length=512 \
    --num_examples=500 \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/distilbert_civil_all_race_val_500/group_balance" \
    --bias_type="race"

python -m explanation_generation.gen_explanation_encoder \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models/distilbert_civil_race/group_balance" \
    --num_labels=2 \
    --batch_size=32 \
    --max_length=512 \
    --num_examples=500 \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/distilbert_civil_race_race_val_500/group_balance" \
    --baseline="pad" \
    --bias_type="race" \
    --seed=42 \
    --only_predicted_class

python -m explanation_generation.gen_explanation_encoder \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models/distilbert_civil_all/group_balance" \
    --num_labels=2 \
    --batch_size=32 \
    --max_length=512 \
    --num_examples=500 \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/distilbert_civil_all_race_val_500/group_balance" \
    --baseline="pad" \
    --bias_type="race" \
    --seed=42 \
    --only_predicted_class

python -m sensitive_attribution_extraction.extract_sensitive_token_attribution \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/distilbert_civil_race_race_val_500/group_balance" \
    --split="val" \
    --bias_type="race" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m model_selection.compute_reliance_statistics \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/distilbert_civil_race_race_val_500/group_balance" \
    --split="val" \
    --bias_type="race" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m sensitive_attribution_extraction.extract_sensitive_token_attribution \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/distilbert_civil_all_race_val_500/group_balance" \
    --split="val" \
    --bias_type="race" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m model_selection.compute_reliance_statistics \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/distilbert_civil_all_race_val_500/group_balance" \
    --split="val" \
    --bias_type="race" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models/distilbert_civil_race/group_class_balance" \
    --batch_size=32 \
    --max_seq_length=512 \
    --num_examples=500 \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/distilbert_civil_race_race_val_500/group_class_balance" \
    --bias_type="race"

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models/distilbert_civil_all/group_class_balance" \
    --batch_size=32 \
    --max_seq_length=512 \
    --num_examples=500 \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/distilbert_civil_all_race_val_500/group_class_balance" \
    --bias_type="race"

python -m explanation_generation.gen_explanation_encoder \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models/distilbert_civil_race/group_class_balance" \
    --num_labels=2 \
    --batch_size=32 \
    --max_length=512 \
    --num_examples=500 \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/distilbert_civil_race_race_val_500/group_class_balance" \
    --baseline="pad" \
    --bias_type="race" \
    --seed=42 \
    --only_predicted_class

python -m explanation_generation.gen_explanation_encoder \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models/distilbert_civil_all/group_class_balance" \
    --num_labels=2 \
    --batch_size=32 \
    --max_length=512 \
    --num_examples=500 \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/distilbert_civil_all_race_val_500/group_class_balance" \
    --baseline="pad" \
    --bias_type="race" \
    --seed=42 \
    --only_predicted_class

python -m sensitive_attribution_extraction.extract_sensitive_token_attribution \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/distilbert_civil_race_race_val_500/group_class_balance" \
    --split="val" \
    --bias_type="race" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m model_selection.compute_reliance_statistics \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/distilbert_civil_race_race_val_500/group_class_balance" \
    --split="val" \
    --bias_type="race" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m sensitive_attribution_extraction.extract_sensitive_token_attribution \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/distilbert_civil_all_race_val_500/group_class_balance" \
    --split="val" \
    --bias_type="race" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m model_selection.compute_reliance_statistics \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/distilbert_civil_all_race_val_500/group_class_balance" \
    --split="val" \
    --bias_type="race" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models/distilbert_civil_race/cda" \
    --batch_size=32 \
    --max_seq_length=512 \
    --num_examples=500 \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/distilbert_civil_race_race_val_500/cda" \
    --bias_type="race"

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models/distilbert_civil_all/cda" \
    --batch_size=32 \
    --max_seq_length=512 \
    --num_examples=500 \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/distilbert_civil_all_race_val_500/cda" \
    --bias_type="race"

python -m explanation_generation.gen_explanation_encoder \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models/distilbert_civil_race/cda" \
    --num_labels=2 \
    --batch_size=32 \
    --max_length=512 \
    --num_examples=500 \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/distilbert_civil_race_race_val_500/cda" \
    --baseline="pad" \
    --bias_type="race" \
    --seed=42 \
    --only_predicted_class

python -m explanation_generation.gen_explanation_encoder \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models/distilbert_civil_all/cda" \
    --num_labels=2 \
    --batch_size=32 \
    --max_length=512 \
    --num_examples=500 \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/distilbert_civil_all_race_val_500/cda" \
    --baseline="pad" \
    --bias_type="race" \
    --seed=42 \
    --only_predicted_class

python -m sensitive_attribution_extraction.extract_sensitive_token_attribution \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/distilbert_civil_race_race_val_500/cda" \
    --split="val" \
    --bias_type="race" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m model_selection.compute_reliance_statistics \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/distilbert_civil_race_race_val_500/cda" \
    --split="val" \
    --bias_type="race" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m sensitive_attribution_extraction.extract_sensitive_token_attribution \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/distilbert_civil_all_race_val_500/cda" \
    --split="val" \
    --bias_type="race" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m model_selection.compute_reliance_statistics \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/distilbert_civil_all_race_val_500/cda" \
    --split="val" \
    --bias_type="race" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models/distilbert_civil_race/dropout" \
    --batch_size=32 \
    --max_seq_length=512 \
    --num_examples=500 \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/distilbert_civil_race_race_val_500/dropout" \
    --bias_type="race"

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models/distilbert_civil_all/dropout" \
    --batch_size=32 \
    --max_seq_length=512 \
    --num_examples=500 \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/distilbert_civil_all_race_val_500/dropout" \
    --bias_type="race"

python -m explanation_generation.gen_explanation_encoder \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models/distilbert_civil_race/dropout" \
    --num_labels=2 \
    --batch_size=32 \
    --max_length=512 \
    --num_examples=500 \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/distilbert_civil_race_race_val_500/dropout" \
    --baseline="pad" \
    --bias_type="race" \
    --seed=42 \
    --only_predicted_class

python -m explanation_generation.gen_explanation_encoder \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models/distilbert_civil_all/dropout" \
    --num_labels=2 \
    --batch_size=32 \
    --max_length=512 \
    --num_examples=500 \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/distilbert_civil_all_race_val_500/dropout" \
    --baseline="pad" \
    --bias_type="race" \
    --seed=42 \
    --only_predicted_class

python -m sensitive_attribution_extraction.extract_sensitive_token_attribution \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/distilbert_civil_race_race_val_500/dropout" \
    --split="val" \
    --bias_type="race" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m model_selection.compute_reliance_statistics \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/distilbert_civil_race_race_val_500/dropout" \
    --split="val" \
    --bias_type="race" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m sensitive_attribution_extraction.extract_sensitive_token_attribution \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/distilbert_civil_all_race_val_500/dropout" \
    --split="val" \
    --bias_type="race" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m model_selection.compute_reliance_statistics \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/distilbert_civil_all_race_val_500/dropout" \
    --split="val" \
    --bias_type="race" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models/distilbert_civil_race/attention_entropy" \
    --batch_size=32 \
    --max_seq_length=512 \
    --num_examples=500 \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/distilbert_civil_race_race_val_500/attention_entropy" \
    --bias_type="race"

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models/distilbert_civil_all/attention_entropy" \
    --batch_size=32 \
    --max_seq_length=512 \
    --num_examples=500 \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/distilbert_civil_all_race_val_500/attention_entropy" \
    --bias_type="race"

python -m explanation_generation.gen_explanation_encoder \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models/distilbert_civil_race/attention_entropy" \
    --num_labels=2 \
    --batch_size=32 \
    --max_length=512 \
    --num_examples=500 \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/distilbert_civil_race_race_val_500/attention_entropy" \
    --baseline="pad" \
    --bias_type="race" \
    --seed=42 \
    --only_predicted_class

python -m explanation_generation.gen_explanation_encoder \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models/distilbert_civil_all/attention_entropy" \
    --num_labels=2 \
    --batch_size=32 \
    --max_length=512 \
    --num_examples=500 \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/distilbert_civil_all_race_val_500/attention_entropy" \
    --baseline="pad" \
    --bias_type="race" \
    --seed=42 \
    --only_predicted_class

python -m sensitive_attribution_extraction.extract_sensitive_token_attribution \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/distilbert_civil_race_race_val_500/attention_entropy" \
    --split="val" \
    --bias_type="race" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m model_selection.compute_reliance_statistics \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/distilbert_civil_race_race_val_500/attention_entropy" \
    --split="val" \
    --bias_type="race" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m sensitive_attribution_extraction.extract_sensitive_token_attribution \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/distilbert_civil_all_race_val_500/attention_entropy" \
    --split="val" \
    --bias_type="race" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m model_selection.compute_reliance_statistics \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/distilbert_civil_all_race_val_500/attention_entropy" \
    --split="val" \
    --bias_type="race" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models/distilbert_civil_race/causal_debias" \
    --batch_size=32 \
    --max_seq_length=512 \
    --num_examples=500 \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/distilbert_civil_race_race_val_500/causal_debias" \
    --bias_type="race"

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models/distilbert_civil_all/causal_debias" \
    --batch_size=32 \
    --max_seq_length=512 \
    --num_examples=500 \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/distilbert_civil_all_race_val_500/causal_debias" \
    --bias_type="race"

python -m explanation_generation.gen_explanation_encoder \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models/distilbert_civil_race/causal_debias" \
    --num_labels=2 \
    --batch_size=32 \
    --max_length=512 \
    --num_examples=500 \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/distilbert_civil_race_race_val_500/causal_debias" \
    --baseline="pad" \
    --bias_type="race" \
    --seed=42 \
    --only_predicted_class

python -m explanation_generation.gen_explanation_encoder \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models/distilbert_civil_all/causal_debias" \
    --num_labels=2 \
    --batch_size=32 \
    --max_length=512 \
    --num_examples=500 \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/distilbert_civil_all_race_val_500/causal_debias" \
    --baseline="pad" \
    --bias_type="race" \
    --seed=42 \
    --only_predicted_class

python -m sensitive_attribution_extraction.extract_sensitive_token_attribution \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/distilbert_civil_race_race_val_500/causal_debias" \
    --split="val" \
    --bias_type="race" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m model_selection.compute_reliance_statistics \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/distilbert_civil_race_race_val_500/causal_debias" \
    --split="val" \
    --bias_type="race" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m sensitive_attribution_extraction.extract_sensitive_token_attribution \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/distilbert_civil_all_race_val_500/causal_debias" \
    --split="val" \
    --bias_type="race" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m model_selection.compute_reliance_statistics \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/distilbert_civil_all_race_val_500/causal_debias" \
    --split="val" \
    --bias_type="race" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m model_selection.compute_model_selection_correlation \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results" \
    --val_results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --bias_type="race" \
    --model_type="distilbert" \
    --num_test_examples=2000 \
    --num_val_examples=500 \
    --test_seed=-1 \
    --val_seed=-1

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models/bert_civil_race/no_debiasing" \
    --batch_size=32 \
    --max_seq_length=512 \
    --num_examples=1000 \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/bert_civil_race_race_val_1000/no_debiasing" \
    --bias_type="race"

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models/bert_civil_all/no_debiasing" \
    --batch_size=32 \
    --max_seq_length=512 \
    --num_examples=1000 \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/bert_civil_all_race_val_1000/no_debiasing" \
    --bias_type="race"

python -m explanation_generation.gen_explanation_encoder \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models/bert_civil_race/no_debiasing" \
    --num_labels=2 \
    --batch_size=32 \
    --max_length=512 \
    --num_examples=1000 \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/bert_civil_race_race_val_1000/no_debiasing" \
    --baseline="pad" \
    --bias_type="race" \
    --seed=42 \
    --only_predicted_class

python -m explanation_generation.gen_explanation_encoder \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models/bert_civil_all/no_debiasing" \
    --num_labels=2 \
    --batch_size=32 \
    --max_length=512 \
    --num_examples=1000 \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/bert_civil_all_race_val_1000/no_debiasing" \
    --baseline="pad" \
    --bias_type="race" \
    --seed=42 \
    --only_predicted_class

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

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models/bert_civil_race/group_balance" \
    --batch_size=32 \
    --max_seq_length=512 \
    --num_examples=1000 \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/bert_civil_race_race_val_1000/group_balance" \
    --bias_type="race"

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models/bert_civil_all/group_balance" \
    --batch_size=32 \
    --max_seq_length=512 \
    --num_examples=1000 \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/bert_civil_all_race_val_1000/group_balance" \
    --bias_type="race"

python -m explanation_generation.gen_explanation_encoder \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models/bert_civil_race/group_balance" \
    --num_labels=2 \
    --batch_size=32 \
    --max_length=512 \
    --num_examples=1000 \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/bert_civil_race_race_val_1000/group_balance" \
    --baseline="pad" \
    --bias_type="race" \
    --seed=42 \
    --only_predicted_class

python -m explanation_generation.gen_explanation_encoder \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models/bert_civil_all/group_balance" \
    --num_labels=2 \
    --batch_size=32 \
    --max_length=512 \
    --num_examples=1000 \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/bert_civil_all_race_val_1000/group_balance" \
    --baseline="pad" \
    --bias_type="race" \
    --seed=42 \
    --only_predicted_class

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

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models/bert_civil_race/group_class_balance" \
    --batch_size=32 \
    --max_seq_length=512 \
    --num_examples=1000 \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/bert_civil_race_race_val_1000/group_class_balance" \
    --bias_type="race"

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models/bert_civil_all/group_class_balance" \
    --batch_size=32 \
    --max_seq_length=512 \
    --num_examples=1000 \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/bert_civil_all_race_val_1000/group_class_balance" \
    --bias_type="race"

python -m explanation_generation.gen_explanation_encoder \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models/bert_civil_race/group_class_balance" \
    --num_labels=2 \
    --batch_size=32 \
    --max_length=512 \
    --num_examples=1000 \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/bert_civil_race_race_val_1000/group_class_balance" \
    --baseline="pad" \
    --bias_type="race" \
    --seed=42 \
    --only_predicted_class

python -m explanation_generation.gen_explanation_encoder \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models/bert_civil_all/group_class_balance" \
    --num_labels=2 \
    --batch_size=32 \
    --max_length=512 \
    --num_examples=1000 \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/bert_civil_all_race_val_1000/group_class_balance" \
    --baseline="pad" \
    --bias_type="race" \
    --seed=42 \
    --only_predicted_class

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

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models/bert_civil_race/cda" \
    --batch_size=32 \
    --max_seq_length=512 \
    --num_examples=1000 \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/bert_civil_race_race_val_1000/cda" \
    --bias_type="race"

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models/bert_civil_all/cda" \
    --batch_size=32 \
    --max_seq_length=512 \
    --num_examples=1000 \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/bert_civil_all_race_val_1000/cda" \
    --bias_type="race"

python -m explanation_generation.gen_explanation_encoder \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models/bert_civil_race/cda" \
    --num_labels=2 \
    --batch_size=32 \
    --max_length=512 \
    --num_examples=1000 \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/bert_civil_race_race_val_1000/cda" \
    --baseline="pad" \
    --bias_type="race" \
    --seed=42 \
    --only_predicted_class

python -m explanation_generation.gen_explanation_encoder \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models/bert_civil_all/cda" \
    --num_labels=2 \
    --batch_size=32 \
    --max_length=512 \
    --num_examples=1000 \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/bert_civil_all_race_val_1000/cda" \
    --baseline="pad" \
    --bias_type="race" \
    --seed=42 \
    --only_predicted_class

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

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models/bert_civil_race/dropout" \
    --batch_size=32 \
    --max_seq_length=512 \
    --num_examples=1000 \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/bert_civil_race_race_val_1000/dropout" \
    --bias_type="race"

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models/bert_civil_all/dropout" \
    --batch_size=32 \
    --max_seq_length=512 \
    --num_examples=1000 \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/bert_civil_all_race_val_1000/dropout" \
    --bias_type="race"

python -m explanation_generation.gen_explanation_encoder \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models/bert_civil_race/dropout" \
    --num_labels=2 \
    --batch_size=32 \
    --max_length=512 \
    --num_examples=1000 \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/bert_civil_race_race_val_1000/dropout" \
    --baseline="pad" \
    --bias_type="race" \
    --seed=42 \
    --only_predicted_class

python -m explanation_generation.gen_explanation_encoder \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models/bert_civil_all/dropout" \
    --num_labels=2 \
    --batch_size=32 \
    --max_length=512 \
    --num_examples=1000 \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/bert_civil_all_race_val_1000/dropout" \
    --baseline="pad" \
    --bias_type="race" \
    --seed=42 \
    --only_predicted_class

python -m sensitive_attribution_extraction.extract_sensitive_token_attribution \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/bert_civil_race_race_val_1000/dropout" \
    --split="val" \
    --bias_type="race" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m model_selection.compute_reliance_statistics \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/bert_civil_race_race_val_1000/dropout" \
    --split="val" \
    --bias_type="race" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m sensitive_attribution_extraction.extract_sensitive_token_attribution \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/bert_civil_all_race_val_1000/dropout" \
    --split="val" \
    --bias_type="race" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m model_selection.compute_reliance_statistics \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/bert_civil_all_race_val_1000/dropout" \
    --split="val" \
    --bias_type="race" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models/bert_civil_race/attention_entropy" \
    --batch_size=32 \
    --max_seq_length=512 \
    --num_examples=1000 \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/bert_civil_race_race_val_1000/attention_entropy" \
    --bias_type="race"

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models/bert_civil_all/attention_entropy" \
    --batch_size=32 \
    --max_seq_length=512 \
    --num_examples=1000 \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/bert_civil_all_race_val_1000/attention_entropy" \
    --bias_type="race"

python -m explanation_generation.gen_explanation_encoder \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models/bert_civil_race/attention_entropy" \
    --num_labels=2 \
    --batch_size=32 \
    --max_length=512 \
    --num_examples=1000 \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/bert_civil_race_race_val_1000/attention_entropy" \
    --baseline="pad" \
    --bias_type="race" \
    --seed=42 \
    --only_predicted_class

python -m explanation_generation.gen_explanation_encoder \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models/bert_civil_all/attention_entropy" \
    --num_labels=2 \
    --batch_size=32 \
    --max_length=512 \
    --num_examples=1000 \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/bert_civil_all_race_val_1000/attention_entropy" \
    --baseline="pad" \
    --bias_type="race" \
    --seed=42 \
    --only_predicted_class

python -m sensitive_attribution_extraction.extract_sensitive_token_attribution \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/bert_civil_race_race_val_1000/attention_entropy" \
    --split="val" \
    --bias_type="race" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m model_selection.compute_reliance_statistics \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/bert_civil_race_race_val_1000/attention_entropy" \
    --split="val" \
    --bias_type="race" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m sensitive_attribution_extraction.extract_sensitive_token_attribution \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/bert_civil_all_race_val_1000/attention_entropy" \
    --split="val" \
    --bias_type="race" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m model_selection.compute_reliance_statistics \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/bert_civil_all_race_val_1000/attention_entropy" \
    --split="val" \
    --bias_type="race" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models/bert_civil_race/causal_debias" \
    --batch_size=32 \
    --max_seq_length=512 \
    --num_examples=1000 \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/bert_civil_race_race_val_1000/causal_debias" \
    --bias_type="race"

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models/bert_civil_all/causal_debias" \
    --batch_size=32 \
    --max_seq_length=512 \
    --num_examples=1000 \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/bert_civil_all_race_val_1000/causal_debias" \
    --bias_type="race"

python -m explanation_generation.gen_explanation_encoder \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models/bert_civil_race/causal_debias" \
    --num_labels=2 \
    --batch_size=32 \
    --max_length=512 \
    --num_examples=1000 \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/bert_civil_race_race_val_1000/causal_debias" \
    --baseline="pad" \
    --bias_type="race" \
    --seed=42 \
    --only_predicted_class

python -m explanation_generation.gen_explanation_encoder \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models/bert_civil_all/causal_debias" \
    --num_labels=2 \
    --batch_size=32 \
    --max_length=512 \
    --num_examples=1000 \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/bert_civil_all_race_val_1000/causal_debias" \
    --baseline="pad" \
    --bias_type="race" \
    --seed=42 \
    --only_predicted_class

python -m sensitive_attribution_extraction.extract_sensitive_token_attribution \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/bert_civil_race_race_val_1000/causal_debias" \
    --split="val" \
    --bias_type="race" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m model_selection.compute_reliance_statistics \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/bert_civil_race_race_val_1000/causal_debias" \
    --split="val" \
    --bias_type="race" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m sensitive_attribution_extraction.extract_sensitive_token_attribution \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/bert_civil_all_race_val_1000/causal_debias" \
    --split="val" \
    --bias_type="race" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m model_selection.compute_reliance_statistics \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/bert_civil_all_race_val_1000/causal_debias" \
    --split="val" \
    --bias_type="race" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m model_selection.compute_model_selection_correlation \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results" \
    --val_results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --bias_type="race" \
    --model_type="bert" \
    --num_test_examples=2000 \
    --num_val_examples=1000 \
    --test_seed=-1 \
    --val_seed=-1

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models/roberta_civil_race/no_debiasing" \
    --batch_size=32 \
    --max_seq_length=512 \
    --num_examples=1000 \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/roberta_civil_race_race_val_1000/no_debiasing" \
    --bias_type="race"

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models/roberta_civil_all/no_debiasing" \
    --batch_size=32 \
    --max_seq_length=512 \
    --num_examples=1000 \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/roberta_civil_all_race_val_1000/no_debiasing" \
    --bias_type="race"

python -m explanation_generation.gen_explanation_encoder \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models/roberta_civil_race/no_debiasing" \
    --num_labels=2 \
    --batch_size=32 \
    --max_length=512 \
    --num_examples=1000 \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/roberta_civil_race_race_val_1000/no_debiasing" \
    --baseline="pad" \
    --bias_type="race" \
    --seed=42 \
    --only_predicted_class

python -m explanation_generation.gen_explanation_encoder \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models/roberta_civil_all/no_debiasing" \
    --num_labels=2 \
    --batch_size=32 \
    --max_length=512 \
    --num_examples=1000 \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/roberta_civil_all_race_val_1000/no_debiasing" \
    --baseline="pad" \
    --bias_type="race" \
    --seed=42 \
    --only_predicted_class

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

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models/roberta_civil_race/group_balance" \
    --batch_size=32 \
    --max_seq_length=512 \
    --num_examples=1000 \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/roberta_civil_race_race_val_1000/group_balance" \
    --bias_type="race"

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models/roberta_civil_all/group_balance" \
    --batch_size=32 \
    --max_seq_length=512 \
    --num_examples=1000 \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/roberta_civil_all_race_val_1000/group_balance" \
    --bias_type="race"

python -m explanation_generation.gen_explanation_encoder \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models/roberta_civil_race/group_balance" \
    --num_labels=2 \
    --batch_size=32 \
    --max_length=512 \
    --num_examples=1000 \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/roberta_civil_race_race_val_1000/group_balance" \
    --baseline="pad" \
    --bias_type="race" \
    --seed=42 \
    --only_predicted_class

python -m explanation_generation.gen_explanation_encoder \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models/roberta_civil_all/group_balance" \
    --num_labels=2 \
    --batch_size=32 \
    --max_length=512 \
    --num_examples=1000 \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/roberta_civil_all_race_val_1000/group_balance" \
    --baseline="pad" \
    --bias_type="race" \
    --seed=42 \
    --only_predicted_class

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

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models/roberta_civil_race/group_class_balance" \
    --batch_size=32 \
    --max_seq_length=512 \
    --num_examples=1000 \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/roberta_civil_race_race_val_1000/group_class_balance" \
    --bias_type="race"

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models/roberta_civil_all/group_class_balance" \
    --batch_size=32 \
    --max_seq_length=512 \
    --num_examples=1000 \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/roberta_civil_all_race_val_1000/group_class_balance" \
    --bias_type="race"

python -m explanation_generation.gen_explanation_encoder \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models/roberta_civil_race/group_class_balance" \
    --num_labels=2 \
    --batch_size=32 \
    --max_length=512 \
    --num_examples=1000 \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/roberta_civil_race_race_val_1000/group_class_balance" \
    --baseline="pad" \
    --bias_type="race" \
    --seed=42 \
    --only_predicted_class

python -m explanation_generation.gen_explanation_encoder \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models/roberta_civil_all/group_class_balance" \
    --num_labels=2 \
    --batch_size=32 \
    --max_length=512 \
    --num_examples=1000 \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/roberta_civil_all_race_val_1000/group_class_balance" \
    --baseline="pad" \
    --bias_type="race" \
    --seed=42 \
    --only_predicted_class

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

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models/roberta_civil_race/cda" \
    --batch_size=32 \
    --max_seq_length=512 \
    --num_examples=1000 \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/roberta_civil_race_race_val_1000/cda" \
    --bias_type="race"

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models/roberta_civil_all/cda" \
    --batch_size=32 \
    --max_seq_length=512 \
    --num_examples=1000 \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/roberta_civil_all_race_val_1000/cda" \
    --bias_type="race"

python -m explanation_generation.gen_explanation_encoder \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models/roberta_civil_race/cda" \
    --num_labels=2 \
    --batch_size=32 \
    --max_length=512 \
    --num_examples=1000 \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/roberta_civil_race_race_val_1000/cda" \
    --baseline="pad" \
    --bias_type="race" \
    --seed=42 \
    --only_predicted_class

python -m explanation_generation.gen_explanation_encoder \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models/roberta_civil_all/cda" \
    --num_labels=2 \
    --batch_size=32 \
    --max_length=512 \
    --num_examples=1000 \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/roberta_civil_all_race_val_1000/cda" \
    --baseline="pad" \
    --bias_type="race" \
    --seed=42 \
    --only_predicted_class

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

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models/roberta_civil_race/dropout" \
    --batch_size=32 \
    --max_seq_length=512 \
    --num_examples=1000 \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/roberta_civil_race_race_val_1000/dropout" \
    --bias_type="race"

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models/roberta_civil_all/dropout" \
    --batch_size=32 \
    --max_seq_length=512 \
    --num_examples=1000 \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/roberta_civil_all_race_val_1000/dropout" \
    --bias_type="race"

python -m explanation_generation.gen_explanation_encoder \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models/roberta_civil_race/dropout" \
    --num_labels=2 \
    --batch_size=32 \
    --max_length=512 \
    --num_examples=1000 \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/roberta_civil_race_race_val_1000/dropout" \
    --baseline="pad" \
    --bias_type="race" \
    --seed=42 \
    --only_predicted_class

python -m explanation_generation.gen_explanation_encoder \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models/roberta_civil_all/dropout" \
    --num_labels=2 \
    --batch_size=32 \
    --max_length=512 \
    --num_examples=1000 \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/roberta_civil_all_race_val_1000/dropout" \
    --baseline="pad" \
    --bias_type="race" \
    --seed=42 \
    --only_predicted_class

python -m sensitive_attribution_extraction.extract_sensitive_token_attribution \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/roberta_civil_race_race_val_1000/dropout" \
    --split="val" \
    --bias_type="race" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m model_selection.compute_reliance_statistics \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/roberta_civil_race_race_val_1000/dropout" \
    --split="val" \
    --bias_type="race" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m sensitive_attribution_extraction.extract_sensitive_token_attribution \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/roberta_civil_all_race_val_1000/dropout" \
    --split="val" \
    --bias_type="race" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m model_selection.compute_reliance_statistics \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/roberta_civil_all_race_val_1000/dropout" \
    --split="val" \
    --bias_type="race" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models/roberta_civil_race/attention_entropy" \
    --batch_size=32 \
    --max_seq_length=512 \
    --num_examples=1000 \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/roberta_civil_race_race_val_1000/attention_entropy" \
    --bias_type="race"

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models/roberta_civil_all/attention_entropy" \
    --batch_size=32 \
    --max_seq_length=512 \
    --num_examples=1000 \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/roberta_civil_all_race_val_1000/attention_entropy" \
    --bias_type="race"

python -m explanation_generation.gen_explanation_encoder \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models/roberta_civil_race/attention_entropy" \
    --num_labels=2 \
    --batch_size=32 \
    --max_length=512 \
    --num_examples=1000 \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/roberta_civil_race_race_val_1000/attention_entropy" \
    --baseline="pad" \
    --bias_type="race" \
    --seed=42 \
    --only_predicted_class

python -m explanation_generation.gen_explanation_encoder \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models/roberta_civil_all/attention_entropy" \
    --num_labels=2 \
    --batch_size=32 \
    --max_length=512 \
    --num_examples=1000 \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/roberta_civil_all_race_val_1000/attention_entropy" \
    --baseline="pad" \
    --bias_type="race" \
    --seed=42 \
    --only_predicted_class

python -m sensitive_attribution_extraction.extract_sensitive_token_attribution \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/roberta_civil_race_race_val_1000/attention_entropy" \
    --split="val" \
    --bias_type="race" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m model_selection.compute_reliance_statistics \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/roberta_civil_race_race_val_1000/attention_entropy" \
    --split="val" \
    --bias_type="race" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m sensitive_attribution_extraction.extract_sensitive_token_attribution \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/roberta_civil_all_race_val_1000/attention_entropy" \
    --split="val" \
    --bias_type="race" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m model_selection.compute_reliance_statistics \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/roberta_civil_all_race_val_1000/attention_entropy" \
    --split="val" \
    --bias_type="race" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models/roberta_civil_race/causal_debias" \
    --batch_size=32 \
    --max_seq_length=512 \
    --num_examples=1000 \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/roberta_civil_race_race_val_1000/causal_debias" \
    --bias_type="race"

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models/roberta_civil_all/causal_debias" \
    --batch_size=32 \
    --max_seq_length=512 \
    --num_examples=1000 \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/roberta_civil_all_race_val_1000/causal_debias" \
    --bias_type="race"

python -m explanation_generation.gen_explanation_encoder \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models/roberta_civil_race/causal_debias" \
    --num_labels=2 \
    --batch_size=32 \
    --max_length=512 \
    --num_examples=1000 \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/roberta_civil_race_race_val_1000/causal_debias" \
    --baseline="pad" \
    --bias_type="race" \
    --seed=42 \
    --only_predicted_class

python -m explanation_generation.gen_explanation_encoder \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models/roberta_civil_all/causal_debias" \
    --num_labels=2 \
    --batch_size=32 \
    --max_length=512 \
    --num_examples=1000 \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/roberta_civil_all_race_val_1000/causal_debias" \
    --baseline="pad" \
    --bias_type="race" \
    --seed=42 \
    --only_predicted_class

python -m sensitive_attribution_extraction.extract_sensitive_token_attribution \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/roberta_civil_race_race_val_1000/causal_debias" \
    --split="val" \
    --bias_type="race" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m model_selection.compute_reliance_statistics \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/roberta_civil_race_race_val_1000/causal_debias" \
    --split="val" \
    --bias_type="race" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m sensitive_attribution_extraction.extract_sensitive_token_attribution \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/roberta_civil_all_race_val_1000/causal_debias" \
    --split="val" \
    --bias_type="race" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m model_selection.compute_reliance_statistics \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/roberta_civil_all_race_val_1000/causal_debias" \
    --split="val" \
    --bias_type="race" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m model_selection.compute_model_selection_correlation \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results" \
    --val_results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --bias_type="race" \
    --model_type="roberta" \
    --num_test_examples=2000 \
    --num_val_examples=1000 \
    --test_seed=-1 \
    --val_seed=-1

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models/distilbert_civil_race/no_debiasing" \
    --batch_size=32 \
    --max_seq_length=512 \
    --num_examples=1000 \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/distilbert_civil_race_race_val_1000/no_debiasing" \
    --bias_type="race"

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models/distilbert_civil_all/no_debiasing" \
    --batch_size=32 \
    --max_seq_length=512 \
    --num_examples=1000 \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/distilbert_civil_all_race_val_1000/no_debiasing" \
    --bias_type="race"

python -m explanation_generation.gen_explanation_encoder \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models/distilbert_civil_race/no_debiasing" \
    --num_labels=2 \
    --batch_size=32 \
    --max_length=512 \
    --num_examples=1000 \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/distilbert_civil_race_race_val_1000/no_debiasing" \
    --baseline="pad" \
    --bias_type="race" \
    --seed=42 \
    --only_predicted_class

python -m explanation_generation.gen_explanation_encoder \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models/distilbert_civil_all/no_debiasing" \
    --num_labels=2 \
    --batch_size=32 \
    --max_length=512 \
    --num_examples=1000 \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/distilbert_civil_all_race_val_1000/no_debiasing" \
    --baseline="pad" \
    --bias_type="race" \
    --seed=42 \
    --only_predicted_class

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

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models/distilbert_civil_race/group_balance" \
    --batch_size=32 \
    --max_seq_length=512 \
    --num_examples=1000 \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/distilbert_civil_race_race_val_1000/group_balance" \
    --bias_type="race"

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models/distilbert_civil_all/group_balance" \
    --batch_size=32 \
    --max_seq_length=512 \
    --num_examples=1000 \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/distilbert_civil_all_race_val_1000/group_balance" \
    --bias_type="race"

python -m explanation_generation.gen_explanation_encoder \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models/distilbert_civil_race/group_balance" \
    --num_labels=2 \
    --batch_size=32 \
    --max_length=512 \
    --num_examples=1000 \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/distilbert_civil_race_race_val_1000/group_balance" \
    --baseline="pad" \
    --bias_type="race" \
    --seed=42 \
    --only_predicted_class

python -m explanation_generation.gen_explanation_encoder \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models/distilbert_civil_all/group_balance" \
    --num_labels=2 \
    --batch_size=32 \
    --max_length=512 \
    --num_examples=1000 \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/distilbert_civil_all_race_val_1000/group_balance" \
    --baseline="pad" \
    --bias_type="race" \
    --seed=42 \
    --only_predicted_class

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

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models/distilbert_civil_race/group_class_balance" \
    --batch_size=32 \
    --max_seq_length=512 \
    --num_examples=1000 \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/distilbert_civil_race_race_val_1000/group_class_balance" \
    --bias_type="race"

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models/distilbert_civil_all/group_class_balance" \
    --batch_size=32 \
    --max_seq_length=512 \
    --num_examples=1000 \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/distilbert_civil_all_race_val_1000/group_class_balance" \
    --bias_type="race"

python -m explanation_generation.gen_explanation_encoder \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models/distilbert_civil_race/group_class_balance" \
    --num_labels=2 \
    --batch_size=32 \
    --max_length=512 \
    --num_examples=1000 \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/distilbert_civil_race_race_val_1000/group_class_balance" \
    --baseline="pad" \
    --bias_type="race" \
    --seed=42 \
    --only_predicted_class

python -m explanation_generation.gen_explanation_encoder \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models/distilbert_civil_all/group_class_balance" \
    --num_labels=2 \
    --batch_size=32 \
    --max_length=512 \
    --num_examples=1000 \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/distilbert_civil_all_race_val_1000/group_class_balance" \
    --baseline="pad" \
    --bias_type="race" \
    --seed=42 \
    --only_predicted_class

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

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models/distilbert_civil_race/cda" \
    --batch_size=32 \
    --max_seq_length=512 \
    --num_examples=1000 \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/distilbert_civil_race_race_val_1000/cda" \
    --bias_type="race"

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models/distilbert_civil_all/cda" \
    --batch_size=32 \
    --max_seq_length=512 \
    --num_examples=1000 \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/distilbert_civil_all_race_val_1000/cda" \
    --bias_type="race"

python -m explanation_generation.gen_explanation_encoder \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models/distilbert_civil_race/cda" \
    --num_labels=2 \
    --batch_size=32 \
    --max_length=512 \
    --num_examples=1000 \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/distilbert_civil_race_race_val_1000/cda" \
    --baseline="pad" \
    --bias_type="race" \
    --seed=42 \
    --only_predicted_class

python -m explanation_generation.gen_explanation_encoder \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models/distilbert_civil_all/cda" \
    --num_labels=2 \
    --batch_size=32 \
    --max_length=512 \
    --num_examples=1000 \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/distilbert_civil_all_race_val_1000/cda" \
    --baseline="pad" \
    --bias_type="race" \
    --seed=42 \
    --only_predicted_class

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

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models/distilbert_civil_race/dropout" \
    --batch_size=32 \
    --max_seq_length=512 \
    --num_examples=1000 \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/distilbert_civil_race_race_val_1000/dropout" \
    --bias_type="race"

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models/distilbert_civil_all/dropout" \
    --batch_size=32 \
    --max_seq_length=512 \
    --num_examples=1000 \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/distilbert_civil_all_race_val_1000/dropout" \
    --bias_type="race"

python -m explanation_generation.gen_explanation_encoder \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models/distilbert_civil_race/dropout" \
    --num_labels=2 \
    --batch_size=32 \
    --max_length=512 \
    --num_examples=1000 \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/distilbert_civil_race_race_val_1000/dropout" \
    --baseline="pad" \
    --bias_type="race" \
    --seed=42 \
    --only_predicted_class

python -m explanation_generation.gen_explanation_encoder \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models/distilbert_civil_all/dropout" \
    --num_labels=2 \
    --batch_size=32 \
    --max_length=512 \
    --num_examples=1000 \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/distilbert_civil_all_race_val_1000/dropout" \
    --baseline="pad" \
    --bias_type="race" \
    --seed=42 \
    --only_predicted_class

python -m sensitive_attribution_extraction.extract_sensitive_token_attribution \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/distilbert_civil_race_race_val_1000/dropout" \
    --split="val" \
    --bias_type="race" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m model_selection.compute_reliance_statistics \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/distilbert_civil_race_race_val_1000/dropout" \
    --split="val" \
    --bias_type="race" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m sensitive_attribution_extraction.extract_sensitive_token_attribution \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/distilbert_civil_all_race_val_1000/dropout" \
    --split="val" \
    --bias_type="race" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m model_selection.compute_reliance_statistics \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/distilbert_civil_all_race_val_1000/dropout" \
    --split="val" \
    --bias_type="race" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models/distilbert_civil_race/attention_entropy" \
    --batch_size=32 \
    --max_seq_length=512 \
    --num_examples=1000 \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/distilbert_civil_race_race_val_1000/attention_entropy" \
    --bias_type="race"

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models/distilbert_civil_all/attention_entropy" \
    --batch_size=32 \
    --max_seq_length=512 \
    --num_examples=1000 \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/distilbert_civil_all_race_val_1000/attention_entropy" \
    --bias_type="race"

python -m explanation_generation.gen_explanation_encoder \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models/distilbert_civil_race/attention_entropy" \
    --num_labels=2 \
    --batch_size=32 \
    --max_length=512 \
    --num_examples=1000 \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/distilbert_civil_race_race_val_1000/attention_entropy" \
    --baseline="pad" \
    --bias_type="race" \
    --seed=42 \
    --only_predicted_class

python -m explanation_generation.gen_explanation_encoder \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models/distilbert_civil_all/attention_entropy" \
    --num_labels=2 \
    --batch_size=32 \
    --max_length=512 \
    --num_examples=1000 \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/distilbert_civil_all_race_val_1000/attention_entropy" \
    --baseline="pad" \
    --bias_type="race" \
    --seed=42 \
    --only_predicted_class

python -m sensitive_attribution_extraction.extract_sensitive_token_attribution \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/distilbert_civil_race_race_val_1000/attention_entropy" \
    --split="val" \
    --bias_type="race" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m model_selection.compute_reliance_statistics \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/distilbert_civil_race_race_val_1000/attention_entropy" \
    --split="val" \
    --bias_type="race" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m sensitive_attribution_extraction.extract_sensitive_token_attribution \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/distilbert_civil_all_race_val_1000/attention_entropy" \
    --split="val" \
    --bias_type="race" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m model_selection.compute_reliance_statistics \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/distilbert_civil_all_race_val_1000/attention_entropy" \
    --split="val" \
    --bias_type="race" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models/distilbert_civil_race/causal_debias" \
    --batch_size=32 \
    --max_seq_length=512 \
    --num_examples=1000 \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/distilbert_civil_race_race_val_1000/causal_debias" \
    --bias_type="race"

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models/distilbert_civil_all/causal_debias" \
    --batch_size=32 \
    --max_seq_length=512 \
    --num_examples=1000 \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/distilbert_civil_all_race_val_1000/causal_debias" \
    --bias_type="race"

python -m explanation_generation.gen_explanation_encoder \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models/distilbert_civil_race/causal_debias" \
    --num_labels=2 \
    --batch_size=32 \
    --max_length=512 \
    --num_examples=1000 \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/distilbert_civil_race_race_val_1000/causal_debias" \
    --baseline="pad" \
    --bias_type="race" \
    --seed=42 \
    --only_predicted_class

python -m explanation_generation.gen_explanation_encoder \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models/distilbert_civil_all/causal_debias" \
    --num_labels=2 \
    --batch_size=32 \
    --max_length=512 \
    --num_examples=1000 \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/distilbert_civil_all_race_val_1000/causal_debias" \
    --baseline="pad" \
    --bias_type="race" \
    --seed=42 \
    --only_predicted_class

python -m sensitive_attribution_extraction.extract_sensitive_token_attribution \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/distilbert_civil_race_race_val_1000/causal_debias" \
    --split="val" \
    --bias_type="race" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m model_selection.compute_reliance_statistics \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/distilbert_civil_race_race_val_1000/causal_debias" \
    --split="val" \
    --bias_type="race" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m sensitive_attribution_extraction.extract_sensitive_token_attribution \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/distilbert_civil_all_race_val_1000/causal_debias" \
    --split="val" \
    --bias_type="race" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m model_selection.compute_reliance_statistics \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/distilbert_civil_all_race_val_1000/causal_debias" \
    --split="val" \
    --bias_type="race" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m model_selection.compute_model_selection_correlation \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results" \
    --val_results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --bias_type="race" \
    --model_type="distilbert" \
    --num_test_examples=2000 \
    --num_val_examples=1000 \
    --test_seed=-1 \
    --val_seed=-1


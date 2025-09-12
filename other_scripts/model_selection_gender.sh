#!/bin/bash

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="google/jigsaw_unintended_bias" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models_jigsaw/roberta_jigsaw_gender/no_debiasing" \
    --batch_size=32 \
    --max_seq_length=512 \
    --num_examples=200 \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results_jigsaw/val_results/roberta_jigsaw_gender_gender_val_200_42/no_debiasing" \
    --bias_type="gender"\
    --shuffle \
    --seed=42

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="google/jigsaw_unintended_bias" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models_jigsaw/roberta_jigsaw_all/no_debiasing" \
    --batch_size=32 \
    --max_seq_length=512 \
    --num_examples=200 \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results_jigsaw/val_results/roberta_jigsaw_all_gender_val_200_42/no_debiasing" \
    --bias_type="gender"\
    --shuffle \
    --seed=42

python -m explanation_generation.gen_explanation_encoder \
    --dataset_name="google/jigsaw_unintended_bias" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models_jigsaw/roberta_jigsaw_gender/no_debiasing" \
    --num_labels=2 \
    --batch_size=32 \
    --max_length=512 \
    --num_examples=200 \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results_jigsaw/val_results/roberta_jigsaw_gender_gender_val_200_42/no_debiasing" \
    --baseline="pad" \
    --bias_type="gender" \
    --only_predicted_class \
    --shuffle \
    --seed=42

python -m explanation_generation.gen_explanation_encoder \
    --dataset_name="google/jigsaw_unintended_bias" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models_jigsaw/roberta_jigsaw_all/no_debiasing" \
    --num_labels=2 \
    --batch_size=32 \
    --max_length=512 \
    --num_examples=200 \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results_jigsaw/val_results/roberta_jigsaw_all_gender_val_200_42/no_debiasing" \
    --baseline="pad" \
    --bias_type="gender" \
    --only_predicted_class \
    --shuffle \
    --seed=42

python -m sensitive_attribution_extraction.extract_sensitive_token_attribution \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results_jigsaw/val_results/roberta_jigsaw_gender_gender_val_200_42/no_debiasing" \
    --split="val" \
    --bias_type="gender" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m model_selection.compute_reliance_statistics \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results_jigsaw/val_results/roberta_jigsaw_gender_gender_val_200_42/no_debiasing" \
    --split="val" \
    --bias_type="gender" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m sensitive_attribution_extraction.extract_sensitive_token_attribution \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results_jigsaw/val_results/roberta_jigsaw_all_gender_val_200_42/no_debiasing" \
    --split="val" \
    --bias_type="gender" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m model_selection.compute_reliance_statistics \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results_jigsaw/val_results/roberta_jigsaw_all_gender_val_200_42/no_debiasing" \
    --split="val" \
    --bias_type="gender" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="google/jigsaw_unintended_bias" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models_jigsaw/roberta_jigsaw_gender/group_balance" \
    --batch_size=32 \
    --max_seq_length=512 \
    --num_examples=200 \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results_jigsaw/val_results/roberta_jigsaw_gender_gender_val_200_42/group_balance" \
    --bias_type="gender"\
    --shuffle \
    --seed=42

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="google/jigsaw_unintended_bias" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models_jigsaw/roberta_jigsaw_all/group_balance" \
    --batch_size=32 \
    --max_seq_length=512 \
    --num_examples=200 \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results_jigsaw/val_results/roberta_jigsaw_all_gender_val_200_42/group_balance" \
    --bias_type="gender"\
    --shuffle \
    --seed=42

python -m explanation_generation.gen_explanation_encoder \
    --dataset_name="google/jigsaw_unintended_bias" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models_jigsaw/roberta_jigsaw_gender/group_balance" \
    --num_labels=2 \
    --batch_size=32 \
    --max_length=512 \
    --num_examples=200 \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results_jigsaw/val_results/roberta_jigsaw_gender_gender_val_200_42/group_balance" \
    --baseline="pad" \
    --bias_type="gender" \
    --only_predicted_class \
    --shuffle \
    --seed=42

python -m explanation_generation.gen_explanation_encoder \
    --dataset_name="google/jigsaw_unintended_bias" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models_jigsaw/roberta_jigsaw_all/group_balance" \
    --num_labels=2 \
    --batch_size=32 \
    --max_length=512 \
    --num_examples=200 \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results_jigsaw/val_results/roberta_jigsaw_all_gender_val_200_42/group_balance" \
    --baseline="pad" \
    --bias_type="gender" \
    --only_predicted_class \
    --shuffle \
    --seed=42

python -m sensitive_attribution_extraction.extract_sensitive_token_attribution \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results_jigsaw/val_results/roberta_jigsaw_gender_gender_val_200_42/group_balance" \
    --split="val" \
    --bias_type="gender" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m model_selection.compute_reliance_statistics \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results_jigsaw/val_results/roberta_jigsaw_gender_gender_val_200_42/group_balance" \
    --split="val" \
    --bias_type="gender" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m sensitive_attribution_extraction.extract_sensitive_token_attribution \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results_jigsaw/val_results/roberta_jigsaw_all_gender_val_200_42/group_balance" \
    --split="val" \
    --bias_type="gender" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m model_selection.compute_reliance_statistics \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results_jigsaw/val_results/roberta_jigsaw_all_gender_val_200_42/group_balance" \
    --split="val" \
    --bias_type="gender" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="google/jigsaw_unintended_bias" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models_jigsaw/roberta_jigsaw_gender/group_class_balance" \
    --batch_size=32 \
    --max_seq_length=512 \
    --num_examples=200 \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results_jigsaw/val_results/roberta_jigsaw_gender_gender_val_200_42/group_class_balance" \
    --bias_type="gender"\
    --shuffle \
    --seed=42

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="google/jigsaw_unintended_bias" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models_jigsaw/roberta_jigsaw_all/group_class_balance" \
    --batch_size=32 \
    --max_seq_length=512 \
    --num_examples=200 \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results_jigsaw/val_results/roberta_jigsaw_all_gender_val_200_42/group_class_balance" \
    --bias_type="gender"\
    --shuffle \
    --seed=42

python -m explanation_generation.gen_explanation_encoder \
    --dataset_name="google/jigsaw_unintended_bias" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models_jigsaw/roberta_jigsaw_gender/group_class_balance" \
    --num_labels=2 \
    --batch_size=32 \
    --max_length=512 \
    --num_examples=200 \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results_jigsaw/val_results/roberta_jigsaw_gender_gender_val_200_42/group_class_balance" \
    --baseline="pad" \
    --bias_type="gender" \
    --only_predicted_class \
    --shuffle \
    --seed=42

python -m explanation_generation.gen_explanation_encoder \
    --dataset_name="google/jigsaw_unintended_bias" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models_jigsaw/roberta_jigsaw_all/group_class_balance" \
    --num_labels=2 \
    --batch_size=32 \
    --max_length=512 \
    --num_examples=200 \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results_jigsaw/val_results/roberta_jigsaw_all_gender_val_200_42/group_class_balance" \
    --baseline="pad" \
    --bias_type="gender" \
    --only_predicted_class \
    --shuffle \
    --seed=42

python -m sensitive_attribution_extraction.extract_sensitive_token_attribution \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results_jigsaw/val_results/roberta_jigsaw_gender_gender_val_200_42/group_class_balance" \
    --split="val" \
    --bias_type="gender" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m model_selection.compute_reliance_statistics \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results_jigsaw/val_results/roberta_jigsaw_gender_gender_val_200_42/group_class_balance" \
    --split="val" \
    --bias_type="gender" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m sensitive_attribution_extraction.extract_sensitive_token_attribution \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results_jigsaw/val_results/roberta_jigsaw_all_gender_val_200_42/group_class_balance" \
    --split="val" \
    --bias_type="gender" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m model_selection.compute_reliance_statistics \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results_jigsaw/val_results/roberta_jigsaw_all_gender_val_200_42/group_class_balance" \
    --split="val" \
    --bias_type="gender" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="google/jigsaw_unintended_bias" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models_jigsaw/roberta_jigsaw_gender/cda" \
    --batch_size=32 \
    --max_seq_length=512 \
    --num_examples=200 \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results_jigsaw/val_results/roberta_jigsaw_gender_gender_val_200_42/cda" \
    --bias_type="gender"\
    --shuffle \
    --seed=42

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="google/jigsaw_unintended_bias" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models_jigsaw/roberta_jigsaw_all/cda" \
    --batch_size=32 \
    --max_seq_length=512 \
    --num_examples=200 \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results_jigsaw/val_results/roberta_jigsaw_all_gender_val_200_42/cda" \
    --bias_type="gender"\
    --shuffle \
    --seed=42

python -m explanation_generation.gen_explanation_encoder \
    --dataset_name="google/jigsaw_unintended_bias" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models_jigsaw/roberta_jigsaw_gender/cda" \
    --num_labels=2 \
    --batch_size=32 \
    --max_length=512 \
    --num_examples=200 \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results_jigsaw/val_results/roberta_jigsaw_gender_gender_val_200_42/cda" \
    --baseline="pad" \
    --bias_type="gender" \
    --only_predicted_class \
    --shuffle \
    --seed=42

python -m explanation_generation.gen_explanation_encoder \
    --dataset_name="google/jigsaw_unintended_bias" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models_jigsaw/roberta_jigsaw_all/cda" \
    --num_labels=2 \
    --batch_size=32 \
    --max_length=512 \
    --num_examples=200 \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results_jigsaw/val_results/roberta_jigsaw_all_gender_val_200_42/cda" \
    --baseline="pad" \
    --bias_type="gender" \
    --only_predicted_class \
    --shuffle \
    --seed=42

python -m sensitive_attribution_extraction.extract_sensitive_token_attribution \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results_jigsaw/val_results/roberta_jigsaw_gender_gender_val_200_42/cda" \
    --split="val" \
    --bias_type="gender" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m model_selection.compute_reliance_statistics \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results_jigsaw/val_results/roberta_jigsaw_gender_gender_val_200_42/cda" \
    --split="val" \
    --bias_type="gender" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m sensitive_attribution_extraction.extract_sensitive_token_attribution \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results_jigsaw/val_results/roberta_jigsaw_all_gender_val_200_42/cda" \
    --split="val" \
    --bias_type="gender" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m model_selection.compute_reliance_statistics \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results_jigsaw/val_results/roberta_jigsaw_all_gender_val_200_42/cda" \
    --split="val" \
    --bias_type="gender" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="google/jigsaw_unintended_bias" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models_jigsaw/roberta_jigsaw_gender/dropout" \
    --batch_size=32 \
    --max_seq_length=512 \
    --num_examples=200 \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results_jigsaw/val_results/roberta_jigsaw_gender_gender_val_200_42/dropout" \
    --bias_type="gender"\
    --shuffle \
    --seed=42

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="google/jigsaw_unintended_bias" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models_jigsaw/roberta_jigsaw_all/dropout" \
    --batch_size=32 \
    --max_seq_length=512 \
    --num_examples=200 \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results_jigsaw/val_results/roberta_jigsaw_all_gender_val_200_42/dropout" \
    --bias_type="gender"\
    --shuffle \
    --seed=42

python -m explanation_generation.gen_explanation_encoder \
    --dataset_name="google/jigsaw_unintended_bias" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models_jigsaw/roberta_jigsaw_gender/dropout" \
    --num_labels=2 \
    --batch_size=32 \
    --max_length=512 \
    --num_examples=200 \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results_jigsaw/val_results/roberta_jigsaw_gender_gender_val_200_42/dropout" \
    --baseline="pad" \
    --bias_type="gender" \
    --only_predicted_class \
    --shuffle \
    --seed=42

python -m explanation_generation.gen_explanation_encoder \
    --dataset_name="google/jigsaw_unintended_bias" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models_jigsaw/roberta_jigsaw_all/dropout" \
    --num_labels=2 \
    --batch_size=32 \
    --max_length=512 \
    --num_examples=200 \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results_jigsaw/val_results/roberta_jigsaw_all_gender_val_200_42/dropout" \
    --baseline="pad" \
    --bias_type="gender" \
    --only_predicted_class \
    --shuffle \
    --seed=42

python -m sensitive_attribution_extraction.extract_sensitive_token_attribution \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results_jigsaw/val_results/roberta_jigsaw_gender_gender_val_200_42/dropout" \
    --split="val" \
    --bias_type="gender" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m model_selection.compute_reliance_statistics \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results_jigsaw/val_results/roberta_jigsaw_gender_gender_val_200_42/dropout" \
    --split="val" \
    --bias_type="gender" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m sensitive_attribution_extraction.extract_sensitive_token_attribution \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results_jigsaw/val_results/roberta_jigsaw_all_gender_val_200_42/dropout" \
    --split="val" \
    --bias_type="gender" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m model_selection.compute_reliance_statistics \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results_jigsaw/val_results/roberta_jigsaw_all_gender_val_200_42/dropout" \
    --split="val" \
    --bias_type="gender" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="google/jigsaw_unintended_bias" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models_jigsaw/roberta_jigsaw_gender/attention_entropy" \
    --batch_size=32 \
    --max_seq_length=512 \
    --num_examples=200 \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results_jigsaw/val_results/roberta_jigsaw_gender_gender_val_200_42/attention_entropy" \
    --bias_type="gender"\
    --shuffle \
    --seed=42

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="google/jigsaw_unintended_bias" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models_jigsaw/roberta_jigsaw_all/attention_entropy" \
    --batch_size=32 \
    --max_seq_length=512 \
    --num_examples=200 \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results_jigsaw/val_results/roberta_jigsaw_all_gender_val_200_42/attention_entropy" \
    --bias_type="gender"\
    --shuffle \
    --seed=42

python -m explanation_generation.gen_explanation_encoder \
    --dataset_name="google/jigsaw_unintended_bias" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models_jigsaw/roberta_jigsaw_gender/attention_entropy" \
    --num_labels=2 \
    --batch_size=32 \
    --max_length=512 \
    --num_examples=200 \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results_jigsaw/val_results/roberta_jigsaw_gender_gender_val_200_42/attention_entropy" \
    --baseline="pad" \
    --bias_type="gender" \
    --only_predicted_class \
    --shuffle \
    --seed=42

python -m explanation_generation.gen_explanation_encoder \
    --dataset_name="google/jigsaw_unintended_bias" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models_jigsaw/roberta_jigsaw_all/attention_entropy" \
    --num_labels=2 \
    --batch_size=32 \
    --max_length=512 \
    --num_examples=200 \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results_jigsaw/val_results/roberta_jigsaw_all_gender_val_200_42/attention_entropy" \
    --baseline="pad" \
    --bias_type="gender" \
    --only_predicted_class \
    --shuffle \
    --seed=42

python -m sensitive_attribution_extraction.extract_sensitive_token_attribution \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results_jigsaw/val_results/roberta_jigsaw_gender_gender_val_200_42/attention_entropy" \
    --split="val" \
    --bias_type="gender" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m model_selection.compute_reliance_statistics \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results_jigsaw/val_results/roberta_jigsaw_gender_gender_val_200_42/attention_entropy" \
    --split="val" \
    --bias_type="gender" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m sensitive_attribution_extraction.extract_sensitive_token_attribution \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results_jigsaw/val_results/roberta_jigsaw_all_gender_val_200_42/attention_entropy" \
    --split="val" \
    --bias_type="gender" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m model_selection.compute_reliance_statistics \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results_jigsaw/val_results/roberta_jigsaw_all_gender_val_200_42/attention_entropy" \
    --split="val" \
    --bias_type="gender" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="google/jigsaw_unintended_bias" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models_jigsaw/roberta_jigsaw_gender/causal_debias" \
    --batch_size=32 \
    --max_seq_length=512 \
    --num_examples=200 \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results_jigsaw/val_results/roberta_jigsaw_gender_gender_val_200_42/causal_debias" \
    --bias_type="gender"\
    --shuffle \
    --seed=42

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="google/jigsaw_unintended_bias" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models_jigsaw/roberta_jigsaw_all/causal_debias" \
    --batch_size=32 \
    --max_seq_length=512 \
    --num_examples=200 \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results_jigsaw/val_results/roberta_jigsaw_all_gender_val_200_42/causal_debias" \
    --bias_type="gender"\
    --shuffle \
    --seed=42

python -m explanation_generation.gen_explanation_encoder \
    --dataset_name="google/jigsaw_unintended_bias" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models_jigsaw/roberta_jigsaw_gender/causal_debias" \
    --num_labels=2 \
    --batch_size=32 \
    --max_length=512 \
    --num_examples=200 \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results_jigsaw/val_results/roberta_jigsaw_gender_gender_val_200_42/causal_debias" \
    --baseline="pad" \
    --bias_type="gender" \
    --only_predicted_class \
    --shuffle \
    --seed=42

python -m explanation_generation.gen_explanation_encoder \
    --dataset_name="google/jigsaw_unintended_bias" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models_jigsaw/roberta_jigsaw_all/causal_debias" \
    --num_labels=2 \
    --batch_size=32 \
    --max_length=512 \
    --num_examples=200 \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results_jigsaw/val_results/roberta_jigsaw_all_gender_val_200_42/causal_debias" \
    --baseline="pad" \
    --bias_type="gender" \
    --only_predicted_class \
    --shuffle \
    --seed=42

python -m sensitive_attribution_extraction.extract_sensitive_token_attribution \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results_jigsaw/val_results/roberta_jigsaw_gender_gender_val_200_42/causal_debias" \
    --split="val" \
    --bias_type="gender" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m model_selection.compute_reliance_statistics \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results_jigsaw/val_results/roberta_jigsaw_gender_gender_val_200_42/causal_debias" \
    --split="val" \
    --bias_type="gender" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m sensitive_attribution_extraction.extract_sensitive_token_attribution \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results_jigsaw/val_results/roberta_jigsaw_all_gender_val_200_42/causal_debias" \
    --split="val" \
    --bias_type="gender" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m model_selection.compute_reliance_statistics \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results_jigsaw/val_results/roberta_jigsaw_all_gender_val_200_42/causal_debias" \
    --split="val" \
    --bias_type="gender" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m model_selection.compute_model_selection_correlation \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results_jigsaw" \
    --val_results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results_jigsaw/val_results" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --experiment_type=encoder \
    --bias_type="gender" \
    --model_type="roberta" \
    --num_test_examples=800 \
    --num_val_examples=200 \
    --test_seed=-1 \
    --val_seed=42 \
    --data_type="jigsaw" 


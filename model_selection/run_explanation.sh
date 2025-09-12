#!/usr/bin/env bash

python -m explanation_generation.gen_explanation_encoder \
    --dataset_name="lighteval/civil_comments_helm" \
    --split_ratio="8, 2" \
    --split="val" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/debiased_models/bert_civil_race/no_debiasing" \
    --num_labels=2 \
    --batch_size=8 \
    --max_length=512 \
    --num_examples=500 \
    --methods "Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/model_selection_results/bert_civil_no_debiasing_val_race_500" \
    --baseline="pad" \
    --bias_type="race" \
    --seed=42 \
    --only_predicted_class


python -m explanation_generation.gen_explanation_encoder \
    --dataset_name="lighteval/civil_comments_helm" \
    --split_ratio="8, 2" \
    --split="val" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/debiased_models/bert_civil_race/group_balance" \
    --num_labels=2 \
    --batch_size=8 \
    --max_length=512 \
    --num_examples=500 \
    --methods "Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/model_selection_results/bert_civil_group_balance_val_race_500" \
    --baseline="pad" \
    --bias_type="race" \
    --seed=42 \
    --only_predicted_class

python -m explanation_generation.gen_explanation_encoder \
    --dataset_name="lighteval/civil_comments_helm" \
    --split_ratio="8, 2" \
    --split="val" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/debiased_models/bert_civil_race/group_class_balance" \
    --num_labels=2 \
    --batch_size=8 \
    --max_length=512 \
    --num_examples=500 \
    --methods "Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/model_selection_results/bert_civil_group_class_balance_val_race_500" \
    --baseline="pad" \
    --bias_type="race" \
    --seed=42 \
    --only_predicted_class

python -m explanation_generation.gen_explanation_encoder \
    --dataset_name="lighteval/civil_comments_helm" \
    --split_ratio="8, 2" \
    --split="val" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/debiased_models/bert_civil_race/cda" \
    --num_labels=2 \
    --batch_size=8 \
    --max_length=512 \
    --num_examples=500 \
    --methods "Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/model_selection_results/bert_civil_cda_val_race_500" \
    --baseline="pad" \
    --bias_type="race" \
    --seed=42 \
    --only_predicted_class

python -m explanation_generation.gen_explanation_encoder \
    --dataset_name="lighteval/civil_comments_helm" \
    --split_ratio="8, 2" \
    --split="val" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/debiased_models/bert_civil_gender/no_debiasing" \
    --num_labels=2 \
    --batch_size=8 \
    --max_length=512 \
    --num_examples=500 \
    --methods "Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/model_selection_results/bert_civil_no_debiasing_val_gender_500" \
    --baseline="pad" \
    --bias_type="gender" \
    --seed=42 \
    --only_predicted_class

python -m explanation_generation.gen_explanation_encoder \
    --dataset_name="lighteval/civil_comments_helm" \
    --split_ratio="8, 2" \
    --split="val" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/debiased_models/bert_civil_gender/group_balance" \
    --num_labels=2 \
    --batch_size=8 \
    --max_length=512 \
    --num_examples=500 \
    --methods "Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/model_selection_results/bert_civil_group_balance_val_gender_500" \
    --baseline="pad" \
    --bias_type="gender" \
    --seed=42 \
    --only_predicted_class

python -m explanation_generation.gen_explanation_encoder \
    --dataset_name="lighteval/civil_comments_helm" \
    --split_ratio="8, 2" \
    --split="val" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/debiased_models/bert_civil_gender/group_class_balance" \
    --num_labels=2 \
    --batch_size=8 \
    --max_length=512 \
    --num_examples=500 \
    --methods "Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/model_selection_results/bert_civil_group_class_balance_val_gender_500" \
    --baseline="pad" \
    --bias_type="gender" \
    --seed=42 \
    --only_predicted_class

python -m explanation_generation.gen_explanation_encoder \
    --dataset_name="lighteval/civil_comments_helm" \
    --split_ratio="8, 2" \
    --split="val" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/debiased_models/bert_civil_gender/cda" \
    --num_labels=2 \
    --batch_size=8 \
    --max_length=512 \
    --num_examples=500 \
    --methods "Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/model_selection_results/bert_civil_cda_val_gender_500" \
    --baseline="pad" \
    --bias_type="gender" \
    --seed=42 \
    --only_predicted_class


python -m explanation_generation.gen_explanation_encoder \
    --dataset_name="lighteval/civil_comments_helm" \
    --split_ratio="8, 2" \
    --split="val" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/debiased_models/roberta_civil_race/no_debiasing" \
    --num_labels=2 \
    --batch_size=8 \
    --max_length=512 \
    --num_examples=500 \
    --methods "Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/model_selection_results/roberta_civil_no_debiasing_val_race_500" \
    --baseline="pad" \
    --bias_type="race" \
    --seed=42 \
    --only_predicted_class
    --only_predicted_class

python -m explanation_generation.gen_explanation_encoder \
    --dataset_name="lighteval/civil_comments_helm" \
    --split_ratio="8, 2" \
    --split="val" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/debiased_models/roberta_civil_race/group_balance" \
    --num_labels=2 \
    --batch_size=8 \
    --max_length=512 \
    --num_examples=500 \
    --methods "Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/model_selection_results/roberta_civil_group_balance_val_race_500" \
    --baseline="pad" \
    --bias_type="race" \
    --seed=42 \
    --only_predicted_class

python -m explanation_generation.gen_explanation_encoder \
    --dataset_name="lighteval/civil_comments_helm" \
    --split_ratio="8, 2" \
    --split="val" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/debiased_models/roberta_civil_race/group_class_balance" \
    --num_labels=2 \
    --batch_size=8 \
    --max_length=512 \
    --num_examples=500 \
    --methods "Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/model_selection_results/roberta_civil_group_class_balance_val_race_500" \
    --baseline="pad" \
    --bias_type="race" \
    --seed=42 \
    --only_predicted_class

python -m explanation_generation.gen_explanation_encoder \
    --dataset_name="lighteval/civil_comments_helm" \
    --split_ratio="8, 2" \
    --split="val" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/debiased_models/roberta_civil_race/cda" \
    --num_labels=2 \
    --batch_size=8 \
    --max_length=512 \
    --num_examples=500 \
    --methods "Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/model_selection_results/roberta_civil_cda_val_race_500" \
    --baseline="pad" \
    --bias_type="race" \
    --seed=42 \
    --only_predicted_class

python -m explanation_generation.gen_explanation_encoder \
    --dataset_name="lighteval/civil_comments_helm" \
    --split_ratio="8, 2" \
    --split="val" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/debiased_models/roberta_civil_gender/no_debiasing" \
    --num_labels=2 \
    --batch_size=8 \
    --max_length=512 \
    --num_examples=500 \
    --methods "Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/model_selection_results/roberta_civil_no_debiasing_val_gender_500" \
    --baseline="pad" \
    --bias_type="gender" \
    --seed=42 \
    --only_predicted_class

python -m explanation_generation.gen_explanation_encoder \
    --dataset_name="lighteval/civil_comments_helm" \
    --split_ratio="8, 2" \
    --split="val" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/debiased_models/roberta_civil_gender/group_balance" \
    --num_labels=2 \
    --batch_size=8 \
    --max_length=512 \
    --num_examples=500 \
    --methods "Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/model_selection_results/roberta_civil_group_balance_val_gender_500" \
    --baseline="pad" \
    --bias_type="gender" \
    --seed=42 \
    --only_predicted_class

python -m explanation_generation.gen_explanation_encoder \
    --dataset_name="lighteval/civil_comments_helm" \
    --split_ratio="8, 2" \
    --split="val" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/debiased_models/roberta_civil_gender/group_class_balance" \
    --num_labels=2 \
    --batch_size=8 \
    --max_length=512 \
    --num_examples=500 \
    --methods "Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/model_selection_results/roberta_civil_group_class_balance_val_gender_500" \
    --baseline="pad" \
    --bias_type="gender" \
    --seed=42 \
    --only_predicted_class

python -m explanation_generation.gen_explanation_encoder \
    --dataset_name="lighteval/civil_comments_helm" \
    --split_ratio="8, 2" \
    --split="val" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/debiased_models/roberta_civil_gender/cda" \
    --num_labels=2 \
    --batch_size=8 \
    --max_length=512 \
    --num_examples=500 \
    --methods "Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/model_selection_results/roberta_civil_cda_val_gender_500" \
    --baseline="pad" \
    --bias_type="gender" \
    --seed=42 \
    --only_predicted_class

python -m explanation_generation.gen_explanation_encoder \
    --dataset_name="lighteval/civil_comments_helm" \
    --split_ratio="8, 2" \
    --split="val" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/debiased_models/distilbert_civil_race/no_debiasing" \
    --num_labels=2 \
    --batch_size=8 \
    --max_length=512 \
    --num_examples=500 \
    --methods "Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/model_selection_results/distilbert_civil_no_debiasing_val_race_500" \
    --baseline="pad" \
    --bias_type="race" \
    --seed=42 \
    --only_predicted_class
    --only_predicted_class

python -m explanation_generation.gen_explanation_encoder \
    --dataset_name="lighteval/civil_comments_helm" \
    --split_ratio="8, 2" \
    --split="val" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/debiased_models/distilbert_civil_race/group_balance" \
    --num_labels=2 \
    --batch_size=8 \
    --max_length=512 \
    --num_examples=500 \
    --methods "Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/model_selection_results/distilbert_civil_group_balance_val_race_500" \
    --baseline="pad" \
    --bias_type="race" \
    --seed=42 \
    --only_predicted_class

python -m explanation_generation.gen_explanation_encoder \
    --dataset_name="lighteval/civil_comments_helm" \
    --split_ratio="8, 2" \
    --split="val" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/debiased_models/distilbert_civil_race/group_class_balance" \
    --num_labels=2 \
    --batch_size=8 \
    --max_length=512 \
    --num_examples=500 \
    --methods "Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/model_selection_results/distilbert_civil_group_class_balance_val_race_500" \
    --baseline="pad" \
    --bias_type="race" \
    --seed=42 \
    --only_predicted_class

python -m explanation_generation.gen_explanation_encoder \
    --dataset_name="lighteval/civil_comments_helm" \
    --split_ratio="8, 2" \
    --split="val" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/debiased_models/distilbert_civil_race/cda" \
    --num_labels=2 \
    --batch_size=8 \
    --max_length=512 \
    --num_examples=500 \
    --methods "Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/model_selection_results/distilbert_civil_cda_val_race_500" \
    --baseline="pad" \
    --bias_type="race" \
    --seed=42 \
    --only_predicted_class

python -m explanation_generation.gen_explanation_encoder \
    --dataset_name="lighteval/civil_comments_helm" \
    --split_ratio="8, 2" \
    --split="val" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/debiased_models/distilbert_civil_gender/no_debiasing" \
    --num_labels=2 \
    --batch_size=8 \
    --max_length=512 \
    --num_examples=500 \
    --methods "Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/model_selection_results/distilbert_civil_no_debiasing_val_gender_500" \
    --baseline="pad" \
    --bias_type="gender" \
    --seed=42 \
    --only_predicted_class

python -m explanation_generation.gen_explanation_encoder \
    --dataset_name="lighteval/civil_comments_helm" \
    --split_ratio="8, 2" \
    --split="val" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/debiased_models/distilbert_civil_gender/group_balance" \
    --num_labels=2 \
    --batch_size=8 \
    --max_length=512 \
    --num_examples=500 \
    --methods "Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/model_selection_results/distilbert_civil_group_balance_val_gender_500" \
    --baseline="pad" \
    --bias_type="gender" \
    --seed=42 \
    --only_predicted_class

python -m explanation_generation.gen_explanation_encoder \
    --dataset_name="lighteval/civil_comments_helm" \
    --split_ratio="8, 2" \
    --split="val" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/debiased_models/distilbert_civil_gender/group_class_balance" \
    --num_labels=2 \
    --batch_size=8 \
    --max_length=512 \
    --num_examples=500 \
    --methods "Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/model_selection_results/distilbert_civil_group_class_balance_val_gender_500" \
    --baseline="pad" \
    --bias_type="gender" \
    --seed=42 \
    --only_predicted_class

python -m explanation_generation.gen_explanation_encoder \
    --dataset_name="lighteval/civil_comments_helm" \
    --split_ratio="8, 2" \
    --split="val" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/debiased_models/distilbert_civil_gender/cda" \
    --num_labels=2 \
    --batch_size=8 \
    --max_length=512 \
    --num_examples=500 \
    --methods "Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/model_selection_results/distilbert_civil_cda_val_gender_500" \
    --baseline="pad" \
    --bias_type="gender" \
    --seed=42 \
    --only_predicted_class

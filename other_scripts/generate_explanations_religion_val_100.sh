#!/bin/bash
python -m explanation_generation.gen_explanation_encoder \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models/bert_civil_religion/no_debiasing" \
    --num_labels=2 \
    --batch_size=16 \
    --max_length=512 \
    --num_examples=100 \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/bert_civil_religion_religion_val_100/no_debiasing" \
    --baseline="pad" \
    --bias_type="religion" \
    --seed=42 \
    --only_predicted_class

python -m explanation_generation.gen_explanation_encoder \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models/bert_civil_all/no_debiasing" \
    --num_labels=2 \
    --batch_size=16 \
    --max_length=512 \
    --num_examples=100 \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/bert_civil_all_religion_val_100/no_debiasing" \
    --baseline="pad" \
    --bias_type="religion" \
    --seed=42 \
    --only_predicted_class

python -m explanation_generation.gen_explanation_encoder \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models/bert_civil_religion/group_balance" \
    --num_labels=2 \
    --batch_size=16 \
    --max_length=512 \
    --num_examples=100 \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/bert_civil_religion_religion_val_100/group_balance" \
    --baseline="pad" \
    --bias_type="religion" \
    --seed=42 \
    --only_predicted_class

python -m explanation_generation.gen_explanation_encoder \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models/bert_civil_all/group_balance" \
    --num_labels=2 \
    --batch_size=16 \
    --max_length=512 \
    --num_examples=100 \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/bert_civil_all_religion_val_100/group_balance" \
    --baseline="pad" \
    --bias_type="religion" \
    --seed=42 \
    --only_predicted_class

python -m explanation_generation.gen_explanation_encoder \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models/bert_civil_religion/group_class_balance" \
    --num_labels=2 \
    --batch_size=16 \
    --max_length=512 \
    --num_examples=100 \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/bert_civil_religion_religion_val_100/group_class_balance" \
    --baseline="pad" \
    --bias_type="religion" \
    --seed=42 \
    --only_predicted_class

python -m explanation_generation.gen_explanation_encoder \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models/bert_civil_all/group_class_balance" \
    --num_labels=2 \
    --batch_size=16 \
    --max_length=512 \
    --num_examples=100 \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/bert_civil_all_religion_val_100/group_class_balance" \
    --baseline="pad" \
    --bias_type="religion" \
    --seed=42 \
    --only_predicted_class

python -m explanation_generation.gen_explanation_encoder \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models/bert_civil_religion/cda" \
    --num_labels=2 \
    --batch_size=16 \
    --max_length=512 \
    --num_examples=100 \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/bert_civil_religion_religion_val_100/cda" \
    --baseline="pad" \
    --bias_type="religion" \
    --seed=42 \
    --only_predicted_class

python -m explanation_generation.gen_explanation_encoder \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models/bert_civil_all/cda" \
    --num_labels=2 \
    --batch_size=16 \
    --max_length=512 \
    --num_examples=100 \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/bert_civil_all_religion_val_100/cda" \
    --baseline="pad" \
    --bias_type="religion" \
    --seed=42 \
    --only_predicted_class

python -m explanation_generation.gen_explanation_encoder \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models/roberta_civil_religion/no_debiasing" \
    --num_labels=2 \
    --batch_size=16 \
    --max_length=512 \
    --num_examples=100 \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/roberta_civil_religion_religion_val_100/no_debiasing" \
    --baseline="pad" \
    --bias_type="religion" \
    --seed=42 \
    --only_predicted_class

python -m explanation_generation.gen_explanation_encoder \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models/roberta_civil_all/no_debiasing" \
    --num_labels=2 \
    --batch_size=16 \
    --max_length=512 \
    --num_examples=100 \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/roberta_civil_all_religion_val_100/no_debiasing" \
    --baseline="pad" \
    --bias_type="religion" \
    --seed=42 \
    --only_predicted_class

python -m explanation_generation.gen_explanation_encoder \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models/roberta_civil_religion/group_balance" \
    --num_labels=2 \
    --batch_size=16 \
    --max_length=512 \
    --num_examples=100 \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/roberta_civil_religion_religion_val_100/group_balance" \
    --baseline="pad" \
    --bias_type="religion" \
    --seed=42 \
    --only_predicted_class

python -m explanation_generation.gen_explanation_encoder \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models/roberta_civil_all/group_balance" \
    --num_labels=2 \
    --batch_size=16 \
    --max_length=512 \
    --num_examples=100 \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/roberta_civil_all_religion_val_100/group_balance" \
    --baseline="pad" \
    --bias_type="religion" \
    --seed=42 \
    --only_predicted_class

python -m explanation_generation.gen_explanation_encoder \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models/roberta_civil_religion/group_class_balance" \
    --num_labels=2 \
    --batch_size=16 \
    --max_length=512 \
    --num_examples=100 \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/roberta_civil_religion_religion_val_100/group_class_balance" \
    --baseline="pad" \
    --bias_type="religion" \
    --seed=42 \
    --only_predicted_class

python -m explanation_generation.gen_explanation_encoder \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models/roberta_civil_all/group_class_balance" \
    --num_labels=2 \
    --batch_size=16 \
    --max_length=512 \
    --num_examples=100 \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/roberta_civil_all_religion_val_100/group_class_balance" \
    --baseline="pad" \
    --bias_type="religion" \
    --seed=42 \
    --only_predicted_class

python -m explanation_generation.gen_explanation_encoder \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models/roberta_civil_religion/cda" \
    --num_labels=2 \
    --batch_size=16 \
    --max_length=512 \
    --num_examples=100 \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/roberta_civil_religion_religion_val_100/cda" \
    --baseline="pad" \
    --bias_type="religion" \
    --seed=42 \
    --only_predicted_class

python -m explanation_generation.gen_explanation_encoder \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models/roberta_civil_all/cda" \
    --num_labels=2 \
    --batch_size=16 \
    --max_length=512 \
    --num_examples=100 \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/roberta_civil_all_religion_val_100/cda" \
    --baseline="pad" \
    --bias_type="religion" \
    --seed=42 \
    --only_predicted_class

python -m explanation_generation.gen_explanation_encoder \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models/distilbert_civil_religion/no_debiasing" \
    --num_labels=2 \
    --batch_size=16 \
    --max_length=512 \
    --num_examples=100 \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/distilbert_civil_religion_religion_val_100/no_debiasing" \
    --baseline="pad" \
    --bias_type="religion" \
    --seed=42 \
    --only_predicted_class

python -m explanation_generation.gen_explanation_encoder \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models/distilbert_civil_all/no_debiasing" \
    --num_labels=2 \
    --batch_size=16 \
    --max_length=512 \
    --num_examples=100 \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/distilbert_civil_all_religion_val_100/no_debiasing" \
    --baseline="pad" \
    --bias_type="religion" \
    --seed=42 \
    --only_predicted_class

python -m explanation_generation.gen_explanation_encoder \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models/distilbert_civil_religion/group_balance" \
    --num_labels=2 \
    --batch_size=16 \
    --max_length=512 \
    --num_examples=100 \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/distilbert_civil_religion_religion_val_100/group_balance" \
    --baseline="pad" \
    --bias_type="religion" \
    --seed=42 \
    --only_predicted_class

python -m explanation_generation.gen_explanation_encoder \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models/distilbert_civil_all/group_balance" \
    --num_labels=2 \
    --batch_size=16 \
    --max_length=512 \
    --num_examples=100 \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/distilbert_civil_all_religion_val_100/group_balance" \
    --baseline="pad" \
    --bias_type="religion" \
    --seed=42 \
    --only_predicted_class

python -m explanation_generation.gen_explanation_encoder \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models/distilbert_civil_religion/group_class_balance" \
    --num_labels=2 \
    --batch_size=16 \
    --max_length=512 \
    --num_examples=100 \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/distilbert_civil_religion_religion_val_100/group_class_balance" \
    --baseline="pad" \
    --bias_type="religion" \
    --seed=42 \
    --only_predicted_class

python -m explanation_generation.gen_explanation_encoder \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models/distilbert_civil_all/group_class_balance" \
    --num_labels=2 \
    --batch_size=16 \
    --max_length=512 \
    --num_examples=100 \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/distilbert_civil_all_religion_val_100/group_class_balance" \
    --baseline="pad" \
    --bias_type="religion" \
    --seed=42 \
    --only_predicted_class

python -m explanation_generation.gen_explanation_encoder \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models/distilbert_civil_religion/cda" \
    --num_labels=2 \
    --batch_size=16 \
    --max_length=512 \
    --num_examples=100 \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/distilbert_civil_religion_religion_val_100/cda" \
    --baseline="pad" \
    --bias_type="religion" \
    --seed=42 \
    --only_predicted_class

python -m explanation_generation.gen_explanation_encoder \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models/distilbert_civil_all/cda" \
    --num_labels=2 \
    --batch_size=16 \
    --max_length=512 \
    --num_examples=100 \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/distilbert_civil_all_religion_val_100/cda" \
    --baseline="pad" \
    --bias_type="religion" \
    --seed=42 \
    --only_predicted_class


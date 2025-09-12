#!/usr/bin/env bash


#--methods "Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \

python -m explanation_generation.gen_explanation_encoder \
    --dataset_name="lighteval/civil_comments_helm" \
    --split_ratio="8, 2" \
    --split="test" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/models/baseline_bert_civil" \
    --num_labels=2 \
    --batch_size=1 \
    --max_length=512 \
    --num_examples=2000 \
    --methods "Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/results/baseline_bert_civil_2000_race" \
    --baseline="pad" \
    --bias_type="race" \
    --seed=42 \

python -m explanation_generation.gen_explanation_encoder \
    --dataset_name="lighteval/civil_comments_helm" \
    --split_ratio="8, 2" \
    --split="test" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/models/baseline_roberta_civil" \
    --num_labels=2 \
    --batch_size=1 \
    --max_length=512 \
    --num_examples=2000 \
    --methods "Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/results/baseline_roberta_civil_2000_race" \
    --baseline="pad" \
    --bias_type="race" \
    --seed=42 \

python -m explanation_generation.gen_explanation_encoder \
    --dataset_name="lighteval/civil_comments_helm" \
    --split_ratio="8, 2" \
    --split="test" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/models/baseline_distilbert_civil" \
    --num_labels=2 \
    --batch_size=1 \
    --max_length=512 \
    --num_examples=2000 \
    --methods "Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/results/baseline_distilbert_civil_2000_race" \
    --baseline="pad" \
    --bias_type="race" \
    --seed=42 \

python -m explanation_generation.gen_explanation_encoder \
    --dataset_name="lighteval/civil_comments_helm" \
    --split_ratio="8, 2" \
    --split="test" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/models/baseline_bert_civil" \
    --num_labels=2 \
    --batch_size=1 \
    --max_length=512 \
    --num_examples=2000 \
    --methods "Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/results/baseline_bert_civil_2000_gender" \
    --baseline="pad" \
    --bias_type="gender" \
    --seed=42 \

python -m explanation_generation.gen_explanation_encoder \
    --dataset_name="lighteval/civil_comments_helm" \
    --split_ratio="8, 2" \
    --split="test" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/models/baseline_roberta_civil" \
    --num_labels=2 \
    --batch_size=1 \
    --max_length=512 \
    --num_examples=2000 \
    --methods "Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/results/baseline_roberta_civil_2000_gender" \
    --baseline="pad" \
    --bias_type="gender" \
    --seed=42 \

python -m explanation_generation.gen_explanation_encoder \
    --dataset_name="lighteval/civil_comments_helm" \
    --split_ratio="8, 2" \
    --split="test" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/models/baseline_distilbert_civil" \
    --num_labels=2 \
    --batch_size=1 \
    --max_length=512 \
    --num_examples=2000 \
    --methods "Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/results/baseline_distilbert_civil_2000_gender" \
    --baseline="pad" \
    --bias_type="gender" \
    --seed=42 \

# python -m explanation_generation.gen_explanation_decoder \
#     --dataset_name="lighteval/civil_comments_helm" \
#     --split_ratio="8, 2" \
#     --split="test" \
#     --model_dir="meta-llama/Llama-3.1-8B-Instruct" \
#     --num_examples=2000 \
#     --methods "Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
#     --output_dir="/scratch/yifwang/new_fairness_x_explainability/results/llama_8b_civil_2000_race" \
#     --baseline="pad" \
#     --bias_type="race" \
#     --seed=42 \

# python -m explanation_generation.gen_explanation_decoder \
#     --dataset_name="lighteval/civil_comments_helm" \
#     --split_ratio="8, 2" \
#     --split="test" \
#     --model_dir="Qwen/Qwen2.5-7B-Instruct" \
#     --num_examples=2000 \
#     --methods "Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
#     --output_dir="/scratch/yifwang/new_fairness_x_explainability/results/qwen_7b_civil_2000_race" \
#     --baseline="pad" \
#     --bias_type="race" \
#     --seed=42 \

# python -m explanation_generation.gen_explanation_decoder \
#     --dataset_name="lighteval/civil_comments_helm" \
#     --split_ratio="8, 2" \
#     --split="test" \
#     --model_dir="meta-llama/Llama-3.1-8B-Instruct" \
#     --num_examples=2000 \
#     --methods "Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
#     --output_dir="/scratch/yifwang/new_fairness_x_explainability/results/llama_8b_civil_2000_race" \
#     --baseline="pad" \
#     --bias_type="race" \
#     --seed=42 \

# python -m explanation_generation.gen_explanation_decoder \
#     --dataset_name="lighteval/civil_comments_helm" \
#     --split_ratio="8, 2" \
#     --split="test" \
#     --model_dir="Qwen/Qwen2.5-7B-Instruct" \
#     --num_examples=2000 \
#     --methods "Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
#     --output_dir="/scratch/yifwang/new_fairness_x_explainability/results/qwen_7b_civil_2000_gender" \
#     --baseline="pad" \
#     --bias_type="gender" \
#     --seed=42 \
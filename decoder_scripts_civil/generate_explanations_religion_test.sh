#!/bin/bash



python -m explanation_generation.gen_explanation_decoder \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="test" \
    --split_ratio="8, 2" \
    --model_dir="Qwen/Qwen3-4B" \
    --num_examples=1000 \
    --methods="IntegratedGradients" \
    --output_dir="/scratch/yifwang/fairness_x_explainability/decoder_results_civil/qwen3_4b_civil_religion_test_1000/few_shot" \
    --baseline="pad" \
    --bias_type="religion" \
    --prompt_type="few_shot" \
    --seed=42 \
    --only_predicted_class

python -m explanation_generation.gen_explanation_decoder \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="test" \
    --split_ratio="8, 2" \
    --model_dir="Qwen/Qwen3-4B" \
    --num_examples=1000 \
    --methods="IntegratedGradients" \
    --output_dir="/scratch/yifwang/fairness_x_explainability/decoder_results_civil/qwen3_4b_civil_religion_test_1000/fairness_instruction" \
    --baseline="pad" \
    --bias_type="religion" \
    --prompt_type="fairness_instruction" \
    --seed=42 \
    --only_predicted_class


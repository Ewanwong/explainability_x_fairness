#!/bin/bash



python -m explanation_generation.gen_explanation_decoder \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="test" \
    --split_ratio="8, 2" \
    --model_dir="Qwen/Qwen3-4B" \
    --num_examples=2000 \
    --methods="IntegratedGradients" \
    --output_dir="/scratch/yifwang/fairness_x_explainability/decoder_results_civil/qwen3_4b_civil_race_test_2000/few_shot" \
    --baseline="pad" \
    --bias_type="race" \
    --prompt_type="few_shot" \
    --seed=42 \
    --only_predicted_class



#!/usr/bin/env bash

# bash encoder_scripts_jigsaw/model_selection_race.sh

python -m explanation_generation.gen_explanation_decoder \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="test" \
    --split_ratio="8, 2" \
    --model_dir="Qwen/Qwen3-4B" \
    --num_examples=2000 \
    --methods="Occlusion, IntegratedGradients" \
    --output_dir="/scratch/yifwang/fairness_x_explainability/decoder_results_civil/qwen3_4b_civil_race_test_2000/fairness_instruction" \
    --baseline="pad" \
    --bias_type="race" \
    --prompt_type="fairness_instruction" \
    --seed=42 \
    --only_predicted_class

bash encoder_scripts_jigsaw/model_selection_religion.sh
bash encoder_scripts_jigsaw/model_selection_gender.sh
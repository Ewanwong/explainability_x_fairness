#!/bin/bash
python -m explanation_generation.gen_explanation_decoder \
    --dataset_name="google/jigsaw_unintended_bias" \
    --split="test" \
    --split_ratio="8, 2" \
    --model_dir="meta-llama/Llama-3.2-3B-Instruct" \
    --num_examples=200 \
    --methods="IntegratedGradients" \
    --output_dir="/scratch/yifwang/fairness_x_explainability/decoder_results_jigsaw/llama_3b_jigsaw_religion_test_200/few_shot" \
    --baseline="pad" \
    --bias_type="religion" \
    --prompt_type="few_shot" \
    --seed=42 \
    --only_predicted_class

python -m explanation_generation.gen_explanation_decoder \
    --dataset_name="google/jigsaw_unintended_bias" \
    --split="test" \
    --split_ratio="8, 2" \
    --model_dir="Qwen/Qwen3-4B" \
    --num_examples=200 \
    --methods="IntegratedGradients" \
    --output_dir="/scratch/yifwang/fairness_x_explainability/decoder_results_jigsaw/qwen3_4b_jigsaw_religion_test_200/zero_shot" \
    --baseline="pad" \
    --bias_type="religion" \
    --prompt_type="zero_shot" \
    --seed=42 \
    --only_predicted_class


python -m explanation_generation.gen_explanation_decoder \
    --dataset_name="google/jigsaw_unintended_bias" \
    --split="test" \
    --split_ratio="8, 2" \
    --model_dir="Qwen/Qwen3-4B" \
    --num_examples=200 \
    --methods="IntegratedGradients" \
    --output_dir="/scratch/yifwang/fairness_x_explainability/decoder_results_jigsaw/qwen3_4b_jigsaw_religion_test_200/few_shot" \
    --baseline="pad" \
    --bias_type="religion" \
    --prompt_type="few_shot" \
    --seed=42 \
    --only_predicted_class





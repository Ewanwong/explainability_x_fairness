#!/bin/bash
python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="google/jigsaw_unintended_bias" \
    --split="test" \
    --split_ratio="8, 2" \
    --model_dir="meta-llama/Llama-3.2-3B-Instruct" \
    --batch_size=1 \
    --max_seq_length=512 \
    --num_examples=200 \
    --output_dir="/scratch/yifwang/fairness_x_explainability/decoder_results_jigsaw/llama_3b_jigsaw_religion_test_200/fairness_imagination" \
    --prompt_type="fairness_imagination" \
    --bias_type="religion"

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="google/jigsaw_unintended_bias" \
    --split="test" \
    --split_ratio="8, 2" \
    --model_dir="meta-llama/Llama-3.2-3B-Instruct" \
    --batch_size=1 \
    --max_seq_length=512 \
    --num_examples=200 \
    --output_dir="/scratch/yifwang/fairness_x_explainability/decoder_results_jigsaw/llama_3b_jigsaw_religion_test_200/fairness_instruction" \
    --prompt_type="fairness_instruction" \
    --bias_type="religion"

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="google/jigsaw_unintended_bias" \
    --split="test" \
    --split_ratio="8, 2" \
    --model_dir="Qwen/Qwen2.5-3B-Instruct" \
    --batch_size=1 \
    --max_seq_length=512 \
    --num_examples=200 \
    --output_dir="/scratch/yifwang/fairness_x_explainability/decoder_results_jigsaw/qwen_3b_jigsaw_religion_test_200/fairness_imagination" \
    --prompt_type="fairness_imagination" \
    --bias_type="religion"

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="google/jigsaw_unintended_bias" \
    --split="test" \
    --split_ratio="8, 2" \
    --model_dir="Qwen/Qwen2.5-3B-Instruct" \
    --batch_size=1 \
    --max_seq_length=512 \
    --num_examples=200 \
    --output_dir="/scratch/yifwang/fairness_x_explainability/decoder_results_jigsaw/qwen_3b_jigsaw_religion_test_200/fairness_instruction" \
    --prompt_type="fairness_instruction" \
    --bias_type="religion"

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="google/jigsaw_unintended_bias" \
    --split="test" \
    --split_ratio="8, 2" \
    --model_dir="Qwen/Qwen3-4B" \
    --batch_size=1 \
    --max_seq_length=512 \
    --num_examples=200 \
    --output_dir="/scratch/yifwang/fairness_x_explainability/decoder_results_jigsaw/qwen3_4b_jigsaw_religion_test_200/fairness_imagination" \
    --prompt_type="fairness_imagination" \
    --bias_type="religion"

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="google/jigsaw_unintended_bias" \
    --split="test" \
    --split_ratio="8, 2" \
    --model_dir="Qwen/Qwen3-4B" \
    --batch_size=1 \
    --max_seq_length=512 \
    --num_examples=200 \
    --output_dir="/scratch/yifwang/fairness_x_explainability/decoder_results_jigsaw/qwen3_4b_jigsaw_religion_test_200/fairness_instruction" \
    --prompt_type="fairness_instruction" \
    --bias_type="religion"


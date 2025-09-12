#!/bin/bash
python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="test" \
    --split_ratio="8, 2" \
    --model_dir="meta-llama/Llama-3.2-3B-Instruct" \
    --batch_size=1 \
    --max_seq_length=512 \
    --num_examples=1000 \
    --output_dir="/scratch/yifwang/fairness_x_explainability/decoder_results_civil/llama_3b_civil_religion_test_1000/fairness_imagination" \
    --prompt_type="fairness_imagination" \
    --bias_type="religion"

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="test" \
    --split_ratio="8, 2" \
    --model_dir="meta-llama/Llama-3.2-3B-Instruct" \
    --batch_size=1 \
    --max_seq_length=512 \
    --num_examples=1000 \
    --output_dir="/scratch/yifwang/fairness_x_explainability/decoder_results_civil/llama_3b_civil_religion_test_1000/fairness_instruction" \
    --prompt_type="fairness_instruction" \
    --bias_type="religion"

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="test" \
    --split_ratio="8, 2" \
    --model_dir="Qwen/Qwen2.5-3B-Instruct" \
    --batch_size=1 \
    --max_seq_length=512 \
    --num_examples=1000 \
    --output_dir="/scratch/yifwang/fairness_x_explainability/decoder_results_civil/qwen_3b_civil_religion_test_1000/fairness_imagination" \
    --prompt_type="fairness_imagination" \
    --bias_type="religion"

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="test" \
    --split_ratio="8, 2" \
    --model_dir="Qwen/Qwen2.5-3B-Instruct" \
    --batch_size=1 \
    --max_seq_length=512 \
    --num_examples=1000 \
    --output_dir="/scratch/yifwang/fairness_x_explainability/decoder_results_civil/qwen_3b_civil_religion_test_1000/fairness_instruction" \
    --prompt_type="fairness_instruction" \
    --bias_type="religion"

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="test" \
    --split_ratio="8, 2" \
    --model_dir="Qwen/Qwen3-4B" \
    --batch_size=1 \
    --max_seq_length=512 \
    --num_examples=1000 \
    --output_dir="/scratch/yifwang/fairness_x_explainability/decoder_results_civil/qwen3_4b_civil_religion_test_1000/fairness_imagination" \
    --prompt_type="fairness_imagination" \
    --bias_type="religion"

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="test" \
    --split_ratio="8, 2" \
    --model_dir="Qwen/Qwen3-4B" \
    --batch_size=1 \
    --max_seq_length=512 \
    --num_examples=1000 \
    --output_dir="/scratch/yifwang/fairness_x_explainability/decoder_results_civil/qwen3_4b_civil_religion_test_1000/fairness_instruction" \
    --prompt_type="fairness_instruction" \
    --bias_type="religion"


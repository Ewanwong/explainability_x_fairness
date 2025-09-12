#!/bin/bash
python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="google/jigsaw_unintended_bias" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="meta-llama/Llama-3.2-3B-Instruct" \
    --batch_size=1 \
    --max_seq_length=512 \
    --num_examples=20 \
    --output_dir="/scratch/yifwang/fairness_x_explainability/decoder_results_jigsaw/val_results/llama_3b_jigsaw_religion_val_20_1/zero_shot" \
    --bias_type="religion"\
    --prompt_type="zero_shot" \
    --shuffle \
    --seed=1

python -m explanation_generation.gen_explanation_decoder \
    --dataset_name="google/jigsaw_unintended_bias" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="meta-llama/Llama-3.2-3B-Instruct" \
    --num_labels=2 \
    --batch_size=1 \
    --max_length=512 \
    --num_examples=20 \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion" \
    --output_dir="/scratch/yifwang/fairness_x_explainability/decoder_results_jigsaw/val_results/llama_3b_jigsaw_religion_val_20_1/zero_shot" \
    --baseline="pad" \
    --bias_type="religion" \
    --prompt_type="zero_shot" \
    --only_predicted_class \
    --shuffle \
    --seed=1

python -m sensitive_attribution_extraction.extract_sensitive_token_attribution \
    --results_dir="/scratch/yifwang/fairness_x_explainability/decoder_results_jigsaw/val_results/llama_3b_jigsaw_religion_val_20_1/zero_shot" \
    --split="val" \
    --bias_type="religion" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion"

python -m model_selection.compute_reliance_statistics \
    --results_dir="/scratch/yifwang/fairness_x_explainability/decoder_results_jigsaw/val_results/llama_3b_jigsaw_religion_val_20_1/zero_shot" \
    --split="val" \
    --bias_type="religion" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion"

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="google/jigsaw_unintended_bias" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="meta-llama/Llama-3.2-3B-Instruct" \
    --batch_size=1 \
    --max_seq_length=512 \
    --num_examples=20 \
    --output_dir="/scratch/yifwang/fairness_x_explainability/decoder_results_jigsaw/val_results/llama_3b_jigsaw_religion_val_20_1/few_shot" \
    --bias_type="religion"\
    --prompt_type="few_shot" \
    --shuffle \
    --seed=1

python -m explanation_generation.gen_explanation_decoder \
    --dataset_name="google/jigsaw_unintended_bias" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="meta-llama/Llama-3.2-3B-Instruct" \
    --num_labels=2 \
    --batch_size=1 \
    --max_length=512 \
    --num_examples=20 \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion" \
    --output_dir="/scratch/yifwang/fairness_x_explainability/decoder_results_jigsaw/val_results/llama_3b_jigsaw_religion_val_20_1/few_shot" \
    --baseline="pad" \
    --bias_type="religion" \
    --prompt_type="few_shot" \
    --only_predicted_class \
    --shuffle \
    --seed=1

python -m sensitive_attribution_extraction.extract_sensitive_token_attribution \
    --results_dir="/scratch/yifwang/fairness_x_explainability/decoder_results_jigsaw/val_results/llama_3b_jigsaw_religion_val_20_1/few_shot" \
    --split="val" \
    --bias_type="religion" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion"

python -m model_selection.compute_reliance_statistics \
    --results_dir="/scratch/yifwang/fairness_x_explainability/decoder_results_jigsaw/val_results/llama_3b_jigsaw_religion_val_20_1/few_shot" \
    --split="val" \
    --bias_type="religion" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion"

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="google/jigsaw_unintended_bias" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="meta-llama/Llama-3.2-3B-Instruct" \
    --batch_size=1 \
    --max_seq_length=512 \
    --num_examples=20 \
    --output_dir="/scratch/yifwang/fairness_x_explainability/decoder_results_jigsaw/val_results/llama_3b_jigsaw_religion_val_20_1/fairness_imagination" \
    --bias_type="religion"\
    --prompt_type="fairness_imagination" \
    --shuffle \
    --seed=1

python -m explanation_generation.gen_explanation_decoder \
    --dataset_name="google/jigsaw_unintended_bias" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="meta-llama/Llama-3.2-3B-Instruct" \
    --num_labels=2 \
    --batch_size=1 \
    --max_length=512 \
    --num_examples=20 \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion" \
    --output_dir="/scratch/yifwang/fairness_x_explainability/decoder_results_jigsaw/val_results/llama_3b_jigsaw_religion_val_20_1/fairness_imagination" \
    --baseline="pad" \
    --bias_type="religion" \
    --prompt_type="fairness_imagination" \
    --only_predicted_class \
    --shuffle \
    --seed=1

python -m sensitive_attribution_extraction.extract_sensitive_token_attribution \
    --results_dir="/scratch/yifwang/fairness_x_explainability/decoder_results_jigsaw/val_results/llama_3b_jigsaw_religion_val_20_1/fairness_imagination" \
    --split="val" \
    --bias_type="religion" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion"

python -m model_selection.compute_reliance_statistics \
    --results_dir="/scratch/yifwang/fairness_x_explainability/decoder_results_jigsaw/val_results/llama_3b_jigsaw_religion_val_20_1/fairness_imagination" \
    --split="val" \
    --bias_type="religion" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion"

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="google/jigsaw_unintended_bias" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="meta-llama/Llama-3.2-3B-Instruct" \
    --batch_size=1 \
    --max_seq_length=512 \
    --num_examples=20 \
    --output_dir="/scratch/yifwang/fairness_x_explainability/decoder_results_jigsaw/val_results/llama_3b_jigsaw_religion_val_20_1/fairness_instruction" \
    --bias_type="religion"\
    --prompt_type="fairness_instruction" \
    --shuffle \
    --seed=1

python -m explanation_generation.gen_explanation_decoder \
    --dataset_name="google/jigsaw_unintended_bias" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="meta-llama/Llama-3.2-3B-Instruct" \
    --num_labels=2 \
    --batch_size=1 \
    --max_length=512 \
    --num_examples=20 \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion" \
    --output_dir="/scratch/yifwang/fairness_x_explainability/decoder_results_jigsaw/val_results/llama_3b_jigsaw_religion_val_20_1/fairness_instruction" \
    --baseline="pad" \
    --bias_type="religion" \
    --prompt_type="fairness_instruction" \
    --only_predicted_class \
    --shuffle \
    --seed=1

python -m sensitive_attribution_extraction.extract_sensitive_token_attribution \
    --results_dir="/scratch/yifwang/fairness_x_explainability/decoder_results_jigsaw/val_results/llama_3b_jigsaw_religion_val_20_1/fairness_instruction" \
    --split="val" \
    --bias_type="religion" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion"

python -m model_selection.compute_reliance_statistics \
    --results_dir="/scratch/yifwang/fairness_x_explainability/decoder_results_jigsaw/val_results/llama_3b_jigsaw_religion_val_20_1/fairness_instruction" \
    --split="val" \
    --bias_type="religion" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion"

python -m model_selection.compute_model_selection_correlation \
    --results_dir="/scratch/yifwang/fairness_x_explainability/decoder_results_jigsaw" \
    --val_results_dir="/scratch/yifwang/fairness_x_explainability/decoder_results_jigsaw/val_results" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion" \
    --experiment_type=decoder \
    --bias_type="religion" \
    --model_type="llama_3b" \
    --num_test_examples=200 \
    --num_val_examples=20 \
    --test_seed=-1 \
    --val_seed=1

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="google/jigsaw_unintended_bias" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="Qwen/Qwen3-4B" \
    --batch_size=1 \
    --max_seq_length=512 \
    --num_examples=20 \
    --output_dir="/scratch/yifwang/fairness_x_explainability/decoder_results_jigsaw/val_results/qwen3_4b_jigsaw_religion_val_20_1/zero_shot" \
    --bias_type="religion"\
    --prompt_type="zero_shot" \
    --shuffle \
    --seed=1

python -m explanation_generation.gen_explanation_decoder \
    --dataset_name="google/jigsaw_unintended_bias" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="Qwen/Qwen3-4B" \
    --num_labels=2 \
    --batch_size=1 \
    --max_length=512 \
    --num_examples=20 \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion" \
    --output_dir="/scratch/yifwang/fairness_x_explainability/decoder_results_jigsaw/val_results/qwen3_4b_jigsaw_religion_val_20_1/zero_shot" \
    --baseline="pad" \
    --bias_type="religion" \
    --prompt_type="zero_shot" \
    --only_predicted_class \
    --shuffle \
    --seed=1

python -m sensitive_attribution_extraction.extract_sensitive_token_attribution \
    --results_dir="/scratch/yifwang/fairness_x_explainability/decoder_results_jigsaw/val_results/qwen3_4b_jigsaw_religion_val_20_1/zero_shot" \
    --split="val" \
    --bias_type="religion" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion"

python -m model_selection.compute_reliance_statistics \
    --results_dir="/scratch/yifwang/fairness_x_explainability/decoder_results_jigsaw/val_results/qwen3_4b_jigsaw_religion_val_20_1/zero_shot" \
    --split="val" \
    --bias_type="religion" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion"

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="google/jigsaw_unintended_bias" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="Qwen/Qwen3-4B" \
    --batch_size=1 \
    --max_seq_length=512 \
    --num_examples=20 \
    --output_dir="/scratch/yifwang/fairness_x_explainability/decoder_results_jigsaw/val_results/qwen3_4b_jigsaw_religion_val_20_1/few_shot" \
    --bias_type="religion"\
    --prompt_type="few_shot" \
    --shuffle \
    --seed=1

python -m explanation_generation.gen_explanation_decoder \
    --dataset_name="google/jigsaw_unintended_bias" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="Qwen/Qwen3-4B" \
    --num_labels=2 \
    --batch_size=1 \
    --max_length=512 \
    --num_examples=20 \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion" \
    --output_dir="/scratch/yifwang/fairness_x_explainability/decoder_results_jigsaw/val_results/qwen3_4b_jigsaw_religion_val_20_1/few_shot" \
    --baseline="pad" \
    --bias_type="religion" \
    --prompt_type="few_shot" \
    --only_predicted_class \
    --shuffle \
    --seed=1

python -m sensitive_attribution_extraction.extract_sensitive_token_attribution \
    --results_dir="/scratch/yifwang/fairness_x_explainability/decoder_results_jigsaw/val_results/qwen3_4b_jigsaw_religion_val_20_1/few_shot" \
    --split="val" \
    --bias_type="religion" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion"

python -m model_selection.compute_reliance_statistics \
    --results_dir="/scratch/yifwang/fairness_x_explainability/decoder_results_jigsaw/val_results/qwen3_4b_jigsaw_religion_val_20_1/few_shot" \
    --split="val" \
    --bias_type="religion" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion"

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="google/jigsaw_unintended_bias" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="Qwen/Qwen3-4B" \
    --batch_size=1 \
    --max_seq_length=512 \
    --num_examples=20 \
    --output_dir="/scratch/yifwang/fairness_x_explainability/decoder_results_jigsaw/val_results/qwen3_4b_jigsaw_religion_val_20_1/fairness_imagination" \
    --bias_type="religion"\
    --prompt_type="fairness_imagination" \
    --shuffle \
    --seed=1

python -m explanation_generation.gen_explanation_decoder \
    --dataset_name="google/jigsaw_unintended_bias" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="Qwen/Qwen3-4B" \
    --num_labels=2 \
    --batch_size=1 \
    --max_length=512 \
    --num_examples=20 \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion" \
    --output_dir="/scratch/yifwang/fairness_x_explainability/decoder_results_jigsaw/val_results/qwen3_4b_jigsaw_religion_val_20_1/fairness_imagination" \
    --baseline="pad" \
    --bias_type="religion" \
    --prompt_type="fairness_imagination" \
    --only_predicted_class \
    --shuffle \
    --seed=1

python -m sensitive_attribution_extraction.extract_sensitive_token_attribution \
    --results_dir="/scratch/yifwang/fairness_x_explainability/decoder_results_jigsaw/val_results/qwen3_4b_jigsaw_religion_val_20_1/fairness_imagination" \
    --split="val" \
    --bias_type="religion" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion"

python -m model_selection.compute_reliance_statistics \
    --results_dir="/scratch/yifwang/fairness_x_explainability/decoder_results_jigsaw/val_results/qwen3_4b_jigsaw_religion_val_20_1/fairness_imagination" \
    --split="val" \
    --bias_type="religion" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion"

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="google/jigsaw_unintended_bias" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="Qwen/Qwen3-4B" \
    --batch_size=1 \
    --max_seq_length=512 \
    --num_examples=20 \
    --output_dir="/scratch/yifwang/fairness_x_explainability/decoder_results_jigsaw/val_results/qwen3_4b_jigsaw_religion_val_20_1/fairness_instruction" \
    --bias_type="religion"\
    --prompt_type="fairness_instruction" \
    --shuffle \
    --seed=1

python -m explanation_generation.gen_explanation_decoder \
    --dataset_name="google/jigsaw_unintended_bias" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="Qwen/Qwen3-4B" \
    --num_labels=2 \
    --batch_size=1 \
    --max_length=512 \
    --num_examples=20 \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion" \
    --output_dir="/scratch/yifwang/fairness_x_explainability/decoder_results_jigsaw/val_results/qwen3_4b_jigsaw_religion_val_20_1/fairness_instruction" \
    --baseline="pad" \
    --bias_type="religion" \
    --prompt_type="fairness_instruction" \
    --only_predicted_class \
    --shuffle \
    --seed=1

python -m sensitive_attribution_extraction.extract_sensitive_token_attribution \
    --results_dir="/scratch/yifwang/fairness_x_explainability/decoder_results_jigsaw/val_results/qwen3_4b_jigsaw_religion_val_20_1/fairness_instruction" \
    --split="val" \
    --bias_type="religion" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion"

python -m model_selection.compute_reliance_statistics \
    --results_dir="/scratch/yifwang/fairness_x_explainability/decoder_results_jigsaw/val_results/qwen3_4b_jigsaw_religion_val_20_1/fairness_instruction" \
    --split="val" \
    --bias_type="religion" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion"

python -m model_selection.compute_model_selection_correlation \
    --results_dir="/scratch/yifwang/fairness_x_explainability/decoder_results_jigsaw" \
    --val_results_dir="/scratch/yifwang/fairness_x_explainability/decoder_results_jigsaw/val_results" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion" \
    --experiment_type=decoder \
    --bias_type="religion" \
    --model_type="qwen3_4b" \
    --num_test_examples=200 \
    --num_val_examples=20 \
    --test_seed=-1 \
    --val_seed=1

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="google/jigsaw_unintended_bias" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="meta-llama/Llama-3.2-3B-Instruct" \
    --batch_size=1 \
    --max_seq_length=512 \
    --num_examples=20 \
    --output_dir="/scratch/yifwang/fairness_x_explainability/decoder_results_jigsaw/val_results/llama_3b_jigsaw_religion_val_20_2/zero_shot" \
    --bias_type="religion"\
    --prompt_type="zero_shot" \
    --shuffle \
    --seed=2

python -m explanation_generation.gen_explanation_decoder \
    --dataset_name="google/jigsaw_unintended_bias" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="meta-llama/Llama-3.2-3B-Instruct" \
    --num_labels=2 \
    --batch_size=1 \
    --max_length=512 \
    --num_examples=20 \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion" \
    --output_dir="/scratch/yifwang/fairness_x_explainability/decoder_results_jigsaw/val_results/llama_3b_jigsaw_religion_val_20_2/zero_shot" \
    --baseline="pad" \
    --bias_type="religion" \
    --prompt_type="zero_shot" \
    --only_predicted_class \
    --shuffle \
    --seed=2

python -m sensitive_attribution_extraction.extract_sensitive_token_attribution \
    --results_dir="/scratch/yifwang/fairness_x_explainability/decoder_results_jigsaw/val_results/llama_3b_jigsaw_religion_val_20_2/zero_shot" \
    --split="val" \
    --bias_type="religion" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion"

python -m model_selection.compute_reliance_statistics \
    --results_dir="/scratch/yifwang/fairness_x_explainability/decoder_results_jigsaw/val_results/llama_3b_jigsaw_religion_val_20_2/zero_shot" \
    --split="val" \
    --bias_type="religion" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion"

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="google/jigsaw_unintended_bias" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="meta-llama/Llama-3.2-3B-Instruct" \
    --batch_size=1 \
    --max_seq_length=512 \
    --num_examples=20 \
    --output_dir="/scratch/yifwang/fairness_x_explainability/decoder_results_jigsaw/val_results/llama_3b_jigsaw_religion_val_20_2/few_shot" \
    --bias_type="religion"\
    --prompt_type="few_shot" \
    --shuffle \
    --seed=2

python -m explanation_generation.gen_explanation_decoder \
    --dataset_name="google/jigsaw_unintended_bias" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="meta-llama/Llama-3.2-3B-Instruct" \
    --num_labels=2 \
    --batch_size=1 \
    --max_length=512 \
    --num_examples=20 \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion" \
    --output_dir="/scratch/yifwang/fairness_x_explainability/decoder_results_jigsaw/val_results/llama_3b_jigsaw_religion_val_20_2/few_shot" \
    --baseline="pad" \
    --bias_type="religion" \
    --prompt_type="few_shot" \
    --only_predicted_class \
    --shuffle \
    --seed=2

python -m sensitive_attribution_extraction.extract_sensitive_token_attribution \
    --results_dir="/scratch/yifwang/fairness_x_explainability/decoder_results_jigsaw/val_results/llama_3b_jigsaw_religion_val_20_2/few_shot" \
    --split="val" \
    --bias_type="religion" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion"

python -m model_selection.compute_reliance_statistics \
    --results_dir="/scratch/yifwang/fairness_x_explainability/decoder_results_jigsaw/val_results/llama_3b_jigsaw_religion_val_20_2/few_shot" \
    --split="val" \
    --bias_type="religion" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion"

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="google/jigsaw_unintended_bias" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="meta-llama/Llama-3.2-3B-Instruct" \
    --batch_size=1 \
    --max_seq_length=512 \
    --num_examples=20 \
    --output_dir="/scratch/yifwang/fairness_x_explainability/decoder_results_jigsaw/val_results/llama_3b_jigsaw_religion_val_20_2/fairness_imagination" \
    --bias_type="religion"\
    --prompt_type="fairness_imagination" \
    --shuffle \
    --seed=2

python -m explanation_generation.gen_explanation_decoder \
    --dataset_name="google/jigsaw_unintended_bias" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="meta-llama/Llama-3.2-3B-Instruct" \
    --num_labels=2 \
    --batch_size=1 \
    --max_length=512 \
    --num_examples=20 \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion" \
    --output_dir="/scratch/yifwang/fairness_x_explainability/decoder_results_jigsaw/val_results/llama_3b_jigsaw_religion_val_20_2/fairness_imagination" \
    --baseline="pad" \
    --bias_type="religion" \
    --prompt_type="fairness_imagination" \
    --only_predicted_class \
    --shuffle \
    --seed=2

python -m sensitive_attribution_extraction.extract_sensitive_token_attribution \
    --results_dir="/scratch/yifwang/fairness_x_explainability/decoder_results_jigsaw/val_results/llama_3b_jigsaw_religion_val_20_2/fairness_imagination" \
    --split="val" \
    --bias_type="religion" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion"

python -m model_selection.compute_reliance_statistics \
    --results_dir="/scratch/yifwang/fairness_x_explainability/decoder_results_jigsaw/val_results/llama_3b_jigsaw_religion_val_20_2/fairness_imagination" \
    --split="val" \
    --bias_type="religion" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion"

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="google/jigsaw_unintended_bias" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="meta-llama/Llama-3.2-3B-Instruct" \
    --batch_size=1 \
    --max_seq_length=512 \
    --num_examples=20 \
    --output_dir="/scratch/yifwang/fairness_x_explainability/decoder_results_jigsaw/val_results/llama_3b_jigsaw_religion_val_20_2/fairness_instruction" \
    --bias_type="religion"\
    --prompt_type="fairness_instruction" \
    --shuffle \
    --seed=2

python -m explanation_generation.gen_explanation_decoder \
    --dataset_name="google/jigsaw_unintended_bias" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="meta-llama/Llama-3.2-3B-Instruct" \
    --num_labels=2 \
    --batch_size=1 \
    --max_length=512 \
    --num_examples=20 \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion" \
    --output_dir="/scratch/yifwang/fairness_x_explainability/decoder_results_jigsaw/val_results/llama_3b_jigsaw_religion_val_20_2/fairness_instruction" \
    --baseline="pad" \
    --bias_type="religion" \
    --prompt_type="fairness_instruction" \
    --only_predicted_class \
    --shuffle \
    --seed=2

python -m sensitive_attribution_extraction.extract_sensitive_token_attribution \
    --results_dir="/scratch/yifwang/fairness_x_explainability/decoder_results_jigsaw/val_results/llama_3b_jigsaw_religion_val_20_2/fairness_instruction" \
    --split="val" \
    --bias_type="religion" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion"

python -m model_selection.compute_reliance_statistics \
    --results_dir="/scratch/yifwang/fairness_x_explainability/decoder_results_jigsaw/val_results/llama_3b_jigsaw_religion_val_20_2/fairness_instruction" \
    --split="val" \
    --bias_type="religion" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion"

python -m model_selection.compute_model_selection_correlation \
    --results_dir="/scratch/yifwang/fairness_x_explainability/decoder_results_jigsaw" \
    --val_results_dir="/scratch/yifwang/fairness_x_explainability/decoder_results_jigsaw/val_results" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion" \
    --experiment_type=decoder \
    --bias_type="religion" \
    --model_type="llama_3b" \
    --num_test_examples=200 \
    --num_val_examples=20 \
    --test_seed=-1 \
    --val_seed=2

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="google/jigsaw_unintended_bias" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="Qwen/Qwen3-4B" \
    --batch_size=1 \
    --max_seq_length=512 \
    --num_examples=20 \
    --output_dir="/scratch/yifwang/fairness_x_explainability/decoder_results_jigsaw/val_results/qwen3_4b_jigsaw_religion_val_20_2/zero_shot" \
    --bias_type="religion"\
    --prompt_type="zero_shot" \
    --shuffle \
    --seed=2

python -m explanation_generation.gen_explanation_decoder \
    --dataset_name="google/jigsaw_unintended_bias" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="Qwen/Qwen3-4B" \
    --num_labels=2 \
    --batch_size=1 \
    --max_length=512 \
    --num_examples=20 \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion" \
    --output_dir="/scratch/yifwang/fairness_x_explainability/decoder_results_jigsaw/val_results/qwen3_4b_jigsaw_religion_val_20_2/zero_shot" \
    --baseline="pad" \
    --bias_type="religion" \
    --prompt_type="zero_shot" \
    --only_predicted_class \
    --shuffle \
    --seed=2

python -m sensitive_attribution_extraction.extract_sensitive_token_attribution \
    --results_dir="/scratch/yifwang/fairness_x_explainability/decoder_results_jigsaw/val_results/qwen3_4b_jigsaw_religion_val_20_2/zero_shot" \
    --split="val" \
    --bias_type="religion" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion"

python -m model_selection.compute_reliance_statistics \
    --results_dir="/scratch/yifwang/fairness_x_explainability/decoder_results_jigsaw/val_results/qwen3_4b_jigsaw_religion_val_20_2/zero_shot" \
    --split="val" \
    --bias_type="religion" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion"

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="google/jigsaw_unintended_bias" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="Qwen/Qwen3-4B" \
    --batch_size=1 \
    --max_seq_length=512 \
    --num_examples=20 \
    --output_dir="/scratch/yifwang/fairness_x_explainability/decoder_results_jigsaw/val_results/qwen3_4b_jigsaw_religion_val_20_2/few_shot" \
    --bias_type="religion"\
    --prompt_type="few_shot" \
    --shuffle \
    --seed=2

python -m explanation_generation.gen_explanation_decoder \
    --dataset_name="google/jigsaw_unintended_bias" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="Qwen/Qwen3-4B" \
    --num_labels=2 \
    --batch_size=1 \
    --max_length=512 \
    --num_examples=20 \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion" \
    --output_dir="/scratch/yifwang/fairness_x_explainability/decoder_results_jigsaw/val_results/qwen3_4b_jigsaw_religion_val_20_2/few_shot" \
    --baseline="pad" \
    --bias_type="religion" \
    --prompt_type="few_shot" \
    --only_predicted_class \
    --shuffle \
    --seed=2

python -m sensitive_attribution_extraction.extract_sensitive_token_attribution \
    --results_dir="/scratch/yifwang/fairness_x_explainability/decoder_results_jigsaw/val_results/qwen3_4b_jigsaw_religion_val_20_2/few_shot" \
    --split="val" \
    --bias_type="religion" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion"

python -m model_selection.compute_reliance_statistics \
    --results_dir="/scratch/yifwang/fairness_x_explainability/decoder_results_jigsaw/val_results/qwen3_4b_jigsaw_religion_val_20_2/few_shot" \
    --split="val" \
    --bias_type="religion" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion"

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="google/jigsaw_unintended_bias" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="Qwen/Qwen3-4B" \
    --batch_size=1 \
    --max_seq_length=512 \
    --num_examples=20 \
    --output_dir="/scratch/yifwang/fairness_x_explainability/decoder_results_jigsaw/val_results/qwen3_4b_jigsaw_religion_val_20_2/fairness_imagination" \
    --bias_type="religion"\
    --prompt_type="fairness_imagination" \
    --shuffle \
    --seed=2

python -m explanation_generation.gen_explanation_decoder \
    --dataset_name="google/jigsaw_unintended_bias" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="Qwen/Qwen3-4B" \
    --num_labels=2 \
    --batch_size=1 \
    --max_length=512 \
    --num_examples=20 \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion" \
    --output_dir="/scratch/yifwang/fairness_x_explainability/decoder_results_jigsaw/val_results/qwen3_4b_jigsaw_religion_val_20_2/fairness_imagination" \
    --baseline="pad" \
    --bias_type="religion" \
    --prompt_type="fairness_imagination" \
    --only_predicted_class \
    --shuffle \
    --seed=2

python -m sensitive_attribution_extraction.extract_sensitive_token_attribution \
    --results_dir="/scratch/yifwang/fairness_x_explainability/decoder_results_jigsaw/val_results/qwen3_4b_jigsaw_religion_val_20_2/fairness_imagination" \
    --split="val" \
    --bias_type="religion" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion"

python -m model_selection.compute_reliance_statistics \
    --results_dir="/scratch/yifwang/fairness_x_explainability/decoder_results_jigsaw/val_results/qwen3_4b_jigsaw_religion_val_20_2/fairness_imagination" \
    --split="val" \
    --bias_type="religion" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion"

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="google/jigsaw_unintended_bias" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="Qwen/Qwen3-4B" \
    --batch_size=1 \
    --max_seq_length=512 \
    --num_examples=20 \
    --output_dir="/scratch/yifwang/fairness_x_explainability/decoder_results_jigsaw/val_results/qwen3_4b_jigsaw_religion_val_20_2/fairness_instruction" \
    --bias_type="religion"\
    --prompt_type="fairness_instruction" \
    --shuffle \
    --seed=2

python -m explanation_generation.gen_explanation_decoder \
    --dataset_name="google/jigsaw_unintended_bias" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="Qwen/Qwen3-4B" \
    --num_labels=2 \
    --batch_size=1 \
    --max_length=512 \
    --num_examples=20 \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion" \
    --output_dir="/scratch/yifwang/fairness_x_explainability/decoder_results_jigsaw/val_results/qwen3_4b_jigsaw_religion_val_20_2/fairness_instruction" \
    --baseline="pad" \
    --bias_type="religion" \
    --prompt_type="fairness_instruction" \
    --only_predicted_class \
    --shuffle \
    --seed=2

python -m sensitive_attribution_extraction.extract_sensitive_token_attribution \
    --results_dir="/scratch/yifwang/fairness_x_explainability/decoder_results_jigsaw/val_results/qwen3_4b_jigsaw_religion_val_20_2/fairness_instruction" \
    --split="val" \
    --bias_type="religion" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion"

python -m model_selection.compute_reliance_statistics \
    --results_dir="/scratch/yifwang/fairness_x_explainability/decoder_results_jigsaw/val_results/qwen3_4b_jigsaw_religion_val_20_2/fairness_instruction" \
    --split="val" \
    --bias_type="religion" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion"

python -m model_selection.compute_model_selection_correlation \
    --results_dir="/scratch/yifwang/fairness_x_explainability/decoder_results_jigsaw" \
    --val_results_dir="/scratch/yifwang/fairness_x_explainability/decoder_results_jigsaw/val_results" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion" \
    --experiment_type=decoder \
    --bias_type="religion" \
    --model_type="qwen3_4b" \
    --num_test_examples=200 \
    --num_val_examples=20 \
    --test_seed=-1 \
    --val_seed=2

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="google/jigsaw_unintended_bias" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="meta-llama/Llama-3.2-3B-Instruct" \
    --batch_size=1 \
    --max_seq_length=512 \
    --num_examples=20 \
    --output_dir="/scratch/yifwang/fairness_x_explainability/decoder_results_jigsaw/val_results/llama_3b_jigsaw_religion_val_20_3/zero_shot" \
    --bias_type="religion"\
    --prompt_type="zero_shot" \
    --shuffle \
    --seed=3

python -m explanation_generation.gen_explanation_decoder \
    --dataset_name="google/jigsaw_unintended_bias" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="meta-llama/Llama-3.2-3B-Instruct" \
    --num_labels=2 \
    --batch_size=1 \
    --max_length=512 \
    --num_examples=20 \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion" \
    --output_dir="/scratch/yifwang/fairness_x_explainability/decoder_results_jigsaw/val_results/llama_3b_jigsaw_religion_val_20_3/zero_shot" \
    --baseline="pad" \
    --bias_type="religion" \
    --prompt_type="zero_shot" \
    --only_predicted_class \
    --shuffle \
    --seed=3

python -m sensitive_attribution_extraction.extract_sensitive_token_attribution \
    --results_dir="/scratch/yifwang/fairness_x_explainability/decoder_results_jigsaw/val_results/llama_3b_jigsaw_religion_val_20_3/zero_shot" \
    --split="val" \
    --bias_type="religion" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion"

python -m model_selection.compute_reliance_statistics \
    --results_dir="/scratch/yifwang/fairness_x_explainability/decoder_results_jigsaw/val_results/llama_3b_jigsaw_religion_val_20_3/zero_shot" \
    --split="val" \
    --bias_type="religion" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion"

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="google/jigsaw_unintended_bias" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="meta-llama/Llama-3.2-3B-Instruct" \
    --batch_size=1 \
    --max_seq_length=512 \
    --num_examples=20 \
    --output_dir="/scratch/yifwang/fairness_x_explainability/decoder_results_jigsaw/val_results/llama_3b_jigsaw_religion_val_20_3/few_shot" \
    --bias_type="religion"\
    --prompt_type="few_shot" \
    --shuffle \
    --seed=3

python -m explanation_generation.gen_explanation_decoder \
    --dataset_name="google/jigsaw_unintended_bias" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="meta-llama/Llama-3.2-3B-Instruct" \
    --num_labels=2 \
    --batch_size=1 \
    --max_length=512 \
    --num_examples=20 \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion" \
    --output_dir="/scratch/yifwang/fairness_x_explainability/decoder_results_jigsaw/val_results/llama_3b_jigsaw_religion_val_20_3/few_shot" \
    --baseline="pad" \
    --bias_type="religion" \
    --prompt_type="few_shot" \
    --only_predicted_class \
    --shuffle \
    --seed=3

python -m sensitive_attribution_extraction.extract_sensitive_token_attribution \
    --results_dir="/scratch/yifwang/fairness_x_explainability/decoder_results_jigsaw/val_results/llama_3b_jigsaw_religion_val_20_3/few_shot" \
    --split="val" \
    --bias_type="religion" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion"

python -m model_selection.compute_reliance_statistics \
    --results_dir="/scratch/yifwang/fairness_x_explainability/decoder_results_jigsaw/val_results/llama_3b_jigsaw_religion_val_20_3/few_shot" \
    --split="val" \
    --bias_type="religion" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion"

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="google/jigsaw_unintended_bias" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="meta-llama/Llama-3.2-3B-Instruct" \
    --batch_size=1 \
    --max_seq_length=512 \
    --num_examples=20 \
    --output_dir="/scratch/yifwang/fairness_x_explainability/decoder_results_jigsaw/val_results/llama_3b_jigsaw_religion_val_20_3/fairness_imagination" \
    --bias_type="religion"\
    --prompt_type="fairness_imagination" \
    --shuffle \
    --seed=3

python -m explanation_generation.gen_explanation_decoder \
    --dataset_name="google/jigsaw_unintended_bias" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="meta-llama/Llama-3.2-3B-Instruct" \
    --num_labels=2 \
    --batch_size=1 \
    --max_length=512 \
    --num_examples=20 \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion" \
    --output_dir="/scratch/yifwang/fairness_x_explainability/decoder_results_jigsaw/val_results/llama_3b_jigsaw_religion_val_20_3/fairness_imagination" \
    --baseline="pad" \
    --bias_type="religion" \
    --prompt_type="fairness_imagination" \
    --only_predicted_class \
    --shuffle \
    --seed=3

python -m sensitive_attribution_extraction.extract_sensitive_token_attribution \
    --results_dir="/scratch/yifwang/fairness_x_explainability/decoder_results_jigsaw/val_results/llama_3b_jigsaw_religion_val_20_3/fairness_imagination" \
    --split="val" \
    --bias_type="religion" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion"

python -m model_selection.compute_reliance_statistics \
    --results_dir="/scratch/yifwang/fairness_x_explainability/decoder_results_jigsaw/val_results/llama_3b_jigsaw_religion_val_20_3/fairness_imagination" \
    --split="val" \
    --bias_type="religion" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion"

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="google/jigsaw_unintended_bias" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="meta-llama/Llama-3.2-3B-Instruct" \
    --batch_size=1 \
    --max_seq_length=512 \
    --num_examples=20 \
    --output_dir="/scratch/yifwang/fairness_x_explainability/decoder_results_jigsaw/val_results/llama_3b_jigsaw_religion_val_20_3/fairness_instruction" \
    --bias_type="religion"\
    --prompt_type="fairness_instruction" \
    --shuffle \
    --seed=3

python -m explanation_generation.gen_explanation_decoder \
    --dataset_name="google/jigsaw_unintended_bias" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="meta-llama/Llama-3.2-3B-Instruct" \
    --num_labels=2 \
    --batch_size=1 \
    --max_length=512 \
    --num_examples=20 \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion" \
    --output_dir="/scratch/yifwang/fairness_x_explainability/decoder_results_jigsaw/val_results/llama_3b_jigsaw_religion_val_20_3/fairness_instruction" \
    --baseline="pad" \
    --bias_type="religion" \
    --prompt_type="fairness_instruction" \
    --only_predicted_class \
    --shuffle \
    --seed=3

python -m sensitive_attribution_extraction.extract_sensitive_token_attribution \
    --results_dir="/scratch/yifwang/fairness_x_explainability/decoder_results_jigsaw/val_results/llama_3b_jigsaw_religion_val_20_3/fairness_instruction" \
    --split="val" \
    --bias_type="religion" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion"

python -m model_selection.compute_reliance_statistics \
    --results_dir="/scratch/yifwang/fairness_x_explainability/decoder_results_jigsaw/val_results/llama_3b_jigsaw_religion_val_20_3/fairness_instruction" \
    --split="val" \
    --bias_type="religion" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion"

python -m model_selection.compute_model_selection_correlation \
    --results_dir="/scratch/yifwang/fairness_x_explainability/decoder_results_jigsaw" \
    --val_results_dir="/scratch/yifwang/fairness_x_explainability/decoder_results_jigsaw/val_results" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion" \
    --experiment_type=decoder \
    --bias_type="religion" \
    --model_type="llama_3b" \
    --num_test_examples=200 \
    --num_val_examples=20 \
    --test_seed=-1 \
    --val_seed=3

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="google/jigsaw_unintended_bias" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="Qwen/Qwen3-4B" \
    --batch_size=1 \
    --max_seq_length=512 \
    --num_examples=20 \
    --output_dir="/scratch/yifwang/fairness_x_explainability/decoder_results_jigsaw/val_results/qwen3_4b_jigsaw_religion_val_20_3/zero_shot" \
    --bias_type="religion"\
    --prompt_type="zero_shot" \
    --shuffle \
    --seed=3

python -m explanation_generation.gen_explanation_decoder \
    --dataset_name="google/jigsaw_unintended_bias" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="Qwen/Qwen3-4B" \
    --num_labels=2 \
    --batch_size=1 \
    --max_length=512 \
    --num_examples=20 \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion" \
    --output_dir="/scratch/yifwang/fairness_x_explainability/decoder_results_jigsaw/val_results/qwen3_4b_jigsaw_religion_val_20_3/zero_shot" \
    --baseline="pad" \
    --bias_type="religion" \
    --prompt_type="zero_shot" \
    --only_predicted_class \
    --shuffle \
    --seed=3

python -m sensitive_attribution_extraction.extract_sensitive_token_attribution \
    --results_dir="/scratch/yifwang/fairness_x_explainability/decoder_results_jigsaw/val_results/qwen3_4b_jigsaw_religion_val_20_3/zero_shot" \
    --split="val" \
    --bias_type="religion" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion"

python -m model_selection.compute_reliance_statistics \
    --results_dir="/scratch/yifwang/fairness_x_explainability/decoder_results_jigsaw/val_results/qwen3_4b_jigsaw_religion_val_20_3/zero_shot" \
    --split="val" \
    --bias_type="religion" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion"

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="google/jigsaw_unintended_bias" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="Qwen/Qwen3-4B" \
    --batch_size=1 \
    --max_seq_length=512 \
    --num_examples=20 \
    --output_dir="/scratch/yifwang/fairness_x_explainability/decoder_results_jigsaw/val_results/qwen3_4b_jigsaw_religion_val_20_3/few_shot" \
    --bias_type="religion"\
    --prompt_type="few_shot" \
    --shuffle \
    --seed=3

python -m explanation_generation.gen_explanation_decoder \
    --dataset_name="google/jigsaw_unintended_bias" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="Qwen/Qwen3-4B" \
    --num_labels=2 \
    --batch_size=1 \
    --max_length=512 \
    --num_examples=20 \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion" \
    --output_dir="/scratch/yifwang/fairness_x_explainability/decoder_results_jigsaw/val_results/qwen3_4b_jigsaw_religion_val_20_3/few_shot" \
    --baseline="pad" \
    --bias_type="religion" \
    --prompt_type="few_shot" \
    --only_predicted_class \
    --shuffle \
    --seed=3

python -m sensitive_attribution_extraction.extract_sensitive_token_attribution \
    --results_dir="/scratch/yifwang/fairness_x_explainability/decoder_results_jigsaw/val_results/qwen3_4b_jigsaw_religion_val_20_3/few_shot" \
    --split="val" \
    --bias_type="religion" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion"

python -m model_selection.compute_reliance_statistics \
    --results_dir="/scratch/yifwang/fairness_x_explainability/decoder_results_jigsaw/val_results/qwen3_4b_jigsaw_religion_val_20_3/few_shot" \
    --split="val" \
    --bias_type="religion" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion"

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="google/jigsaw_unintended_bias" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="Qwen/Qwen3-4B" \
    --batch_size=1 \
    --max_seq_length=512 \
    --num_examples=20 \
    --output_dir="/scratch/yifwang/fairness_x_explainability/decoder_results_jigsaw/val_results/qwen3_4b_jigsaw_religion_val_20_3/fairness_imagination" \
    --bias_type="religion"\
    --prompt_type="fairness_imagination" \
    --shuffle \
    --seed=3

python -m explanation_generation.gen_explanation_decoder \
    --dataset_name="google/jigsaw_unintended_bias" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="Qwen/Qwen3-4B" \
    --num_labels=2 \
    --batch_size=1 \
    --max_length=512 \
    --num_examples=20 \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion" \
    --output_dir="/scratch/yifwang/fairness_x_explainability/decoder_results_jigsaw/val_results/qwen3_4b_jigsaw_religion_val_20_3/fairness_imagination" \
    --baseline="pad" \
    --bias_type="religion" \
    --prompt_type="fairness_imagination" \
    --only_predicted_class \
    --shuffle \
    --seed=3

python -m sensitive_attribution_extraction.extract_sensitive_token_attribution \
    --results_dir="/scratch/yifwang/fairness_x_explainability/decoder_results_jigsaw/val_results/qwen3_4b_jigsaw_religion_val_20_3/fairness_imagination" \
    --split="val" \
    --bias_type="religion" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion"

python -m model_selection.compute_reliance_statistics \
    --results_dir="/scratch/yifwang/fairness_x_explainability/decoder_results_jigsaw/val_results/qwen3_4b_jigsaw_religion_val_20_3/fairness_imagination" \
    --split="val" \
    --bias_type="religion" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion"

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="google/jigsaw_unintended_bias" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="Qwen/Qwen3-4B" \
    --batch_size=1 \
    --max_seq_length=512 \
    --num_examples=20 \
    --output_dir="/scratch/yifwang/fairness_x_explainability/decoder_results_jigsaw/val_results/qwen3_4b_jigsaw_religion_val_20_3/fairness_instruction" \
    --bias_type="religion"\
    --prompt_type="fairness_instruction" \
    --shuffle \
    --seed=3

python -m explanation_generation.gen_explanation_decoder \
    --dataset_name="google/jigsaw_unintended_bias" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="Qwen/Qwen3-4B" \
    --num_labels=2 \
    --batch_size=1 \
    --max_length=512 \
    --num_examples=20 \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion" \
    --output_dir="/scratch/yifwang/fairness_x_explainability/decoder_results_jigsaw/val_results/qwen3_4b_jigsaw_religion_val_20_3/fairness_instruction" \
    --baseline="pad" \
    --bias_type="religion" \
    --prompt_type="fairness_instruction" \
    --only_predicted_class \
    --shuffle \
    --seed=3

python -m sensitive_attribution_extraction.extract_sensitive_token_attribution \
    --results_dir="/scratch/yifwang/fairness_x_explainability/decoder_results_jigsaw/val_results/qwen3_4b_jigsaw_religion_val_20_3/fairness_instruction" \
    --split="val" \
    --bias_type="religion" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion"

python -m model_selection.compute_reliance_statistics \
    --results_dir="/scratch/yifwang/fairness_x_explainability/decoder_results_jigsaw/val_results/qwen3_4b_jigsaw_religion_val_20_3/fairness_instruction" \
    --split="val" \
    --bias_type="religion" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion"

python -m model_selection.compute_model_selection_correlation \
    --results_dir="/scratch/yifwang/fairness_x_explainability/decoder_results_jigsaw" \
    --val_results_dir="/scratch/yifwang/fairness_x_explainability/decoder_results_jigsaw/val_results" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion" \
    --experiment_type=decoder \
    --bias_type="religion" \
    --model_type="qwen3_4b" \
    --num_test_examples=200 \
    --num_val_examples=20 \
    --test_seed=-1 \
    --val_seed=3

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="google/jigsaw_unintended_bias" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="meta-llama/Llama-3.2-3B-Instruct" \
    --batch_size=1 \
    --max_seq_length=512 \
    --num_examples=20 \
    --output_dir="/scratch/yifwang/fairness_x_explainability/decoder_results_jigsaw/val_results/llama_3b_jigsaw_religion_val_20_4/zero_shot" \
    --bias_type="religion"\
    --prompt_type="zero_shot" \
    --shuffle \
    --seed=4

python -m explanation_generation.gen_explanation_decoder \
    --dataset_name="google/jigsaw_unintended_bias" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="meta-llama/Llama-3.2-3B-Instruct" \
    --num_labels=2 \
    --batch_size=1 \
    --max_length=512 \
    --num_examples=20 \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion" \
    --output_dir="/scratch/yifwang/fairness_x_explainability/decoder_results_jigsaw/val_results/llama_3b_jigsaw_religion_val_20_4/zero_shot" \
    --baseline="pad" \
    --bias_type="religion" \
    --prompt_type="zero_shot" \
    --only_predicted_class \
    --shuffle \
    --seed=4

python -m sensitive_attribution_extraction.extract_sensitive_token_attribution \
    --results_dir="/scratch/yifwang/fairness_x_explainability/decoder_results_jigsaw/val_results/llama_3b_jigsaw_religion_val_20_4/zero_shot" \
    --split="val" \
    --bias_type="religion" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion"

python -m model_selection.compute_reliance_statistics \
    --results_dir="/scratch/yifwang/fairness_x_explainability/decoder_results_jigsaw/val_results/llama_3b_jigsaw_religion_val_20_4/zero_shot" \
    --split="val" \
    --bias_type="religion" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion"

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="google/jigsaw_unintended_bias" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="meta-llama/Llama-3.2-3B-Instruct" \
    --batch_size=1 \
    --max_seq_length=512 \
    --num_examples=20 \
    --output_dir="/scratch/yifwang/fairness_x_explainability/decoder_results_jigsaw/val_results/llama_3b_jigsaw_religion_val_20_4/few_shot" \
    --bias_type="religion"\
    --prompt_type="few_shot" \
    --shuffle \
    --seed=4

python -m explanation_generation.gen_explanation_decoder \
    --dataset_name="google/jigsaw_unintended_bias" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="meta-llama/Llama-3.2-3B-Instruct" \
    --num_labels=2 \
    --batch_size=1 \
    --max_length=512 \
    --num_examples=20 \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion" \
    --output_dir="/scratch/yifwang/fairness_x_explainability/decoder_results_jigsaw/val_results/llama_3b_jigsaw_religion_val_20_4/few_shot" \
    --baseline="pad" \
    --bias_type="religion" \
    --prompt_type="few_shot" \
    --only_predicted_class \
    --shuffle \
    --seed=4

python -m sensitive_attribution_extraction.extract_sensitive_token_attribution \
    --results_dir="/scratch/yifwang/fairness_x_explainability/decoder_results_jigsaw/val_results/llama_3b_jigsaw_religion_val_20_4/few_shot" \
    --split="val" \
    --bias_type="religion" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion"

python -m model_selection.compute_reliance_statistics \
    --results_dir="/scratch/yifwang/fairness_x_explainability/decoder_results_jigsaw/val_results/llama_3b_jigsaw_religion_val_20_4/few_shot" \
    --split="val" \
    --bias_type="religion" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion"

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="google/jigsaw_unintended_bias" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="meta-llama/Llama-3.2-3B-Instruct" \
    --batch_size=1 \
    --max_seq_length=512 \
    --num_examples=20 \
    --output_dir="/scratch/yifwang/fairness_x_explainability/decoder_results_jigsaw/val_results/llama_3b_jigsaw_religion_val_20_4/fairness_imagination" \
    --bias_type="religion"\
    --prompt_type="fairness_imagination" \
    --shuffle \
    --seed=4

python -m explanation_generation.gen_explanation_decoder \
    --dataset_name="google/jigsaw_unintended_bias" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="meta-llama/Llama-3.2-3B-Instruct" \
    --num_labels=2 \
    --batch_size=1 \
    --max_length=512 \
    --num_examples=20 \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion" \
    --output_dir="/scratch/yifwang/fairness_x_explainability/decoder_results_jigsaw/val_results/llama_3b_jigsaw_religion_val_20_4/fairness_imagination" \
    --baseline="pad" \
    --bias_type="religion" \
    --prompt_type="fairness_imagination" \
    --only_predicted_class \
    --shuffle \
    --seed=4

python -m sensitive_attribution_extraction.extract_sensitive_token_attribution \
    --results_dir="/scratch/yifwang/fairness_x_explainability/decoder_results_jigsaw/val_results/llama_3b_jigsaw_religion_val_20_4/fairness_imagination" \
    --split="val" \
    --bias_type="religion" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion"

python -m model_selection.compute_reliance_statistics \
    --results_dir="/scratch/yifwang/fairness_x_explainability/decoder_results_jigsaw/val_results/llama_3b_jigsaw_religion_val_20_4/fairness_imagination" \
    --split="val" \
    --bias_type="religion" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion"

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="google/jigsaw_unintended_bias" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="meta-llama/Llama-3.2-3B-Instruct" \
    --batch_size=1 \
    --max_seq_length=512 \
    --num_examples=20 \
    --output_dir="/scratch/yifwang/fairness_x_explainability/decoder_results_jigsaw/val_results/llama_3b_jigsaw_religion_val_20_4/fairness_instruction" \
    --bias_type="religion"\
    --prompt_type="fairness_instruction" \
    --shuffle \
    --seed=4

python -m explanation_generation.gen_explanation_decoder \
    --dataset_name="google/jigsaw_unintended_bias" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="meta-llama/Llama-3.2-3B-Instruct" \
    --num_labels=2 \
    --batch_size=1 \
    --max_length=512 \
    --num_examples=20 \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion" \
    --output_dir="/scratch/yifwang/fairness_x_explainability/decoder_results_jigsaw/val_results/llama_3b_jigsaw_religion_val_20_4/fairness_instruction" \
    --baseline="pad" \
    --bias_type="religion" \
    --prompt_type="fairness_instruction" \
    --only_predicted_class \
    --shuffle \
    --seed=4

python -m sensitive_attribution_extraction.extract_sensitive_token_attribution \
    --results_dir="/scratch/yifwang/fairness_x_explainability/decoder_results_jigsaw/val_results/llama_3b_jigsaw_religion_val_20_4/fairness_instruction" \
    --split="val" \
    --bias_type="religion" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion"

python -m model_selection.compute_reliance_statistics \
    --results_dir="/scratch/yifwang/fairness_x_explainability/decoder_results_jigsaw/val_results/llama_3b_jigsaw_religion_val_20_4/fairness_instruction" \
    --split="val" \
    --bias_type="religion" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion"

python -m model_selection.compute_model_selection_correlation \
    --results_dir="/scratch/yifwang/fairness_x_explainability/decoder_results_jigsaw" \
    --val_results_dir="/scratch/yifwang/fairness_x_explainability/decoder_results_jigsaw/val_results" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion" \
    --experiment_type=decoder \
    --bias_type="religion" \
    --model_type="llama_3b" \
    --num_test_examples=200 \
    --num_val_examples=20 \
    --test_seed=-1 \
    --val_seed=4

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="google/jigsaw_unintended_bias" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="Qwen/Qwen3-4B" \
    --batch_size=1 \
    --max_seq_length=512 \
    --num_examples=20 \
    --output_dir="/scratch/yifwang/fairness_x_explainability/decoder_results_jigsaw/val_results/qwen3_4b_jigsaw_religion_val_20_4/zero_shot" \
    --bias_type="religion"\
    --prompt_type="zero_shot" \
    --shuffle \
    --seed=4

python -m explanation_generation.gen_explanation_decoder \
    --dataset_name="google/jigsaw_unintended_bias" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="Qwen/Qwen3-4B" \
    --num_labels=2 \
    --batch_size=1 \
    --max_length=512 \
    --num_examples=20 \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion" \
    --output_dir="/scratch/yifwang/fairness_x_explainability/decoder_results_jigsaw/val_results/qwen3_4b_jigsaw_religion_val_20_4/zero_shot" \
    --baseline="pad" \
    --bias_type="religion" \
    --prompt_type="zero_shot" \
    --only_predicted_class \
    --shuffle \
    --seed=4

python -m sensitive_attribution_extraction.extract_sensitive_token_attribution \
    --results_dir="/scratch/yifwang/fairness_x_explainability/decoder_results_jigsaw/val_results/qwen3_4b_jigsaw_religion_val_20_4/zero_shot" \
    --split="val" \
    --bias_type="religion" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion"

python -m model_selection.compute_reliance_statistics \
    --results_dir="/scratch/yifwang/fairness_x_explainability/decoder_results_jigsaw/val_results/qwen3_4b_jigsaw_religion_val_20_4/zero_shot" \
    --split="val" \
    --bias_type="religion" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion"

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="google/jigsaw_unintended_bias" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="Qwen/Qwen3-4B" \
    --batch_size=1 \
    --max_seq_length=512 \
    --num_examples=20 \
    --output_dir="/scratch/yifwang/fairness_x_explainability/decoder_results_jigsaw/val_results/qwen3_4b_jigsaw_religion_val_20_4/few_shot" \
    --bias_type="religion"\
    --prompt_type="few_shot" \
    --shuffle \
    --seed=4

python -m explanation_generation.gen_explanation_decoder \
    --dataset_name="google/jigsaw_unintended_bias" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="Qwen/Qwen3-4B" \
    --num_labels=2 \
    --batch_size=1 \
    --max_length=512 \
    --num_examples=20 \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion" \
    --output_dir="/scratch/yifwang/fairness_x_explainability/decoder_results_jigsaw/val_results/qwen3_4b_jigsaw_religion_val_20_4/few_shot" \
    --baseline="pad" \
    --bias_type="religion" \
    --prompt_type="few_shot" \
    --only_predicted_class \
    --shuffle \
    --seed=4

python -m sensitive_attribution_extraction.extract_sensitive_token_attribution \
    --results_dir="/scratch/yifwang/fairness_x_explainability/decoder_results_jigsaw/val_results/qwen3_4b_jigsaw_religion_val_20_4/few_shot" \
    --split="val" \
    --bias_type="religion" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion"

python -m model_selection.compute_reliance_statistics \
    --results_dir="/scratch/yifwang/fairness_x_explainability/decoder_results_jigsaw/val_results/qwen3_4b_jigsaw_religion_val_20_4/few_shot" \
    --split="val" \
    --bias_type="religion" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion"

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="google/jigsaw_unintended_bias" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="Qwen/Qwen3-4B" \
    --batch_size=1 \
    --max_seq_length=512 \
    --num_examples=20 \
    --output_dir="/scratch/yifwang/fairness_x_explainability/decoder_results_jigsaw/val_results/qwen3_4b_jigsaw_religion_val_20_4/fairness_imagination" \
    --bias_type="religion"\
    --prompt_type="fairness_imagination" \
    --shuffle \
    --seed=4

python -m explanation_generation.gen_explanation_decoder \
    --dataset_name="google/jigsaw_unintended_bias" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="Qwen/Qwen3-4B" \
    --num_labels=2 \
    --batch_size=1 \
    --max_length=512 \
    --num_examples=20 \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion" \
    --output_dir="/scratch/yifwang/fairness_x_explainability/decoder_results_jigsaw/val_results/qwen3_4b_jigsaw_religion_val_20_4/fairness_imagination" \
    --baseline="pad" \
    --bias_type="religion" \
    --prompt_type="fairness_imagination" \
    --only_predicted_class \
    --shuffle \
    --seed=4

python -m sensitive_attribution_extraction.extract_sensitive_token_attribution \
    --results_dir="/scratch/yifwang/fairness_x_explainability/decoder_results_jigsaw/val_results/qwen3_4b_jigsaw_religion_val_20_4/fairness_imagination" \
    --split="val" \
    --bias_type="religion" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion"

python -m model_selection.compute_reliance_statistics \
    --results_dir="/scratch/yifwang/fairness_x_explainability/decoder_results_jigsaw/val_results/qwen3_4b_jigsaw_religion_val_20_4/fairness_imagination" \
    --split="val" \
    --bias_type="religion" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion"

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="google/jigsaw_unintended_bias" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="Qwen/Qwen3-4B" \
    --batch_size=1 \
    --max_seq_length=512 \
    --num_examples=20 \
    --output_dir="/scratch/yifwang/fairness_x_explainability/decoder_results_jigsaw/val_results/qwen3_4b_jigsaw_religion_val_20_4/fairness_instruction" \
    --bias_type="religion"\
    --prompt_type="fairness_instruction" \
    --shuffle \
    --seed=4

python -m explanation_generation.gen_explanation_decoder \
    --dataset_name="google/jigsaw_unintended_bias" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="Qwen/Qwen3-4B" \
    --num_labels=2 \
    --batch_size=1 \
    --max_length=512 \
    --num_examples=20 \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion" \
    --output_dir="/scratch/yifwang/fairness_x_explainability/decoder_results_jigsaw/val_results/qwen3_4b_jigsaw_religion_val_20_4/fairness_instruction" \
    --baseline="pad" \
    --bias_type="religion" \
    --prompt_type="fairness_instruction" \
    --only_predicted_class \
    --shuffle \
    --seed=4

python -m sensitive_attribution_extraction.extract_sensitive_token_attribution \
    --results_dir="/scratch/yifwang/fairness_x_explainability/decoder_results_jigsaw/val_results/qwen3_4b_jigsaw_religion_val_20_4/fairness_instruction" \
    --split="val" \
    --bias_type="religion" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion"

python -m model_selection.compute_reliance_statistics \
    --results_dir="/scratch/yifwang/fairness_x_explainability/decoder_results_jigsaw/val_results/qwen3_4b_jigsaw_religion_val_20_4/fairness_instruction" \
    --split="val" \
    --bias_type="religion" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion"

python -m model_selection.compute_model_selection_correlation \
    --results_dir="/scratch/yifwang/fairness_x_explainability/decoder_results_jigsaw" \
    --val_results_dir="/scratch/yifwang/fairness_x_explainability/decoder_results_jigsaw/val_results" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion" \
    --experiment_type=decoder \
    --bias_type="religion" \
    --model_type="qwen3_4b" \
    --num_test_examples=200 \
    --num_val_examples=20 \
    --test_seed=-1 \
    --val_seed=4

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="google/jigsaw_unintended_bias" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="meta-llama/Llama-3.2-3B-Instruct" \
    --batch_size=1 \
    --max_seq_length=512 \
    --num_examples=20 \
    --output_dir="/scratch/yifwang/fairness_x_explainability/decoder_results_jigsaw/val_results/llama_3b_jigsaw_religion_val_20_5/zero_shot" \
    --bias_type="religion"\
    --prompt_type="zero_shot" \
    --shuffle \
    --seed=5

python -m explanation_generation.gen_explanation_decoder \
    --dataset_name="google/jigsaw_unintended_bias" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="meta-llama/Llama-3.2-3B-Instruct" \
    --num_labels=2 \
    --batch_size=1 \
    --max_length=512 \
    --num_examples=20 \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion" \
    --output_dir="/scratch/yifwang/fairness_x_explainability/decoder_results_jigsaw/val_results/llama_3b_jigsaw_religion_val_20_5/zero_shot" \
    --baseline="pad" \
    --bias_type="religion" \
    --prompt_type="zero_shot" \
    --only_predicted_class \
    --shuffle \
    --seed=5

python -m sensitive_attribution_extraction.extract_sensitive_token_attribution \
    --results_dir="/scratch/yifwang/fairness_x_explainability/decoder_results_jigsaw/val_results/llama_3b_jigsaw_religion_val_20_5/zero_shot" \
    --split="val" \
    --bias_type="religion" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion"

python -m model_selection.compute_reliance_statistics \
    --results_dir="/scratch/yifwang/fairness_x_explainability/decoder_results_jigsaw/val_results/llama_3b_jigsaw_religion_val_20_5/zero_shot" \
    --split="val" \
    --bias_type="religion" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion"

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="google/jigsaw_unintended_bias" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="meta-llama/Llama-3.2-3B-Instruct" \
    --batch_size=1 \
    --max_seq_length=512 \
    --num_examples=20 \
    --output_dir="/scratch/yifwang/fairness_x_explainability/decoder_results_jigsaw/val_results/llama_3b_jigsaw_religion_val_20_5/few_shot" \
    --bias_type="religion"\
    --prompt_type="few_shot" \
    --shuffle \
    --seed=5

python -m explanation_generation.gen_explanation_decoder \
    --dataset_name="google/jigsaw_unintended_bias" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="meta-llama/Llama-3.2-3B-Instruct" \
    --num_labels=2 \
    --batch_size=1 \
    --max_length=512 \
    --num_examples=20 \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion" \
    --output_dir="/scratch/yifwang/fairness_x_explainability/decoder_results_jigsaw/val_results/llama_3b_jigsaw_religion_val_20_5/few_shot" \
    --baseline="pad" \
    --bias_type="religion" \
    --prompt_type="few_shot" \
    --only_predicted_class \
    --shuffle \
    --seed=5

python -m sensitive_attribution_extraction.extract_sensitive_token_attribution \
    --results_dir="/scratch/yifwang/fairness_x_explainability/decoder_results_jigsaw/val_results/llama_3b_jigsaw_religion_val_20_5/few_shot" \
    --split="val" \
    --bias_type="religion" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion"

python -m model_selection.compute_reliance_statistics \
    --results_dir="/scratch/yifwang/fairness_x_explainability/decoder_results_jigsaw/val_results/llama_3b_jigsaw_religion_val_20_5/few_shot" \
    --split="val" \
    --bias_type="religion" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion"

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="google/jigsaw_unintended_bias" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="meta-llama/Llama-3.2-3B-Instruct" \
    --batch_size=1 \
    --max_seq_length=512 \
    --num_examples=20 \
    --output_dir="/scratch/yifwang/fairness_x_explainability/decoder_results_jigsaw/val_results/llama_3b_jigsaw_religion_val_20_5/fairness_imagination" \
    --bias_type="religion"\
    --prompt_type="fairness_imagination" \
    --shuffle \
    --seed=5

python -m explanation_generation.gen_explanation_decoder \
    --dataset_name="google/jigsaw_unintended_bias" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="meta-llama/Llama-3.2-3B-Instruct" \
    --num_labels=2 \
    --batch_size=1 \
    --max_length=512 \
    --num_examples=20 \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion" \
    --output_dir="/scratch/yifwang/fairness_x_explainability/decoder_results_jigsaw/val_results/llama_3b_jigsaw_religion_val_20_5/fairness_imagination" \
    --baseline="pad" \
    --bias_type="religion" \
    --prompt_type="fairness_imagination" \
    --only_predicted_class \
    --shuffle \
    --seed=5

python -m sensitive_attribution_extraction.extract_sensitive_token_attribution \
    --results_dir="/scratch/yifwang/fairness_x_explainability/decoder_results_jigsaw/val_results/llama_3b_jigsaw_religion_val_20_5/fairness_imagination" \
    --split="val" \
    --bias_type="religion" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion"

python -m model_selection.compute_reliance_statistics \
    --results_dir="/scratch/yifwang/fairness_x_explainability/decoder_results_jigsaw/val_results/llama_3b_jigsaw_religion_val_20_5/fairness_imagination" \
    --split="val" \
    --bias_type="religion" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion"

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="google/jigsaw_unintended_bias" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="meta-llama/Llama-3.2-3B-Instruct" \
    --batch_size=1 \
    --max_seq_length=512 \
    --num_examples=20 \
    --output_dir="/scratch/yifwang/fairness_x_explainability/decoder_results_jigsaw/val_results/llama_3b_jigsaw_religion_val_20_5/fairness_instruction" \
    --bias_type="religion"\
    --prompt_type="fairness_instruction" \
    --shuffle \
    --seed=5

python -m explanation_generation.gen_explanation_decoder \
    --dataset_name="google/jigsaw_unintended_bias" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="meta-llama/Llama-3.2-3B-Instruct" \
    --num_labels=2 \
    --batch_size=1 \
    --max_length=512 \
    --num_examples=20 \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion" \
    --output_dir="/scratch/yifwang/fairness_x_explainability/decoder_results_jigsaw/val_results/llama_3b_jigsaw_religion_val_20_5/fairness_instruction" \
    --baseline="pad" \
    --bias_type="religion" \
    --prompt_type="fairness_instruction" \
    --only_predicted_class \
    --shuffle \
    --seed=5

python -m sensitive_attribution_extraction.extract_sensitive_token_attribution \
    --results_dir="/scratch/yifwang/fairness_x_explainability/decoder_results_jigsaw/val_results/llama_3b_jigsaw_religion_val_20_5/fairness_instruction" \
    --split="val" \
    --bias_type="religion" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion"

python -m model_selection.compute_reliance_statistics \
    --results_dir="/scratch/yifwang/fairness_x_explainability/decoder_results_jigsaw/val_results/llama_3b_jigsaw_religion_val_20_5/fairness_instruction" \
    --split="val" \
    --bias_type="religion" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion"

python -m model_selection.compute_model_selection_correlation \
    --results_dir="/scratch/yifwang/fairness_x_explainability/decoder_results_jigsaw" \
    --val_results_dir="/scratch/yifwang/fairness_x_explainability/decoder_results_jigsaw/val_results" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion" \
    --experiment_type=decoder \
    --bias_type="religion" \
    --model_type="llama_3b" \
    --num_test_examples=200 \
    --num_val_examples=20 \
    --test_seed=-1 \
    --val_seed=5

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="google/jigsaw_unintended_bias" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="Qwen/Qwen3-4B" \
    --batch_size=1 \
    --max_seq_length=512 \
    --num_examples=20 \
    --output_dir="/scratch/yifwang/fairness_x_explainability/decoder_results_jigsaw/val_results/qwen3_4b_jigsaw_religion_val_20_5/zero_shot" \
    --bias_type="religion"\
    --prompt_type="zero_shot" \
    --shuffle \
    --seed=5

python -m explanation_generation.gen_explanation_decoder \
    --dataset_name="google/jigsaw_unintended_bias" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="Qwen/Qwen3-4B" \
    --num_labels=2 \
    --batch_size=1 \
    --max_length=512 \
    --num_examples=20 \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion" \
    --output_dir="/scratch/yifwang/fairness_x_explainability/decoder_results_jigsaw/val_results/qwen3_4b_jigsaw_religion_val_20_5/zero_shot" \
    --baseline="pad" \
    --bias_type="religion" \
    --prompt_type="zero_shot" \
    --only_predicted_class \
    --shuffle \
    --seed=5

python -m sensitive_attribution_extraction.extract_sensitive_token_attribution \
    --results_dir="/scratch/yifwang/fairness_x_explainability/decoder_results_jigsaw/val_results/qwen3_4b_jigsaw_religion_val_20_5/zero_shot" \
    --split="val" \
    --bias_type="religion" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion"

python -m model_selection.compute_reliance_statistics \
    --results_dir="/scratch/yifwang/fairness_x_explainability/decoder_results_jigsaw/val_results/qwen3_4b_jigsaw_religion_val_20_5/zero_shot" \
    --split="val" \
    --bias_type="religion" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion"

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="google/jigsaw_unintended_bias" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="Qwen/Qwen3-4B" \
    --batch_size=1 \
    --max_seq_length=512 \
    --num_examples=20 \
    --output_dir="/scratch/yifwang/fairness_x_explainability/decoder_results_jigsaw/val_results/qwen3_4b_jigsaw_religion_val_20_5/few_shot" \
    --bias_type="religion"\
    --prompt_type="few_shot" \
    --shuffle \
    --seed=5

python -m explanation_generation.gen_explanation_decoder \
    --dataset_name="google/jigsaw_unintended_bias" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="Qwen/Qwen3-4B" \
    --num_labels=2 \
    --batch_size=1 \
    --max_length=512 \
    --num_examples=20 \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion" \
    --output_dir="/scratch/yifwang/fairness_x_explainability/decoder_results_jigsaw/val_results/qwen3_4b_jigsaw_religion_val_20_5/few_shot" \
    --baseline="pad" \
    --bias_type="religion" \
    --prompt_type="few_shot" \
    --only_predicted_class \
    --shuffle \
    --seed=5

python -m sensitive_attribution_extraction.extract_sensitive_token_attribution \
    --results_dir="/scratch/yifwang/fairness_x_explainability/decoder_results_jigsaw/val_results/qwen3_4b_jigsaw_religion_val_20_5/few_shot" \
    --split="val" \
    --bias_type="religion" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion"

python -m model_selection.compute_reliance_statistics \
    --results_dir="/scratch/yifwang/fairness_x_explainability/decoder_results_jigsaw/val_results/qwen3_4b_jigsaw_religion_val_20_5/few_shot" \
    --split="val" \
    --bias_type="religion" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion"

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="google/jigsaw_unintended_bias" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="Qwen/Qwen3-4B" \
    --batch_size=1 \
    --max_seq_length=512 \
    --num_examples=20 \
    --output_dir="/scratch/yifwang/fairness_x_explainability/decoder_results_jigsaw/val_results/qwen3_4b_jigsaw_religion_val_20_5/fairness_imagination" \
    --bias_type="religion"\
    --prompt_type="fairness_imagination" \
    --shuffle \
    --seed=5

python -m explanation_generation.gen_explanation_decoder \
    --dataset_name="google/jigsaw_unintended_bias" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="Qwen/Qwen3-4B" \
    --num_labels=2 \
    --batch_size=1 \
    --max_length=512 \
    --num_examples=20 \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion" \
    --output_dir="/scratch/yifwang/fairness_x_explainability/decoder_results_jigsaw/val_results/qwen3_4b_jigsaw_religion_val_20_5/fairness_imagination" \
    --baseline="pad" \
    --bias_type="religion" \
    --prompt_type="fairness_imagination" \
    --only_predicted_class \
    --shuffle \
    --seed=5

python -m sensitive_attribution_extraction.extract_sensitive_token_attribution \
    --results_dir="/scratch/yifwang/fairness_x_explainability/decoder_results_jigsaw/val_results/qwen3_4b_jigsaw_religion_val_20_5/fairness_imagination" \
    --split="val" \
    --bias_type="religion" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion"

python -m model_selection.compute_reliance_statistics \
    --results_dir="/scratch/yifwang/fairness_x_explainability/decoder_results_jigsaw/val_results/qwen3_4b_jigsaw_religion_val_20_5/fairness_imagination" \
    --split="val" \
    --bias_type="religion" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion"

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="google/jigsaw_unintended_bias" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="Qwen/Qwen3-4B" \
    --batch_size=1 \
    --max_seq_length=512 \
    --num_examples=20 \
    --output_dir="/scratch/yifwang/fairness_x_explainability/decoder_results_jigsaw/val_results/qwen3_4b_jigsaw_religion_val_20_5/fairness_instruction" \
    --bias_type="religion"\
    --prompt_type="fairness_instruction" \
    --shuffle \
    --seed=5

python -m explanation_generation.gen_explanation_decoder \
    --dataset_name="google/jigsaw_unintended_bias" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="Qwen/Qwen3-4B" \
    --num_labels=2 \
    --batch_size=1 \
    --max_length=512 \
    --num_examples=20 \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion" \
    --output_dir="/scratch/yifwang/fairness_x_explainability/decoder_results_jigsaw/val_results/qwen3_4b_jigsaw_religion_val_20_5/fairness_instruction" \
    --baseline="pad" \
    --bias_type="religion" \
    --prompt_type="fairness_instruction" \
    --only_predicted_class \
    --shuffle \
    --seed=5

python -m sensitive_attribution_extraction.extract_sensitive_token_attribution \
    --results_dir="/scratch/yifwang/fairness_x_explainability/decoder_results_jigsaw/val_results/qwen3_4b_jigsaw_religion_val_20_5/fairness_instruction" \
    --split="val" \
    --bias_type="religion" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion"

python -m model_selection.compute_reliance_statistics \
    --results_dir="/scratch/yifwang/fairness_x_explainability/decoder_results_jigsaw/val_results/qwen3_4b_jigsaw_religion_val_20_5/fairness_instruction" \
    --split="val" \
    --bias_type="religion" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion"

python -m model_selection.compute_model_selection_correlation \
    --results_dir="/scratch/yifwang/fairness_x_explainability/decoder_results_jigsaw" \
    --val_results_dir="/scratch/yifwang/fairness_x_explainability/decoder_results_jigsaw/val_results" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion" \
    --experiment_type=decoder \
    --bias_type="religion" \
    --model_type="qwen3_4b" \
    --num_test_examples=200 \
    --num_val_examples=20 \
    --test_seed=-1 \
    --val_seed=5

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="google/jigsaw_unintended_bias" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="meta-llama/Llama-3.2-3B-Instruct" \
    --batch_size=1 \
    --max_seq_length=512 \
    --num_examples=20 \
    --output_dir="/scratch/yifwang/fairness_x_explainability/decoder_results_jigsaw/val_results/llama_3b_jigsaw_religion_val_20_42/zero_shot" \
    --bias_type="religion"\
    --prompt_type="zero_shot" \
    --shuffle \
    --seed=42

python -m explanation_generation.gen_explanation_decoder \
    --dataset_name="google/jigsaw_unintended_bias" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="meta-llama/Llama-3.2-3B-Instruct" \
    --num_labels=2 \
    --batch_size=1 \
    --max_length=512 \
    --num_examples=20 \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion" \
    --output_dir="/scratch/yifwang/fairness_x_explainability/decoder_results_jigsaw/val_results/llama_3b_jigsaw_religion_val_20_42/zero_shot" \
    --baseline="pad" \
    --bias_type="religion" \
    --prompt_type="zero_shot" \
    --only_predicted_class \
    --shuffle \
    --seed=42

python -m sensitive_attribution_extraction.extract_sensitive_token_attribution \
    --results_dir="/scratch/yifwang/fairness_x_explainability/decoder_results_jigsaw/val_results/llama_3b_jigsaw_religion_val_20_42/zero_shot" \
    --split="val" \
    --bias_type="religion" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion"

python -m model_selection.compute_reliance_statistics \
    --results_dir="/scratch/yifwang/fairness_x_explainability/decoder_results_jigsaw/val_results/llama_3b_jigsaw_religion_val_20_42/zero_shot" \
    --split="val" \
    --bias_type="religion" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion"

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="google/jigsaw_unintended_bias" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="meta-llama/Llama-3.2-3B-Instruct" \
    --batch_size=1 \
    --max_seq_length=512 \
    --num_examples=20 \
    --output_dir="/scratch/yifwang/fairness_x_explainability/decoder_results_jigsaw/val_results/llama_3b_jigsaw_religion_val_20_42/few_shot" \
    --bias_type="religion"\
    --prompt_type="few_shot" \
    --shuffle \
    --seed=42

python -m explanation_generation.gen_explanation_decoder \
    --dataset_name="google/jigsaw_unintended_bias" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="meta-llama/Llama-3.2-3B-Instruct" \
    --num_labels=2 \
    --batch_size=1 \
    --max_length=512 \
    --num_examples=20 \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion" \
    --output_dir="/scratch/yifwang/fairness_x_explainability/decoder_results_jigsaw/val_results/llama_3b_jigsaw_religion_val_20_42/few_shot" \
    --baseline="pad" \
    --bias_type="religion" \
    --prompt_type="few_shot" \
    --only_predicted_class \
    --shuffle \
    --seed=42

python -m sensitive_attribution_extraction.extract_sensitive_token_attribution \
    --results_dir="/scratch/yifwang/fairness_x_explainability/decoder_results_jigsaw/val_results/llama_3b_jigsaw_religion_val_20_42/few_shot" \
    --split="val" \
    --bias_type="religion" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion"

python -m model_selection.compute_reliance_statistics \
    --results_dir="/scratch/yifwang/fairness_x_explainability/decoder_results_jigsaw/val_results/llama_3b_jigsaw_religion_val_20_42/few_shot" \
    --split="val" \
    --bias_type="religion" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion"

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="google/jigsaw_unintended_bias" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="meta-llama/Llama-3.2-3B-Instruct" \
    --batch_size=1 \
    --max_seq_length=512 \
    --num_examples=20 \
    --output_dir="/scratch/yifwang/fairness_x_explainability/decoder_results_jigsaw/val_results/llama_3b_jigsaw_religion_val_20_42/fairness_imagination" \
    --bias_type="religion"\
    --prompt_type="fairness_imagination" \
    --shuffle \
    --seed=42

python -m explanation_generation.gen_explanation_decoder \
    --dataset_name="google/jigsaw_unintended_bias" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="meta-llama/Llama-3.2-3B-Instruct" \
    --num_labels=2 \
    --batch_size=1 \
    --max_length=512 \
    --num_examples=20 \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion" \
    --output_dir="/scratch/yifwang/fairness_x_explainability/decoder_results_jigsaw/val_results/llama_3b_jigsaw_religion_val_20_42/fairness_imagination" \
    --baseline="pad" \
    --bias_type="religion" \
    --prompt_type="fairness_imagination" \
    --only_predicted_class \
    --shuffle \
    --seed=42

python -m sensitive_attribution_extraction.extract_sensitive_token_attribution \
    --results_dir="/scratch/yifwang/fairness_x_explainability/decoder_results_jigsaw/val_results/llama_3b_jigsaw_religion_val_20_42/fairness_imagination" \
    --split="val" \
    --bias_type="religion" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion"

python -m model_selection.compute_reliance_statistics \
    --results_dir="/scratch/yifwang/fairness_x_explainability/decoder_results_jigsaw/val_results/llama_3b_jigsaw_religion_val_20_42/fairness_imagination" \
    --split="val" \
    --bias_type="religion" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion"

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="google/jigsaw_unintended_bias" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="meta-llama/Llama-3.2-3B-Instruct" \
    --batch_size=1 \
    --max_seq_length=512 \
    --num_examples=20 \
    --output_dir="/scratch/yifwang/fairness_x_explainability/decoder_results_jigsaw/val_results/llama_3b_jigsaw_religion_val_20_42/fairness_instruction" \
    --bias_type="religion"\
    --prompt_type="fairness_instruction" \
    --shuffle \
    --seed=42

python -m explanation_generation.gen_explanation_decoder \
    --dataset_name="google/jigsaw_unintended_bias" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="meta-llama/Llama-3.2-3B-Instruct" \
    --num_labels=2 \
    --batch_size=1 \
    --max_length=512 \
    --num_examples=20 \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion" \
    --output_dir="/scratch/yifwang/fairness_x_explainability/decoder_results_jigsaw/val_results/llama_3b_jigsaw_religion_val_20_42/fairness_instruction" \
    --baseline="pad" \
    --bias_type="religion" \
    --prompt_type="fairness_instruction" \
    --only_predicted_class \
    --shuffle \
    --seed=42

python -m sensitive_attribution_extraction.extract_sensitive_token_attribution \
    --results_dir="/scratch/yifwang/fairness_x_explainability/decoder_results_jigsaw/val_results/llama_3b_jigsaw_religion_val_20_42/fairness_instruction" \
    --split="val" \
    --bias_type="religion" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion"

python -m model_selection.compute_reliance_statistics \
    --results_dir="/scratch/yifwang/fairness_x_explainability/decoder_results_jigsaw/val_results/llama_3b_jigsaw_religion_val_20_42/fairness_instruction" \
    --split="val" \
    --bias_type="religion" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion"

python -m model_selection.compute_model_selection_correlation \
    --results_dir="/scratch/yifwang/fairness_x_explainability/decoder_results_jigsaw" \
    --val_results_dir="/scratch/yifwang/fairness_x_explainability/decoder_results_jigsaw/val_results" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion" \
    --experiment_type=decoder \
    --bias_type="religion" \
    --model_type="llama_3b" \
    --num_test_examples=200 \
    --num_val_examples=20 \
    --test_seed=-1 \
    --val_seed=42

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="google/jigsaw_unintended_bias" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="Qwen/Qwen3-4B" \
    --batch_size=1 \
    --max_seq_length=512 \
    --num_examples=20 \
    --output_dir="/scratch/yifwang/fairness_x_explainability/decoder_results_jigsaw/val_results/qwen3_4b_jigsaw_religion_val_20_42/zero_shot" \
    --bias_type="religion"\
    --prompt_type="zero_shot" \
    --shuffle \
    --seed=42

python -m explanation_generation.gen_explanation_decoder \
    --dataset_name="google/jigsaw_unintended_bias" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="Qwen/Qwen3-4B" \
    --num_labels=2 \
    --batch_size=1 \
    --max_length=512 \
    --num_examples=20 \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion" \
    --output_dir="/scratch/yifwang/fairness_x_explainability/decoder_results_jigsaw/val_results/qwen3_4b_jigsaw_religion_val_20_42/zero_shot" \
    --baseline="pad" \
    --bias_type="religion" \
    --prompt_type="zero_shot" \
    --only_predicted_class \
    --shuffle \
    --seed=42

python -m sensitive_attribution_extraction.extract_sensitive_token_attribution \
    --results_dir="/scratch/yifwang/fairness_x_explainability/decoder_results_jigsaw/val_results/qwen3_4b_jigsaw_religion_val_20_42/zero_shot" \
    --split="val" \
    --bias_type="religion" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion"

python -m model_selection.compute_reliance_statistics \
    --results_dir="/scratch/yifwang/fairness_x_explainability/decoder_results_jigsaw/val_results/qwen3_4b_jigsaw_religion_val_20_42/zero_shot" \
    --split="val" \
    --bias_type="religion" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion"

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="google/jigsaw_unintended_bias" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="Qwen/Qwen3-4B" \
    --batch_size=1 \
    --max_seq_length=512 \
    --num_examples=20 \
    --output_dir="/scratch/yifwang/fairness_x_explainability/decoder_results_jigsaw/val_results/qwen3_4b_jigsaw_religion_val_20_42/few_shot" \
    --bias_type="religion"\
    --prompt_type="few_shot" \
    --shuffle \
    --seed=42

python -m explanation_generation.gen_explanation_decoder \
    --dataset_name="google/jigsaw_unintended_bias" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="Qwen/Qwen3-4B" \
    --num_labels=2 \
    --batch_size=1 \
    --max_length=512 \
    --num_examples=20 \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion" \
    --output_dir="/scratch/yifwang/fairness_x_explainability/decoder_results_jigsaw/val_results/qwen3_4b_jigsaw_religion_val_20_42/few_shot" \
    --baseline="pad" \
    --bias_type="religion" \
    --prompt_type="few_shot" \
    --only_predicted_class \
    --shuffle \
    --seed=42

python -m sensitive_attribution_extraction.extract_sensitive_token_attribution \
    --results_dir="/scratch/yifwang/fairness_x_explainability/decoder_results_jigsaw/val_results/qwen3_4b_jigsaw_religion_val_20_42/few_shot" \
    --split="val" \
    --bias_type="religion" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion"

python -m model_selection.compute_reliance_statistics \
    --results_dir="/scratch/yifwang/fairness_x_explainability/decoder_results_jigsaw/val_results/qwen3_4b_jigsaw_religion_val_20_42/few_shot" \
    --split="val" \
    --bias_type="religion" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion"

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="google/jigsaw_unintended_bias" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="Qwen/Qwen3-4B" \
    --batch_size=1 \
    --max_seq_length=512 \
    --num_examples=20 \
    --output_dir="/scratch/yifwang/fairness_x_explainability/decoder_results_jigsaw/val_results/qwen3_4b_jigsaw_religion_val_20_42/fairness_imagination" \
    --bias_type="religion"\
    --prompt_type="fairness_imagination" \
    --shuffle \
    --seed=42

python -m explanation_generation.gen_explanation_decoder \
    --dataset_name="google/jigsaw_unintended_bias" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="Qwen/Qwen3-4B" \
    --num_labels=2 \
    --batch_size=1 \
    --max_length=512 \
    --num_examples=20 \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion" \
    --output_dir="/scratch/yifwang/fairness_x_explainability/decoder_results_jigsaw/val_results/qwen3_4b_jigsaw_religion_val_20_42/fairness_imagination" \
    --baseline="pad" \
    --bias_type="religion" \
    --prompt_type="fairness_imagination" \
    --only_predicted_class \
    --shuffle \
    --seed=42

python -m sensitive_attribution_extraction.extract_sensitive_token_attribution \
    --results_dir="/scratch/yifwang/fairness_x_explainability/decoder_results_jigsaw/val_results/qwen3_4b_jigsaw_religion_val_20_42/fairness_imagination" \
    --split="val" \
    --bias_type="religion" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion"

python -m model_selection.compute_reliance_statistics \
    --results_dir="/scratch/yifwang/fairness_x_explainability/decoder_results_jigsaw/val_results/qwen3_4b_jigsaw_religion_val_20_42/fairness_imagination" \
    --split="val" \
    --bias_type="religion" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion"

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="google/jigsaw_unintended_bias" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="Qwen/Qwen3-4B" \
    --batch_size=1 \
    --max_seq_length=512 \
    --num_examples=20 \
    --output_dir="/scratch/yifwang/fairness_x_explainability/decoder_results_jigsaw/val_results/qwen3_4b_jigsaw_religion_val_20_42/fairness_instruction" \
    --bias_type="religion"\
    --prompt_type="fairness_instruction" \
    --shuffle \
    --seed=42

python -m explanation_generation.gen_explanation_decoder \
    --dataset_name="google/jigsaw_unintended_bias" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="Qwen/Qwen3-4B" \
    --num_labels=2 \
    --batch_size=1 \
    --max_length=512 \
    --num_examples=20 \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion" \
    --output_dir="/scratch/yifwang/fairness_x_explainability/decoder_results_jigsaw/val_results/qwen3_4b_jigsaw_religion_val_20_42/fairness_instruction" \
    --baseline="pad" \
    --bias_type="religion" \
    --prompt_type="fairness_instruction" \
    --only_predicted_class \
    --shuffle \
    --seed=42

python -m sensitive_attribution_extraction.extract_sensitive_token_attribution \
    --results_dir="/scratch/yifwang/fairness_x_explainability/decoder_results_jigsaw/val_results/qwen3_4b_jigsaw_religion_val_20_42/fairness_instruction" \
    --split="val" \
    --bias_type="religion" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion"

python -m model_selection.compute_reliance_statistics \
    --results_dir="/scratch/yifwang/fairness_x_explainability/decoder_results_jigsaw/val_results/qwen3_4b_jigsaw_religion_val_20_42/fairness_instruction" \
    --split="val" \
    --bias_type="religion" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion"

python -m model_selection.compute_model_selection_correlation \
    --results_dir="/scratch/yifwang/fairness_x_explainability/decoder_results_jigsaw" \
    --val_results_dir="/scratch/yifwang/fairness_x_explainability/decoder_results_jigsaw/val_results" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion" \
    --experiment_type=decoder \
    --bias_type="religion" \
    --model_type="qwen3_4b" \
    --num_test_examples=200 \
    --num_val_examples=20 \
    --test_seed=-1 \
    --val_seed=42

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="google/jigsaw_unintended_bias" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="meta-llama/Llama-3.2-3B-Instruct" \
    --batch_size=1 \
    --max_seq_length=512 \
    --num_examples=50 \
    --output_dir="/scratch/yifwang/fairness_x_explainability/decoder_results_jigsaw/val_results/llama_3b_jigsaw_religion_val_50_1/zero_shot" \
    --bias_type="religion"\
    --prompt_type="zero_shot" \
    --shuffle \
    --seed=1

python -m explanation_generation.gen_explanation_decoder \
    --dataset_name="google/jigsaw_unintended_bias" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="meta-llama/Llama-3.2-3B-Instruct" \
    --num_labels=2 \
    --batch_size=1 \
    --max_length=512 \
    --num_examples=50 \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion" \
    --output_dir="/scratch/yifwang/fairness_x_explainability/decoder_results_jigsaw/val_results/llama_3b_jigsaw_religion_val_50_1/zero_shot" \
    --baseline="pad" \
    --bias_type="religion" \
    --prompt_type="zero_shot" \
    --only_predicted_class \
    --shuffle \
    --seed=1

python -m sensitive_attribution_extraction.extract_sensitive_token_attribution \
    --results_dir="/scratch/yifwang/fairness_x_explainability/decoder_results_jigsaw/val_results/llama_3b_jigsaw_religion_val_50_1/zero_shot" \
    --split="val" \
    --bias_type="religion" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion"

python -m model_selection.compute_reliance_statistics \
    --results_dir="/scratch/yifwang/fairness_x_explainability/decoder_results_jigsaw/val_results/llama_3b_jigsaw_religion_val_50_1/zero_shot" \
    --split="val" \
    --bias_type="religion" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion"

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="google/jigsaw_unintended_bias" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="meta-llama/Llama-3.2-3B-Instruct" \
    --batch_size=1 \
    --max_seq_length=512 \
    --num_examples=50 \
    --output_dir="/scratch/yifwang/fairness_x_explainability/decoder_results_jigsaw/val_results/llama_3b_jigsaw_religion_val_50_1/few_shot" \
    --bias_type="religion"\
    --prompt_type="few_shot" \
    --shuffle \
    --seed=1

python -m explanation_generation.gen_explanation_decoder \
    --dataset_name="google/jigsaw_unintended_bias" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="meta-llama/Llama-3.2-3B-Instruct" \
    --num_labels=2 \
    --batch_size=1 \
    --max_length=512 \
    --num_examples=50 \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion" \
    --output_dir="/scratch/yifwang/fairness_x_explainability/decoder_results_jigsaw/val_results/llama_3b_jigsaw_religion_val_50_1/few_shot" \
    --baseline="pad" \
    --bias_type="religion" \
    --prompt_type="few_shot" \
    --only_predicted_class \
    --shuffle \
    --seed=1

python -m sensitive_attribution_extraction.extract_sensitive_token_attribution \
    --results_dir="/scratch/yifwang/fairness_x_explainability/decoder_results_jigsaw/val_results/llama_3b_jigsaw_religion_val_50_1/few_shot" \
    --split="val" \
    --bias_type="religion" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion"

python -m model_selection.compute_reliance_statistics \
    --results_dir="/scratch/yifwang/fairness_x_explainability/decoder_results_jigsaw/val_results/llama_3b_jigsaw_religion_val_50_1/few_shot" \
    --split="val" \
    --bias_type="religion" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion"

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="google/jigsaw_unintended_bias" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="meta-llama/Llama-3.2-3B-Instruct" \
    --batch_size=1 \
    --max_seq_length=512 \
    --num_examples=50 \
    --output_dir="/scratch/yifwang/fairness_x_explainability/decoder_results_jigsaw/val_results/llama_3b_jigsaw_religion_val_50_1/fairness_imagination" \
    --bias_type="religion"\
    --prompt_type="fairness_imagination" \
    --shuffle \
    --seed=1

python -m explanation_generation.gen_explanation_decoder \
    --dataset_name="google/jigsaw_unintended_bias" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="meta-llama/Llama-3.2-3B-Instruct" \
    --num_labels=2 \
    --batch_size=1 \
    --max_length=512 \
    --num_examples=50 \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion" \
    --output_dir="/scratch/yifwang/fairness_x_explainability/decoder_results_jigsaw/val_results/llama_3b_jigsaw_religion_val_50_1/fairness_imagination" \
    --baseline="pad" \
    --bias_type="religion" \
    --prompt_type="fairness_imagination" \
    --only_predicted_class \
    --shuffle \
    --seed=1

python -m sensitive_attribution_extraction.extract_sensitive_token_attribution \
    --results_dir="/scratch/yifwang/fairness_x_explainability/decoder_results_jigsaw/val_results/llama_3b_jigsaw_religion_val_50_1/fairness_imagination" \
    --split="val" \
    --bias_type="religion" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion"

python -m model_selection.compute_reliance_statistics \
    --results_dir="/scratch/yifwang/fairness_x_explainability/decoder_results_jigsaw/val_results/llama_3b_jigsaw_religion_val_50_1/fairness_imagination" \
    --split="val" \
    --bias_type="religion" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion"

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="google/jigsaw_unintended_bias" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="meta-llama/Llama-3.2-3B-Instruct" \
    --batch_size=1 \
    --max_seq_length=512 \
    --num_examples=50 \
    --output_dir="/scratch/yifwang/fairness_x_explainability/decoder_results_jigsaw/val_results/llama_3b_jigsaw_religion_val_50_1/fairness_instruction" \
    --bias_type="religion"\
    --prompt_type="fairness_instruction" \
    --shuffle \
    --seed=1

python -m explanation_generation.gen_explanation_decoder \
    --dataset_name="google/jigsaw_unintended_bias" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="meta-llama/Llama-3.2-3B-Instruct" \
    --num_labels=2 \
    --batch_size=1 \
    --max_length=512 \
    --num_examples=50 \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion" \
    --output_dir="/scratch/yifwang/fairness_x_explainability/decoder_results_jigsaw/val_results/llama_3b_jigsaw_religion_val_50_1/fairness_instruction" \
    --baseline="pad" \
    --bias_type="religion" \
    --prompt_type="fairness_instruction" \
    --only_predicted_class \
    --shuffle \
    --seed=1

python -m sensitive_attribution_extraction.extract_sensitive_token_attribution \
    --results_dir="/scratch/yifwang/fairness_x_explainability/decoder_results_jigsaw/val_results/llama_3b_jigsaw_religion_val_50_1/fairness_instruction" \
    --split="val" \
    --bias_type="religion" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion"

python -m model_selection.compute_reliance_statistics \
    --results_dir="/scratch/yifwang/fairness_x_explainability/decoder_results_jigsaw/val_results/llama_3b_jigsaw_religion_val_50_1/fairness_instruction" \
    --split="val" \
    --bias_type="religion" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion"

python -m model_selection.compute_model_selection_correlation \
    --results_dir="/scratch/yifwang/fairness_x_explainability/decoder_results_jigsaw" \
    --val_results_dir="/scratch/yifwang/fairness_x_explainability/decoder_results_jigsaw/val_results" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion" \
    --experiment_type=decoder \
    --bias_type="religion" \
    --model_type="llama_3b" \
    --num_test_examples=200 \
    --num_val_examples=50 \
    --test_seed=-1 \
    --val_seed=1

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="google/jigsaw_unintended_bias" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="Qwen/Qwen3-4B" \
    --batch_size=1 \
    --max_seq_length=512 \
    --num_examples=50 \
    --output_dir="/scratch/yifwang/fairness_x_explainability/decoder_results_jigsaw/val_results/qwen3_4b_jigsaw_religion_val_50_1/zero_shot" \
    --bias_type="religion"\
    --prompt_type="zero_shot" \
    --shuffle \
    --seed=1

python -m explanation_generation.gen_explanation_decoder \
    --dataset_name="google/jigsaw_unintended_bias" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="Qwen/Qwen3-4B" \
    --num_labels=2 \
    --batch_size=1 \
    --max_length=512 \
    --num_examples=50 \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion" \
    --output_dir="/scratch/yifwang/fairness_x_explainability/decoder_results_jigsaw/val_results/qwen3_4b_jigsaw_religion_val_50_1/zero_shot" \
    --baseline="pad" \
    --bias_type="religion" \
    --prompt_type="zero_shot" \
    --only_predicted_class \
    --shuffle \
    --seed=1

python -m sensitive_attribution_extraction.extract_sensitive_token_attribution \
    --results_dir="/scratch/yifwang/fairness_x_explainability/decoder_results_jigsaw/val_results/qwen3_4b_jigsaw_religion_val_50_1/zero_shot" \
    --split="val" \
    --bias_type="religion" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion"

python -m model_selection.compute_reliance_statistics \
    --results_dir="/scratch/yifwang/fairness_x_explainability/decoder_results_jigsaw/val_results/qwen3_4b_jigsaw_religion_val_50_1/zero_shot" \
    --split="val" \
    --bias_type="religion" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion"

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="google/jigsaw_unintended_bias" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="Qwen/Qwen3-4B" \
    --batch_size=1 \
    --max_seq_length=512 \
    --num_examples=50 \
    --output_dir="/scratch/yifwang/fairness_x_explainability/decoder_results_jigsaw/val_results/qwen3_4b_jigsaw_religion_val_50_1/few_shot" \
    --bias_type="religion"\
    --prompt_type="few_shot" \
    --shuffle \
    --seed=1

python -m explanation_generation.gen_explanation_decoder \
    --dataset_name="google/jigsaw_unintended_bias" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="Qwen/Qwen3-4B" \
    --num_labels=2 \
    --batch_size=1 \
    --max_length=512 \
    --num_examples=50 \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion" \
    --output_dir="/scratch/yifwang/fairness_x_explainability/decoder_results_jigsaw/val_results/qwen3_4b_jigsaw_religion_val_50_1/few_shot" \
    --baseline="pad" \
    --bias_type="religion" \
    --prompt_type="few_shot" \
    --only_predicted_class \
    --shuffle \
    --seed=1

python -m sensitive_attribution_extraction.extract_sensitive_token_attribution \
    --results_dir="/scratch/yifwang/fairness_x_explainability/decoder_results_jigsaw/val_results/qwen3_4b_jigsaw_religion_val_50_1/few_shot" \
    --split="val" \
    --bias_type="religion" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion"

python -m model_selection.compute_reliance_statistics \
    --results_dir="/scratch/yifwang/fairness_x_explainability/decoder_results_jigsaw/val_results/qwen3_4b_jigsaw_religion_val_50_1/few_shot" \
    --split="val" \
    --bias_type="religion" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion"

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="google/jigsaw_unintended_bias" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="Qwen/Qwen3-4B" \
    --batch_size=1 \
    --max_seq_length=512 \
    --num_examples=50 \
    --output_dir="/scratch/yifwang/fairness_x_explainability/decoder_results_jigsaw/val_results/qwen3_4b_jigsaw_religion_val_50_1/fairness_imagination" \
    --bias_type="religion"\
    --prompt_type="fairness_imagination" \
    --shuffle \
    --seed=1

python -m explanation_generation.gen_explanation_decoder \
    --dataset_name="google/jigsaw_unintended_bias" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="Qwen/Qwen3-4B" \
    --num_labels=2 \
    --batch_size=1 \
    --max_length=512 \
    --num_examples=50 \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion" \
    --output_dir="/scratch/yifwang/fairness_x_explainability/decoder_results_jigsaw/val_results/qwen3_4b_jigsaw_religion_val_50_1/fairness_imagination" \
    --baseline="pad" \
    --bias_type="religion" \
    --prompt_type="fairness_imagination" \
    --only_predicted_class \
    --shuffle \
    --seed=1

python -m sensitive_attribution_extraction.extract_sensitive_token_attribution \
    --results_dir="/scratch/yifwang/fairness_x_explainability/decoder_results_jigsaw/val_results/qwen3_4b_jigsaw_religion_val_50_1/fairness_imagination" \
    --split="val" \
    --bias_type="religion" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion"

python -m model_selection.compute_reliance_statistics \
    --results_dir="/scratch/yifwang/fairness_x_explainability/decoder_results_jigsaw/val_results/qwen3_4b_jigsaw_religion_val_50_1/fairness_imagination" \
    --split="val" \
    --bias_type="religion" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion"

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="google/jigsaw_unintended_bias" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="Qwen/Qwen3-4B" \
    --batch_size=1 \
    --max_seq_length=512 \
    --num_examples=50 \
    --output_dir="/scratch/yifwang/fairness_x_explainability/decoder_results_jigsaw/val_results/qwen3_4b_jigsaw_religion_val_50_1/fairness_instruction" \
    --bias_type="religion"\
    --prompt_type="fairness_instruction" \
    --shuffle \
    --seed=1

python -m explanation_generation.gen_explanation_decoder \
    --dataset_name="google/jigsaw_unintended_bias" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="Qwen/Qwen3-4B" \
    --num_labels=2 \
    --batch_size=1 \
    --max_length=512 \
    --num_examples=50 \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion" \
    --output_dir="/scratch/yifwang/fairness_x_explainability/decoder_results_jigsaw/val_results/qwen3_4b_jigsaw_religion_val_50_1/fairness_instruction" \
    --baseline="pad" \
    --bias_type="religion" \
    --prompt_type="fairness_instruction" \
    --only_predicted_class \
    --shuffle \
    --seed=1

python -m sensitive_attribution_extraction.extract_sensitive_token_attribution \
    --results_dir="/scratch/yifwang/fairness_x_explainability/decoder_results_jigsaw/val_results/qwen3_4b_jigsaw_religion_val_50_1/fairness_instruction" \
    --split="val" \
    --bias_type="religion" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion"

python -m model_selection.compute_reliance_statistics \
    --results_dir="/scratch/yifwang/fairness_x_explainability/decoder_results_jigsaw/val_results/qwen3_4b_jigsaw_religion_val_50_1/fairness_instruction" \
    --split="val" \
    --bias_type="religion" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion"

python -m model_selection.compute_model_selection_correlation \
    --results_dir="/scratch/yifwang/fairness_x_explainability/decoder_results_jigsaw" \
    --val_results_dir="/scratch/yifwang/fairness_x_explainability/decoder_results_jigsaw/val_results" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion" \
    --experiment_type=decoder \
    --bias_type="religion" \
    --model_type="qwen3_4b" \
    --num_test_examples=200 \
    --num_val_examples=50 \
    --test_seed=-1 \
    --val_seed=1

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="google/jigsaw_unintended_bias" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="meta-llama/Llama-3.2-3B-Instruct" \
    --batch_size=1 \
    --max_seq_length=512 \
    --num_examples=50 \
    --output_dir="/scratch/yifwang/fairness_x_explainability/decoder_results_jigsaw/val_results/llama_3b_jigsaw_religion_val_50_2/zero_shot" \
    --bias_type="religion"\
    --prompt_type="zero_shot" \
    --shuffle \
    --seed=2

python -m explanation_generation.gen_explanation_decoder \
    --dataset_name="google/jigsaw_unintended_bias" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="meta-llama/Llama-3.2-3B-Instruct" \
    --num_labels=2 \
    --batch_size=1 \
    --max_length=512 \
    --num_examples=50 \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion" \
    --output_dir="/scratch/yifwang/fairness_x_explainability/decoder_results_jigsaw/val_results/llama_3b_jigsaw_religion_val_50_2/zero_shot" \
    --baseline="pad" \
    --bias_type="religion" \
    --prompt_type="zero_shot" \
    --only_predicted_class \
    --shuffle \
    --seed=2

python -m sensitive_attribution_extraction.extract_sensitive_token_attribution \
    --results_dir="/scratch/yifwang/fairness_x_explainability/decoder_results_jigsaw/val_results/llama_3b_jigsaw_religion_val_50_2/zero_shot" \
    --split="val" \
    --bias_type="religion" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion"

python -m model_selection.compute_reliance_statistics \
    --results_dir="/scratch/yifwang/fairness_x_explainability/decoder_results_jigsaw/val_results/llama_3b_jigsaw_religion_val_50_2/zero_shot" \
    --split="val" \
    --bias_type="religion" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion"

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="google/jigsaw_unintended_bias" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="meta-llama/Llama-3.2-3B-Instruct" \
    --batch_size=1 \
    --max_seq_length=512 \
    --num_examples=50 \
    --output_dir="/scratch/yifwang/fairness_x_explainability/decoder_results_jigsaw/val_results/llama_3b_jigsaw_religion_val_50_2/few_shot" \
    --bias_type="religion"\
    --prompt_type="few_shot" \
    --shuffle \
    --seed=2

python -m explanation_generation.gen_explanation_decoder \
    --dataset_name="google/jigsaw_unintended_bias" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="meta-llama/Llama-3.2-3B-Instruct" \
    --num_labels=2 \
    --batch_size=1 \
    --max_length=512 \
    --num_examples=50 \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion" \
    --output_dir="/scratch/yifwang/fairness_x_explainability/decoder_results_jigsaw/val_results/llama_3b_jigsaw_religion_val_50_2/few_shot" \
    --baseline="pad" \
    --bias_type="religion" \
    --prompt_type="few_shot" \
    --only_predicted_class \
    --shuffle \
    --seed=2

python -m sensitive_attribution_extraction.extract_sensitive_token_attribution \
    --results_dir="/scratch/yifwang/fairness_x_explainability/decoder_results_jigsaw/val_results/llama_3b_jigsaw_religion_val_50_2/few_shot" \
    --split="val" \
    --bias_type="religion" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion"

python -m model_selection.compute_reliance_statistics \
    --results_dir="/scratch/yifwang/fairness_x_explainability/decoder_results_jigsaw/val_results/llama_3b_jigsaw_religion_val_50_2/few_shot" \
    --split="val" \
    --bias_type="religion" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion"

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="google/jigsaw_unintended_bias" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="meta-llama/Llama-3.2-3B-Instruct" \
    --batch_size=1 \
    --max_seq_length=512 \
    --num_examples=50 \
    --output_dir="/scratch/yifwang/fairness_x_explainability/decoder_results_jigsaw/val_results/llama_3b_jigsaw_religion_val_50_2/fairness_imagination" \
    --bias_type="religion"\
    --prompt_type="fairness_imagination" \
    --shuffle \
    --seed=2

python -m explanation_generation.gen_explanation_decoder \
    --dataset_name="google/jigsaw_unintended_bias" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="meta-llama/Llama-3.2-3B-Instruct" \
    --num_labels=2 \
    --batch_size=1 \
    --max_length=512 \
    --num_examples=50 \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion" \
    --output_dir="/scratch/yifwang/fairness_x_explainability/decoder_results_jigsaw/val_results/llama_3b_jigsaw_religion_val_50_2/fairness_imagination" \
    --baseline="pad" \
    --bias_type="religion" \
    --prompt_type="fairness_imagination" \
    --only_predicted_class \
    --shuffle \
    --seed=2

python -m sensitive_attribution_extraction.extract_sensitive_token_attribution \
    --results_dir="/scratch/yifwang/fairness_x_explainability/decoder_results_jigsaw/val_results/llama_3b_jigsaw_religion_val_50_2/fairness_imagination" \
    --split="val" \
    --bias_type="religion" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion"

python -m model_selection.compute_reliance_statistics \
    --results_dir="/scratch/yifwang/fairness_x_explainability/decoder_results_jigsaw/val_results/llama_3b_jigsaw_religion_val_50_2/fairness_imagination" \
    --split="val" \
    --bias_type="religion" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion"

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="google/jigsaw_unintended_bias" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="meta-llama/Llama-3.2-3B-Instruct" \
    --batch_size=1 \
    --max_seq_length=512 \
    --num_examples=50 \
    --output_dir="/scratch/yifwang/fairness_x_explainability/decoder_results_jigsaw/val_results/llama_3b_jigsaw_religion_val_50_2/fairness_instruction" \
    --bias_type="religion"\
    --prompt_type="fairness_instruction" \
    --shuffle \
    --seed=2

python -m explanation_generation.gen_explanation_decoder \
    --dataset_name="google/jigsaw_unintended_bias" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="meta-llama/Llama-3.2-3B-Instruct" \
    --num_labels=2 \
    --batch_size=1 \
    --max_length=512 \
    --num_examples=50 \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion" \
    --output_dir="/scratch/yifwang/fairness_x_explainability/decoder_results_jigsaw/val_results/llama_3b_jigsaw_religion_val_50_2/fairness_instruction" \
    --baseline="pad" \
    --bias_type="religion" \
    --prompt_type="fairness_instruction" \
    --only_predicted_class \
    --shuffle \
    --seed=2

python -m sensitive_attribution_extraction.extract_sensitive_token_attribution \
    --results_dir="/scratch/yifwang/fairness_x_explainability/decoder_results_jigsaw/val_results/llama_3b_jigsaw_religion_val_50_2/fairness_instruction" \
    --split="val" \
    --bias_type="religion" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion"

python -m model_selection.compute_reliance_statistics \
    --results_dir="/scratch/yifwang/fairness_x_explainability/decoder_results_jigsaw/val_results/llama_3b_jigsaw_religion_val_50_2/fairness_instruction" \
    --split="val" \
    --bias_type="religion" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion"

python -m model_selection.compute_model_selection_correlation \
    --results_dir="/scratch/yifwang/fairness_x_explainability/decoder_results_jigsaw" \
    --val_results_dir="/scratch/yifwang/fairness_x_explainability/decoder_results_jigsaw/val_results" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion" \
    --experiment_type=decoder \
    --bias_type="religion" \
    --model_type="llama_3b" \
    --num_test_examples=200 \
    --num_val_examples=50 \
    --test_seed=-1 \
    --val_seed=2

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="google/jigsaw_unintended_bias" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="Qwen/Qwen3-4B" \
    --batch_size=1 \
    --max_seq_length=512 \
    --num_examples=50 \
    --output_dir="/scratch/yifwang/fairness_x_explainability/decoder_results_jigsaw/val_results/qwen3_4b_jigsaw_religion_val_50_2/zero_shot" \
    --bias_type="religion"\
    --prompt_type="zero_shot" \
    --shuffle \
    --seed=2

python -m explanation_generation.gen_explanation_decoder \
    --dataset_name="google/jigsaw_unintended_bias" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="Qwen/Qwen3-4B" \
    --num_labels=2 \
    --batch_size=1 \
    --max_length=512 \
    --num_examples=50 \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion" \
    --output_dir="/scratch/yifwang/fairness_x_explainability/decoder_results_jigsaw/val_results/qwen3_4b_jigsaw_religion_val_50_2/zero_shot" \
    --baseline="pad" \
    --bias_type="religion" \
    --prompt_type="zero_shot" \
    --only_predicted_class \
    --shuffle \
    --seed=2

python -m sensitive_attribution_extraction.extract_sensitive_token_attribution \
    --results_dir="/scratch/yifwang/fairness_x_explainability/decoder_results_jigsaw/val_results/qwen3_4b_jigsaw_religion_val_50_2/zero_shot" \
    --split="val" \
    --bias_type="religion" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion"

python -m model_selection.compute_reliance_statistics \
    --results_dir="/scratch/yifwang/fairness_x_explainability/decoder_results_jigsaw/val_results/qwen3_4b_jigsaw_religion_val_50_2/zero_shot" \
    --split="val" \
    --bias_type="religion" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion"

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="google/jigsaw_unintended_bias" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="Qwen/Qwen3-4B" \
    --batch_size=1 \
    --max_seq_length=512 \
    --num_examples=50 \
    --output_dir="/scratch/yifwang/fairness_x_explainability/decoder_results_jigsaw/val_results/qwen3_4b_jigsaw_religion_val_50_2/few_shot" \
    --bias_type="religion"\
    --prompt_type="few_shot" \
    --shuffle \
    --seed=2

python -m explanation_generation.gen_explanation_decoder \
    --dataset_name="google/jigsaw_unintended_bias" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="Qwen/Qwen3-4B" \
    --num_labels=2 \
    --batch_size=1 \
    --max_length=512 \
    --num_examples=50 \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion" \
    --output_dir="/scratch/yifwang/fairness_x_explainability/decoder_results_jigsaw/val_results/qwen3_4b_jigsaw_religion_val_50_2/few_shot" \
    --baseline="pad" \
    --bias_type="religion" \
    --prompt_type="few_shot" \
    --only_predicted_class \
    --shuffle \
    --seed=2

python -m sensitive_attribution_extraction.extract_sensitive_token_attribution \
    --results_dir="/scratch/yifwang/fairness_x_explainability/decoder_results_jigsaw/val_results/qwen3_4b_jigsaw_religion_val_50_2/few_shot" \
    --split="val" \
    --bias_type="religion" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion"

python -m model_selection.compute_reliance_statistics \
    --results_dir="/scratch/yifwang/fairness_x_explainability/decoder_results_jigsaw/val_results/qwen3_4b_jigsaw_religion_val_50_2/few_shot" \
    --split="val" \
    --bias_type="religion" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion"

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="google/jigsaw_unintended_bias" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="Qwen/Qwen3-4B" \
    --batch_size=1 \
    --max_seq_length=512 \
    --num_examples=50 \
    --output_dir="/scratch/yifwang/fairness_x_explainability/decoder_results_jigsaw/val_results/qwen3_4b_jigsaw_religion_val_50_2/fairness_imagination" \
    --bias_type="religion"\
    --prompt_type="fairness_imagination" \
    --shuffle \
    --seed=2

python -m explanation_generation.gen_explanation_decoder \
    --dataset_name="google/jigsaw_unintended_bias" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="Qwen/Qwen3-4B" \
    --num_labels=2 \
    --batch_size=1 \
    --max_length=512 \
    --num_examples=50 \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion" \
    --output_dir="/scratch/yifwang/fairness_x_explainability/decoder_results_jigsaw/val_results/qwen3_4b_jigsaw_religion_val_50_2/fairness_imagination" \
    --baseline="pad" \
    --bias_type="religion" \
    --prompt_type="fairness_imagination" \
    --only_predicted_class \
    --shuffle \
    --seed=2

python -m sensitive_attribution_extraction.extract_sensitive_token_attribution \
    --results_dir="/scratch/yifwang/fairness_x_explainability/decoder_results_jigsaw/val_results/qwen3_4b_jigsaw_religion_val_50_2/fairness_imagination" \
    --split="val" \
    --bias_type="religion" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion"

python -m model_selection.compute_reliance_statistics \
    --results_dir="/scratch/yifwang/fairness_x_explainability/decoder_results_jigsaw/val_results/qwen3_4b_jigsaw_religion_val_50_2/fairness_imagination" \
    --split="val" \
    --bias_type="religion" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion"

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="google/jigsaw_unintended_bias" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="Qwen/Qwen3-4B" \
    --batch_size=1 \
    --max_seq_length=512 \
    --num_examples=50 \
    --output_dir="/scratch/yifwang/fairness_x_explainability/decoder_results_jigsaw/val_results/qwen3_4b_jigsaw_religion_val_50_2/fairness_instruction" \
    --bias_type="religion"\
    --prompt_type="fairness_instruction" \
    --shuffle \
    --seed=2

python -m explanation_generation.gen_explanation_decoder \
    --dataset_name="google/jigsaw_unintended_bias" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="Qwen/Qwen3-4B" \
    --num_labels=2 \
    --batch_size=1 \
    --max_length=512 \
    --num_examples=50 \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion" \
    --output_dir="/scratch/yifwang/fairness_x_explainability/decoder_results_jigsaw/val_results/qwen3_4b_jigsaw_religion_val_50_2/fairness_instruction" \
    --baseline="pad" \
    --bias_type="religion" \
    --prompt_type="fairness_instruction" \
    --only_predicted_class \
    --shuffle \
    --seed=2

python -m sensitive_attribution_extraction.extract_sensitive_token_attribution \
    --results_dir="/scratch/yifwang/fairness_x_explainability/decoder_results_jigsaw/val_results/qwen3_4b_jigsaw_religion_val_50_2/fairness_instruction" \
    --split="val" \
    --bias_type="religion" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion"

python -m model_selection.compute_reliance_statistics \
    --results_dir="/scratch/yifwang/fairness_x_explainability/decoder_results_jigsaw/val_results/qwen3_4b_jigsaw_religion_val_50_2/fairness_instruction" \
    --split="val" \
    --bias_type="religion" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion"

python -m model_selection.compute_model_selection_correlation \
    --results_dir="/scratch/yifwang/fairness_x_explainability/decoder_results_jigsaw" \
    --val_results_dir="/scratch/yifwang/fairness_x_explainability/decoder_results_jigsaw/val_results" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion" \
    --experiment_type=decoder \
    --bias_type="religion" \
    --model_type="qwen3_4b" \
    --num_test_examples=200 \
    --num_val_examples=50 \
    --test_seed=-1 \
    --val_seed=2

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="google/jigsaw_unintended_bias" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="meta-llama/Llama-3.2-3B-Instruct" \
    --batch_size=1 \
    --max_seq_length=512 \
    --num_examples=50 \
    --output_dir="/scratch/yifwang/fairness_x_explainability/decoder_results_jigsaw/val_results/llama_3b_jigsaw_religion_val_50_3/zero_shot" \
    --bias_type="religion"\
    --prompt_type="zero_shot" \
    --shuffle \
    --seed=3

python -m explanation_generation.gen_explanation_decoder \
    --dataset_name="google/jigsaw_unintended_bias" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="meta-llama/Llama-3.2-3B-Instruct" \
    --num_labels=2 \
    --batch_size=1 \
    --max_length=512 \
    --num_examples=50 \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion" \
    --output_dir="/scratch/yifwang/fairness_x_explainability/decoder_results_jigsaw/val_results/llama_3b_jigsaw_religion_val_50_3/zero_shot" \
    --baseline="pad" \
    --bias_type="religion" \
    --prompt_type="zero_shot" \
    --only_predicted_class \
    --shuffle \
    --seed=3

python -m sensitive_attribution_extraction.extract_sensitive_token_attribution \
    --results_dir="/scratch/yifwang/fairness_x_explainability/decoder_results_jigsaw/val_results/llama_3b_jigsaw_religion_val_50_3/zero_shot" \
    --split="val" \
    --bias_type="religion" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion"

python -m model_selection.compute_reliance_statistics \
    --results_dir="/scratch/yifwang/fairness_x_explainability/decoder_results_jigsaw/val_results/llama_3b_jigsaw_religion_val_50_3/zero_shot" \
    --split="val" \
    --bias_type="religion" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion"

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="google/jigsaw_unintended_bias" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="meta-llama/Llama-3.2-3B-Instruct" \
    --batch_size=1 \
    --max_seq_length=512 \
    --num_examples=50 \
    --output_dir="/scratch/yifwang/fairness_x_explainability/decoder_results_jigsaw/val_results/llama_3b_jigsaw_religion_val_50_3/few_shot" \
    --bias_type="religion"\
    --prompt_type="few_shot" \
    --shuffle \
    --seed=3

python -m explanation_generation.gen_explanation_decoder \
    --dataset_name="google/jigsaw_unintended_bias" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="meta-llama/Llama-3.2-3B-Instruct" \
    --num_labels=2 \
    --batch_size=1 \
    --max_length=512 \
    --num_examples=50 \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion" \
    --output_dir="/scratch/yifwang/fairness_x_explainability/decoder_results_jigsaw/val_results/llama_3b_jigsaw_religion_val_50_3/few_shot" \
    --baseline="pad" \
    --bias_type="religion" \
    --prompt_type="few_shot" \
    --only_predicted_class \
    --shuffle \
    --seed=3

python -m sensitive_attribution_extraction.extract_sensitive_token_attribution \
    --results_dir="/scratch/yifwang/fairness_x_explainability/decoder_results_jigsaw/val_results/llama_3b_jigsaw_religion_val_50_3/few_shot" \
    --split="val" \
    --bias_type="religion" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion"

python -m model_selection.compute_reliance_statistics \
    --results_dir="/scratch/yifwang/fairness_x_explainability/decoder_results_jigsaw/val_results/llama_3b_jigsaw_religion_val_50_3/few_shot" \
    --split="val" \
    --bias_type="religion" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion"

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="google/jigsaw_unintended_bias" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="meta-llama/Llama-3.2-3B-Instruct" \
    --batch_size=1 \
    --max_seq_length=512 \
    --num_examples=50 \
    --output_dir="/scratch/yifwang/fairness_x_explainability/decoder_results_jigsaw/val_results/llama_3b_jigsaw_religion_val_50_3/fairness_imagination" \
    --bias_type="religion"\
    --prompt_type="fairness_imagination" \
    --shuffle \
    --seed=3

python -m explanation_generation.gen_explanation_decoder \
    --dataset_name="google/jigsaw_unintended_bias" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="meta-llama/Llama-3.2-3B-Instruct" \
    --num_labels=2 \
    --batch_size=1 \
    --max_length=512 \
    --num_examples=50 \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion" \
    --output_dir="/scratch/yifwang/fairness_x_explainability/decoder_results_jigsaw/val_results/llama_3b_jigsaw_religion_val_50_3/fairness_imagination" \
    --baseline="pad" \
    --bias_type="religion" \
    --prompt_type="fairness_imagination" \
    --only_predicted_class \
    --shuffle \
    --seed=3

python -m sensitive_attribution_extraction.extract_sensitive_token_attribution \
    --results_dir="/scratch/yifwang/fairness_x_explainability/decoder_results_jigsaw/val_results/llama_3b_jigsaw_religion_val_50_3/fairness_imagination" \
    --split="val" \
    --bias_type="religion" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion"

python -m model_selection.compute_reliance_statistics \
    --results_dir="/scratch/yifwang/fairness_x_explainability/decoder_results_jigsaw/val_results/llama_3b_jigsaw_religion_val_50_3/fairness_imagination" \
    --split="val" \
    --bias_type="religion" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion"

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="google/jigsaw_unintended_bias" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="meta-llama/Llama-3.2-3B-Instruct" \
    --batch_size=1 \
    --max_seq_length=512 \
    --num_examples=50 \
    --output_dir="/scratch/yifwang/fairness_x_explainability/decoder_results_jigsaw/val_results/llama_3b_jigsaw_religion_val_50_3/fairness_instruction" \
    --bias_type="religion"\
    --prompt_type="fairness_instruction" \
    --shuffle \
    --seed=3

python -m explanation_generation.gen_explanation_decoder \
    --dataset_name="google/jigsaw_unintended_bias" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="meta-llama/Llama-3.2-3B-Instruct" \
    --num_labels=2 \
    --batch_size=1 \
    --max_length=512 \
    --num_examples=50 \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion" \
    --output_dir="/scratch/yifwang/fairness_x_explainability/decoder_results_jigsaw/val_results/llama_3b_jigsaw_religion_val_50_3/fairness_instruction" \
    --baseline="pad" \
    --bias_type="religion" \
    --prompt_type="fairness_instruction" \
    --only_predicted_class \
    --shuffle \
    --seed=3

python -m sensitive_attribution_extraction.extract_sensitive_token_attribution \
    --results_dir="/scratch/yifwang/fairness_x_explainability/decoder_results_jigsaw/val_results/llama_3b_jigsaw_religion_val_50_3/fairness_instruction" \
    --split="val" \
    --bias_type="religion" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion"

python -m model_selection.compute_reliance_statistics \
    --results_dir="/scratch/yifwang/fairness_x_explainability/decoder_results_jigsaw/val_results/llama_3b_jigsaw_religion_val_50_3/fairness_instruction" \
    --split="val" \
    --bias_type="religion" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion"

python -m model_selection.compute_model_selection_correlation \
    --results_dir="/scratch/yifwang/fairness_x_explainability/decoder_results_jigsaw" \
    --val_results_dir="/scratch/yifwang/fairness_x_explainability/decoder_results_jigsaw/val_results" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion" \
    --experiment_type=decoder \
    --bias_type="religion" \
    --model_type="llama_3b" \
    --num_test_examples=200 \
    --num_val_examples=50 \
    --test_seed=-1 \
    --val_seed=3

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="google/jigsaw_unintended_bias" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="Qwen/Qwen3-4B" \
    --batch_size=1 \
    --max_seq_length=512 \
    --num_examples=50 \
    --output_dir="/scratch/yifwang/fairness_x_explainability/decoder_results_jigsaw/val_results/qwen3_4b_jigsaw_religion_val_50_3/zero_shot" \
    --bias_type="religion"\
    --prompt_type="zero_shot" \
    --shuffle \
    --seed=3

python -m explanation_generation.gen_explanation_decoder \
    --dataset_name="google/jigsaw_unintended_bias" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="Qwen/Qwen3-4B" \
    --num_labels=2 \
    --batch_size=1 \
    --max_length=512 \
    --num_examples=50 \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion" \
    --output_dir="/scratch/yifwang/fairness_x_explainability/decoder_results_jigsaw/val_results/qwen3_4b_jigsaw_religion_val_50_3/zero_shot" \
    --baseline="pad" \
    --bias_type="religion" \
    --prompt_type="zero_shot" \
    --only_predicted_class \
    --shuffle \
    --seed=3

python -m sensitive_attribution_extraction.extract_sensitive_token_attribution \
    --results_dir="/scratch/yifwang/fairness_x_explainability/decoder_results_jigsaw/val_results/qwen3_4b_jigsaw_religion_val_50_3/zero_shot" \
    --split="val" \
    --bias_type="religion" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion"

python -m model_selection.compute_reliance_statistics \
    --results_dir="/scratch/yifwang/fairness_x_explainability/decoder_results_jigsaw/val_results/qwen3_4b_jigsaw_religion_val_50_3/zero_shot" \
    --split="val" \
    --bias_type="religion" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion"

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="google/jigsaw_unintended_bias" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="Qwen/Qwen3-4B" \
    --batch_size=1 \
    --max_seq_length=512 \
    --num_examples=50 \
    --output_dir="/scratch/yifwang/fairness_x_explainability/decoder_results_jigsaw/val_results/qwen3_4b_jigsaw_religion_val_50_3/few_shot" \
    --bias_type="religion"\
    --prompt_type="few_shot" \
    --shuffle \
    --seed=3

python -m explanation_generation.gen_explanation_decoder \
    --dataset_name="google/jigsaw_unintended_bias" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="Qwen/Qwen3-4B" \
    --num_labels=2 \
    --batch_size=1 \
    --max_length=512 \
    --num_examples=50 \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion" \
    --output_dir="/scratch/yifwang/fairness_x_explainability/decoder_results_jigsaw/val_results/qwen3_4b_jigsaw_religion_val_50_3/few_shot" \
    --baseline="pad" \
    --bias_type="religion" \
    --prompt_type="few_shot" \
    --only_predicted_class \
    --shuffle \
    --seed=3

python -m sensitive_attribution_extraction.extract_sensitive_token_attribution \
    --results_dir="/scratch/yifwang/fairness_x_explainability/decoder_results_jigsaw/val_results/qwen3_4b_jigsaw_religion_val_50_3/few_shot" \
    --split="val" \
    --bias_type="religion" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion"

python -m model_selection.compute_reliance_statistics \
    --results_dir="/scratch/yifwang/fairness_x_explainability/decoder_results_jigsaw/val_results/qwen3_4b_jigsaw_religion_val_50_3/few_shot" \
    --split="val" \
    --bias_type="religion" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion"

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="google/jigsaw_unintended_bias" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="Qwen/Qwen3-4B" \
    --batch_size=1 \
    --max_seq_length=512 \
    --num_examples=50 \
    --output_dir="/scratch/yifwang/fairness_x_explainability/decoder_results_jigsaw/val_results/qwen3_4b_jigsaw_religion_val_50_3/fairness_imagination" \
    --bias_type="religion"\
    --prompt_type="fairness_imagination" \
    --shuffle \
    --seed=3

python -m explanation_generation.gen_explanation_decoder \
    --dataset_name="google/jigsaw_unintended_bias" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="Qwen/Qwen3-4B" \
    --num_labels=2 \
    --batch_size=1 \
    --max_length=512 \
    --num_examples=50 \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion" \
    --output_dir="/scratch/yifwang/fairness_x_explainability/decoder_results_jigsaw/val_results/qwen3_4b_jigsaw_religion_val_50_3/fairness_imagination" \
    --baseline="pad" \
    --bias_type="religion" \
    --prompt_type="fairness_imagination" \
    --only_predicted_class \
    --shuffle \
    --seed=3

python -m sensitive_attribution_extraction.extract_sensitive_token_attribution \
    --results_dir="/scratch/yifwang/fairness_x_explainability/decoder_results_jigsaw/val_results/qwen3_4b_jigsaw_religion_val_50_3/fairness_imagination" \
    --split="val" \
    --bias_type="religion" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion"

python -m model_selection.compute_reliance_statistics \
    --results_dir="/scratch/yifwang/fairness_x_explainability/decoder_results_jigsaw/val_results/qwen3_4b_jigsaw_religion_val_50_3/fairness_imagination" \
    --split="val" \
    --bias_type="religion" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion"

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="google/jigsaw_unintended_bias" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="Qwen/Qwen3-4B" \
    --batch_size=1 \
    --max_seq_length=512 \
    --num_examples=50 \
    --output_dir="/scratch/yifwang/fairness_x_explainability/decoder_results_jigsaw/val_results/qwen3_4b_jigsaw_religion_val_50_3/fairness_instruction" \
    --bias_type="religion"\
    --prompt_type="fairness_instruction" \
    --shuffle \
    --seed=3

python -m explanation_generation.gen_explanation_decoder \
    --dataset_name="google/jigsaw_unintended_bias" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="Qwen/Qwen3-4B" \
    --num_labels=2 \
    --batch_size=1 \
    --max_length=512 \
    --num_examples=50 \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion" \
    --output_dir="/scratch/yifwang/fairness_x_explainability/decoder_results_jigsaw/val_results/qwen3_4b_jigsaw_religion_val_50_3/fairness_instruction" \
    --baseline="pad" \
    --bias_type="religion" \
    --prompt_type="fairness_instruction" \
    --only_predicted_class \
    --shuffle \
    --seed=3

python -m sensitive_attribution_extraction.extract_sensitive_token_attribution \
    --results_dir="/scratch/yifwang/fairness_x_explainability/decoder_results_jigsaw/val_results/qwen3_4b_jigsaw_religion_val_50_3/fairness_instruction" \
    --split="val" \
    --bias_type="religion" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion"

python -m model_selection.compute_reliance_statistics \
    --results_dir="/scratch/yifwang/fairness_x_explainability/decoder_results_jigsaw/val_results/qwen3_4b_jigsaw_religion_val_50_3/fairness_instruction" \
    --split="val" \
    --bias_type="religion" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion"

python -m model_selection.compute_model_selection_correlation \
    --results_dir="/scratch/yifwang/fairness_x_explainability/decoder_results_jigsaw" \
    --val_results_dir="/scratch/yifwang/fairness_x_explainability/decoder_results_jigsaw/val_results" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion" \
    --experiment_type=decoder \
    --bias_type="religion" \
    --model_type="qwen3_4b" \
    --num_test_examples=200 \
    --num_val_examples=50 \
    --test_seed=-1 \
    --val_seed=3

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="google/jigsaw_unintended_bias" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="meta-llama/Llama-3.2-3B-Instruct" \
    --batch_size=1 \
    --max_seq_length=512 \
    --num_examples=50 \
    --output_dir="/scratch/yifwang/fairness_x_explainability/decoder_results_jigsaw/val_results/llama_3b_jigsaw_religion_val_50_42/zero_shot" \
    --bias_type="religion"\
    --prompt_type="zero_shot" \
    --shuffle \
    --seed=42

python -m explanation_generation.gen_explanation_decoder \
    --dataset_name="google/jigsaw_unintended_bias" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="meta-llama/Llama-3.2-3B-Instruct" \
    --num_labels=2 \
    --batch_size=1 \
    --max_length=512 \
    --num_examples=50 \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion" \
    --output_dir="/scratch/yifwang/fairness_x_explainability/decoder_results_jigsaw/val_results/llama_3b_jigsaw_religion_val_50_42/zero_shot" \
    --baseline="pad" \
    --bias_type="religion" \
    --prompt_type="zero_shot" \
    --only_predicted_class \
    --shuffle \
    --seed=42

python -m sensitive_attribution_extraction.extract_sensitive_token_attribution \
    --results_dir="/scratch/yifwang/fairness_x_explainability/decoder_results_jigsaw/val_results/llama_3b_jigsaw_religion_val_50_42/zero_shot" \
    --split="val" \
    --bias_type="religion" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion"

python -m model_selection.compute_reliance_statistics \
    --results_dir="/scratch/yifwang/fairness_x_explainability/decoder_results_jigsaw/val_results/llama_3b_jigsaw_religion_val_50_42/zero_shot" \
    --split="val" \
    --bias_type="religion" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion"

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="google/jigsaw_unintended_bias" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="meta-llama/Llama-3.2-3B-Instruct" \
    --batch_size=1 \
    --max_seq_length=512 \
    --num_examples=50 \
    --output_dir="/scratch/yifwang/fairness_x_explainability/decoder_results_jigsaw/val_results/llama_3b_jigsaw_religion_val_50_42/few_shot" \
    --bias_type="religion"\
    --prompt_type="few_shot" \
    --shuffle \
    --seed=42

python -m explanation_generation.gen_explanation_decoder \
    --dataset_name="google/jigsaw_unintended_bias" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="meta-llama/Llama-3.2-3B-Instruct" \
    --num_labels=2 \
    --batch_size=1 \
    --max_length=512 \
    --num_examples=50 \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion" \
    --output_dir="/scratch/yifwang/fairness_x_explainability/decoder_results_jigsaw/val_results/llama_3b_jigsaw_religion_val_50_42/few_shot" \
    --baseline="pad" \
    --bias_type="religion" \
    --prompt_type="few_shot" \
    --only_predicted_class \
    --shuffle \
    --seed=42

python -m sensitive_attribution_extraction.extract_sensitive_token_attribution \
    --results_dir="/scratch/yifwang/fairness_x_explainability/decoder_results_jigsaw/val_results/llama_3b_jigsaw_religion_val_50_42/few_shot" \
    --split="val" \
    --bias_type="religion" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion"

python -m model_selection.compute_reliance_statistics \
    --results_dir="/scratch/yifwang/fairness_x_explainability/decoder_results_jigsaw/val_results/llama_3b_jigsaw_religion_val_50_42/few_shot" \
    --split="val" \
    --bias_type="religion" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion"

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="google/jigsaw_unintended_bias" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="meta-llama/Llama-3.2-3B-Instruct" \
    --batch_size=1 \
    --max_seq_length=512 \
    --num_examples=50 \
    --output_dir="/scratch/yifwang/fairness_x_explainability/decoder_results_jigsaw/val_results/llama_3b_jigsaw_religion_val_50_42/fairness_imagination" \
    --bias_type="religion"\
    --prompt_type="fairness_imagination" \
    --shuffle \
    --seed=42

python -m explanation_generation.gen_explanation_decoder \
    --dataset_name="google/jigsaw_unintended_bias" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="meta-llama/Llama-3.2-3B-Instruct" \
    --num_labels=2 \
    --batch_size=1 \
    --max_length=512 \
    --num_examples=50 \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion" \
    --output_dir="/scratch/yifwang/fairness_x_explainability/decoder_results_jigsaw/val_results/llama_3b_jigsaw_religion_val_50_42/fairness_imagination" \
    --baseline="pad" \
    --bias_type="religion" \
    --prompt_type="fairness_imagination" \
    --only_predicted_class \
    --shuffle \
    --seed=42

python -m sensitive_attribution_extraction.extract_sensitive_token_attribution \
    --results_dir="/scratch/yifwang/fairness_x_explainability/decoder_results_jigsaw/val_results/llama_3b_jigsaw_religion_val_50_42/fairness_imagination" \
    --split="val" \
    --bias_type="religion" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion"

python -m model_selection.compute_reliance_statistics \
    --results_dir="/scratch/yifwang/fairness_x_explainability/decoder_results_jigsaw/val_results/llama_3b_jigsaw_religion_val_50_42/fairness_imagination" \
    --split="val" \
    --bias_type="religion" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion"

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="google/jigsaw_unintended_bias" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="meta-llama/Llama-3.2-3B-Instruct" \
    --batch_size=1 \
    --max_seq_length=512 \
    --num_examples=50 \
    --output_dir="/scratch/yifwang/fairness_x_explainability/decoder_results_jigsaw/val_results/llama_3b_jigsaw_religion_val_50_42/fairness_instruction" \
    --bias_type="religion"\
    --prompt_type="fairness_instruction" \
    --shuffle \
    --seed=42

python -m explanation_generation.gen_explanation_decoder \
    --dataset_name="google/jigsaw_unintended_bias" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="meta-llama/Llama-3.2-3B-Instruct" \
    --num_labels=2 \
    --batch_size=1 \
    --max_length=512 \
    --num_examples=50 \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion" \
    --output_dir="/scratch/yifwang/fairness_x_explainability/decoder_results_jigsaw/val_results/llama_3b_jigsaw_religion_val_50_42/fairness_instruction" \
    --baseline="pad" \
    --bias_type="religion" \
    --prompt_type="fairness_instruction" \
    --only_predicted_class \
    --shuffle \
    --seed=42

python -m sensitive_attribution_extraction.extract_sensitive_token_attribution \
    --results_dir="/scratch/yifwang/fairness_x_explainability/decoder_results_jigsaw/val_results/llama_3b_jigsaw_religion_val_50_42/fairness_instruction" \
    --split="val" \
    --bias_type="religion" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion"

python -m model_selection.compute_reliance_statistics \
    --results_dir="/scratch/yifwang/fairness_x_explainability/decoder_results_jigsaw/val_results/llama_3b_jigsaw_religion_val_50_42/fairness_instruction" \
    --split="val" \
    --bias_type="religion" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion"

python -m model_selection.compute_model_selection_correlation \
    --results_dir="/scratch/yifwang/fairness_x_explainability/decoder_results_jigsaw" \
    --val_results_dir="/scratch/yifwang/fairness_x_explainability/decoder_results_jigsaw/val_results" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion" \
    --experiment_type=decoder \
    --bias_type="religion" \
    --model_type="llama_3b" \
    --num_test_examples=200 \
    --num_val_examples=50 \
    --test_seed=-1 \
    --val_seed=42

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="google/jigsaw_unintended_bias" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="Qwen/Qwen3-4B" \
    --batch_size=1 \
    --max_seq_length=512 \
    --num_examples=50 \
    --output_dir="/scratch/yifwang/fairness_x_explainability/decoder_results_jigsaw/val_results/qwen3_4b_jigsaw_religion_val_50_42/zero_shot" \
    --bias_type="religion"\
    --prompt_type="zero_shot" \
    --shuffle \
    --seed=42

python -m explanation_generation.gen_explanation_decoder \
    --dataset_name="google/jigsaw_unintended_bias" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="Qwen/Qwen3-4B" \
    --num_labels=2 \
    --batch_size=1 \
    --max_length=512 \
    --num_examples=50 \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion" \
    --output_dir="/scratch/yifwang/fairness_x_explainability/decoder_results_jigsaw/val_results/qwen3_4b_jigsaw_religion_val_50_42/zero_shot" \
    --baseline="pad" \
    --bias_type="religion" \
    --prompt_type="zero_shot" \
    --only_predicted_class \
    --shuffle \
    --seed=42

python -m sensitive_attribution_extraction.extract_sensitive_token_attribution \
    --results_dir="/scratch/yifwang/fairness_x_explainability/decoder_results_jigsaw/val_results/qwen3_4b_jigsaw_religion_val_50_42/zero_shot" \
    --split="val" \
    --bias_type="religion" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion"

python -m model_selection.compute_reliance_statistics \
    --results_dir="/scratch/yifwang/fairness_x_explainability/decoder_results_jigsaw/val_results/qwen3_4b_jigsaw_religion_val_50_42/zero_shot" \
    --split="val" \
    --bias_type="religion" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion"

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="google/jigsaw_unintended_bias" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="Qwen/Qwen3-4B" \
    --batch_size=1 \
    --max_seq_length=512 \
    --num_examples=50 \
    --output_dir="/scratch/yifwang/fairness_x_explainability/decoder_results_jigsaw/val_results/qwen3_4b_jigsaw_religion_val_50_42/few_shot" \
    --bias_type="religion"\
    --prompt_type="few_shot" \
    --shuffle \
    --seed=42

python -m explanation_generation.gen_explanation_decoder \
    --dataset_name="google/jigsaw_unintended_bias" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="Qwen/Qwen3-4B" \
    --num_labels=2 \
    --batch_size=1 \
    --max_length=512 \
    --num_examples=50 \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion" \
    --output_dir="/scratch/yifwang/fairness_x_explainability/decoder_results_jigsaw/val_results/qwen3_4b_jigsaw_religion_val_50_42/few_shot" \
    --baseline="pad" \
    --bias_type="religion" \
    --prompt_type="few_shot" \
    --only_predicted_class \
    --shuffle \
    --seed=42

python -m sensitive_attribution_extraction.extract_sensitive_token_attribution \
    --results_dir="/scratch/yifwang/fairness_x_explainability/decoder_results_jigsaw/val_results/qwen3_4b_jigsaw_religion_val_50_42/few_shot" \
    --split="val" \
    --bias_type="religion" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion"

python -m model_selection.compute_reliance_statistics \
    --results_dir="/scratch/yifwang/fairness_x_explainability/decoder_results_jigsaw/val_results/qwen3_4b_jigsaw_religion_val_50_42/few_shot" \
    --split="val" \
    --bias_type="religion" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion"

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="google/jigsaw_unintended_bias" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="Qwen/Qwen3-4B" \
    --batch_size=1 \
    --max_seq_length=512 \
    --num_examples=50 \
    --output_dir="/scratch/yifwang/fairness_x_explainability/decoder_results_jigsaw/val_results/qwen3_4b_jigsaw_religion_val_50_42/fairness_imagination" \
    --bias_type="religion"\
    --prompt_type="fairness_imagination" \
    --shuffle \
    --seed=42

python -m explanation_generation.gen_explanation_decoder \
    --dataset_name="google/jigsaw_unintended_bias" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="Qwen/Qwen3-4B" \
    --num_labels=2 \
    --batch_size=1 \
    --max_length=512 \
    --num_examples=50 \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion" \
    --output_dir="/scratch/yifwang/fairness_x_explainability/decoder_results_jigsaw/val_results/qwen3_4b_jigsaw_religion_val_50_42/fairness_imagination" \
    --baseline="pad" \
    --bias_type="religion" \
    --prompt_type="fairness_imagination" \
    --only_predicted_class \
    --shuffle \
    --seed=42

python -m sensitive_attribution_extraction.extract_sensitive_token_attribution \
    --results_dir="/scratch/yifwang/fairness_x_explainability/decoder_results_jigsaw/val_results/qwen3_4b_jigsaw_religion_val_50_42/fairness_imagination" \
    --split="val" \
    --bias_type="religion" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion"

python -m model_selection.compute_reliance_statistics \
    --results_dir="/scratch/yifwang/fairness_x_explainability/decoder_results_jigsaw/val_results/qwen3_4b_jigsaw_religion_val_50_42/fairness_imagination" \
    --split="val" \
    --bias_type="religion" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion"

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="google/jigsaw_unintended_bias" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="Qwen/Qwen3-4B" \
    --batch_size=1 \
    --max_seq_length=512 \
    --num_examples=50 \
    --output_dir="/scratch/yifwang/fairness_x_explainability/decoder_results_jigsaw/val_results/qwen3_4b_jigsaw_religion_val_50_42/fairness_instruction" \
    --bias_type="religion"\
    --prompt_type="fairness_instruction" \
    --shuffle \
    --seed=42

python -m explanation_generation.gen_explanation_decoder \
    --dataset_name="google/jigsaw_unintended_bias" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="Qwen/Qwen3-4B" \
    --num_labels=2 \
    --batch_size=1 \
    --max_length=512 \
    --num_examples=50 \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion" \
    --output_dir="/scratch/yifwang/fairness_x_explainability/decoder_results_jigsaw/val_results/qwen3_4b_jigsaw_religion_val_50_42/fairness_instruction" \
    --baseline="pad" \
    --bias_type="religion" \
    --prompt_type="fairness_instruction" \
    --only_predicted_class \
    --shuffle \
    --seed=42

python -m sensitive_attribution_extraction.extract_sensitive_token_attribution \
    --results_dir="/scratch/yifwang/fairness_x_explainability/decoder_results_jigsaw/val_results/qwen3_4b_jigsaw_religion_val_50_42/fairness_instruction" \
    --split="val" \
    --bias_type="religion" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion"

python -m model_selection.compute_reliance_statistics \
    --results_dir="/scratch/yifwang/fairness_x_explainability/decoder_results_jigsaw/val_results/qwen3_4b_jigsaw_religion_val_50_42/fairness_instruction" \
    --split="val" \
    --bias_type="religion" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion"

python -m model_selection.compute_model_selection_correlation \
    --results_dir="/scratch/yifwang/fairness_x_explainability/decoder_results_jigsaw" \
    --val_results_dir="/scratch/yifwang/fairness_x_explainability/decoder_results_jigsaw/val_results" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion" \
    --experiment_type=decoder \
    --bias_type="religion" \
    --model_type="qwen3_4b" \
    --num_test_examples=200 \
    --num_val_examples=50 \
    --test_seed=-1 \
    --val_seed=42

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="google/jigsaw_unintended_bias" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="meta-llama/Llama-3.2-3B-Instruct" \
    --batch_size=1 \
    --max_seq_length=512 \
    --num_examples=100 \
    --output_dir="/scratch/yifwang/fairness_x_explainability/decoder_results_jigsaw/val_results/llama_3b_jigsaw_religion_val_100_1/zero_shot" \
    --bias_type="religion"\
    --prompt_type="zero_shot" \
    --shuffle \
    --seed=1

python -m explanation_generation.gen_explanation_decoder \
    --dataset_name="google/jigsaw_unintended_bias" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="meta-llama/Llama-3.2-3B-Instruct" \
    --num_labels=2 \
    --batch_size=1 \
    --max_length=512 \
    --num_examples=100 \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion" \
    --output_dir="/scratch/yifwang/fairness_x_explainability/decoder_results_jigsaw/val_results/llama_3b_jigsaw_religion_val_100_1/zero_shot" \
    --baseline="pad" \
    --bias_type="religion" \
    --prompt_type="zero_shot" \
    --only_predicted_class \
    --shuffle \
    --seed=1

python -m sensitive_attribution_extraction.extract_sensitive_token_attribution \
    --results_dir="/scratch/yifwang/fairness_x_explainability/decoder_results_jigsaw/val_results/llama_3b_jigsaw_religion_val_100_1/zero_shot" \
    --split="val" \
    --bias_type="religion" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion"

python -m model_selection.compute_reliance_statistics \
    --results_dir="/scratch/yifwang/fairness_x_explainability/decoder_results_jigsaw/val_results/llama_3b_jigsaw_religion_val_100_1/zero_shot" \
    --split="val" \
    --bias_type="religion" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion"

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="google/jigsaw_unintended_bias" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="meta-llama/Llama-3.2-3B-Instruct" \
    --batch_size=1 \
    --max_seq_length=512 \
    --num_examples=100 \
    --output_dir="/scratch/yifwang/fairness_x_explainability/decoder_results_jigsaw/val_results/llama_3b_jigsaw_religion_val_100_1/few_shot" \
    --bias_type="religion"\
    --prompt_type="few_shot" \
    --shuffle \
    --seed=1

python -m explanation_generation.gen_explanation_decoder \
    --dataset_name="google/jigsaw_unintended_bias" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="meta-llama/Llama-3.2-3B-Instruct" \
    --num_labels=2 \
    --batch_size=1 \
    --max_length=512 \
    --num_examples=100 \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion" \
    --output_dir="/scratch/yifwang/fairness_x_explainability/decoder_results_jigsaw/val_results/llama_3b_jigsaw_religion_val_100_1/few_shot" \
    --baseline="pad" \
    --bias_type="religion" \
    --prompt_type="few_shot" \
    --only_predicted_class \
    --shuffle \
    --seed=1

python -m sensitive_attribution_extraction.extract_sensitive_token_attribution \
    --results_dir="/scratch/yifwang/fairness_x_explainability/decoder_results_jigsaw/val_results/llama_3b_jigsaw_religion_val_100_1/few_shot" \
    --split="val" \
    --bias_type="religion" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion"

python -m model_selection.compute_reliance_statistics \
    --results_dir="/scratch/yifwang/fairness_x_explainability/decoder_results_jigsaw/val_results/llama_3b_jigsaw_religion_val_100_1/few_shot" \
    --split="val" \
    --bias_type="religion" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion"

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="google/jigsaw_unintended_bias" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="meta-llama/Llama-3.2-3B-Instruct" \
    --batch_size=1 \
    --max_seq_length=512 \
    --num_examples=100 \
    --output_dir="/scratch/yifwang/fairness_x_explainability/decoder_results_jigsaw/val_results/llama_3b_jigsaw_religion_val_100_1/fairness_imagination" \
    --bias_type="religion"\
    --prompt_type="fairness_imagination" \
    --shuffle \
    --seed=1

python -m explanation_generation.gen_explanation_decoder \
    --dataset_name="google/jigsaw_unintended_bias" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="meta-llama/Llama-3.2-3B-Instruct" \
    --num_labels=2 \
    --batch_size=1 \
    --max_length=512 \
    --num_examples=100 \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion" \
    --output_dir="/scratch/yifwang/fairness_x_explainability/decoder_results_jigsaw/val_results/llama_3b_jigsaw_religion_val_100_1/fairness_imagination" \
    --baseline="pad" \
    --bias_type="religion" \
    --prompt_type="fairness_imagination" \
    --only_predicted_class \
    --shuffle \
    --seed=1

python -m sensitive_attribution_extraction.extract_sensitive_token_attribution \
    --results_dir="/scratch/yifwang/fairness_x_explainability/decoder_results_jigsaw/val_results/llama_3b_jigsaw_religion_val_100_1/fairness_imagination" \
    --split="val" \
    --bias_type="religion" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion"

python -m model_selection.compute_reliance_statistics \
    --results_dir="/scratch/yifwang/fairness_x_explainability/decoder_results_jigsaw/val_results/llama_3b_jigsaw_religion_val_100_1/fairness_imagination" \
    --split="val" \
    --bias_type="religion" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion"

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="google/jigsaw_unintended_bias" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="meta-llama/Llama-3.2-3B-Instruct" \
    --batch_size=1 \
    --max_seq_length=512 \
    --num_examples=100 \
    --output_dir="/scratch/yifwang/fairness_x_explainability/decoder_results_jigsaw/val_results/llama_3b_jigsaw_religion_val_100_1/fairness_instruction" \
    --bias_type="religion"\
    --prompt_type="fairness_instruction" \
    --shuffle \
    --seed=1

python -m explanation_generation.gen_explanation_decoder \
    --dataset_name="google/jigsaw_unintended_bias" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="meta-llama/Llama-3.2-3B-Instruct" \
    --num_labels=2 \
    --batch_size=1 \
    --max_length=512 \
    --num_examples=100 \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion" \
    --output_dir="/scratch/yifwang/fairness_x_explainability/decoder_results_jigsaw/val_results/llama_3b_jigsaw_religion_val_100_1/fairness_instruction" \
    --baseline="pad" \
    --bias_type="religion" \
    --prompt_type="fairness_instruction" \
    --only_predicted_class \
    --shuffle \
    --seed=1

python -m sensitive_attribution_extraction.extract_sensitive_token_attribution \
    --results_dir="/scratch/yifwang/fairness_x_explainability/decoder_results_jigsaw/val_results/llama_3b_jigsaw_religion_val_100_1/fairness_instruction" \
    --split="val" \
    --bias_type="religion" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion"

python -m model_selection.compute_reliance_statistics \
    --results_dir="/scratch/yifwang/fairness_x_explainability/decoder_results_jigsaw/val_results/llama_3b_jigsaw_religion_val_100_1/fairness_instruction" \
    --split="val" \
    --bias_type="religion" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion"

python -m model_selection.compute_model_selection_correlation \
    --results_dir="/scratch/yifwang/fairness_x_explainability/decoder_results_jigsaw" \
    --val_results_dir="/scratch/yifwang/fairness_x_explainability/decoder_results_jigsaw/val_results" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion" \
    --experiment_type=decoder \
    --bias_type="religion" \
    --model_type="llama_3b" \
    --num_test_examples=200 \
    --num_val_examples=100 \
    --test_seed=-1 \
    --val_seed=1

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="google/jigsaw_unintended_bias" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="Qwen/Qwen3-4B" \
    --batch_size=1 \
    --max_seq_length=512 \
    --num_examples=100 \
    --output_dir="/scratch/yifwang/fairness_x_explainability/decoder_results_jigsaw/val_results/qwen3_4b_jigsaw_religion_val_100_1/zero_shot" \
    --bias_type="religion"\
    --prompt_type="zero_shot" \
    --shuffle \
    --seed=1

python -m explanation_generation.gen_explanation_decoder \
    --dataset_name="google/jigsaw_unintended_bias" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="Qwen/Qwen3-4B" \
    --num_labels=2 \
    --batch_size=1 \
    --max_length=512 \
    --num_examples=100 \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion" \
    --output_dir="/scratch/yifwang/fairness_x_explainability/decoder_results_jigsaw/val_results/qwen3_4b_jigsaw_religion_val_100_1/zero_shot" \
    --baseline="pad" \
    --bias_type="religion" \
    --prompt_type="zero_shot" \
    --only_predicted_class \
    --shuffle \
    --seed=1

python -m sensitive_attribution_extraction.extract_sensitive_token_attribution \
    --results_dir="/scratch/yifwang/fairness_x_explainability/decoder_results_jigsaw/val_results/qwen3_4b_jigsaw_religion_val_100_1/zero_shot" \
    --split="val" \
    --bias_type="religion" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion"

python -m model_selection.compute_reliance_statistics \
    --results_dir="/scratch/yifwang/fairness_x_explainability/decoder_results_jigsaw/val_results/qwen3_4b_jigsaw_religion_val_100_1/zero_shot" \
    --split="val" \
    --bias_type="religion" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion"

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="google/jigsaw_unintended_bias" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="Qwen/Qwen3-4B" \
    --batch_size=1 \
    --max_seq_length=512 \
    --num_examples=100 \
    --output_dir="/scratch/yifwang/fairness_x_explainability/decoder_results_jigsaw/val_results/qwen3_4b_jigsaw_religion_val_100_1/few_shot" \
    --bias_type="religion"\
    --prompt_type="few_shot" \
    --shuffle \
    --seed=1

python -m explanation_generation.gen_explanation_decoder \
    --dataset_name="google/jigsaw_unintended_bias" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="Qwen/Qwen3-4B" \
    --num_labels=2 \
    --batch_size=1 \
    --max_length=512 \
    --num_examples=100 \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion" \
    --output_dir="/scratch/yifwang/fairness_x_explainability/decoder_results_jigsaw/val_results/qwen3_4b_jigsaw_religion_val_100_1/few_shot" \
    --baseline="pad" \
    --bias_type="religion" \
    --prompt_type="few_shot" \
    --only_predicted_class \
    --shuffle \
    --seed=1

python -m sensitive_attribution_extraction.extract_sensitive_token_attribution \
    --results_dir="/scratch/yifwang/fairness_x_explainability/decoder_results_jigsaw/val_results/qwen3_4b_jigsaw_religion_val_100_1/few_shot" \
    --split="val" \
    --bias_type="religion" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion"

python -m model_selection.compute_reliance_statistics \
    --results_dir="/scratch/yifwang/fairness_x_explainability/decoder_results_jigsaw/val_results/qwen3_4b_jigsaw_religion_val_100_1/few_shot" \
    --split="val" \
    --bias_type="religion" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion"

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="google/jigsaw_unintended_bias" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="Qwen/Qwen3-4B" \
    --batch_size=1 \
    --max_seq_length=512 \
    --num_examples=100 \
    --output_dir="/scratch/yifwang/fairness_x_explainability/decoder_results_jigsaw/val_results/qwen3_4b_jigsaw_religion_val_100_1/fairness_imagination" \
    --bias_type="religion"\
    --prompt_type="fairness_imagination" \
    --shuffle \
    --seed=1

python -m explanation_generation.gen_explanation_decoder \
    --dataset_name="google/jigsaw_unintended_bias" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="Qwen/Qwen3-4B" \
    --num_labels=2 \
    --batch_size=1 \
    --max_length=512 \
    --num_examples=100 \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion" \
    --output_dir="/scratch/yifwang/fairness_x_explainability/decoder_results_jigsaw/val_results/qwen3_4b_jigsaw_religion_val_100_1/fairness_imagination" \
    --baseline="pad" \
    --bias_type="religion" \
    --prompt_type="fairness_imagination" \
    --only_predicted_class \
    --shuffle \
    --seed=1

python -m sensitive_attribution_extraction.extract_sensitive_token_attribution \
    --results_dir="/scratch/yifwang/fairness_x_explainability/decoder_results_jigsaw/val_results/qwen3_4b_jigsaw_religion_val_100_1/fairness_imagination" \
    --split="val" \
    --bias_type="religion" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion"

python -m model_selection.compute_reliance_statistics \
    --results_dir="/scratch/yifwang/fairness_x_explainability/decoder_results_jigsaw/val_results/qwen3_4b_jigsaw_religion_val_100_1/fairness_imagination" \
    --split="val" \
    --bias_type="religion" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion"

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="google/jigsaw_unintended_bias" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="Qwen/Qwen3-4B" \
    --batch_size=1 \
    --max_seq_length=512 \
    --num_examples=100 \
    --output_dir="/scratch/yifwang/fairness_x_explainability/decoder_results_jigsaw/val_results/qwen3_4b_jigsaw_religion_val_100_1/fairness_instruction" \
    --bias_type="religion"\
    --prompt_type="fairness_instruction" \
    --shuffle \
    --seed=1

python -m explanation_generation.gen_explanation_decoder \
    --dataset_name="google/jigsaw_unintended_bias" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="Qwen/Qwen3-4B" \
    --num_labels=2 \
    --batch_size=1 \
    --max_length=512 \
    --num_examples=100 \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion" \
    --output_dir="/scratch/yifwang/fairness_x_explainability/decoder_results_jigsaw/val_results/qwen3_4b_jigsaw_religion_val_100_1/fairness_instruction" \
    --baseline="pad" \
    --bias_type="religion" \
    --prompt_type="fairness_instruction" \
    --only_predicted_class \
    --shuffle \
    --seed=1

python -m sensitive_attribution_extraction.extract_sensitive_token_attribution \
    --results_dir="/scratch/yifwang/fairness_x_explainability/decoder_results_jigsaw/val_results/qwen3_4b_jigsaw_religion_val_100_1/fairness_instruction" \
    --split="val" \
    --bias_type="religion" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion"

python -m model_selection.compute_reliance_statistics \
    --results_dir="/scratch/yifwang/fairness_x_explainability/decoder_results_jigsaw/val_results/qwen3_4b_jigsaw_religion_val_100_1/fairness_instruction" \
    --split="val" \
    --bias_type="religion" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion"

python -m model_selection.compute_model_selection_correlation \
    --results_dir="/scratch/yifwang/fairness_x_explainability/decoder_results_jigsaw" \
    --val_results_dir="/scratch/yifwang/fairness_x_explainability/decoder_results_jigsaw/val_results" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion" \
    --experiment_type=decoder \
    --bias_type="religion" \
    --model_type="qwen3_4b" \
    --num_test_examples=200 \
    --num_val_examples=100 \
    --test_seed=-1 \
    --val_seed=1

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="google/jigsaw_unintended_bias" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="meta-llama/Llama-3.2-3B-Instruct" \
    --batch_size=1 \
    --max_seq_length=512 \
    --num_examples=100 \
    --output_dir="/scratch/yifwang/fairness_x_explainability/decoder_results_jigsaw/val_results/llama_3b_jigsaw_religion_val_100_2/zero_shot" \
    --bias_type="religion"\
    --prompt_type="zero_shot" \
    --shuffle \
    --seed=2

python -m explanation_generation.gen_explanation_decoder \
    --dataset_name="google/jigsaw_unintended_bias" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="meta-llama/Llama-3.2-3B-Instruct" \
    --num_labels=2 \
    --batch_size=1 \
    --max_length=512 \
    --num_examples=100 \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion" \
    --output_dir="/scratch/yifwang/fairness_x_explainability/decoder_results_jigsaw/val_results/llama_3b_jigsaw_religion_val_100_2/zero_shot" \
    --baseline="pad" \
    --bias_type="religion" \
    --prompt_type="zero_shot" \
    --only_predicted_class \
    --shuffle \
    --seed=2

python -m sensitive_attribution_extraction.extract_sensitive_token_attribution \
    --results_dir="/scratch/yifwang/fairness_x_explainability/decoder_results_jigsaw/val_results/llama_3b_jigsaw_religion_val_100_2/zero_shot" \
    --split="val" \
    --bias_type="religion" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion"

python -m model_selection.compute_reliance_statistics \
    --results_dir="/scratch/yifwang/fairness_x_explainability/decoder_results_jigsaw/val_results/llama_3b_jigsaw_religion_val_100_2/zero_shot" \
    --split="val" \
    --bias_type="religion" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion"

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="google/jigsaw_unintended_bias" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="meta-llama/Llama-3.2-3B-Instruct" \
    --batch_size=1 \
    --max_seq_length=512 \
    --num_examples=100 \
    --output_dir="/scratch/yifwang/fairness_x_explainability/decoder_results_jigsaw/val_results/llama_3b_jigsaw_religion_val_100_2/few_shot" \
    --bias_type="religion"\
    --prompt_type="few_shot" \
    --shuffle \
    --seed=2

python -m explanation_generation.gen_explanation_decoder \
    --dataset_name="google/jigsaw_unintended_bias" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="meta-llama/Llama-3.2-3B-Instruct" \
    --num_labels=2 \
    --batch_size=1 \
    --max_length=512 \
    --num_examples=100 \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion" \
    --output_dir="/scratch/yifwang/fairness_x_explainability/decoder_results_jigsaw/val_results/llama_3b_jigsaw_religion_val_100_2/few_shot" \
    --baseline="pad" \
    --bias_type="religion" \
    --prompt_type="few_shot" \
    --only_predicted_class \
    --shuffle \
    --seed=2

python -m sensitive_attribution_extraction.extract_sensitive_token_attribution \
    --results_dir="/scratch/yifwang/fairness_x_explainability/decoder_results_jigsaw/val_results/llama_3b_jigsaw_religion_val_100_2/few_shot" \
    --split="val" \
    --bias_type="religion" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion"

python -m model_selection.compute_reliance_statistics \
    --results_dir="/scratch/yifwang/fairness_x_explainability/decoder_results_jigsaw/val_results/llama_3b_jigsaw_religion_val_100_2/few_shot" \
    --split="val" \
    --bias_type="religion" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion"

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="google/jigsaw_unintended_bias" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="meta-llama/Llama-3.2-3B-Instruct" \
    --batch_size=1 \
    --max_seq_length=512 \
    --num_examples=100 \
    --output_dir="/scratch/yifwang/fairness_x_explainability/decoder_results_jigsaw/val_results/llama_3b_jigsaw_religion_val_100_2/fairness_imagination" \
    --bias_type="religion"\
    --prompt_type="fairness_imagination" \
    --shuffle \
    --seed=2

python -m explanation_generation.gen_explanation_decoder \
    --dataset_name="google/jigsaw_unintended_bias" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="meta-llama/Llama-3.2-3B-Instruct" \
    --num_labels=2 \
    --batch_size=1 \
    --max_length=512 \
    --num_examples=100 \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion" \
    --output_dir="/scratch/yifwang/fairness_x_explainability/decoder_results_jigsaw/val_results/llama_3b_jigsaw_religion_val_100_2/fairness_imagination" \
    --baseline="pad" \
    --bias_type="religion" \
    --prompt_type="fairness_imagination" \
    --only_predicted_class \
    --shuffle \
    --seed=2

python -m sensitive_attribution_extraction.extract_sensitive_token_attribution \
    --results_dir="/scratch/yifwang/fairness_x_explainability/decoder_results_jigsaw/val_results/llama_3b_jigsaw_religion_val_100_2/fairness_imagination" \
    --split="val" \
    --bias_type="religion" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion"

python -m model_selection.compute_reliance_statistics \
    --results_dir="/scratch/yifwang/fairness_x_explainability/decoder_results_jigsaw/val_results/llama_3b_jigsaw_religion_val_100_2/fairness_imagination" \
    --split="val" \
    --bias_type="religion" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion"

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="google/jigsaw_unintended_bias" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="meta-llama/Llama-3.2-3B-Instruct" \
    --batch_size=1 \
    --max_seq_length=512 \
    --num_examples=100 \
    --output_dir="/scratch/yifwang/fairness_x_explainability/decoder_results_jigsaw/val_results/llama_3b_jigsaw_religion_val_100_2/fairness_instruction" \
    --bias_type="religion"\
    --prompt_type="fairness_instruction" \
    --shuffle \
    --seed=2

python -m explanation_generation.gen_explanation_decoder \
    --dataset_name="google/jigsaw_unintended_bias" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="meta-llama/Llama-3.2-3B-Instruct" \
    --num_labels=2 \
    --batch_size=1 \
    --max_length=512 \
    --num_examples=100 \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion" \
    --output_dir="/scratch/yifwang/fairness_x_explainability/decoder_results_jigsaw/val_results/llama_3b_jigsaw_religion_val_100_2/fairness_instruction" \
    --baseline="pad" \
    --bias_type="religion" \
    --prompt_type="fairness_instruction" \
    --only_predicted_class \
    --shuffle \
    --seed=2

python -m sensitive_attribution_extraction.extract_sensitive_token_attribution \
    --results_dir="/scratch/yifwang/fairness_x_explainability/decoder_results_jigsaw/val_results/llama_3b_jigsaw_religion_val_100_2/fairness_instruction" \
    --split="val" \
    --bias_type="religion" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion"

python -m model_selection.compute_reliance_statistics \
    --results_dir="/scratch/yifwang/fairness_x_explainability/decoder_results_jigsaw/val_results/llama_3b_jigsaw_religion_val_100_2/fairness_instruction" \
    --split="val" \
    --bias_type="religion" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion"

python -m model_selection.compute_model_selection_correlation \
    --results_dir="/scratch/yifwang/fairness_x_explainability/decoder_results_jigsaw" \
    --val_results_dir="/scratch/yifwang/fairness_x_explainability/decoder_results_jigsaw/val_results" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion" \
    --experiment_type=decoder \
    --bias_type="religion" \
    --model_type="llama_3b" \
    --num_test_examples=200 \
    --num_val_examples=100 \
    --test_seed=-1 \
    --val_seed=2

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="google/jigsaw_unintended_bias" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="Qwen/Qwen3-4B" \
    --batch_size=1 \
    --max_seq_length=512 \
    --num_examples=100 \
    --output_dir="/scratch/yifwang/fairness_x_explainability/decoder_results_jigsaw/val_results/qwen3_4b_jigsaw_religion_val_100_2/zero_shot" \
    --bias_type="religion"\
    --prompt_type="zero_shot" \
    --shuffle \
    --seed=2

python -m explanation_generation.gen_explanation_decoder \
    --dataset_name="google/jigsaw_unintended_bias" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="Qwen/Qwen3-4B" \
    --num_labels=2 \
    --batch_size=1 \
    --max_length=512 \
    --num_examples=100 \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion" \
    --output_dir="/scratch/yifwang/fairness_x_explainability/decoder_results_jigsaw/val_results/qwen3_4b_jigsaw_religion_val_100_2/zero_shot" \
    --baseline="pad" \
    --bias_type="religion" \
    --prompt_type="zero_shot" \
    --only_predicted_class \
    --shuffle \
    --seed=2

python -m sensitive_attribution_extraction.extract_sensitive_token_attribution \
    --results_dir="/scratch/yifwang/fairness_x_explainability/decoder_results_jigsaw/val_results/qwen3_4b_jigsaw_religion_val_100_2/zero_shot" \
    --split="val" \
    --bias_type="religion" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion"

python -m model_selection.compute_reliance_statistics \
    --results_dir="/scratch/yifwang/fairness_x_explainability/decoder_results_jigsaw/val_results/qwen3_4b_jigsaw_religion_val_100_2/zero_shot" \
    --split="val" \
    --bias_type="religion" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion"

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="google/jigsaw_unintended_bias" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="Qwen/Qwen3-4B" \
    --batch_size=1 \
    --max_seq_length=512 \
    --num_examples=100 \
    --output_dir="/scratch/yifwang/fairness_x_explainability/decoder_results_jigsaw/val_results/qwen3_4b_jigsaw_religion_val_100_2/few_shot" \
    --bias_type="religion"\
    --prompt_type="few_shot" \
    --shuffle \
    --seed=2

python -m explanation_generation.gen_explanation_decoder \
    --dataset_name="google/jigsaw_unintended_bias" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="Qwen/Qwen3-4B" \
    --num_labels=2 \
    --batch_size=1 \
    --max_length=512 \
    --num_examples=100 \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion" \
    --output_dir="/scratch/yifwang/fairness_x_explainability/decoder_results_jigsaw/val_results/qwen3_4b_jigsaw_religion_val_100_2/few_shot" \
    --baseline="pad" \
    --bias_type="religion" \
    --prompt_type="few_shot" \
    --only_predicted_class \
    --shuffle \
    --seed=2

python -m sensitive_attribution_extraction.extract_sensitive_token_attribution \
    --results_dir="/scratch/yifwang/fairness_x_explainability/decoder_results_jigsaw/val_results/qwen3_4b_jigsaw_religion_val_100_2/few_shot" \
    --split="val" \
    --bias_type="religion" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion"

python -m model_selection.compute_reliance_statistics \
    --results_dir="/scratch/yifwang/fairness_x_explainability/decoder_results_jigsaw/val_results/qwen3_4b_jigsaw_religion_val_100_2/few_shot" \
    --split="val" \
    --bias_type="religion" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion"

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="google/jigsaw_unintended_bias" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="Qwen/Qwen3-4B" \
    --batch_size=1 \
    --max_seq_length=512 \
    --num_examples=100 \
    --output_dir="/scratch/yifwang/fairness_x_explainability/decoder_results_jigsaw/val_results/qwen3_4b_jigsaw_religion_val_100_2/fairness_imagination" \
    --bias_type="religion"\
    --prompt_type="fairness_imagination" \
    --shuffle \
    --seed=2

python -m explanation_generation.gen_explanation_decoder \
    --dataset_name="google/jigsaw_unintended_bias" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="Qwen/Qwen3-4B" \
    --num_labels=2 \
    --batch_size=1 \
    --max_length=512 \
    --num_examples=100 \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion" \
    --output_dir="/scratch/yifwang/fairness_x_explainability/decoder_results_jigsaw/val_results/qwen3_4b_jigsaw_religion_val_100_2/fairness_imagination" \
    --baseline="pad" \
    --bias_type="religion" \
    --prompt_type="fairness_imagination" \
    --only_predicted_class \
    --shuffle \
    --seed=2

python -m sensitive_attribution_extraction.extract_sensitive_token_attribution \
    --results_dir="/scratch/yifwang/fairness_x_explainability/decoder_results_jigsaw/val_results/qwen3_4b_jigsaw_religion_val_100_2/fairness_imagination" \
    --split="val" \
    --bias_type="religion" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion"

python -m model_selection.compute_reliance_statistics \
    --results_dir="/scratch/yifwang/fairness_x_explainability/decoder_results_jigsaw/val_results/qwen3_4b_jigsaw_religion_val_100_2/fairness_imagination" \
    --split="val" \
    --bias_type="religion" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion"

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="google/jigsaw_unintended_bias" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="Qwen/Qwen3-4B" \
    --batch_size=1 \
    --max_seq_length=512 \
    --num_examples=100 \
    --output_dir="/scratch/yifwang/fairness_x_explainability/decoder_results_jigsaw/val_results/qwen3_4b_jigsaw_religion_val_100_2/fairness_instruction" \
    --bias_type="religion"\
    --prompt_type="fairness_instruction" \
    --shuffle \
    --seed=2

python -m explanation_generation.gen_explanation_decoder \
    --dataset_name="google/jigsaw_unintended_bias" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="Qwen/Qwen3-4B" \
    --num_labels=2 \
    --batch_size=1 \
    --max_length=512 \
    --num_examples=100 \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion" \
    --output_dir="/scratch/yifwang/fairness_x_explainability/decoder_results_jigsaw/val_results/qwen3_4b_jigsaw_religion_val_100_2/fairness_instruction" \
    --baseline="pad" \
    --bias_type="religion" \
    --prompt_type="fairness_instruction" \
    --only_predicted_class \
    --shuffle \
    --seed=2

python -m sensitive_attribution_extraction.extract_sensitive_token_attribution \
    --results_dir="/scratch/yifwang/fairness_x_explainability/decoder_results_jigsaw/val_results/qwen3_4b_jigsaw_religion_val_100_2/fairness_instruction" \
    --split="val" \
    --bias_type="religion" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion"

python -m model_selection.compute_reliance_statistics \
    --results_dir="/scratch/yifwang/fairness_x_explainability/decoder_results_jigsaw/val_results/qwen3_4b_jigsaw_religion_val_100_2/fairness_instruction" \
    --split="val" \
    --bias_type="religion" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion"

python -m model_selection.compute_model_selection_correlation \
    --results_dir="/scratch/yifwang/fairness_x_explainability/decoder_results_jigsaw" \
    --val_results_dir="/scratch/yifwang/fairness_x_explainability/decoder_results_jigsaw/val_results" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion" \
    --experiment_type=decoder \
    --bias_type="religion" \
    --model_type="qwen3_4b" \
    --num_test_examples=200 \
    --num_val_examples=100 \
    --test_seed=-1 \
    --val_seed=2

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="google/jigsaw_unintended_bias" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="meta-llama/Llama-3.2-3B-Instruct" \
    --batch_size=1 \
    --max_seq_length=512 \
    --num_examples=100 \
    --output_dir="/scratch/yifwang/fairness_x_explainability/decoder_results_jigsaw/val_results/llama_3b_jigsaw_religion_val_100_42/zero_shot" \
    --bias_type="religion"\
    --prompt_type="zero_shot" \
    --shuffle \
    --seed=42

python -m explanation_generation.gen_explanation_decoder \
    --dataset_name="google/jigsaw_unintended_bias" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="meta-llama/Llama-3.2-3B-Instruct" \
    --num_labels=2 \
    --batch_size=1 \
    --max_length=512 \
    --num_examples=100 \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion" \
    --output_dir="/scratch/yifwang/fairness_x_explainability/decoder_results_jigsaw/val_results/llama_3b_jigsaw_religion_val_100_42/zero_shot" \
    --baseline="pad" \
    --bias_type="religion" \
    --prompt_type="zero_shot" \
    --only_predicted_class \
    --shuffle \
    --seed=42

python -m sensitive_attribution_extraction.extract_sensitive_token_attribution \
    --results_dir="/scratch/yifwang/fairness_x_explainability/decoder_results_jigsaw/val_results/llama_3b_jigsaw_religion_val_100_42/zero_shot" \
    --split="val" \
    --bias_type="religion" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion"

python -m model_selection.compute_reliance_statistics \
    --results_dir="/scratch/yifwang/fairness_x_explainability/decoder_results_jigsaw/val_results/llama_3b_jigsaw_religion_val_100_42/zero_shot" \
    --split="val" \
    --bias_type="religion" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion"

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="google/jigsaw_unintended_bias" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="meta-llama/Llama-3.2-3B-Instruct" \
    --batch_size=1 \
    --max_seq_length=512 \
    --num_examples=100 \
    --output_dir="/scratch/yifwang/fairness_x_explainability/decoder_results_jigsaw/val_results/llama_3b_jigsaw_religion_val_100_42/few_shot" \
    --bias_type="religion"\
    --prompt_type="few_shot" \
    --shuffle \
    --seed=42

python -m explanation_generation.gen_explanation_decoder \
    --dataset_name="google/jigsaw_unintended_bias" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="meta-llama/Llama-3.2-3B-Instruct" \
    --num_labels=2 \
    --batch_size=1 \
    --max_length=512 \
    --num_examples=100 \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion" \
    --output_dir="/scratch/yifwang/fairness_x_explainability/decoder_results_jigsaw/val_results/llama_3b_jigsaw_religion_val_100_42/few_shot" \
    --baseline="pad" \
    --bias_type="religion" \
    --prompt_type="few_shot" \
    --only_predicted_class \
    --shuffle \
    --seed=42

python -m sensitive_attribution_extraction.extract_sensitive_token_attribution \
    --results_dir="/scratch/yifwang/fairness_x_explainability/decoder_results_jigsaw/val_results/llama_3b_jigsaw_religion_val_100_42/few_shot" \
    --split="val" \
    --bias_type="religion" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion"

python -m model_selection.compute_reliance_statistics \
    --results_dir="/scratch/yifwang/fairness_x_explainability/decoder_results_jigsaw/val_results/llama_3b_jigsaw_religion_val_100_42/few_shot" \
    --split="val" \
    --bias_type="religion" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion"

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="google/jigsaw_unintended_bias" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="meta-llama/Llama-3.2-3B-Instruct" \
    --batch_size=1 \
    --max_seq_length=512 \
    --num_examples=100 \
    --output_dir="/scratch/yifwang/fairness_x_explainability/decoder_results_jigsaw/val_results/llama_3b_jigsaw_religion_val_100_42/fairness_imagination" \
    --bias_type="religion"\
    --prompt_type="fairness_imagination" \
    --shuffle \
    --seed=42

python -m explanation_generation.gen_explanation_decoder \
    --dataset_name="google/jigsaw_unintended_bias" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="meta-llama/Llama-3.2-3B-Instruct" \
    --num_labels=2 \
    --batch_size=1 \
    --max_length=512 \
    --num_examples=100 \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion" \
    --output_dir="/scratch/yifwang/fairness_x_explainability/decoder_results_jigsaw/val_results/llama_3b_jigsaw_religion_val_100_42/fairness_imagination" \
    --baseline="pad" \
    --bias_type="religion" \
    --prompt_type="fairness_imagination" \
    --only_predicted_class \
    --shuffle \
    --seed=42

python -m sensitive_attribution_extraction.extract_sensitive_token_attribution \
    --results_dir="/scratch/yifwang/fairness_x_explainability/decoder_results_jigsaw/val_results/llama_3b_jigsaw_religion_val_100_42/fairness_imagination" \
    --split="val" \
    --bias_type="religion" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion"

python -m model_selection.compute_reliance_statistics \
    --results_dir="/scratch/yifwang/fairness_x_explainability/decoder_results_jigsaw/val_results/llama_3b_jigsaw_religion_val_100_42/fairness_imagination" \
    --split="val" \
    --bias_type="religion" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion"

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="google/jigsaw_unintended_bias" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="meta-llama/Llama-3.2-3B-Instruct" \
    --batch_size=1 \
    --max_seq_length=512 \
    --num_examples=100 \
    --output_dir="/scratch/yifwang/fairness_x_explainability/decoder_results_jigsaw/val_results/llama_3b_jigsaw_religion_val_100_42/fairness_instruction" \
    --bias_type="religion"\
    --prompt_type="fairness_instruction" \
    --shuffle \
    --seed=42

python -m explanation_generation.gen_explanation_decoder \
    --dataset_name="google/jigsaw_unintended_bias" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="meta-llama/Llama-3.2-3B-Instruct" \
    --num_labels=2 \
    --batch_size=1 \
    --max_length=512 \
    --num_examples=100 \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion" \
    --output_dir="/scratch/yifwang/fairness_x_explainability/decoder_results_jigsaw/val_results/llama_3b_jigsaw_religion_val_100_42/fairness_instruction" \
    --baseline="pad" \
    --bias_type="religion" \
    --prompt_type="fairness_instruction" \
    --only_predicted_class \
    --shuffle \
    --seed=42

python -m sensitive_attribution_extraction.extract_sensitive_token_attribution \
    --results_dir="/scratch/yifwang/fairness_x_explainability/decoder_results_jigsaw/val_results/llama_3b_jigsaw_religion_val_100_42/fairness_instruction" \
    --split="val" \
    --bias_type="religion" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion"

python -m model_selection.compute_reliance_statistics \
    --results_dir="/scratch/yifwang/fairness_x_explainability/decoder_results_jigsaw/val_results/llama_3b_jigsaw_religion_val_100_42/fairness_instruction" \
    --split="val" \
    --bias_type="religion" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion"

python -m model_selection.compute_model_selection_correlation \
    --results_dir="/scratch/yifwang/fairness_x_explainability/decoder_results_jigsaw" \
    --val_results_dir="/scratch/yifwang/fairness_x_explainability/decoder_results_jigsaw/val_results" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion" \
    --experiment_type=decoder \
    --bias_type="religion" \
    --model_type="llama_3b" \
    --num_test_examples=200 \
    --num_val_examples=100 \
    --test_seed=-1 \
    --val_seed=42

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="google/jigsaw_unintended_bias" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="Qwen/Qwen3-4B" \
    --batch_size=1 \
    --max_seq_length=512 \
    --num_examples=100 \
    --output_dir="/scratch/yifwang/fairness_x_explainability/decoder_results_jigsaw/val_results/qwen3_4b_jigsaw_religion_val_100_42/zero_shot" \
    --bias_type="religion"\
    --prompt_type="zero_shot" \
    --shuffle \
    --seed=42

python -m explanation_generation.gen_explanation_decoder \
    --dataset_name="google/jigsaw_unintended_bias" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="Qwen/Qwen3-4B" \
    --num_labels=2 \
    --batch_size=1 \
    --max_length=512 \
    --num_examples=100 \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion" \
    --output_dir="/scratch/yifwang/fairness_x_explainability/decoder_results_jigsaw/val_results/qwen3_4b_jigsaw_religion_val_100_42/zero_shot" \
    --baseline="pad" \
    --bias_type="religion" \
    --prompt_type="zero_shot" \
    --only_predicted_class \
    --shuffle \
    --seed=42

python -m sensitive_attribution_extraction.extract_sensitive_token_attribution \
    --results_dir="/scratch/yifwang/fairness_x_explainability/decoder_results_jigsaw/val_results/qwen3_4b_jigsaw_religion_val_100_42/zero_shot" \
    --split="val" \
    --bias_type="religion" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion"

python -m model_selection.compute_reliance_statistics \
    --results_dir="/scratch/yifwang/fairness_x_explainability/decoder_results_jigsaw/val_results/qwen3_4b_jigsaw_religion_val_100_42/zero_shot" \
    --split="val" \
    --bias_type="religion" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion"

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="google/jigsaw_unintended_bias" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="Qwen/Qwen3-4B" \
    --batch_size=1 \
    --max_seq_length=512 \
    --num_examples=100 \
    --output_dir="/scratch/yifwang/fairness_x_explainability/decoder_results_jigsaw/val_results/qwen3_4b_jigsaw_religion_val_100_42/few_shot" \
    --bias_type="religion"\
    --prompt_type="few_shot" \
    --shuffle \
    --seed=42

python -m explanation_generation.gen_explanation_decoder \
    --dataset_name="google/jigsaw_unintended_bias" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="Qwen/Qwen3-4B" \
    --num_labels=2 \
    --batch_size=1 \
    --max_length=512 \
    --num_examples=100 \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion" \
    --output_dir="/scratch/yifwang/fairness_x_explainability/decoder_results_jigsaw/val_results/qwen3_4b_jigsaw_religion_val_100_42/few_shot" \
    --baseline="pad" \
    --bias_type="religion" \
    --prompt_type="few_shot" \
    --only_predicted_class \
    --shuffle \
    --seed=42

python -m sensitive_attribution_extraction.extract_sensitive_token_attribution \
    --results_dir="/scratch/yifwang/fairness_x_explainability/decoder_results_jigsaw/val_results/qwen3_4b_jigsaw_religion_val_100_42/few_shot" \
    --split="val" \
    --bias_type="religion" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion"

python -m model_selection.compute_reliance_statistics \
    --results_dir="/scratch/yifwang/fairness_x_explainability/decoder_results_jigsaw/val_results/qwen3_4b_jigsaw_religion_val_100_42/few_shot" \
    --split="val" \
    --bias_type="religion" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion"

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="google/jigsaw_unintended_bias" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="Qwen/Qwen3-4B" \
    --batch_size=1 \
    --max_seq_length=512 \
    --num_examples=100 \
    --output_dir="/scratch/yifwang/fairness_x_explainability/decoder_results_jigsaw/val_results/qwen3_4b_jigsaw_religion_val_100_42/fairness_imagination" \
    --bias_type="religion"\
    --prompt_type="fairness_imagination" \
    --shuffle \
    --seed=42

python -m explanation_generation.gen_explanation_decoder \
    --dataset_name="google/jigsaw_unintended_bias" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="Qwen/Qwen3-4B" \
    --num_labels=2 \
    --batch_size=1 \
    --max_length=512 \
    --num_examples=100 \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion" \
    --output_dir="/scratch/yifwang/fairness_x_explainability/decoder_results_jigsaw/val_results/qwen3_4b_jigsaw_religion_val_100_42/fairness_imagination" \
    --baseline="pad" \
    --bias_type="religion" \
    --prompt_type="fairness_imagination" \
    --only_predicted_class \
    --shuffle \
    --seed=42

python -m sensitive_attribution_extraction.extract_sensitive_token_attribution \
    --results_dir="/scratch/yifwang/fairness_x_explainability/decoder_results_jigsaw/val_results/qwen3_4b_jigsaw_religion_val_100_42/fairness_imagination" \
    --split="val" \
    --bias_type="religion" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion"

python -m model_selection.compute_reliance_statistics \
    --results_dir="/scratch/yifwang/fairness_x_explainability/decoder_results_jigsaw/val_results/qwen3_4b_jigsaw_religion_val_100_42/fairness_imagination" \
    --split="val" \
    --bias_type="religion" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion"

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="google/jigsaw_unintended_bias" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="Qwen/Qwen3-4B" \
    --batch_size=1 \
    --max_seq_length=512 \
    --num_examples=100 \
    --output_dir="/scratch/yifwang/fairness_x_explainability/decoder_results_jigsaw/val_results/qwen3_4b_jigsaw_religion_val_100_42/fairness_instruction" \
    --bias_type="religion"\
    --prompt_type="fairness_instruction" \
    --shuffle \
    --seed=42

python -m explanation_generation.gen_explanation_decoder \
    --dataset_name="google/jigsaw_unintended_bias" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="Qwen/Qwen3-4B" \
    --num_labels=2 \
    --batch_size=1 \
    --max_length=512 \
    --num_examples=100 \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion" \
    --output_dir="/scratch/yifwang/fairness_x_explainability/decoder_results_jigsaw/val_results/qwen3_4b_jigsaw_religion_val_100_42/fairness_instruction" \
    --baseline="pad" \
    --bias_type="religion" \
    --prompt_type="fairness_instruction" \
    --only_predicted_class \
    --shuffle \
    --seed=42

python -m sensitive_attribution_extraction.extract_sensitive_token_attribution \
    --results_dir="/scratch/yifwang/fairness_x_explainability/decoder_results_jigsaw/val_results/qwen3_4b_jigsaw_religion_val_100_42/fairness_instruction" \
    --split="val" \
    --bias_type="religion" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion"

python -m model_selection.compute_reliance_statistics \
    --results_dir="/scratch/yifwang/fairness_x_explainability/decoder_results_jigsaw/val_results/qwen3_4b_jigsaw_religion_val_100_42/fairness_instruction" \
    --split="val" \
    --bias_type="religion" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion"

python -m model_selection.compute_model_selection_correlation \
    --results_dir="/scratch/yifwang/fairness_x_explainability/decoder_results_jigsaw" \
    --val_results_dir="/scratch/yifwang/fairness_x_explainability/decoder_results_jigsaw/val_results" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion" \
    --experiment_type=decoder \
    --bias_type="religion" \
    --model_type="qwen3_4b" \
    --num_test_examples=200 \
    --num_val_examples=100 \
    --test_seed=-1 \
    --val_seed=42

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="google/jigsaw_unintended_bias" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="meta-llama/Llama-3.2-3B-Instruct" \
    --batch_size=1 \
    --max_seq_length=512 \
    --num_examples=200 \
    --output_dir="/scratch/yifwang/fairness_x_explainability/decoder_results_jigsaw/val_results/llama_3b_jigsaw_religion_val_200_42/zero_shot" \
    --bias_type="religion"\
    --prompt_type="zero_shot" \
    --shuffle \
    --seed=42

python -m explanation_generation.gen_explanation_decoder \
    --dataset_name="google/jigsaw_unintended_bias" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="meta-llama/Llama-3.2-3B-Instruct" \
    --num_labels=2 \
    --batch_size=1 \
    --max_length=512 \
    --num_examples=200 \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion" \
    --output_dir="/scratch/yifwang/fairness_x_explainability/decoder_results_jigsaw/val_results/llama_3b_jigsaw_religion_val_200_42/zero_shot" \
    --baseline="pad" \
    --bias_type="religion" \
    --prompt_type="zero_shot" \
    --only_predicted_class \
    --shuffle \
    --seed=42

python -m sensitive_attribution_extraction.extract_sensitive_token_attribution \
    --results_dir="/scratch/yifwang/fairness_x_explainability/decoder_results_jigsaw/val_results/llama_3b_jigsaw_religion_val_200_42/zero_shot" \
    --split="val" \
    --bias_type="religion" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion"

python -m model_selection.compute_reliance_statistics \
    --results_dir="/scratch/yifwang/fairness_x_explainability/decoder_results_jigsaw/val_results/llama_3b_jigsaw_religion_val_200_42/zero_shot" \
    --split="val" \
    --bias_type="religion" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion"

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="google/jigsaw_unintended_bias" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="meta-llama/Llama-3.2-3B-Instruct" \
    --batch_size=1 \
    --max_seq_length=512 \
    --num_examples=200 \
    --output_dir="/scratch/yifwang/fairness_x_explainability/decoder_results_jigsaw/val_results/llama_3b_jigsaw_religion_val_200_42/few_shot" \
    --bias_type="religion"\
    --prompt_type="few_shot" \
    --shuffle \
    --seed=42

python -m explanation_generation.gen_explanation_decoder \
    --dataset_name="google/jigsaw_unintended_bias" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="meta-llama/Llama-3.2-3B-Instruct" \
    --num_labels=2 \
    --batch_size=1 \
    --max_length=512 \
    --num_examples=200 \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion" \
    --output_dir="/scratch/yifwang/fairness_x_explainability/decoder_results_jigsaw/val_results/llama_3b_jigsaw_religion_val_200_42/few_shot" \
    --baseline="pad" \
    --bias_type="religion" \
    --prompt_type="few_shot" \
    --only_predicted_class \
    --shuffle \
    --seed=42

python -m sensitive_attribution_extraction.extract_sensitive_token_attribution \
    --results_dir="/scratch/yifwang/fairness_x_explainability/decoder_results_jigsaw/val_results/llama_3b_jigsaw_religion_val_200_42/few_shot" \
    --split="val" \
    --bias_type="religion" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion"

python -m model_selection.compute_reliance_statistics \
    --results_dir="/scratch/yifwang/fairness_x_explainability/decoder_results_jigsaw/val_results/llama_3b_jigsaw_religion_val_200_42/few_shot" \
    --split="val" \
    --bias_type="religion" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion"

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="google/jigsaw_unintended_bias" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="meta-llama/Llama-3.2-3B-Instruct" \
    --batch_size=1 \
    --max_seq_length=512 \
    --num_examples=200 \
    --output_dir="/scratch/yifwang/fairness_x_explainability/decoder_results_jigsaw/val_results/llama_3b_jigsaw_religion_val_200_42/fairness_imagination" \
    --bias_type="religion"\
    --prompt_type="fairness_imagination" \
    --shuffle \
    --seed=42

python -m explanation_generation.gen_explanation_decoder \
    --dataset_name="google/jigsaw_unintended_bias" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="meta-llama/Llama-3.2-3B-Instruct" \
    --num_labels=2 \
    --batch_size=1 \
    --max_length=512 \
    --num_examples=200 \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion" \
    --output_dir="/scratch/yifwang/fairness_x_explainability/decoder_results_jigsaw/val_results/llama_3b_jigsaw_religion_val_200_42/fairness_imagination" \
    --baseline="pad" \
    --bias_type="religion" \
    --prompt_type="fairness_imagination" \
    --only_predicted_class \
    --shuffle \
    --seed=42

python -m sensitive_attribution_extraction.extract_sensitive_token_attribution \
    --results_dir="/scratch/yifwang/fairness_x_explainability/decoder_results_jigsaw/val_results/llama_3b_jigsaw_religion_val_200_42/fairness_imagination" \
    --split="val" \
    --bias_type="religion" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion"

python -m model_selection.compute_reliance_statistics \
    --results_dir="/scratch/yifwang/fairness_x_explainability/decoder_results_jigsaw/val_results/llama_3b_jigsaw_religion_val_200_42/fairness_imagination" \
    --split="val" \
    --bias_type="religion" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion"

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="google/jigsaw_unintended_bias" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="meta-llama/Llama-3.2-3B-Instruct" \
    --batch_size=1 \
    --max_seq_length=512 \
    --num_examples=200 \
    --output_dir="/scratch/yifwang/fairness_x_explainability/decoder_results_jigsaw/val_results/llama_3b_jigsaw_religion_val_200_42/fairness_instruction" \
    --bias_type="religion"\
    --prompt_type="fairness_instruction" \
    --shuffle \
    --seed=42

python -m explanation_generation.gen_explanation_decoder \
    --dataset_name="google/jigsaw_unintended_bias" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="meta-llama/Llama-3.2-3B-Instruct" \
    --num_labels=2 \
    --batch_size=1 \
    --max_length=512 \
    --num_examples=200 \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion" \
    --output_dir="/scratch/yifwang/fairness_x_explainability/decoder_results_jigsaw/val_results/llama_3b_jigsaw_religion_val_200_42/fairness_instruction" \
    --baseline="pad" \
    --bias_type="religion" \
    --prompt_type="fairness_instruction" \
    --only_predicted_class \
    --shuffle \
    --seed=42

python -m sensitive_attribution_extraction.extract_sensitive_token_attribution \
    --results_dir="/scratch/yifwang/fairness_x_explainability/decoder_results_jigsaw/val_results/llama_3b_jigsaw_religion_val_200_42/fairness_instruction" \
    --split="val" \
    --bias_type="religion" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion"

python -m model_selection.compute_reliance_statistics \
    --results_dir="/scratch/yifwang/fairness_x_explainability/decoder_results_jigsaw/val_results/llama_3b_jigsaw_religion_val_200_42/fairness_instruction" \
    --split="val" \
    --bias_type="religion" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion"

python -m model_selection.compute_model_selection_correlation \
    --results_dir="/scratch/yifwang/fairness_x_explainability/decoder_results_jigsaw" \
    --val_results_dir="/scratch/yifwang/fairness_x_explainability/decoder_results_jigsaw/val_results" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion" \
    --experiment_type=decoder \
    --bias_type="religion" \
    --model_type="llama_3b" \
    --num_test_examples=200 \
    --num_val_examples=200 \
    --test_seed=-1 \
    --val_seed=42

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="google/jigsaw_unintended_bias" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="Qwen/Qwen3-4B" \
    --batch_size=1 \
    --max_seq_length=512 \
    --num_examples=200 \
    --output_dir="/scratch/yifwang/fairness_x_explainability/decoder_results_jigsaw/val_results/qwen3_4b_jigsaw_religion_val_200_42/zero_shot" \
    --bias_type="religion"\
    --prompt_type="zero_shot" \
    --shuffle \
    --seed=42

python -m explanation_generation.gen_explanation_decoder \
    --dataset_name="google/jigsaw_unintended_bias" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="Qwen/Qwen3-4B" \
    --num_labels=2 \
    --batch_size=1 \
    --max_length=512 \
    --num_examples=200 \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion" \
    --output_dir="/scratch/yifwang/fairness_x_explainability/decoder_results_jigsaw/val_results/qwen3_4b_jigsaw_religion_val_200_42/zero_shot" \
    --baseline="pad" \
    --bias_type="religion" \
    --prompt_type="zero_shot" \
    --only_predicted_class \
    --shuffle \
    --seed=42

python -m sensitive_attribution_extraction.extract_sensitive_token_attribution \
    --results_dir="/scratch/yifwang/fairness_x_explainability/decoder_results_jigsaw/val_results/qwen3_4b_jigsaw_religion_val_200_42/zero_shot" \
    --split="val" \
    --bias_type="religion" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion"

python -m model_selection.compute_reliance_statistics \
    --results_dir="/scratch/yifwang/fairness_x_explainability/decoder_results_jigsaw/val_results/qwen3_4b_jigsaw_religion_val_200_42/zero_shot" \
    --split="val" \
    --bias_type="religion" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion"

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="google/jigsaw_unintended_bias" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="Qwen/Qwen3-4B" \
    --batch_size=1 \
    --max_seq_length=512 \
    --num_examples=200 \
    --output_dir="/scratch/yifwang/fairness_x_explainability/decoder_results_jigsaw/val_results/qwen3_4b_jigsaw_religion_val_200_42/few_shot" \
    --bias_type="religion"\
    --prompt_type="few_shot" \
    --shuffle \
    --seed=42

python -m explanation_generation.gen_explanation_decoder \
    --dataset_name="google/jigsaw_unintended_bias" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="Qwen/Qwen3-4B" \
    --num_labels=2 \
    --batch_size=1 \
    --max_length=512 \
    --num_examples=200 \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion" \
    --output_dir="/scratch/yifwang/fairness_x_explainability/decoder_results_jigsaw/val_results/qwen3_4b_jigsaw_religion_val_200_42/few_shot" \
    --baseline="pad" \
    --bias_type="religion" \
    --prompt_type="few_shot" \
    --only_predicted_class \
    --shuffle \
    --seed=42

python -m sensitive_attribution_extraction.extract_sensitive_token_attribution \
    --results_dir="/scratch/yifwang/fairness_x_explainability/decoder_results_jigsaw/val_results/qwen3_4b_jigsaw_religion_val_200_42/few_shot" \
    --split="val" \
    --bias_type="religion" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion"

python -m model_selection.compute_reliance_statistics \
    --results_dir="/scratch/yifwang/fairness_x_explainability/decoder_results_jigsaw/val_results/qwen3_4b_jigsaw_religion_val_200_42/few_shot" \
    --split="val" \
    --bias_type="religion" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion"

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="google/jigsaw_unintended_bias" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="Qwen/Qwen3-4B" \
    --batch_size=1 \
    --max_seq_length=512 \
    --num_examples=200 \
    --output_dir="/scratch/yifwang/fairness_x_explainability/decoder_results_jigsaw/val_results/qwen3_4b_jigsaw_religion_val_200_42/fairness_imagination" \
    --bias_type="religion"\
    --prompt_type="fairness_imagination" \
    --shuffle \
    --seed=42

python -m explanation_generation.gen_explanation_decoder \
    --dataset_name="google/jigsaw_unintended_bias" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="Qwen/Qwen3-4B" \
    --num_labels=2 \
    --batch_size=1 \
    --max_length=512 \
    --num_examples=200 \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion" \
    --output_dir="/scratch/yifwang/fairness_x_explainability/decoder_results_jigsaw/val_results/qwen3_4b_jigsaw_religion_val_200_42/fairness_imagination" \
    --baseline="pad" \
    --bias_type="religion" \
    --prompt_type="fairness_imagination" \
    --only_predicted_class \
    --shuffle \
    --seed=42

python -m sensitive_attribution_extraction.extract_sensitive_token_attribution \
    --results_dir="/scratch/yifwang/fairness_x_explainability/decoder_results_jigsaw/val_results/qwen3_4b_jigsaw_religion_val_200_42/fairness_imagination" \
    --split="val" \
    --bias_type="religion" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion"

python -m model_selection.compute_reliance_statistics \
    --results_dir="/scratch/yifwang/fairness_x_explainability/decoder_results_jigsaw/val_results/qwen3_4b_jigsaw_religion_val_200_42/fairness_imagination" \
    --split="val" \
    --bias_type="religion" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion"

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="google/jigsaw_unintended_bias" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="Qwen/Qwen3-4B" \
    --batch_size=1 \
    --max_seq_length=512 \
    --num_examples=200 \
    --output_dir="/scratch/yifwang/fairness_x_explainability/decoder_results_jigsaw/val_results/qwen3_4b_jigsaw_religion_val_200_42/fairness_instruction" \
    --bias_type="religion"\
    --prompt_type="fairness_instruction" \
    --shuffle \
    --seed=42

python -m explanation_generation.gen_explanation_decoder \
    --dataset_name="google/jigsaw_unintended_bias" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="Qwen/Qwen3-4B" \
    --num_labels=2 \
    --batch_size=1 \
    --max_length=512 \
    --num_examples=200 \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion" \
    --output_dir="/scratch/yifwang/fairness_x_explainability/decoder_results_jigsaw/val_results/qwen3_4b_jigsaw_religion_val_200_42/fairness_instruction" \
    --baseline="pad" \
    --bias_type="religion" \
    --prompt_type="fairness_instruction" \
    --only_predicted_class \
    --shuffle \
    --seed=42

python -m sensitive_attribution_extraction.extract_sensitive_token_attribution \
    --results_dir="/scratch/yifwang/fairness_x_explainability/decoder_results_jigsaw/val_results/qwen3_4b_jigsaw_religion_val_200_42/fairness_instruction" \
    --split="val" \
    --bias_type="religion" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion"

python -m model_selection.compute_reliance_statistics \
    --results_dir="/scratch/yifwang/fairness_x_explainability/decoder_results_jigsaw/val_results/qwen3_4b_jigsaw_religion_val_200_42/fairness_instruction" \
    --split="val" \
    --bias_type="religion" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion"

python -m model_selection.compute_model_selection_correlation \
    --results_dir="/scratch/yifwang/fairness_x_explainability/decoder_results_jigsaw" \
    --val_results_dir="/scratch/yifwang/fairness_x_explainability/decoder_results_jigsaw/val_results" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion" \
    --experiment_type=decoder \
    --bias_type="religion" \
    --model_type="qwen3_4b" \
    --num_test_examples=200 \
    --num_val_examples=200 \
    --test_seed=-1 \
    --val_seed=42


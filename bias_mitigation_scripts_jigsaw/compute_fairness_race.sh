#!/bin/bash

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="google/jigsaw_unintended_bias" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/fairness_x_explainability/explanation_debiased_models_jigsaw/bert_jigsaw_race_1/Saliency/L1_100.0" \
    --batch_size=64 \
    --max_seq_length=512 \
    --num_examples=200 \
    --output_dir="/scratch/yifwang/fairness_x_explainability/bias_mitigation_results_jigsaw/bert_jigsaw_race_race_val_200_1/Saliency/L1_100.0" \
    --bias_type="race" 

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="google/jigsaw_unintended_bias" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/fairness_x_explainability/explanation_debiased_models_jigsaw/bert_jigsaw_race_2/Saliency/L1_100.0" \
    --batch_size=64 \
    --max_seq_length=512 \
    --num_examples=200 \
    --output_dir="/scratch/yifwang/fairness_x_explainability/bias_mitigation_results_jigsaw/bert_jigsaw_race_race_val_200_2/Saliency/L1_100.0" \
    --bias_type="race" 

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="google/jigsaw_unintended_bias" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/fairness_x_explainability/explanation_debiased_models_jigsaw/bert_jigsaw_race_1/Saliency/L2_100.0" \
    --batch_size=64 \
    --max_seq_length=512 \
    --num_examples=200 \
    --output_dir="/scratch/yifwang/fairness_x_explainability/bias_mitigation_results_jigsaw/bert_jigsaw_race_race_val_200_1/Saliency/L2_100.0" \
    --bias_type="race" 

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="google/jigsaw_unintended_bias" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/fairness_x_explainability/explanation_debiased_models_jigsaw/bert_jigsaw_race_2/Saliency/L2_100.0" \
    --batch_size=64 \
    --max_seq_length=512 \
    --num_examples=200 \
    --output_dir="/scratch/yifwang/fairness_x_explainability/bias_mitigation_results_jigsaw/bert_jigsaw_race_race_val_200_2/Saliency/L2_100.0" \
    --bias_type="race" 

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="google/jigsaw_unintended_bias" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/fairness_x_explainability/explanation_debiased_models_jigsaw/bert_jigsaw_race_1/Saliency/L1_10.0" \
    --batch_size=64 \
    --max_seq_length=512 \
    --num_examples=200 \
    --output_dir="/scratch/yifwang/fairness_x_explainability/bias_mitigation_results_jigsaw/bert_jigsaw_race_race_val_200_1/Saliency/L1_10.0" \
    --bias_type="race" 

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="google/jigsaw_unintended_bias" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/fairness_x_explainability/explanation_debiased_models_jigsaw/bert_jigsaw_race_2/Saliency/L1_10.0" \
    --batch_size=64 \
    --max_seq_length=512 \
    --num_examples=200 \
    --output_dir="/scratch/yifwang/fairness_x_explainability/bias_mitigation_results_jigsaw/bert_jigsaw_race_race_val_200_2/Saliency/L1_10.0" \
    --bias_type="race" 

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="google/jigsaw_unintended_bias" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/fairness_x_explainability/explanation_debiased_models_jigsaw/bert_jigsaw_race_1/Saliency/L2_10.0" \
    --batch_size=64 \
    --max_seq_length=512 \
    --num_examples=200 \
    --output_dir="/scratch/yifwang/fairness_x_explainability/bias_mitigation_results_jigsaw/bert_jigsaw_race_race_val_200_1/Saliency/L2_10.0" \
    --bias_type="race" 

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="google/jigsaw_unintended_bias" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/fairness_x_explainability/explanation_debiased_models_jigsaw/bert_jigsaw_race_2/Saliency/L2_10.0" \
    --batch_size=64 \
    --max_seq_length=512 \
    --num_examples=200 \
    --output_dir="/scratch/yifwang/fairness_x_explainability/bias_mitigation_results_jigsaw/bert_jigsaw_race_race_val_200_2/Saliency/L2_10.0" \
    --bias_type="race" 

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="google/jigsaw_unintended_bias" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/fairness_x_explainability/explanation_debiased_models_jigsaw/bert_jigsaw_race_1/Saliency/L1_1.0" \
    --batch_size=64 \
    --max_seq_length=512 \
    --num_examples=200 \
    --output_dir="/scratch/yifwang/fairness_x_explainability/bias_mitigation_results_jigsaw/bert_jigsaw_race_race_val_200_1/Saliency/L1_1.0" \
    --bias_type="race" 

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="google/jigsaw_unintended_bias" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/fairness_x_explainability/explanation_debiased_models_jigsaw/bert_jigsaw_race_2/Saliency/L1_1.0" \
    --batch_size=64 \
    --max_seq_length=512 \
    --num_examples=200 \
    --output_dir="/scratch/yifwang/fairness_x_explainability/bias_mitigation_results_jigsaw/bert_jigsaw_race_race_val_200_2/Saliency/L1_1.0" \
    --bias_type="race" 

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="google/jigsaw_unintended_bias" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/fairness_x_explainability/explanation_debiased_models_jigsaw/bert_jigsaw_race_1/Saliency/L2_1.0" \
    --batch_size=64 \
    --max_seq_length=512 \
    --num_examples=200 \
    --output_dir="/scratch/yifwang/fairness_x_explainability/bias_mitigation_results_jigsaw/bert_jigsaw_race_race_val_200_1/Saliency/L2_1.0" \
    --bias_type="race" 

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="google/jigsaw_unintended_bias" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/fairness_x_explainability/explanation_debiased_models_jigsaw/bert_jigsaw_race_2/Saliency/L2_1.0" \
    --batch_size=64 \
    --max_seq_length=512 \
    --num_examples=200 \
    --output_dir="/scratch/yifwang/fairness_x_explainability/bias_mitigation_results_jigsaw/bert_jigsaw_race_race_val_200_2/Saliency/L2_1.0" \
    --bias_type="race" 

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="google/jigsaw_unintended_bias" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/fairness_x_explainability/explanation_debiased_models_jigsaw/bert_jigsaw_race_1/Saliency/L1_0.1" \
    --batch_size=64 \
    --max_seq_length=512 \
    --num_examples=200 \
    --output_dir="/scratch/yifwang/fairness_x_explainability/bias_mitigation_results_jigsaw/bert_jigsaw_race_race_val_200_1/Saliency/L1_0.1" \
    --bias_type="race" 

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="google/jigsaw_unintended_bias" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/fairness_x_explainability/explanation_debiased_models_jigsaw/bert_jigsaw_race_2/Saliency/L1_0.1" \
    --batch_size=64 \
    --max_seq_length=512 \
    --num_examples=200 \
    --output_dir="/scratch/yifwang/fairness_x_explainability/bias_mitigation_results_jigsaw/bert_jigsaw_race_race_val_200_2/Saliency/L1_0.1" \
    --bias_type="race" 

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="google/jigsaw_unintended_bias" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/fairness_x_explainability/explanation_debiased_models_jigsaw/bert_jigsaw_race_1/Saliency/L2_0.1" \
    --batch_size=64 \
    --max_seq_length=512 \
    --num_examples=200 \
    --output_dir="/scratch/yifwang/fairness_x_explainability/bias_mitigation_results_jigsaw/bert_jigsaw_race_race_val_200_1/Saliency/L2_0.1" \
    --bias_type="race" 

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="google/jigsaw_unintended_bias" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/fairness_x_explainability/explanation_debiased_models_jigsaw/bert_jigsaw_race_2/Saliency/L2_0.1" \
    --batch_size=64 \
    --max_seq_length=512 \
    --num_examples=200 \
    --output_dir="/scratch/yifwang/fairness_x_explainability/bias_mitigation_results_jigsaw/bert_jigsaw_race_race_val_200_2/Saliency/L2_0.1" \
    --bias_type="race" 

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="google/jigsaw_unintended_bias" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/fairness_x_explainability/explanation_debiased_models_jigsaw/bert_jigsaw_race_1/Saliency/L1_0.01" \
    --batch_size=64 \
    --max_seq_length=512 \
    --num_examples=200 \
    --output_dir="/scratch/yifwang/fairness_x_explainability/bias_mitigation_results_jigsaw/bert_jigsaw_race_race_val_200_1/Saliency/L1_0.01" \
    --bias_type="race" 

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="google/jigsaw_unintended_bias" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/fairness_x_explainability/explanation_debiased_models_jigsaw/bert_jigsaw_race_2/Saliency/L1_0.01" \
    --batch_size=64 \
    --max_seq_length=512 \
    --num_examples=200 \
    --output_dir="/scratch/yifwang/fairness_x_explainability/bias_mitigation_results_jigsaw/bert_jigsaw_race_race_val_200_2/Saliency/L1_0.01" \
    --bias_type="race" 

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="google/jigsaw_unintended_bias" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/fairness_x_explainability/explanation_debiased_models_jigsaw/bert_jigsaw_race_1/Saliency/L2_0.01" \
    --batch_size=64 \
    --max_seq_length=512 \
    --num_examples=200 \
    --output_dir="/scratch/yifwang/fairness_x_explainability/bias_mitigation_results_jigsaw/bert_jigsaw_race_race_val_200_1/Saliency/L2_0.01" \
    --bias_type="race" 

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="google/jigsaw_unintended_bias" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/fairness_x_explainability/explanation_debiased_models_jigsaw/bert_jigsaw_race_2/Saliency/L2_0.01" \
    --batch_size=64 \
    --max_seq_length=512 \
    --num_examples=200 \
    --output_dir="/scratch/yifwang/fairness_x_explainability/bias_mitigation_results_jigsaw/bert_jigsaw_race_race_val_200_2/Saliency/L2_0.01" \
    --bias_type="race" 

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="google/jigsaw_unintended_bias" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/fairness_x_explainability/explanation_debiased_models_jigsaw/bert_jigsaw_race_1/InputXGradient/L1_100.0" \
    --batch_size=64 \
    --max_seq_length=512 \
    --num_examples=200 \
    --output_dir="/scratch/yifwang/fairness_x_explainability/bias_mitigation_results_jigsaw/bert_jigsaw_race_race_val_200_1/InputXGradient/L1_100.0" \
    --bias_type="race" 

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="google/jigsaw_unintended_bias" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/fairness_x_explainability/explanation_debiased_models_jigsaw/bert_jigsaw_race_2/InputXGradient/L1_100.0" \
    --batch_size=64 \
    --max_seq_length=512 \
    --num_examples=200 \
    --output_dir="/scratch/yifwang/fairness_x_explainability/bias_mitigation_results_jigsaw/bert_jigsaw_race_race_val_200_2/InputXGradient/L1_100.0" \
    --bias_type="race" 

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="google/jigsaw_unintended_bias" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/fairness_x_explainability/explanation_debiased_models_jigsaw/bert_jigsaw_race_1/InputXGradient/L2_100.0" \
    --batch_size=64 \
    --max_seq_length=512 \
    --num_examples=200 \
    --output_dir="/scratch/yifwang/fairness_x_explainability/bias_mitigation_results_jigsaw/bert_jigsaw_race_race_val_200_1/InputXGradient/L2_100.0" \
    --bias_type="race" 

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="google/jigsaw_unintended_bias" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/fairness_x_explainability/explanation_debiased_models_jigsaw/bert_jigsaw_race_2/InputXGradient/L2_100.0" \
    --batch_size=64 \
    --max_seq_length=512 \
    --num_examples=200 \
    --output_dir="/scratch/yifwang/fairness_x_explainability/bias_mitigation_results_jigsaw/bert_jigsaw_race_race_val_200_2/InputXGradient/L2_100.0" \
    --bias_type="race" 

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="google/jigsaw_unintended_bias" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/fairness_x_explainability/explanation_debiased_models_jigsaw/bert_jigsaw_race_1/InputXGradient/L1_10.0" \
    --batch_size=64 \
    --max_seq_length=512 \
    --num_examples=200 \
    --output_dir="/scratch/yifwang/fairness_x_explainability/bias_mitigation_results_jigsaw/bert_jigsaw_race_race_val_200_1/InputXGradient/L1_10.0" \
    --bias_type="race" 

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="google/jigsaw_unintended_bias" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/fairness_x_explainability/explanation_debiased_models_jigsaw/bert_jigsaw_race_2/InputXGradient/L1_10.0" \
    --batch_size=64 \
    --max_seq_length=512 \
    --num_examples=200 \
    --output_dir="/scratch/yifwang/fairness_x_explainability/bias_mitigation_results_jigsaw/bert_jigsaw_race_race_val_200_2/InputXGradient/L1_10.0" \
    --bias_type="race" 

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="google/jigsaw_unintended_bias" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/fairness_x_explainability/explanation_debiased_models_jigsaw/bert_jigsaw_race_1/InputXGradient/L2_10.0" \
    --batch_size=64 \
    --max_seq_length=512 \
    --num_examples=200 \
    --output_dir="/scratch/yifwang/fairness_x_explainability/bias_mitigation_results_jigsaw/bert_jigsaw_race_race_val_200_1/InputXGradient/L2_10.0" \
    --bias_type="race" 

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="google/jigsaw_unintended_bias" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/fairness_x_explainability/explanation_debiased_models_jigsaw/bert_jigsaw_race_2/InputXGradient/L2_10.0" \
    --batch_size=64 \
    --max_seq_length=512 \
    --num_examples=200 \
    --output_dir="/scratch/yifwang/fairness_x_explainability/bias_mitigation_results_jigsaw/bert_jigsaw_race_race_val_200_2/InputXGradient/L2_10.0" \
    --bias_type="race" 

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="google/jigsaw_unintended_bias" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/fairness_x_explainability/explanation_debiased_models_jigsaw/bert_jigsaw_race_1/InputXGradient/L1_1.0" \
    --batch_size=64 \
    --max_seq_length=512 \
    --num_examples=200 \
    --output_dir="/scratch/yifwang/fairness_x_explainability/bias_mitigation_results_jigsaw/bert_jigsaw_race_race_val_200_1/InputXGradient/L1_1.0" \
    --bias_type="race" 

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="google/jigsaw_unintended_bias" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/fairness_x_explainability/explanation_debiased_models_jigsaw/bert_jigsaw_race_2/InputXGradient/L1_1.0" \
    --batch_size=64 \
    --max_seq_length=512 \
    --num_examples=200 \
    --output_dir="/scratch/yifwang/fairness_x_explainability/bias_mitigation_results_jigsaw/bert_jigsaw_race_race_val_200_2/InputXGradient/L1_1.0" \
    --bias_type="race" 

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="google/jigsaw_unintended_bias" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/fairness_x_explainability/explanation_debiased_models_jigsaw/bert_jigsaw_race_1/InputXGradient/L2_1.0" \
    --batch_size=64 \
    --max_seq_length=512 \
    --num_examples=200 \
    --output_dir="/scratch/yifwang/fairness_x_explainability/bias_mitigation_results_jigsaw/bert_jigsaw_race_race_val_200_1/InputXGradient/L2_1.0" \
    --bias_type="race" 

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="google/jigsaw_unintended_bias" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/fairness_x_explainability/explanation_debiased_models_jigsaw/bert_jigsaw_race_2/InputXGradient/L2_1.0" \
    --batch_size=64 \
    --max_seq_length=512 \
    --num_examples=200 \
    --output_dir="/scratch/yifwang/fairness_x_explainability/bias_mitigation_results_jigsaw/bert_jigsaw_race_race_val_200_2/InputXGradient/L2_1.0" \
    --bias_type="race" 

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="google/jigsaw_unintended_bias" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/fairness_x_explainability/explanation_debiased_models_jigsaw/bert_jigsaw_race_1/InputXGradient/L1_0.1" \
    --batch_size=64 \
    --max_seq_length=512 \
    --num_examples=200 \
    --output_dir="/scratch/yifwang/fairness_x_explainability/bias_mitigation_results_jigsaw/bert_jigsaw_race_race_val_200_1/InputXGradient/L1_0.1" \
    --bias_type="race" 

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="google/jigsaw_unintended_bias" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/fairness_x_explainability/explanation_debiased_models_jigsaw/bert_jigsaw_race_2/InputXGradient/L1_0.1" \
    --batch_size=64 \
    --max_seq_length=512 \
    --num_examples=200 \
    --output_dir="/scratch/yifwang/fairness_x_explainability/bias_mitigation_results_jigsaw/bert_jigsaw_race_race_val_200_2/InputXGradient/L1_0.1" \
    --bias_type="race" 

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="google/jigsaw_unintended_bias" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/fairness_x_explainability/explanation_debiased_models_jigsaw/bert_jigsaw_race_1/InputXGradient/L2_0.1" \
    --batch_size=64 \
    --max_seq_length=512 \
    --num_examples=200 \
    --output_dir="/scratch/yifwang/fairness_x_explainability/bias_mitigation_results_jigsaw/bert_jigsaw_race_race_val_200_1/InputXGradient/L2_0.1" \
    --bias_type="race" 

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="google/jigsaw_unintended_bias" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/fairness_x_explainability/explanation_debiased_models_jigsaw/bert_jigsaw_race_2/InputXGradient/L2_0.1" \
    --batch_size=64 \
    --max_seq_length=512 \
    --num_examples=200 \
    --output_dir="/scratch/yifwang/fairness_x_explainability/bias_mitigation_results_jigsaw/bert_jigsaw_race_race_val_200_2/InputXGradient/L2_0.1" \
    --bias_type="race" 

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="google/jigsaw_unintended_bias" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/fairness_x_explainability/explanation_debiased_models_jigsaw/bert_jigsaw_race_1/InputXGradient/L1_0.01" \
    --batch_size=64 \
    --max_seq_length=512 \
    --num_examples=200 \
    --output_dir="/scratch/yifwang/fairness_x_explainability/bias_mitigation_results_jigsaw/bert_jigsaw_race_race_val_200_1/InputXGradient/L1_0.01" \
    --bias_type="race" 

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="google/jigsaw_unintended_bias" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/fairness_x_explainability/explanation_debiased_models_jigsaw/bert_jigsaw_race_2/InputXGradient/L1_0.01" \
    --batch_size=64 \
    --max_seq_length=512 \
    --num_examples=200 \
    --output_dir="/scratch/yifwang/fairness_x_explainability/bias_mitigation_results_jigsaw/bert_jigsaw_race_race_val_200_2/InputXGradient/L1_0.01" \
    --bias_type="race" 

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="google/jigsaw_unintended_bias" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/fairness_x_explainability/explanation_debiased_models_jigsaw/bert_jigsaw_race_1/InputXGradient/L2_0.01" \
    --batch_size=64 \
    --max_seq_length=512 \
    --num_examples=200 \
    --output_dir="/scratch/yifwang/fairness_x_explainability/bias_mitigation_results_jigsaw/bert_jigsaw_race_race_val_200_1/InputXGradient/L2_0.01" \
    --bias_type="race" 

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="google/jigsaw_unintended_bias" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/fairness_x_explainability/explanation_debiased_models_jigsaw/bert_jigsaw_race_2/InputXGradient/L2_0.01" \
    --batch_size=64 \
    --max_seq_length=512 \
    --num_examples=200 \
    --output_dir="/scratch/yifwang/fairness_x_explainability/bias_mitigation_results_jigsaw/bert_jigsaw_race_race_val_200_2/InputXGradient/L2_0.01" \
    --bias_type="race" 

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="google/jigsaw_unintended_bias" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/fairness_x_explainability/explanation_debiased_models_jigsaw/bert_jigsaw_race_1/raw_attention/100.0" \
    --batch_size=64 \
    --max_seq_length=512 \
    --num_examples=200 \
    --output_dir="/scratch/yifwang/fairness_x_explainability/bias_mitigation_results_jigsaw/bert_jigsaw_race_race_val_200_1/raw_attention/100.0" \
    --bias_type="race" 

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="google/jigsaw_unintended_bias" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/fairness_x_explainability/explanation_debiased_models_jigsaw/bert_jigsaw_race_2/raw_attention/100.0" \
    --batch_size=64 \
    --max_seq_length=512 \
    --num_examples=200 \
    --output_dir="/scratch/yifwang/fairness_x_explainability/bias_mitigation_results_jigsaw/bert_jigsaw_race_race_val_200_2/raw_attention/100.0" \
    --bias_type="race" 

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="google/jigsaw_unintended_bias" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/fairness_x_explainability/explanation_debiased_models_jigsaw/bert_jigsaw_race_1/raw_attention/10.0" \
    --batch_size=64 \
    --max_seq_length=512 \
    --num_examples=200 \
    --output_dir="/scratch/yifwang/fairness_x_explainability/bias_mitigation_results_jigsaw/bert_jigsaw_race_race_val_200_1/raw_attention/10.0" \
    --bias_type="race" 

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="google/jigsaw_unintended_bias" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/fairness_x_explainability/explanation_debiased_models_jigsaw/bert_jigsaw_race_2/raw_attention/10.0" \
    --batch_size=64 \
    --max_seq_length=512 \
    --num_examples=200 \
    --output_dir="/scratch/yifwang/fairness_x_explainability/bias_mitigation_results_jigsaw/bert_jigsaw_race_race_val_200_2/raw_attention/10.0" \
    --bias_type="race" 

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="google/jigsaw_unintended_bias" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/fairness_x_explainability/explanation_debiased_models_jigsaw/bert_jigsaw_race_1/raw_attention/1.0" \
    --batch_size=64 \
    --max_seq_length=512 \
    --num_examples=200 \
    --output_dir="/scratch/yifwang/fairness_x_explainability/bias_mitigation_results_jigsaw/bert_jigsaw_race_race_val_200_1/raw_attention/1.0" \
    --bias_type="race" 

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="google/jigsaw_unintended_bias" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/fairness_x_explainability/explanation_debiased_models_jigsaw/bert_jigsaw_race_2/raw_attention/1.0" \
    --batch_size=64 \
    --max_seq_length=512 \
    --num_examples=200 \
    --output_dir="/scratch/yifwang/fairness_x_explainability/bias_mitigation_results_jigsaw/bert_jigsaw_race_race_val_200_2/raw_attention/1.0" \
    --bias_type="race" 

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="google/jigsaw_unintended_bias" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/fairness_x_explainability/explanation_debiased_models_jigsaw/bert_jigsaw_race_1/raw_attention/0.1" \
    --batch_size=64 \
    --max_seq_length=512 \
    --num_examples=200 \
    --output_dir="/scratch/yifwang/fairness_x_explainability/bias_mitigation_results_jigsaw/bert_jigsaw_race_race_val_200_1/raw_attention/0.1" \
    --bias_type="race" 

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="google/jigsaw_unintended_bias" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/fairness_x_explainability/explanation_debiased_models_jigsaw/bert_jigsaw_race_2/raw_attention/0.1" \
    --batch_size=64 \
    --max_seq_length=512 \
    --num_examples=200 \
    --output_dir="/scratch/yifwang/fairness_x_explainability/bias_mitigation_results_jigsaw/bert_jigsaw_race_race_val_200_2/raw_attention/0.1" \
    --bias_type="race" 

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="google/jigsaw_unintended_bias" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/fairness_x_explainability/explanation_debiased_models_jigsaw/bert_jigsaw_race_1/raw_attention/0.01" \
    --batch_size=64 \
    --max_seq_length=512 \
    --num_examples=200 \
    --output_dir="/scratch/yifwang/fairness_x_explainability/bias_mitigation_results_jigsaw/bert_jigsaw_race_race_val_200_1/raw_attention/0.01" \
    --bias_type="race" 

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="google/jigsaw_unintended_bias" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/fairness_x_explainability/explanation_debiased_models_jigsaw/bert_jigsaw_race_2/raw_attention/0.01" \
    --batch_size=64 \
    --max_seq_length=512 \
    --num_examples=200 \
    --output_dir="/scratch/yifwang/fairness_x_explainability/bias_mitigation_results_jigsaw/bert_jigsaw_race_race_val_200_2/raw_attention/0.01" \
    --bias_type="race" 

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="google/jigsaw_unintended_bias" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/fairness_x_explainability/explanation_debiased_models_jigsaw/bert_jigsaw_race_1/attention_flow/100.0" \
    --batch_size=64 \
    --max_seq_length=512 \
    --num_examples=200 \
    --output_dir="/scratch/yifwang/fairness_x_explainability/bias_mitigation_results_jigsaw/bert_jigsaw_race_race_val_200_1/attention_flow/100.0" \
    --bias_type="race" 

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="google/jigsaw_unintended_bias" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/fairness_x_explainability/explanation_debiased_models_jigsaw/bert_jigsaw_race_2/attention_flow/100.0" \
    --batch_size=64 \
    --max_seq_length=512 \
    --num_examples=200 \
    --output_dir="/scratch/yifwang/fairness_x_explainability/bias_mitigation_results_jigsaw/bert_jigsaw_race_race_val_200_2/attention_flow/100.0" \
    --bias_type="race" 

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="google/jigsaw_unintended_bias" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/fairness_x_explainability/explanation_debiased_models_jigsaw/bert_jigsaw_race_1/attention_flow/10.0" \
    --batch_size=64 \
    --max_seq_length=512 \
    --num_examples=200 \
    --output_dir="/scratch/yifwang/fairness_x_explainability/bias_mitigation_results_jigsaw/bert_jigsaw_race_race_val_200_1/attention_flow/10.0" \
    --bias_type="race" 

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="google/jigsaw_unintended_bias" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/fairness_x_explainability/explanation_debiased_models_jigsaw/bert_jigsaw_race_2/attention_flow/10.0" \
    --batch_size=64 \
    --max_seq_length=512 \
    --num_examples=200 \
    --output_dir="/scratch/yifwang/fairness_x_explainability/bias_mitigation_results_jigsaw/bert_jigsaw_race_race_val_200_2/attention_flow/10.0" \
    --bias_type="race" 

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="google/jigsaw_unintended_bias" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/fairness_x_explainability/explanation_debiased_models_jigsaw/bert_jigsaw_race_1/attention_flow/1.0" \
    --batch_size=64 \
    --max_seq_length=512 \
    --num_examples=200 \
    --output_dir="/scratch/yifwang/fairness_x_explainability/bias_mitigation_results_jigsaw/bert_jigsaw_race_race_val_200_1/attention_flow/1.0" \
    --bias_type="race" 

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="google/jigsaw_unintended_bias" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/fairness_x_explainability/explanation_debiased_models_jigsaw/bert_jigsaw_race_2/attention_flow/1.0" \
    --batch_size=64 \
    --max_seq_length=512 \
    --num_examples=200 \
    --output_dir="/scratch/yifwang/fairness_x_explainability/bias_mitigation_results_jigsaw/bert_jigsaw_race_race_val_200_2/attention_flow/1.0" \
    --bias_type="race" 

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="google/jigsaw_unintended_bias" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/fairness_x_explainability/explanation_debiased_models_jigsaw/bert_jigsaw_race_1/attention_flow/0.1" \
    --batch_size=64 \
    --max_seq_length=512 \
    --num_examples=200 \
    --output_dir="/scratch/yifwang/fairness_x_explainability/bias_mitigation_results_jigsaw/bert_jigsaw_race_race_val_200_1/attention_flow/0.1" \
    --bias_type="race" 

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="google/jigsaw_unintended_bias" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/fairness_x_explainability/explanation_debiased_models_jigsaw/bert_jigsaw_race_2/attention_flow/0.1" \
    --batch_size=64 \
    --max_seq_length=512 \
    --num_examples=200 \
    --output_dir="/scratch/yifwang/fairness_x_explainability/bias_mitigation_results_jigsaw/bert_jigsaw_race_race_val_200_2/attention_flow/0.1" \
    --bias_type="race" 

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="google/jigsaw_unintended_bias" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/fairness_x_explainability/explanation_debiased_models_jigsaw/bert_jigsaw_race_1/attention_flow/0.01" \
    --batch_size=64 \
    --max_seq_length=512 \
    --num_examples=200 \
    --output_dir="/scratch/yifwang/fairness_x_explainability/bias_mitigation_results_jigsaw/bert_jigsaw_race_race_val_200_1/attention_flow/0.01" \
    --bias_type="race" 

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="google/jigsaw_unintended_bias" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/fairness_x_explainability/explanation_debiased_models_jigsaw/bert_jigsaw_race_2/attention_flow/0.01" \
    --batch_size=64 \
    --max_seq_length=512 \
    --num_examples=200 \
    --output_dir="/scratch/yifwang/fairness_x_explainability/bias_mitigation_results_jigsaw/bert_jigsaw_race_race_val_200_2/attention_flow/0.01" \
    --bias_type="race" 

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="google/jigsaw_unintended_bias" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/fairness_x_explainability/explanation_debiased_models_jigsaw/bert_jigsaw_race_1/attention_rollout/100.0" \
    --batch_size=64 \
    --max_seq_length=512 \
    --num_examples=200 \
    --output_dir="/scratch/yifwang/fairness_x_explainability/bias_mitigation_results_jigsaw/bert_jigsaw_race_race_val_200_1/attention_rollout/100.0" \
    --bias_type="race" 

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="google/jigsaw_unintended_bias" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/fairness_x_explainability/explanation_debiased_models_jigsaw/bert_jigsaw_race_2/attention_rollout/100.0" \
    --batch_size=64 \
    --max_seq_length=512 \
    --num_examples=200 \
    --output_dir="/scratch/yifwang/fairness_x_explainability/bias_mitigation_results_jigsaw/bert_jigsaw_race_race_val_200_2/attention_rollout/100.0" \
    --bias_type="race" 

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="google/jigsaw_unintended_bias" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/fairness_x_explainability/explanation_debiased_models_jigsaw/bert_jigsaw_race_1/attention_rollout/10.0" \
    --batch_size=64 \
    --max_seq_length=512 \
    --num_examples=200 \
    --output_dir="/scratch/yifwang/fairness_x_explainability/bias_mitigation_results_jigsaw/bert_jigsaw_race_race_val_200_1/attention_rollout/10.0" \
    --bias_type="race" 

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="google/jigsaw_unintended_bias" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/fairness_x_explainability/explanation_debiased_models_jigsaw/bert_jigsaw_race_2/attention_rollout/10.0" \
    --batch_size=64 \
    --max_seq_length=512 \
    --num_examples=200 \
    --output_dir="/scratch/yifwang/fairness_x_explainability/bias_mitigation_results_jigsaw/bert_jigsaw_race_race_val_200_2/attention_rollout/10.0" \
    --bias_type="race" 

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="google/jigsaw_unintended_bias" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/fairness_x_explainability/explanation_debiased_models_jigsaw/bert_jigsaw_race_1/attention_rollout/1.0" \
    --batch_size=64 \
    --max_seq_length=512 \
    --num_examples=200 \
    --output_dir="/scratch/yifwang/fairness_x_explainability/bias_mitigation_results_jigsaw/bert_jigsaw_race_race_val_200_1/attention_rollout/1.0" \
    --bias_type="race" 

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="google/jigsaw_unintended_bias" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/fairness_x_explainability/explanation_debiased_models_jigsaw/bert_jigsaw_race_2/attention_rollout/1.0" \
    --batch_size=64 \
    --max_seq_length=512 \
    --num_examples=200 \
    --output_dir="/scratch/yifwang/fairness_x_explainability/bias_mitigation_results_jigsaw/bert_jigsaw_race_race_val_200_2/attention_rollout/1.0" \
    --bias_type="race" 

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="google/jigsaw_unintended_bias" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/fairness_x_explainability/explanation_debiased_models_jigsaw/bert_jigsaw_race_1/attention_rollout/0.1" \
    --batch_size=64 \
    --max_seq_length=512 \
    --num_examples=200 \
    --output_dir="/scratch/yifwang/fairness_x_explainability/bias_mitigation_results_jigsaw/bert_jigsaw_race_race_val_200_1/attention_rollout/0.1" \
    --bias_type="race" 

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="google/jigsaw_unintended_bias" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/fairness_x_explainability/explanation_debiased_models_jigsaw/bert_jigsaw_race_2/attention_rollout/0.1" \
    --batch_size=64 \
    --max_seq_length=512 \
    --num_examples=200 \
    --output_dir="/scratch/yifwang/fairness_x_explainability/bias_mitigation_results_jigsaw/bert_jigsaw_race_race_val_200_2/attention_rollout/0.1" \
    --bias_type="race" 

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="google/jigsaw_unintended_bias" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/fairness_x_explainability/explanation_debiased_models_jigsaw/bert_jigsaw_race_1/attention_rollout/0.01" \
    --batch_size=64 \
    --max_seq_length=512 \
    --num_examples=200 \
    --output_dir="/scratch/yifwang/fairness_x_explainability/bias_mitigation_results_jigsaw/bert_jigsaw_race_race_val_200_1/attention_rollout/0.01" \
    --bias_type="race" 

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="google/jigsaw_unintended_bias" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/fairness_x_explainability/explanation_debiased_models_jigsaw/bert_jigsaw_race_2/attention_rollout/0.01" \
    --batch_size=64 \
    --max_seq_length=512 \
    --num_examples=200 \
    --output_dir="/scratch/yifwang/fairness_x_explainability/bias_mitigation_results_jigsaw/bert_jigsaw_race_race_val_200_2/attention_rollout/0.01" \
    --bias_type="race" 

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="google/jigsaw_unintended_bias" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/fairness_x_explainability/explanation_debiased_models_jigsaw/bert_jigsaw_race_1/Occlusion/100.0" \
    --batch_size=64 \
    --max_seq_length=512 \
    --num_examples=200 \
    --output_dir="/scratch/yifwang/fairness_x_explainability/bias_mitigation_results_jigsaw/bert_jigsaw_race_race_val_200_1/Occlusion/100.0" \
    --bias_type="race" 

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="google/jigsaw_unintended_bias" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/fairness_x_explainability/explanation_debiased_models_jigsaw/bert_jigsaw_race_2/Occlusion/100.0" \
    --batch_size=64 \
    --max_seq_length=512 \
    --num_examples=200 \
    --output_dir="/scratch/yifwang/fairness_x_explainability/bias_mitigation_results_jigsaw/bert_jigsaw_race_race_val_200_2/Occlusion/100.0" \
    --bias_type="race" 

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="google/jigsaw_unintended_bias" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/fairness_x_explainability/explanation_debiased_models_jigsaw/bert_jigsaw_race_1/Occlusion/10.0" \
    --batch_size=64 \
    --max_seq_length=512 \
    --num_examples=200 \
    --output_dir="/scratch/yifwang/fairness_x_explainability/bias_mitigation_results_jigsaw/bert_jigsaw_race_race_val_200_1/Occlusion/10.0" \
    --bias_type="race" 

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="google/jigsaw_unintended_bias" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/fairness_x_explainability/explanation_debiased_models_jigsaw/bert_jigsaw_race_2/Occlusion/10.0" \
    --batch_size=64 \
    --max_seq_length=512 \
    --num_examples=200 \
    --output_dir="/scratch/yifwang/fairness_x_explainability/bias_mitigation_results_jigsaw/bert_jigsaw_race_race_val_200_2/Occlusion/10.0" \
    --bias_type="race" 

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="google/jigsaw_unintended_bias" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/fairness_x_explainability/explanation_debiased_models_jigsaw/bert_jigsaw_race_1/Occlusion/1.0" \
    --batch_size=64 \
    --max_seq_length=512 \
    --num_examples=200 \
    --output_dir="/scratch/yifwang/fairness_x_explainability/bias_mitigation_results_jigsaw/bert_jigsaw_race_race_val_200_1/Occlusion/1.0" \
    --bias_type="race" 

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="google/jigsaw_unintended_bias" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/fairness_x_explainability/explanation_debiased_models_jigsaw/bert_jigsaw_race_2/Occlusion/1.0" \
    --batch_size=64 \
    --max_seq_length=512 \
    --num_examples=200 \
    --output_dir="/scratch/yifwang/fairness_x_explainability/bias_mitigation_results_jigsaw/bert_jigsaw_race_race_val_200_2/Occlusion/1.0" \
    --bias_type="race" 

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="google/jigsaw_unintended_bias" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/fairness_x_explainability/explanation_debiased_models_jigsaw/bert_jigsaw_race_1/Occlusion/0.1" \
    --batch_size=64 \
    --max_seq_length=512 \
    --num_examples=200 \
    --output_dir="/scratch/yifwang/fairness_x_explainability/bias_mitigation_results_jigsaw/bert_jigsaw_race_race_val_200_1/Occlusion/0.1" \
    --bias_type="race" 

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="google/jigsaw_unintended_bias" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/fairness_x_explainability/explanation_debiased_models_jigsaw/bert_jigsaw_race_2/Occlusion/0.1" \
    --batch_size=64 \
    --max_seq_length=512 \
    --num_examples=200 \
    --output_dir="/scratch/yifwang/fairness_x_explainability/bias_mitigation_results_jigsaw/bert_jigsaw_race_race_val_200_2/Occlusion/0.1" \
    --bias_type="race" 

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="google/jigsaw_unintended_bias" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/fairness_x_explainability/explanation_debiased_models_jigsaw/bert_jigsaw_race_1/Occlusion/0.01" \
    --batch_size=64 \
    --max_seq_length=512 \
    --num_examples=200 \
    --output_dir="/scratch/yifwang/fairness_x_explainability/bias_mitigation_results_jigsaw/bert_jigsaw_race_race_val_200_1/Occlusion/0.01" \
    --bias_type="race" 

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="google/jigsaw_unintended_bias" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/fairness_x_explainability/explanation_debiased_models_jigsaw/bert_jigsaw_race_2/Occlusion/0.01" \
    --batch_size=64 \
    --max_seq_length=512 \
    --num_examples=200 \
    --output_dir="/scratch/yifwang/fairness_x_explainability/bias_mitigation_results_jigsaw/bert_jigsaw_race_race_val_200_2/Occlusion/0.01" \
    --bias_type="race" 

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="google/jigsaw_unintended_bias" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/fairness_x_explainability/explanation_debiased_models_jigsaw/roberta_jigsaw_race_1/Saliency/L1_100.0" \
    --batch_size=64 \
    --max_seq_length=512 \
    --num_examples=200 \
    --output_dir="/scratch/yifwang/fairness_x_explainability/bias_mitigation_results_jigsaw/roberta_jigsaw_race_race_val_200_1/Saliency/L1_100.0" \
    --bias_type="race" 

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="google/jigsaw_unintended_bias" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/fairness_x_explainability/explanation_debiased_models_jigsaw/roberta_jigsaw_race_2/Saliency/L1_100.0" \
    --batch_size=64 \
    --max_seq_length=512 \
    --num_examples=200 \
    --output_dir="/scratch/yifwang/fairness_x_explainability/bias_mitigation_results_jigsaw/roberta_jigsaw_race_race_val_200_2/Saliency/L1_100.0" \
    --bias_type="race" 

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="google/jigsaw_unintended_bias" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/fairness_x_explainability/explanation_debiased_models_jigsaw/roberta_jigsaw_race_1/Saliency/L2_100.0" \
    --batch_size=64 \
    --max_seq_length=512 \
    --num_examples=200 \
    --output_dir="/scratch/yifwang/fairness_x_explainability/bias_mitigation_results_jigsaw/roberta_jigsaw_race_race_val_200_1/Saliency/L2_100.0" \
    --bias_type="race" 

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="google/jigsaw_unintended_bias" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/fairness_x_explainability/explanation_debiased_models_jigsaw/roberta_jigsaw_race_2/Saliency/L2_100.0" \
    --batch_size=64 \
    --max_seq_length=512 \
    --num_examples=200 \
    --output_dir="/scratch/yifwang/fairness_x_explainability/bias_mitigation_results_jigsaw/roberta_jigsaw_race_race_val_200_2/Saliency/L2_100.0" \
    --bias_type="race" 

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="google/jigsaw_unintended_bias" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/fairness_x_explainability/explanation_debiased_models_jigsaw/roberta_jigsaw_race_1/Saliency/L1_10.0" \
    --batch_size=64 \
    --max_seq_length=512 \
    --num_examples=200 \
    --output_dir="/scratch/yifwang/fairness_x_explainability/bias_mitigation_results_jigsaw/roberta_jigsaw_race_race_val_200_1/Saliency/L1_10.0" \
    --bias_type="race" 

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="google/jigsaw_unintended_bias" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/fairness_x_explainability/explanation_debiased_models_jigsaw/roberta_jigsaw_race_2/Saliency/L1_10.0" \
    --batch_size=64 \
    --max_seq_length=512 \
    --num_examples=200 \
    --output_dir="/scratch/yifwang/fairness_x_explainability/bias_mitigation_results_jigsaw/roberta_jigsaw_race_race_val_200_2/Saliency/L1_10.0" \
    --bias_type="race" 

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="google/jigsaw_unintended_bias" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/fairness_x_explainability/explanation_debiased_models_jigsaw/roberta_jigsaw_race_1/Saliency/L2_10.0" \
    --batch_size=64 \
    --max_seq_length=512 \
    --num_examples=200 \
    --output_dir="/scratch/yifwang/fairness_x_explainability/bias_mitigation_results_jigsaw/roberta_jigsaw_race_race_val_200_1/Saliency/L2_10.0" \
    --bias_type="race" 

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="google/jigsaw_unintended_bias" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/fairness_x_explainability/explanation_debiased_models_jigsaw/roberta_jigsaw_race_2/Saliency/L2_10.0" \
    --batch_size=64 \
    --max_seq_length=512 \
    --num_examples=200 \
    --output_dir="/scratch/yifwang/fairness_x_explainability/bias_mitigation_results_jigsaw/roberta_jigsaw_race_race_val_200_2/Saliency/L2_10.0" \
    --bias_type="race" 

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="google/jigsaw_unintended_bias" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/fairness_x_explainability/explanation_debiased_models_jigsaw/roberta_jigsaw_race_1/Saliency/L1_1.0" \
    --batch_size=64 \
    --max_seq_length=512 \
    --num_examples=200 \
    --output_dir="/scratch/yifwang/fairness_x_explainability/bias_mitigation_results_jigsaw/roberta_jigsaw_race_race_val_200_1/Saliency/L1_1.0" \
    --bias_type="race" 

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="google/jigsaw_unintended_bias" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/fairness_x_explainability/explanation_debiased_models_jigsaw/roberta_jigsaw_race_2/Saliency/L1_1.0" \
    --batch_size=64 \
    --max_seq_length=512 \
    --num_examples=200 \
    --output_dir="/scratch/yifwang/fairness_x_explainability/bias_mitigation_results_jigsaw/roberta_jigsaw_race_race_val_200_2/Saliency/L1_1.0" \
    --bias_type="race" 

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="google/jigsaw_unintended_bias" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/fairness_x_explainability/explanation_debiased_models_jigsaw/roberta_jigsaw_race_1/Saliency/L2_1.0" \
    --batch_size=64 \
    --max_seq_length=512 \
    --num_examples=200 \
    --output_dir="/scratch/yifwang/fairness_x_explainability/bias_mitigation_results_jigsaw/roberta_jigsaw_race_race_val_200_1/Saliency/L2_1.0" \
    --bias_type="race" 

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="google/jigsaw_unintended_bias" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/fairness_x_explainability/explanation_debiased_models_jigsaw/roberta_jigsaw_race_2/Saliency/L2_1.0" \
    --batch_size=64 \
    --max_seq_length=512 \
    --num_examples=200 \
    --output_dir="/scratch/yifwang/fairness_x_explainability/bias_mitigation_results_jigsaw/roberta_jigsaw_race_race_val_200_2/Saliency/L2_1.0" \
    --bias_type="race" 

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="google/jigsaw_unintended_bias" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/fairness_x_explainability/explanation_debiased_models_jigsaw/roberta_jigsaw_race_1/Saliency/L1_0.1" \
    --batch_size=64 \
    --max_seq_length=512 \
    --num_examples=200 \
    --output_dir="/scratch/yifwang/fairness_x_explainability/bias_mitigation_results_jigsaw/roberta_jigsaw_race_race_val_200_1/Saliency/L1_0.1" \
    --bias_type="race" 

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="google/jigsaw_unintended_bias" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/fairness_x_explainability/explanation_debiased_models_jigsaw/roberta_jigsaw_race_2/Saliency/L1_0.1" \
    --batch_size=64 \
    --max_seq_length=512 \
    --num_examples=200 \
    --output_dir="/scratch/yifwang/fairness_x_explainability/bias_mitigation_results_jigsaw/roberta_jigsaw_race_race_val_200_2/Saliency/L1_0.1" \
    --bias_type="race" 

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="google/jigsaw_unintended_bias" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/fairness_x_explainability/explanation_debiased_models_jigsaw/roberta_jigsaw_race_1/Saliency/L2_0.1" \
    --batch_size=64 \
    --max_seq_length=512 \
    --num_examples=200 \
    --output_dir="/scratch/yifwang/fairness_x_explainability/bias_mitigation_results_jigsaw/roberta_jigsaw_race_race_val_200_1/Saliency/L2_0.1" \
    --bias_type="race" 

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="google/jigsaw_unintended_bias" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/fairness_x_explainability/explanation_debiased_models_jigsaw/roberta_jigsaw_race_2/Saliency/L2_0.1" \
    --batch_size=64 \
    --max_seq_length=512 \
    --num_examples=200 \
    --output_dir="/scratch/yifwang/fairness_x_explainability/bias_mitigation_results_jigsaw/roberta_jigsaw_race_race_val_200_2/Saliency/L2_0.1" \
    --bias_type="race" 

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="google/jigsaw_unintended_bias" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/fairness_x_explainability/explanation_debiased_models_jigsaw/roberta_jigsaw_race_1/Saliency/L1_0.01" \
    --batch_size=64 \
    --max_seq_length=512 \
    --num_examples=200 \
    --output_dir="/scratch/yifwang/fairness_x_explainability/bias_mitigation_results_jigsaw/roberta_jigsaw_race_race_val_200_1/Saliency/L1_0.01" \
    --bias_type="race" 

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="google/jigsaw_unintended_bias" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/fairness_x_explainability/explanation_debiased_models_jigsaw/roberta_jigsaw_race_2/Saliency/L1_0.01" \
    --batch_size=64 \
    --max_seq_length=512 \
    --num_examples=200 \
    --output_dir="/scratch/yifwang/fairness_x_explainability/bias_mitigation_results_jigsaw/roberta_jigsaw_race_race_val_200_2/Saliency/L1_0.01" \
    --bias_type="race" 

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="google/jigsaw_unintended_bias" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/fairness_x_explainability/explanation_debiased_models_jigsaw/roberta_jigsaw_race_1/Saliency/L2_0.01" \
    --batch_size=64 \
    --max_seq_length=512 \
    --num_examples=200 \
    --output_dir="/scratch/yifwang/fairness_x_explainability/bias_mitigation_results_jigsaw/roberta_jigsaw_race_race_val_200_1/Saliency/L2_0.01" \
    --bias_type="race" 

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="google/jigsaw_unintended_bias" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/fairness_x_explainability/explanation_debiased_models_jigsaw/roberta_jigsaw_race_2/Saliency/L2_0.01" \
    --batch_size=64 \
    --max_seq_length=512 \
    --num_examples=200 \
    --output_dir="/scratch/yifwang/fairness_x_explainability/bias_mitigation_results_jigsaw/roberta_jigsaw_race_race_val_200_2/Saliency/L2_0.01" \
    --bias_type="race" 

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="google/jigsaw_unintended_bias" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/fairness_x_explainability/explanation_debiased_models_jigsaw/roberta_jigsaw_race_1/InputXGradient/L1_100.0" \
    --batch_size=64 \
    --max_seq_length=512 \
    --num_examples=200 \
    --output_dir="/scratch/yifwang/fairness_x_explainability/bias_mitigation_results_jigsaw/roberta_jigsaw_race_race_val_200_1/InputXGradient/L1_100.0" \
    --bias_type="race" 

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="google/jigsaw_unintended_bias" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/fairness_x_explainability/explanation_debiased_models_jigsaw/roberta_jigsaw_race_2/InputXGradient/L1_100.0" \
    --batch_size=64 \
    --max_seq_length=512 \
    --num_examples=200 \
    --output_dir="/scratch/yifwang/fairness_x_explainability/bias_mitigation_results_jigsaw/roberta_jigsaw_race_race_val_200_2/InputXGradient/L1_100.0" \
    --bias_type="race" 

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="google/jigsaw_unintended_bias" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/fairness_x_explainability/explanation_debiased_models_jigsaw/roberta_jigsaw_race_1/InputXGradient/L2_100.0" \
    --batch_size=64 \
    --max_seq_length=512 \
    --num_examples=200 \
    --output_dir="/scratch/yifwang/fairness_x_explainability/bias_mitigation_results_jigsaw/roberta_jigsaw_race_race_val_200_1/InputXGradient/L2_100.0" \
    --bias_type="race" 

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="google/jigsaw_unintended_bias" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/fairness_x_explainability/explanation_debiased_models_jigsaw/roberta_jigsaw_race_2/InputXGradient/L2_100.0" \
    --batch_size=64 \
    --max_seq_length=512 \
    --num_examples=200 \
    --output_dir="/scratch/yifwang/fairness_x_explainability/bias_mitigation_results_jigsaw/roberta_jigsaw_race_race_val_200_2/InputXGradient/L2_100.0" \
    --bias_type="race" 

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="google/jigsaw_unintended_bias" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/fairness_x_explainability/explanation_debiased_models_jigsaw/roberta_jigsaw_race_1/InputXGradient/L1_10.0" \
    --batch_size=64 \
    --max_seq_length=512 \
    --num_examples=200 \
    --output_dir="/scratch/yifwang/fairness_x_explainability/bias_mitigation_results_jigsaw/roberta_jigsaw_race_race_val_200_1/InputXGradient/L1_10.0" \
    --bias_type="race" 

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="google/jigsaw_unintended_bias" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/fairness_x_explainability/explanation_debiased_models_jigsaw/roberta_jigsaw_race_2/InputXGradient/L1_10.0" \
    --batch_size=64 \
    --max_seq_length=512 \
    --num_examples=200 \
    --output_dir="/scratch/yifwang/fairness_x_explainability/bias_mitigation_results_jigsaw/roberta_jigsaw_race_race_val_200_2/InputXGradient/L1_10.0" \
    --bias_type="race" 

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="google/jigsaw_unintended_bias" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/fairness_x_explainability/explanation_debiased_models_jigsaw/roberta_jigsaw_race_1/InputXGradient/L2_10.0" \
    --batch_size=64 \
    --max_seq_length=512 \
    --num_examples=200 \
    --output_dir="/scratch/yifwang/fairness_x_explainability/bias_mitigation_results_jigsaw/roberta_jigsaw_race_race_val_200_1/InputXGradient/L2_10.0" \
    --bias_type="race" 

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="google/jigsaw_unintended_bias" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/fairness_x_explainability/explanation_debiased_models_jigsaw/roberta_jigsaw_race_2/InputXGradient/L2_10.0" \
    --batch_size=64 \
    --max_seq_length=512 \
    --num_examples=200 \
    --output_dir="/scratch/yifwang/fairness_x_explainability/bias_mitigation_results_jigsaw/roberta_jigsaw_race_race_val_200_2/InputXGradient/L2_10.0" \
    --bias_type="race" 

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="google/jigsaw_unintended_bias" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/fairness_x_explainability/explanation_debiased_models_jigsaw/roberta_jigsaw_race_1/InputXGradient/L1_1.0" \
    --batch_size=64 \
    --max_seq_length=512 \
    --num_examples=200 \
    --output_dir="/scratch/yifwang/fairness_x_explainability/bias_mitigation_results_jigsaw/roberta_jigsaw_race_race_val_200_1/InputXGradient/L1_1.0" \
    --bias_type="race" 

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="google/jigsaw_unintended_bias" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/fairness_x_explainability/explanation_debiased_models_jigsaw/roberta_jigsaw_race_2/InputXGradient/L1_1.0" \
    --batch_size=64 \
    --max_seq_length=512 \
    --num_examples=200 \
    --output_dir="/scratch/yifwang/fairness_x_explainability/bias_mitigation_results_jigsaw/roberta_jigsaw_race_race_val_200_2/InputXGradient/L1_1.0" \
    --bias_type="race" 

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="google/jigsaw_unintended_bias" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/fairness_x_explainability/explanation_debiased_models_jigsaw/roberta_jigsaw_race_1/InputXGradient/L2_1.0" \
    --batch_size=64 \
    --max_seq_length=512 \
    --num_examples=200 \
    --output_dir="/scratch/yifwang/fairness_x_explainability/bias_mitigation_results_jigsaw/roberta_jigsaw_race_race_val_200_1/InputXGradient/L2_1.0" \
    --bias_type="race" 

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="google/jigsaw_unintended_bias" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/fairness_x_explainability/explanation_debiased_models_jigsaw/roberta_jigsaw_race_2/InputXGradient/L2_1.0" \
    --batch_size=64 \
    --max_seq_length=512 \
    --num_examples=200 \
    --output_dir="/scratch/yifwang/fairness_x_explainability/bias_mitigation_results_jigsaw/roberta_jigsaw_race_race_val_200_2/InputXGradient/L2_1.0" \
    --bias_type="race" 

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="google/jigsaw_unintended_bias" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/fairness_x_explainability/explanation_debiased_models_jigsaw/roberta_jigsaw_race_1/InputXGradient/L1_0.1" \
    --batch_size=64 \
    --max_seq_length=512 \
    --num_examples=200 \
    --output_dir="/scratch/yifwang/fairness_x_explainability/bias_mitigation_results_jigsaw/roberta_jigsaw_race_race_val_200_1/InputXGradient/L1_0.1" \
    --bias_type="race" 

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="google/jigsaw_unintended_bias" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/fairness_x_explainability/explanation_debiased_models_jigsaw/roberta_jigsaw_race_2/InputXGradient/L1_0.1" \
    --batch_size=64 \
    --max_seq_length=512 \
    --num_examples=200 \
    --output_dir="/scratch/yifwang/fairness_x_explainability/bias_mitigation_results_jigsaw/roberta_jigsaw_race_race_val_200_2/InputXGradient/L1_0.1" \
    --bias_type="race" 

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="google/jigsaw_unintended_bias" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/fairness_x_explainability/explanation_debiased_models_jigsaw/roberta_jigsaw_race_1/InputXGradient/L2_0.1" \
    --batch_size=64 \
    --max_seq_length=512 \
    --num_examples=200 \
    --output_dir="/scratch/yifwang/fairness_x_explainability/bias_mitigation_results_jigsaw/roberta_jigsaw_race_race_val_200_1/InputXGradient/L2_0.1" \
    --bias_type="race" 

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="google/jigsaw_unintended_bias" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/fairness_x_explainability/explanation_debiased_models_jigsaw/roberta_jigsaw_race_2/InputXGradient/L2_0.1" \
    --batch_size=64 \
    --max_seq_length=512 \
    --num_examples=200 \
    --output_dir="/scratch/yifwang/fairness_x_explainability/bias_mitigation_results_jigsaw/roberta_jigsaw_race_race_val_200_2/InputXGradient/L2_0.1" \
    --bias_type="race" 

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="google/jigsaw_unintended_bias" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/fairness_x_explainability/explanation_debiased_models_jigsaw/roberta_jigsaw_race_1/InputXGradient/L1_0.01" \
    --batch_size=64 \
    --max_seq_length=512 \
    --num_examples=200 \
    --output_dir="/scratch/yifwang/fairness_x_explainability/bias_mitigation_results_jigsaw/roberta_jigsaw_race_race_val_200_1/InputXGradient/L1_0.01" \
    --bias_type="race" 

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="google/jigsaw_unintended_bias" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/fairness_x_explainability/explanation_debiased_models_jigsaw/roberta_jigsaw_race_2/InputXGradient/L1_0.01" \
    --batch_size=64 \
    --max_seq_length=512 \
    --num_examples=200 \
    --output_dir="/scratch/yifwang/fairness_x_explainability/bias_mitigation_results_jigsaw/roberta_jigsaw_race_race_val_200_2/InputXGradient/L1_0.01" \
    --bias_type="race" 

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="google/jigsaw_unintended_bias" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/fairness_x_explainability/explanation_debiased_models_jigsaw/roberta_jigsaw_race_1/InputXGradient/L2_0.01" \
    --batch_size=64 \
    --max_seq_length=512 \
    --num_examples=200 \
    --output_dir="/scratch/yifwang/fairness_x_explainability/bias_mitigation_results_jigsaw/roberta_jigsaw_race_race_val_200_1/InputXGradient/L2_0.01" \
    --bias_type="race" 

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="google/jigsaw_unintended_bias" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/fairness_x_explainability/explanation_debiased_models_jigsaw/roberta_jigsaw_race_2/InputXGradient/L2_0.01" \
    --batch_size=64 \
    --max_seq_length=512 \
    --num_examples=200 \
    --output_dir="/scratch/yifwang/fairness_x_explainability/bias_mitigation_results_jigsaw/roberta_jigsaw_race_race_val_200_2/InputXGradient/L2_0.01" \
    --bias_type="race" 

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="google/jigsaw_unintended_bias" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/fairness_x_explainability/explanation_debiased_models_jigsaw/roberta_jigsaw_race_1/raw_attention/100.0" \
    --batch_size=64 \
    --max_seq_length=512 \
    --num_examples=200 \
    --output_dir="/scratch/yifwang/fairness_x_explainability/bias_mitigation_results_jigsaw/roberta_jigsaw_race_race_val_200_1/raw_attention/100.0" \
    --bias_type="race" 

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="google/jigsaw_unintended_bias" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/fairness_x_explainability/explanation_debiased_models_jigsaw/roberta_jigsaw_race_2/raw_attention/100.0" \
    --batch_size=64 \
    --max_seq_length=512 \
    --num_examples=200 \
    --output_dir="/scratch/yifwang/fairness_x_explainability/bias_mitigation_results_jigsaw/roberta_jigsaw_race_race_val_200_2/raw_attention/100.0" \
    --bias_type="race" 

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="google/jigsaw_unintended_bias" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/fairness_x_explainability/explanation_debiased_models_jigsaw/roberta_jigsaw_race_1/raw_attention/10.0" \
    --batch_size=64 \
    --max_seq_length=512 \
    --num_examples=200 \
    --output_dir="/scratch/yifwang/fairness_x_explainability/bias_mitigation_results_jigsaw/roberta_jigsaw_race_race_val_200_1/raw_attention/10.0" \
    --bias_type="race" 

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="google/jigsaw_unintended_bias" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/fairness_x_explainability/explanation_debiased_models_jigsaw/roberta_jigsaw_race_2/raw_attention/10.0" \
    --batch_size=64 \
    --max_seq_length=512 \
    --num_examples=200 \
    --output_dir="/scratch/yifwang/fairness_x_explainability/bias_mitigation_results_jigsaw/roberta_jigsaw_race_race_val_200_2/raw_attention/10.0" \
    --bias_type="race" 

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="google/jigsaw_unintended_bias" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/fairness_x_explainability/explanation_debiased_models_jigsaw/roberta_jigsaw_race_1/raw_attention/1.0" \
    --batch_size=64 \
    --max_seq_length=512 \
    --num_examples=200 \
    --output_dir="/scratch/yifwang/fairness_x_explainability/bias_mitigation_results_jigsaw/roberta_jigsaw_race_race_val_200_1/raw_attention/1.0" \
    --bias_type="race" 

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="google/jigsaw_unintended_bias" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/fairness_x_explainability/explanation_debiased_models_jigsaw/roberta_jigsaw_race_2/raw_attention/1.0" \
    --batch_size=64 \
    --max_seq_length=512 \
    --num_examples=200 \
    --output_dir="/scratch/yifwang/fairness_x_explainability/bias_mitigation_results_jigsaw/roberta_jigsaw_race_race_val_200_2/raw_attention/1.0" \
    --bias_type="race" 

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="google/jigsaw_unintended_bias" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/fairness_x_explainability/explanation_debiased_models_jigsaw/roberta_jigsaw_race_1/raw_attention/0.1" \
    --batch_size=64 \
    --max_seq_length=512 \
    --num_examples=200 \
    --output_dir="/scratch/yifwang/fairness_x_explainability/bias_mitigation_results_jigsaw/roberta_jigsaw_race_race_val_200_1/raw_attention/0.1" \
    --bias_type="race" 

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="google/jigsaw_unintended_bias" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/fairness_x_explainability/explanation_debiased_models_jigsaw/roberta_jigsaw_race_2/raw_attention/0.1" \
    --batch_size=64 \
    --max_seq_length=512 \
    --num_examples=200 \
    --output_dir="/scratch/yifwang/fairness_x_explainability/bias_mitigation_results_jigsaw/roberta_jigsaw_race_race_val_200_2/raw_attention/0.1" \
    --bias_type="race" 

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="google/jigsaw_unintended_bias" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/fairness_x_explainability/explanation_debiased_models_jigsaw/roberta_jigsaw_race_1/raw_attention/0.01" \
    --batch_size=64 \
    --max_seq_length=512 \
    --num_examples=200 \
    --output_dir="/scratch/yifwang/fairness_x_explainability/bias_mitigation_results_jigsaw/roberta_jigsaw_race_race_val_200_1/raw_attention/0.01" \
    --bias_type="race" 

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="google/jigsaw_unintended_bias" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/fairness_x_explainability/explanation_debiased_models_jigsaw/roberta_jigsaw_race_2/raw_attention/0.01" \
    --batch_size=64 \
    --max_seq_length=512 \
    --num_examples=200 \
    --output_dir="/scratch/yifwang/fairness_x_explainability/bias_mitigation_results_jigsaw/roberta_jigsaw_race_race_val_200_2/raw_attention/0.01" \
    --bias_type="race" 

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="google/jigsaw_unintended_bias" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/fairness_x_explainability/explanation_debiased_models_jigsaw/roberta_jigsaw_race_1/attention_flow/100.0" \
    --batch_size=64 \
    --max_seq_length=512 \
    --num_examples=200 \
    --output_dir="/scratch/yifwang/fairness_x_explainability/bias_mitigation_results_jigsaw/roberta_jigsaw_race_race_val_200_1/attention_flow/100.0" \
    --bias_type="race" 

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="google/jigsaw_unintended_bias" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/fairness_x_explainability/explanation_debiased_models_jigsaw/roberta_jigsaw_race_2/attention_flow/100.0" \
    --batch_size=64 \
    --max_seq_length=512 \
    --num_examples=200 \
    --output_dir="/scratch/yifwang/fairness_x_explainability/bias_mitigation_results_jigsaw/roberta_jigsaw_race_race_val_200_2/attention_flow/100.0" \
    --bias_type="race" 

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="google/jigsaw_unintended_bias" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/fairness_x_explainability/explanation_debiased_models_jigsaw/roberta_jigsaw_race_1/attention_flow/10.0" \
    --batch_size=64 \
    --max_seq_length=512 \
    --num_examples=200 \
    --output_dir="/scratch/yifwang/fairness_x_explainability/bias_mitigation_results_jigsaw/roberta_jigsaw_race_race_val_200_1/attention_flow/10.0" \
    --bias_type="race" 

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="google/jigsaw_unintended_bias" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/fairness_x_explainability/explanation_debiased_models_jigsaw/roberta_jigsaw_race_2/attention_flow/10.0" \
    --batch_size=64 \
    --max_seq_length=512 \
    --num_examples=200 \
    --output_dir="/scratch/yifwang/fairness_x_explainability/bias_mitigation_results_jigsaw/roberta_jigsaw_race_race_val_200_2/attention_flow/10.0" \
    --bias_type="race" 

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="google/jigsaw_unintended_bias" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/fairness_x_explainability/explanation_debiased_models_jigsaw/roberta_jigsaw_race_1/attention_flow/1.0" \
    --batch_size=64 \
    --max_seq_length=512 \
    --num_examples=200 \
    --output_dir="/scratch/yifwang/fairness_x_explainability/bias_mitigation_results_jigsaw/roberta_jigsaw_race_race_val_200_1/attention_flow/1.0" \
    --bias_type="race" 

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="google/jigsaw_unintended_bias" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/fairness_x_explainability/explanation_debiased_models_jigsaw/roberta_jigsaw_race_2/attention_flow/1.0" \
    --batch_size=64 \
    --max_seq_length=512 \
    --num_examples=200 \
    --output_dir="/scratch/yifwang/fairness_x_explainability/bias_mitigation_results_jigsaw/roberta_jigsaw_race_race_val_200_2/attention_flow/1.0" \
    --bias_type="race" 

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="google/jigsaw_unintended_bias" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/fairness_x_explainability/explanation_debiased_models_jigsaw/roberta_jigsaw_race_1/attention_flow/0.1" \
    --batch_size=64 \
    --max_seq_length=512 \
    --num_examples=200 \
    --output_dir="/scratch/yifwang/fairness_x_explainability/bias_mitigation_results_jigsaw/roberta_jigsaw_race_race_val_200_1/attention_flow/0.1" \
    --bias_type="race" 

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="google/jigsaw_unintended_bias" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/fairness_x_explainability/explanation_debiased_models_jigsaw/roberta_jigsaw_race_2/attention_flow/0.1" \
    --batch_size=64 \
    --max_seq_length=512 \
    --num_examples=200 \
    --output_dir="/scratch/yifwang/fairness_x_explainability/bias_mitigation_results_jigsaw/roberta_jigsaw_race_race_val_200_2/attention_flow/0.1" \
    --bias_type="race" 

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="google/jigsaw_unintended_bias" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/fairness_x_explainability/explanation_debiased_models_jigsaw/roberta_jigsaw_race_1/attention_flow/0.01" \
    --batch_size=64 \
    --max_seq_length=512 \
    --num_examples=200 \
    --output_dir="/scratch/yifwang/fairness_x_explainability/bias_mitigation_results_jigsaw/roberta_jigsaw_race_race_val_200_1/attention_flow/0.01" \
    --bias_type="race" 

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="google/jigsaw_unintended_bias" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/fairness_x_explainability/explanation_debiased_models_jigsaw/roberta_jigsaw_race_2/attention_flow/0.01" \
    --batch_size=64 \
    --max_seq_length=512 \
    --num_examples=200 \
    --output_dir="/scratch/yifwang/fairness_x_explainability/bias_mitigation_results_jigsaw/roberta_jigsaw_race_race_val_200_2/attention_flow/0.01" \
    --bias_type="race" 

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="google/jigsaw_unintended_bias" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/fairness_x_explainability/explanation_debiased_models_jigsaw/roberta_jigsaw_race_1/attention_rollout/100.0" \
    --batch_size=64 \
    --max_seq_length=512 \
    --num_examples=200 \
    --output_dir="/scratch/yifwang/fairness_x_explainability/bias_mitigation_results_jigsaw/roberta_jigsaw_race_race_val_200_1/attention_rollout/100.0" \
    --bias_type="race" 

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="google/jigsaw_unintended_bias" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/fairness_x_explainability/explanation_debiased_models_jigsaw/roberta_jigsaw_race_2/attention_rollout/100.0" \
    --batch_size=64 \
    --max_seq_length=512 \
    --num_examples=200 \
    --output_dir="/scratch/yifwang/fairness_x_explainability/bias_mitigation_results_jigsaw/roberta_jigsaw_race_race_val_200_2/attention_rollout/100.0" \
    --bias_type="race" 

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="google/jigsaw_unintended_bias" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/fairness_x_explainability/explanation_debiased_models_jigsaw/roberta_jigsaw_race_1/attention_rollout/10.0" \
    --batch_size=64 \
    --max_seq_length=512 \
    --num_examples=200 \
    --output_dir="/scratch/yifwang/fairness_x_explainability/bias_mitigation_results_jigsaw/roberta_jigsaw_race_race_val_200_1/attention_rollout/10.0" \
    --bias_type="race" 

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="google/jigsaw_unintended_bias" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/fairness_x_explainability/explanation_debiased_models_jigsaw/roberta_jigsaw_race_2/attention_rollout/10.0" \
    --batch_size=64 \
    --max_seq_length=512 \
    --num_examples=200 \
    --output_dir="/scratch/yifwang/fairness_x_explainability/bias_mitigation_results_jigsaw/roberta_jigsaw_race_race_val_200_2/attention_rollout/10.0" \
    --bias_type="race" 

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="google/jigsaw_unintended_bias" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/fairness_x_explainability/explanation_debiased_models_jigsaw/roberta_jigsaw_race_1/attention_rollout/1.0" \
    --batch_size=64 \
    --max_seq_length=512 \
    --num_examples=200 \
    --output_dir="/scratch/yifwang/fairness_x_explainability/bias_mitigation_results_jigsaw/roberta_jigsaw_race_race_val_200_1/attention_rollout/1.0" \
    --bias_type="race" 

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="google/jigsaw_unintended_bias" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/fairness_x_explainability/explanation_debiased_models_jigsaw/roberta_jigsaw_race_2/attention_rollout/1.0" \
    --batch_size=64 \
    --max_seq_length=512 \
    --num_examples=200 \
    --output_dir="/scratch/yifwang/fairness_x_explainability/bias_mitigation_results_jigsaw/roberta_jigsaw_race_race_val_200_2/attention_rollout/1.0" \
    --bias_type="race" 

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="google/jigsaw_unintended_bias" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/fairness_x_explainability/explanation_debiased_models_jigsaw/roberta_jigsaw_race_1/attention_rollout/0.1" \
    --batch_size=64 \
    --max_seq_length=512 \
    --num_examples=200 \
    --output_dir="/scratch/yifwang/fairness_x_explainability/bias_mitigation_results_jigsaw/roberta_jigsaw_race_race_val_200_1/attention_rollout/0.1" \
    --bias_type="race" 

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="google/jigsaw_unintended_bias" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/fairness_x_explainability/explanation_debiased_models_jigsaw/roberta_jigsaw_race_2/attention_rollout/0.1" \
    --batch_size=64 \
    --max_seq_length=512 \
    --num_examples=200 \
    --output_dir="/scratch/yifwang/fairness_x_explainability/bias_mitigation_results_jigsaw/roberta_jigsaw_race_race_val_200_2/attention_rollout/0.1" \
    --bias_type="race" 

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="google/jigsaw_unintended_bias" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/fairness_x_explainability/explanation_debiased_models_jigsaw/roberta_jigsaw_race_1/attention_rollout/0.01" \
    --batch_size=64 \
    --max_seq_length=512 \
    --num_examples=200 \
    --output_dir="/scratch/yifwang/fairness_x_explainability/bias_mitigation_results_jigsaw/roberta_jigsaw_race_race_val_200_1/attention_rollout/0.01" \
    --bias_type="race" 

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="google/jigsaw_unintended_bias" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/fairness_x_explainability/explanation_debiased_models_jigsaw/roberta_jigsaw_race_2/attention_rollout/0.01" \
    --batch_size=64 \
    --max_seq_length=512 \
    --num_examples=200 \
    --output_dir="/scratch/yifwang/fairness_x_explainability/bias_mitigation_results_jigsaw/roberta_jigsaw_race_race_val_200_2/attention_rollout/0.01" \
    --bias_type="race" 

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="google/jigsaw_unintended_bias" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/fairness_x_explainability/explanation_debiased_models_jigsaw/roberta_jigsaw_race_1/Occlusion/100.0" \
    --batch_size=64 \
    --max_seq_length=512 \
    --num_examples=200 \
    --output_dir="/scratch/yifwang/fairness_x_explainability/bias_mitigation_results_jigsaw/roberta_jigsaw_race_race_val_200_1/Occlusion/100.0" \
    --bias_type="race" 

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="google/jigsaw_unintended_bias" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/fairness_x_explainability/explanation_debiased_models_jigsaw/roberta_jigsaw_race_2/Occlusion/100.0" \
    --batch_size=64 \
    --max_seq_length=512 \
    --num_examples=200 \
    --output_dir="/scratch/yifwang/fairness_x_explainability/bias_mitigation_results_jigsaw/roberta_jigsaw_race_race_val_200_2/Occlusion/100.0" \
    --bias_type="race" 

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="google/jigsaw_unintended_bias" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/fairness_x_explainability/explanation_debiased_models_jigsaw/roberta_jigsaw_race_1/Occlusion/10.0" \
    --batch_size=64 \
    --max_seq_length=512 \
    --num_examples=200 \
    --output_dir="/scratch/yifwang/fairness_x_explainability/bias_mitigation_results_jigsaw/roberta_jigsaw_race_race_val_200_1/Occlusion/10.0" \
    --bias_type="race" 

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="google/jigsaw_unintended_bias" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/fairness_x_explainability/explanation_debiased_models_jigsaw/roberta_jigsaw_race_2/Occlusion/10.0" \
    --batch_size=64 \
    --max_seq_length=512 \
    --num_examples=200 \
    --output_dir="/scratch/yifwang/fairness_x_explainability/bias_mitigation_results_jigsaw/roberta_jigsaw_race_race_val_200_2/Occlusion/10.0" \
    --bias_type="race" 

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="google/jigsaw_unintended_bias" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/fairness_x_explainability/explanation_debiased_models_jigsaw/roberta_jigsaw_race_1/Occlusion/1.0" \
    --batch_size=64 \
    --max_seq_length=512 \
    --num_examples=200 \
    --output_dir="/scratch/yifwang/fairness_x_explainability/bias_mitigation_results_jigsaw/roberta_jigsaw_race_race_val_200_1/Occlusion/1.0" \
    --bias_type="race" 

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="google/jigsaw_unintended_bias" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/fairness_x_explainability/explanation_debiased_models_jigsaw/roberta_jigsaw_race_2/Occlusion/1.0" \
    --batch_size=64 \
    --max_seq_length=512 \
    --num_examples=200 \
    --output_dir="/scratch/yifwang/fairness_x_explainability/bias_mitigation_results_jigsaw/roberta_jigsaw_race_race_val_200_2/Occlusion/1.0" \
    --bias_type="race" 

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="google/jigsaw_unintended_bias" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/fairness_x_explainability/explanation_debiased_models_jigsaw/roberta_jigsaw_race_1/Occlusion/0.1" \
    --batch_size=64 \
    --max_seq_length=512 \
    --num_examples=200 \
    --output_dir="/scratch/yifwang/fairness_x_explainability/bias_mitigation_results_jigsaw/roberta_jigsaw_race_race_val_200_1/Occlusion/0.1" \
    --bias_type="race" 

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="google/jigsaw_unintended_bias" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/fairness_x_explainability/explanation_debiased_models_jigsaw/roberta_jigsaw_race_2/Occlusion/0.1" \
    --batch_size=64 \
    --max_seq_length=512 \
    --num_examples=200 \
    --output_dir="/scratch/yifwang/fairness_x_explainability/bias_mitigation_results_jigsaw/roberta_jigsaw_race_race_val_200_2/Occlusion/0.1" \
    --bias_type="race" 

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="google/jigsaw_unintended_bias" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/fairness_x_explainability/explanation_debiased_models_jigsaw/roberta_jigsaw_race_1/Occlusion/0.01" \
    --batch_size=64 \
    --max_seq_length=512 \
    --num_examples=200 \
    --output_dir="/scratch/yifwang/fairness_x_explainability/bias_mitigation_results_jigsaw/roberta_jigsaw_race_race_val_200_1/Occlusion/0.01" \
    --bias_type="race" 

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="google/jigsaw_unintended_bias" \
    --split="val" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/fairness_x_explainability/explanation_debiased_models_jigsaw/roberta_jigsaw_race_2/Occlusion/0.01" \
    --batch_size=64 \
    --max_seq_length=512 \
    --num_examples=200 \
    --output_dir="/scratch/yifwang/fairness_x_explainability/bias_mitigation_results_jigsaw/roberta_jigsaw_race_race_val_200_2/Occlusion/0.01" \
    --bias_type="race" 


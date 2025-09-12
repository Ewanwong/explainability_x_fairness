#!/bin/bash

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="test" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/fairness_x_explainability/explanation_debiased_models_civil/bert_civil_race/Saliency/L1_100.0" \
    --batch_size=64 \
    --max_seq_length=512 \
    --num_examples=2000 \
    --output_dir="/scratch/yifwang/fairness_x_explainability/bias_mitigation_results_civil/bert_civil_race_race_test_2000/Saliency/L1_100.0" \
    --bias_type="race" 

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="test" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/fairness_x_explainability/explanation_debiased_models_civil/bert_civil_race/Saliency/L2_100.0" \
    --batch_size=64 \
    --max_seq_length=512 \
    --num_examples=2000 \
    --output_dir="/scratch/yifwang/fairness_x_explainability/bias_mitigation_results_civil/bert_civil_race_race_test_2000/Saliency/L2_100.0" \
    --bias_type="race" 

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="test" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/fairness_x_explainability/explanation_debiased_models_civil/bert_civil_race/Saliency/L1_10.0" \
    --batch_size=64 \
    --max_seq_length=512 \
    --num_examples=2000 \
    --output_dir="/scratch/yifwang/fairness_x_explainability/bias_mitigation_results_civil/bert_civil_race_race_test_2000/Saliency/L1_10.0" \
    --bias_type="race" 

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="test" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/fairness_x_explainability/explanation_debiased_models_civil/bert_civil_race/Saliency/L2_10.0" \
    --batch_size=64 \
    --max_seq_length=512 \
    --num_examples=2000 \
    --output_dir="/scratch/yifwang/fairness_x_explainability/bias_mitigation_results_civil/bert_civil_race_race_test_2000/Saliency/L2_10.0" \
    --bias_type="race" 

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="test" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/fairness_x_explainability/explanation_debiased_models_civil/bert_civil_race/Saliency/L1_1.0" \
    --batch_size=64 \
    --max_seq_length=512 \
    --num_examples=2000 \
    --output_dir="/scratch/yifwang/fairness_x_explainability/bias_mitigation_results_civil/bert_civil_race_race_test_2000/Saliency/L1_1.0" \
    --bias_type="race" 

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="test" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/fairness_x_explainability/explanation_debiased_models_civil/bert_civil_race/Saliency/L2_1.0" \
    --batch_size=64 \
    --max_seq_length=512 \
    --num_examples=2000 \
    --output_dir="/scratch/yifwang/fairness_x_explainability/bias_mitigation_results_civil/bert_civil_race_race_test_2000/Saliency/L2_1.0" \
    --bias_type="race" 

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="test" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/fairness_x_explainability/explanation_debiased_models_civil/bert_civil_race/Saliency/L1_0.1" \
    --batch_size=64 \
    --max_seq_length=512 \
    --num_examples=2000 \
    --output_dir="/scratch/yifwang/fairness_x_explainability/bias_mitigation_results_civil/bert_civil_race_race_test_2000/Saliency/L1_0.1" \
    --bias_type="race" 

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="test" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/fairness_x_explainability/explanation_debiased_models_civil/bert_civil_race/Saliency/L2_0.1" \
    --batch_size=64 \
    --max_seq_length=512 \
    --num_examples=2000 \
    --output_dir="/scratch/yifwang/fairness_x_explainability/bias_mitigation_results_civil/bert_civil_race_race_test_2000/Saliency/L2_0.1" \
    --bias_type="race" 

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="test" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/fairness_x_explainability/explanation_debiased_models_civil/bert_civil_race/Saliency/L1_0.01" \
    --batch_size=64 \
    --max_seq_length=512 \
    --num_examples=2000 \
    --output_dir="/scratch/yifwang/fairness_x_explainability/bias_mitigation_results_civil/bert_civil_race_race_test_2000/Saliency/L1_0.01" \
    --bias_type="race" 

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="test" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/fairness_x_explainability/explanation_debiased_models_civil/bert_civil_race/Saliency/L2_0.01" \
    --batch_size=64 \
    --max_seq_length=512 \
    --num_examples=2000 \
    --output_dir="/scratch/yifwang/fairness_x_explainability/bias_mitigation_results_civil/bert_civil_race_race_test_2000/Saliency/L2_0.01" \
    --bias_type="race" 

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="test" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/fairness_x_explainability/explanation_debiased_models_civil/bert_civil_race/InputXGradient/L1_100.0" \
    --batch_size=64 \
    --max_seq_length=512 \
    --num_examples=2000 \
    --output_dir="/scratch/yifwang/fairness_x_explainability/bias_mitigation_results_civil/bert_civil_race_race_test_2000/InputXGradient/L1_100.0" \
    --bias_type="race" 

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="test" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/fairness_x_explainability/explanation_debiased_models_civil/bert_civil_race/InputXGradient/L2_100.0" \
    --batch_size=64 \
    --max_seq_length=512 \
    --num_examples=2000 \
    --output_dir="/scratch/yifwang/fairness_x_explainability/bias_mitigation_results_civil/bert_civil_race_race_test_2000/InputXGradient/L2_100.0" \
    --bias_type="race" 

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="test" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/fairness_x_explainability/explanation_debiased_models_civil/bert_civil_race/InputXGradient/L1_10.0" \
    --batch_size=64 \
    --max_seq_length=512 \
    --num_examples=2000 \
    --output_dir="/scratch/yifwang/fairness_x_explainability/bias_mitigation_results_civil/bert_civil_race_race_test_2000/InputXGradient/L1_10.0" \
    --bias_type="race" 

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="test" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/fairness_x_explainability/explanation_debiased_models_civil/bert_civil_race/InputXGradient/L2_10.0" \
    --batch_size=64 \
    --max_seq_length=512 \
    --num_examples=2000 \
    --output_dir="/scratch/yifwang/fairness_x_explainability/bias_mitigation_results_civil/bert_civil_race_race_test_2000/InputXGradient/L2_10.0" \
    --bias_type="race" 

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="test" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/fairness_x_explainability/explanation_debiased_models_civil/bert_civil_race/InputXGradient/L1_1.0" \
    --batch_size=64 \
    --max_seq_length=512 \
    --num_examples=2000 \
    --output_dir="/scratch/yifwang/fairness_x_explainability/bias_mitigation_results_civil/bert_civil_race_race_test_2000/InputXGradient/L1_1.0" \
    --bias_type="race" 

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="test" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/fairness_x_explainability/explanation_debiased_models_civil/bert_civil_race/InputXGradient/L2_1.0" \
    --batch_size=64 \
    --max_seq_length=512 \
    --num_examples=2000 \
    --output_dir="/scratch/yifwang/fairness_x_explainability/bias_mitigation_results_civil/bert_civil_race_race_test_2000/InputXGradient/L2_1.0" \
    --bias_type="race" 

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="test" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/fairness_x_explainability/explanation_debiased_models_civil/bert_civil_race/InputXGradient/L1_0.1" \
    --batch_size=64 \
    --max_seq_length=512 \
    --num_examples=2000 \
    --output_dir="/scratch/yifwang/fairness_x_explainability/bias_mitigation_results_civil/bert_civil_race_race_test_2000/InputXGradient/L1_0.1" \
    --bias_type="race" 

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="test" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/fairness_x_explainability/explanation_debiased_models_civil/bert_civil_race/InputXGradient/L2_0.1" \
    --batch_size=64 \
    --max_seq_length=512 \
    --num_examples=2000 \
    --output_dir="/scratch/yifwang/fairness_x_explainability/bias_mitigation_results_civil/bert_civil_race_race_test_2000/InputXGradient/L2_0.1" \
    --bias_type="race" 

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="test" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/fairness_x_explainability/explanation_debiased_models_civil/bert_civil_race/InputXGradient/L1_0.01" \
    --batch_size=64 \
    --max_seq_length=512 \
    --num_examples=2000 \
    --output_dir="/scratch/yifwang/fairness_x_explainability/bias_mitigation_results_civil/bert_civil_race_race_test_2000/InputXGradient/L1_0.01" \
    --bias_type="race" 

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="test" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/fairness_x_explainability/explanation_debiased_models_civil/bert_civil_race/InputXGradient/L2_0.01" \
    --batch_size=64 \
    --max_seq_length=512 \
    --num_examples=2000 \
    --output_dir="/scratch/yifwang/fairness_x_explainability/bias_mitigation_results_civil/bert_civil_race_race_test_2000/InputXGradient/L2_0.01" \
    --bias_type="race" 

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="test" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/fairness_x_explainability/explanation_debiased_models_civil/bert_civil_race/raw_attention/100.0" \
    --batch_size=64 \
    --max_seq_length=512 \
    --num_examples=2000 \
    --output_dir="/scratch/yifwang/fairness_x_explainability/bias_mitigation_results_civil/bert_civil_race_race_test_2000/raw_attention/100.0" \
    --bias_type="race" 

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="test" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/fairness_x_explainability/explanation_debiased_models_civil/bert_civil_race/raw_attention/10.0" \
    --batch_size=64 \
    --max_seq_length=512 \
    --num_examples=2000 \
    --output_dir="/scratch/yifwang/fairness_x_explainability/bias_mitigation_results_civil/bert_civil_race_race_test_2000/raw_attention/10.0" \
    --bias_type="race" 

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="test" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/fairness_x_explainability/explanation_debiased_models_civil/bert_civil_race/raw_attention/1.0" \
    --batch_size=64 \
    --max_seq_length=512 \
    --num_examples=2000 \
    --output_dir="/scratch/yifwang/fairness_x_explainability/bias_mitigation_results_civil/bert_civil_race_race_test_2000/raw_attention/1.0" \
    --bias_type="race" 

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="test" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/fairness_x_explainability/explanation_debiased_models_civil/bert_civil_race/raw_attention/0.1" \
    --batch_size=64 \
    --max_seq_length=512 \
    --num_examples=2000 \
    --output_dir="/scratch/yifwang/fairness_x_explainability/bias_mitigation_results_civil/bert_civil_race_race_test_2000/raw_attention/0.1" \
    --bias_type="race" 

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="test" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/fairness_x_explainability/explanation_debiased_models_civil/bert_civil_race/raw_attention/0.01" \
    --batch_size=64 \
    --max_seq_length=512 \
    --num_examples=2000 \
    --output_dir="/scratch/yifwang/fairness_x_explainability/bias_mitigation_results_civil/bert_civil_race_race_test_2000/raw_attention/0.01" \
    --bias_type="race" 

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="test" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/fairness_x_explainability/explanation_debiased_models_civil/bert_civil_race/attention_flow/100.0" \
    --batch_size=64 \
    --max_seq_length=512 \
    --num_examples=2000 \
    --output_dir="/scratch/yifwang/fairness_x_explainability/bias_mitigation_results_civil/bert_civil_race_race_test_2000/attention_flow/100.0" \
    --bias_type="race" 

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="test" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/fairness_x_explainability/explanation_debiased_models_civil/bert_civil_race/attention_flow/10.0" \
    --batch_size=64 \
    --max_seq_length=512 \
    --num_examples=2000 \
    --output_dir="/scratch/yifwang/fairness_x_explainability/bias_mitigation_results_civil/bert_civil_race_race_test_2000/attention_flow/10.0" \
    --bias_type="race" 

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="test" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/fairness_x_explainability/explanation_debiased_models_civil/bert_civil_race/attention_flow/1.0" \
    --batch_size=64 \
    --max_seq_length=512 \
    --num_examples=2000 \
    --output_dir="/scratch/yifwang/fairness_x_explainability/bias_mitigation_results_civil/bert_civil_race_race_test_2000/attention_flow/1.0" \
    --bias_type="race" 

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="test" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/fairness_x_explainability/explanation_debiased_models_civil/bert_civil_race/attention_flow/0.1" \
    --batch_size=64 \
    --max_seq_length=512 \
    --num_examples=2000 \
    --output_dir="/scratch/yifwang/fairness_x_explainability/bias_mitigation_results_civil/bert_civil_race_race_test_2000/attention_flow/0.1" \
    --bias_type="race" 

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="test" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/fairness_x_explainability/explanation_debiased_models_civil/bert_civil_race/attention_flow/0.01" \
    --batch_size=64 \
    --max_seq_length=512 \
    --num_examples=2000 \
    --output_dir="/scratch/yifwang/fairness_x_explainability/bias_mitigation_results_civil/bert_civil_race_race_test_2000/attention_flow/0.01" \
    --bias_type="race" 

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="test" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/fairness_x_explainability/explanation_debiased_models_civil/bert_civil_race/attention_rollout/100.0" \
    --batch_size=64 \
    --max_seq_length=512 \
    --num_examples=2000 \
    --output_dir="/scratch/yifwang/fairness_x_explainability/bias_mitigation_results_civil/bert_civil_race_race_test_2000/attention_rollout/100.0" \
    --bias_type="race" 

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="test" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/fairness_x_explainability/explanation_debiased_models_civil/bert_civil_race/attention_rollout/10.0" \
    --batch_size=64 \
    --max_seq_length=512 \
    --num_examples=2000 \
    --output_dir="/scratch/yifwang/fairness_x_explainability/bias_mitigation_results_civil/bert_civil_race_race_test_2000/attention_rollout/10.0" \
    --bias_type="race" 

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="test" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/fairness_x_explainability/explanation_debiased_models_civil/bert_civil_race/attention_rollout/1.0" \
    --batch_size=64 \
    --max_seq_length=512 \
    --num_examples=2000 \
    --output_dir="/scratch/yifwang/fairness_x_explainability/bias_mitigation_results_civil/bert_civil_race_race_test_2000/attention_rollout/1.0" \
    --bias_type="race" 

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="test" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/fairness_x_explainability/explanation_debiased_models_civil/bert_civil_race/attention_rollout/0.1" \
    --batch_size=64 \
    --max_seq_length=512 \
    --num_examples=2000 \
    --output_dir="/scratch/yifwang/fairness_x_explainability/bias_mitigation_results_civil/bert_civil_race_race_test_2000/attention_rollout/0.1" \
    --bias_type="race" 

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="test" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/fairness_x_explainability/explanation_debiased_models_civil/bert_civil_race/attention_rollout/0.01" \
    --batch_size=64 \
    --max_seq_length=512 \
    --num_examples=2000 \
    --output_dir="/scratch/yifwang/fairness_x_explainability/bias_mitigation_results_civil/bert_civil_race_race_test_2000/attention_rollout/0.01" \
    --bias_type="race" 

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="test" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/fairness_x_explainability/explanation_debiased_models_civil/bert_civil_race/Occlusion/100.0" \
    --batch_size=64 \
    --max_seq_length=512 \
    --num_examples=2000 \
    --output_dir="/scratch/yifwang/fairness_x_explainability/bias_mitigation_results_civil/bert_civil_race_race_test_2000/Occlusion/100.0" \
    --bias_type="race" 

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="test" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/fairness_x_explainability/explanation_debiased_models_civil/bert_civil_race/Occlusion/10.0" \
    --batch_size=64 \
    --max_seq_length=512 \
    --num_examples=2000 \
    --output_dir="/scratch/yifwang/fairness_x_explainability/bias_mitigation_results_civil/bert_civil_race_race_test_2000/Occlusion/10.0" \
    --bias_type="race" 

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="test" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/fairness_x_explainability/explanation_debiased_models_civil/bert_civil_race/Occlusion/1.0" \
    --batch_size=64 \
    --max_seq_length=512 \
    --num_examples=2000 \
    --output_dir="/scratch/yifwang/fairness_x_explainability/bias_mitigation_results_civil/bert_civil_race_race_test_2000/Occlusion/1.0" \
    --bias_type="race" 

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="test" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/fairness_x_explainability/explanation_debiased_models_civil/bert_civil_race/Occlusion/0.1" \
    --batch_size=64 \
    --max_seq_length=512 \
    --num_examples=2000 \
    --output_dir="/scratch/yifwang/fairness_x_explainability/bias_mitigation_results_civil/bert_civil_race_race_test_2000/Occlusion/0.1" \
    --bias_type="race" 

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="test" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/fairness_x_explainability/explanation_debiased_models_civil/bert_civil_race/Occlusion/0.01" \
    --batch_size=64 \
    --max_seq_length=512 \
    --num_examples=2000 \
    --output_dir="/scratch/yifwang/fairness_x_explainability/bias_mitigation_results_civil/bert_civil_race_race_test_2000/Occlusion/0.01" \
    --bias_type="race" 

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="test" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/fairness_x_explainability/explanation_debiased_models_civil/bert_civil_gender/Saliency/L1_100.0" \
    --batch_size=64 \
    --max_seq_length=512 \
    --num_examples=2000 \
    --output_dir="/scratch/yifwang/fairness_x_explainability/bias_mitigation_results_civil/bert_civil_gender_gender_test_2000/Saliency/L1_100.0" \
    --bias_type="gender" 

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="test" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/fairness_x_explainability/explanation_debiased_models_civil/bert_civil_gender/Saliency/L2_100.0" \
    --batch_size=64 \
    --max_seq_length=512 \
    --num_examples=2000 \
    --output_dir="/scratch/yifwang/fairness_x_explainability/bias_mitigation_results_civil/bert_civil_gender_gender_test_2000/Saliency/L2_100.0" \
    --bias_type="gender" 

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="test" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/fairness_x_explainability/explanation_debiased_models_civil/bert_civil_gender/Saliency/L1_10.0" \
    --batch_size=64 \
    --max_seq_length=512 \
    --num_examples=2000 \
    --output_dir="/scratch/yifwang/fairness_x_explainability/bias_mitigation_results_civil/bert_civil_gender_gender_test_2000/Saliency/L1_10.0" \
    --bias_type="gender" 

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="test" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/fairness_x_explainability/explanation_debiased_models_civil/bert_civil_gender/Saliency/L2_10.0" \
    --batch_size=64 \
    --max_seq_length=512 \
    --num_examples=2000 \
    --output_dir="/scratch/yifwang/fairness_x_explainability/bias_mitigation_results_civil/bert_civil_gender_gender_test_2000/Saliency/L2_10.0" \
    --bias_type="gender" 

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="test" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/fairness_x_explainability/explanation_debiased_models_civil/bert_civil_gender/Saliency/L1_1.0" \
    --batch_size=64 \
    --max_seq_length=512 \
    --num_examples=2000 \
    --output_dir="/scratch/yifwang/fairness_x_explainability/bias_mitigation_results_civil/bert_civil_gender_gender_test_2000/Saliency/L1_1.0" \
    --bias_type="gender" 

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="test" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/fairness_x_explainability/explanation_debiased_models_civil/bert_civil_gender/Saliency/L2_1.0" \
    --batch_size=64 \
    --max_seq_length=512 \
    --num_examples=2000 \
    --output_dir="/scratch/yifwang/fairness_x_explainability/bias_mitigation_results_civil/bert_civil_gender_gender_test_2000/Saliency/L2_1.0" \
    --bias_type="gender" 

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="test" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/fairness_x_explainability/explanation_debiased_models_civil/bert_civil_gender/Saliency/L1_0.1" \
    --batch_size=64 \
    --max_seq_length=512 \
    --num_examples=2000 \
    --output_dir="/scratch/yifwang/fairness_x_explainability/bias_mitigation_results_civil/bert_civil_gender_gender_test_2000/Saliency/L1_0.1" \
    --bias_type="gender" 

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="test" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/fairness_x_explainability/explanation_debiased_models_civil/bert_civil_gender/Saliency/L2_0.1" \
    --batch_size=64 \
    --max_seq_length=512 \
    --num_examples=2000 \
    --output_dir="/scratch/yifwang/fairness_x_explainability/bias_mitigation_results_civil/bert_civil_gender_gender_test_2000/Saliency/L2_0.1" \
    --bias_type="gender" 

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="test" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/fairness_x_explainability/explanation_debiased_models_civil/bert_civil_gender/Saliency/L1_0.01" \
    --batch_size=64 \
    --max_seq_length=512 \
    --num_examples=2000 \
    --output_dir="/scratch/yifwang/fairness_x_explainability/bias_mitigation_results_civil/bert_civil_gender_gender_test_2000/Saliency/L1_0.01" \
    --bias_type="gender" 

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="test" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/fairness_x_explainability/explanation_debiased_models_civil/bert_civil_gender/Saliency/L2_0.01" \
    --batch_size=64 \
    --max_seq_length=512 \
    --num_examples=2000 \
    --output_dir="/scratch/yifwang/fairness_x_explainability/bias_mitigation_results_civil/bert_civil_gender_gender_test_2000/Saliency/L2_0.01" \
    --bias_type="gender" 

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="test" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/fairness_x_explainability/explanation_debiased_models_civil/bert_civil_gender/InputXGradient/L1_100.0" \
    --batch_size=64 \
    --max_seq_length=512 \
    --num_examples=2000 \
    --output_dir="/scratch/yifwang/fairness_x_explainability/bias_mitigation_results_civil/bert_civil_gender_gender_test_2000/InputXGradient/L1_100.0" \
    --bias_type="gender" 

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="test" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/fairness_x_explainability/explanation_debiased_models_civil/bert_civil_gender/InputXGradient/L2_100.0" \
    --batch_size=64 \
    --max_seq_length=512 \
    --num_examples=2000 \
    --output_dir="/scratch/yifwang/fairness_x_explainability/bias_mitigation_results_civil/bert_civil_gender_gender_test_2000/InputXGradient/L2_100.0" \
    --bias_type="gender" 

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="test" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/fairness_x_explainability/explanation_debiased_models_civil/bert_civil_gender/InputXGradient/L1_10.0" \
    --batch_size=64 \
    --max_seq_length=512 \
    --num_examples=2000 \
    --output_dir="/scratch/yifwang/fairness_x_explainability/bias_mitigation_results_civil/bert_civil_gender_gender_test_2000/InputXGradient/L1_10.0" \
    --bias_type="gender" 

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="test" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/fairness_x_explainability/explanation_debiased_models_civil/bert_civil_gender/InputXGradient/L2_10.0" \
    --batch_size=64 \
    --max_seq_length=512 \
    --num_examples=2000 \
    --output_dir="/scratch/yifwang/fairness_x_explainability/bias_mitigation_results_civil/bert_civil_gender_gender_test_2000/InputXGradient/L2_10.0" \
    --bias_type="gender" 

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="test" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/fairness_x_explainability/explanation_debiased_models_civil/bert_civil_gender/InputXGradient/L1_1.0" \
    --batch_size=64 \
    --max_seq_length=512 \
    --num_examples=2000 \
    --output_dir="/scratch/yifwang/fairness_x_explainability/bias_mitigation_results_civil/bert_civil_gender_gender_test_2000/InputXGradient/L1_1.0" \
    --bias_type="gender" 

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="test" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/fairness_x_explainability/explanation_debiased_models_civil/bert_civil_gender/InputXGradient/L2_1.0" \
    --batch_size=64 \
    --max_seq_length=512 \
    --num_examples=2000 \
    --output_dir="/scratch/yifwang/fairness_x_explainability/bias_mitigation_results_civil/bert_civil_gender_gender_test_2000/InputXGradient/L2_1.0" \
    --bias_type="gender" 

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="test" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/fairness_x_explainability/explanation_debiased_models_civil/bert_civil_gender/InputXGradient/L1_0.1" \
    --batch_size=64 \
    --max_seq_length=512 \
    --num_examples=2000 \
    --output_dir="/scratch/yifwang/fairness_x_explainability/bias_mitigation_results_civil/bert_civil_gender_gender_test_2000/InputXGradient/L1_0.1" \
    --bias_type="gender" 

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="test" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/fairness_x_explainability/explanation_debiased_models_civil/bert_civil_gender/InputXGradient/L2_0.1" \
    --batch_size=64 \
    --max_seq_length=512 \
    --num_examples=2000 \
    --output_dir="/scratch/yifwang/fairness_x_explainability/bias_mitigation_results_civil/bert_civil_gender_gender_test_2000/InputXGradient/L2_0.1" \
    --bias_type="gender" 

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="test" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/fairness_x_explainability/explanation_debiased_models_civil/bert_civil_gender/InputXGradient/L1_0.01" \
    --batch_size=64 \
    --max_seq_length=512 \
    --num_examples=2000 \
    --output_dir="/scratch/yifwang/fairness_x_explainability/bias_mitigation_results_civil/bert_civil_gender_gender_test_2000/InputXGradient/L1_0.01" \
    --bias_type="gender" 

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="test" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/fairness_x_explainability/explanation_debiased_models_civil/bert_civil_gender/InputXGradient/L2_0.01" \
    --batch_size=64 \
    --max_seq_length=512 \
    --num_examples=2000 \
    --output_dir="/scratch/yifwang/fairness_x_explainability/bias_mitigation_results_civil/bert_civil_gender_gender_test_2000/InputXGradient/L2_0.01" \
    --bias_type="gender" 

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="test" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/fairness_x_explainability/explanation_debiased_models_civil/bert_civil_gender/raw_attention/100.0" \
    --batch_size=64 \
    --max_seq_length=512 \
    --num_examples=2000 \
    --output_dir="/scratch/yifwang/fairness_x_explainability/bias_mitigation_results_civil/bert_civil_gender_gender_test_2000/raw_attention/100.0" \
    --bias_type="gender" 

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="test" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/fairness_x_explainability/explanation_debiased_models_civil/bert_civil_gender/raw_attention/10.0" \
    --batch_size=64 \
    --max_seq_length=512 \
    --num_examples=2000 \
    --output_dir="/scratch/yifwang/fairness_x_explainability/bias_mitigation_results_civil/bert_civil_gender_gender_test_2000/raw_attention/10.0" \
    --bias_type="gender" 

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="test" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/fairness_x_explainability/explanation_debiased_models_civil/bert_civil_gender/raw_attention/1.0" \
    --batch_size=64 \
    --max_seq_length=512 \
    --num_examples=2000 \
    --output_dir="/scratch/yifwang/fairness_x_explainability/bias_mitigation_results_civil/bert_civil_gender_gender_test_2000/raw_attention/1.0" \
    --bias_type="gender" 

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="test" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/fairness_x_explainability/explanation_debiased_models_civil/bert_civil_gender/raw_attention/0.1" \
    --batch_size=64 \
    --max_seq_length=512 \
    --num_examples=2000 \
    --output_dir="/scratch/yifwang/fairness_x_explainability/bias_mitigation_results_civil/bert_civil_gender_gender_test_2000/raw_attention/0.1" \
    --bias_type="gender" 

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="test" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/fairness_x_explainability/explanation_debiased_models_civil/bert_civil_gender/raw_attention/0.01" \
    --batch_size=64 \
    --max_seq_length=512 \
    --num_examples=2000 \
    --output_dir="/scratch/yifwang/fairness_x_explainability/bias_mitigation_results_civil/bert_civil_gender_gender_test_2000/raw_attention/0.01" \
    --bias_type="gender" 

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="test" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/fairness_x_explainability/explanation_debiased_models_civil/bert_civil_gender/attention_flow/100.0" \
    --batch_size=64 \
    --max_seq_length=512 \
    --num_examples=2000 \
    --output_dir="/scratch/yifwang/fairness_x_explainability/bias_mitigation_results_civil/bert_civil_gender_gender_test_2000/attention_flow/100.0" \
    --bias_type="gender" 

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="test" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/fairness_x_explainability/explanation_debiased_models_civil/bert_civil_gender/attention_flow/10.0" \
    --batch_size=64 \
    --max_seq_length=512 \
    --num_examples=2000 \
    --output_dir="/scratch/yifwang/fairness_x_explainability/bias_mitigation_results_civil/bert_civil_gender_gender_test_2000/attention_flow/10.0" \
    --bias_type="gender" 

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="test" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/fairness_x_explainability/explanation_debiased_models_civil/bert_civil_gender/attention_flow/1.0" \
    --batch_size=64 \
    --max_seq_length=512 \
    --num_examples=2000 \
    --output_dir="/scratch/yifwang/fairness_x_explainability/bias_mitigation_results_civil/bert_civil_gender_gender_test_2000/attention_flow/1.0" \
    --bias_type="gender" 

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="test" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/fairness_x_explainability/explanation_debiased_models_civil/bert_civil_gender/attention_flow/0.1" \
    --batch_size=64 \
    --max_seq_length=512 \
    --num_examples=2000 \
    --output_dir="/scratch/yifwang/fairness_x_explainability/bias_mitigation_results_civil/bert_civil_gender_gender_test_2000/attention_flow/0.1" \
    --bias_type="gender" 

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="test" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/fairness_x_explainability/explanation_debiased_models_civil/bert_civil_gender/attention_flow/0.01" \
    --batch_size=64 \
    --max_seq_length=512 \
    --num_examples=2000 \
    --output_dir="/scratch/yifwang/fairness_x_explainability/bias_mitigation_results_civil/bert_civil_gender_gender_test_2000/attention_flow/0.01" \
    --bias_type="gender" 

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="test" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/fairness_x_explainability/explanation_debiased_models_civil/bert_civil_gender/attention_rollout/100.0" \
    --batch_size=64 \
    --max_seq_length=512 \
    --num_examples=2000 \
    --output_dir="/scratch/yifwang/fairness_x_explainability/bias_mitigation_results_civil/bert_civil_gender_gender_test_2000/attention_rollout/100.0" \
    --bias_type="gender" 

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="test" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/fairness_x_explainability/explanation_debiased_models_civil/bert_civil_gender/attention_rollout/10.0" \
    --batch_size=64 \
    --max_seq_length=512 \
    --num_examples=2000 \
    --output_dir="/scratch/yifwang/fairness_x_explainability/bias_mitigation_results_civil/bert_civil_gender_gender_test_2000/attention_rollout/10.0" \
    --bias_type="gender" 

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="test" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/fairness_x_explainability/explanation_debiased_models_civil/bert_civil_gender/attention_rollout/1.0" \
    --batch_size=64 \
    --max_seq_length=512 \
    --num_examples=2000 \
    --output_dir="/scratch/yifwang/fairness_x_explainability/bias_mitigation_results_civil/bert_civil_gender_gender_test_2000/attention_rollout/1.0" \
    --bias_type="gender" 

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="test" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/fairness_x_explainability/explanation_debiased_models_civil/bert_civil_gender/attention_rollout/0.1" \
    --batch_size=64 \
    --max_seq_length=512 \
    --num_examples=2000 \
    --output_dir="/scratch/yifwang/fairness_x_explainability/bias_mitigation_results_civil/bert_civil_gender_gender_test_2000/attention_rollout/0.1" \
    --bias_type="gender" 

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="test" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/fairness_x_explainability/explanation_debiased_models_civil/bert_civil_gender/attention_rollout/0.01" \
    --batch_size=64 \
    --max_seq_length=512 \
    --num_examples=2000 \
    --output_dir="/scratch/yifwang/fairness_x_explainability/bias_mitigation_results_civil/bert_civil_gender_gender_test_2000/attention_rollout/0.01" \
    --bias_type="gender" 

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="test" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/fairness_x_explainability/explanation_debiased_models_civil/bert_civil_gender/Occlusion/100.0" \
    --batch_size=64 \
    --max_seq_length=512 \
    --num_examples=2000 \
    --output_dir="/scratch/yifwang/fairness_x_explainability/bias_mitigation_results_civil/bert_civil_gender_gender_test_2000/Occlusion/100.0" \
    --bias_type="gender" 

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="test" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/fairness_x_explainability/explanation_debiased_models_civil/bert_civil_gender/Occlusion/10.0" \
    --batch_size=64 \
    --max_seq_length=512 \
    --num_examples=2000 \
    --output_dir="/scratch/yifwang/fairness_x_explainability/bias_mitigation_results_civil/bert_civil_gender_gender_test_2000/Occlusion/10.0" \
    --bias_type="gender" 

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="test" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/fairness_x_explainability/explanation_debiased_models_civil/bert_civil_gender/Occlusion/1.0" \
    --batch_size=64 \
    --max_seq_length=512 \
    --num_examples=2000 \
    --output_dir="/scratch/yifwang/fairness_x_explainability/bias_mitigation_results_civil/bert_civil_gender_gender_test_2000/Occlusion/1.0" \
    --bias_type="gender" 

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="test" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/fairness_x_explainability/explanation_debiased_models_civil/bert_civil_gender/Occlusion/0.1" \
    --batch_size=64 \
    --max_seq_length=512 \
    --num_examples=2000 \
    --output_dir="/scratch/yifwang/fairness_x_explainability/bias_mitigation_results_civil/bert_civil_gender_gender_test_2000/Occlusion/0.1" \
    --bias_type="gender" 

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="test" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/fairness_x_explainability/explanation_debiased_models_civil/bert_civil_gender/Occlusion/0.01" \
    --batch_size=64 \
    --max_seq_length=512 \
    --num_examples=2000 \
    --output_dir="/scratch/yifwang/fairness_x_explainability/bias_mitigation_results_civil/bert_civil_gender_gender_test_2000/Occlusion/0.01" \
    --bias_type="gender" 

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="test" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/fairness_x_explainability/explanation_debiased_models_civil/bert_civil_religion/Saliency/L1_100.0" \
    --batch_size=64 \
    --max_seq_length=512 \
    --num_examples=1000 \
    --output_dir="/scratch/yifwang/fairness_x_explainability/bias_mitigation_results_civil/bert_civil_religion_religion_test_1000/Saliency/L1_100.0" \
    --bias_type="religion" 

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="test" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/fairness_x_explainability/explanation_debiased_models_civil/bert_civil_religion/Saliency/L2_100.0" \
    --batch_size=64 \
    --max_seq_length=512 \
    --num_examples=1000 \
    --output_dir="/scratch/yifwang/fairness_x_explainability/bias_mitigation_results_civil/bert_civil_religion_religion_test_1000/Saliency/L2_100.0" \
    --bias_type="religion" 

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="test" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/fairness_x_explainability/explanation_debiased_models_civil/bert_civil_religion/Saliency/L1_10.0" \
    --batch_size=64 \
    --max_seq_length=512 \
    --num_examples=1000 \
    --output_dir="/scratch/yifwang/fairness_x_explainability/bias_mitigation_results_civil/bert_civil_religion_religion_test_1000/Saliency/L1_10.0" \
    --bias_type="religion" 

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="test" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/fairness_x_explainability/explanation_debiased_models_civil/bert_civil_religion/Saliency/L2_10.0" \
    --batch_size=64 \
    --max_seq_length=512 \
    --num_examples=1000 \
    --output_dir="/scratch/yifwang/fairness_x_explainability/bias_mitigation_results_civil/bert_civil_religion_religion_test_1000/Saliency/L2_10.0" \
    --bias_type="religion" 

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="test" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/fairness_x_explainability/explanation_debiased_models_civil/bert_civil_religion/Saliency/L1_1.0" \
    --batch_size=64 \
    --max_seq_length=512 \
    --num_examples=1000 \
    --output_dir="/scratch/yifwang/fairness_x_explainability/bias_mitigation_results_civil/bert_civil_religion_religion_test_1000/Saliency/L1_1.0" \
    --bias_type="religion" 

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="test" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/fairness_x_explainability/explanation_debiased_models_civil/bert_civil_religion/Saliency/L2_1.0" \
    --batch_size=64 \
    --max_seq_length=512 \
    --num_examples=1000 \
    --output_dir="/scratch/yifwang/fairness_x_explainability/bias_mitigation_results_civil/bert_civil_religion_religion_test_1000/Saliency/L2_1.0" \
    --bias_type="religion" 

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="test" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/fairness_x_explainability/explanation_debiased_models_civil/bert_civil_religion/Saliency/L1_0.1" \
    --batch_size=64 \
    --max_seq_length=512 \
    --num_examples=1000 \
    --output_dir="/scratch/yifwang/fairness_x_explainability/bias_mitigation_results_civil/bert_civil_religion_religion_test_1000/Saliency/L1_0.1" \
    --bias_type="religion" 

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="test" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/fairness_x_explainability/explanation_debiased_models_civil/bert_civil_religion/Saliency/L2_0.1" \
    --batch_size=64 \
    --max_seq_length=512 \
    --num_examples=1000 \
    --output_dir="/scratch/yifwang/fairness_x_explainability/bias_mitigation_results_civil/bert_civil_religion_religion_test_1000/Saliency/L2_0.1" \
    --bias_type="religion" 

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="test" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/fairness_x_explainability/explanation_debiased_models_civil/bert_civil_religion/Saliency/L1_0.01" \
    --batch_size=64 \
    --max_seq_length=512 \
    --num_examples=1000 \
    --output_dir="/scratch/yifwang/fairness_x_explainability/bias_mitigation_results_civil/bert_civil_religion_religion_test_1000/Saliency/L1_0.01" \
    --bias_type="religion" 

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="test" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/fairness_x_explainability/explanation_debiased_models_civil/bert_civil_religion/Saliency/L2_0.01" \
    --batch_size=64 \
    --max_seq_length=512 \
    --num_examples=1000 \
    --output_dir="/scratch/yifwang/fairness_x_explainability/bias_mitigation_results_civil/bert_civil_religion_religion_test_1000/Saliency/L2_0.01" \
    --bias_type="religion" 

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="test" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/fairness_x_explainability/explanation_debiased_models_civil/bert_civil_religion/InputXGradient/L1_100.0" \
    --batch_size=64 \
    --max_seq_length=512 \
    --num_examples=1000 \
    --output_dir="/scratch/yifwang/fairness_x_explainability/bias_mitigation_results_civil/bert_civil_religion_religion_test_1000/InputXGradient/L1_100.0" \
    --bias_type="religion" 

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="test" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/fairness_x_explainability/explanation_debiased_models_civil/bert_civil_religion/InputXGradient/L2_100.0" \
    --batch_size=64 \
    --max_seq_length=512 \
    --num_examples=1000 \
    --output_dir="/scratch/yifwang/fairness_x_explainability/bias_mitigation_results_civil/bert_civil_religion_religion_test_1000/InputXGradient/L2_100.0" \
    --bias_type="religion" 

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="test" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/fairness_x_explainability/explanation_debiased_models_civil/bert_civil_religion/InputXGradient/L1_10.0" \
    --batch_size=64 \
    --max_seq_length=512 \
    --num_examples=1000 \
    --output_dir="/scratch/yifwang/fairness_x_explainability/bias_mitigation_results_civil/bert_civil_religion_religion_test_1000/InputXGradient/L1_10.0" \
    --bias_type="religion" 

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="test" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/fairness_x_explainability/explanation_debiased_models_civil/bert_civil_religion/InputXGradient/L2_10.0" \
    --batch_size=64 \
    --max_seq_length=512 \
    --num_examples=1000 \
    --output_dir="/scratch/yifwang/fairness_x_explainability/bias_mitigation_results_civil/bert_civil_religion_religion_test_1000/InputXGradient/L2_10.0" \
    --bias_type="religion" 

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="test" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/fairness_x_explainability/explanation_debiased_models_civil/bert_civil_religion/InputXGradient/L1_1.0" \
    --batch_size=64 \
    --max_seq_length=512 \
    --num_examples=1000 \
    --output_dir="/scratch/yifwang/fairness_x_explainability/bias_mitigation_results_civil/bert_civil_religion_religion_test_1000/InputXGradient/L1_1.0" \
    --bias_type="religion" 

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="test" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/fairness_x_explainability/explanation_debiased_models_civil/bert_civil_religion/InputXGradient/L2_1.0" \
    --batch_size=64 \
    --max_seq_length=512 \
    --num_examples=1000 \
    --output_dir="/scratch/yifwang/fairness_x_explainability/bias_mitigation_results_civil/bert_civil_religion_religion_test_1000/InputXGradient/L2_1.0" \
    --bias_type="religion" 

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="test" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/fairness_x_explainability/explanation_debiased_models_civil/bert_civil_religion/InputXGradient/L1_0.1" \
    --batch_size=64 \
    --max_seq_length=512 \
    --num_examples=1000 \
    --output_dir="/scratch/yifwang/fairness_x_explainability/bias_mitigation_results_civil/bert_civil_religion_religion_test_1000/InputXGradient/L1_0.1" \
    --bias_type="religion" 

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="test" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/fairness_x_explainability/explanation_debiased_models_civil/bert_civil_religion/InputXGradient/L2_0.1" \
    --batch_size=64 \
    --max_seq_length=512 \
    --num_examples=1000 \
    --output_dir="/scratch/yifwang/fairness_x_explainability/bias_mitigation_results_civil/bert_civil_religion_religion_test_1000/InputXGradient/L2_0.1" \
    --bias_type="religion" 

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="test" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/fairness_x_explainability/explanation_debiased_models_civil/bert_civil_religion/InputXGradient/L1_0.01" \
    --batch_size=64 \
    --max_seq_length=512 \
    --num_examples=1000 \
    --output_dir="/scratch/yifwang/fairness_x_explainability/bias_mitigation_results_civil/bert_civil_religion_religion_test_1000/InputXGradient/L1_0.01" \
    --bias_type="religion" 

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="test" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/fairness_x_explainability/explanation_debiased_models_civil/bert_civil_religion/InputXGradient/L2_0.01" \
    --batch_size=64 \
    --max_seq_length=512 \
    --num_examples=1000 \
    --output_dir="/scratch/yifwang/fairness_x_explainability/bias_mitigation_results_civil/bert_civil_religion_religion_test_1000/InputXGradient/L2_0.01" \
    --bias_type="religion" 

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="test" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/fairness_x_explainability/explanation_debiased_models_civil/bert_civil_religion/raw_attention/100.0" \
    --batch_size=64 \
    --max_seq_length=512 \
    --num_examples=1000 \
    --output_dir="/scratch/yifwang/fairness_x_explainability/bias_mitigation_results_civil/bert_civil_religion_religion_test_1000/raw_attention/100.0" \
    --bias_type="religion" 

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="test" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/fairness_x_explainability/explanation_debiased_models_civil/bert_civil_religion/raw_attention/10.0" \
    --batch_size=64 \
    --max_seq_length=512 \
    --num_examples=1000 \
    --output_dir="/scratch/yifwang/fairness_x_explainability/bias_mitigation_results_civil/bert_civil_religion_religion_test_1000/raw_attention/10.0" \
    --bias_type="religion" 

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="test" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/fairness_x_explainability/explanation_debiased_models_civil/bert_civil_religion/raw_attention/1.0" \
    --batch_size=64 \
    --max_seq_length=512 \
    --num_examples=1000 \
    --output_dir="/scratch/yifwang/fairness_x_explainability/bias_mitigation_results_civil/bert_civil_religion_religion_test_1000/raw_attention/1.0" \
    --bias_type="religion" 

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="test" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/fairness_x_explainability/explanation_debiased_models_civil/bert_civil_religion/raw_attention/0.1" \
    --batch_size=64 \
    --max_seq_length=512 \
    --num_examples=1000 \
    --output_dir="/scratch/yifwang/fairness_x_explainability/bias_mitigation_results_civil/bert_civil_religion_religion_test_1000/raw_attention/0.1" \
    --bias_type="religion" 

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="test" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/fairness_x_explainability/explanation_debiased_models_civil/bert_civil_religion/raw_attention/0.01" \
    --batch_size=64 \
    --max_seq_length=512 \
    --num_examples=1000 \
    --output_dir="/scratch/yifwang/fairness_x_explainability/bias_mitigation_results_civil/bert_civil_religion_religion_test_1000/raw_attention/0.01" \
    --bias_type="religion" 

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="test" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/fairness_x_explainability/explanation_debiased_models_civil/bert_civil_religion/attention_flow/100.0" \
    --batch_size=64 \
    --max_seq_length=512 \
    --num_examples=1000 \
    --output_dir="/scratch/yifwang/fairness_x_explainability/bias_mitigation_results_civil/bert_civil_religion_religion_test_1000/attention_flow/100.0" \
    --bias_type="religion" 

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="test" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/fairness_x_explainability/explanation_debiased_models_civil/bert_civil_religion/attention_flow/10.0" \
    --batch_size=64 \
    --max_seq_length=512 \
    --num_examples=1000 \
    --output_dir="/scratch/yifwang/fairness_x_explainability/bias_mitigation_results_civil/bert_civil_religion_religion_test_1000/attention_flow/10.0" \
    --bias_type="religion" 

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="test" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/fairness_x_explainability/explanation_debiased_models_civil/bert_civil_religion/attention_flow/1.0" \
    --batch_size=64 \
    --max_seq_length=512 \
    --num_examples=1000 \
    --output_dir="/scratch/yifwang/fairness_x_explainability/bias_mitigation_results_civil/bert_civil_religion_religion_test_1000/attention_flow/1.0" \
    --bias_type="religion" 

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="test" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/fairness_x_explainability/explanation_debiased_models_civil/bert_civil_religion/attention_flow/0.1" \
    --batch_size=64 \
    --max_seq_length=512 \
    --num_examples=1000 \
    --output_dir="/scratch/yifwang/fairness_x_explainability/bias_mitigation_results_civil/bert_civil_religion_religion_test_1000/attention_flow/0.1" \
    --bias_type="religion" 

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="test" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/fairness_x_explainability/explanation_debiased_models_civil/bert_civil_religion/attention_flow/0.01" \
    --batch_size=64 \
    --max_seq_length=512 \
    --num_examples=1000 \
    --output_dir="/scratch/yifwang/fairness_x_explainability/bias_mitigation_results_civil/bert_civil_religion_religion_test_1000/attention_flow/0.01" \
    --bias_type="religion" 

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="test" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/fairness_x_explainability/explanation_debiased_models_civil/bert_civil_religion/attention_rollout/100.0" \
    --batch_size=64 \
    --max_seq_length=512 \
    --num_examples=1000 \
    --output_dir="/scratch/yifwang/fairness_x_explainability/bias_mitigation_results_civil/bert_civil_religion_religion_test_1000/attention_rollout/100.0" \
    --bias_type="religion" 

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="test" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/fairness_x_explainability/explanation_debiased_models_civil/bert_civil_religion/attention_rollout/10.0" \
    --batch_size=64 \
    --max_seq_length=512 \
    --num_examples=1000 \
    --output_dir="/scratch/yifwang/fairness_x_explainability/bias_mitigation_results_civil/bert_civil_religion_religion_test_1000/attention_rollout/10.0" \
    --bias_type="religion" 

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="test" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/fairness_x_explainability/explanation_debiased_models_civil/bert_civil_religion/attention_rollout/1.0" \
    --batch_size=64 \
    --max_seq_length=512 \
    --num_examples=1000 \
    --output_dir="/scratch/yifwang/fairness_x_explainability/bias_mitigation_results_civil/bert_civil_religion_religion_test_1000/attention_rollout/1.0" \
    --bias_type="religion" 

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="test" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/fairness_x_explainability/explanation_debiased_models_civil/bert_civil_religion/attention_rollout/0.1" \
    --batch_size=64 \
    --max_seq_length=512 \
    --num_examples=1000 \
    --output_dir="/scratch/yifwang/fairness_x_explainability/bias_mitigation_results_civil/bert_civil_religion_religion_test_1000/attention_rollout/0.1" \
    --bias_type="religion" 

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="test" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/fairness_x_explainability/explanation_debiased_models_civil/bert_civil_religion/attention_rollout/0.01" \
    --batch_size=64 \
    --max_seq_length=512 \
    --num_examples=1000 \
    --output_dir="/scratch/yifwang/fairness_x_explainability/bias_mitigation_results_civil/bert_civil_religion_religion_test_1000/attention_rollout/0.01" \
    --bias_type="religion" 

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="test" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/fairness_x_explainability/explanation_debiased_models_civil/bert_civil_religion/Occlusion/100.0" \
    --batch_size=64 \
    --max_seq_length=512 \
    --num_examples=1000 \
    --output_dir="/scratch/yifwang/fairness_x_explainability/bias_mitigation_results_civil/bert_civil_religion_religion_test_1000/Occlusion/100.0" \
    --bias_type="religion" 

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="test" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/fairness_x_explainability/explanation_debiased_models_civil/bert_civil_religion/Occlusion/10.0" \
    --batch_size=64 \
    --max_seq_length=512 \
    --num_examples=1000 \
    --output_dir="/scratch/yifwang/fairness_x_explainability/bias_mitigation_results_civil/bert_civil_religion_religion_test_1000/Occlusion/10.0" \
    --bias_type="religion" 

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="test" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/fairness_x_explainability/explanation_debiased_models_civil/bert_civil_religion/Occlusion/1.0" \
    --batch_size=64 \
    --max_seq_length=512 \
    --num_examples=1000 \
    --output_dir="/scratch/yifwang/fairness_x_explainability/bias_mitigation_results_civil/bert_civil_religion_religion_test_1000/Occlusion/1.0" \
    --bias_type="religion" 

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="test" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/fairness_x_explainability/explanation_debiased_models_civil/bert_civil_religion/Occlusion/0.1" \
    --batch_size=64 \
    --max_seq_length=512 \
    --num_examples=1000 \
    --output_dir="/scratch/yifwang/fairness_x_explainability/bias_mitigation_results_civil/bert_civil_religion_religion_test_1000/Occlusion/0.1" \
    --bias_type="religion" 

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="test" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/fairness_x_explainability/explanation_debiased_models_civil/bert_civil_religion/Occlusion/0.01" \
    --batch_size=64 \
    --max_seq_length=512 \
    --num_examples=1000 \
    --output_dir="/scratch/yifwang/fairness_x_explainability/bias_mitigation_results_civil/bert_civil_religion_religion_test_1000/Occlusion/0.01" \
    --bias_type="religion" 
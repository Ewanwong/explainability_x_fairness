#!/bin/bash

python -m explanation_generation.gen_explanation_encoder \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="test" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/explanation_debiased_models_civil/bert_civil_race/Saliency/L1_1.0" \
    --num_labels=2 \
    --batch_size=64 \
    --max_length=512 \
    --num_examples=2000 \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/bias_mitigation_results_civil/bert_civil_race_race_test_2000/Saliency/L1_1.0" \
    --baseline="pad" \
    --bias_type="race" \
    --seed=42 \
    --only_predicted_class

python -m explanation_generation.gen_explanation_encoder \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="test" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/explanation_debiased_models_civil/bert_civil_race/Saliency/L1_0.1" \
    --num_labels=2 \
    --batch_size=64 \
    --max_length=512 \
    --num_examples=2000 \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/bias_mitigation_results_civil/bert_civil_race_race_test_2000/Saliency/L1_0.1" \
    --baseline="pad" \
    --bias_type="race" \
    --seed=42 \
    --only_predicted_class

python -m explanation_generation.gen_explanation_encoder \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="test" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/explanation_debiased_models_civil/bert_civil_race/Saliency/L1_0.01" \
    --num_labels=2 \
    --batch_size=64 \
    --max_length=512 \
    --num_examples=2000 \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/bias_mitigation_results_civil/bert_civil_race_race_test_2000/Saliency/L1_0.01" \
    --baseline="pad" \
    --bias_type="race" \
    --seed=42 \
    --only_predicted_class

python -m explanation_generation.gen_explanation_encoder \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="test" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/explanation_debiased_models_civil/bert_civil_race/Saliency/L1_0.001" \
    --num_labels=2 \
    --batch_size=64 \
    --max_length=512 \
    --num_examples=2000 \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/bias_mitigation_results_civil/bert_civil_race_race_test_2000/Saliency/L1_0.001" \
    --baseline="pad" \
    --bias_type="race" \
    --seed=42 \
    --only_predicted_class

python -m explanation_generation.gen_explanation_encoder \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="test" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/explanation_debiased_models_civil/bert_civil_race/Saliency/L1_0.0001" \
    --num_labels=2 \
    --batch_size=64 \
    --max_length=512 \
    --num_examples=2000 \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/bias_mitigation_results_civil/bert_civil_race_race_test_2000/Saliency/L1_0.0001" \
    --baseline="pad" \
    --bias_type="race" \
    --seed=42 \
    --only_predicted_class

python -m explanation_generation.gen_explanation_encoder \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="test" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/explanation_debiased_models_civil/bert_civil_race/Saliency/L2_1.0" \
    --num_labels=2 \
    --batch_size=64 \
    --max_length=512 \
    --num_examples=2000 \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/bias_mitigation_results_civil/bert_civil_race_race_test_2000/Saliency/L2_1.0" \
    --baseline="pad" \
    --bias_type="race" \
    --seed=42 \
    --only_predicted_class

python -m explanation_generation.gen_explanation_encoder \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="test" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/explanation_debiased_models_civil/bert_civil_race/Saliency/L2_0.1" \
    --num_labels=2 \
    --batch_size=64 \
    --max_length=512 \
    --num_examples=2000 \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/bias_mitigation_results_civil/bert_civil_race_race_test_2000/Saliency/L2_0.1" \
    --baseline="pad" \
    --bias_type="race" \
    --seed=42 \
    --only_predicted_class

python -m explanation_generation.gen_explanation_encoder \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="test" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/explanation_debiased_models_civil/bert_civil_race/Saliency/L2_0.01" \
    --num_labels=2 \
    --batch_size=64 \
    --max_length=512 \
    --num_examples=2000 \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/bias_mitigation_results_civil/bert_civil_race_race_test_2000/Saliency/L2_0.01" \
    --baseline="pad" \
    --bias_type="race" \
    --seed=42 \
    --only_predicted_class

python -m explanation_generation.gen_explanation_encoder \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="test" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/explanation_debiased_models_civil/bert_civil_race/Saliency/L2_0.001" \
    --num_labels=2 \
    --batch_size=64 \
    --max_length=512 \
    --num_examples=2000 \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/bias_mitigation_results_civil/bert_civil_race_race_test_2000/Saliency/L2_0.001" \
    --baseline="pad" \
    --bias_type="race" \
    --seed=42 \
    --only_predicted_class

python -m explanation_generation.gen_explanation_encoder \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="test" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/explanation_debiased_models_civil/bert_civil_race/Saliency/L2_0.0001" \
    --num_labels=2 \
    --batch_size=64 \
    --max_length=512 \
    --num_examples=2000 \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/bias_mitigation_results_civil/bert_civil_race_race_test_2000/Saliency/L2_0.0001" \
    --baseline="pad" \
    --bias_type="race" \
    --seed=42 \
    --only_predicted_class

python -m explanation_generation.gen_explanation_encoder \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="test" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/explanation_debiased_models_civil/bert_civil_race/InputXGradient/L1_1.0" \
    --num_labels=2 \
    --batch_size=64 \
    --max_length=512 \
    --num_examples=2000 \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/bias_mitigation_results_civil/bert_civil_race_race_test_2000/InputXGradient/L1_1.0" \
    --baseline="pad" \
    --bias_type="race" \
    --seed=42 \
    --only_predicted_class

python -m explanation_generation.gen_explanation_encoder \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="test" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/explanation_debiased_models_civil/bert_civil_race/InputXGradient/L1_0.1" \
    --num_labels=2 \
    --batch_size=64 \
    --max_length=512 \
    --num_examples=2000 \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/bias_mitigation_results_civil/bert_civil_race_race_test_2000/InputXGradient/L1_0.1" \
    --baseline="pad" \
    --bias_type="race" \
    --seed=42 \
    --only_predicted_class

python -m explanation_generation.gen_explanation_encoder \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="test" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/explanation_debiased_models_civil/bert_civil_race/InputXGradient/L1_0.01" \
    --num_labels=2 \
    --batch_size=64 \
    --max_length=512 \
    --num_examples=2000 \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/bias_mitigation_results_civil/bert_civil_race_race_test_2000/InputXGradient/L1_0.01" \
    --baseline="pad" \
    --bias_type="race" \
    --seed=42 \
    --only_predicted_class

python -m explanation_generation.gen_explanation_encoder \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="test" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/explanation_debiased_models_civil/bert_civil_race/InputXGradient/L1_0.001" \
    --num_labels=2 \
    --batch_size=64 \
    --max_length=512 \
    --num_examples=2000 \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/bias_mitigation_results_civil/bert_civil_race_race_test_2000/InputXGradient/L1_0.001" \
    --baseline="pad" \
    --bias_type="race" \
    --seed=42 \
    --only_predicted_class

python -m explanation_generation.gen_explanation_encoder \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="test" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/explanation_debiased_models_civil/bert_civil_race/InputXGradient/L1_0.0001" \
    --num_labels=2 \
    --batch_size=64 \
    --max_length=512 \
    --num_examples=2000 \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/bias_mitigation_results_civil/bert_civil_race_race_test_2000/InputXGradient/L1_0.0001" \
    --baseline="pad" \
    --bias_type="race" \
    --seed=42 \
    --only_predicted_class

python -m explanation_generation.gen_explanation_encoder \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="test" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/explanation_debiased_models_civil/bert_civil_race/InputXGradient/L2_1.0" \
    --num_labels=2 \
    --batch_size=64 \
    --max_length=512 \
    --num_examples=2000 \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/bias_mitigation_results_civil/bert_civil_race_race_test_2000/InputXGradient/L2_1.0" \
    --baseline="pad" \
    --bias_type="race" \
    --seed=42 \
    --only_predicted_class

python -m explanation_generation.gen_explanation_encoder \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="test" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/explanation_debiased_models_civil/bert_civil_race/InputXGradient/L2_0.1" \
    --num_labels=2 \
    --batch_size=64 \
    --max_length=512 \
    --num_examples=2000 \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/bias_mitigation_results_civil/bert_civil_race_race_test_2000/InputXGradient/L2_0.1" \
    --baseline="pad" \
    --bias_type="race" \
    --seed=42 \
    --only_predicted_class

python -m explanation_generation.gen_explanation_encoder \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="test" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/explanation_debiased_models_civil/bert_civil_race/InputXGradient/L2_0.01" \
    --num_labels=2 \
    --batch_size=64 \
    --max_length=512 \
    --num_examples=2000 \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/bias_mitigation_results_civil/bert_civil_race_race_test_2000/InputXGradient/L2_0.01" \
    --baseline="pad" \
    --bias_type="race" \
    --seed=42 \
    --only_predicted_class

python -m explanation_generation.gen_explanation_encoder \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="test" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/explanation_debiased_models_civil/bert_civil_race/InputXGradient/L2_0.001" \
    --num_labels=2 \
    --batch_size=64 \
    --max_length=512 \
    --num_examples=2000 \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/bias_mitigation_results_civil/bert_civil_race_race_test_2000/InputXGradient/L2_0.001" \
    --baseline="pad" \
    --bias_type="race" \
    --seed=42 \
    --only_predicted_class

python -m explanation_generation.gen_explanation_encoder \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="test" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/explanation_debiased_models_civil/bert_civil_race/InputXGradient/L2_0.0001" \
    --num_labels=2 \
    --batch_size=64 \
    --max_length=512 \
    --num_examples=2000 \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/bias_mitigation_results_civil/bert_civil_race_race_test_2000/InputXGradient/L2_0.0001" \
    --baseline="pad" \
    --bias_type="race" \
    --seed=42 \
    --only_predicted_class

python -m explanation_generation.gen_explanation_encoder \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="test" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/explanation_debiased_models_civil/bert_civil_race/IntegratedGradients/L1_1.0" \
    --num_labels=2 \
    --batch_size=64 \
    --max_length=512 \
    --num_examples=2000 \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/bias_mitigation_results_civil/bert_civil_race_race_test_2000/IntegratedGradients/L1_1.0" \
    --baseline="pad" \
    --bias_type="race" \
    --seed=42 \
    --only_predicted_class

python -m explanation_generation.gen_explanation_encoder \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="test" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/explanation_debiased_models_civil/bert_civil_race/IntegratedGradients/L1_0.1" \
    --num_labels=2 \
    --batch_size=64 \
    --max_length=512 \
    --num_examples=2000 \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/bias_mitigation_results_civil/bert_civil_race_race_test_2000/IntegratedGradients/L1_0.1" \
    --baseline="pad" \
    --bias_type="race" \
    --seed=42 \
    --only_predicted_class

python -m explanation_generation.gen_explanation_encoder \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="test" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/explanation_debiased_models_civil/bert_civil_race/IntegratedGradients/L1_0.01" \
    --num_labels=2 \
    --batch_size=64 \
    --max_length=512 \
    --num_examples=2000 \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/bias_mitigation_results_civil/bert_civil_race_race_test_2000/IntegratedGradients/L1_0.01" \
    --baseline="pad" \
    --bias_type="race" \
    --seed=42 \
    --only_predicted_class

python -m explanation_generation.gen_explanation_encoder \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="test" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/explanation_debiased_models_civil/bert_civil_race/IntegratedGradients/L1_0.001" \
    --num_labels=2 \
    --batch_size=64 \
    --max_length=512 \
    --num_examples=2000 \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/bias_mitigation_results_civil/bert_civil_race_race_test_2000/IntegratedGradients/L1_0.001" \
    --baseline="pad" \
    --bias_type="race" \
    --seed=42 \
    --only_predicted_class

python -m explanation_generation.gen_explanation_encoder \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="test" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/explanation_debiased_models_civil/bert_civil_race/IntegratedGradients/L1_0.0001" \
    --num_labels=2 \
    --batch_size=64 \
    --max_length=512 \
    --num_examples=2000 \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/bias_mitigation_results_civil/bert_civil_race_race_test_2000/IntegratedGradients/L1_0.0001" \
    --baseline="pad" \
    --bias_type="race" \
    --seed=42 \
    --only_predicted_class

python -m explanation_generation.gen_explanation_encoder \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="test" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/explanation_debiased_models_civil/bert_civil_race/IntegratedGradients/L2_1.0" \
    --num_labels=2 \
    --batch_size=64 \
    --max_length=512 \
    --num_examples=2000 \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/bias_mitigation_results_civil/bert_civil_race_race_test_2000/IntegratedGradients/L2_1.0" \
    --baseline="pad" \
    --bias_type="race" \
    --seed=42 \
    --only_predicted_class

python -m explanation_generation.gen_explanation_encoder \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="test" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/explanation_debiased_models_civil/bert_civil_race/IntegratedGradients/L2_0.1" \
    --num_labels=2 \
    --batch_size=64 \
    --max_length=512 \
    --num_examples=2000 \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/bias_mitigation_results_civil/bert_civil_race_race_test_2000/IntegratedGradients/L2_0.1" \
    --baseline="pad" \
    --bias_type="race" \
    --seed=42 \
    --only_predicted_class

python -m explanation_generation.gen_explanation_encoder \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="test" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/explanation_debiased_models_civil/bert_civil_race/IntegratedGradients/L2_0.01" \
    --num_labels=2 \
    --batch_size=64 \
    --max_length=512 \
    --num_examples=2000 \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/bias_mitigation_results_civil/bert_civil_race_race_test_2000/IntegratedGradients/L2_0.01" \
    --baseline="pad" \
    --bias_type="race" \
    --seed=42 \
    --only_predicted_class

python -m explanation_generation.gen_explanation_encoder \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="test" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/explanation_debiased_models_civil/bert_civil_race/IntegratedGradients/L2_0.001" \
    --num_labels=2 \
    --batch_size=64 \
    --max_length=512 \
    --num_examples=2000 \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/bias_mitigation_results_civil/bert_civil_race_race_test_2000/IntegratedGradients/L2_0.001" \
    --baseline="pad" \
    --bias_type="race" \
    --seed=42 \
    --only_predicted_class

python -m explanation_generation.gen_explanation_encoder \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="test" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/explanation_debiased_models_civil/bert_civil_race/IntegratedGradients/L2_0.0001" \
    --num_labels=2 \
    --batch_size=64 \
    --max_length=512 \
    --num_examples=2000 \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/bias_mitigation_results_civil/bert_civil_race_race_test_2000/IntegratedGradients/L2_0.0001" \
    --baseline="pad" \
    --bias_type="race" \
    --seed=42 \
    --only_predicted_class

python -m explanation_generation.gen_explanation_encoder \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="test" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/explanation_debiased_models_civil/bert_civil_race/raw_attention/L1_1.0" \
    --num_labels=2 \
    --batch_size=64 \
    --max_length=512 \
    --num_examples=2000 \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/bias_mitigation_results_civil/bert_civil_race_race_test_2000/raw_attention/L1_1.0" \
    --baseline="pad" \
    --bias_type="race" \
    --seed=42 \
    --only_predicted_class

python -m explanation_generation.gen_explanation_encoder \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="test" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/explanation_debiased_models_civil/bert_civil_race/raw_attention/L1_0.1" \
    --num_labels=2 \
    --batch_size=64 \
    --max_length=512 \
    --num_examples=2000 \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/bias_mitigation_results_civil/bert_civil_race_race_test_2000/raw_attention/L1_0.1" \
    --baseline="pad" \
    --bias_type="race" \
    --seed=42 \
    --only_predicted_class

python -m explanation_generation.gen_explanation_encoder \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="test" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/explanation_debiased_models_civil/bert_civil_race/raw_attention/L1_0.01" \
    --num_labels=2 \
    --batch_size=64 \
    --max_length=512 \
    --num_examples=2000 \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/bias_mitigation_results_civil/bert_civil_race_race_test_2000/raw_attention/L1_0.01" \
    --baseline="pad" \
    --bias_type="race" \
    --seed=42 \
    --only_predicted_class

python -m explanation_generation.gen_explanation_encoder \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="test" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/explanation_debiased_models_civil/bert_civil_race/raw_attention/L1_0.001" \
    --num_labels=2 \
    --batch_size=64 \
    --max_length=512 \
    --num_examples=2000 \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/bias_mitigation_results_civil/bert_civil_race_race_test_2000/raw_attention/L1_0.001" \
    --baseline="pad" \
    --bias_type="race" \
    --seed=42 \
    --only_predicted_class

python -m explanation_generation.gen_explanation_encoder \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="test" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/explanation_debiased_models_civil/bert_civil_race/raw_attention/L1_0.0001" \
    --num_labels=2 \
    --batch_size=64 \
    --max_length=512 \
    --num_examples=2000 \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/bias_mitigation_results_civil/bert_civil_race_race_test_2000/raw_attention/L1_0.0001" \
    --baseline="pad" \
    --bias_type="race" \
    --seed=42 \
    --only_predicted_class

python -m explanation_generation.gen_explanation_encoder \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="test" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/explanation_debiased_models_civil/bert_civil_race/raw_attention/L2_1.0" \
    --num_labels=2 \
    --batch_size=64 \
    --max_length=512 \
    --num_examples=2000 \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/bias_mitigation_results_civil/bert_civil_race_race_test_2000/raw_attention/L2_1.0" \
    --baseline="pad" \
    --bias_type="race" \
    --seed=42 \
    --only_predicted_class

python -m explanation_generation.gen_explanation_encoder \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="test" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/explanation_debiased_models_civil/bert_civil_race/raw_attention/L2_0.1" \
    --num_labels=2 \
    --batch_size=64 \
    --max_length=512 \
    --num_examples=2000 \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/bias_mitigation_results_civil/bert_civil_race_race_test_2000/raw_attention/L2_0.1" \
    --baseline="pad" \
    --bias_type="race" \
    --seed=42 \
    --only_predicted_class

python -m explanation_generation.gen_explanation_encoder \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="test" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/explanation_debiased_models_civil/bert_civil_race/raw_attention/L2_0.01" \
    --num_labels=2 \
    --batch_size=64 \
    --max_length=512 \
    --num_examples=2000 \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/bias_mitigation_results_civil/bert_civil_race_race_test_2000/raw_attention/L2_0.01" \
    --baseline="pad" \
    --bias_type="race" \
    --seed=42 \
    --only_predicted_class

python -m explanation_generation.gen_explanation_encoder \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="test" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/explanation_debiased_models_civil/bert_civil_race/raw_attention/L2_0.001" \
    --num_labels=2 \
    --batch_size=64 \
    --max_length=512 \
    --num_examples=2000 \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/bias_mitigation_results_civil/bert_civil_race_race_test_2000/raw_attention/L2_0.001" \
    --baseline="pad" \
    --bias_type="race" \
    --seed=42 \
    --only_predicted_class

python -m explanation_generation.gen_explanation_encoder \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="test" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/explanation_debiased_models_civil/bert_civil_race/raw_attention/L2_0.0001" \
    --num_labels=2 \
    --batch_size=64 \
    --max_length=512 \
    --num_examples=2000 \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/bias_mitigation_results_civil/bert_civil_race_race_test_2000/raw_attention/L2_0.0001" \
    --baseline="pad" \
    --bias_type="race" \
    --seed=42 \
    --only_predicted_class

python -m explanation_generation.gen_explanation_encoder \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="test" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/explanation_debiased_models_civil/bert_civil_race/attention_flow/L1_1.0" \
    --num_labels=2 \
    --batch_size=64 \
    --max_length=512 \
    --num_examples=2000 \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/bias_mitigation_results_civil/bert_civil_race_race_test_2000/attention_flow/L1_1.0" \
    --baseline="pad" \
    --bias_type="race" \
    --seed=42 \
    --only_predicted_class

python -m explanation_generation.gen_explanation_encoder \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="test" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/explanation_debiased_models_civil/bert_civil_race/attention_flow/L1_0.1" \
    --num_labels=2 \
    --batch_size=64 \
    --max_length=512 \
    --num_examples=2000 \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/bias_mitigation_results_civil/bert_civil_race_race_test_2000/attention_flow/L1_0.1" \
    --baseline="pad" \
    --bias_type="race" \
    --seed=42 \
    --only_predicted_class

python -m explanation_generation.gen_explanation_encoder \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="test" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/explanation_debiased_models_civil/bert_civil_race/attention_flow/L1_0.01" \
    --num_labels=2 \
    --batch_size=64 \
    --max_length=512 \
    --num_examples=2000 \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/bias_mitigation_results_civil/bert_civil_race_race_test_2000/attention_flow/L1_0.01" \
    --baseline="pad" \
    --bias_type="race" \
    --seed=42 \
    --only_predicted_class

python -m explanation_generation.gen_explanation_encoder \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="test" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/explanation_debiased_models_civil/bert_civil_race/attention_flow/L1_0.001" \
    --num_labels=2 \
    --batch_size=64 \
    --max_length=512 \
    --num_examples=2000 \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/bias_mitigation_results_civil/bert_civil_race_race_test_2000/attention_flow/L1_0.001" \
    --baseline="pad" \
    --bias_type="race" \
    --seed=42 \
    --only_predicted_class

python -m explanation_generation.gen_explanation_encoder \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="test" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/explanation_debiased_models_civil/bert_civil_race/attention_flow/L1_0.0001" \
    --num_labels=2 \
    --batch_size=64 \
    --max_length=512 \
    --num_examples=2000 \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/bias_mitigation_results_civil/bert_civil_race_race_test_2000/attention_flow/L1_0.0001" \
    --baseline="pad" \
    --bias_type="race" \
    --seed=42 \
    --only_predicted_class

python -m explanation_generation.gen_explanation_encoder \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="test" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/explanation_debiased_models_civil/bert_civil_race/attention_flow/L2_1.0" \
    --num_labels=2 \
    --batch_size=64 \
    --max_length=512 \
    --num_examples=2000 \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/bias_mitigation_results_civil/bert_civil_race_race_test_2000/attention_flow/L2_1.0" \
    --baseline="pad" \
    --bias_type="race" \
    --seed=42 \
    --only_predicted_class

python -m explanation_generation.gen_explanation_encoder \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="test" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/explanation_debiased_models_civil/bert_civil_race/attention_flow/L2_0.1" \
    --num_labels=2 \
    --batch_size=64 \
    --max_length=512 \
    --num_examples=2000 \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/bias_mitigation_results_civil/bert_civil_race_race_test_2000/attention_flow/L2_0.1" \
    --baseline="pad" \
    --bias_type="race" \
    --seed=42 \
    --only_predicted_class

python -m explanation_generation.gen_explanation_encoder \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="test" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/explanation_debiased_models_civil/bert_civil_race/attention_flow/L2_0.01" \
    --num_labels=2 \
    --batch_size=64 \
    --max_length=512 \
    --num_examples=2000 \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/bias_mitigation_results_civil/bert_civil_race_race_test_2000/attention_flow/L2_0.01" \
    --baseline="pad" \
    --bias_type="race" \
    --seed=42 \
    --only_predicted_class

python -m explanation_generation.gen_explanation_encoder \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="test" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/explanation_debiased_models_civil/bert_civil_race/attention_flow/L2_0.001" \
    --num_labels=2 \
    --batch_size=64 \
    --max_length=512 \
    --num_examples=2000 \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/bias_mitigation_results_civil/bert_civil_race_race_test_2000/attention_flow/L2_0.001" \
    --baseline="pad" \
    --bias_type="race" \
    --seed=42 \
    --only_predicted_class

python -m explanation_generation.gen_explanation_encoder \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="test" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/explanation_debiased_models_civil/bert_civil_race/attention_flow/L2_0.0001" \
    --num_labels=2 \
    --batch_size=64 \
    --max_length=512 \
    --num_examples=2000 \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/bias_mitigation_results_civil/bert_civil_race_race_test_2000/attention_flow/L2_0.0001" \
    --baseline="pad" \
    --bias_type="race" \
    --seed=42 \
    --only_predicted_class

python -m explanation_generation.gen_explanation_encoder \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="test" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/explanation_debiased_models_civil/bert_civil_race/attention_rollout/L1_1.0" \
    --num_labels=2 \
    --batch_size=64 \
    --max_length=512 \
    --num_examples=2000 \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/bias_mitigation_results_civil/bert_civil_race_race_test_2000/attention_rollout/L1_1.0" \
    --baseline="pad" \
    --bias_type="race" \
    --seed=42 \
    --only_predicted_class

python -m explanation_generation.gen_explanation_encoder \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="test" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/explanation_debiased_models_civil/bert_civil_race/attention_rollout/L1_0.1" \
    --num_labels=2 \
    --batch_size=64 \
    --max_length=512 \
    --num_examples=2000 \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/bias_mitigation_results_civil/bert_civil_race_race_test_2000/attention_rollout/L1_0.1" \
    --baseline="pad" \
    --bias_type="race" \
    --seed=42 \
    --only_predicted_class

python -m explanation_generation.gen_explanation_encoder \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="test" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/explanation_debiased_models_civil/bert_civil_race/attention_rollout/L1_0.01" \
    --num_labels=2 \
    --batch_size=64 \
    --max_length=512 \
    --num_examples=2000 \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/bias_mitigation_results_civil/bert_civil_race_race_test_2000/attention_rollout/L1_0.01" \
    --baseline="pad" \
    --bias_type="race" \
    --seed=42 \
    --only_predicted_class

python -m explanation_generation.gen_explanation_encoder \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="test" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/explanation_debiased_models_civil/bert_civil_race/attention_rollout/L1_0.001" \
    --num_labels=2 \
    --batch_size=64 \
    --max_length=512 \
    --num_examples=2000 \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/bias_mitigation_results_civil/bert_civil_race_race_test_2000/attention_rollout/L1_0.001" \
    --baseline="pad" \
    --bias_type="race" \
    --seed=42 \
    --only_predicted_class

python -m explanation_generation.gen_explanation_encoder \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="test" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/explanation_debiased_models_civil/bert_civil_race/attention_rollout/L1_0.0001" \
    --num_labels=2 \
    --batch_size=64 \
    --max_length=512 \
    --num_examples=2000 \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/bias_mitigation_results_civil/bert_civil_race_race_test_2000/attention_rollout/L1_0.0001" \
    --baseline="pad" \
    --bias_type="race" \
    --seed=42 \
    --only_predicted_class

python -m explanation_generation.gen_explanation_encoder \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="test" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/explanation_debiased_models_civil/bert_civil_race/attention_rollout/L2_1.0" \
    --num_labels=2 \
    --batch_size=64 \
    --max_length=512 \
    --num_examples=2000 \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/bias_mitigation_results_civil/bert_civil_race_race_test_2000/attention_rollout/L2_1.0" \
    --baseline="pad" \
    --bias_type="race" \
    --seed=42 \
    --only_predicted_class

python -m explanation_generation.gen_explanation_encoder \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="test" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/explanation_debiased_models_civil/bert_civil_race/attention_rollout/L2_0.1" \
    --num_labels=2 \
    --batch_size=64 \
    --max_length=512 \
    --num_examples=2000 \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/bias_mitigation_results_civil/bert_civil_race_race_test_2000/attention_rollout/L2_0.1" \
    --baseline="pad" \
    --bias_type="race" \
    --seed=42 \
    --only_predicted_class

python -m explanation_generation.gen_explanation_encoder \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="test" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/explanation_debiased_models_civil/bert_civil_race/attention_rollout/L2_0.01" \
    --num_labels=2 \
    --batch_size=64 \
    --max_length=512 \
    --num_examples=2000 \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/bias_mitigation_results_civil/bert_civil_race_race_test_2000/attention_rollout/L2_0.01" \
    --baseline="pad" \
    --bias_type="race" \
    --seed=42 \
    --only_predicted_class

python -m explanation_generation.gen_explanation_encoder \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="test" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/explanation_debiased_models_civil/bert_civil_race/attention_rollout/L2_0.001" \
    --num_labels=2 \
    --batch_size=64 \
    --max_length=512 \
    --num_examples=2000 \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/bias_mitigation_results_civil/bert_civil_race_race_test_2000/attention_rollout/L2_0.001" \
    --baseline="pad" \
    --bias_type="race" \
    --seed=42 \
    --only_predicted_class

python -m explanation_generation.gen_explanation_encoder \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="test" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/explanation_debiased_models_civil/bert_civil_race/attention_rollout/L2_0.0001" \
    --num_labels=2 \
    --batch_size=64 \
    --max_length=512 \
    --num_examples=2000 \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/bias_mitigation_results_civil/bert_civil_race_race_test_2000/attention_rollout/L2_0.0001" \
    --baseline="pad" \
    --bias_type="race" \
    --seed=42 \
    --only_predicted_class

python -m explanation_generation.gen_explanation_encoder \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="test" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/explanation_debiased_models_civil/bert_civil_race/Occlusion/L1_1.0" \
    --num_labels=2 \
    --batch_size=64 \
    --max_length=512 \
    --num_examples=2000 \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/bias_mitigation_results_civil/bert_civil_race_race_test_2000/Occlusion/L1_1.0" \
    --baseline="pad" \
    --bias_type="race" \
    --seed=42 \
    --only_predicted_class

python -m explanation_generation.gen_explanation_encoder \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="test" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/explanation_debiased_models_civil/bert_civil_race/Occlusion/L1_0.1" \
    --num_labels=2 \
    --batch_size=64 \
    --max_length=512 \
    --num_examples=2000 \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/bias_mitigation_results_civil/bert_civil_race_race_test_2000/Occlusion/L1_0.1" \
    --baseline="pad" \
    --bias_type="race" \
    --seed=42 \
    --only_predicted_class

python -m explanation_generation.gen_explanation_encoder \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="test" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/explanation_debiased_models_civil/bert_civil_race/Occlusion/L1_0.01" \
    --num_labels=2 \
    --batch_size=64 \
    --max_length=512 \
    --num_examples=2000 \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/bias_mitigation_results_civil/bert_civil_race_race_test_2000/Occlusion/L1_0.01" \
    --baseline="pad" \
    --bias_type="race" \
    --seed=42 \
    --only_predicted_class

python -m explanation_generation.gen_explanation_encoder \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="test" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/explanation_debiased_models_civil/bert_civil_race/Occlusion/L1_0.001" \
    --num_labels=2 \
    --batch_size=64 \
    --max_length=512 \
    --num_examples=2000 \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/bias_mitigation_results_civil/bert_civil_race_race_test_2000/Occlusion/L1_0.001" \
    --baseline="pad" \
    --bias_type="race" \
    --seed=42 \
    --only_predicted_class

python -m explanation_generation.gen_explanation_encoder \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="test" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/explanation_debiased_models_civil/bert_civil_race/Occlusion/L1_0.0001" \
    --num_labels=2 \
    --batch_size=64 \
    --max_length=512 \
    --num_examples=2000 \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/bias_mitigation_results_civil/bert_civil_race_race_test_2000/Occlusion/L1_0.0001" \
    --baseline="pad" \
    --bias_type="race" \
    --seed=42 \
    --only_predicted_class

python -m explanation_generation.gen_explanation_encoder \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="test" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/explanation_debiased_models_civil/bert_civil_race/Occlusion/L2_1.0" \
    --num_labels=2 \
    --batch_size=64 \
    --max_length=512 \
    --num_examples=2000 \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/bias_mitigation_results_civil/bert_civil_race_race_test_2000/Occlusion/L2_1.0" \
    --baseline="pad" \
    --bias_type="race" \
    --seed=42 \
    --only_predicted_class

python -m explanation_generation.gen_explanation_encoder \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="test" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/explanation_debiased_models_civil/bert_civil_race/Occlusion/L2_0.1" \
    --num_labels=2 \
    --batch_size=64 \
    --max_length=512 \
    --num_examples=2000 \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/bias_mitigation_results_civil/bert_civil_race_race_test_2000/Occlusion/L2_0.1" \
    --baseline="pad" \
    --bias_type="race" \
    --seed=42 \
    --only_predicted_class

python -m explanation_generation.gen_explanation_encoder \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="test" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/explanation_debiased_models_civil/bert_civil_race/Occlusion/L2_0.01" \
    --num_labels=2 \
    --batch_size=64 \
    --max_length=512 \
    --num_examples=2000 \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/bias_mitigation_results_civil/bert_civil_race_race_test_2000/Occlusion/L2_0.01" \
    --baseline="pad" \
    --bias_type="race" \
    --seed=42 \
    --only_predicted_class

python -m explanation_generation.gen_explanation_encoder \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="test" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/explanation_debiased_models_civil/bert_civil_race/Occlusion/L2_0.001" \
    --num_labels=2 \
    --batch_size=64 \
    --max_length=512 \
    --num_examples=2000 \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/bias_mitigation_results_civil/bert_civil_race_race_test_2000/Occlusion/L2_0.001" \
    --baseline="pad" \
    --bias_type="race" \
    --seed=42 \
    --only_predicted_class

python -m explanation_generation.gen_explanation_encoder \
    --dataset_name="lighteval/civil_comments_helm" \
    --split="test" \
    --split_ratio="8, 2" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/explanation_debiased_models_civil/bert_civil_race/Occlusion/L2_0.0001" \
    --num_labels=2 \
    --batch_size=64 \
    --max_length=512 \
    --num_examples=2000 \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/bias_mitigation_results_civil/bert_civil_race_race_test_2000/Occlusion/L2_0.0001" \
    --baseline="pad" \
    --bias_type="race" \
    --seed=42 \
    --only_predicted_class


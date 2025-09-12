#!/bin/bash

python -m bias_correlation.compute_bias_correlation \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results_jigsaw/bert_jigsaw_race_race_test_400/no_debiasing" \
    --split="test" \
    --bias_type="race" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m visualization.visualize_bias_correlation \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results_jigsaw/bert_jigsaw_race_race_test_400/no_debiasing" \
    --split="test" \
    --bias_type="race" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m bias_correlation.compute_bias_correlation \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results_jigsaw/bert_jigsaw_all_race_test_400/no_debiasing" \
    --split="test" \
    --bias_type="race" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m visualization.visualize_bias_correlation \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results_jigsaw/bert_jigsaw_all_race_test_400/no_debiasing" \
    --split="test" \
    --bias_type="race" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m bias_correlation.compute_bias_correlation \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results_jigsaw/bert_jigsaw_race_race_test_400/group_balance" \
    --split="test" \
    --bias_type="race" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m visualization.visualize_bias_correlation \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results_jigsaw/bert_jigsaw_race_race_test_400/group_balance" \
    --split="test" \
    --bias_type="race" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m bias_correlation.compute_bias_correlation \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results_jigsaw/bert_jigsaw_all_race_test_400/group_balance" \
    --split="test" \
    --bias_type="race" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m visualization.visualize_bias_correlation \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results_jigsaw/bert_jigsaw_all_race_test_400/group_balance" \
    --split="test" \
    --bias_type="race" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m bias_correlation.compute_bias_correlation \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results_jigsaw/bert_jigsaw_race_race_test_400/group_class_balance" \
    --split="test" \
    --bias_type="race" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m visualization.visualize_bias_correlation \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results_jigsaw/bert_jigsaw_race_race_test_400/group_class_balance" \
    --split="test" \
    --bias_type="race" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m bias_correlation.compute_bias_correlation \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results_jigsaw/bert_jigsaw_all_race_test_400/group_class_balance" \
    --split="test" \
    --bias_type="race" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m visualization.visualize_bias_correlation \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results_jigsaw/bert_jigsaw_all_race_test_400/group_class_balance" \
    --split="test" \
    --bias_type="race" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m bias_correlation.compute_bias_correlation \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results_jigsaw/bert_jigsaw_race_race_test_400/cda" \
    --split="test" \
    --bias_type="race" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m visualization.visualize_bias_correlation \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results_jigsaw/bert_jigsaw_race_race_test_400/cda" \
    --split="test" \
    --bias_type="race" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m bias_correlation.compute_bias_correlation \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results_jigsaw/bert_jigsaw_all_race_test_400/cda" \
    --split="test" \
    --bias_type="race" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m visualization.visualize_bias_correlation \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results_jigsaw/bert_jigsaw_all_race_test_400/cda" \
    --split="test" \
    --bias_type="race" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m bias_correlation.compute_bias_correlation \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results_jigsaw/bert_jigsaw_race_race_test_400/dropout" \
    --split="test" \
    --bias_type="race" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m visualization.visualize_bias_correlation \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results_jigsaw/bert_jigsaw_race_race_test_400/dropout" \
    --split="test" \
    --bias_type="race" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m bias_correlation.compute_bias_correlation \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results_jigsaw/bert_jigsaw_all_race_test_400/dropout" \
    --split="test" \
    --bias_type="race" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m visualization.visualize_bias_correlation \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results_jigsaw/bert_jigsaw_all_race_test_400/dropout" \
    --split="test" \
    --bias_type="race" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m bias_correlation.compute_bias_correlation \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results_jigsaw/bert_jigsaw_race_race_test_400/attention_entropy" \
    --split="test" \
    --bias_type="race" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m visualization.visualize_bias_correlation \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results_jigsaw/bert_jigsaw_race_race_test_400/attention_entropy" \
    --split="test" \
    --bias_type="race" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m bias_correlation.compute_bias_correlation \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results_jigsaw/bert_jigsaw_all_race_test_400/attention_entropy" \
    --split="test" \
    --bias_type="race" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m visualization.visualize_bias_correlation \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results_jigsaw/bert_jigsaw_all_race_test_400/attention_entropy" \
    --split="test" \
    --bias_type="race" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m bias_correlation.compute_bias_correlation \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results_jigsaw/bert_jigsaw_race_race_test_400/causal_debias" \
    --split="test" \
    --bias_type="race" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m visualization.visualize_bias_correlation \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results_jigsaw/bert_jigsaw_race_race_test_400/causal_debias" \
    --split="test" \
    --bias_type="race" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m bias_correlation.compute_bias_correlation \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results_jigsaw/bert_jigsaw_all_race_test_400/causal_debias" \
    --split="test" \
    --bias_type="race" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m visualization.visualize_bias_correlation \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results_jigsaw/bert_jigsaw_all_race_test_400/causal_debias" \
    --split="test" \
    --bias_type="race" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m bias_correlation.compute_bias_correlation \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results_jigsaw/roberta_jigsaw_race_race_test_400/no_debiasing" \
    --split="test" \
    --bias_type="race" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m visualization.visualize_bias_correlation \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results_jigsaw/roberta_jigsaw_race_race_test_400/no_debiasing" \
    --split="test" \
    --bias_type="race" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m bias_correlation.compute_bias_correlation \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results_jigsaw/roberta_jigsaw_all_race_test_400/no_debiasing" \
    --split="test" \
    --bias_type="race" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m visualization.visualize_bias_correlation \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results_jigsaw/roberta_jigsaw_all_race_test_400/no_debiasing" \
    --split="test" \
    --bias_type="race" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m bias_correlation.compute_bias_correlation \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results_jigsaw/roberta_jigsaw_race_race_test_400/group_balance" \
    --split="test" \
    --bias_type="race" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m visualization.visualize_bias_correlation \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results_jigsaw/roberta_jigsaw_race_race_test_400/group_balance" \
    --split="test" \
    --bias_type="race" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m bias_correlation.compute_bias_correlation \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results_jigsaw/roberta_jigsaw_all_race_test_400/group_balance" \
    --split="test" \
    --bias_type="race" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m visualization.visualize_bias_correlation \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results_jigsaw/roberta_jigsaw_all_race_test_400/group_balance" \
    --split="test" \
    --bias_type="race" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m bias_correlation.compute_bias_correlation \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results_jigsaw/roberta_jigsaw_race_race_test_400/group_class_balance" \
    --split="test" \
    --bias_type="race" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m visualization.visualize_bias_correlation \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results_jigsaw/roberta_jigsaw_race_race_test_400/group_class_balance" \
    --split="test" \
    --bias_type="race" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m bias_correlation.compute_bias_correlation \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results_jigsaw/roberta_jigsaw_all_race_test_400/group_class_balance" \
    --split="test" \
    --bias_type="race" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m visualization.visualize_bias_correlation \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results_jigsaw/roberta_jigsaw_all_race_test_400/group_class_balance" \
    --split="test" \
    --bias_type="race" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m bias_correlation.compute_bias_correlation \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results_jigsaw/roberta_jigsaw_race_race_test_400/cda" \
    --split="test" \
    --bias_type="race" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m visualization.visualize_bias_correlation \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results_jigsaw/roberta_jigsaw_race_race_test_400/cda" \
    --split="test" \
    --bias_type="race" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m bias_correlation.compute_bias_correlation \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results_jigsaw/roberta_jigsaw_all_race_test_400/cda" \
    --split="test" \
    --bias_type="race" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m visualization.visualize_bias_correlation \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results_jigsaw/roberta_jigsaw_all_race_test_400/cda" \
    --split="test" \
    --bias_type="race" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m bias_correlation.compute_bias_correlation \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results_jigsaw/roberta_jigsaw_race_race_test_400/dropout" \
    --split="test" \
    --bias_type="race" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m visualization.visualize_bias_correlation \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results_jigsaw/roberta_jigsaw_race_race_test_400/dropout" \
    --split="test" \
    --bias_type="race" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m bias_correlation.compute_bias_correlation \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results_jigsaw/roberta_jigsaw_all_race_test_400/dropout" \
    --split="test" \
    --bias_type="race" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m visualization.visualize_bias_correlation \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results_jigsaw/roberta_jigsaw_all_race_test_400/dropout" \
    --split="test" \
    --bias_type="race" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m bias_correlation.compute_bias_correlation \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results_jigsaw/roberta_jigsaw_race_race_test_400/attention_entropy" \
    --split="test" \
    --bias_type="race" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m visualization.visualize_bias_correlation \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results_jigsaw/roberta_jigsaw_race_race_test_400/attention_entropy" \
    --split="test" \
    --bias_type="race" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m bias_correlation.compute_bias_correlation \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results_jigsaw/roberta_jigsaw_all_race_test_400/attention_entropy" \
    --split="test" \
    --bias_type="race" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m visualization.visualize_bias_correlation \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results_jigsaw/roberta_jigsaw_all_race_test_400/attention_entropy" \
    --split="test" \
    --bias_type="race" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m bias_correlation.compute_bias_correlation \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results_jigsaw/roberta_jigsaw_race_race_test_400/causal_debias" \
    --split="test" \
    --bias_type="race" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m visualization.visualize_bias_correlation \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results_jigsaw/roberta_jigsaw_race_race_test_400/causal_debias" \
    --split="test" \
    --bias_type="race" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m bias_correlation.compute_bias_correlation \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results_jigsaw/roberta_jigsaw_all_race_test_400/causal_debias" \
    --split="test" \
    --bias_type="race" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m visualization.visualize_bias_correlation \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results_jigsaw/roberta_jigsaw_all_race_test_400/causal_debias" \
    --split="test" \
    --bias_type="race" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m bias_correlation.compute_bias_correlation \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results_jigsaw/distilbert_jigsaw_race_race_test_400/no_debiasing" \
    --split="test" \
    --bias_type="race" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m visualization.visualize_bias_correlation \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results_jigsaw/distilbert_jigsaw_race_race_test_400/no_debiasing" \
    --split="test" \
    --bias_type="race" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m bias_correlation.compute_bias_correlation \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results_jigsaw/distilbert_jigsaw_all_race_test_400/no_debiasing" \
    --split="test" \
    --bias_type="race" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m visualization.visualize_bias_correlation \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results_jigsaw/distilbert_jigsaw_all_race_test_400/no_debiasing" \
    --split="test" \
    --bias_type="race" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m bias_correlation.compute_bias_correlation \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results_jigsaw/distilbert_jigsaw_race_race_test_400/group_balance" \
    --split="test" \
    --bias_type="race" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m visualization.visualize_bias_correlation \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results_jigsaw/distilbert_jigsaw_race_race_test_400/group_balance" \
    --split="test" \
    --bias_type="race" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m bias_correlation.compute_bias_correlation \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results_jigsaw/distilbert_jigsaw_all_race_test_400/group_balance" \
    --split="test" \
    --bias_type="race" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m visualization.visualize_bias_correlation \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results_jigsaw/distilbert_jigsaw_all_race_test_400/group_balance" \
    --split="test" \
    --bias_type="race" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m bias_correlation.compute_bias_correlation \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results_jigsaw/distilbert_jigsaw_race_race_test_400/group_class_balance" \
    --split="test" \
    --bias_type="race" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m visualization.visualize_bias_correlation \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results_jigsaw/distilbert_jigsaw_race_race_test_400/group_class_balance" \
    --split="test" \
    --bias_type="race" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m bias_correlation.compute_bias_correlation \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results_jigsaw/distilbert_jigsaw_all_race_test_400/group_class_balance" \
    --split="test" \
    --bias_type="race" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m visualization.visualize_bias_correlation \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results_jigsaw/distilbert_jigsaw_all_race_test_400/group_class_balance" \
    --split="test" \
    --bias_type="race" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m bias_correlation.compute_bias_correlation \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results_jigsaw/distilbert_jigsaw_race_race_test_400/cda" \
    --split="test" \
    --bias_type="race" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m visualization.visualize_bias_correlation \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results_jigsaw/distilbert_jigsaw_race_race_test_400/cda" \
    --split="test" \
    --bias_type="race" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m bias_correlation.compute_bias_correlation \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results_jigsaw/distilbert_jigsaw_all_race_test_400/cda" \
    --split="test" \
    --bias_type="race" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m visualization.visualize_bias_correlation \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results_jigsaw/distilbert_jigsaw_all_race_test_400/cda" \
    --split="test" \
    --bias_type="race" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m bias_correlation.compute_bias_correlation \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results_jigsaw/distilbert_jigsaw_race_race_test_400/dropout" \
    --split="test" \
    --bias_type="race" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m visualization.visualize_bias_correlation \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results_jigsaw/distilbert_jigsaw_race_race_test_400/dropout" \
    --split="test" \
    --bias_type="race" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m bias_correlation.compute_bias_correlation \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results_jigsaw/distilbert_jigsaw_all_race_test_400/dropout" \
    --split="test" \
    --bias_type="race" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m visualization.visualize_bias_correlation \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results_jigsaw/distilbert_jigsaw_all_race_test_400/dropout" \
    --split="test" \
    --bias_type="race" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m bias_correlation.compute_bias_correlation \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results_jigsaw/distilbert_jigsaw_race_race_test_400/attention_entropy" \
    --split="test" \
    --bias_type="race" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m visualization.visualize_bias_correlation \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results_jigsaw/distilbert_jigsaw_race_race_test_400/attention_entropy" \
    --split="test" \
    --bias_type="race" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m bias_correlation.compute_bias_correlation \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results_jigsaw/distilbert_jigsaw_all_race_test_400/attention_entropy" \
    --split="test" \
    --bias_type="race" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m visualization.visualize_bias_correlation \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results_jigsaw/distilbert_jigsaw_all_race_test_400/attention_entropy" \
    --split="test" \
    --bias_type="race" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m bias_correlation.compute_bias_correlation \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results_jigsaw/distilbert_jigsaw_race_race_test_400/causal_debias" \
    --split="test" \
    --bias_type="race" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m visualization.visualize_bias_correlation \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results_jigsaw/distilbert_jigsaw_race_race_test_400/causal_debias" \
    --split="test" \
    --bias_type="race" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m bias_correlation.compute_bias_correlation \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results_jigsaw/distilbert_jigsaw_all_race_test_400/causal_debias" \
    --split="test" \
    --bias_type="race" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m visualization.visualize_bias_correlation \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results_jigsaw/distilbert_jigsaw_all_race_test_400/causal_debias" \
    --split="test" \
    --bias_type="race" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"


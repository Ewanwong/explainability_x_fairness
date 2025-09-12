#!/bin/bash
python -m analysis.correlation_w_abs \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/bert_civil_gender_gender_test_2000/no_debiasing" \
    --split="test" \
    --bias_type="gender" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m analysis.visualization_w_abs \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/bert_civil_gender_gender_test_2000/no_debiasing" \
    --split="test" \
    --bias_type="gender" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m analysis.correlation_w_abs \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/bert_civil_all_gender_test_2000/no_debiasing" \
    --split="test" \
    --bias_type="gender" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m analysis.visualization_w_abs \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/bert_civil_all_gender_test_2000/no_debiasing" \
    --split="test" \
    --bias_type="gender" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m analysis.correlation_w_abs \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/bert_civil_gender_gender_test_2000/group_balance" \
    --split="test" \
    --bias_type="gender" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m analysis.visualization_w_abs \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/bert_civil_gender_gender_test_2000/group_balance" \
    --split="test" \
    --bias_type="gender" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m analysis.correlation_w_abs \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/bert_civil_all_gender_test_2000/group_balance" \
    --split="test" \
    --bias_type="gender" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m analysis.visualization_w_abs \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/bert_civil_all_gender_test_2000/group_balance" \
    --split="test" \
    --bias_type="gender" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m analysis.correlation_w_abs \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/bert_civil_gender_gender_test_2000/group_class_balance" \
    --split="test" \
    --bias_type="gender" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m analysis.visualization_w_abs \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/bert_civil_gender_gender_test_2000/group_class_balance" \
    --split="test" \
    --bias_type="gender" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m analysis.correlation_w_abs \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/bert_civil_all_gender_test_2000/group_class_balance" \
    --split="test" \
    --bias_type="gender" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m analysis.visualization_w_abs \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/bert_civil_all_gender_test_2000/group_class_balance" \
    --split="test" \
    --bias_type="gender" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m analysis.correlation_w_abs \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/bert_civil_gender_gender_test_2000/cda" \
    --split="test" \
    --bias_type="gender" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m analysis.visualization_w_abs \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/bert_civil_gender_gender_test_2000/cda" \
    --split="test" \
    --bias_type="gender" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m analysis.correlation_w_abs \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/bert_civil_all_gender_test_2000/cda" \
    --split="test" \
    --bias_type="gender" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m analysis.visualization_w_abs \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/bert_civil_all_gender_test_2000/cda" \
    --split="test" \
    --bias_type="gender" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m analysis.correlation_w_abs \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/bert_civil_gender_gender_test_2000/dropout" \
    --split="test" \
    --bias_type="gender" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m analysis.visualization_w_abs \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/bert_civil_gender_gender_test_2000/dropout" \
    --split="test" \
    --bias_type="gender" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m analysis.correlation_w_abs \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/bert_civil_all_gender_test_2000/dropout" \
    --split="test" \
    --bias_type="gender" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m analysis.visualization_w_abs \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/bert_civil_all_gender_test_2000/dropout" \
    --split="test" \
    --bias_type="gender" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m analysis.correlation_w_abs \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/bert_civil_gender_gender_test_2000/attention_entropy" \
    --split="test" \
    --bias_type="gender" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m analysis.visualization_w_abs \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/bert_civil_gender_gender_test_2000/attention_entropy" \
    --split="test" \
    --bias_type="gender" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m analysis.correlation_w_abs \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/bert_civil_all_gender_test_2000/attention_entropy" \
    --split="test" \
    --bias_type="gender" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m analysis.visualization_w_abs \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/bert_civil_all_gender_test_2000/attention_entropy" \
    --split="test" \
    --bias_type="gender" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m analysis.correlation_w_abs \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/bert_civil_gender_gender_test_2000/causal_debias" \
    --split="test" \
    --bias_type="gender" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m analysis.visualization_w_abs \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/bert_civil_gender_gender_test_2000/causal_debias" \
    --split="test" \
    --bias_type="gender" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m analysis.correlation_w_abs \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/bert_civil_all_gender_test_2000/causal_debias" \
    --split="test" \
    --bias_type="gender" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"

python -m analysis.visualization_w_abs \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results/bert_civil_all_gender_test_2000/causal_debias" \
    --split="test" \
    --bias_type="gender" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"
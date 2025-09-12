#!/bin/bash
python -m model_selection.compute_model_selection_correlation \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --bias_type="gender" \
    --model_type="bert" \
    --num_test_examples=2000 \
    --num_val_examples=100 \

python -m model_selection.compute_model_selection_correlation \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --bias_type="gender" \
    --model_type="roberta" \
    --num_test_examples=2000 \
    --num_val_examples=100 \

python -m model_selection.compute_model_selection_correlation \
    --results_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --bias_type="gender" \
    --model_type="distilbert" \
    --num_test_examples=2000 \
    --num_val_examples=100 \


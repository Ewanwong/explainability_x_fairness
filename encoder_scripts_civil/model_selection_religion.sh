#!/bin/bash

python -m model_selection.compute_model_selection_correlation \
    --results_dir="/scratch/yifwang/fairness_x_explainability/encoder_results_civil" \
    --val_results_dir="/scratch/yifwang/fairness_x_explainability/encoder_results_civil/val_results" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --experiment_type=encoder \
    --train_type=single \
    --bias_type="religion" \
    --model_type="bert" \
    --num_test_examples=1000 \
    --num_val_examples=20 \
    --test_seed=-1 \
    --val_seed=1

python -m model_selection.compute_model_selection_correlation \
    --results_dir="/scratch/yifwang/fairness_x_explainability/encoder_results_civil" \
    --val_results_dir="/scratch/yifwang/fairness_x_explainability/encoder_results_civil/val_results" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --experiment_type=encoder \
    --train_type=all \
    --bias_type="religion" \
    --model_type="bert" \
    --num_test_examples=1000 \
    --num_val_examples=20 \
    --test_seed=-1 \
    --val_seed=1

python -m model_selection.compute_model_selection_correlation \
    --results_dir="/scratch/yifwang/fairness_x_explainability/encoder_results_civil" \
    --val_results_dir="/scratch/yifwang/fairness_x_explainability/encoder_results_civil/val_results" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --experiment_type=encoder \
    --train_type=single \
    --bias_type="religion" \
    --model_type="roberta" \
    --num_test_examples=1000 \
    --num_val_examples=20 \
    --test_seed=-1 \
    --val_seed=1

python -m model_selection.compute_model_selection_correlation \
    --results_dir="/scratch/yifwang/fairness_x_explainability/encoder_results_civil" \
    --val_results_dir="/scratch/yifwang/fairness_x_explainability/encoder_results_civil/val_results" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --experiment_type=encoder \
    --train_type=all \
    --bias_type="religion" \
    --model_type="roberta" \
    --num_test_examples=1000 \
    --num_val_examples=20 \
    --test_seed=-1 \
    --val_seed=1

python -m model_selection.compute_model_selection_correlation \
    --results_dir="/scratch/yifwang/fairness_x_explainability/encoder_results_civil" \
    --val_results_dir="/scratch/yifwang/fairness_x_explainability/encoder_results_civil/val_results" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --experiment_type=encoder \
    --train_type=single \
    --bias_type="religion" \
    --model_type="distilbert" \
    --num_test_examples=1000 \
    --num_val_examples=20 \
    --test_seed=-1 \
    --val_seed=1

python -m model_selection.compute_model_selection_correlation \
    --results_dir="/scratch/yifwang/fairness_x_explainability/encoder_results_civil" \
    --val_results_dir="/scratch/yifwang/fairness_x_explainability/encoder_results_civil/val_results" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --experiment_type=encoder \
    --train_type=all \
    --bias_type="religion" \
    --model_type="distilbert" \
    --num_test_examples=1000 \
    --num_val_examples=20 \
    --test_seed=-1 \
    --val_seed=1

python -m model_selection.compute_model_selection_correlation \
    --results_dir="/scratch/yifwang/fairness_x_explainability/encoder_results_civil" \
    --val_results_dir="/scratch/yifwang/fairness_x_explainability/encoder_results_civil/val_results" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --experiment_type=encoder \
    --train_type=single \
    --bias_type="religion" \
    --model_type="bert" \
    --num_test_examples=1000 \
    --num_val_examples=20 \
    --test_seed=-1 \
    --val_seed=2

python -m model_selection.compute_model_selection_correlation \
    --results_dir="/scratch/yifwang/fairness_x_explainability/encoder_results_civil" \
    --val_results_dir="/scratch/yifwang/fairness_x_explainability/encoder_results_civil/val_results" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --experiment_type=encoder \
    --train_type=all \
    --bias_type="religion" \
    --model_type="bert" \
    --num_test_examples=1000 \
    --num_val_examples=20 \
    --test_seed=-1 \
    --val_seed=2

python -m model_selection.compute_model_selection_correlation \
    --results_dir="/scratch/yifwang/fairness_x_explainability/encoder_results_civil" \
    --val_results_dir="/scratch/yifwang/fairness_x_explainability/encoder_results_civil/val_results" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --experiment_type=encoder \
    --train_type=single \
    --bias_type="religion" \
    --model_type="roberta" \
    --num_test_examples=1000 \
    --num_val_examples=20 \
    --test_seed=-1 \
    --val_seed=2

python -m model_selection.compute_model_selection_correlation \
    --results_dir="/scratch/yifwang/fairness_x_explainability/encoder_results_civil" \
    --val_results_dir="/scratch/yifwang/fairness_x_explainability/encoder_results_civil/val_results" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --experiment_type=encoder \
    --train_type=all \
    --bias_type="religion" \
    --model_type="roberta" \
    --num_test_examples=1000 \
    --num_val_examples=20 \
    --test_seed=-1 \
    --val_seed=2

python -m model_selection.compute_model_selection_correlation \
    --results_dir="/scratch/yifwang/fairness_x_explainability/encoder_results_civil" \
    --val_results_dir="/scratch/yifwang/fairness_x_explainability/encoder_results_civil/val_results" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --experiment_type=encoder \
    --train_type=single \
    --bias_type="religion" \
    --model_type="distilbert" \
    --num_test_examples=1000 \
    --num_val_examples=20 \
    --test_seed=-1 \
    --val_seed=2

python -m model_selection.compute_model_selection_correlation \
    --results_dir="/scratch/yifwang/fairness_x_explainability/encoder_results_civil" \
    --val_results_dir="/scratch/yifwang/fairness_x_explainability/encoder_results_civil/val_results" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --experiment_type=encoder \
    --train_type=all \
    --bias_type="religion" \
    --model_type="distilbert" \
    --num_test_examples=1000 \
    --num_val_examples=20 \
    --test_seed=-1 \
    --val_seed=2

python -m model_selection.compute_model_selection_correlation \
    --results_dir="/scratch/yifwang/fairness_x_explainability/encoder_results_civil" \
    --val_results_dir="/scratch/yifwang/fairness_x_explainability/encoder_results_civil/val_results" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --experiment_type=encoder \
    --train_type=single \
    --bias_type="religion" \
    --model_type="bert" \
    --num_test_examples=1000 \
    --num_val_examples=20 \
    --test_seed=-1 \
    --val_seed=3

python -m model_selection.compute_model_selection_correlation \
    --results_dir="/scratch/yifwang/fairness_x_explainability/encoder_results_civil" \
    --val_results_dir="/scratch/yifwang/fairness_x_explainability/encoder_results_civil/val_results" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --experiment_type=encoder \
    --train_type=all \
    --bias_type="religion" \
    --model_type="bert" \
    --num_test_examples=1000 \
    --num_val_examples=20 \
    --test_seed=-1 \
    --val_seed=3

python -m model_selection.compute_model_selection_correlation \
    --results_dir="/scratch/yifwang/fairness_x_explainability/encoder_results_civil" \
    --val_results_dir="/scratch/yifwang/fairness_x_explainability/encoder_results_civil/val_results" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --experiment_type=encoder \
    --train_type=single \
    --bias_type="religion" \
    --model_type="roberta" \
    --num_test_examples=1000 \
    --num_val_examples=20 \
    --test_seed=-1 \
    --val_seed=3

python -m model_selection.compute_model_selection_correlation \
    --results_dir="/scratch/yifwang/fairness_x_explainability/encoder_results_civil" \
    --val_results_dir="/scratch/yifwang/fairness_x_explainability/encoder_results_civil/val_results" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --experiment_type=encoder \
    --train_type=all \
    --bias_type="religion" \
    --model_type="roberta" \
    --num_test_examples=1000 \
    --num_val_examples=20 \
    --test_seed=-1 \
    --val_seed=3

python -m model_selection.compute_model_selection_correlation \
    --results_dir="/scratch/yifwang/fairness_x_explainability/encoder_results_civil" \
    --val_results_dir="/scratch/yifwang/fairness_x_explainability/encoder_results_civil/val_results" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --experiment_type=encoder \
    --train_type=single \
    --bias_type="religion" \
    --model_type="distilbert" \
    --num_test_examples=1000 \
    --num_val_examples=20 \
    --test_seed=-1 \
    --val_seed=3

python -m model_selection.compute_model_selection_correlation \
    --results_dir="/scratch/yifwang/fairness_x_explainability/encoder_results_civil" \
    --val_results_dir="/scratch/yifwang/fairness_x_explainability/encoder_results_civil/val_results" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --experiment_type=encoder \
    --train_type=all \
    --bias_type="religion" \
    --model_type="distilbert" \
    --num_test_examples=1000 \
    --num_val_examples=20 \
    --test_seed=-1 \
    --val_seed=3

python -m model_selection.compute_model_selection_correlation \
    --results_dir="/scratch/yifwang/fairness_x_explainability/encoder_results_civil" \
    --val_results_dir="/scratch/yifwang/fairness_x_explainability/encoder_results_civil/val_results" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --experiment_type=encoder \
    --train_type=single \
    --bias_type="religion" \
    --model_type="bert" \
    --num_test_examples=1000 \
    --num_val_examples=20 \
    --test_seed=-1 \
    --val_seed=4

python -m model_selection.compute_model_selection_correlation \
    --results_dir="/scratch/yifwang/fairness_x_explainability/encoder_results_civil" \
    --val_results_dir="/scratch/yifwang/fairness_x_explainability/encoder_results_civil/val_results" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --experiment_type=encoder \
    --train_type=all \
    --bias_type="religion" \
    --model_type="bert" \
    --num_test_examples=1000 \
    --num_val_examples=20 \
    --test_seed=-1 \
    --val_seed=4

python -m model_selection.compute_model_selection_correlation \
    --results_dir="/scratch/yifwang/fairness_x_explainability/encoder_results_civil" \
    --val_results_dir="/scratch/yifwang/fairness_x_explainability/encoder_results_civil/val_results" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --experiment_type=encoder \
    --train_type=single \
    --bias_type="religion" \
    --model_type="roberta" \
    --num_test_examples=1000 \
    --num_val_examples=20 \
    --test_seed=-1 \
    --val_seed=4

python -m model_selection.compute_model_selection_correlation \
    --results_dir="/scratch/yifwang/fairness_x_explainability/encoder_results_civil" \
    --val_results_dir="/scratch/yifwang/fairness_x_explainability/encoder_results_civil/val_results" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --experiment_type=encoder \
    --train_type=all \
    --bias_type="religion" \
    --model_type="roberta" \
    --num_test_examples=1000 \
    --num_val_examples=20 \
    --test_seed=-1 \
    --val_seed=4

python -m model_selection.compute_model_selection_correlation \
    --results_dir="/scratch/yifwang/fairness_x_explainability/encoder_results_civil" \
    --val_results_dir="/scratch/yifwang/fairness_x_explainability/encoder_results_civil/val_results" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --experiment_type=encoder \
    --train_type=single \
    --bias_type="religion" \
    --model_type="distilbert" \
    --num_test_examples=1000 \
    --num_val_examples=20 \
    --test_seed=-1 \
    --val_seed=4

python -m model_selection.compute_model_selection_correlation \
    --results_dir="/scratch/yifwang/fairness_x_explainability/encoder_results_civil" \
    --val_results_dir="/scratch/yifwang/fairness_x_explainability/encoder_results_civil/val_results" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --experiment_type=encoder \
    --train_type=all \
    --bias_type="religion" \
    --model_type="distilbert" \
    --num_test_examples=1000 \
    --num_val_examples=20 \
    --test_seed=-1 \
    --val_seed=4

python -m model_selection.compute_model_selection_correlation \
    --results_dir="/scratch/yifwang/fairness_x_explainability/encoder_results_civil" \
    --val_results_dir="/scratch/yifwang/fairness_x_explainability/encoder_results_civil/val_results" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --experiment_type=encoder \
    --train_type=single \
    --bias_type="religion" \
    --model_type="bert" \
    --num_test_examples=1000 \
    --num_val_examples=20 \
    --test_seed=-1 \
    --val_seed=5

python -m model_selection.compute_model_selection_correlation \
    --results_dir="/scratch/yifwang/fairness_x_explainability/encoder_results_civil" \
    --val_results_dir="/scratch/yifwang/fairness_x_explainability/encoder_results_civil/val_results" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --experiment_type=encoder \
    --train_type=all \
    --bias_type="religion" \
    --model_type="bert" \
    --num_test_examples=1000 \
    --num_val_examples=20 \
    --test_seed=-1 \
    --val_seed=5

python -m model_selection.compute_model_selection_correlation \
    --results_dir="/scratch/yifwang/fairness_x_explainability/encoder_results_civil" \
    --val_results_dir="/scratch/yifwang/fairness_x_explainability/encoder_results_civil/val_results" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --experiment_type=encoder \
    --train_type=single \
    --bias_type="religion" \
    --model_type="roberta" \
    --num_test_examples=1000 \
    --num_val_examples=20 \
    --test_seed=-1 \
    --val_seed=5

python -m model_selection.compute_model_selection_correlation \
    --results_dir="/scratch/yifwang/fairness_x_explainability/encoder_results_civil" \
    --val_results_dir="/scratch/yifwang/fairness_x_explainability/encoder_results_civil/val_results" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --experiment_type=encoder \
    --train_type=all \
    --bias_type="religion" \
    --model_type="roberta" \
    --num_test_examples=1000 \
    --num_val_examples=20 \
    --test_seed=-1 \
    --val_seed=5

python -m model_selection.compute_model_selection_correlation \
    --results_dir="/scratch/yifwang/fairness_x_explainability/encoder_results_civil" \
    --val_results_dir="/scratch/yifwang/fairness_x_explainability/encoder_results_civil/val_results" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --experiment_type=encoder \
    --train_type=single \
    --bias_type="religion" \
    --model_type="distilbert" \
    --num_test_examples=1000 \
    --num_val_examples=20 \
    --test_seed=-1 \
    --val_seed=5

python -m model_selection.compute_model_selection_correlation \
    --results_dir="/scratch/yifwang/fairness_x_explainability/encoder_results_civil" \
    --val_results_dir="/scratch/yifwang/fairness_x_explainability/encoder_results_civil/val_results" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --experiment_type=encoder \
    --train_type=all \
    --bias_type="religion" \
    --model_type="distilbert" \
    --num_test_examples=1000 \
    --num_val_examples=20 \
    --test_seed=-1 \
    --val_seed=5

python -m model_selection.compute_model_selection_correlation \
    --results_dir="/scratch/yifwang/fairness_x_explainability/encoder_results_civil" \
    --val_results_dir="/scratch/yifwang/fairness_x_explainability/encoder_results_civil/val_results" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --experiment_type=encoder \
    --train_type=single \
    --bias_type="religion" \
    --model_type="bert" \
    --num_test_examples=1000 \
    --num_val_examples=20 \
    --test_seed=-1 \
    --val_seed=6

python -m model_selection.compute_model_selection_correlation \
    --results_dir="/scratch/yifwang/fairness_x_explainability/encoder_results_civil" \
    --val_results_dir="/scratch/yifwang/fairness_x_explainability/encoder_results_civil/val_results" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --experiment_type=encoder \
    --train_type=all \
    --bias_type="religion" \
    --model_type="bert" \
    --num_test_examples=1000 \
    --num_val_examples=20 \
    --test_seed=-1 \
    --val_seed=6

python -m model_selection.compute_model_selection_correlation \
    --results_dir="/scratch/yifwang/fairness_x_explainability/encoder_results_civil" \
    --val_results_dir="/scratch/yifwang/fairness_x_explainability/encoder_results_civil/val_results" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --experiment_type=encoder \
    --train_type=single \
    --bias_type="religion" \
    --model_type="roberta" \
    --num_test_examples=1000 \
    --num_val_examples=20 \
    --test_seed=-1 \
    --val_seed=6

python -m model_selection.compute_model_selection_correlation \
    --results_dir="/scratch/yifwang/fairness_x_explainability/encoder_results_civil" \
    --val_results_dir="/scratch/yifwang/fairness_x_explainability/encoder_results_civil/val_results" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --experiment_type=encoder \
    --train_type=all \
    --bias_type="religion" \
    --model_type="roberta" \
    --num_test_examples=1000 \
    --num_val_examples=20 \
    --test_seed=-1 \
    --val_seed=6

python -m model_selection.compute_model_selection_correlation \
    --results_dir="/scratch/yifwang/fairness_x_explainability/encoder_results_civil" \
    --val_results_dir="/scratch/yifwang/fairness_x_explainability/encoder_results_civil/val_results" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --experiment_type=encoder \
    --train_type=single \
    --bias_type="religion" \
    --model_type="distilbert" \
    --num_test_examples=1000 \
    --num_val_examples=20 \
    --test_seed=-1 \
    --val_seed=6

python -m model_selection.compute_model_selection_correlation \
    --results_dir="/scratch/yifwang/fairness_x_explainability/encoder_results_civil" \
    --val_results_dir="/scratch/yifwang/fairness_x_explainability/encoder_results_civil/val_results" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --experiment_type=encoder \
    --train_type=all \
    --bias_type="religion" \
    --model_type="distilbert" \
    --num_test_examples=1000 \
    --num_val_examples=20 \
    --test_seed=-1 \
    --val_seed=6

python -m model_selection.compute_model_selection_correlation \
    --results_dir="/scratch/yifwang/fairness_x_explainability/encoder_results_civil" \
    --val_results_dir="/scratch/yifwang/fairness_x_explainability/encoder_results_civil/val_results" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --experiment_type=encoder \
    --train_type=single \
    --bias_type="religion" \
    --model_type="bert" \
    --num_test_examples=1000 \
    --num_val_examples=20 \
    --test_seed=-1 \
    --val_seed=7

python -m model_selection.compute_model_selection_correlation \
    --results_dir="/scratch/yifwang/fairness_x_explainability/encoder_results_civil" \
    --val_results_dir="/scratch/yifwang/fairness_x_explainability/encoder_results_civil/val_results" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --experiment_type=encoder \
    --train_type=all \
    --bias_type="religion" \
    --model_type="bert" \
    --num_test_examples=1000 \
    --num_val_examples=20 \
    --test_seed=-1 \
    --val_seed=7

python -m model_selection.compute_model_selection_correlation \
    --results_dir="/scratch/yifwang/fairness_x_explainability/encoder_results_civil" \
    --val_results_dir="/scratch/yifwang/fairness_x_explainability/encoder_results_civil/val_results" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --experiment_type=encoder \
    --train_type=single \
    --bias_type="religion" \
    --model_type="roberta" \
    --num_test_examples=1000 \
    --num_val_examples=20 \
    --test_seed=-1 \
    --val_seed=7

python -m model_selection.compute_model_selection_correlation \
    --results_dir="/scratch/yifwang/fairness_x_explainability/encoder_results_civil" \
    --val_results_dir="/scratch/yifwang/fairness_x_explainability/encoder_results_civil/val_results" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --experiment_type=encoder \
    --train_type=all \
    --bias_type="religion" \
    --model_type="roberta" \
    --num_test_examples=1000 \
    --num_val_examples=20 \
    --test_seed=-1 \
    --val_seed=7

python -m model_selection.compute_model_selection_correlation \
    --results_dir="/scratch/yifwang/fairness_x_explainability/encoder_results_civil" \
    --val_results_dir="/scratch/yifwang/fairness_x_explainability/encoder_results_civil/val_results" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --experiment_type=encoder \
    --train_type=single \
    --bias_type="religion" \
    --model_type="distilbert" \
    --num_test_examples=1000 \
    --num_val_examples=20 \
    --test_seed=-1 \
    --val_seed=7

python -m model_selection.compute_model_selection_correlation \
    --results_dir="/scratch/yifwang/fairness_x_explainability/encoder_results_civil" \
    --val_results_dir="/scratch/yifwang/fairness_x_explainability/encoder_results_civil/val_results" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --experiment_type=encoder \
    --train_type=all \
    --bias_type="religion" \
    --model_type="distilbert" \
    --num_test_examples=1000 \
    --num_val_examples=20 \
    --test_seed=-1 \
    --val_seed=7

python -m model_selection.compute_model_selection_correlation \
    --results_dir="/scratch/yifwang/fairness_x_explainability/encoder_results_civil" \
    --val_results_dir="/scratch/yifwang/fairness_x_explainability/encoder_results_civil/val_results" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --experiment_type=encoder \
    --train_type=single \
    --bias_type="religion" \
    --model_type="bert" \
    --num_test_examples=1000 \
    --num_val_examples=20 \
    --test_seed=-1 \
    --val_seed=8

python -m model_selection.compute_model_selection_correlation \
    --results_dir="/scratch/yifwang/fairness_x_explainability/encoder_results_civil" \
    --val_results_dir="/scratch/yifwang/fairness_x_explainability/encoder_results_civil/val_results" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --experiment_type=encoder \
    --train_type=all \
    --bias_type="religion" \
    --model_type="bert" \
    --num_test_examples=1000 \
    --num_val_examples=20 \
    --test_seed=-1 \
    --val_seed=8

python -m model_selection.compute_model_selection_correlation \
    --results_dir="/scratch/yifwang/fairness_x_explainability/encoder_results_civil" \
    --val_results_dir="/scratch/yifwang/fairness_x_explainability/encoder_results_civil/val_results" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --experiment_type=encoder \
    --train_type=single \
    --bias_type="religion" \
    --model_type="roberta" \
    --num_test_examples=1000 \
    --num_val_examples=20 \
    --test_seed=-1 \
    --val_seed=8

python -m model_selection.compute_model_selection_correlation \
    --results_dir="/scratch/yifwang/fairness_x_explainability/encoder_results_civil" \
    --val_results_dir="/scratch/yifwang/fairness_x_explainability/encoder_results_civil/val_results" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --experiment_type=encoder \
    --train_type=all \
    --bias_type="religion" \
    --model_type="roberta" \
    --num_test_examples=1000 \
    --num_val_examples=20 \
    --test_seed=-1 \
    --val_seed=8

python -m model_selection.compute_model_selection_correlation \
    --results_dir="/scratch/yifwang/fairness_x_explainability/encoder_results_civil" \
    --val_results_dir="/scratch/yifwang/fairness_x_explainability/encoder_results_civil/val_results" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --experiment_type=encoder \
    --train_type=single \
    --bias_type="religion" \
    --model_type="distilbert" \
    --num_test_examples=1000 \
    --num_val_examples=20 \
    --test_seed=-1 \
    --val_seed=8

python -m model_selection.compute_model_selection_correlation \
    --results_dir="/scratch/yifwang/fairness_x_explainability/encoder_results_civil" \
    --val_results_dir="/scratch/yifwang/fairness_x_explainability/encoder_results_civil/val_results" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --experiment_type=encoder \
    --train_type=all \
    --bias_type="religion" \
    --model_type="distilbert" \
    --num_test_examples=1000 \
    --num_val_examples=20 \
    --test_seed=-1 \
    --val_seed=8

python -m model_selection.compute_model_selection_correlation \
    --results_dir="/scratch/yifwang/fairness_x_explainability/encoder_results_civil" \
    --val_results_dir="/scratch/yifwang/fairness_x_explainability/encoder_results_civil/val_results" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --experiment_type=encoder \
    --train_type=single \
    --bias_type="religion" \
    --model_type="bert" \
    --num_test_examples=1000 \
    --num_val_examples=20 \
    --test_seed=-1 \
    --val_seed=9

python -m model_selection.compute_model_selection_correlation \
    --results_dir="/scratch/yifwang/fairness_x_explainability/encoder_results_civil" \
    --val_results_dir="/scratch/yifwang/fairness_x_explainability/encoder_results_civil/val_results" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --experiment_type=encoder \
    --train_type=all \
    --bias_type="religion" \
    --model_type="bert" \
    --num_test_examples=1000 \
    --num_val_examples=20 \
    --test_seed=-1 \
    --val_seed=9

python -m model_selection.compute_model_selection_correlation \
    --results_dir="/scratch/yifwang/fairness_x_explainability/encoder_results_civil" \
    --val_results_dir="/scratch/yifwang/fairness_x_explainability/encoder_results_civil/val_results" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --experiment_type=encoder \
    --train_type=single \
    --bias_type="religion" \
    --model_type="roberta" \
    --num_test_examples=1000 \
    --num_val_examples=20 \
    --test_seed=-1 \
    --val_seed=9

python -m model_selection.compute_model_selection_correlation \
    --results_dir="/scratch/yifwang/fairness_x_explainability/encoder_results_civil" \
    --val_results_dir="/scratch/yifwang/fairness_x_explainability/encoder_results_civil/val_results" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --experiment_type=encoder \
    --train_type=all \
    --bias_type="religion" \
    --model_type="roberta" \
    --num_test_examples=1000 \
    --num_val_examples=20 \
    --test_seed=-1 \
    --val_seed=9

python -m model_selection.compute_model_selection_correlation \
    --results_dir="/scratch/yifwang/fairness_x_explainability/encoder_results_civil" \
    --val_results_dir="/scratch/yifwang/fairness_x_explainability/encoder_results_civil/val_results" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --experiment_type=encoder \
    --train_type=single \
    --bias_type="religion" \
    --model_type="distilbert" \
    --num_test_examples=1000 \
    --num_val_examples=20 \
    --test_seed=-1 \
    --val_seed=9

python -m model_selection.compute_model_selection_correlation \
    --results_dir="/scratch/yifwang/fairness_x_explainability/encoder_results_civil" \
    --val_results_dir="/scratch/yifwang/fairness_x_explainability/encoder_results_civil/val_results" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --experiment_type=encoder \
    --train_type=all \
    --bias_type="religion" \
    --model_type="distilbert" \
    --num_test_examples=1000 \
    --num_val_examples=20 \
    --test_seed=-1 \
    --val_seed=9

python -m model_selection.compute_model_selection_correlation \
    --results_dir="/scratch/yifwang/fairness_x_explainability/encoder_results_civil" \
    --val_results_dir="/scratch/yifwang/fairness_x_explainability/encoder_results_civil/val_results" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --experiment_type=encoder \
    --train_type=single \
    --bias_type="religion" \
    --model_type="bert" \
    --num_test_examples=1000 \
    --num_val_examples=20 \
    --test_seed=-1 \
    --val_seed=10

python -m model_selection.compute_model_selection_correlation \
    --results_dir="/scratch/yifwang/fairness_x_explainability/encoder_results_civil" \
    --val_results_dir="/scratch/yifwang/fairness_x_explainability/encoder_results_civil/val_results" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --experiment_type=encoder \
    --train_type=all \
    --bias_type="religion" \
    --model_type="bert" \
    --num_test_examples=1000 \
    --num_val_examples=20 \
    --test_seed=-1 \
    --val_seed=10

python -m model_selection.compute_model_selection_correlation \
    --results_dir="/scratch/yifwang/fairness_x_explainability/encoder_results_civil" \
    --val_results_dir="/scratch/yifwang/fairness_x_explainability/encoder_results_civil/val_results" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --experiment_type=encoder \
    --train_type=single \
    --bias_type="religion" \
    --model_type="roberta" \
    --num_test_examples=1000 \
    --num_val_examples=20 \
    --test_seed=-1 \
    --val_seed=10

python -m model_selection.compute_model_selection_correlation \
    --results_dir="/scratch/yifwang/fairness_x_explainability/encoder_results_civil" \
    --val_results_dir="/scratch/yifwang/fairness_x_explainability/encoder_results_civil/val_results" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --experiment_type=encoder \
    --train_type=all \
    --bias_type="religion" \
    --model_type="roberta" \
    --num_test_examples=1000 \
    --num_val_examples=20 \
    --test_seed=-1 \
    --val_seed=10

python -m model_selection.compute_model_selection_correlation \
    --results_dir="/scratch/yifwang/fairness_x_explainability/encoder_results_civil" \
    --val_results_dir="/scratch/yifwang/fairness_x_explainability/encoder_results_civil/val_results" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --experiment_type=encoder \
    --train_type=single \
    --bias_type="religion" \
    --model_type="distilbert" \
    --num_test_examples=1000 \
    --num_val_examples=20 \
    --test_seed=-1 \
    --val_seed=10

python -m model_selection.compute_model_selection_correlation \
    --results_dir="/scratch/yifwang/fairness_x_explainability/encoder_results_civil" \
    --val_results_dir="/scratch/yifwang/fairness_x_explainability/encoder_results_civil/val_results" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --experiment_type=encoder \
    --train_type=all \
    --bias_type="religion" \
    --model_type="distilbert" \
    --num_test_examples=1000 \
    --num_val_examples=20 \
    --test_seed=-1 \
    --val_seed=10

python -m model_selection.compute_model_selection_correlation \
    --results_dir="/scratch/yifwang/fairness_x_explainability/encoder_results_civil" \
    --val_results_dir="/scratch/yifwang/fairness_x_explainability/encoder_results_civil/val_results" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --experiment_type=encoder \
    --train_type=single \
    --bias_type="religion" \
    --model_type="bert" \
    --num_test_examples=1000 \
    --num_val_examples=20 \
    --test_seed=-1 \
    --val_seed=11

python -m model_selection.compute_model_selection_correlation \
    --results_dir="/scratch/yifwang/fairness_x_explainability/encoder_results_civil" \
    --val_results_dir="/scratch/yifwang/fairness_x_explainability/encoder_results_civil/val_results" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --experiment_type=encoder \
    --train_type=all \
    --bias_type="religion" \
    --model_type="bert" \
    --num_test_examples=1000 \
    --num_val_examples=20 \
    --test_seed=-1 \
    --val_seed=11

python -m model_selection.compute_model_selection_correlation \
    --results_dir="/scratch/yifwang/fairness_x_explainability/encoder_results_civil" \
    --val_results_dir="/scratch/yifwang/fairness_x_explainability/encoder_results_civil/val_results" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --experiment_type=encoder \
    --train_type=single \
    --bias_type="religion" \
    --model_type="roberta" \
    --num_test_examples=1000 \
    --num_val_examples=20 \
    --test_seed=-1 \
    --val_seed=11

python -m model_selection.compute_model_selection_correlation \
    --results_dir="/scratch/yifwang/fairness_x_explainability/encoder_results_civil" \
    --val_results_dir="/scratch/yifwang/fairness_x_explainability/encoder_results_civil/val_results" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --experiment_type=encoder \
    --train_type=all \
    --bias_type="religion" \
    --model_type="roberta" \
    --num_test_examples=1000 \
    --num_val_examples=20 \
    --test_seed=-1 \
    --val_seed=11

python -m model_selection.compute_model_selection_correlation \
    --results_dir="/scratch/yifwang/fairness_x_explainability/encoder_results_civil" \
    --val_results_dir="/scratch/yifwang/fairness_x_explainability/encoder_results_civil/val_results" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --experiment_type=encoder \
    --train_type=single \
    --bias_type="religion" \
    --model_type="distilbert" \
    --num_test_examples=1000 \
    --num_val_examples=20 \
    --test_seed=-1 \
    --val_seed=11

python -m model_selection.compute_model_selection_correlation \
    --results_dir="/scratch/yifwang/fairness_x_explainability/encoder_results_civil" \
    --val_results_dir="/scratch/yifwang/fairness_x_explainability/encoder_results_civil/val_results" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --experiment_type=encoder \
    --train_type=all \
    --bias_type="religion" \
    --model_type="distilbert" \
    --num_test_examples=1000 \
    --num_val_examples=20 \
    --test_seed=-1 \
    --val_seed=11

python -m model_selection.compute_model_selection_correlation \
    --results_dir="/scratch/yifwang/fairness_x_explainability/encoder_results_civil" \
    --val_results_dir="/scratch/yifwang/fairness_x_explainability/encoder_results_civil/val_results" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --experiment_type=encoder \
    --train_type=single \
    --bias_type="religion" \
    --model_type="bert" \
    --num_test_examples=1000 \
    --num_val_examples=20 \
    --test_seed=-1 \
    --val_seed=12

python -m model_selection.compute_model_selection_correlation \
    --results_dir="/scratch/yifwang/fairness_x_explainability/encoder_results_civil" \
    --val_results_dir="/scratch/yifwang/fairness_x_explainability/encoder_results_civil/val_results" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --experiment_type=encoder \
    --train_type=all \
    --bias_type="religion" \
    --model_type="bert" \
    --num_test_examples=1000 \
    --num_val_examples=20 \
    --test_seed=-1 \
    --val_seed=12

python -m model_selection.compute_model_selection_correlation \
    --results_dir="/scratch/yifwang/fairness_x_explainability/encoder_results_civil" \
    --val_results_dir="/scratch/yifwang/fairness_x_explainability/encoder_results_civil/val_results" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --experiment_type=encoder \
    --train_type=single \
    --bias_type="religion" \
    --model_type="roberta" \
    --num_test_examples=1000 \
    --num_val_examples=20 \
    --test_seed=-1 \
    --val_seed=12

python -m model_selection.compute_model_selection_correlation \
    --results_dir="/scratch/yifwang/fairness_x_explainability/encoder_results_civil" \
    --val_results_dir="/scratch/yifwang/fairness_x_explainability/encoder_results_civil/val_results" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --experiment_type=encoder \
    --train_type=all \
    --bias_type="religion" \
    --model_type="roberta" \
    --num_test_examples=1000 \
    --num_val_examples=20 \
    --test_seed=-1 \
    --val_seed=12

python -m model_selection.compute_model_selection_correlation \
    --results_dir="/scratch/yifwang/fairness_x_explainability/encoder_results_civil" \
    --val_results_dir="/scratch/yifwang/fairness_x_explainability/encoder_results_civil/val_results" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --experiment_type=encoder \
    --train_type=single \
    --bias_type="religion" \
    --model_type="distilbert" \
    --num_test_examples=1000 \
    --num_val_examples=20 \
    --test_seed=-1 \
    --val_seed=12

python -m model_selection.compute_model_selection_correlation \
    --results_dir="/scratch/yifwang/fairness_x_explainability/encoder_results_civil" \
    --val_results_dir="/scratch/yifwang/fairness_x_explainability/encoder_results_civil/val_results" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --experiment_type=encoder \
    --train_type=all \
    --bias_type="religion" \
    --model_type="distilbert" \
    --num_test_examples=1000 \
    --num_val_examples=20 \
    --test_seed=-1 \
    --val_seed=12

python -m model_selection.compute_model_selection_correlation \
    --results_dir="/scratch/yifwang/fairness_x_explainability/encoder_results_civil" \
    --val_results_dir="/scratch/yifwang/fairness_x_explainability/encoder_results_civil/val_results" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --experiment_type=encoder \
    --train_type=single \
    --bias_type="religion" \
    --model_type="bert" \
    --num_test_examples=1000 \
    --num_val_examples=20 \
    --test_seed=-1 \
    --val_seed=13

python -m model_selection.compute_model_selection_correlation \
    --results_dir="/scratch/yifwang/fairness_x_explainability/encoder_results_civil" \
    --val_results_dir="/scratch/yifwang/fairness_x_explainability/encoder_results_civil/val_results" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --experiment_type=encoder \
    --train_type=all \
    --bias_type="religion" \
    --model_type="bert" \
    --num_test_examples=1000 \
    --num_val_examples=20 \
    --test_seed=-1 \
    --val_seed=13

python -m model_selection.compute_model_selection_correlation \
    --results_dir="/scratch/yifwang/fairness_x_explainability/encoder_results_civil" \
    --val_results_dir="/scratch/yifwang/fairness_x_explainability/encoder_results_civil/val_results" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --experiment_type=encoder \
    --train_type=single \
    --bias_type="religion" \
    --model_type="roberta" \
    --num_test_examples=1000 \
    --num_val_examples=20 \
    --test_seed=-1 \
    --val_seed=13

python -m model_selection.compute_model_selection_correlation \
    --results_dir="/scratch/yifwang/fairness_x_explainability/encoder_results_civil" \
    --val_results_dir="/scratch/yifwang/fairness_x_explainability/encoder_results_civil/val_results" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --experiment_type=encoder \
    --train_type=all \
    --bias_type="religion" \
    --model_type="roberta" \
    --num_test_examples=1000 \
    --num_val_examples=20 \
    --test_seed=-1 \
    --val_seed=13

python -m model_selection.compute_model_selection_correlation \
    --results_dir="/scratch/yifwang/fairness_x_explainability/encoder_results_civil" \
    --val_results_dir="/scratch/yifwang/fairness_x_explainability/encoder_results_civil/val_results" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --experiment_type=encoder \
    --train_type=single \
    --bias_type="religion" \
    --model_type="distilbert" \
    --num_test_examples=1000 \
    --num_val_examples=20 \
    --test_seed=-1 \
    --val_seed=13

python -m model_selection.compute_model_selection_correlation \
    --results_dir="/scratch/yifwang/fairness_x_explainability/encoder_results_civil" \
    --val_results_dir="/scratch/yifwang/fairness_x_explainability/encoder_results_civil/val_results" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --experiment_type=encoder \
    --train_type=all \
    --bias_type="religion" \
    --model_type="distilbert" \
    --num_test_examples=1000 \
    --num_val_examples=20 \
    --test_seed=-1 \
    --val_seed=13

python -m model_selection.compute_model_selection_correlation \
    --results_dir="/scratch/yifwang/fairness_x_explainability/encoder_results_civil" \
    --val_results_dir="/scratch/yifwang/fairness_x_explainability/encoder_results_civil/val_results" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --experiment_type=encoder \
    --train_type=single \
    --bias_type="religion" \
    --model_type="bert" \
    --num_test_examples=1000 \
    --num_val_examples=20 \
    --test_seed=-1 \
    --val_seed=14

python -m model_selection.compute_model_selection_correlation \
    --results_dir="/scratch/yifwang/fairness_x_explainability/encoder_results_civil" \
    --val_results_dir="/scratch/yifwang/fairness_x_explainability/encoder_results_civil/val_results" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --experiment_type=encoder \
    --train_type=all \
    --bias_type="religion" \
    --model_type="bert" \
    --num_test_examples=1000 \
    --num_val_examples=20 \
    --test_seed=-1 \
    --val_seed=14

python -m model_selection.compute_model_selection_correlation \
    --results_dir="/scratch/yifwang/fairness_x_explainability/encoder_results_civil" \
    --val_results_dir="/scratch/yifwang/fairness_x_explainability/encoder_results_civil/val_results" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --experiment_type=encoder \
    --train_type=single \
    --bias_type="religion" \
    --model_type="roberta" \
    --num_test_examples=1000 \
    --num_val_examples=20 \
    --test_seed=-1 \
    --val_seed=14

python -m model_selection.compute_model_selection_correlation \
    --results_dir="/scratch/yifwang/fairness_x_explainability/encoder_results_civil" \
    --val_results_dir="/scratch/yifwang/fairness_x_explainability/encoder_results_civil/val_results" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --experiment_type=encoder \
    --train_type=all \
    --bias_type="religion" \
    --model_type="roberta" \
    --num_test_examples=1000 \
    --num_val_examples=20 \
    --test_seed=-1 \
    --val_seed=14

python -m model_selection.compute_model_selection_correlation \
    --results_dir="/scratch/yifwang/fairness_x_explainability/encoder_results_civil" \
    --val_results_dir="/scratch/yifwang/fairness_x_explainability/encoder_results_civil/val_results" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --experiment_type=encoder \
    --train_type=single \
    --bias_type="religion" \
    --model_type="distilbert" \
    --num_test_examples=1000 \
    --num_val_examples=20 \
    --test_seed=-1 \
    --val_seed=14

python -m model_selection.compute_model_selection_correlation \
    --results_dir="/scratch/yifwang/fairness_x_explainability/encoder_results_civil" \
    --val_results_dir="/scratch/yifwang/fairness_x_explainability/encoder_results_civil/val_results" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --experiment_type=encoder \
    --train_type=all \
    --bias_type="religion" \
    --model_type="distilbert" \
    --num_test_examples=1000 \
    --num_val_examples=20 \
    --test_seed=-1 \
    --val_seed=14

python -m model_selection.compute_model_selection_correlation \
    --results_dir="/scratch/yifwang/fairness_x_explainability/encoder_results_civil" \
    --val_results_dir="/scratch/yifwang/fairness_x_explainability/encoder_results_civil/val_results" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --experiment_type=encoder \
    --train_type=single \
    --bias_type="religion" \
    --model_type="bert" \
    --num_test_examples=1000 \
    --num_val_examples=20 \
    --test_seed=-1 \
    --val_seed=15

python -m model_selection.compute_model_selection_correlation \
    --results_dir="/scratch/yifwang/fairness_x_explainability/encoder_results_civil" \
    --val_results_dir="/scratch/yifwang/fairness_x_explainability/encoder_results_civil/val_results" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --experiment_type=encoder \
    --train_type=all \
    --bias_type="religion" \
    --model_type="bert" \
    --num_test_examples=1000 \
    --num_val_examples=20 \
    --test_seed=-1 \
    --val_seed=15

python -m model_selection.compute_model_selection_correlation \
    --results_dir="/scratch/yifwang/fairness_x_explainability/encoder_results_civil" \
    --val_results_dir="/scratch/yifwang/fairness_x_explainability/encoder_results_civil/val_results" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --experiment_type=encoder \
    --train_type=single \
    --bias_type="religion" \
    --model_type="roberta" \
    --num_test_examples=1000 \
    --num_val_examples=20 \
    --test_seed=-1 \
    --val_seed=15

python -m model_selection.compute_model_selection_correlation \
    --results_dir="/scratch/yifwang/fairness_x_explainability/encoder_results_civil" \
    --val_results_dir="/scratch/yifwang/fairness_x_explainability/encoder_results_civil/val_results" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --experiment_type=encoder \
    --train_type=all \
    --bias_type="religion" \
    --model_type="roberta" \
    --num_test_examples=1000 \
    --num_val_examples=20 \
    --test_seed=-1 \
    --val_seed=15

python -m model_selection.compute_model_selection_correlation \
    --results_dir="/scratch/yifwang/fairness_x_explainability/encoder_results_civil" \
    --val_results_dir="/scratch/yifwang/fairness_x_explainability/encoder_results_civil/val_results" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --experiment_type=encoder \
    --train_type=single \
    --bias_type="religion" \
    --model_type="distilbert" \
    --num_test_examples=1000 \
    --num_val_examples=20 \
    --test_seed=-1 \
    --val_seed=15

python -m model_selection.compute_model_selection_correlation \
    --results_dir="/scratch/yifwang/fairness_x_explainability/encoder_results_civil" \
    --val_results_dir="/scratch/yifwang/fairness_x_explainability/encoder_results_civil/val_results" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --experiment_type=encoder \
    --train_type=all \
    --bias_type="religion" \
    --model_type="distilbert" \
    --num_test_examples=1000 \
    --num_val_examples=20 \
    --test_seed=-1 \
    --val_seed=15

python -m model_selection.compute_model_selection_correlation \
    --results_dir="/scratch/yifwang/fairness_x_explainability/encoder_results_civil" \
    --val_results_dir="/scratch/yifwang/fairness_x_explainability/encoder_results_civil/val_results" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --experiment_type=encoder \
    --train_type=single \
    --bias_type="religion" \
    --model_type="bert" \
    --num_test_examples=1000 \
    --num_val_examples=20 \
    --test_seed=-1 \
    --val_seed=16

python -m model_selection.compute_model_selection_correlation \
    --results_dir="/scratch/yifwang/fairness_x_explainability/encoder_results_civil" \
    --val_results_dir="/scratch/yifwang/fairness_x_explainability/encoder_results_civil/val_results" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --experiment_type=encoder \
    --train_type=all \
    --bias_type="religion" \
    --model_type="bert" \
    --num_test_examples=1000 \
    --num_val_examples=20 \
    --test_seed=-1 \
    --val_seed=16

python -m model_selection.compute_model_selection_correlation \
    --results_dir="/scratch/yifwang/fairness_x_explainability/encoder_results_civil" \
    --val_results_dir="/scratch/yifwang/fairness_x_explainability/encoder_results_civil/val_results" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --experiment_type=encoder \
    --train_type=single \
    --bias_type="religion" \
    --model_type="roberta" \
    --num_test_examples=1000 \
    --num_val_examples=20 \
    --test_seed=-1 \
    --val_seed=16

python -m model_selection.compute_model_selection_correlation \
    --results_dir="/scratch/yifwang/fairness_x_explainability/encoder_results_civil" \
    --val_results_dir="/scratch/yifwang/fairness_x_explainability/encoder_results_civil/val_results" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --experiment_type=encoder \
    --train_type=all \
    --bias_type="religion" \
    --model_type="roberta" \
    --num_test_examples=1000 \
    --num_val_examples=20 \
    --test_seed=-1 \
    --val_seed=16

python -m model_selection.compute_model_selection_correlation \
    --results_dir="/scratch/yifwang/fairness_x_explainability/encoder_results_civil" \
    --val_results_dir="/scratch/yifwang/fairness_x_explainability/encoder_results_civil/val_results" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --experiment_type=encoder \
    --train_type=single \
    --bias_type="religion" \
    --model_type="distilbert" \
    --num_test_examples=1000 \
    --num_val_examples=20 \
    --test_seed=-1 \
    --val_seed=16

python -m model_selection.compute_model_selection_correlation \
    --results_dir="/scratch/yifwang/fairness_x_explainability/encoder_results_civil" \
    --val_results_dir="/scratch/yifwang/fairness_x_explainability/encoder_results_civil/val_results" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --experiment_type=encoder \
    --train_type=all \
    --bias_type="religion" \
    --model_type="distilbert" \
    --num_test_examples=1000 \
    --num_val_examples=20 \
    --test_seed=-1 \
    --val_seed=16

python -m model_selection.compute_model_selection_correlation \
    --results_dir="/scratch/yifwang/fairness_x_explainability/encoder_results_civil" \
    --val_results_dir="/scratch/yifwang/fairness_x_explainability/encoder_results_civil/val_results" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --experiment_type=encoder \
    --train_type=single \
    --bias_type="religion" \
    --model_type="bert" \
    --num_test_examples=1000 \
    --num_val_examples=20 \
    --test_seed=-1 \
    --val_seed=17

python -m model_selection.compute_model_selection_correlation \
    --results_dir="/scratch/yifwang/fairness_x_explainability/encoder_results_civil" \
    --val_results_dir="/scratch/yifwang/fairness_x_explainability/encoder_results_civil/val_results" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --experiment_type=encoder \
    --train_type=all \
    --bias_type="religion" \
    --model_type="bert" \
    --num_test_examples=1000 \
    --num_val_examples=20 \
    --test_seed=-1 \
    --val_seed=17

python -m model_selection.compute_model_selection_correlation \
    --results_dir="/scratch/yifwang/fairness_x_explainability/encoder_results_civil" \
    --val_results_dir="/scratch/yifwang/fairness_x_explainability/encoder_results_civil/val_results" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --experiment_type=encoder \
    --train_type=single \
    --bias_type="religion" \
    --model_type="roberta" \
    --num_test_examples=1000 \
    --num_val_examples=20 \
    --test_seed=-1 \
    --val_seed=17

python -m model_selection.compute_model_selection_correlation \
    --results_dir="/scratch/yifwang/fairness_x_explainability/encoder_results_civil" \
    --val_results_dir="/scratch/yifwang/fairness_x_explainability/encoder_results_civil/val_results" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --experiment_type=encoder \
    --train_type=all \
    --bias_type="religion" \
    --model_type="roberta" \
    --num_test_examples=1000 \
    --num_val_examples=20 \
    --test_seed=-1 \
    --val_seed=17

python -m model_selection.compute_model_selection_correlation \
    --results_dir="/scratch/yifwang/fairness_x_explainability/encoder_results_civil" \
    --val_results_dir="/scratch/yifwang/fairness_x_explainability/encoder_results_civil/val_results" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --experiment_type=encoder \
    --train_type=single \
    --bias_type="religion" \
    --model_type="distilbert" \
    --num_test_examples=1000 \
    --num_val_examples=20 \
    --test_seed=-1 \
    --val_seed=17

python -m model_selection.compute_model_selection_correlation \
    --results_dir="/scratch/yifwang/fairness_x_explainability/encoder_results_civil" \
    --val_results_dir="/scratch/yifwang/fairness_x_explainability/encoder_results_civil/val_results" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --experiment_type=encoder \
    --train_type=all \
    --bias_type="religion" \
    --model_type="distilbert" \
    --num_test_examples=1000 \
    --num_val_examples=20 \
    --test_seed=-1 \
    --val_seed=17

python -m model_selection.compute_model_selection_correlation \
    --results_dir="/scratch/yifwang/fairness_x_explainability/encoder_results_civil" \
    --val_results_dir="/scratch/yifwang/fairness_x_explainability/encoder_results_civil/val_results" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --experiment_type=encoder \
    --train_type=single \
    --bias_type="religion" \
    --model_type="bert" \
    --num_test_examples=1000 \
    --num_val_examples=20 \
    --test_seed=-1 \
    --val_seed=18

python -m model_selection.compute_model_selection_correlation \
    --results_dir="/scratch/yifwang/fairness_x_explainability/encoder_results_civil" \
    --val_results_dir="/scratch/yifwang/fairness_x_explainability/encoder_results_civil/val_results" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --experiment_type=encoder \
    --train_type=all \
    --bias_type="religion" \
    --model_type="bert" \
    --num_test_examples=1000 \
    --num_val_examples=20 \
    --test_seed=-1 \
    --val_seed=18

python -m model_selection.compute_model_selection_correlation \
    --results_dir="/scratch/yifwang/fairness_x_explainability/encoder_results_civil" \
    --val_results_dir="/scratch/yifwang/fairness_x_explainability/encoder_results_civil/val_results" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --experiment_type=encoder \
    --train_type=single \
    --bias_type="religion" \
    --model_type="roberta" \
    --num_test_examples=1000 \
    --num_val_examples=20 \
    --test_seed=-1 \
    --val_seed=18

python -m model_selection.compute_model_selection_correlation \
    --results_dir="/scratch/yifwang/fairness_x_explainability/encoder_results_civil" \
    --val_results_dir="/scratch/yifwang/fairness_x_explainability/encoder_results_civil/val_results" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --experiment_type=encoder \
    --train_type=all \
    --bias_type="religion" \
    --model_type="roberta" \
    --num_test_examples=1000 \
    --num_val_examples=20 \
    --test_seed=-1 \
    --val_seed=18

python -m model_selection.compute_model_selection_correlation \
    --results_dir="/scratch/yifwang/fairness_x_explainability/encoder_results_civil" \
    --val_results_dir="/scratch/yifwang/fairness_x_explainability/encoder_results_civil/val_results" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --experiment_type=encoder \
    --train_type=single \
    --bias_type="religion" \
    --model_type="distilbert" \
    --num_test_examples=1000 \
    --num_val_examples=20 \
    --test_seed=-1 \
    --val_seed=18

python -m model_selection.compute_model_selection_correlation \
    --results_dir="/scratch/yifwang/fairness_x_explainability/encoder_results_civil" \
    --val_results_dir="/scratch/yifwang/fairness_x_explainability/encoder_results_civil/val_results" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --experiment_type=encoder \
    --train_type=all \
    --bias_type="religion" \
    --model_type="distilbert" \
    --num_test_examples=1000 \
    --num_val_examples=20 \
    --test_seed=-1 \
    --val_seed=18

python -m model_selection.compute_model_selection_correlation \
    --results_dir="/scratch/yifwang/fairness_x_explainability/encoder_results_civil" \
    --val_results_dir="/scratch/yifwang/fairness_x_explainability/encoder_results_civil/val_results" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --experiment_type=encoder \
    --train_type=single \
    --bias_type="religion" \
    --model_type="bert" \
    --num_test_examples=1000 \
    --num_val_examples=20 \
    --test_seed=-1 \
    --val_seed=19

python -m model_selection.compute_model_selection_correlation \
    --results_dir="/scratch/yifwang/fairness_x_explainability/encoder_results_civil" \
    --val_results_dir="/scratch/yifwang/fairness_x_explainability/encoder_results_civil/val_results" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --experiment_type=encoder \
    --train_type=all \
    --bias_type="religion" \
    --model_type="bert" \
    --num_test_examples=1000 \
    --num_val_examples=20 \
    --test_seed=-1 \
    --val_seed=19

python -m model_selection.compute_model_selection_correlation \
    --results_dir="/scratch/yifwang/fairness_x_explainability/encoder_results_civil" \
    --val_results_dir="/scratch/yifwang/fairness_x_explainability/encoder_results_civil/val_results" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --experiment_type=encoder \
    --train_type=single \
    --bias_type="religion" \
    --model_type="roberta" \
    --num_test_examples=1000 \
    --num_val_examples=20 \
    --test_seed=-1 \
    --val_seed=19

python -m model_selection.compute_model_selection_correlation \
    --results_dir="/scratch/yifwang/fairness_x_explainability/encoder_results_civil" \
    --val_results_dir="/scratch/yifwang/fairness_x_explainability/encoder_results_civil/val_results" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --experiment_type=encoder \
    --train_type=all \
    --bias_type="religion" \
    --model_type="roberta" \
    --num_test_examples=1000 \
    --num_val_examples=20 \
    --test_seed=-1 \
    --val_seed=19

python -m model_selection.compute_model_selection_correlation \
    --results_dir="/scratch/yifwang/fairness_x_explainability/encoder_results_civil" \
    --val_results_dir="/scratch/yifwang/fairness_x_explainability/encoder_results_civil/val_results" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --experiment_type=encoder \
    --train_type=single \
    --bias_type="religion" \
    --model_type="distilbert" \
    --num_test_examples=1000 \
    --num_val_examples=20 \
    --test_seed=-1 \
    --val_seed=19

python -m model_selection.compute_model_selection_correlation \
    --results_dir="/scratch/yifwang/fairness_x_explainability/encoder_results_civil" \
    --val_results_dir="/scratch/yifwang/fairness_x_explainability/encoder_results_civil/val_results" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --experiment_type=encoder \
    --train_type=all \
    --bias_type="religion" \
    --model_type="distilbert" \
    --num_test_examples=1000 \
    --num_val_examples=20 \
    --test_seed=-1 \
    --val_seed=19

python -m model_selection.compute_model_selection_correlation \
    --results_dir="/scratch/yifwang/fairness_x_explainability/encoder_results_civil" \
    --val_results_dir="/scratch/yifwang/fairness_x_explainability/encoder_results_civil/val_results" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --experiment_type=encoder \
    --train_type=single \
    --bias_type="religion" \
    --model_type="bert" \
    --num_test_examples=1000 \
    --num_val_examples=20 \
    --test_seed=-1 \
    --val_seed=42

python -m model_selection.compute_model_selection_correlation \
    --results_dir="/scratch/yifwang/fairness_x_explainability/encoder_results_civil" \
    --val_results_dir="/scratch/yifwang/fairness_x_explainability/encoder_results_civil/val_results" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --experiment_type=encoder \
    --train_type=all \
    --bias_type="religion" \
    --model_type="bert" \
    --num_test_examples=1000 \
    --num_val_examples=20 \
    --test_seed=-1 \
    --val_seed=42

python -m model_selection.compute_model_selection_correlation \
    --results_dir="/scratch/yifwang/fairness_x_explainability/encoder_results_civil" \
    --val_results_dir="/scratch/yifwang/fairness_x_explainability/encoder_results_civil/val_results" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --experiment_type=encoder \
    --train_type=single \
    --bias_type="religion" \
    --model_type="roberta" \
    --num_test_examples=1000 \
    --num_val_examples=20 \
    --test_seed=-1 \
    --val_seed=42

python -m model_selection.compute_model_selection_correlation \
    --results_dir="/scratch/yifwang/fairness_x_explainability/encoder_results_civil" \
    --val_results_dir="/scratch/yifwang/fairness_x_explainability/encoder_results_civil/val_results" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --experiment_type=encoder \
    --train_type=all \
    --bias_type="religion" \
    --model_type="roberta" \
    --num_test_examples=1000 \
    --num_val_examples=20 \
    --test_seed=-1 \
    --val_seed=42

python -m model_selection.compute_model_selection_correlation \
    --results_dir="/scratch/yifwang/fairness_x_explainability/encoder_results_civil" \
    --val_results_dir="/scratch/yifwang/fairness_x_explainability/encoder_results_civil/val_results" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --experiment_type=encoder \
    --train_type=single \
    --bias_type="religion" \
    --model_type="distilbert" \
    --num_test_examples=1000 \
    --num_val_examples=20 \
    --test_seed=-1 \
    --val_seed=42

python -m model_selection.compute_model_selection_correlation \
    --results_dir="/scratch/yifwang/fairness_x_explainability/encoder_results_civil" \
    --val_results_dir="/scratch/yifwang/fairness_x_explainability/encoder_results_civil/val_results" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --experiment_type=encoder \
    --train_type=all \
    --bias_type="religion" \
    --model_type="distilbert" \
    --num_test_examples=1000 \
    --num_val_examples=20 \
    --test_seed=-1 \
    --val_seed=42

python -m model_selection.compute_model_selection_correlation \
    --results_dir="/scratch/yifwang/fairness_x_explainability/encoder_results_civil" \
    --val_results_dir="/scratch/yifwang/fairness_x_explainability/encoder_results_civil/val_results" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --experiment_type=encoder \
    --train_type=single \
    --bias_type="religion" \
    --model_type="bert" \
    --num_test_examples=1000 \
    --num_val_examples=50 \
    --test_seed=-1 \
    --val_seed=1

python -m model_selection.compute_model_selection_correlation \
    --results_dir="/scratch/yifwang/fairness_x_explainability/encoder_results_civil" \
    --val_results_dir="/scratch/yifwang/fairness_x_explainability/encoder_results_civil/val_results" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --experiment_type=encoder \
    --train_type=all \
    --bias_type="religion" \
    --model_type="bert" \
    --num_test_examples=1000 \
    --num_val_examples=50 \
    --test_seed=-1 \
    --val_seed=1

python -m model_selection.compute_model_selection_correlation \
    --results_dir="/scratch/yifwang/fairness_x_explainability/encoder_results_civil" \
    --val_results_dir="/scratch/yifwang/fairness_x_explainability/encoder_results_civil/val_results" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --experiment_type=encoder \
    --train_type=single \
    --bias_type="religion" \
    --model_type="roberta" \
    --num_test_examples=1000 \
    --num_val_examples=50 \
    --test_seed=-1 \
    --val_seed=1

python -m model_selection.compute_model_selection_correlation \
    --results_dir="/scratch/yifwang/fairness_x_explainability/encoder_results_civil" \
    --val_results_dir="/scratch/yifwang/fairness_x_explainability/encoder_results_civil/val_results" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --experiment_type=encoder \
    --train_type=all \
    --bias_type="religion" \
    --model_type="roberta" \
    --num_test_examples=1000 \
    --num_val_examples=50 \
    --test_seed=-1 \
    --val_seed=1

python -m model_selection.compute_model_selection_correlation \
    --results_dir="/scratch/yifwang/fairness_x_explainability/encoder_results_civil" \
    --val_results_dir="/scratch/yifwang/fairness_x_explainability/encoder_results_civil/val_results" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --experiment_type=encoder \
    --train_type=single \
    --bias_type="religion" \
    --model_type="distilbert" \
    --num_test_examples=1000 \
    --num_val_examples=50 \
    --test_seed=-1 \
    --val_seed=1

python -m model_selection.compute_model_selection_correlation \
    --results_dir="/scratch/yifwang/fairness_x_explainability/encoder_results_civil" \
    --val_results_dir="/scratch/yifwang/fairness_x_explainability/encoder_results_civil/val_results" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --experiment_type=encoder \
    --train_type=all \
    --bias_type="religion" \
    --model_type="distilbert" \
    --num_test_examples=1000 \
    --num_val_examples=50 \
    --test_seed=-1 \
    --val_seed=1

python -m model_selection.compute_model_selection_correlation \
    --results_dir="/scratch/yifwang/fairness_x_explainability/encoder_results_civil" \
    --val_results_dir="/scratch/yifwang/fairness_x_explainability/encoder_results_civil/val_results" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --experiment_type=encoder \
    --train_type=single \
    --bias_type="religion" \
    --model_type="bert" \
    --num_test_examples=1000 \
    --num_val_examples=50 \
    --test_seed=-1 \
    --val_seed=2

python -m model_selection.compute_model_selection_correlation \
    --results_dir="/scratch/yifwang/fairness_x_explainability/encoder_results_civil" \
    --val_results_dir="/scratch/yifwang/fairness_x_explainability/encoder_results_civil/val_results" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --experiment_type=encoder \
    --train_type=all \
    --bias_type="religion" \
    --model_type="bert" \
    --num_test_examples=1000 \
    --num_val_examples=50 \
    --test_seed=-1 \
    --val_seed=2

python -m model_selection.compute_model_selection_correlation \
    --results_dir="/scratch/yifwang/fairness_x_explainability/encoder_results_civil" \
    --val_results_dir="/scratch/yifwang/fairness_x_explainability/encoder_results_civil/val_results" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --experiment_type=encoder \
    --train_type=single \
    --bias_type="religion" \
    --model_type="roberta" \
    --num_test_examples=1000 \
    --num_val_examples=50 \
    --test_seed=-1 \
    --val_seed=2

python -m model_selection.compute_model_selection_correlation \
    --results_dir="/scratch/yifwang/fairness_x_explainability/encoder_results_civil" \
    --val_results_dir="/scratch/yifwang/fairness_x_explainability/encoder_results_civil/val_results" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --experiment_type=encoder \
    --train_type=all \
    --bias_type="religion" \
    --model_type="roberta" \
    --num_test_examples=1000 \
    --num_val_examples=50 \
    --test_seed=-1 \
    --val_seed=2

python -m model_selection.compute_model_selection_correlation \
    --results_dir="/scratch/yifwang/fairness_x_explainability/encoder_results_civil" \
    --val_results_dir="/scratch/yifwang/fairness_x_explainability/encoder_results_civil/val_results" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --experiment_type=encoder \
    --train_type=single \
    --bias_type="religion" \
    --model_type="distilbert" \
    --num_test_examples=1000 \
    --num_val_examples=50 \
    --test_seed=-1 \
    --val_seed=2

python -m model_selection.compute_model_selection_correlation \
    --results_dir="/scratch/yifwang/fairness_x_explainability/encoder_results_civil" \
    --val_results_dir="/scratch/yifwang/fairness_x_explainability/encoder_results_civil/val_results" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --experiment_type=encoder \
    --train_type=all \
    --bias_type="religion" \
    --model_type="distilbert" \
    --num_test_examples=1000 \
    --num_val_examples=50 \
    --test_seed=-1 \
    --val_seed=2

python -m model_selection.compute_model_selection_correlation \
    --results_dir="/scratch/yifwang/fairness_x_explainability/encoder_results_civil" \
    --val_results_dir="/scratch/yifwang/fairness_x_explainability/encoder_results_civil/val_results" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --experiment_type=encoder \
    --train_type=single \
    --bias_type="religion" \
    --model_type="bert" \
    --num_test_examples=1000 \
    --num_val_examples=50 \
    --test_seed=-1 \
    --val_seed=3

python -m model_selection.compute_model_selection_correlation \
    --results_dir="/scratch/yifwang/fairness_x_explainability/encoder_results_civil" \
    --val_results_dir="/scratch/yifwang/fairness_x_explainability/encoder_results_civil/val_results" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --experiment_type=encoder \
    --train_type=all \
    --bias_type="religion" \
    --model_type="bert" \
    --num_test_examples=1000 \
    --num_val_examples=50 \
    --test_seed=-1 \
    --val_seed=3

python -m model_selection.compute_model_selection_correlation \
    --results_dir="/scratch/yifwang/fairness_x_explainability/encoder_results_civil" \
    --val_results_dir="/scratch/yifwang/fairness_x_explainability/encoder_results_civil/val_results" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --experiment_type=encoder \
    --train_type=single \
    --bias_type="religion" \
    --model_type="roberta" \
    --num_test_examples=1000 \
    --num_val_examples=50 \
    --test_seed=-1 \
    --val_seed=3

python -m model_selection.compute_model_selection_correlation \
    --results_dir="/scratch/yifwang/fairness_x_explainability/encoder_results_civil" \
    --val_results_dir="/scratch/yifwang/fairness_x_explainability/encoder_results_civil/val_results" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --experiment_type=encoder \
    --train_type=all \
    --bias_type="religion" \
    --model_type="roberta" \
    --num_test_examples=1000 \
    --num_val_examples=50 \
    --test_seed=-1 \
    --val_seed=3

python -m model_selection.compute_model_selection_correlation \
    --results_dir="/scratch/yifwang/fairness_x_explainability/encoder_results_civil" \
    --val_results_dir="/scratch/yifwang/fairness_x_explainability/encoder_results_civil/val_results" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --experiment_type=encoder \
    --train_type=single \
    --bias_type="religion" \
    --model_type="distilbert" \
    --num_test_examples=1000 \
    --num_val_examples=50 \
    --test_seed=-1 \
    --val_seed=3

python -m model_selection.compute_model_selection_correlation \
    --results_dir="/scratch/yifwang/fairness_x_explainability/encoder_results_civil" \
    --val_results_dir="/scratch/yifwang/fairness_x_explainability/encoder_results_civil/val_results" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --experiment_type=encoder \
    --train_type=all \
    --bias_type="religion" \
    --model_type="distilbert" \
    --num_test_examples=1000 \
    --num_val_examples=50 \
    --test_seed=-1 \
    --val_seed=3

python -m model_selection.compute_model_selection_correlation \
    --results_dir="/scratch/yifwang/fairness_x_explainability/encoder_results_civil" \
    --val_results_dir="/scratch/yifwang/fairness_x_explainability/encoder_results_civil/val_results" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --experiment_type=encoder \
    --train_type=single \
    --bias_type="religion" \
    --model_type="bert" \
    --num_test_examples=1000 \
    --num_val_examples=50 \
    --test_seed=-1 \
    --val_seed=4

python -m model_selection.compute_model_selection_correlation \
    --results_dir="/scratch/yifwang/fairness_x_explainability/encoder_results_civil" \
    --val_results_dir="/scratch/yifwang/fairness_x_explainability/encoder_results_civil/val_results" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --experiment_type=encoder \
    --train_type=all \
    --bias_type="religion" \
    --model_type="bert" \
    --num_test_examples=1000 \
    --num_val_examples=50 \
    --test_seed=-1 \
    --val_seed=4

python -m model_selection.compute_model_selection_correlation \
    --results_dir="/scratch/yifwang/fairness_x_explainability/encoder_results_civil" \
    --val_results_dir="/scratch/yifwang/fairness_x_explainability/encoder_results_civil/val_results" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --experiment_type=encoder \
    --train_type=single \
    --bias_type="religion" \
    --model_type="roberta" \
    --num_test_examples=1000 \
    --num_val_examples=50 \
    --test_seed=-1 \
    --val_seed=4

python -m model_selection.compute_model_selection_correlation \
    --results_dir="/scratch/yifwang/fairness_x_explainability/encoder_results_civil" \
    --val_results_dir="/scratch/yifwang/fairness_x_explainability/encoder_results_civil/val_results" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --experiment_type=encoder \
    --train_type=all \
    --bias_type="religion" \
    --model_type="roberta" \
    --num_test_examples=1000 \
    --num_val_examples=50 \
    --test_seed=-1 \
    --val_seed=4

python -m model_selection.compute_model_selection_correlation \
    --results_dir="/scratch/yifwang/fairness_x_explainability/encoder_results_civil" \
    --val_results_dir="/scratch/yifwang/fairness_x_explainability/encoder_results_civil/val_results" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --experiment_type=encoder \
    --train_type=single \
    --bias_type="religion" \
    --model_type="distilbert" \
    --num_test_examples=1000 \
    --num_val_examples=50 \
    --test_seed=-1 \
    --val_seed=4

python -m model_selection.compute_model_selection_correlation \
    --results_dir="/scratch/yifwang/fairness_x_explainability/encoder_results_civil" \
    --val_results_dir="/scratch/yifwang/fairness_x_explainability/encoder_results_civil/val_results" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --experiment_type=encoder \
    --train_type=all \
    --bias_type="religion" \
    --model_type="distilbert" \
    --num_test_examples=1000 \
    --num_val_examples=50 \
    --test_seed=-1 \
    --val_seed=4

python -m model_selection.compute_model_selection_correlation \
    --results_dir="/scratch/yifwang/fairness_x_explainability/encoder_results_civil" \
    --val_results_dir="/scratch/yifwang/fairness_x_explainability/encoder_results_civil/val_results" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --experiment_type=encoder \
    --train_type=single \
    --bias_type="religion" \
    --model_type="bert" \
    --num_test_examples=1000 \
    --num_val_examples=50 \
    --test_seed=-1 \
    --val_seed=5

python -m model_selection.compute_model_selection_correlation \
    --results_dir="/scratch/yifwang/fairness_x_explainability/encoder_results_civil" \
    --val_results_dir="/scratch/yifwang/fairness_x_explainability/encoder_results_civil/val_results" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --experiment_type=encoder \
    --train_type=all \
    --bias_type="religion" \
    --model_type="bert" \
    --num_test_examples=1000 \
    --num_val_examples=50 \
    --test_seed=-1 \
    --val_seed=5

python -m model_selection.compute_model_selection_correlation \
    --results_dir="/scratch/yifwang/fairness_x_explainability/encoder_results_civil" \
    --val_results_dir="/scratch/yifwang/fairness_x_explainability/encoder_results_civil/val_results" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --experiment_type=encoder \
    --train_type=single \
    --bias_type="religion" \
    --model_type="roberta" \
    --num_test_examples=1000 \
    --num_val_examples=50 \
    --test_seed=-1 \
    --val_seed=5

python -m model_selection.compute_model_selection_correlation \
    --results_dir="/scratch/yifwang/fairness_x_explainability/encoder_results_civil" \
    --val_results_dir="/scratch/yifwang/fairness_x_explainability/encoder_results_civil/val_results" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --experiment_type=encoder \
    --train_type=all \
    --bias_type="religion" \
    --model_type="roberta" \
    --num_test_examples=1000 \
    --num_val_examples=50 \
    --test_seed=-1 \
    --val_seed=5

python -m model_selection.compute_model_selection_correlation \
    --results_dir="/scratch/yifwang/fairness_x_explainability/encoder_results_civil" \
    --val_results_dir="/scratch/yifwang/fairness_x_explainability/encoder_results_civil/val_results" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --experiment_type=encoder \
    --train_type=single \
    --bias_type="religion" \
    --model_type="distilbert" \
    --num_test_examples=1000 \
    --num_val_examples=50 \
    --test_seed=-1 \
    --val_seed=5

python -m model_selection.compute_model_selection_correlation \
    --results_dir="/scratch/yifwang/fairness_x_explainability/encoder_results_civil" \
    --val_results_dir="/scratch/yifwang/fairness_x_explainability/encoder_results_civil/val_results" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --experiment_type=encoder \
    --train_type=all \
    --bias_type="religion" \
    --model_type="distilbert" \
    --num_test_examples=1000 \
    --num_val_examples=50 \
    --test_seed=-1 \
    --val_seed=5

python -m model_selection.compute_model_selection_correlation \
    --results_dir="/scratch/yifwang/fairness_x_explainability/encoder_results_civil" \
    --val_results_dir="/scratch/yifwang/fairness_x_explainability/encoder_results_civil/val_results" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --experiment_type=encoder \
    --train_type=single \
    --bias_type="religion" \
    --model_type="bert" \
    --num_test_examples=1000 \
    --num_val_examples=50 \
    --test_seed=-1 \
    --val_seed=6

python -m model_selection.compute_model_selection_correlation \
    --results_dir="/scratch/yifwang/fairness_x_explainability/encoder_results_civil" \
    --val_results_dir="/scratch/yifwang/fairness_x_explainability/encoder_results_civil/val_results" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --experiment_type=encoder \
    --train_type=all \
    --bias_type="religion" \
    --model_type="bert" \
    --num_test_examples=1000 \
    --num_val_examples=50 \
    --test_seed=-1 \
    --val_seed=6

python -m model_selection.compute_model_selection_correlation \
    --results_dir="/scratch/yifwang/fairness_x_explainability/encoder_results_civil" \
    --val_results_dir="/scratch/yifwang/fairness_x_explainability/encoder_results_civil/val_results" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --experiment_type=encoder \
    --train_type=single \
    --bias_type="religion" \
    --model_type="roberta" \
    --num_test_examples=1000 \
    --num_val_examples=50 \
    --test_seed=-1 \
    --val_seed=6

python -m model_selection.compute_model_selection_correlation \
    --results_dir="/scratch/yifwang/fairness_x_explainability/encoder_results_civil" \
    --val_results_dir="/scratch/yifwang/fairness_x_explainability/encoder_results_civil/val_results" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --experiment_type=encoder \
    --train_type=all \
    --bias_type="religion" \
    --model_type="roberta" \
    --num_test_examples=1000 \
    --num_val_examples=50 \
    --test_seed=-1 \
    --val_seed=6

python -m model_selection.compute_model_selection_correlation \
    --results_dir="/scratch/yifwang/fairness_x_explainability/encoder_results_civil" \
    --val_results_dir="/scratch/yifwang/fairness_x_explainability/encoder_results_civil/val_results" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --experiment_type=encoder \
    --train_type=single \
    --bias_type="religion" \
    --model_type="distilbert" \
    --num_test_examples=1000 \
    --num_val_examples=50 \
    --test_seed=-1 \
    --val_seed=6

python -m model_selection.compute_model_selection_correlation \
    --results_dir="/scratch/yifwang/fairness_x_explainability/encoder_results_civil" \
    --val_results_dir="/scratch/yifwang/fairness_x_explainability/encoder_results_civil/val_results" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --experiment_type=encoder \
    --train_type=all \
    --bias_type="religion" \
    --model_type="distilbert" \
    --num_test_examples=1000 \
    --num_val_examples=50 \
    --test_seed=-1 \
    --val_seed=6

python -m model_selection.compute_model_selection_correlation \
    --results_dir="/scratch/yifwang/fairness_x_explainability/encoder_results_civil" \
    --val_results_dir="/scratch/yifwang/fairness_x_explainability/encoder_results_civil/val_results" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --experiment_type=encoder \
    --train_type=single \
    --bias_type="religion" \
    --model_type="bert" \
    --num_test_examples=1000 \
    --num_val_examples=50 \
    --test_seed=-1 \
    --val_seed=7

python -m model_selection.compute_model_selection_correlation \
    --results_dir="/scratch/yifwang/fairness_x_explainability/encoder_results_civil" \
    --val_results_dir="/scratch/yifwang/fairness_x_explainability/encoder_results_civil/val_results" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --experiment_type=encoder \
    --train_type=all \
    --bias_type="religion" \
    --model_type="bert" \
    --num_test_examples=1000 \
    --num_val_examples=50 \
    --test_seed=-1 \
    --val_seed=7

python -m model_selection.compute_model_selection_correlation \
    --results_dir="/scratch/yifwang/fairness_x_explainability/encoder_results_civil" \
    --val_results_dir="/scratch/yifwang/fairness_x_explainability/encoder_results_civil/val_results" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --experiment_type=encoder \
    --train_type=single \
    --bias_type="religion" \
    --model_type="roberta" \
    --num_test_examples=1000 \
    --num_val_examples=50 \
    --test_seed=-1 \
    --val_seed=7

python -m model_selection.compute_model_selection_correlation \
    --results_dir="/scratch/yifwang/fairness_x_explainability/encoder_results_civil" \
    --val_results_dir="/scratch/yifwang/fairness_x_explainability/encoder_results_civil/val_results" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --experiment_type=encoder \
    --train_type=all \
    --bias_type="religion" \
    --model_type="roberta" \
    --num_test_examples=1000 \
    --num_val_examples=50 \
    --test_seed=-1 \
    --val_seed=7

python -m model_selection.compute_model_selection_correlation \
    --results_dir="/scratch/yifwang/fairness_x_explainability/encoder_results_civil" \
    --val_results_dir="/scratch/yifwang/fairness_x_explainability/encoder_results_civil/val_results" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --experiment_type=encoder \
    --train_type=single \
    --bias_type="religion" \
    --model_type="distilbert" \
    --num_test_examples=1000 \
    --num_val_examples=50 \
    --test_seed=-1 \
    --val_seed=7

python -m model_selection.compute_model_selection_correlation \
    --results_dir="/scratch/yifwang/fairness_x_explainability/encoder_results_civil" \
    --val_results_dir="/scratch/yifwang/fairness_x_explainability/encoder_results_civil/val_results" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --experiment_type=encoder \
    --train_type=all \
    --bias_type="religion" \
    --model_type="distilbert" \
    --num_test_examples=1000 \
    --num_val_examples=50 \
    --test_seed=-1 \
    --val_seed=7

python -m model_selection.compute_model_selection_correlation \
    --results_dir="/scratch/yifwang/fairness_x_explainability/encoder_results_civil" \
    --val_results_dir="/scratch/yifwang/fairness_x_explainability/encoder_results_civil/val_results" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --experiment_type=encoder \
    --train_type=single \
    --bias_type="religion" \
    --model_type="bert" \
    --num_test_examples=1000 \
    --num_val_examples=50 \
    --test_seed=-1 \
    --val_seed=8

python -m model_selection.compute_model_selection_correlation \
    --results_dir="/scratch/yifwang/fairness_x_explainability/encoder_results_civil" \
    --val_results_dir="/scratch/yifwang/fairness_x_explainability/encoder_results_civil/val_results" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --experiment_type=encoder \
    --train_type=all \
    --bias_type="religion" \
    --model_type="bert" \
    --num_test_examples=1000 \
    --num_val_examples=50 \
    --test_seed=-1 \
    --val_seed=8

python -m model_selection.compute_model_selection_correlation \
    --results_dir="/scratch/yifwang/fairness_x_explainability/encoder_results_civil" \
    --val_results_dir="/scratch/yifwang/fairness_x_explainability/encoder_results_civil/val_results" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --experiment_type=encoder \
    --train_type=single \
    --bias_type="religion" \
    --model_type="roberta" \
    --num_test_examples=1000 \
    --num_val_examples=50 \
    --test_seed=-1 \
    --val_seed=8

python -m model_selection.compute_model_selection_correlation \
    --results_dir="/scratch/yifwang/fairness_x_explainability/encoder_results_civil" \
    --val_results_dir="/scratch/yifwang/fairness_x_explainability/encoder_results_civil/val_results" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --experiment_type=encoder \
    --train_type=all \
    --bias_type="religion" \
    --model_type="roberta" \
    --num_test_examples=1000 \
    --num_val_examples=50 \
    --test_seed=-1 \
    --val_seed=8

python -m model_selection.compute_model_selection_correlation \
    --results_dir="/scratch/yifwang/fairness_x_explainability/encoder_results_civil" \
    --val_results_dir="/scratch/yifwang/fairness_x_explainability/encoder_results_civil/val_results" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --experiment_type=encoder \
    --train_type=single \
    --bias_type="religion" \
    --model_type="distilbert" \
    --num_test_examples=1000 \
    --num_val_examples=50 \
    --test_seed=-1 \
    --val_seed=8

python -m model_selection.compute_model_selection_correlation \
    --results_dir="/scratch/yifwang/fairness_x_explainability/encoder_results_civil" \
    --val_results_dir="/scratch/yifwang/fairness_x_explainability/encoder_results_civil/val_results" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --experiment_type=encoder \
    --train_type=all \
    --bias_type="religion" \
    --model_type="distilbert" \
    --num_test_examples=1000 \
    --num_val_examples=50 \
    --test_seed=-1 \
    --val_seed=8

python -m model_selection.compute_model_selection_correlation \
    --results_dir="/scratch/yifwang/fairness_x_explainability/encoder_results_civil" \
    --val_results_dir="/scratch/yifwang/fairness_x_explainability/encoder_results_civil/val_results" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --experiment_type=encoder \
    --train_type=single \
    --bias_type="religion" \
    --model_type="bert" \
    --num_test_examples=1000 \
    --num_val_examples=50 \
    --test_seed=-1 \
    --val_seed=9

python -m model_selection.compute_model_selection_correlation \
    --results_dir="/scratch/yifwang/fairness_x_explainability/encoder_results_civil" \
    --val_results_dir="/scratch/yifwang/fairness_x_explainability/encoder_results_civil/val_results" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --experiment_type=encoder \
    --train_type=all \
    --bias_type="religion" \
    --model_type="bert" \
    --num_test_examples=1000 \
    --num_val_examples=50 \
    --test_seed=-1 \
    --val_seed=9

python -m model_selection.compute_model_selection_correlation \
    --results_dir="/scratch/yifwang/fairness_x_explainability/encoder_results_civil" \
    --val_results_dir="/scratch/yifwang/fairness_x_explainability/encoder_results_civil/val_results" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --experiment_type=encoder \
    --train_type=single \
    --bias_type="religion" \
    --model_type="roberta" \
    --num_test_examples=1000 \
    --num_val_examples=50 \
    --test_seed=-1 \
    --val_seed=9

python -m model_selection.compute_model_selection_correlation \
    --results_dir="/scratch/yifwang/fairness_x_explainability/encoder_results_civil" \
    --val_results_dir="/scratch/yifwang/fairness_x_explainability/encoder_results_civil/val_results" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --experiment_type=encoder \
    --train_type=all \
    --bias_type="religion" \
    --model_type="roberta" \
    --num_test_examples=1000 \
    --num_val_examples=50 \
    --test_seed=-1 \
    --val_seed=9

python -m model_selection.compute_model_selection_correlation \
    --results_dir="/scratch/yifwang/fairness_x_explainability/encoder_results_civil" \
    --val_results_dir="/scratch/yifwang/fairness_x_explainability/encoder_results_civil/val_results" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --experiment_type=encoder \
    --train_type=single \
    --bias_type="religion" \
    --model_type="distilbert" \
    --num_test_examples=1000 \
    --num_val_examples=50 \
    --test_seed=-1 \
    --val_seed=9

python -m model_selection.compute_model_selection_correlation \
    --results_dir="/scratch/yifwang/fairness_x_explainability/encoder_results_civil" \
    --val_results_dir="/scratch/yifwang/fairness_x_explainability/encoder_results_civil/val_results" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --experiment_type=encoder \
    --train_type=all \
    --bias_type="religion" \
    --model_type="distilbert" \
    --num_test_examples=1000 \
    --num_val_examples=50 \
    --test_seed=-1 \
    --val_seed=9

python -m model_selection.compute_model_selection_correlation \
    --results_dir="/scratch/yifwang/fairness_x_explainability/encoder_results_civil" \
    --val_results_dir="/scratch/yifwang/fairness_x_explainability/encoder_results_civil/val_results" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --experiment_type=encoder \
    --train_type=single \
    --bias_type="religion" \
    --model_type="bert" \
    --num_test_examples=1000 \
    --num_val_examples=50 \
    --test_seed=-1 \
    --val_seed=42

python -m model_selection.compute_model_selection_correlation \
    --results_dir="/scratch/yifwang/fairness_x_explainability/encoder_results_civil" \
    --val_results_dir="/scratch/yifwang/fairness_x_explainability/encoder_results_civil/val_results" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --experiment_type=encoder \
    --train_type=all \
    --bias_type="religion" \
    --model_type="bert" \
    --num_test_examples=1000 \
    --num_val_examples=50 \
    --test_seed=-1 \
    --val_seed=42

python -m model_selection.compute_model_selection_correlation \
    --results_dir="/scratch/yifwang/fairness_x_explainability/encoder_results_civil" \
    --val_results_dir="/scratch/yifwang/fairness_x_explainability/encoder_results_civil/val_results" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --experiment_type=encoder \
    --train_type=single \
    --bias_type="religion" \
    --model_type="roberta" \
    --num_test_examples=1000 \
    --num_val_examples=50 \
    --test_seed=-1 \
    --val_seed=42

python -m model_selection.compute_model_selection_correlation \
    --results_dir="/scratch/yifwang/fairness_x_explainability/encoder_results_civil" \
    --val_results_dir="/scratch/yifwang/fairness_x_explainability/encoder_results_civil/val_results" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --experiment_type=encoder \
    --train_type=all \
    --bias_type="religion" \
    --model_type="roberta" \
    --num_test_examples=1000 \
    --num_val_examples=50 \
    --test_seed=-1 \
    --val_seed=42

python -m model_selection.compute_model_selection_correlation \
    --results_dir="/scratch/yifwang/fairness_x_explainability/encoder_results_civil" \
    --val_results_dir="/scratch/yifwang/fairness_x_explainability/encoder_results_civil/val_results" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --experiment_type=encoder \
    --train_type=single \
    --bias_type="religion" \
    --model_type="distilbert" \
    --num_test_examples=1000 \
    --num_val_examples=50 \
    --test_seed=-1 \
    --val_seed=42

python -m model_selection.compute_model_selection_correlation \
    --results_dir="/scratch/yifwang/fairness_x_explainability/encoder_results_civil" \
    --val_results_dir="/scratch/yifwang/fairness_x_explainability/encoder_results_civil/val_results" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --experiment_type=encoder \
    --train_type=all \
    --bias_type="religion" \
    --model_type="distilbert" \
    --num_test_examples=1000 \
    --num_val_examples=50 \
    --test_seed=-1 \
    --val_seed=42

python -m model_selection.compute_model_selection_correlation \
    --results_dir="/scratch/yifwang/fairness_x_explainability/encoder_results_civil" \
    --val_results_dir="/scratch/yifwang/fairness_x_explainability/encoder_results_civil/val_results" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --experiment_type=encoder \
    --train_type=single \
    --bias_type="religion" \
    --model_type="bert" \
    --num_test_examples=1000 \
    --num_val_examples=100 \
    --test_seed=-1 \
    --val_seed=1

python -m model_selection.compute_model_selection_correlation \
    --results_dir="/scratch/yifwang/fairness_x_explainability/encoder_results_civil" \
    --val_results_dir="/scratch/yifwang/fairness_x_explainability/encoder_results_civil/val_results" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --experiment_type=encoder \
    --train_type=all \
    --bias_type="religion" \
    --model_type="bert" \
    --num_test_examples=1000 \
    --num_val_examples=100 \
    --test_seed=-1 \
    --val_seed=1

python -m model_selection.compute_model_selection_correlation \
    --results_dir="/scratch/yifwang/fairness_x_explainability/encoder_results_civil" \
    --val_results_dir="/scratch/yifwang/fairness_x_explainability/encoder_results_civil/val_results" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --experiment_type=encoder \
    --train_type=single \
    --bias_type="religion" \
    --model_type="roberta" \
    --num_test_examples=1000 \
    --num_val_examples=100 \
    --test_seed=-1 \
    --val_seed=1

python -m model_selection.compute_model_selection_correlation \
    --results_dir="/scratch/yifwang/fairness_x_explainability/encoder_results_civil" \
    --val_results_dir="/scratch/yifwang/fairness_x_explainability/encoder_results_civil/val_results" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --experiment_type=encoder \
    --train_type=all \
    --bias_type="religion" \
    --model_type="roberta" \
    --num_test_examples=1000 \
    --num_val_examples=100 \
    --test_seed=-1 \
    --val_seed=1

python -m model_selection.compute_model_selection_correlation \
    --results_dir="/scratch/yifwang/fairness_x_explainability/encoder_results_civil" \
    --val_results_dir="/scratch/yifwang/fairness_x_explainability/encoder_results_civil/val_results" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --experiment_type=encoder \
    --train_type=single \
    --bias_type="religion" \
    --model_type="distilbert" \
    --num_test_examples=1000 \
    --num_val_examples=100 \
    --test_seed=-1 \
    --val_seed=1

python -m model_selection.compute_model_selection_correlation \
    --results_dir="/scratch/yifwang/fairness_x_explainability/encoder_results_civil" \
    --val_results_dir="/scratch/yifwang/fairness_x_explainability/encoder_results_civil/val_results" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --experiment_type=encoder \
    --train_type=all \
    --bias_type="religion" \
    --model_type="distilbert" \
    --num_test_examples=1000 \
    --num_val_examples=100 \
    --test_seed=-1 \
    --val_seed=1

python -m model_selection.compute_model_selection_correlation \
    --results_dir="/scratch/yifwang/fairness_x_explainability/encoder_results_civil" \
    --val_results_dir="/scratch/yifwang/fairness_x_explainability/encoder_results_civil/val_results" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --experiment_type=encoder \
    --train_type=single \
    --bias_type="religion" \
    --model_type="bert" \
    --num_test_examples=1000 \
    --num_val_examples=100 \
    --test_seed=-1 \
    --val_seed=2

python -m model_selection.compute_model_selection_correlation \
    --results_dir="/scratch/yifwang/fairness_x_explainability/encoder_results_civil" \
    --val_results_dir="/scratch/yifwang/fairness_x_explainability/encoder_results_civil/val_results" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --experiment_type=encoder \
    --train_type=all \
    --bias_type="religion" \
    --model_type="bert" \
    --num_test_examples=1000 \
    --num_val_examples=100 \
    --test_seed=-1 \
    --val_seed=2

python -m model_selection.compute_model_selection_correlation \
    --results_dir="/scratch/yifwang/fairness_x_explainability/encoder_results_civil" \
    --val_results_dir="/scratch/yifwang/fairness_x_explainability/encoder_results_civil/val_results" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --experiment_type=encoder \
    --train_type=single \
    --bias_type="religion" \
    --model_type="roberta" \
    --num_test_examples=1000 \
    --num_val_examples=100 \
    --test_seed=-1 \
    --val_seed=2

python -m model_selection.compute_model_selection_correlation \
    --results_dir="/scratch/yifwang/fairness_x_explainability/encoder_results_civil" \
    --val_results_dir="/scratch/yifwang/fairness_x_explainability/encoder_results_civil/val_results" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --experiment_type=encoder \
    --train_type=all \
    --bias_type="religion" \
    --model_type="roberta" \
    --num_test_examples=1000 \
    --num_val_examples=100 \
    --test_seed=-1 \
    --val_seed=2

python -m model_selection.compute_model_selection_correlation \
    --results_dir="/scratch/yifwang/fairness_x_explainability/encoder_results_civil" \
    --val_results_dir="/scratch/yifwang/fairness_x_explainability/encoder_results_civil/val_results" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --experiment_type=encoder \
    --train_type=single \
    --bias_type="religion" \
    --model_type="distilbert" \
    --num_test_examples=1000 \
    --num_val_examples=100 \
    --test_seed=-1 \
    --val_seed=2

python -m model_selection.compute_model_selection_correlation \
    --results_dir="/scratch/yifwang/fairness_x_explainability/encoder_results_civil" \
    --val_results_dir="/scratch/yifwang/fairness_x_explainability/encoder_results_civil/val_results" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --experiment_type=encoder \
    --train_type=all \
    --bias_type="religion" \
    --model_type="distilbert" \
    --num_test_examples=1000 \
    --num_val_examples=100 \
    --test_seed=-1 \
    --val_seed=2

python -m model_selection.compute_model_selection_correlation \
    --results_dir="/scratch/yifwang/fairness_x_explainability/encoder_results_civil" \
    --val_results_dir="/scratch/yifwang/fairness_x_explainability/encoder_results_civil/val_results" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --experiment_type=encoder \
    --train_type=single \
    --bias_type="religion" \
    --model_type="bert" \
    --num_test_examples=1000 \
    --num_val_examples=100 \
    --test_seed=-1 \
    --val_seed=3

python -m model_selection.compute_model_selection_correlation \
    --results_dir="/scratch/yifwang/fairness_x_explainability/encoder_results_civil" \
    --val_results_dir="/scratch/yifwang/fairness_x_explainability/encoder_results_civil/val_results" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --experiment_type=encoder \
    --train_type=all \
    --bias_type="religion" \
    --model_type="bert" \
    --num_test_examples=1000 \
    --num_val_examples=100 \
    --test_seed=-1 \
    --val_seed=3

python -m model_selection.compute_model_selection_correlation \
    --results_dir="/scratch/yifwang/fairness_x_explainability/encoder_results_civil" \
    --val_results_dir="/scratch/yifwang/fairness_x_explainability/encoder_results_civil/val_results" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --experiment_type=encoder \
    --train_type=single \
    --bias_type="religion" \
    --model_type="roberta" \
    --num_test_examples=1000 \
    --num_val_examples=100 \
    --test_seed=-1 \
    --val_seed=3

python -m model_selection.compute_model_selection_correlation \
    --results_dir="/scratch/yifwang/fairness_x_explainability/encoder_results_civil" \
    --val_results_dir="/scratch/yifwang/fairness_x_explainability/encoder_results_civil/val_results" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --experiment_type=encoder \
    --train_type=all \
    --bias_type="religion" \
    --model_type="roberta" \
    --num_test_examples=1000 \
    --num_val_examples=100 \
    --test_seed=-1 \
    --val_seed=3

python -m model_selection.compute_model_selection_correlation \
    --results_dir="/scratch/yifwang/fairness_x_explainability/encoder_results_civil" \
    --val_results_dir="/scratch/yifwang/fairness_x_explainability/encoder_results_civil/val_results" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --experiment_type=encoder \
    --train_type=single \
    --bias_type="religion" \
    --model_type="distilbert" \
    --num_test_examples=1000 \
    --num_val_examples=100 \
    --test_seed=-1 \
    --val_seed=3

python -m model_selection.compute_model_selection_correlation \
    --results_dir="/scratch/yifwang/fairness_x_explainability/encoder_results_civil" \
    --val_results_dir="/scratch/yifwang/fairness_x_explainability/encoder_results_civil/val_results" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --experiment_type=encoder \
    --train_type=all \
    --bias_type="religion" \
    --model_type="distilbert" \
    --num_test_examples=1000 \
    --num_val_examples=100 \
    --test_seed=-1 \
    --val_seed=3

python -m model_selection.compute_model_selection_correlation \
    --results_dir="/scratch/yifwang/fairness_x_explainability/encoder_results_civil" \
    --val_results_dir="/scratch/yifwang/fairness_x_explainability/encoder_results_civil/val_results" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --experiment_type=encoder \
    --train_type=single \
    --bias_type="religion" \
    --model_type="bert" \
    --num_test_examples=1000 \
    --num_val_examples=100 \
    --test_seed=-1 \
    --val_seed=4

python -m model_selection.compute_model_selection_correlation \
    --results_dir="/scratch/yifwang/fairness_x_explainability/encoder_results_civil" \
    --val_results_dir="/scratch/yifwang/fairness_x_explainability/encoder_results_civil/val_results" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --experiment_type=encoder \
    --train_type=all \
    --bias_type="religion" \
    --model_type="bert" \
    --num_test_examples=1000 \
    --num_val_examples=100 \
    --test_seed=-1 \
    --val_seed=4

python -m model_selection.compute_model_selection_correlation \
    --results_dir="/scratch/yifwang/fairness_x_explainability/encoder_results_civil" \
    --val_results_dir="/scratch/yifwang/fairness_x_explainability/encoder_results_civil/val_results" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --experiment_type=encoder \
    --train_type=single \
    --bias_type="religion" \
    --model_type="roberta" \
    --num_test_examples=1000 \
    --num_val_examples=100 \
    --test_seed=-1 \
    --val_seed=4

python -m model_selection.compute_model_selection_correlation \
    --results_dir="/scratch/yifwang/fairness_x_explainability/encoder_results_civil" \
    --val_results_dir="/scratch/yifwang/fairness_x_explainability/encoder_results_civil/val_results" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --experiment_type=encoder \
    --train_type=all \
    --bias_type="religion" \
    --model_type="roberta" \
    --num_test_examples=1000 \
    --num_val_examples=100 \
    --test_seed=-1 \
    --val_seed=4

python -m model_selection.compute_model_selection_correlation \
    --results_dir="/scratch/yifwang/fairness_x_explainability/encoder_results_civil" \
    --val_results_dir="/scratch/yifwang/fairness_x_explainability/encoder_results_civil/val_results" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --experiment_type=encoder \
    --train_type=single \
    --bias_type="religion" \
    --model_type="distilbert" \
    --num_test_examples=1000 \
    --num_val_examples=100 \
    --test_seed=-1 \
    --val_seed=4

python -m model_selection.compute_model_selection_correlation \
    --results_dir="/scratch/yifwang/fairness_x_explainability/encoder_results_civil" \
    --val_results_dir="/scratch/yifwang/fairness_x_explainability/encoder_results_civil/val_results" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --experiment_type=encoder \
    --train_type=all \
    --bias_type="religion" \
    --model_type="distilbert" \
    --num_test_examples=1000 \
    --num_val_examples=100 \
    --test_seed=-1 \
    --val_seed=4

python -m model_selection.compute_model_selection_correlation \
    --results_dir="/scratch/yifwang/fairness_x_explainability/encoder_results_civil" \
    --val_results_dir="/scratch/yifwang/fairness_x_explainability/encoder_results_civil/val_results" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --experiment_type=encoder \
    --train_type=single \
    --bias_type="religion" \
    --model_type="bert" \
    --num_test_examples=1000 \
    --num_val_examples=100 \
    --test_seed=-1 \
    --val_seed=5

python -m model_selection.compute_model_selection_correlation \
    --results_dir="/scratch/yifwang/fairness_x_explainability/encoder_results_civil" \
    --val_results_dir="/scratch/yifwang/fairness_x_explainability/encoder_results_civil/val_results" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --experiment_type=encoder \
    --train_type=all \
    --bias_type="religion" \
    --model_type="bert" \
    --num_test_examples=1000 \
    --num_val_examples=100 \
    --test_seed=-1 \
    --val_seed=5

python -m model_selection.compute_model_selection_correlation \
    --results_dir="/scratch/yifwang/fairness_x_explainability/encoder_results_civil" \
    --val_results_dir="/scratch/yifwang/fairness_x_explainability/encoder_results_civil/val_results" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --experiment_type=encoder \
    --train_type=single \
    --bias_type="religion" \
    --model_type="roberta" \
    --num_test_examples=1000 \
    --num_val_examples=100 \
    --test_seed=-1 \
    --val_seed=5

python -m model_selection.compute_model_selection_correlation \
    --results_dir="/scratch/yifwang/fairness_x_explainability/encoder_results_civil" \
    --val_results_dir="/scratch/yifwang/fairness_x_explainability/encoder_results_civil/val_results" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --experiment_type=encoder \
    --train_type=all \
    --bias_type="religion" \
    --model_type="roberta" \
    --num_test_examples=1000 \
    --num_val_examples=100 \
    --test_seed=-1 \
    --val_seed=5

python -m model_selection.compute_model_selection_correlation \
    --results_dir="/scratch/yifwang/fairness_x_explainability/encoder_results_civil" \
    --val_results_dir="/scratch/yifwang/fairness_x_explainability/encoder_results_civil/val_results" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --experiment_type=encoder \
    --train_type=single \
    --bias_type="religion" \
    --model_type="distilbert" \
    --num_test_examples=1000 \
    --num_val_examples=100 \
    --test_seed=-1 \
    --val_seed=5

python -m model_selection.compute_model_selection_correlation \
    --results_dir="/scratch/yifwang/fairness_x_explainability/encoder_results_civil" \
    --val_results_dir="/scratch/yifwang/fairness_x_explainability/encoder_results_civil/val_results" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --experiment_type=encoder \
    --train_type=all \
    --bias_type="religion" \
    --model_type="distilbert" \
    --num_test_examples=1000 \
    --num_val_examples=100 \
    --test_seed=-1 \
    --val_seed=5

python -m model_selection.compute_model_selection_correlation \
    --results_dir="/scratch/yifwang/fairness_x_explainability/encoder_results_civil" \
    --val_results_dir="/scratch/yifwang/fairness_x_explainability/encoder_results_civil/val_results" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --experiment_type=encoder \
    --train_type=single \
    --bias_type="religion" \
    --model_type="bert" \
    --num_test_examples=1000 \
    --num_val_examples=100 \
    --test_seed=-1 \
    --val_seed=42

python -m model_selection.compute_model_selection_correlation \
    --results_dir="/scratch/yifwang/fairness_x_explainability/encoder_results_civil" \
    --val_results_dir="/scratch/yifwang/fairness_x_explainability/encoder_results_civil/val_results" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --experiment_type=encoder \
    --train_type=all \
    --bias_type="religion" \
    --model_type="bert" \
    --num_test_examples=1000 \
    --num_val_examples=100 \
    --test_seed=-1 \
    --val_seed=42

python -m model_selection.compute_model_selection_correlation \
    --results_dir="/scratch/yifwang/fairness_x_explainability/encoder_results_civil" \
    --val_results_dir="/scratch/yifwang/fairness_x_explainability/encoder_results_civil/val_results" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --experiment_type=encoder \
    --train_type=single \
    --bias_type="religion" \
    --model_type="roberta" \
    --num_test_examples=1000 \
    --num_val_examples=100 \
    --test_seed=-1 \
    --val_seed=42

python -m model_selection.compute_model_selection_correlation \
    --results_dir="/scratch/yifwang/fairness_x_explainability/encoder_results_civil" \
    --val_results_dir="/scratch/yifwang/fairness_x_explainability/encoder_results_civil/val_results" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --experiment_type=encoder \
    --train_type=all \
    --bias_type="religion" \
    --model_type="roberta" \
    --num_test_examples=1000 \
    --num_val_examples=100 \
    --test_seed=-1 \
    --val_seed=42

python -m model_selection.compute_model_selection_correlation \
    --results_dir="/scratch/yifwang/fairness_x_explainability/encoder_results_civil" \
    --val_results_dir="/scratch/yifwang/fairness_x_explainability/encoder_results_civil/val_results" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --experiment_type=encoder \
    --train_type=single \
    --bias_type="religion" \
    --model_type="distilbert" \
    --num_test_examples=1000 \
    --num_val_examples=100 \
    --test_seed=-1 \
    --val_seed=42

python -m model_selection.compute_model_selection_correlation \
    --results_dir="/scratch/yifwang/fairness_x_explainability/encoder_results_civil" \
    --val_results_dir="/scratch/yifwang/fairness_x_explainability/encoder_results_civil/val_results" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --experiment_type=encoder \
    --train_type=all \
    --bias_type="religion" \
    --model_type="distilbert" \
    --num_test_examples=1000 \
    --num_val_examples=100 \
    --test_seed=-1 \
    --val_seed=42

python -m model_selection.compute_model_selection_correlation \
    --results_dir="/scratch/yifwang/fairness_x_explainability/encoder_results_civil" \
    --val_results_dir="/scratch/yifwang/fairness_x_explainability/encoder_results_civil/val_results" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --experiment_type=encoder \
    --train_type=single \
    --bias_type="religion" \
    --model_type="bert" \
    --num_test_examples=1000 \
    --num_val_examples=200 \
    --test_seed=-1 \
    --val_seed=1

python -m model_selection.compute_model_selection_correlation \
    --results_dir="/scratch/yifwang/fairness_x_explainability/encoder_results_civil" \
    --val_results_dir="/scratch/yifwang/fairness_x_explainability/encoder_results_civil/val_results" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --experiment_type=encoder \
    --train_type=all \
    --bias_type="religion" \
    --model_type="bert" \
    --num_test_examples=1000 \
    --num_val_examples=200 \
    --test_seed=-1 \
    --val_seed=1

python -m model_selection.compute_model_selection_correlation \
    --results_dir="/scratch/yifwang/fairness_x_explainability/encoder_results_civil" \
    --val_results_dir="/scratch/yifwang/fairness_x_explainability/encoder_results_civil/val_results" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --experiment_type=encoder \
    --train_type=single \
    --bias_type="religion" \
    --model_type="roberta" \
    --num_test_examples=1000 \
    --num_val_examples=200 \
    --test_seed=-1 \
    --val_seed=1

python -m model_selection.compute_model_selection_correlation \
    --results_dir="/scratch/yifwang/fairness_x_explainability/encoder_results_civil" \
    --val_results_dir="/scratch/yifwang/fairness_x_explainability/encoder_results_civil/val_results" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --experiment_type=encoder \
    --train_type=all \
    --bias_type="religion" \
    --model_type="roberta" \
    --num_test_examples=1000 \
    --num_val_examples=200 \
    --test_seed=-1 \
    --val_seed=1

python -m model_selection.compute_model_selection_correlation \
    --results_dir="/scratch/yifwang/fairness_x_explainability/encoder_results_civil" \
    --val_results_dir="/scratch/yifwang/fairness_x_explainability/encoder_results_civil/val_results" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --experiment_type=encoder \
    --train_type=single \
    --bias_type="religion" \
    --model_type="distilbert" \
    --num_test_examples=1000 \
    --num_val_examples=200 \
    --test_seed=-1 \
    --val_seed=1

python -m model_selection.compute_model_selection_correlation \
    --results_dir="/scratch/yifwang/fairness_x_explainability/encoder_results_civil" \
    --val_results_dir="/scratch/yifwang/fairness_x_explainability/encoder_results_civil/val_results" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --experiment_type=encoder \
    --train_type=all \
    --bias_type="religion" \
    --model_type="distilbert" \
    --num_test_examples=1000 \
    --num_val_examples=200 \
    --test_seed=-1 \
    --val_seed=1

python -m model_selection.compute_model_selection_correlation \
    --results_dir="/scratch/yifwang/fairness_x_explainability/encoder_results_civil" \
    --val_results_dir="/scratch/yifwang/fairness_x_explainability/encoder_results_civil/val_results" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --experiment_type=encoder \
    --train_type=single \
    --bias_type="religion" \
    --model_type="bert" \
    --num_test_examples=1000 \
    --num_val_examples=200 \
    --test_seed=-1 \
    --val_seed=2

python -m model_selection.compute_model_selection_correlation \
    --results_dir="/scratch/yifwang/fairness_x_explainability/encoder_results_civil" \
    --val_results_dir="/scratch/yifwang/fairness_x_explainability/encoder_results_civil/val_results" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --experiment_type=encoder \
    --train_type=all \
    --bias_type="religion" \
    --model_type="bert" \
    --num_test_examples=1000 \
    --num_val_examples=200 \
    --test_seed=-1 \
    --val_seed=2

python -m model_selection.compute_model_selection_correlation \
    --results_dir="/scratch/yifwang/fairness_x_explainability/encoder_results_civil" \
    --val_results_dir="/scratch/yifwang/fairness_x_explainability/encoder_results_civil/val_results" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --experiment_type=encoder \
    --train_type=single \
    --bias_type="religion" \
    --model_type="roberta" \
    --num_test_examples=1000 \
    --num_val_examples=200 \
    --test_seed=-1 \
    --val_seed=2

python -m model_selection.compute_model_selection_correlation \
    --results_dir="/scratch/yifwang/fairness_x_explainability/encoder_results_civil" \
    --val_results_dir="/scratch/yifwang/fairness_x_explainability/encoder_results_civil/val_results" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --experiment_type=encoder \
    --train_type=all \
    --bias_type="religion" \
    --model_type="roberta" \
    --num_test_examples=1000 \
    --num_val_examples=200 \
    --test_seed=-1 \
    --val_seed=2

python -m model_selection.compute_model_selection_correlation \
    --results_dir="/scratch/yifwang/fairness_x_explainability/encoder_results_civil" \
    --val_results_dir="/scratch/yifwang/fairness_x_explainability/encoder_results_civil/val_results" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --experiment_type=encoder \
    --train_type=single \
    --bias_type="religion" \
    --model_type="distilbert" \
    --num_test_examples=1000 \
    --num_val_examples=200 \
    --test_seed=-1 \
    --val_seed=2

python -m model_selection.compute_model_selection_correlation \
    --results_dir="/scratch/yifwang/fairness_x_explainability/encoder_results_civil" \
    --val_results_dir="/scratch/yifwang/fairness_x_explainability/encoder_results_civil/val_results" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --experiment_type=encoder \
    --train_type=all \
    --bias_type="religion" \
    --model_type="distilbert" \
    --num_test_examples=1000 \
    --num_val_examples=200 \
    --test_seed=-1 \
    --val_seed=2

python -m model_selection.compute_model_selection_correlation \
    --results_dir="/scratch/yifwang/fairness_x_explainability/encoder_results_civil" \
    --val_results_dir="/scratch/yifwang/fairness_x_explainability/encoder_results_civil/val_results" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --experiment_type=encoder \
    --train_type=single \
    --bias_type="religion" \
    --model_type="bert" \
    --num_test_examples=1000 \
    --num_val_examples=200 \
    --test_seed=-1 \
    --val_seed=3

python -m model_selection.compute_model_selection_correlation \
    --results_dir="/scratch/yifwang/fairness_x_explainability/encoder_results_civil" \
    --val_results_dir="/scratch/yifwang/fairness_x_explainability/encoder_results_civil/val_results" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --experiment_type=encoder \
    --train_type=all \
    --bias_type="religion" \
    --model_type="bert" \
    --num_test_examples=1000 \
    --num_val_examples=200 \
    --test_seed=-1 \
    --val_seed=3

python -m model_selection.compute_model_selection_correlation \
    --results_dir="/scratch/yifwang/fairness_x_explainability/encoder_results_civil" \
    --val_results_dir="/scratch/yifwang/fairness_x_explainability/encoder_results_civil/val_results" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --experiment_type=encoder \
    --train_type=single \
    --bias_type="religion" \
    --model_type="roberta" \
    --num_test_examples=1000 \
    --num_val_examples=200 \
    --test_seed=-1 \
    --val_seed=3

python -m model_selection.compute_model_selection_correlation \
    --results_dir="/scratch/yifwang/fairness_x_explainability/encoder_results_civil" \
    --val_results_dir="/scratch/yifwang/fairness_x_explainability/encoder_results_civil/val_results" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --experiment_type=encoder \
    --train_type=all \
    --bias_type="religion" \
    --model_type="roberta" \
    --num_test_examples=1000 \
    --num_val_examples=200 \
    --test_seed=-1 \
    --val_seed=3

python -m model_selection.compute_model_selection_correlation \
    --results_dir="/scratch/yifwang/fairness_x_explainability/encoder_results_civil" \
    --val_results_dir="/scratch/yifwang/fairness_x_explainability/encoder_results_civil/val_results" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --experiment_type=encoder \
    --train_type=single \
    --bias_type="religion" \
    --model_type="distilbert" \
    --num_test_examples=1000 \
    --num_val_examples=200 \
    --test_seed=-1 \
    --val_seed=3

python -m model_selection.compute_model_selection_correlation \
    --results_dir="/scratch/yifwang/fairness_x_explainability/encoder_results_civil" \
    --val_results_dir="/scratch/yifwang/fairness_x_explainability/encoder_results_civil/val_results" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --experiment_type=encoder \
    --train_type=all \
    --bias_type="religion" \
    --model_type="distilbert" \
    --num_test_examples=1000 \
    --num_val_examples=200 \
    --test_seed=-1 \
    --val_seed=3

python -m model_selection.compute_model_selection_correlation \
    --results_dir="/scratch/yifwang/fairness_x_explainability/encoder_results_civil" \
    --val_results_dir="/scratch/yifwang/fairness_x_explainability/encoder_results_civil/val_results" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --experiment_type=encoder \
    --train_type=single \
    --bias_type="religion" \
    --model_type="bert" \
    --num_test_examples=1000 \
    --num_val_examples=200 \
    --test_seed=-1 \
    --val_seed=4

python -m model_selection.compute_model_selection_correlation \
    --results_dir="/scratch/yifwang/fairness_x_explainability/encoder_results_civil" \
    --val_results_dir="/scratch/yifwang/fairness_x_explainability/encoder_results_civil/val_results" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --experiment_type=encoder \
    --train_type=all \
    --bias_type="religion" \
    --model_type="bert" \
    --num_test_examples=1000 \
    --num_val_examples=200 \
    --test_seed=-1 \
    --val_seed=4

python -m model_selection.compute_model_selection_correlation \
    --results_dir="/scratch/yifwang/fairness_x_explainability/encoder_results_civil" \
    --val_results_dir="/scratch/yifwang/fairness_x_explainability/encoder_results_civil/val_results" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --experiment_type=encoder \
    --train_type=single \
    --bias_type="religion" \
    --model_type="roberta" \
    --num_test_examples=1000 \
    --num_val_examples=200 \
    --test_seed=-1 \
    --val_seed=4

python -m model_selection.compute_model_selection_correlation \
    --results_dir="/scratch/yifwang/fairness_x_explainability/encoder_results_civil" \
    --val_results_dir="/scratch/yifwang/fairness_x_explainability/encoder_results_civil/val_results" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --experiment_type=encoder \
    --train_type=all \
    --bias_type="religion" \
    --model_type="roberta" \
    --num_test_examples=1000 \
    --num_val_examples=200 \
    --test_seed=-1 \
    --val_seed=4

python -m model_selection.compute_model_selection_correlation \
    --results_dir="/scratch/yifwang/fairness_x_explainability/encoder_results_civil" \
    --val_results_dir="/scratch/yifwang/fairness_x_explainability/encoder_results_civil/val_results" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --experiment_type=encoder \
    --train_type=single \
    --bias_type="religion" \
    --model_type="distilbert" \
    --num_test_examples=1000 \
    --num_val_examples=200 \
    --test_seed=-1 \
    --val_seed=4

python -m model_selection.compute_model_selection_correlation \
    --results_dir="/scratch/yifwang/fairness_x_explainability/encoder_results_civil" \
    --val_results_dir="/scratch/yifwang/fairness_x_explainability/encoder_results_civil/val_results" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --experiment_type=encoder \
    --train_type=all \
    --bias_type="religion" \
    --model_type="distilbert" \
    --num_test_examples=1000 \
    --num_val_examples=200 \
    --test_seed=-1 \
    --val_seed=4

python -m model_selection.compute_model_selection_correlation \
    --results_dir="/scratch/yifwang/fairness_x_explainability/encoder_results_civil" \
    --val_results_dir="/scratch/yifwang/fairness_x_explainability/encoder_results_civil/val_results" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --experiment_type=encoder \
    --train_type=single \
    --bias_type="religion" \
    --model_type="bert" \
    --num_test_examples=1000 \
    --num_val_examples=200 \
    --test_seed=-1 \
    --val_seed=5

python -m model_selection.compute_model_selection_correlation \
    --results_dir="/scratch/yifwang/fairness_x_explainability/encoder_results_civil" \
    --val_results_dir="/scratch/yifwang/fairness_x_explainability/encoder_results_civil/val_results" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --experiment_type=encoder \
    --train_type=all \
    --bias_type="religion" \
    --model_type="bert" \
    --num_test_examples=1000 \
    --num_val_examples=200 \
    --test_seed=-1 \
    --val_seed=5

python -m model_selection.compute_model_selection_correlation \
    --results_dir="/scratch/yifwang/fairness_x_explainability/encoder_results_civil" \
    --val_results_dir="/scratch/yifwang/fairness_x_explainability/encoder_results_civil/val_results" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --experiment_type=encoder \
    --train_type=single \
    --bias_type="religion" \
    --model_type="roberta" \
    --num_test_examples=1000 \
    --num_val_examples=200 \
    --test_seed=-1 \
    --val_seed=5

python -m model_selection.compute_model_selection_correlation \
    --results_dir="/scratch/yifwang/fairness_x_explainability/encoder_results_civil" \
    --val_results_dir="/scratch/yifwang/fairness_x_explainability/encoder_results_civil/val_results" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --experiment_type=encoder \
    --train_type=all \
    --bias_type="religion" \
    --model_type="roberta" \
    --num_test_examples=1000 \
    --num_val_examples=200 \
    --test_seed=-1 \
    --val_seed=5

python -m model_selection.compute_model_selection_correlation \
    --results_dir="/scratch/yifwang/fairness_x_explainability/encoder_results_civil" \
    --val_results_dir="/scratch/yifwang/fairness_x_explainability/encoder_results_civil/val_results" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --experiment_type=encoder \
    --train_type=single \
    --bias_type="religion" \
    --model_type="distilbert" \
    --num_test_examples=1000 \
    --num_val_examples=200 \
    --test_seed=-1 \
    --val_seed=5

python -m model_selection.compute_model_selection_correlation \
    --results_dir="/scratch/yifwang/fairness_x_explainability/encoder_results_civil" \
    --val_results_dir="/scratch/yifwang/fairness_x_explainability/encoder_results_civil/val_results" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --experiment_type=encoder \
    --train_type=all \
    --bias_type="religion" \
    --model_type="distilbert" \
    --num_test_examples=1000 \
    --num_val_examples=200 \
    --test_seed=-1 \
    --val_seed=5

python -m model_selection.compute_model_selection_correlation \
    --results_dir="/scratch/yifwang/fairness_x_explainability/encoder_results_civil" \
    --val_results_dir="/scratch/yifwang/fairness_x_explainability/encoder_results_civil/val_results" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --experiment_type=encoder \
    --train_type=single \
    --bias_type="religion" \
    --model_type="bert" \
    --num_test_examples=1000 \
    --num_val_examples=200 \
    --test_seed=-1 \
    --val_seed=42

python -m model_selection.compute_model_selection_correlation \
    --results_dir="/scratch/yifwang/fairness_x_explainability/encoder_results_civil" \
    --val_results_dir="/scratch/yifwang/fairness_x_explainability/encoder_results_civil/val_results" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --experiment_type=encoder \
    --train_type=all \
    --bias_type="religion" \
    --model_type="bert" \
    --num_test_examples=1000 \
    --num_val_examples=200 \
    --test_seed=-1 \
    --val_seed=42

python -m model_selection.compute_model_selection_correlation \
    --results_dir="/scratch/yifwang/fairness_x_explainability/encoder_results_civil" \
    --val_results_dir="/scratch/yifwang/fairness_x_explainability/encoder_results_civil/val_results" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --experiment_type=encoder \
    --train_type=single \
    --bias_type="religion" \
    --model_type="roberta" \
    --num_test_examples=1000 \
    --num_val_examples=200 \
    --test_seed=-1 \
    --val_seed=42

python -m model_selection.compute_model_selection_correlation \
    --results_dir="/scratch/yifwang/fairness_x_explainability/encoder_results_civil" \
    --val_results_dir="/scratch/yifwang/fairness_x_explainability/encoder_results_civil/val_results" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --experiment_type=encoder \
    --train_type=all \
    --bias_type="religion" \
    --model_type="roberta" \
    --num_test_examples=1000 \
    --num_val_examples=200 \
    --test_seed=-1 \
    --val_seed=42

python -m model_selection.compute_model_selection_correlation \
    --results_dir="/scratch/yifwang/fairness_x_explainability/encoder_results_civil" \
    --val_results_dir="/scratch/yifwang/fairness_x_explainability/encoder_results_civil/val_results" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --experiment_type=encoder \
    --train_type=single \
    --bias_type="religion" \
    --model_type="distilbert" \
    --num_test_examples=1000 \
    --num_val_examples=200 \
    --test_seed=-1 \
    --val_seed=42

python -m model_selection.compute_model_selection_correlation \
    --results_dir="/scratch/yifwang/fairness_x_explainability/encoder_results_civil" \
    --val_results_dir="/scratch/yifwang/fairness_x_explainability/encoder_results_civil/val_results" \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --experiment_type=encoder \
    --train_type=all \
    --bias_type="religion" \
    --model_type="distilbert" \
    --num_test_examples=1000 \
    --num_val_examples=200 \
    --test_seed=-1 \
    --val_seed=42


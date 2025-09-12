#!/bin/bash

python -m explanation_generation.gen_explanation_encoder \
    --dataset_name="google/jigsaw_unintended_bias" \
    --split="test" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models_jigsaw/bert_jigsaw_religion/no_debiasing" \
    --num_labels=2 \
    --batch_size=8 \
    --max_length=512 \
    --num_examples=200 \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results_jigsaw/bert_jigsaw_religion_religion_test_200/no_debiasing" \
    --baseline="pad" \
    --bias_type="religion" \
    --seed=42 \
    --only_predicted_class

python -m explanation_generation.gen_explanation_encoder \
    --dataset_name="google/jigsaw_unintended_bias" \
    --split="test" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models_jigsaw/bert_jigsaw_all/no_debiasing" \
    --num_labels=2 \
    --batch_size=8 \
    --max_length=512 \
    --num_examples=200 \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results_jigsaw/bert_jigsaw_all_religion_test_200/no_debiasing" \
    --baseline="pad" \
    --bias_type="religion" \
    --seed=42 \
    --only_predicted_class

python -m explanation_generation.gen_explanation_encoder \
    --dataset_name="google/jigsaw_unintended_bias" \
    --split="test" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models_jigsaw/bert_jigsaw_religion/group_balance" \
    --num_labels=2 \
    --batch_size=8 \
    --max_length=512 \
    --num_examples=200 \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results_jigsaw/bert_jigsaw_religion_religion_test_200/group_balance" \
    --baseline="pad" \
    --bias_type="religion" \
    --seed=42 \
    --only_predicted_class

python -m explanation_generation.gen_explanation_encoder \
    --dataset_name="google/jigsaw_unintended_bias" \
    --split="test" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models_jigsaw/bert_jigsaw_all/group_balance" \
    --num_labels=2 \
    --batch_size=8 \
    --max_length=512 \
    --num_examples=200 \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results_jigsaw/bert_jigsaw_all_religion_test_200/group_balance" \
    --baseline="pad" \
    --bias_type="religion" \
    --seed=42 \
    --only_predicted_class

python -m explanation_generation.gen_explanation_encoder \
    --dataset_name="google/jigsaw_unintended_bias" \
    --split="test" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models_jigsaw/bert_jigsaw_religion/group_class_balance" \
    --num_labels=2 \
    --batch_size=8 \
    --max_length=512 \
    --num_examples=200 \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results_jigsaw/bert_jigsaw_religion_religion_test_200/group_class_balance" \
    --baseline="pad" \
    --bias_type="religion" \
    --seed=42 \
    --only_predicted_class

python -m explanation_generation.gen_explanation_encoder \
    --dataset_name="google/jigsaw_unintended_bias" \
    --split="test" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models_jigsaw/bert_jigsaw_all/group_class_balance" \
    --num_labels=2 \
    --batch_size=8 \
    --max_length=512 \
    --num_examples=200 \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results_jigsaw/bert_jigsaw_all_religion_test_200/group_class_balance" \
    --baseline="pad" \
    --bias_type="religion" \
    --seed=42 \
    --only_predicted_class

python -m explanation_generation.gen_explanation_encoder \
    --dataset_name="google/jigsaw_unintended_bias" \
    --split="test" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models_jigsaw/bert_jigsaw_religion/cda" \
    --num_labels=2 \
    --batch_size=8 \
    --max_length=512 \
    --num_examples=200 \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results_jigsaw/bert_jigsaw_religion_religion_test_200/cda" \
    --baseline="pad" \
    --bias_type="religion" \
    --seed=42 \
    --only_predicted_class

python -m explanation_generation.gen_explanation_encoder \
    --dataset_name="google/jigsaw_unintended_bias" \
    --split="test" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models_jigsaw/bert_jigsaw_all/cda" \
    --num_labels=2 \
    --batch_size=8 \
    --max_length=512 \
    --num_examples=200 \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results_jigsaw/bert_jigsaw_all_religion_test_200/cda" \
    --baseline="pad" \
    --bias_type="religion" \
    --seed=42 \
    --only_predicted_class

python -m explanation_generation.gen_explanation_encoder \
    --dataset_name="google/jigsaw_unintended_bias" \
    --split="test" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models_jigsaw/bert_jigsaw_religion/dropout" \
    --num_labels=2 \
    --batch_size=8 \
    --max_length=512 \
    --num_examples=200 \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results_jigsaw/bert_jigsaw_religion_religion_test_200/dropout" \
    --baseline="pad" \
    --bias_type="religion" \
    --seed=42 \
    --only_predicted_class

python -m explanation_generation.gen_explanation_encoder \
    --dataset_name="google/jigsaw_unintended_bias" \
    --split="test" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models_jigsaw/bert_jigsaw_all/dropout" \
    --num_labels=2 \
    --batch_size=8 \
    --max_length=512 \
    --num_examples=200 \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results_jigsaw/bert_jigsaw_all_religion_test_200/dropout" \
    --baseline="pad" \
    --bias_type="religion" \
    --seed=42 \
    --only_predicted_class

python -m explanation_generation.gen_explanation_encoder \
    --dataset_name="google/jigsaw_unintended_bias" \
    --split="test" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models_jigsaw/bert_jigsaw_religion/attention_entropy" \
    --num_labels=2 \
    --batch_size=8 \
    --max_length=512 \
    --num_examples=200 \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results_jigsaw/bert_jigsaw_religion_religion_test_200/attention_entropy" \
    --baseline="pad" \
    --bias_type="religion" \
    --seed=42 \
    --only_predicted_class

python -m explanation_generation.gen_explanation_encoder \
    --dataset_name="google/jigsaw_unintended_bias" \
    --split="test" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models_jigsaw/bert_jigsaw_all/attention_entropy" \
    --num_labels=2 \
    --batch_size=8 \
    --max_length=512 \
    --num_examples=200 \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results_jigsaw/bert_jigsaw_all_religion_test_200/attention_entropy" \
    --baseline="pad" \
    --bias_type="religion" \
    --seed=42 \
    --only_predicted_class

python -m explanation_generation.gen_explanation_encoder \
    --dataset_name="google/jigsaw_unintended_bias" \
    --split="test" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models_jigsaw/bert_jigsaw_religion/causal_debias" \
    --num_labels=2 \
    --batch_size=8 \
    --max_length=512 \
    --num_examples=200 \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results_jigsaw/bert_jigsaw_religion_religion_test_200/causal_debias" \
    --baseline="pad" \
    --bias_type="religion" \
    --seed=42 \
    --only_predicted_class

python -m explanation_generation.gen_explanation_encoder \
    --dataset_name="google/jigsaw_unintended_bias" \
    --split="test" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models_jigsaw/bert_jigsaw_all/causal_debias" \
    --num_labels=2 \
    --batch_size=8 \
    --max_length=512 \
    --num_examples=200 \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results_jigsaw/bert_jigsaw_all_religion_test_200/causal_debias" \
    --baseline="pad" \
    --bias_type="religion" \
    --seed=42 \
    --only_predicted_class

python -m explanation_generation.gen_explanation_encoder \
    --dataset_name="google/jigsaw_unintended_bias" \
    --split="test" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models_jigsaw/roberta_jigsaw_religion/no_debiasing" \
    --num_labels=2 \
    --batch_size=8 \
    --max_length=512 \
    --num_examples=200 \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results_jigsaw/roberta_jigsaw_religion_religion_test_200/no_debiasing" \
    --baseline="pad" \
    --bias_type="religion" \
    --seed=42 \
    --only_predicted_class

python -m explanation_generation.gen_explanation_encoder \
    --dataset_name="google/jigsaw_unintended_bias" \
    --split="test" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models_jigsaw/roberta_jigsaw_all/no_debiasing" \
    --num_labels=2 \
    --batch_size=8 \
    --max_length=512 \
    --num_examples=200 \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results_jigsaw/roberta_jigsaw_all_religion_test_200/no_debiasing" \
    --baseline="pad" \
    --bias_type="religion" \
    --seed=42 \
    --only_predicted_class

python -m explanation_generation.gen_explanation_encoder \
    --dataset_name="google/jigsaw_unintended_bias" \
    --split="test" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models_jigsaw/roberta_jigsaw_religion/group_balance" \
    --num_labels=2 \
    --batch_size=8 \
    --max_length=512 \
    --num_examples=200 \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results_jigsaw/roberta_jigsaw_religion_religion_test_200/group_balance" \
    --baseline="pad" \
    --bias_type="religion" \
    --seed=42 \
    --only_predicted_class

python -m explanation_generation.gen_explanation_encoder \
    --dataset_name="google/jigsaw_unintended_bias" \
    --split="test" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models_jigsaw/roberta_jigsaw_all/group_balance" \
    --num_labels=2 \
    --batch_size=8 \
    --max_length=512 \
    --num_examples=200 \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results_jigsaw/roberta_jigsaw_all_religion_test_200/group_balance" \
    --baseline="pad" \
    --bias_type="religion" \
    --seed=42 \
    --only_predicted_class

python -m explanation_generation.gen_explanation_encoder \
    --dataset_name="google/jigsaw_unintended_bias" \
    --split="test" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models_jigsaw/roberta_jigsaw_religion/group_class_balance" \
    --num_labels=2 \
    --batch_size=8 \
    --max_length=512 \
    --num_examples=200 \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results_jigsaw/roberta_jigsaw_religion_religion_test_200/group_class_balance" \
    --baseline="pad" \
    --bias_type="religion" \
    --seed=42 \
    --only_predicted_class

python -m explanation_generation.gen_explanation_encoder \
    --dataset_name="google/jigsaw_unintended_bias" \
    --split="test" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models_jigsaw/roberta_jigsaw_all/group_class_balance" \
    --num_labels=2 \
    --batch_size=8 \
    --max_length=512 \
    --num_examples=200 \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results_jigsaw/roberta_jigsaw_all_religion_test_200/group_class_balance" \
    --baseline="pad" \
    --bias_type="religion" \
    --seed=42 \
    --only_predicted_class

python -m explanation_generation.gen_explanation_encoder \
    --dataset_name="google/jigsaw_unintended_bias" \
    --split="test" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models_jigsaw/roberta_jigsaw_religion/cda" \
    --num_labels=2 \
    --batch_size=8 \
    --max_length=512 \
    --num_examples=200 \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results_jigsaw/roberta_jigsaw_religion_religion_test_200/cda" \
    --baseline="pad" \
    --bias_type="religion" \
    --seed=42 \
    --only_predicted_class

python -m explanation_generation.gen_explanation_encoder \
    --dataset_name="google/jigsaw_unintended_bias" \
    --split="test" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models_jigsaw/roberta_jigsaw_all/cda" \
    --num_labels=2 \
    --batch_size=8 \
    --max_length=512 \
    --num_examples=200 \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results_jigsaw/roberta_jigsaw_all_religion_test_200/cda" \
    --baseline="pad" \
    --bias_type="religion" \
    --seed=42 \
    --only_predicted_class

python -m explanation_generation.gen_explanation_encoder \
    --dataset_name="google/jigsaw_unintended_bias" \
    --split="test" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models_jigsaw/roberta_jigsaw_religion/dropout" \
    --num_labels=2 \
    --batch_size=8 \
    --max_length=512 \
    --num_examples=200 \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results_jigsaw/roberta_jigsaw_religion_religion_test_200/dropout" \
    --baseline="pad" \
    --bias_type="religion" \
    --seed=42 \
    --only_predicted_class

python -m explanation_generation.gen_explanation_encoder \
    --dataset_name="google/jigsaw_unintended_bias" \
    --split="test" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models_jigsaw/roberta_jigsaw_all/dropout" \
    --num_labels=2 \
    --batch_size=8 \
    --max_length=512 \
    --num_examples=200 \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results_jigsaw/roberta_jigsaw_all_religion_test_200/dropout" \
    --baseline="pad" \
    --bias_type="religion" \
    --seed=42 \
    --only_predicted_class

python -m explanation_generation.gen_explanation_encoder \
    --dataset_name="google/jigsaw_unintended_bias" \
    --split="test" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models_jigsaw/roberta_jigsaw_religion/attention_entropy" \
    --num_labels=2 \
    --batch_size=8 \
    --max_length=512 \
    --num_examples=200 \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results_jigsaw/roberta_jigsaw_religion_religion_test_200/attention_entropy" \
    --baseline="pad" \
    --bias_type="religion" \
    --seed=42 \
    --only_predicted_class

python -m explanation_generation.gen_explanation_encoder \
    --dataset_name="google/jigsaw_unintended_bias" \
    --split="test" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models_jigsaw/roberta_jigsaw_all/attention_entropy" \
    --num_labels=2 \
    --batch_size=8 \
    --max_length=512 \
    --num_examples=200 \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results_jigsaw/roberta_jigsaw_all_religion_test_200/attention_entropy" \
    --baseline="pad" \
    --bias_type="religion" \
    --seed=42 \
    --only_predicted_class

python -m explanation_generation.gen_explanation_encoder \
    --dataset_name="google/jigsaw_unintended_bias" \
    --split="test" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models_jigsaw/roberta_jigsaw_religion/causal_debias" \
    --num_labels=2 \
    --batch_size=8 \
    --max_length=512 \
    --num_examples=200 \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results_jigsaw/roberta_jigsaw_religion_religion_test_200/causal_debias" \
    --baseline="pad" \
    --bias_type="religion" \
    --seed=42 \
    --only_predicted_class

python -m explanation_generation.gen_explanation_encoder \
    --dataset_name="google/jigsaw_unintended_bias" \
    --split="test" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models_jigsaw/roberta_jigsaw_all/causal_debias" \
    --num_labels=2 \
    --batch_size=8 \
    --max_length=512 \
    --num_examples=200 \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results_jigsaw/roberta_jigsaw_all_religion_test_200/causal_debias" \
    --baseline="pad" \
    --bias_type="religion" \
    --seed=42 \
    --only_predicted_class

python -m explanation_generation.gen_explanation_encoder \
    --dataset_name="google/jigsaw_unintended_bias" \
    --split="test" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models_jigsaw/distilbert_jigsaw_religion/no_debiasing" \
    --num_labels=2 \
    --batch_size=8 \
    --max_length=512 \
    --num_examples=200 \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results_jigsaw/distilbert_jigsaw_religion_religion_test_200/no_debiasing" \
    --baseline="pad" \
    --bias_type="religion" \
    --seed=42 \
    --only_predicted_class

python -m explanation_generation.gen_explanation_encoder \
    --dataset_name="google/jigsaw_unintended_bias" \
    --split="test" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models_jigsaw/distilbert_jigsaw_all/no_debiasing" \
    --num_labels=2 \
    --batch_size=8 \
    --max_length=512 \
    --num_examples=200 \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results_jigsaw/distilbert_jigsaw_all_religion_test_200/no_debiasing" \
    --baseline="pad" \
    --bias_type="religion" \
    --seed=42 \
    --only_predicted_class

python -m explanation_generation.gen_explanation_encoder \
    --dataset_name="google/jigsaw_unintended_bias" \
    --split="test" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models_jigsaw/distilbert_jigsaw_religion/group_balance" \
    --num_labels=2 \
    --batch_size=8 \
    --max_length=512 \
    --num_examples=200 \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results_jigsaw/distilbert_jigsaw_religion_religion_test_200/group_balance" \
    --baseline="pad" \
    --bias_type="religion" \
    --seed=42 \
    --only_predicted_class

python -m explanation_generation.gen_explanation_encoder \
    --dataset_name="google/jigsaw_unintended_bias" \
    --split="test" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models_jigsaw/distilbert_jigsaw_all/group_balance" \
    --num_labels=2 \
    --batch_size=8 \
    --max_length=512 \
    --num_examples=200 \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results_jigsaw/distilbert_jigsaw_all_religion_test_200/group_balance" \
    --baseline="pad" \
    --bias_type="religion" \
    --seed=42 \
    --only_predicted_class

python -m explanation_generation.gen_explanation_encoder \
    --dataset_name="google/jigsaw_unintended_bias" \
    --split="test" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models_jigsaw/distilbert_jigsaw_religion/group_class_balance" \
    --num_labels=2 \
    --batch_size=8 \
    --max_length=512 \
    --num_examples=200 \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results_jigsaw/distilbert_jigsaw_religion_religion_test_200/group_class_balance" \
    --baseline="pad" \
    --bias_type="religion" \
    --seed=42 \
    --only_predicted_class

python -m explanation_generation.gen_explanation_encoder \
    --dataset_name="google/jigsaw_unintended_bias" \
    --split="test" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models_jigsaw/distilbert_jigsaw_all/group_class_balance" \
    --num_labels=2 \
    --batch_size=8 \
    --max_length=512 \
    --num_examples=200 \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results_jigsaw/distilbert_jigsaw_all_religion_test_200/group_class_balance" \
    --baseline="pad" \
    --bias_type="religion" \
    --seed=42 \
    --only_predicted_class

python -m explanation_generation.gen_explanation_encoder \
    --dataset_name="google/jigsaw_unintended_bias" \
    --split="test" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models_jigsaw/distilbert_jigsaw_religion/cda" \
    --num_labels=2 \
    --batch_size=8 \
    --max_length=512 \
    --num_examples=200 \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results_jigsaw/distilbert_jigsaw_religion_religion_test_200/cda" \
    --baseline="pad" \
    --bias_type="religion" \
    --seed=42 \
    --only_predicted_class

python -m explanation_generation.gen_explanation_encoder \
    --dataset_name="google/jigsaw_unintended_bias" \
    --split="test" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models_jigsaw/distilbert_jigsaw_all/cda" \
    --num_labels=2 \
    --batch_size=8 \
    --max_length=512 \
    --num_examples=200 \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results_jigsaw/distilbert_jigsaw_all_religion_test_200/cda" \
    --baseline="pad" \
    --bias_type="religion" \
    --seed=42 \
    --only_predicted_class

python -m explanation_generation.gen_explanation_encoder \
    --dataset_name="google/jigsaw_unintended_bias" \
    --split="test" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models_jigsaw/distilbert_jigsaw_religion/dropout" \
    --num_labels=2 \
    --batch_size=8 \
    --max_length=512 \
    --num_examples=200 \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results_jigsaw/distilbert_jigsaw_religion_religion_test_200/dropout" \
    --baseline="pad" \
    --bias_type="religion" \
    --seed=42 \
    --only_predicted_class

python -m explanation_generation.gen_explanation_encoder \
    --dataset_name="google/jigsaw_unintended_bias" \
    --split="test" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models_jigsaw/distilbert_jigsaw_all/dropout" \
    --num_labels=2 \
    --batch_size=8 \
    --max_length=512 \
    --num_examples=200 \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results_jigsaw/distilbert_jigsaw_all_religion_test_200/dropout" \
    --baseline="pad" \
    --bias_type="religion" \
    --seed=42 \
    --only_predicted_class

python -m explanation_generation.gen_explanation_encoder \
    --dataset_name="google/jigsaw_unintended_bias" \
    --split="test" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models_jigsaw/distilbert_jigsaw_religion/attention_entropy" \
    --num_labels=2 \
    --batch_size=8 \
    --max_length=512 \
    --num_examples=200 \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results_jigsaw/distilbert_jigsaw_religion_religion_test_200/attention_entropy" \
    --baseline="pad" \
    --bias_type="religion" \
    --seed=42 \
    --only_predicted_class

python -m explanation_generation.gen_explanation_encoder \
    --dataset_name="google/jigsaw_unintended_bias" \
    --split="test" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models_jigsaw/distilbert_jigsaw_all/attention_entropy" \
    --num_labels=2 \
    --batch_size=8 \
    --max_length=512 \
    --num_examples=200 \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results_jigsaw/distilbert_jigsaw_all_religion_test_200/attention_entropy" \
    --baseline="pad" \
    --bias_type="religion" \
    --seed=42 \
    --only_predicted_class

python -m explanation_generation.gen_explanation_encoder \
    --dataset_name="google/jigsaw_unintended_bias" \
    --split="test" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models_jigsaw/distilbert_jigsaw_religion/causal_debias" \
    --num_labels=2 \
    --batch_size=8 \
    --max_length=512 \
    --num_examples=200 \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results_jigsaw/distilbert_jigsaw_religion_religion_test_200/causal_debias" \
    --baseline="pad" \
    --bias_type="religion" \
    --seed=42 \
    --only_predicted_class

python -m explanation_generation.gen_explanation_encoder \
    --dataset_name="google/jigsaw_unintended_bias" \
    --split="test" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models_jigsaw/distilbert_jigsaw_all/causal_debias" \
    --num_labels=2 \
    --batch_size=8 \
    --max_length=512 \
    --num_examples=200 \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results_jigsaw/distilbert_jigsaw_all_religion_test_200/causal_debias" \
    --baseline="pad" \
    --bias_type="religion" \
    --seed=42 \
    --only_predicted_class


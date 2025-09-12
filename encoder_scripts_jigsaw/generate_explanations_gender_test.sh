#!/bin/bash

python -m explanation_generation.gen_explanation_encoder \
    --dataset_name="google/jigsaw_unintended_bias" \
    --split="test" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models_jigsaw/bert_jigsaw_gender/no_debiasing" \
    --num_labels=2 \
    --batch_size=8 \
    --max_length=512 \
    --num_examples=800 \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results_jigsaw/bert_jigsaw_gender_gender_test_800/no_debiasing" \
    --baseline="pad" \
    --bias_type="gender" \
    --seed=42 \
    --only_predicted_class

python -m explanation_generation.gen_explanation_encoder \
    --dataset_name="google/jigsaw_unintended_bias" \
    --split="test" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models_jigsaw/bert_jigsaw_all/no_debiasing" \
    --num_labels=2 \
    --batch_size=8 \
    --max_length=512 \
    --num_examples=800 \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results_jigsaw/bert_jigsaw_all_gender_test_800/no_debiasing" \
    --baseline="pad" \
    --bias_type="gender" \
    --seed=42 \
    --only_predicted_class

python -m explanation_generation.gen_explanation_encoder \
    --dataset_name="google/jigsaw_unintended_bias" \
    --split="test" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models_jigsaw/bert_jigsaw_gender/group_balance" \
    --num_labels=2 \
    --batch_size=8 \
    --max_length=512 \
    --num_examples=800 \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results_jigsaw/bert_jigsaw_gender_gender_test_800/group_balance" \
    --baseline="pad" \
    --bias_type="gender" \
    --seed=42 \
    --only_predicted_class

python -m explanation_generation.gen_explanation_encoder \
    --dataset_name="google/jigsaw_unintended_bias" \
    --split="test" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models_jigsaw/bert_jigsaw_all/group_balance" \
    --num_labels=2 \
    --batch_size=8 \
    --max_length=512 \
    --num_examples=800 \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results_jigsaw/bert_jigsaw_all_gender_test_800/group_balance" \
    --baseline="pad" \
    --bias_type="gender" \
    --seed=42 \
    --only_predicted_class

python -m explanation_generation.gen_explanation_encoder \
    --dataset_name="google/jigsaw_unintended_bias" \
    --split="test" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models_jigsaw/bert_jigsaw_gender/group_class_balance" \
    --num_labels=2 \
    --batch_size=8 \
    --max_length=512 \
    --num_examples=800 \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results_jigsaw/bert_jigsaw_gender_gender_test_800/group_class_balance" \
    --baseline="pad" \
    --bias_type="gender" \
    --seed=42 \
    --only_predicted_class

python -m explanation_generation.gen_explanation_encoder \
    --dataset_name="google/jigsaw_unintended_bias" \
    --split="test" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models_jigsaw/bert_jigsaw_all/group_class_balance" \
    --num_labels=2 \
    --batch_size=8 \
    --max_length=512 \
    --num_examples=800 \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results_jigsaw/bert_jigsaw_all_gender_test_800/group_class_balance" \
    --baseline="pad" \
    --bias_type="gender" \
    --seed=42 \
    --only_predicted_class

python -m explanation_generation.gen_explanation_encoder \
    --dataset_name="google/jigsaw_unintended_bias" \
    --split="test" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models_jigsaw/bert_jigsaw_gender/cda" \
    --num_labels=2 \
    --batch_size=8 \
    --max_length=512 \
    --num_examples=800 \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results_jigsaw/bert_jigsaw_gender_gender_test_800/cda" \
    --baseline="pad" \
    --bias_type="gender" \
    --seed=42 \
    --only_predicted_class

python -m explanation_generation.gen_explanation_encoder \
    --dataset_name="google/jigsaw_unintended_bias" \
    --split="test" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models_jigsaw/bert_jigsaw_all/cda" \
    --num_labels=2 \
    --batch_size=8 \
    --max_length=512 \
    --num_examples=800 \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results_jigsaw/bert_jigsaw_all_gender_test_800/cda" \
    --baseline="pad" \
    --bias_type="gender" \
    --seed=42 \
    --only_predicted_class

python -m explanation_generation.gen_explanation_encoder \
    --dataset_name="google/jigsaw_unintended_bias" \
    --split="test" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models_jigsaw/bert_jigsaw_gender/dropout" \
    --num_labels=2 \
    --batch_size=8 \
    --max_length=512 \
    --num_examples=800 \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results_jigsaw/bert_jigsaw_gender_gender_test_800/dropout" \
    --baseline="pad" \
    --bias_type="gender" \
    --seed=42 \
    --only_predicted_class

python -m explanation_generation.gen_explanation_encoder \
    --dataset_name="google/jigsaw_unintended_bias" \
    --split="test" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models_jigsaw/bert_jigsaw_all/dropout" \
    --num_labels=2 \
    --batch_size=8 \
    --max_length=512 \
    --num_examples=800 \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results_jigsaw/bert_jigsaw_all_gender_test_800/dropout" \
    --baseline="pad" \
    --bias_type="gender" \
    --seed=42 \
    --only_predicted_class

python -m explanation_generation.gen_explanation_encoder \
    --dataset_name="google/jigsaw_unintended_bias" \
    --split="test" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models_jigsaw/bert_jigsaw_gender/attention_entropy" \
    --num_labels=2 \
    --batch_size=8 \
    --max_length=512 \
    --num_examples=800 \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results_jigsaw/bert_jigsaw_gender_gender_test_800/attention_entropy" \
    --baseline="pad" \
    --bias_type="gender" \
    --seed=42 \
    --only_predicted_class

python -m explanation_generation.gen_explanation_encoder \
    --dataset_name="google/jigsaw_unintended_bias" \
    --split="test" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models_jigsaw/bert_jigsaw_all/attention_entropy" \
    --num_labels=2 \
    --batch_size=8 \
    --max_length=512 \
    --num_examples=800 \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results_jigsaw/bert_jigsaw_all_gender_test_800/attention_entropy" \
    --baseline="pad" \
    --bias_type="gender" \
    --seed=42 \
    --only_predicted_class

python -m explanation_generation.gen_explanation_encoder \
    --dataset_name="google/jigsaw_unintended_bias" \
    --split="test" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models_jigsaw/bert_jigsaw_gender/causal_debias" \
    --num_labels=2 \
    --batch_size=8 \
    --max_length=512 \
    --num_examples=800 \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results_jigsaw/bert_jigsaw_gender_gender_test_800/causal_debias" \
    --baseline="pad" \
    --bias_type="gender" \
    --seed=42 \
    --only_predicted_class

python -m explanation_generation.gen_explanation_encoder \
    --dataset_name="google/jigsaw_unintended_bias" \
    --split="test" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models_jigsaw/bert_jigsaw_all/causal_debias" \
    --num_labels=2 \
    --batch_size=8 \
    --max_length=512 \
    --num_examples=800 \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results_jigsaw/bert_jigsaw_all_gender_test_800/causal_debias" \
    --baseline="pad" \
    --bias_type="gender" \
    --seed=42 \
    --only_predicted_class

python -m explanation_generation.gen_explanation_encoder \
    --dataset_name="google/jigsaw_unintended_bias" \
    --split="test" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models_jigsaw/roberta_jigsaw_gender/no_debiasing" \
    --num_labels=2 \
    --batch_size=8 \
    --max_length=512 \
    --num_examples=800 \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results_jigsaw/roberta_jigsaw_gender_gender_test_800/no_debiasing" \
    --baseline="pad" \
    --bias_type="gender" \
    --seed=42 \
    --only_predicted_class

python -m explanation_generation.gen_explanation_encoder \
    --dataset_name="google/jigsaw_unintended_bias" \
    --split="test" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models_jigsaw/roberta_jigsaw_all/no_debiasing" \
    --num_labels=2 \
    --batch_size=8 \
    --max_length=512 \
    --num_examples=800 \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results_jigsaw/roberta_jigsaw_all_gender_test_800/no_debiasing" \
    --baseline="pad" \
    --bias_type="gender" \
    --seed=42 \
    --only_predicted_class

python -m explanation_generation.gen_explanation_encoder \
    --dataset_name="google/jigsaw_unintended_bias" \
    --split="test" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models_jigsaw/roberta_jigsaw_gender/group_balance" \
    --num_labels=2 \
    --batch_size=8 \
    --max_length=512 \
    --num_examples=800 \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results_jigsaw/roberta_jigsaw_gender_gender_test_800/group_balance" \
    --baseline="pad" \
    --bias_type="gender" \
    --seed=42 \
    --only_predicted_class

python -m explanation_generation.gen_explanation_encoder \
    --dataset_name="google/jigsaw_unintended_bias" \
    --split="test" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models_jigsaw/roberta_jigsaw_all/group_balance" \
    --num_labels=2 \
    --batch_size=8 \
    --max_length=512 \
    --num_examples=800 \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results_jigsaw/roberta_jigsaw_all_gender_test_800/group_balance" \
    --baseline="pad" \
    --bias_type="gender" \
    --seed=42 \
    --only_predicted_class

python -m explanation_generation.gen_explanation_encoder \
    --dataset_name="google/jigsaw_unintended_bias" \
    --split="test" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models_jigsaw/roberta_jigsaw_gender/group_class_balance" \
    --num_labels=2 \
    --batch_size=8 \
    --max_length=512 \
    --num_examples=800 \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results_jigsaw/roberta_jigsaw_gender_gender_test_800/group_class_balance" \
    --baseline="pad" \
    --bias_type="gender" \
    --seed=42 \
    --only_predicted_class

python -m explanation_generation.gen_explanation_encoder \
    --dataset_name="google/jigsaw_unintended_bias" \
    --split="test" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models_jigsaw/roberta_jigsaw_all/group_class_balance" \
    --num_labels=2 \
    --batch_size=8 \
    --max_length=512 \
    --num_examples=800 \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results_jigsaw/roberta_jigsaw_all_gender_test_800/group_class_balance" \
    --baseline="pad" \
    --bias_type="gender" \
    --seed=42 \
    --only_predicted_class

python -m explanation_generation.gen_explanation_encoder \
    --dataset_name="google/jigsaw_unintended_bias" \
    --split="test" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models_jigsaw/roberta_jigsaw_gender/cda" \
    --num_labels=2 \
    --batch_size=8 \
    --max_length=512 \
    --num_examples=800 \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results_jigsaw/roberta_jigsaw_gender_gender_test_800/cda" \
    --baseline="pad" \
    --bias_type="gender" \
    --seed=42 \
    --only_predicted_class

python -m explanation_generation.gen_explanation_encoder \
    --dataset_name="google/jigsaw_unintended_bias" \
    --split="test" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models_jigsaw/roberta_jigsaw_all/cda" \
    --num_labels=2 \
    --batch_size=8 \
    --max_length=512 \
    --num_examples=800 \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results_jigsaw/roberta_jigsaw_all_gender_test_800/cda" \
    --baseline="pad" \
    --bias_type="gender" \
    --seed=42 \
    --only_predicted_class

python -m explanation_generation.gen_explanation_encoder \
    --dataset_name="google/jigsaw_unintended_bias" \
    --split="test" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models_jigsaw/roberta_jigsaw_gender/dropout" \
    --num_labels=2 \
    --batch_size=8 \
    --max_length=512 \
    --num_examples=800 \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results_jigsaw/roberta_jigsaw_gender_gender_test_800/dropout" \
    --baseline="pad" \
    --bias_type="gender" \
    --seed=42 \
    --only_predicted_class

python -m explanation_generation.gen_explanation_encoder \
    --dataset_name="google/jigsaw_unintended_bias" \
    --split="test" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models_jigsaw/roberta_jigsaw_all/dropout" \
    --num_labels=2 \
    --batch_size=8 \
    --max_length=512 \
    --num_examples=800 \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results_jigsaw/roberta_jigsaw_all_gender_test_800/dropout" \
    --baseline="pad" \
    --bias_type="gender" \
    --seed=42 \
    --only_predicted_class

python -m explanation_generation.gen_explanation_encoder \
    --dataset_name="google/jigsaw_unintended_bias" \
    --split="test" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models_jigsaw/roberta_jigsaw_gender/attention_entropy" \
    --num_labels=2 \
    --batch_size=8 \
    --max_length=512 \
    --num_examples=800 \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results_jigsaw/roberta_jigsaw_gender_gender_test_800/attention_entropy" \
    --baseline="pad" \
    --bias_type="gender" \
    --seed=42 \
    --only_predicted_class

python -m explanation_generation.gen_explanation_encoder \
    --dataset_name="google/jigsaw_unintended_bias" \
    --split="test" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models_jigsaw/roberta_jigsaw_all/attention_entropy" \
    --num_labels=2 \
    --batch_size=8 \
    --max_length=512 \
    --num_examples=800 \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results_jigsaw/roberta_jigsaw_all_gender_test_800/attention_entropy" \
    --baseline="pad" \
    --bias_type="gender" \
    --seed=42 \
    --only_predicted_class

python -m explanation_generation.gen_explanation_encoder \
    --dataset_name="google/jigsaw_unintended_bias" \
    --split="test" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models_jigsaw/roberta_jigsaw_gender/causal_debias" \
    --num_labels=2 \
    --batch_size=8 \
    --max_length=512 \
    --num_examples=800 \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results_jigsaw/roberta_jigsaw_gender_gender_test_800/causal_debias" \
    --baseline="pad" \
    --bias_type="gender" \
    --seed=42 \
    --only_predicted_class

python -m explanation_generation.gen_explanation_encoder \
    --dataset_name="google/jigsaw_unintended_bias" \
    --split="test" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models_jigsaw/roberta_jigsaw_all/causal_debias" \
    --num_labels=2 \
    --batch_size=8 \
    --max_length=512 \
    --num_examples=800 \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results_jigsaw/roberta_jigsaw_all_gender_test_800/causal_debias" \
    --baseline="pad" \
    --bias_type="gender" \
    --seed=42 \
    --only_predicted_class

python -m explanation_generation.gen_explanation_encoder \
    --dataset_name="google/jigsaw_unintended_bias" \
    --split="test" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models_jigsaw/distilbert_jigsaw_gender/no_debiasing" \
    --num_labels=2 \
    --batch_size=8 \
    --max_length=512 \
    --num_examples=800 \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results_jigsaw/distilbert_jigsaw_gender_gender_test_800/no_debiasing" \
    --baseline="pad" \
    --bias_type="gender" \
    --seed=42 \
    --only_predicted_class

python -m explanation_generation.gen_explanation_encoder \
    --dataset_name="google/jigsaw_unintended_bias" \
    --split="test" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models_jigsaw/distilbert_jigsaw_all/no_debiasing" \
    --num_labels=2 \
    --batch_size=8 \
    --max_length=512 \
    --num_examples=800 \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results_jigsaw/distilbert_jigsaw_all_gender_test_800/no_debiasing" \
    --baseline="pad" \
    --bias_type="gender" \
    --seed=42 \
    --only_predicted_class

python -m explanation_generation.gen_explanation_encoder \
    --dataset_name="google/jigsaw_unintended_bias" \
    --split="test" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models_jigsaw/distilbert_jigsaw_gender/group_balance" \
    --num_labels=2 \
    --batch_size=8 \
    --max_length=512 \
    --num_examples=800 \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results_jigsaw/distilbert_jigsaw_gender_gender_test_800/group_balance" \
    --baseline="pad" \
    --bias_type="gender" \
    --seed=42 \
    --only_predicted_class

python -m explanation_generation.gen_explanation_encoder \
    --dataset_name="google/jigsaw_unintended_bias" \
    --split="test" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models_jigsaw/distilbert_jigsaw_all/group_balance" \
    --num_labels=2 \
    --batch_size=8 \
    --max_length=512 \
    --num_examples=800 \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results_jigsaw/distilbert_jigsaw_all_gender_test_800/group_balance" \
    --baseline="pad" \
    --bias_type="gender" \
    --seed=42 \
    --only_predicted_class

python -m explanation_generation.gen_explanation_encoder \
    --dataset_name="google/jigsaw_unintended_bias" \
    --split="test" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models_jigsaw/distilbert_jigsaw_gender/group_class_balance" \
    --num_labels=2 \
    --batch_size=8 \
    --max_length=512 \
    --num_examples=800 \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results_jigsaw/distilbert_jigsaw_gender_gender_test_800/group_class_balance" \
    --baseline="pad" \
    --bias_type="gender" \
    --seed=42 \
    --only_predicted_class

python -m explanation_generation.gen_explanation_encoder \
    --dataset_name="google/jigsaw_unintended_bias" \
    --split="test" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models_jigsaw/distilbert_jigsaw_all/group_class_balance" \
    --num_labels=2 \
    --batch_size=8 \
    --max_length=512 \
    --num_examples=800 \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results_jigsaw/distilbert_jigsaw_all_gender_test_800/group_class_balance" \
    --baseline="pad" \
    --bias_type="gender" \
    --seed=42 \
    --only_predicted_class

python -m explanation_generation.gen_explanation_encoder \
    --dataset_name="google/jigsaw_unintended_bias" \
    --split="test" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models_jigsaw/distilbert_jigsaw_gender/cda" \
    --num_labels=2 \
    --batch_size=8 \
    --max_length=512 \
    --num_examples=800 \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results_jigsaw/distilbert_jigsaw_gender_gender_test_800/cda" \
    --baseline="pad" \
    --bias_type="gender" \
    --seed=42 \
    --only_predicted_class

python -m explanation_generation.gen_explanation_encoder \
    --dataset_name="google/jigsaw_unintended_bias" \
    --split="test" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models_jigsaw/distilbert_jigsaw_all/cda" \
    --num_labels=2 \
    --batch_size=8 \
    --max_length=512 \
    --num_examples=800 \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results_jigsaw/distilbert_jigsaw_all_gender_test_800/cda" \
    --baseline="pad" \
    --bias_type="gender" \
    --seed=42 \
    --only_predicted_class

python -m explanation_generation.gen_explanation_encoder \
    --dataset_name="google/jigsaw_unintended_bias" \
    --split="test" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models_jigsaw/distilbert_jigsaw_gender/dropout" \
    --num_labels=2 \
    --batch_size=8 \
    --max_length=512 \
    --num_examples=800 \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results_jigsaw/distilbert_jigsaw_gender_gender_test_800/dropout" \
    --baseline="pad" \
    --bias_type="gender" \
    --seed=42 \
    --only_predicted_class

python -m explanation_generation.gen_explanation_encoder \
    --dataset_name="google/jigsaw_unintended_bias" \
    --split="test" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models_jigsaw/distilbert_jigsaw_all/dropout" \
    --num_labels=2 \
    --batch_size=8 \
    --max_length=512 \
    --num_examples=800 \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results_jigsaw/distilbert_jigsaw_all_gender_test_800/dropout" \
    --baseline="pad" \
    --bias_type="gender" \
    --seed=42 \
    --only_predicted_class

python -m explanation_generation.gen_explanation_encoder \
    --dataset_name="google/jigsaw_unintended_bias" \
    --split="test" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models_jigsaw/distilbert_jigsaw_gender/attention_entropy" \
    --num_labels=2 \
    --batch_size=8 \
    --max_length=512 \
    --num_examples=800 \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results_jigsaw/distilbert_jigsaw_gender_gender_test_800/attention_entropy" \
    --baseline="pad" \
    --bias_type="gender" \
    --seed=42 \
    --only_predicted_class

python -m explanation_generation.gen_explanation_encoder \
    --dataset_name="google/jigsaw_unintended_bias" \
    --split="test" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models_jigsaw/distilbert_jigsaw_all/attention_entropy" \
    --num_labels=2 \
    --batch_size=8 \
    --max_length=512 \
    --num_examples=800 \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results_jigsaw/distilbert_jigsaw_all_gender_test_800/attention_entropy" \
    --baseline="pad" \
    --bias_type="gender" \
    --seed=42 \
    --only_predicted_class

python -m explanation_generation.gen_explanation_encoder \
    --dataset_name="google/jigsaw_unintended_bias" \
    --split="test" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models_jigsaw/distilbert_jigsaw_gender/causal_debias" \
    --num_labels=2 \
    --batch_size=8 \
    --max_length=512 \
    --num_examples=800 \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results_jigsaw/distilbert_jigsaw_gender_gender_test_800/causal_debias" \
    --baseline="pad" \
    --bias_type="gender" \
    --seed=42 \
    --only_predicted_class

python -m explanation_generation.gen_explanation_encoder \
    --dataset_name="google/jigsaw_unintended_bias" \
    --split="test" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models_jigsaw/distilbert_jigsaw_all/causal_debias" \
    --num_labels=2 \
    --batch_size=8 \
    --max_length=512 \
    --num_examples=800 \
    --methods="Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients" \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results_jigsaw/distilbert_jigsaw_all_gender_test_800/causal_debias" \
    --baseline="pad" \
    --bias_type="gender" \
    --seed=42 \
    --only_predicted_class


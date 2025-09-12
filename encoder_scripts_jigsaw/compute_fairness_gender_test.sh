#!/bin/bash

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="google/jigsaw_unintended_bias" \
    --split="test" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models_jigsaw/bert_jigsaw_gender/no_debiasing" \
    --batch_size=32 \
    --max_seq_length=512 \
    --num_examples=800 \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results_jigsaw/bert_jigsaw_gender_gender_test_800/no_debiasing" \
    --bias_type="gender"

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="google/jigsaw_unintended_bias" \
    --split="test" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models_jigsaw/bert_jigsaw_all/no_debiasing" \
    --batch_size=32 \
    --max_seq_length=512 \
    --num_examples=800 \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results_jigsaw/bert_jigsaw_all_gender_test_800/no_debiasing" \
    --bias_type="gender"

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="google/jigsaw_unintended_bias" \
    --split="test" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models_jigsaw/bert_jigsaw_gender/group_balance" \
    --batch_size=32 \
    --max_seq_length=512 \
    --num_examples=800 \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results_jigsaw/bert_jigsaw_gender_gender_test_800/group_balance" \
    --bias_type="gender"

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="google/jigsaw_unintended_bias" \
    --split="test" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models_jigsaw/bert_jigsaw_all/group_balance" \
    --batch_size=32 \
    --max_seq_length=512 \
    --num_examples=800 \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results_jigsaw/bert_jigsaw_all_gender_test_800/group_balance" \
    --bias_type="gender"

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="google/jigsaw_unintended_bias" \
    --split="test" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models_jigsaw/bert_jigsaw_gender/group_class_balance" \
    --batch_size=32 \
    --max_seq_length=512 \
    --num_examples=800 \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results_jigsaw/bert_jigsaw_gender_gender_test_800/group_class_balance" \
    --bias_type="gender"

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="google/jigsaw_unintended_bias" \
    --split="test" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models_jigsaw/bert_jigsaw_all/group_class_balance" \
    --batch_size=32 \
    --max_seq_length=512 \
    --num_examples=800 \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results_jigsaw/bert_jigsaw_all_gender_test_800/group_class_balance" \
    --bias_type="gender"

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="google/jigsaw_unintended_bias" \
    --split="test" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models_jigsaw/bert_jigsaw_gender/cda" \
    --batch_size=32 \
    --max_seq_length=512 \
    --num_examples=800 \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results_jigsaw/bert_jigsaw_gender_gender_test_800/cda" \
    --bias_type="gender"

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="google/jigsaw_unintended_bias" \
    --split="test" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models_jigsaw/bert_jigsaw_all/cda" \
    --batch_size=32 \
    --max_seq_length=512 \
    --num_examples=800 \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results_jigsaw/bert_jigsaw_all_gender_test_800/cda" \
    --bias_type="gender"

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="google/jigsaw_unintended_bias" \
    --split="test" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models_jigsaw/bert_jigsaw_gender/dropout" \
    --batch_size=32 \
    --max_seq_length=512 \
    --num_examples=800 \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results_jigsaw/bert_jigsaw_gender_gender_test_800/dropout" \
    --bias_type="gender"

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="google/jigsaw_unintended_bias" \
    --split="test" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models_jigsaw/bert_jigsaw_all/dropout" \
    --batch_size=32 \
    --max_seq_length=512 \
    --num_examples=800 \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results_jigsaw/bert_jigsaw_all_gender_test_800/dropout" \
    --bias_type="gender"

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="google/jigsaw_unintended_bias" \
    --split="test" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models_jigsaw/bert_jigsaw_gender/attention_entropy" \
    --batch_size=32 \
    --max_seq_length=512 \
    --num_examples=800 \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results_jigsaw/bert_jigsaw_gender_gender_test_800/attention_entropy" \
    --bias_type="gender"

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="google/jigsaw_unintended_bias" \
    --split="test" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models_jigsaw/bert_jigsaw_all/attention_entropy" \
    --batch_size=32 \
    --max_seq_length=512 \
    --num_examples=800 \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results_jigsaw/bert_jigsaw_all_gender_test_800/attention_entropy" \
    --bias_type="gender"

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="google/jigsaw_unintended_bias" \
    --split="test" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models_jigsaw/bert_jigsaw_gender/causal_debias" \
    --batch_size=32 \
    --max_seq_length=512 \
    --num_examples=800 \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results_jigsaw/bert_jigsaw_gender_gender_test_800/causal_debias" \
    --bias_type="gender"

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="google/jigsaw_unintended_bias" \
    --split="test" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models_jigsaw/bert_jigsaw_all/causal_debias" \
    --batch_size=32 \
    --max_seq_length=512 \
    --num_examples=800 \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results_jigsaw/bert_jigsaw_all_gender_test_800/causal_debias" \
    --bias_type="gender"

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="google/jigsaw_unintended_bias" \
    --split="test" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models_jigsaw/roberta_jigsaw_gender/no_debiasing" \
    --batch_size=32 \
    --max_seq_length=512 \
    --num_examples=800 \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results_jigsaw/roberta_jigsaw_gender_gender_test_800/no_debiasing" \
    --bias_type="gender"

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="google/jigsaw_unintended_bias" \
    --split="test" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models_jigsaw/roberta_jigsaw_all/no_debiasing" \
    --batch_size=32 \
    --max_seq_length=512 \
    --num_examples=800 \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results_jigsaw/roberta_jigsaw_all_gender_test_800/no_debiasing" \
    --bias_type="gender"

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="google/jigsaw_unintended_bias" \
    --split="test" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models_jigsaw/roberta_jigsaw_gender/group_balance" \
    --batch_size=32 \
    --max_seq_length=512 \
    --num_examples=800 \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results_jigsaw/roberta_jigsaw_gender_gender_test_800/group_balance" \
    --bias_type="gender"

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="google/jigsaw_unintended_bias" \
    --split="test" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models_jigsaw/roberta_jigsaw_all/group_balance" \
    --batch_size=32 \
    --max_seq_length=512 \
    --num_examples=800 \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results_jigsaw/roberta_jigsaw_all_gender_test_800/group_balance" \
    --bias_type="gender"

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="google/jigsaw_unintended_bias" \
    --split="test" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models_jigsaw/roberta_jigsaw_gender/group_class_balance" \
    --batch_size=32 \
    --max_seq_length=512 \
    --num_examples=800 \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results_jigsaw/roberta_jigsaw_gender_gender_test_800/group_class_balance" \
    --bias_type="gender"

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="google/jigsaw_unintended_bias" \
    --split="test" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models_jigsaw/roberta_jigsaw_all/group_class_balance" \
    --batch_size=32 \
    --max_seq_length=512 \
    --num_examples=800 \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results_jigsaw/roberta_jigsaw_all_gender_test_800/group_class_balance" \
    --bias_type="gender"

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="google/jigsaw_unintended_bias" \
    --split="test" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models_jigsaw/roberta_jigsaw_gender/cda" \
    --batch_size=32 \
    --max_seq_length=512 \
    --num_examples=800 \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results_jigsaw/roberta_jigsaw_gender_gender_test_800/cda" \
    --bias_type="gender"

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="google/jigsaw_unintended_bias" \
    --split="test" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models_jigsaw/roberta_jigsaw_all/cda" \
    --batch_size=32 \
    --max_seq_length=512 \
    --num_examples=800 \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results_jigsaw/roberta_jigsaw_all_gender_test_800/cda" \
    --bias_type="gender"

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="google/jigsaw_unintended_bias" \
    --split="test" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models_jigsaw/roberta_jigsaw_gender/dropout" \
    --batch_size=32 \
    --max_seq_length=512 \
    --num_examples=800 \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results_jigsaw/roberta_jigsaw_gender_gender_test_800/dropout" \
    --bias_type="gender"

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="google/jigsaw_unintended_bias" \
    --split="test" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models_jigsaw/roberta_jigsaw_all/dropout" \
    --batch_size=32 \
    --max_seq_length=512 \
    --num_examples=800 \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results_jigsaw/roberta_jigsaw_all_gender_test_800/dropout" \
    --bias_type="gender"

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="google/jigsaw_unintended_bias" \
    --split="test" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models_jigsaw/roberta_jigsaw_gender/attention_entropy" \
    --batch_size=32 \
    --max_seq_length=512 \
    --num_examples=800 \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results_jigsaw/roberta_jigsaw_gender_gender_test_800/attention_entropy" \
    --bias_type="gender"

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="google/jigsaw_unintended_bias" \
    --split="test" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models_jigsaw/roberta_jigsaw_all/attention_entropy" \
    --batch_size=32 \
    --max_seq_length=512 \
    --num_examples=800 \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results_jigsaw/roberta_jigsaw_all_gender_test_800/attention_entropy" \
    --bias_type="gender"

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="google/jigsaw_unintended_bias" \
    --split="test" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models_jigsaw/roberta_jigsaw_gender/causal_debias" \
    --batch_size=32 \
    --max_seq_length=512 \
    --num_examples=800 \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results_jigsaw/roberta_jigsaw_gender_gender_test_800/causal_debias" \
    --bias_type="gender"

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="google/jigsaw_unintended_bias" \
    --split="test" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models_jigsaw/roberta_jigsaw_all/causal_debias" \
    --batch_size=32 \
    --max_seq_length=512 \
    --num_examples=800 \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results_jigsaw/roberta_jigsaw_all_gender_test_800/causal_debias" \
    --bias_type="gender"

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="google/jigsaw_unintended_bias" \
    --split="test" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models_jigsaw/distilbert_jigsaw_gender/no_debiasing" \
    --batch_size=32 \
    --max_seq_length=512 \
    --num_examples=800 \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results_jigsaw/distilbert_jigsaw_gender_gender_test_800/no_debiasing" \
    --bias_type="gender"

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="google/jigsaw_unintended_bias" \
    --split="test" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models_jigsaw/distilbert_jigsaw_all/no_debiasing" \
    --batch_size=32 \
    --max_seq_length=512 \
    --num_examples=800 \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results_jigsaw/distilbert_jigsaw_all_gender_test_800/no_debiasing" \
    --bias_type="gender"

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="google/jigsaw_unintended_bias" \
    --split="test" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models_jigsaw/distilbert_jigsaw_gender/group_balance" \
    --batch_size=32 \
    --max_seq_length=512 \
    --num_examples=800 \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results_jigsaw/distilbert_jigsaw_gender_gender_test_800/group_balance" \
    --bias_type="gender"

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="google/jigsaw_unintended_bias" \
    --split="test" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models_jigsaw/distilbert_jigsaw_all/group_balance" \
    --batch_size=32 \
    --max_seq_length=512 \
    --num_examples=800 \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results_jigsaw/distilbert_jigsaw_all_gender_test_800/group_balance" \
    --bias_type="gender"

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="google/jigsaw_unintended_bias" \
    --split="test" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models_jigsaw/distilbert_jigsaw_gender/group_class_balance" \
    --batch_size=32 \
    --max_seq_length=512 \
    --num_examples=800 \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results_jigsaw/distilbert_jigsaw_gender_gender_test_800/group_class_balance" \
    --bias_type="gender"

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="google/jigsaw_unintended_bias" \
    --split="test" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models_jigsaw/distilbert_jigsaw_all/group_class_balance" \
    --batch_size=32 \
    --max_seq_length=512 \
    --num_examples=800 \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results_jigsaw/distilbert_jigsaw_all_gender_test_800/group_class_balance" \
    --bias_type="gender"

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="google/jigsaw_unintended_bias" \
    --split="test" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models_jigsaw/distilbert_jigsaw_gender/cda" \
    --batch_size=32 \
    --max_seq_length=512 \
    --num_examples=800 \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results_jigsaw/distilbert_jigsaw_gender_gender_test_800/cda" \
    --bias_type="gender"

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="google/jigsaw_unintended_bias" \
    --split="test" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models_jigsaw/distilbert_jigsaw_all/cda" \
    --batch_size=32 \
    --max_seq_length=512 \
    --num_examples=800 \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results_jigsaw/distilbert_jigsaw_all_gender_test_800/cda" \
    --bias_type="gender"

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="google/jigsaw_unintended_bias" \
    --split="test" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models_jigsaw/distilbert_jigsaw_gender/dropout" \
    --batch_size=32 \
    --max_seq_length=512 \
    --num_examples=800 \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results_jigsaw/distilbert_jigsaw_gender_gender_test_800/dropout" \
    --bias_type="gender"

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="google/jigsaw_unintended_bias" \
    --split="test" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models_jigsaw/distilbert_jigsaw_all/dropout" \
    --batch_size=32 \
    --max_seq_length=512 \
    --num_examples=800 \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results_jigsaw/distilbert_jigsaw_all_gender_test_800/dropout" \
    --bias_type="gender"

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="google/jigsaw_unintended_bias" \
    --split="test" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models_jigsaw/distilbert_jigsaw_gender/attention_entropy" \
    --batch_size=32 \
    --max_seq_length=512 \
    --num_examples=800 \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results_jigsaw/distilbert_jigsaw_gender_gender_test_800/attention_entropy" \
    --bias_type="gender"

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="google/jigsaw_unintended_bias" \
    --split="test" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models_jigsaw/distilbert_jigsaw_all/attention_entropy" \
    --batch_size=32 \
    --max_seq_length=512 \
    --num_examples=800 \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results_jigsaw/distilbert_jigsaw_all_gender_test_800/attention_entropy" \
    --bias_type="gender"

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="google/jigsaw_unintended_bias" \
    --split="test" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models_jigsaw/distilbert_jigsaw_gender/causal_debias" \
    --batch_size=32 \
    --max_seq_length=512 \
    --num_examples=800 \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results_jigsaw/distilbert_jigsaw_gender_gender_test_800/causal_debias" \
    --bias_type="gender"

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="google/jigsaw_unintended_bias" \
    --split="test" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models_jigsaw/distilbert_jigsaw_all/causal_debias" \
    --batch_size=32 \
    --max_seq_length=512 \
    --num_examples=800 \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results_jigsaw/distilbert_jigsaw_all_gender_test_800/causal_debias" \
    --bias_type="gender"


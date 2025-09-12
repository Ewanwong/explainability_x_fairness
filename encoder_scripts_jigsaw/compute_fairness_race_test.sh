#!/bin/bash

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="google/jigsaw_unintended_bias" \
    --split="test" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models_jigsaw/bert_jigsaw_race/no_debiasing" \
    --batch_size=32 \
    --max_seq_length=512 \
    --num_examples=400 \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results_jigsaw/bert_jigsaw_race_race_test_400/no_debiasing" \
    --bias_type="race"

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="google/jigsaw_unintended_bias" \
    --split="test" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models_jigsaw/bert_jigsaw_all/no_debiasing" \
    --batch_size=32 \
    --max_seq_length=512 \
    --num_examples=400 \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results_jigsaw/bert_jigsaw_all_race_test_400/no_debiasing" \
    --bias_type="race"

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="google/jigsaw_unintended_bias" \
    --split="test" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models_jigsaw/bert_jigsaw_race/group_balance" \
    --batch_size=32 \
    --max_seq_length=512 \
    --num_examples=400 \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results_jigsaw/bert_jigsaw_race_race_test_400/group_balance" \
    --bias_type="race"

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="google/jigsaw_unintended_bias" \
    --split="test" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models_jigsaw/bert_jigsaw_all/group_balance" \
    --batch_size=32 \
    --max_seq_length=512 \
    --num_examples=400 \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results_jigsaw/bert_jigsaw_all_race_test_400/group_balance" \
    --bias_type="race"

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="google/jigsaw_unintended_bias" \
    --split="test" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models_jigsaw/bert_jigsaw_race/group_class_balance" \
    --batch_size=32 \
    --max_seq_length=512 \
    --num_examples=400 \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results_jigsaw/bert_jigsaw_race_race_test_400/group_class_balance" \
    --bias_type="race"

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="google/jigsaw_unintended_bias" \
    --split="test" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models_jigsaw/bert_jigsaw_all/group_class_balance" \
    --batch_size=32 \
    --max_seq_length=512 \
    --num_examples=400 \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results_jigsaw/bert_jigsaw_all_race_test_400/group_class_balance" \
    --bias_type="race"

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="google/jigsaw_unintended_bias" \
    --split="test" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models_jigsaw/bert_jigsaw_race/cda" \
    --batch_size=32 \
    --max_seq_length=512 \
    --num_examples=400 \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results_jigsaw/bert_jigsaw_race_race_test_400/cda" \
    --bias_type="race"

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="google/jigsaw_unintended_bias" \
    --split="test" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models_jigsaw/bert_jigsaw_all/cda" \
    --batch_size=32 \
    --max_seq_length=512 \
    --num_examples=400 \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results_jigsaw/bert_jigsaw_all_race_test_400/cda" \
    --bias_type="race"

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="google/jigsaw_unintended_bias" \
    --split="test" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models_jigsaw/bert_jigsaw_race/dropout" \
    --batch_size=32 \
    --max_seq_length=512 \
    --num_examples=400 \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results_jigsaw/bert_jigsaw_race_race_test_400/dropout" \
    --bias_type="race"

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="google/jigsaw_unintended_bias" \
    --split="test" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models_jigsaw/bert_jigsaw_all/dropout" \
    --batch_size=32 \
    --max_seq_length=512 \
    --num_examples=400 \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results_jigsaw/bert_jigsaw_all_race_test_400/dropout" \
    --bias_type="race"

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="google/jigsaw_unintended_bias" \
    --split="test" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models_jigsaw/bert_jigsaw_race/attention_entropy" \
    --batch_size=32 \
    --max_seq_length=512 \
    --num_examples=400 \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results_jigsaw/bert_jigsaw_race_race_test_400/attention_entropy" \
    --bias_type="race"

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="google/jigsaw_unintended_bias" \
    --split="test" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models_jigsaw/bert_jigsaw_all/attention_entropy" \
    --batch_size=32 \
    --max_seq_length=512 \
    --num_examples=400 \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results_jigsaw/bert_jigsaw_all_race_test_400/attention_entropy" \
    --bias_type="race"

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="google/jigsaw_unintended_bias" \
    --split="test" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models_jigsaw/bert_jigsaw_race/causal_debias" \
    --batch_size=32 \
    --max_seq_length=512 \
    --num_examples=400 \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results_jigsaw/bert_jigsaw_race_race_test_400/causal_debias" \
    --bias_type="race"

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="google/jigsaw_unintended_bias" \
    --split="test" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models_jigsaw/bert_jigsaw_all/causal_debias" \
    --batch_size=32 \
    --max_seq_length=512 \
    --num_examples=400 \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results_jigsaw/bert_jigsaw_all_race_test_400/causal_debias" \
    --bias_type="race"

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="google/jigsaw_unintended_bias" \
    --split="test" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models_jigsaw/roberta_jigsaw_race/no_debiasing" \
    --batch_size=32 \
    --max_seq_length=512 \
    --num_examples=400 \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results_jigsaw/roberta_jigsaw_race_race_test_400/no_debiasing" \
    --bias_type="race"

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="google/jigsaw_unintended_bias" \
    --split="test" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models_jigsaw/roberta_jigsaw_all/no_debiasing" \
    --batch_size=32 \
    --max_seq_length=512 \
    --num_examples=400 \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results_jigsaw/roberta_jigsaw_all_race_test_400/no_debiasing" \
    --bias_type="race"

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="google/jigsaw_unintended_bias" \
    --split="test" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models_jigsaw/roberta_jigsaw_race/group_balance" \
    --batch_size=32 \
    --max_seq_length=512 \
    --num_examples=400 \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results_jigsaw/roberta_jigsaw_race_race_test_400/group_balance" \
    --bias_type="race"

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="google/jigsaw_unintended_bias" \
    --split="test" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models_jigsaw/roberta_jigsaw_all/group_balance" \
    --batch_size=32 \
    --max_seq_length=512 \
    --num_examples=400 \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results_jigsaw/roberta_jigsaw_all_race_test_400/group_balance" \
    --bias_type="race"

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="google/jigsaw_unintended_bias" \
    --split="test" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models_jigsaw/roberta_jigsaw_race/group_class_balance" \
    --batch_size=32 \
    --max_seq_length=512 \
    --num_examples=400 \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results_jigsaw/roberta_jigsaw_race_race_test_400/group_class_balance" \
    --bias_type="race"

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="google/jigsaw_unintended_bias" \
    --split="test" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models_jigsaw/roberta_jigsaw_all/group_class_balance" \
    --batch_size=32 \
    --max_seq_length=512 \
    --num_examples=400 \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results_jigsaw/roberta_jigsaw_all_race_test_400/group_class_balance" \
    --bias_type="race"

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="google/jigsaw_unintended_bias" \
    --split="test" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models_jigsaw/roberta_jigsaw_race/cda" \
    --batch_size=32 \
    --max_seq_length=512 \
    --num_examples=400 \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results_jigsaw/roberta_jigsaw_race_race_test_400/cda" \
    --bias_type="race"

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="google/jigsaw_unintended_bias" \
    --split="test" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models_jigsaw/roberta_jigsaw_all/cda" \
    --batch_size=32 \
    --max_seq_length=512 \
    --num_examples=400 \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results_jigsaw/roberta_jigsaw_all_race_test_400/cda" \
    --bias_type="race"

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="google/jigsaw_unintended_bias" \
    --split="test" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models_jigsaw/roberta_jigsaw_race/dropout" \
    --batch_size=32 \
    --max_seq_length=512 \
    --num_examples=400 \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results_jigsaw/roberta_jigsaw_race_race_test_400/dropout" \
    --bias_type="race"

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="google/jigsaw_unintended_bias" \
    --split="test" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models_jigsaw/roberta_jigsaw_all/dropout" \
    --batch_size=32 \
    --max_seq_length=512 \
    --num_examples=400 \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results_jigsaw/roberta_jigsaw_all_race_test_400/dropout" \
    --bias_type="race"

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="google/jigsaw_unintended_bias" \
    --split="test" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models_jigsaw/roberta_jigsaw_race/attention_entropy" \
    --batch_size=32 \
    --max_seq_length=512 \
    --num_examples=400 \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results_jigsaw/roberta_jigsaw_race_race_test_400/attention_entropy" \
    --bias_type="race"

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="google/jigsaw_unintended_bias" \
    --split="test" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models_jigsaw/roberta_jigsaw_all/attention_entropy" \
    --batch_size=32 \
    --max_seq_length=512 \
    --num_examples=400 \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results_jigsaw/roberta_jigsaw_all_race_test_400/attention_entropy" \
    --bias_type="race"

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="google/jigsaw_unintended_bias" \
    --split="test" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models_jigsaw/roberta_jigsaw_race/causal_debias" \
    --batch_size=32 \
    --max_seq_length=512 \
    --num_examples=400 \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results_jigsaw/roberta_jigsaw_race_race_test_400/causal_debias" \
    --bias_type="race"

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="google/jigsaw_unintended_bias" \
    --split="test" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models_jigsaw/roberta_jigsaw_all/causal_debias" \
    --batch_size=32 \
    --max_seq_length=512 \
    --num_examples=400 \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results_jigsaw/roberta_jigsaw_all_race_test_400/causal_debias" \
    --bias_type="race"

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="google/jigsaw_unintended_bias" \
    --split="test" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models_jigsaw/distilbert_jigsaw_race/no_debiasing" \
    --batch_size=32 \
    --max_seq_length=512 \
    --num_examples=400 \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results_jigsaw/distilbert_jigsaw_race_race_test_400/no_debiasing" \
    --bias_type="race"

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="google/jigsaw_unintended_bias" \
    --split="test" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models_jigsaw/distilbert_jigsaw_all/no_debiasing" \
    --batch_size=32 \
    --max_seq_length=512 \
    --num_examples=400 \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results_jigsaw/distilbert_jigsaw_all_race_test_400/no_debiasing" \
    --bias_type="race"

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="google/jigsaw_unintended_bias" \
    --split="test" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models_jigsaw/distilbert_jigsaw_race/group_balance" \
    --batch_size=32 \
    --max_seq_length=512 \
    --num_examples=400 \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results_jigsaw/distilbert_jigsaw_race_race_test_400/group_balance" \
    --bias_type="race"

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="google/jigsaw_unintended_bias" \
    --split="test" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models_jigsaw/distilbert_jigsaw_all/group_balance" \
    --batch_size=32 \
    --max_seq_length=512 \
    --num_examples=400 \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results_jigsaw/distilbert_jigsaw_all_race_test_400/group_balance" \
    --bias_type="race"

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="google/jigsaw_unintended_bias" \
    --split="test" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models_jigsaw/distilbert_jigsaw_race/group_class_balance" \
    --batch_size=32 \
    --max_seq_length=512 \
    --num_examples=400 \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results_jigsaw/distilbert_jigsaw_race_race_test_400/group_class_balance" \
    --bias_type="race"

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="google/jigsaw_unintended_bias" \
    --split="test" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models_jigsaw/distilbert_jigsaw_all/group_class_balance" \
    --batch_size=32 \
    --max_seq_length=512 \
    --num_examples=400 \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results_jigsaw/distilbert_jigsaw_all_race_test_400/group_class_balance" \
    --bias_type="race"

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="google/jigsaw_unintended_bias" \
    --split="test" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models_jigsaw/distilbert_jigsaw_race/cda" \
    --batch_size=32 \
    --max_seq_length=512 \
    --num_examples=400 \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results_jigsaw/distilbert_jigsaw_race_race_test_400/cda" \
    --bias_type="race"

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="google/jigsaw_unintended_bias" \
    --split="test" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models_jigsaw/distilbert_jigsaw_all/cda" \
    --batch_size=32 \
    --max_seq_length=512 \
    --num_examples=400 \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results_jigsaw/distilbert_jigsaw_all_race_test_400/cda" \
    --bias_type="race"

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="google/jigsaw_unintended_bias" \
    --split="test" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models_jigsaw/distilbert_jigsaw_race/dropout" \
    --batch_size=32 \
    --max_seq_length=512 \
    --num_examples=400 \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results_jigsaw/distilbert_jigsaw_race_race_test_400/dropout" \
    --bias_type="race"

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="google/jigsaw_unintended_bias" \
    --split="test" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models_jigsaw/distilbert_jigsaw_all/dropout" \
    --batch_size=32 \
    --max_seq_length=512 \
    --num_examples=400 \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results_jigsaw/distilbert_jigsaw_all_race_test_400/dropout" \
    --bias_type="race"

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="google/jigsaw_unintended_bias" \
    --split="test" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models_jigsaw/distilbert_jigsaw_race/attention_entropy" \
    --batch_size=32 \
    --max_seq_length=512 \
    --num_examples=400 \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results_jigsaw/distilbert_jigsaw_race_race_test_400/attention_entropy" \
    --bias_type="race"

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="google/jigsaw_unintended_bias" \
    --split="test" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models_jigsaw/distilbert_jigsaw_all/attention_entropy" \
    --batch_size=32 \
    --max_seq_length=512 \
    --num_examples=400 \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results_jigsaw/distilbert_jigsaw_all_race_test_400/attention_entropy" \
    --bias_type="race"

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="google/jigsaw_unintended_bias" \
    --split="test" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models_jigsaw/distilbert_jigsaw_race/causal_debias" \
    --batch_size=32 \
    --max_seq_length=512 \
    --num_examples=400 \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results_jigsaw/distilbert_jigsaw_race_race_test_400/causal_debias" \
    --bias_type="race"

python -m fairness_evaluation.compute_fairness_results \
    --dataset_name="google/jigsaw_unintended_bias" \
    --split="test" \
    --model_dir="/scratch/yifwang/new_fairness_x_explainability/new_debiased_models_jigsaw/distilbert_jigsaw_all/causal_debias" \
    --batch_size=32 \
    --max_seq_length=512 \
    --num_examples=400 \
    --output_dir="/scratch/yifwang/new_fairness_x_explainability/encoder_results_jigsaw/distilbert_jigsaw_all_race_test_400/causal_debias" \
    --bias_type="race"


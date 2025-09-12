data = "jigsaw"  # jigsaw or civil
splits = ["val"]
bias_types = ["race", "gender", "religion"]
model_types = ["bert", "roberta"]
explanation_debiasing_methods = ["Saliency", "InputXGradient", "raw_attention", "attention_flow", "attention_rollout", "Occlusion"]
aggregation_methods = ["L1", "L2"]
alphas = [100.0, 10.0, 1.0, 0.1, 0.01] # [100.0, 10.0, 1.0, 0.1, 0.01] # [1000.0, 100.0, 10.0, 1.0, 0.1, 0.01, 0.001]
alphas = [float(alpha) for alpha in alphas]  # Ensure alphas are floats
if data == "civil":
    num_examples_dict_test = {"race": 2000, "gender": 2000, "religion": 1000}
elif data == "jigsaw":
    num_examples_dict_test = {"race": 400, "gender": 800, "religion": 200}
if data == "civil":
    num_examples_dict_val = {"race": 500, "gender": 500, "religion": 200}
elif data == "jigsaw":
    num_examples_dict_val = {"race": 200, "gender": 200, "religion": 200}
model_name_or_path_dict = {
    "bert": "bert-base-uncased",
    "roberta": "FacebookAI/roberta-base",
    "distilbert": "distilbert/distilbert-base-uncased"}
if data == "civil":
    dataset_name = "lighteval/civil_comments_helm"
elif data == "jigsaw":
    dataset_name = "google/jigsaw_unintended_bias"
num_labels = 2
seeds = [1,2]  # [42, 1, 2]
batch_size = 64
max_seq_length = 512
split_ratio = "8, 2"
learning_rate = 2e-5
warmup_steps_or_ratio = 0.1
num_train_epochs = 5
early_stopping_patience = -1
eval_steps = 1000
save_steps = 1000

eval_metric = "accuracy"
n_steps=50

model_dir_root = f"/scratch/yifwang/fairness_x_explainability/explanation_debiased_models_{data}"
output_dir_root = f"/scratch/yifwang/fairness_x_explainability/bias_mitigation_results_{data}"


for bias_type in bias_types:
    for split in splits:
        if split == "test":
            num_examples = num_examples_dict_test[bias_type]
        else:
            num_examples = num_examples_dict_val[bias_type]
        bash_script = f"/nethome/yifwang/fairness_x_explainability/new_fairness_explainability/bias_mitigation_scripts_{data}/compute_fairness_{bias_type}.sh"
        with open(bash_script, "w") as f:
            f.write("#!/bin/bash\n\n")
            for model_type in model_types:
                for explanation_debiasing_method in explanation_debiasing_methods:
                    for alpha in alphas:
                        if explanation_debiasing_method in ["raw_attention", "attention_flow", "attention_rollout", "Occlusion"]:
                            # skip iteration for aggregation methods for these explanation methods
                            aggregation_methods = ["L1"]  # These methods have only one aggregation method
                        else:
                            aggregation_methods = ["L1", "L2"]
                        for aggregation_method in aggregation_methods:
                            for seed in seeds:
                                # write the command to run fairness evaluation
                                if explanation_debiasing_method in ["raw_attention", "attention_flow", "attention_rollout", "Occlusion"]:
                                    model_dir= f"{model_dir_root}/{model_type}_{data}_{bias_type}_{seed}/{explanation_debiasing_method}/{alpha}"
                                    output_dir = f"{output_dir_root}/{model_type}_{data}_{bias_type}_{bias_type}_{split}_{num_examples}_{seed}/{explanation_debiasing_method}/{alpha}"
                                else:
                                    model_dir= f"{model_dir_root}/{model_type}_{data}_{bias_type}_{seed}/{explanation_debiasing_method}/{aggregation_method}_{alpha}"
                                    output_dir = f"{output_dir_root}/{model_type}_{data}_{bias_type}_{bias_type}_{split}_{num_examples}_{seed}/{explanation_debiasing_method}/{aggregation_method}_{alpha}"
                                f.write(f"python -m fairness_evaluation.compute_fairness_results \\\n")
                                f.write(f"    --dataset_name=\"{dataset_name}\" \\\n")
                                f.write(f"    --split=\"{split}\" \\\n")
                                f.write(f"    --split_ratio=\"{split_ratio}\" \\\n")
                                f.write(f"    --model_dir=\"{model_dir}\" \\\n")
                                f.write(f"    --batch_size={batch_size} \\\n")
                                f.write(f"    --max_seq_length={max_seq_length} \\\n")
                                f.write(f"    --num_examples={num_examples} \\\n")
                                f.write(f"    --output_dir=\"{output_dir}\" \\\n")
                                f.write(f"    --bias_type=\"{bias_type}\" \n\n")

                                # if explanation_debiasing_method in ["raw_attention", "attention_flow", "attention_rollout", "Occlusion"]:
                                #     model_dir= f"{model_dir_root}/{model_type}_{data}_all_{seed}/{explanation_debiasing_method}/{alpha}"
                                #     output_dir = f"{output_dir_root}/{model_type}_{data}_all_{bias_type}_{split}_{num_examples}_{seed}/{explanation_debiasing_method}/{alpha}"
                                # else:
                                #     model_dir= f"{model_dir_root}/{model_type}_{data}_all_{seed}/{explanation_debiasing_method}/{aggregation_method}_{alpha}"
                                #     output_dir = f"{output_dir_root}/{model_type}_{data}_all_{bias_type}_{split}_{num_examples}_{seed}/{explanation_debiasing_method}/{aggregation_method}_{alpha}"
                                # f.write(f"python -m fairness_evaluation.compute_fairness_results \\\n")
                                # f.write(f"    --dataset_name=\"{dataset_name}\" \\\n")
                                # f.write(f"    --split=\"test\" \\\n")
                                # f.write(f"    --split_ratio=\"{split_ratio}\" \\\n")
                                # f.write(f"    --model_dir=\"{model_dir}\" \\\n")
                                # f.write(f"    --batch_size={batch_size} \\\n")
                                # f.write(f"    --max_seq_length={max_seq_length} \\\n")
                                # f.write(f"    --num_examples={num_examples} \\\n")
                                # f.write(f"    --output_dir=\"{output_dir}\" \\\n")
                                # f.write(f"    --bias_type=\"{bias_type}\" \n\n")
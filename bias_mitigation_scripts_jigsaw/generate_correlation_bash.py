data = "jigsaw"  # jigsaw or civil
splits = ["test"]
bias_types = ["race"] # ["race", "gender", "religion"]
model_types = ["bert"] # ["bert", "roberta", "distilbert"]
explanation_debiasing_methods = ["Saliency", "InputXGradient", "IntegratedGradients", "raw_attention", "attention_flow", "attention_rollout", "Occlusion"]
aggregation_methods = ["L1", "L2"]
alphas = [1.0, 0.1, 0.01, 0.001, 0.0001] # [100.0, 10.0, 1.0, 0.1, 0.01]
alphas = [float(alpha) for alpha in alphas]  # Ensure alphas are floats
if data == "civil":
    num_examples_dict_test = {"race": 2000, "gender": 2000, "religion": 1000}
elif data == "jigsaw":
    num_examples_dict_test = {"race": 400, "gender": 800, "religion": 200}
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
batch_size = 64
max_seq_length = 512
split_ratio = "8, 2"
learning_rate = 2e-5
warmup_steps_or_ratio = 0.1
num_train_epochs = 5
early_stopping_patience = -1
eval_steps = 1000
save_steps = 1000
seed = 42
eval_metric = "accuracy"
n_steps=50

model_dir_root = f"/scratch/yifwang/fairness_x_explainability/explanation_debiased_models_{data}"
output_dir_root = f"/scratch/yifwang/fairness_x_explainability/bias_mitigation_results_{data}"
explanation_methods = "Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"
# explanation_methods = "Occlusion"


if __name__ == "__main__":
    for bias_type in bias_types:
        for split in splits:
            num_examples = num_examples_dict_test[bias_type] if split == "test" else num_examples_dict_val[bias_type]
            bash_script = f"/nethome/yifwang/fairness_x_explainability/new_fairness_explainability/bias_mitigation_scripts_{data}/compute_correlation_{bias_type}_{split}.sh"
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
                                # write the command to run fairness evaluation
                                if explanation_debiasing_method in ["raw_attention", "attention_flow", "attention_rollout", "Occlusion"]:
                                    output_dir = f"{output_dir_root}/{model_type}_{data}_all_{bias_type}_{split}_{num_examples}/{explanation_debiasing_method}/{alpha}"
                                else:
                                    output_dir = f"{output_dir_root}/{model_type}_{data}_all_{bias_type}_{split}_{num_examples}/{explanation_debiasing_method}/{aggregation_method}_{alpha}"
                                f.write(f"python -m bias_correlation.compute_bias_correlation \\\n")
                                f.write(f"    --results_dir=\"{output_dir}\" \\\n")
                                f.write(f"    --split=\"{split}\" \\\n")
                                f.write(f"    --bias_type=\"{bias_type}\" \\\n")
                                f.write(f"    --methods=\"{explanation_methods}\"\n\n")

                                f.write(f"python -m visualization.visualize_bias_correlation \\\n")
                                f.write(f"    --results_dir=\"{output_dir}\" \\\n")
                                f.write(f"    --split=\"{split}\" \\\n")
                                f.write(f"    --bias_type=\"{bias_type}\" \\\n")
                                f.write(f"    --methods=\"{explanation_methods}\"\n\n")

                                if explanation_debiasing_method in ["raw_attention", "attention_flow", "attention_rollout", "Occlusion"]:
                                    output_dir = f"{output_dir_root}/{model_type}_{data}_all_{bias_type}_{split}_{num_examples}/{explanation_debiasing_method}/{alpha}"
                                else:
                                    output_dir = f"{output_dir_root}/{model_type}_{data}_all_{bias_type}_{split}_{num_examples}/{explanation_debiasing_method}/{aggregation_method}_{alpha}"
                                f.write(f"python -m bias_correlation.compute_bias_correlation \\\n")
                                f.write(f"    --results_dir=\"{output_dir}\" \\\n")
                                f.write(f"    --split=\"{split}\" \\\n")
                                f.write(f"    --bias_type=\"{bias_type}\" \\\n")
                                f.write(f"    --methods=\"{explanation_methods}\"\n\n")
                        
                                f.write(f"python -m visualization.visualize_bias_correlation \\\n")
                                f.write(f"    --results_dir=\"{output_dir}\" \\\n")
                                f.write(f"    --split=\"{split}\" \\\n")
                                f.write(f"    --bias_type=\"{bias_type}\" \\\n")
                                f.write(f"    --methods=\"{explanation_methods}\"\n\n")
                        

                



data = "civil" # jigsaw or civil
splits = ["test"]
bias_types = ["race", "gender", "religion"]
model_types = ["bert", "roberta", "distilbert"]
debiasing_methods = ["no_debiasing", "group_balance", "group_class_balance", "cda", "dropout", "attention_entropy", "causal_debias"]
# debiasing_methods = ["dropout", "attention_entropy", "causal_debias"]
if data == "civil":
    num_examples_dict_test = {"race": 2000, "gender": 2000, "religion": 1000}
elif data == "jigsaw":
    num_examples_dict_test = {"race": 400, "gender": 800, "religion": 200}
num_examples_dict_val = {"race": 200, "gender": 200, "religion": 200}

if data == "civil":
    dataset_name = "lighteval/civil_comments_helm"
elif data == "jigsaw":
    dataset_name = "google/jigsaw_unintended_bias"
num_labels = 2
batch_size = 32
max_seq_length = 512
split_ratio = "8, 2"

model_dir_root = f"/scratch/yifwang/fairness_x_explainability/debiased_models_{data}"
output_dir_root = f"/scratch/yifwang/fairness_x_explainability/encoder_results_{data}"
# explanation_methods = "Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"
explanation_methods = "Occlusion"
import os

if __name__ == "__main__":
    for bias_type in bias_types:
        for split in splits:
            num_examples = num_examples_dict_test[bias_type] if split == "test" else num_examples_dict_val[bias_type]
            bash_script = f"/nethome/yifwang/fairness_x_explainability/new_fairness_explainability/encoder_scripts_{data}/compute_sensitive_feature_reliance_{bias_type}_{split}.sh"
            with open(bash_script, "w") as f:
                f.write("#!/bin/bash\n\n")
                for model_type in model_types:
                    for debiasing_method in debiasing_methods:
                        output_dir = f"{output_dir_root}/{model_type}_{data}_{bias_type}_{bias_type}_{split}_{num_examples}/{debiasing_method}"
                        f.write(f"python -m sensitive_attribution_extraction.extract_sensitive_token_attribution \\\n")
                        f.write(f"    --results_dir=\"{output_dir}\" \\\n")
                        f.write(f"    --split=\"{split}\" \\\n")
                        f.write(f"    --bias_type=\"{bias_type}\" \\\n")
                        f.write(f"    --methods=\"{explanation_methods}\"\n\n")

                        f.write(f"python -m model_selection.compute_reliance_statistics \\\n")
                        f.write(f"    --results_dir=\"{output_dir}\" \\\n")
                        f.write(f"    --split=\"{split}\" \\\n")
                        f.write(f"    --bias_type=\"{bias_type}\" \\\n")
                        f.write(f"    --methods=\"{explanation_methods}\"\n\n")
                        
                        output_dir = f"{output_dir_root}/{model_type}_{data}_all_{bias_type}_{split}_{num_examples}/{debiasing_method}"
                        f.write(f"python -m sensitive_attribution_extraction.extract_sensitive_token_attribution \\\n")
                        f.write(f"    --results_dir=\"{output_dir}\" \\\n")
                        f.write(f"    --split=\"{split}\" \\\n")
                        f.write(f"    --bias_type=\"{bias_type}\" \\\n")
                        f.write(f"    --methods=\"{explanation_methods}\"\n\n")

                        f.write(f"python -m model_selection.compute_reliance_statistics \\\n")
                        f.write(f"    --results_dir=\"{output_dir}\" \\\n")
                        f.write(f"    --split=\"{split}\" \\\n")
                        f.write(f"    --bias_type=\"{bias_type}\" \\\n")
                        f.write(f"    --methods=\"{explanation_methods}\"\n\n")
                        

                



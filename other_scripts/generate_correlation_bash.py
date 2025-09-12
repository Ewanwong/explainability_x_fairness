
splits = ["test", "val"]
bias_types = ["race", "gender", "religion"]
model_types = ["bert", "roberta", "distilbert"]
#debiasing_methods = ["no_debiasing", "group_balance", "group_class_balance", "cda", "dropout", "attention_entropy", "causal_debias"]
debiasing_methods = ["dropout", "attention_entropy", "causal_debias"]
num_examples_dict_test = {"race": 2000, "gender": 2000, "religion": 1000}
num_examples_dict_val = {"race": 200, "gender": 200, "religion": 200}

dataset_name = "lighteval/civil_comments_helm"
num_labels = 2
batch_size = 32
max_seq_length = 512
split_ratio = "8, 2"

model_dir_root = "/scratch/yifwang/new_fairness_x_explainability/new_debiased_models"
output_dir_root = "/scratch/yifwang/new_fairness_x_explainability/encoder_results"
explanation_methods = "Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"
    
import os

if __name__ == "__main__":
    for bias_type in bias_types:
        for split in splits:
            num_examples = num_examples_dict_test[bias_type] if split == "test" else num_examples_dict_val[bias_type]
            bash_script = f"/nethome/yifwang/fairness_x_explainability/new_fairness_explainability/other_scripts/compute_correlation_{bias_type}_{split}.sh"
            with open(bash_script, "w") as f:
                f.write("#!/bin/bash\n")
                for model_type in model_types:
                    for debiasing_method in debiasing_methods:
                        """
                        output_dir = f"{output_dir_root}/{model_type}_civil_{bias_type}_{bias_type}_{split}_{num_examples}/{debiasing_method}"
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
                        """
                        output_dir = f"{output_dir_root}/{model_type}_civil_all_{bias_type}_{split}_{num_examples}/{debiasing_method}"
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
                        

                



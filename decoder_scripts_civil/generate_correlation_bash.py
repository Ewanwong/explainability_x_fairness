data = "civil" # jigsaw or civil
splits = ["test"]
bias_types = ["race", "gender", "religion"] # ["race", "gender", "religion"]
model_types = ["llama", "qwen3"] #["llama", "qwen", "qwen3"]
model_size = "3b" # 7b / 8b
model_names = {"llama": "meta-llama/Llama-3.2-3B-Instruct", "qwen": "Qwen/Qwen2.5-3B-Instruct", "qwen3": "Qwen/Qwen3-4B"} if model_size == "3b" else {"llama": "meta-llama/Llama-3.1-8B-Instruct", "qwen": "Qwen/Qwen2.5-7B-Instruct", "qwen3": "Qwen/Qwen3-8B"}
param_num = {"llama": "3b", "qwen": "3b", "qwen3": "4b"} if model_size == "3b" else {"llama": "8b", "qwen": "7b", "qwen3": "8b"}
prompt_types = ["few_shot", "fairness_imagination", "fairness_instruction"] # ["zero_shot", "few_shot", "fairness_imagination", "fairness_instruction"]
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
batch_size = 1
max_seq_length = 512
split_ratio = "8, 2"

output_dir_root = f"/scratch/yifwang/fairness_x_explainability/decoder_results_{data}"
#explanation_methods = "Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"
explanation_methods = "IntegratedGradients"
    
import os

if __name__ == "__main__":
    for bias_type in bias_types:
        for split in splits:
            num_examples = num_examples_dict_test[bias_type] if split == "test" else num_examples_dict_val[bias_type]
            bash_script = f"/nethome/yifwang/fairness_x_explainability/new_fairness_explainability/decoder_scripts_{data}/compute_correlation_{bias_type}_{split}.sh"
            with open(bash_script, "w") as f:
                f.write("#!/bin/bash\n")
                for model_type in model_types:
                    for prompt_type in prompt_types:
                        output_dir = f"{output_dir_root}/{model_type}_{param_num[model_type]}_{data}_{bias_type}_{split}_{num_examples}/{prompt_type}"
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
                        

                



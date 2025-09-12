data = "civil" # jigsaw or civil
splits = ["test"]
bias_types = ["race", "gender", "religion"] # ["race", "gender", "religion"]
model_types = ["llama", "qwen", "qwen3"]
model_size = "3b" # 7b / 8b
model_names = {"llama": "meta-llama/Llama-3.2-3B-Instruct", "qwen": "Qwen/Qwen2.5-3B-Instruct", "qwen3": "Qwen/Qwen3-4B"} if model_size == "3b" else {"llama": "meta-llama/Llama-3.1-8B-Instruct", "qwen": "Qwen/Qwen2.5-7B-Instruct", "qwen3": "Qwen/Qwen3-8B"}
param_num = {"llama": "3b", "qwen": "3b", "qwen3": "4b"} if model_size == "3b" else {"llama": "8b", "qwen": "7b", "qwen3": "8b"}
prompt_types = ["fairness_imagination", "fairness_instruction"] # ["zero_shot", "few_shot", "fairness_imagination", "fairness_instruction"]
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

import os

if __name__ == "__main__":
    for bias_type in bias_types:
        for split in splits:
            num_examples = num_examples_dict_test[bias_type] if split == "test" else num_examples_dict_val[bias_type]
            bash_script = f"/nethome/yifwang/fairness_x_explainability/new_fairness_explainability/decoder_scripts_{data}/compute_fairness_{bias_type}_{split}.sh"
            with open(bash_script, "w") as f:
                f.write("#!/bin/bash\n")
                for model_type in model_types:
                    for prompt_type in prompt_types:
                        model_dir = model_names[model_type]
                        output_dir = f"{output_dir_root}/{model_type}_{param_num[model_type]}_{data}_{bias_type}_{split}_{num_examples}/{prompt_type}"
                        f.write(f"python -m fairness_evaluation.compute_fairness_results \\\n")
                        f.write(f"    --dataset_name=\"{dataset_name}\" \\\n")
                        f.write(f"    --split=\"{split}\" \\\n")
                        f.write(f"    --split_ratio=\"{split_ratio}\" \\\n")
                        f.write(f"    --model_dir=\"{model_dir}\" \\\n")
                        f.write(f"    --batch_size={batch_size} \\\n")
                        f.write(f"    --max_seq_length={max_seq_length} \\\n")
                        f.write(f"    --num_examples={num_examples} \\\n")
                        f.write(f"    --output_dir=\"{output_dir}\" \\\n")
                        f.write(f"    --prompt_type=\"{prompt_type}\" \\\n")
                        f.write(f"    --bias_type=\"{bias_type}\"\n\n")



                



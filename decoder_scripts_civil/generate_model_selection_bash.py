data = "civil" # jigsaw or civil
splits = ["val"]
bias_types = ["race", "gender", "religion"] # ["race", "gender", "religion"]
model_types = ["llama", "qwen3"]
model_size = "3b" # 7b / 8b
model_names = {"llama": "meta-llama/Llama-3.2-3B-Instruct", "qwen": "Qwen/Qwen2.5-3B-Instruct", "qwen3": "Qwen/Qwen3-4B"} if model_size == "3b" else {"llama": "meta-llama/Llama-3.1-8B-Instruct", "qwen": "Qwen/Qwen2.5-7B-Instruct", "qwen3": "Qwen/Qwen3-8B"}
param_num = {"llama": "3b", "qwen": "3b", "qwen3": "4b"} if model_size == "3b" else {"llama": "8b", "qwen": "7b", "qwen3": "8b"}
prompt_types = ["zero_shot", "few_shot", "fairness_imagination", "fairness_instruction"]
if data == "civil":
    num_examples_dict_test = {"race": 2000, "gender": 2000, "religion": 1000}
elif data == "jigsaw":
    num_examples_dict_test = {"race": 400, "gender": 800, "religion": 200}
num_examples_dict_val_list = [{"race": 20, "gender": 20, "religion": 20}, {"race": 50, "gender": 50, "religion": 50}, {"race": 100, "gender": 100, "religion": 100},{"race": 200, "gender": 200, "religion": 200}, {"race": 500, "gender": 500}]
seeds = {20: list(range(1, 6))+[42], 50: list(range(1, 4))+[42], 100: [1,2,42], 200: [42], 500: [42]}
if data == "civil":
    dataset_name = "lighteval/civil_comments_helm"
elif data == "jigsaw":
    dataset_name = "google/jigsaw_unintended_bias"
num_labels = 2
batch_size = 1
max_seq_length = 512
split_ratio = "8, 2"

test_output_dir_root = f"/scratch/yifwang/fairness_x_explainability/decoder_results_{data}"
val_output_dir_root = f"/scratch/yifwang/fairness_x_explainability/decoder_results_{data}/val_results"
# explanation_methods = "Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion, IntegratedGradients"
explanation_methods = "Attention, Saliency, DeepLift, InputXGradient, KernelShap, Occlusion"
import os

if __name__ == "__main__":
    for bias_type in bias_types:
        bash_script = f"/nethome/yifwang/fairness_x_explainability/new_fairness_explainability/decoder_scripts_{data}/model_selection_{bias_type}.sh"
        with open(bash_script, "w") as f:
            f.write("#!/bin/bash\n")
            
            for num_examples_dict_val in num_examples_dict_val_list:
                if bias_type not in num_examples_dict_val:
                    continue
                num_examples_val = num_examples_dict_val[bias_type]
                num_examples_test = num_examples_dict_test[bias_type]
                for seed in seeds[num_examples_val]:
                    for model_type in model_types:
                        for prompt_type in prompt_types:
                            # first fairness evaluation
                            model_dir = model_names[model_type]
                            output_dir = f"{val_output_dir_root}/{model_type}_{param_num[model_type]}_{data}_{bias_type}_val_{num_examples_val}_{seed}/{prompt_type}"
                            f.write(f"python -m fairness_evaluation.compute_fairness_results \\\n")
                            f.write(f"    --dataset_name=\"{dataset_name}\" \\\n")
                            f.write(f"    --split=\"val\" \\\n")
                            f.write(f"    --split_ratio=\"{split_ratio}\" \\\n")
                            f.write(f"    --model_dir=\"{model_dir}\" \\\n")
                            f.write(f"    --batch_size={batch_size} \\\n")
                            f.write(f"    --max_seq_length={max_seq_length} \\\n")
                            f.write(f"    --num_examples={num_examples_val} \\\n")
                            f.write(f"    --output_dir=\"{output_dir}\" \\\n")
                            f.write(f"    --bias_type=\"{bias_type}\"\\\n")
                            f.write(f"    --prompt_type=\"{prompt_type}\" \\\n")
                            f.write(f"    --shuffle \\\n")
                            f.write(f"    --seed={seed}\n\n")
                            
                            # run the explanation generation

                            model_dir = model_names[model_type]
                            output_dir = f"{val_output_dir_root}/{model_type}_{param_num[model_type]}_{data}_{bias_type}_val_{num_examples_val}_{seed}/{prompt_type}"
                            f.write(f"python -m explanation_generation.gen_explanation_decoder \\\n")
                            f.write(f"    --dataset_name=\"{dataset_name}\" \\\n")
                            f.write(f"    --split=\"val\" \\\n")
                            f.write(f"    --split_ratio=\"{split_ratio}\" \\\n")
                            f.write(f"    --model_dir=\"{model_dir}\" \\\n")
                            f.write(f"    --num_labels={num_labels} \\\n")
                            f.write(f"    --batch_size={batch_size} \\\n")
                            f.write(f"    --max_length={max_seq_length} \\\n")
                            f.write(f"    --num_examples={num_examples_val} \\\n")
                            f.write(f"    --methods=\"{explanation_methods}\" \\\n")
                            f.write(f"    --output_dir=\"{output_dir}\" \\\n")
                            f.write(f"    --baseline=\"pad\" \\\n")
                            f.write(f"    --bias_type=\"{bias_type}\" \\\n")
                            f.write(f"    --prompt_type=\"{prompt_type}\" \\\n")
                            f.write(f"    --only_predicted_class \\\n")
                            f.write(f"    --shuffle \\\n")
                            f.write(f"    --seed={seed}\n\n")

                           
                            # run the attribution extraction
                                                    
                            output_dir = f"{val_output_dir_root}/{model_type}_{param_num[model_type]}_{data}_{bias_type}_val_{num_examples_val}_{seed}/{prompt_type}"
                            f.write(f"python -m sensitive_attribution_extraction.extract_sensitive_token_attribution \\\n")
                            f.write(f"    --results_dir=\"{output_dir}\" \\\n")
                            f.write(f"    --split=\"val\" \\\n")
                            f.write(f"    --bias_type=\"{bias_type}\" \\\n")
                            f.write(f"    --methods=\"{explanation_methods}\"\n\n")

                            f.write(f"python -m model_selection.compute_reliance_statistics \\\n")
                            f.write(f"    --results_dir=\"{output_dir}\" \\\n")
                            f.write(f"    --split=\"val\" \\\n")
                            f.write(f"    --bias_type=\"{bias_type}\" \\\n")
                            f.write(f"    --methods=\"{explanation_methods}\"\n\n")
                            

                        # run the model selection
                        f.write(f"python -m model_selection.compute_model_selection_correlation \\\n")
                        f.write(f"    --results_dir=\"{test_output_dir_root}\" \\\n")
                        f.write(f"    --val_results_dir=\"{val_output_dir_root}\" \\\n")
                        f.write(f"    --methods=\"{explanation_methods}\" \\\n")
                        f.write(f"    --experiment_type=decoder \\\n")
                        f.write(f"    --bias_type=\"{bias_type}\" \\\n")
                        f.write(f"    --model_type=\"{model_type}_{param_num[model_type]}\" \\\n")
                        f.write(f"    --num_test_examples={num_examples_test} \\\n")
                        f.write(f"    --num_val_examples={num_examples_val} \\\n")
                        f.write(f"    --test_seed=-1 \\\n")
                        f.write(f"    --val_seed={seed}\n\n")

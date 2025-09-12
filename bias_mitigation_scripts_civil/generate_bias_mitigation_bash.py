data = "civil"  # jigsaw or civil
splits = ["test"]
bias_types = ["all"] # ["race", "gender", "religion", "all"] 
model_types = ["bert", "roberta", "distilbert"]
explanation_debiasing_methods = ["Saliency", "InputXGradient", "raw_attention", "attention_flow", "attention_rollout", "Occlusion"]
aggregation_methods = ["L1", "L2"]

alphas = [100.0, 10.0, 1.0, 0.1, 0.01] # [100.0, 10.0, 1.0, 0.1, 0.01] # [1000.0, 100.0, 10.0, 1.0, 0.1, 0.01, 0.001]
alphas = [float(alpha) for alpha in alphas]  # Ensure alphas are floats

seeds = [1, 2] # [42, 1, 2]

num_examples_dict = {"race": 8000, "gender": 8000, "religion": 6300, "all":"8000,8000,6300"}
model_name_or_path_dict = {
    "bert": "bert-base-uncased",
    "roberta": "FacebookAI/roberta-base",
    "distilbert": "distilbert/distilbert-base-uncased"}
if data == "civil":
    dataset_name = "lighteval/civil_comments_helm"
elif data == "jigsaw":
    dataset_name = "google/jigsaw_unintended_bias"
num_labels = 2

batch_size=8

batch_size = {expl: 8 for expl in explanation_debiasing_methods}  # Adjust batch size for each explanation method
batch_size["IntegratedGradients"] = 4  # Set a smaller batch size for IntegratedGradients due to memory constraints
gradient_accumulation_steps = {expl: 1 for expl in explanation_debiasing_methods}  # Adjust gradient accumulation steps for each explanation method
gradient_accumulation_steps["IntegratedGradients"] = 2  # Set a smaller gradient

max_seq_length = 512
split_ratio = "8, 2"
learning_rate = 2e-5
warmup_steps_or_ratio = 0.1
num_train_epochs = 5
early_stopping_patience = -1
eval_steps = 1000
save_steps = 1000
# seed = 42
eval_metric = "accuracy"
n_steps=20

model_dir_root = f"/scratch/yifwang/fairness_x_explainability/explanation_debiased_models_{data}"
output_dir_root = f"/scratch/yifwang/fairness_x_explainability/bias_mitigation_results_{data}"


for bias_type in bias_types:
    for split in splits:
        bias_type_detail = bias_type if bias_type != "all" else "race,gender,religion"
        num_examples = num_examples_dict[bias_type]
        bash_script = f"/nethome/yifwang/fairness_x_explainability/new_fairness_explainability/bias_mitigation_scripts_{data}/bias_mitigation_{bias_type}.sh"
        with open(bash_script, "w") as f:
            f.write("#!/bin/bash\n\n")
            for model_type in model_types:
                for seed in seeds:
                    model_dir = f"{model_dir_root}/{model_type}_{data}_{bias_type}_{seed}"
                    for explanation_debiasing_method in explanation_debiasing_methods:
                        for alpha in alphas:
                            if explanation_debiasing_method in ["raw_attention", "attention_flow", "attention_rollout", "Occlusion"]:
                                # skip iteration for aggregation methods for these explanation methods
                                aggregation_methods = ["L1"]  # These methods have only one aggregation method
                            else:
                                aggregation_methods = ["L1", "L2"]
                            for aggregation_method in aggregation_methods:
                                
                                f.write(f"python -m bias_mitigation.bias_mitigation \\\n")
                                f.write(f"    --model_name_or_path=\"{model_name_or_path_dict[model_type]}\" \\\n")
                                f.write(f"    --dataset_name=\"{dataset_name}\" \\\n")
                                f.write(f"    --num_labels={num_labels} \\\n")
                                f.write(f"    --output_dir=\"{model_dir}\" \\\n")
                                f.write(f"    --batch_size={batch_size[explanation_debiasing_method]} \\\n")  
                                f.write(f"    --gradient_accumulation_steps={gradient_accumulation_steps[explanation_debiasing_method]} \\\n")
                                f.write(f"    --max_seq_length={max_seq_length} \\\n")
                                f.write(f"    --learning_rate={learning_rate} \\\n")
                                f.write(f"    --warmup_steps_or_ratio={warmup_steps_or_ratio} \\\n")
                                f.write(f"    --num_train_epochs={num_train_epochs} \\\n")
                                f.write(f"    --early_stopping_patience={early_stopping_patience} \\\n")
                                f.write(f"    --eval_steps={eval_steps} \\\n")
                                f.write(f"    --save_steps={save_steps} \\\n")
                                f.write(f"    --split_ratio=\"{split_ratio}\" \\\n")
                                f.write(f"    --seed={seed} \\\n")
                                f.write(f"    --eval_metric=\"{eval_metric}\" \\\n")
                                f.write(f"    --bias_type=\"{bias_type_detail}\" \\\n")
                                f.write(f"    --total_num_examples={num_examples}\\\n")
                                f.write(f"    --explanation_method=\"{explanation_debiasing_method}\" \\\n")
                                f.write(f"    --aggregation=\"{aggregation_method}\" \\\n")
                                f.write(f"    --alpha={alpha} \\\n")
                                f.write(f"    --n_steps={n_steps} \n\n")
                                

                    
                
                    
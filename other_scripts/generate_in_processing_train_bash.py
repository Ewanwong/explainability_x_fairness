bias_types = ["race", "gender", "religion", "all"]
model_types = ["bert", "roberta", "distilbert"]
debiasing_methods = ["causal_debias"]
# debiasing_methods = ["dropout", "attention_entropy", "causal_debias"]
num_examples_dict = {"race": 8000, "gender": 8000, "religion": 6300, "all":"8000,8000,6300"}
model_name_or_path_dict = {
    "bert": "bert-base-uncased",
    "roberta": "FacebookAI/roberta-base",
    "distilbert": "distilbert/distilbert-base-uncased"}

dataset_name = "lighteval/civil_comments_helm"
num_labels = 2
batch_size = 8
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
entropy_weight = 0.1
causal_debias_weight = 0.5

model_dir_root = "/scratch/yifwang/new_fairness_x_explainability/new_debiased_models"
output_dir_root = "/scratch/yifwang/new_fairness_x_explainability/new_debiased_models_results"

import os

for bias_type in bias_types:

    num_examples = num_examples_dict[bias_type]
    bash_script = f"/nethome/yifwang/fairness_x_explainability/new_fairness_explainability/encoder_scripts/train_models_in_processing_{bias_type}.sh"
    with open(bash_script, "w") as f:
        f.write("#!/bin/bash\n")
        for model_type in model_types:
            for debiasing_method in debiasing_methods:
                model_dir = f"{model_dir_root}/{model_type}_civil_{bias_type}"
                bias_type_detail = bias_type if bias_type != "all" else "race,gender,religion"
                f.write(f"python -m model_selection.debiasing_inprocessing \\\n")
                f.write(f"    --model_name_or_path=\"{model_name_or_path_dict[model_type]}\" \\\n")
                f.write(f"    --dataset_name=\"{dataset_name}\" \\\n")
                f.write(f"    --num_labels={num_labels} \\\n")
                f.write(f"    --output_dir=\"{model_dir}\" \\\n")
                f.write(f"    --batch_size={batch_size} \\\n")  
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
                f.write(f"    --debiasing_method=\"{debiasing_method}\" \\\n")
                f.write(f"    --total_num_examples={num_examples}\\\n")
                f.write(f"    --entropy_weight={entropy_weight} \\\n")
                f.write(f"    --causal_debias_weight={causal_debias_weight} \n\n")
            
                

        



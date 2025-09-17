from explainer.Explainer_Decoder import BcosExplainer, AttentionExplainer, GradientNPropabationExplainer, OcclusionExplainer, ShapleyValueExplainer
from utils.utils import set_random_seed
from utils.dataset_utils import customized_load_dataset, customized_split_dataset, load_subsets_for_fairness_explainability
import argparse
import torch
from torch.utils.data import DataLoader, Subset
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM

import numpy as np
import json
import os
import random
from tqdm import tqdm

from utils.vocabulary import *
from utils.prompt import *

EXPLANATION_METHODS_MAPPING = {
    #"Bcos": BcosExplainer,
    "Attention": AttentionExplainer,
    "Saliency": GradientNPropabationExplainer,
    "DeepLift": GradientNPropabationExplainer,
    #"GuidedBackprop": GradientNPropabationExplainer,
    "InputXGradient": GradientNPropabationExplainer,
    "IntegratedGradients": GradientNPropabationExplainer,
    #"SIG": GradientNPropabationExplainer,
    "Occlusion": OcclusionExplainer,
    #"ShapleyValue": ShapleyValueExplainer,
    "KernelShap": ShapleyValueExplainer,
    #"Lime": LimeExplainer,
}

def main(args):

    # Set random seed for reproducibility
    set_random_seed(args.seed)
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if "civil" in args.dataset_name:
        dataset = "civil"
    elif "jigsaw" in args.dataset_name:
        dataset = "jigsaw"
    else:
        dataset = "civil"
        print("Make sure the dataset is either civil comments or jigsaw. Default: civil comments")

    # Load the model
    needs_attn = (args.methods is not None and "Attention" in args.methods) or args.methods is None
    model = AutoModelForCausalLM.from_pretrained(args.model_dir, output_attentions=needs_attn, device_map="auto")
    model.config.use_cache = False
    model.eval()
    #model.to(device)

    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    template = TEMPLATE
    if args.prompt_type == "zero_shot":
        system_prompt = construct_zero_shot_prompt(dataset, args.bias_type)
    elif args.prompt_type == "few_shot":
        system_prompt = construct_few_shot_prompt(dataset, args.bias_type)
    elif args.prompt_type == "fairness_imagination":
        system_prompt = construct_fairness_imagination_prompt(dataset, args.bias_type)
    elif args.prompt_type == "fairness_instruction":
        system_prompt = construct_fairness_instruction_prompt(dataset, args.bias_type)
    else:
        raise ValueError(f"Unknown prompt type: {args.prompt_type}")
    # Load a dataset from HuggingFace datasets library
    print("Loading dataset...")
    target_datasets = load_subsets_for_fairness_explainability(args.dataset_name, args.split, args.split_ratio, args.bias_type, args.num_examples, shuffle=args.shuffle, seed=args.seed)


    # Initialize the explainer
    print("Running attribution methods...")
    all_methods = EXPLANATION_METHODS_MAPPING.keys()
    if args.methods:
        attribution_methods = args.methods.replace(' ', '').split(',')   
    else:
        attribution_methods = all_methods  # Use all methods if none specified


    # Create output directory if it does not exist
    output_dir = os.path.join(args.output_dir, "explanations")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for method in attribution_methods:
        print(f"\nRunning {method} explainer...")
        if method == "IntegratedGradients":
            model.config.use_cache = False
            model.gradient_checkpointing_enable()  # Enable gradient checkpointing for memory efficiency
            print("Gradient checkpointing enabled for IntegratedGradients.")
        else:
            model.gradient_checkpointing_disable()
        if EXPLANATION_METHODS_MAPPING[method] == BcosExplainer:
            explainer = BcosExplainer(model, tokenizer)
        elif EXPLANATION_METHODS_MAPPING[method] == ShapleyValueExplainer:
            explainer = ShapleyValueExplainer(model, tokenizer, method, args.baseline, args.shap_n_samples)
        # for GradientNPropabationExplainer, we need to specify the method
        elif EXPLANATION_METHODS_MAPPING[method] == GradientNPropabationExplainer:
            explainer = EXPLANATION_METHODS_MAPPING[method](model, tokenizer, method, args.baseline)
        else:
            explainer = EXPLANATION_METHODS_MAPPING[method](model, tokenizer) 

        for group in target_datasets.keys():
            target_dataset = target_datasets[group].to_dict()
            target_dataset['index'] = list(range(len(target_dataset['text'])))
            if not args.only_predicted_classes:   
                target_dataset['target'] = [["No", "Yes"] for _ in range(len(target_dataset['text']))]
            prompts = [tokenizer.apply_chat_template(fill_in_template(template, system_prompt.replace("[TEST EXAMPLE]", text)),tokenize=False,add_generation_prompt=True, enable_thinking=False, date_string="2025-07-01") for text in target_dataset['text']]
            target_dataset["prompt"] = prompts
            target_dataset["raw_input"] = [text+"\n\n" for text in target_dataset['text']]
            
            explanation_results = explainer.explain_dataset(target_dataset)
            result = explanation_results

            # Save the results to a JSON file
            output_file = os.path.join(output_dir, f'{method}_{group}_{args.split}_explanations.json')
            with open(output_file, 'w') as f:
                json.dump(result, f, indent=4)
            print(f"\nAttribution results saved to {output_file}")
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='BERT Attribution with Captum')

    parser.add_argument('--dataset_name', type=str, default='stanfordnlp/imdb', help='Name of the HuggingFace dataset to use') #fancyzhx/ag_news, stanfordnlp/imdb
    parser.add_argument('--split', type=str, default='test', help='Dataset split to use (e.g., train, test)')
    parser.add_argument('--split_ratio', type=str, default="0.8, 0.2",
                    help='Ratio to split the train set into train and validation sets')
    parser.add_argument('--model_dir', type=str, default='models/bert_base_imdb_512', help='Name of the pre-trained model')
    parser.add_argument('--num_labels', type=int, default=2, help='Not used for decoder models, but required for compatibility with the interface')
    parser.add_argument('--batch_size', type=int, default=1, help='only batch size=1 is supported for decoder models')
    parser.add_argument('--max_length', type=int, default=512, help='Not used for decoder models, but required for compatibility with the interface')
    parser.add_argument('--baseline', type=str, default='pad', help='Baseline for the attribution methods, select from zero, mask, pad')    
    parser.add_argument('--num_examples', type=int, default=1000, help='Number of examples to process (-1 for all)')
    parser.add_argument('--methods', type=str, default=None, help='List of attribution methods to use separated by commas')
    parser.add_argument('--output_dir', type=str, default='baseline_saliency_results/all_methods_1000_examples_512', help='Directory to save the output files')
    parser.add_argument('--shuffle', action='store_true', help='Shuffle the dataset before processing')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--shap_n_samples', type=int, default=25, help='Number of samples for Shapley Value Sampling')
    parser.add_argument('--only_predicted_classes', action='store_true', help='Only explain the predicted class')
    parser.add_argument('--bias_type', type=str, default="race", choices=["race", "gender", "religion"])
    parser.add_argument('--prompt_type', type=str, default="zero_shot", choices=["zero_shot", "few_shot", "fairness_imagination", "fairness_instruction"], help='Type of prompt to use for the model')


    args = parser.parse_args()
    main(args)

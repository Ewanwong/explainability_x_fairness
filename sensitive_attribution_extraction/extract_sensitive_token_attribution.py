import argparse
import numpy as np
import json
import os
import random
from tqdm import tqdm
from transformers import AutoTokenizer
from utils.reliance_utils import extract_sensitive_attributions
from utils.utils import EXPLANATION_METHODS
from utils.vocabulary import *


def main(args):

    if args.tokenizer_dir is not None:
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_dir)
    else:
        # Load the tokenizer
        if "roberta" in args.results_dir.lower():
            tokenizer = AutoTokenizer.from_pretrained("FacebookAI/roberta-base")
        elif "distilbert" in args.results_dir.lower():
            tokenizer = AutoTokenizer.from_pretrained("distilbert/distilbert-base-uncased")
        elif "bert" in args.results_dir.lower():
            tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        elif "llama" in args.results_dir.lower():
            tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-3B-Instruct")
        elif "qwen" in args.results_dir.lower():
            tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-3B-Instruct")
        elif "gpt" in args.results_dir.lower():
            tokenizer = AutoTokenizer.from_pretrained("gpt2")
        else:
            raise ValueError("Tokenizer is not provided and cannot be inferred from the explanation directory name.")
    
    if args.methods is not None:
        methods = args.methods.replace(' ', '').split(",")
    else:
        methods = EXPLANATION_METHODS
    
    groups = SOCIAL_GROUPS[args.bias_type]
    explanation_dir = os.path.join(args.results_dir, "explanations")
    reliance_dir = os.path.join(args.results_dir, "reliance")
    if not os.path.exists(reliance_dir):
        os.makedirs(reliance_dir)

    for method in methods:
        for group in groups:
            sensitive_tokens = SENSITIVE_TOKENS[args.bias_type][group]           
            group_explanation_file = os.path.join(explanation_dir, f"{method}_{group}_{args.split}_explanations.json")
            if not os.path.exists(group_explanation_file):
                #print(f"File {group_explanation_file} does not exist. Skipping...")
                continue
            print(f"Extracting sensitive attribution from {group_explanation_file}")

            sensitive_attribution_results = {}
            with open(group_explanation_file) as f:
                explanation_data = json.load(f)
            aggregations = list(explanation_data.keys())
            
            for aggregation in aggregations:
                sensitive_attribution_results[aggregation] = {}
                explanations = explanation_data[aggregation]
                sensitive_attribution_results[aggregation] = extract_sensitive_attributions(explanations, sensitive_tokens, tokenizer)

                if aggregation == "Occlusion":
                    # for occlusion, add one more key for first taking absolute values for all attribution scores in examples, then computing the reliance scores
                    sensitive_attribution_results[aggregation + "_abs"] = {}
                    abs_explanations = explanations.copy()
                    for expls in abs_explanations:
                        for expl in expls:
                            expl["attribution"] = [[token, abs(score)] for token, score in expl["attribution"]]
                    sensitive_attribution_results[aggregation + "_abs"] = extract_sensitive_attributions(abs_explanations, sensitive_tokens, tokenizer)
                        
            output_file = os.path.join(reliance_dir, f"{method}_{group}_{args.split}_sensitive_attribution.json")
            with open(output_file, 'w') as f:
                json.dump(sensitive_attribution_results, f, indent=4)
            print(f"Sensitive attribution results saved to {output_file}")                                                        

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='BERT Attribution with Captum')

    parser.add_argument('--results_dir', type=str, default=None, help='Path to the saliency data')
    parser.add_argument('--split', type=str, default='test', help='Dataset split to use (e.g., train, test)')
    parser.add_argument('--methods', type=str, default=None, help='List of attribution methods to use separated by commas')
    parser.add_argument('--bias_type', type=str, default='race', help='Bias type to explain')
    parser.add_argument('--tokenizer_dir', type=str, default=None, help='Path to the tokenizer directory')
    args = parser.parse_args()
    main(args)
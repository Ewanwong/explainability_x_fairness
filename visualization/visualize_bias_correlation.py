import argparse
import numpy as np
import json
import os
from tqdm import tqdm
import scipy.stats
from utils.reliance_utils import compute_reliance_score
from utils.utils import EXPLANATION_METHODS
from utils.vocabulary import *

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd

RELIANCE_KEYS = ["raw", "max", "len", "norm"]


def compute_reliance_scores(attribution_results):
    reliance_scores_dict = {key: [] for key in RELIANCE_KEYS}
    for key in RELIANCE_KEYS:
        reliance_scores = [compute_reliance_score(attr["predicted_class"]["sensitive_attribution"], attr["predicted_class"]["total_attribution"], method=key) for attr in attribution_results]
        reliance_scores_dict[key] = reliance_scores
    
    return reliance_scores_dict
    
def main(args):
    
    if args.methods is not None:
        methods = args.methods.replace(' ', '').split(",")
    else:
        methods = EXPLANATION_METHODS

    groups = SOCIAL_GROUPS[args.bias_type]

    fairness_results_dir = os.path.join(args.results_dir, "fairness")
    reliance_dir = os.path.join(args.results_dir, "reliance")
    # load fairness results    
    fairness_file = os.path.join(fairness_results_dir, f"fairness_{args.bias_type}_{args.split}_individual_stats.json")
    if not os.path.exists(fairness_file):
        raise ValueError(f"File {fairness_file} does not exist")
    with open(fairness_file) as f:
        individual_fairness_results = json.load(f)   
    

    for method in methods:
        
        individual_fairness_scores = {}
        reliance_scores = {}

        for group in groups:
            sensitive_attribution_file = os.path.join(reliance_dir, f"{method}_{group}_{args.split}_sensitive_attribution.json")
            if not os.path.exists(sensitive_attribution_file):
                #print(f"File {sensitive_sensitive_attribution_file} does not exist. Skipping...")
                continue
            
            # collect all predictions of the group (original)
            group_individual_fairness_scores= individual_fairness_results["Individual_Differences"][group]["predicted_class"]
            group_predictions = individual_fairness_results["Predictions"][group][group]

            with open(sensitive_attribution_file) as f:
                data = json.load(f)

            # mean or L2
            aggregations = list(data.keys())
            for aggregation in aggregations:
                if aggregation not in individual_fairness_scores:
                    individual_fairness_scores[aggregation] = {}
                if aggregation not in reliance_scores:
                    reliance_scores[aggregation] = {}
                attribution_results = data[aggregation]
                # make sure the attribution and fairness files have the same predictions
                # assert [attr["prediction"] for attr in attribution_results] == group_predictions

                positive_prediction_indexes = [i for i in range(len(group_predictions)) if group_predictions[i] == 1]
                negative_prediction_indexes = [i for i in range(len(group_predictions)) if group_predictions[i] == 0]

                positive_prediction_individual_fairness_scores = [group_individual_fairness_scores[i] for i in positive_prediction_indexes]
                negative_prediction_individual_fairness_scores = [group_individual_fairness_scores[i] for i in negative_prediction_indexes]

                positive_prediction_attributions = [attribution_results[i] for i in positive_prediction_indexes]
                negative_prediction_attributions = [attribution_results[i] for i in negative_prediction_indexes]

                group_positive_prediction_reliance_scores = compute_reliance_scores(positive_prediction_attributions)
                group_negative_prediction_reliance_scores = compute_reliance_scores(negative_prediction_attributions)

                individual_fairness_scores[aggregation][f"{group}_positive"] = positive_prediction_individual_fairness_scores
                individual_fairness_scores[aggregation][f"{group}_negative"] = negative_prediction_individual_fairness_scores

                reliance_scores[aggregation][f"{group}_positive"] = group_positive_prediction_reliance_scores
                reliance_scores[aggregation][f"{group}_negative"] = group_negative_prediction_reliance_scores

        
        # make plots and save 
        visualization_dir = os.path.join(args.results_dir, "visualization")
        if not os.path.exists(visualization_dir):
            os.makedirs(visualization_dir)

        for aggregation in individual_fairness_scores.keys():
            all_categories = []
            all_individual_fairness_scores = []
            for k, v in individual_fairness_scores[aggregation].items():
                all_categories += [k] * len(v)
                all_individual_fairness_scores += v
                
            for key in RELIANCE_KEYS:
                plot_path = os.path.join(visualization_dir, f"{method}_{aggregation}_{key}_correlation_visualization.png")
                all_reliance_scores = []
                for k, v in individual_fairness_scores[aggregation].items():
                    all_reliance_scores += reliance_scores[aggregation][k][key]

                df = pd.DataFrame({
                    'Sensitive_reliance': all_reliance_scores,
                    'Individual_fairness': all_individual_fairness_scores,
                    'Category': all_categories
                })

                # Plot using seaborn
                plt.figure(figsize=(16, 10))
                sns.scatterplot(data=df, x='Sensitive_reliance', y=f'Individual_fairness', hue='Category', palette='tab10')
                # add x=0 and y=0 lines
                plt.axhline(0, color='gray', linestyle='--')
                plt.axvline(0, color='gray', linestyle='--')
                # Show the plot
                plt.legend(title='Categories (group/predicted class)')
                plt.show()
                plt.savefig(plot_path)
                plt.close()
                print(f"Saved plot to {plot_path}")
            


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='BERT Attribution with Captum')

    parser.add_argument('--results_dir', type=str, default='baseline_saliency_results/all_methods_1000_examples_512', help='Path to the saliency data')
    parser.add_argument('--split', type=str, default='test', help='Dataset split to use (e.g., train, test)')
    parser.add_argument('--methods', type=str, default=None, help='List of attribution methods to use separated by commas')
    parser.add_argument('--bias_type', type=str, default="race", choices=["race", "gender", "religion"])


    args = parser.parse_args()
    main(args)
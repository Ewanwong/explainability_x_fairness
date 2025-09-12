import argparse
import numpy as np
import json
import os
from tqdm import tqdm
import scipy.stats
from utils.reliance_utils import compute_reliance_score
from utils.utils import EXPLANATION_METHODS
from utils.vocabulary import *

RELIANCE_KEYS = ["raw", "max", "len", "norm"]


def compute_fairness_reliance_correlation(fairness_scores, reliance_scores_dict):

    correlation_results = {}
    fairness_scores = np.array(fairness_scores)
    for key in RELIANCE_KEYS:
        reliance_score = reliance_scores_dict[key]
        # deal with inf and nan values by applying masks to fairness and reliance scores
        reliance_score = np.array(reliance_score)
        mask = np.isfinite(fairness_scores) & np.isfinite(reliance_score)
        corr = scipy.stats.pearsonr(fairness_scores[mask], reliance_score[mask])
        correlation_results[key] = corr
        
    return correlation_results

def compute_bias_correlation(fairness_scores, attribution_results):
    reliance_scores_dict = {key: [] for key in RELIANCE_KEYS}
    for key in RELIANCE_KEYS:
        reliance_scores = [compute_reliance_score(attr["predicted_class"]["sensitive_attribution"], attr["predicted_class"]["total_attribution"], method=key) for attr in attribution_results]
        reliance_scores_dict[key] = reliance_scores
    
    correlation_results = compute_fairness_reliance_correlation(fairness_scores, reliance_scores_dict)
    return correlation_results
    
def main(args):
    
    if args.methods is not None:
        methods = args.methods.replace(' ', '').split(",")
    else:
        methods = EXPLANATION_METHODS

    groups = SOCIAL_GROUPS[args.bias_type]

    fairness_results_dir = os.path.join(args.results_dir, "fairness")
    reliance_dir = os.path.join(args.results_dir, "reliance")
    correlation_results_dir = os.path.join(args.results_dir, "correlation")
    if not os.path.exists(correlation_results_dir):
        os.makedirs(correlation_results_dir)

    # load fairness results    
    fairness_file = os.path.join(fairness_results_dir, f"fairness_{args.bias_type}_{args.split}_individual_stats.json")
    if not os.path.exists(fairness_file):
        raise ValueError(f"File {fairness_file} does not exist")
    with open(fairness_file) as f:
        individual_fairness_results = json.load(f)   
    


    for method in methods:
        correlation_results = {}
        
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
                if aggregation not in correlation_results:
                    correlation_results[aggregation] = {}
                attribution_results = data[aggregation]
                # make sure the attribution and fairness files have the same predictions
                assert [attr["prediction"] for attr in attribution_results] == group_predictions

                positive_prediction_indexes = [i for i in range(len(group_predictions)) if group_predictions[i] == 1]
                negative_prediction_indexes = [i for i in range(len(group_predictions)) if group_predictions[i] == 0]

                positive_prediction_individual_fairness_scores = [group_individual_fairness_scores[i] for i in positive_prediction_indexes]
                negative_prediction_individual_fairness_scores = [group_individual_fairness_scores[i] for i in negative_prediction_indexes]

                positive_prediction_attributions = [attribution_results[i] for i in positive_prediction_indexes]
                negative_prediction_attributions = [attribution_results[i] for i in negative_prediction_indexes]

                group_positive_prediction_correlation_results = compute_bias_correlation(positive_prediction_individual_fairness_scores, positive_prediction_attributions)
                group_negative_prediction_correlation_results = compute_bias_correlation(negative_prediction_individual_fairness_scores, negative_prediction_attributions)

                correlation_results[aggregation][f"{group}_positive"] = group_positive_prediction_correlation_results
                correlation_results[aggregation][f"{group}_negative"] = group_negative_prediction_correlation_results
        
        # compute average correlation for all groups
        # compute average correlation for all labels
        # compute average correlation for all combinations of groups and labels
        for aggregation in correlation_results:
            correlation_results[aggregation]["abs_average"] = {}
            for group in groups:
                correlation_results[aggregation][f"{group}_abs_average"] = {}
                for key in RELIANCE_KEYS:
                    correlation_results[aggregation][f"{group}_abs_average"][key] = (abs(correlation_results[aggregation][f"{group}_positive"][key][0]) + abs(correlation_results[aggregation][f"{group}_negative"][key][0])) / 2
            correlation_results[aggregation]["positive_abs_average"] = {}
            correlation_results[aggregation]["negative_abs_average"] = {}
            for key in RELIANCE_KEYS:
                correlation_results[aggregation]["positive_abs_average"][key] = float(np.mean([abs(correlation_results[aggregation][f"{group}_positive"][key][0]) for group in groups]))
                correlation_results[aggregation]["negative_abs_average"][key] = float(np.mean([abs(correlation_results[aggregation][f"{group}_negative"][key][0]) for group in groups]))
                correlation_results[aggregation]["abs_average"][key] = float(np.mean([abs(correlation_results[aggregation][f"{group}_positive"][key][0]) + abs(correlation_results[aggregation][f"{group}_negative"][key][0]) for group in groups])) / 2

            # reorder the dictionary keys to make sure the summary statistics are shown first
            correlation_results[aggregation] = {k: correlation_results[aggregation][k] for k in ["abs_average", "positive_abs_average", "negative_abs_average"] + [f"{group}_abs_average" for group in groups] + [f"{group}_positive" for group in groups] + [f"{group}_negative" for group in groups] if k in correlation_results[aggregation]}
        
        correlation_file = os.path.join(correlation_results_dir, f"correlation_{method}_{args.bias_type}_{args.split}.json")
        with open(correlation_file, "w") as f:
            json.dump(correlation_results, f, indent=4)
        
        print(f"Correlation results for {method} saved to {correlation_file}")
            


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='BERT Attribution with Captum')

    parser.add_argument('--results_dir', type=str, default='baseline_saliency_results/all_methods_1000_examples_512', help='Path to the saliency data')
    parser.add_argument('--split', type=str, default='test', help='Dataset split to use (e.g., train, test)')
    parser.add_argument('--methods', type=str, default=None, help='List of attribution methods to use separated by commas')
    parser.add_argument('--bias_type', type=str, default="race", choices=["race", "gender", "religion"])
    

    args = parser.parse_args()
    main(args)
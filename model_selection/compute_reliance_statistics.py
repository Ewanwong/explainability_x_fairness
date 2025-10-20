# average absolute sensitive reliance score 
# difference between the average sensitive reliance score across groups
# difference between the average sensitive reliance score across groups by predicted class
# difference between the average sensitive reliance score across groups by true label

import argparse
import numpy as np
import json
import os
from tqdm import tqdm
from utils.reliance_utils import compute_reliance_score
from utils.utils import EXPLANATION_METHODS
from utils.vocabulary import *

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
    reliance_statistics_dir = os.path.join(args.results_dir, "reliance_statistics")
    if not os.path.exists(reliance_statistics_dir):
        os.makedirs(reliance_statistics_dir)
    # load fairness results    
    fairness_file = os.path.join(fairness_results_dir, f"fairness_{args.bias_type}_{args.split}_individual_stats.json")
    if not os.path.exists(fairness_file):
        raise ValueError(f"File {fairness_file} does not exist")
    with open(fairness_file) as f:
        individual_fairness_results = json.load(f)   
    

    for method in methods:
        print(f"Computing reliance statistics for method {method}...")
        reliance_scores = {}
        by_group_reliance_statistics = {}
        summary_reliance_statistics = {}
        for group in groups:
            sensitive_attribution_file = os.path.join(reliance_dir, f"{method}_{group}_{args.split}_sensitive_attribution.json")
            if not os.path.exists(sensitive_attribution_file):
                #print(f"File {sensitive_sensitive_attribution_file} does not exist. Skipping...")
                continue
            
            # collect all predictions of the group (original)
            group_predictions = individual_fairness_results["Predictions"][group][group]
            group_labels = individual_fairness_results["Labels"][group]
            with open(sensitive_attribution_file) as f:
                data = json.load(f)

            # mean or L2
            aggregations = list(data.keys())
            for aggregation in aggregations:
                if aggregation not in reliance_scores:
                    reliance_scores[aggregation] = {}

                attribution_results = data[aggregation]
                # make sure the attribution and fairness files have the same predictions
                # assert [attr["prediction"] for attr in attribution_results] == group_predictions

                positive_prediction_indexes = [i for i in range(len(group_predictions)) if group_predictions[i] == 1]
                negative_prediction_indexes = [i for i in range(len(group_predictions)) if group_predictions[i] == 0]

                positive_label_indexes = [i for i in range(len(group_labels)) if group_labels[i] == 1]
                negative_label_indexes = [i for i in range(len(group_labels)) if group_labels[i] == 0]

                positive_prediction_attributions = [attribution_results[i] for i in positive_prediction_indexes]
                negative_prediction_attributions = [attribution_results[i] for i in negative_prediction_indexes]

                positive_label_attributions = [attribution_results[i] for i in positive_label_indexes]
                negative_label_attributions = [attribution_results[i] for i in negative_label_indexes]

                group_positive_prediction_reliance_scores = compute_reliance_scores(positive_prediction_attributions)
                group_negative_prediction_reliance_scores = compute_reliance_scores(negative_prediction_attributions)

                group_positive_label_reliance_scores = compute_reliance_scores(positive_label_attributions)
                group_negative_label_reliance_scores = compute_reliance_scores(negative_label_attributions)

                reliance_scores[aggregation][f"{group}_positive_prediction"] = group_positive_prediction_reliance_scores
                reliance_scores[aggregation][f"{group}_negative_prediction"] = group_negative_prediction_reliance_scores
                reliance_scores[aggregation][f"{group}_positive_label"] = group_positive_label_reliance_scores
                reliance_scores[aggregation][f"{group}_negative_label"] = group_negative_label_reliance_scores
        
        # compute average correlation for all groups
        for aggregation in reliance_scores.keys():
            if aggregation not in by_group_reliance_statistics:
                by_group_reliance_statistics[aggregation] = {"abs_overall": {}, "overall": {}}

            # compute average correlation for all groups
            for group in groups:
                for cls in ['positive', 'negative']:
                    #by_group_reliance_statistics[aggregation]["abs_average"][f"{group}_{cls}_prediction"] = {key: float(np.mean(np.abs(reliance_scores[aggregation][f"{group}_{cls}_prediction"][key]))) for key in RELIANCE_KEYS}
                    #by_group_reliance_statistics[aggregation]["abs_average"][f"{group}_{cls}_label"] = {key: float(np.mean(np.abs(reliance_scores[aggregation][f"{group}_{cls}_label"][key]))) for key in RELIANCE_KEYS}
                    by_group_reliance_statistics[aggregation]["abs_overall"][f"{group}_{cls}_prediction"] = {key: float(np.mean(np.abs(reliance_scores[aggregation][f"{group}_{cls}_prediction"][key]))) for key in RELIANCE_KEYS}
                    by_group_reliance_statistics[aggregation]["abs_overall"][f"{group}_{cls}_label"] = {key: float(np.mean(np.abs(reliance_scores[aggregation][f"{group}_{cls}_label"][key]))) for key in RELIANCE_KEYS}
                
                    #by_group_reliance_statistics[aggregation]["average"][f"{group}_{cls}_prediction"] = {key: float(np.mean(reliance_scores[aggregation][f"{group}_{cls}_prediction"][key])) for key in RELIANCE_KEYS}
                    #by_group_reliance_statistics[aggregation]["average"][f"{group}_{cls}_label"] = {key: float(np.mean(reliance_scores[aggregation][f"{group}_{cls}_label"][key])) for key in RELIANCE_KEYS}
                    by_group_reliance_statistics[aggregation]["overall"][f"{group}_{cls}_prediction"] = {key: float(np.mean(reliance_scores[aggregation][f"{group}_{cls}_prediction"][key])) for key in RELIANCE_KEYS}
                    by_group_reliance_statistics[aggregation]["overall"][f"{group}_{cls}_label"] = {key: float(np.mean(reliance_scores[aggregation][f"{group}_{cls}_label"][key])) for key in RELIANCE_KEYS}
                

                #by_group_reliance_statistics[aggregation]["abs_average"][f"{group}_by_prediction"] = {key: float(np.mean(np.abs([by_group_reliance_statistics[aggregation]["abs_average"][f"{group}_{predicted_class}_prediction"][key] for predicted_class in ['positive', 'negative']]))) for key in RELIANCE_KEYS}
                #by_group_reliance_statistics[aggregation]["abs_average"][f"{group}_by_label"] = {key: float(np.mean(np.abs([by_group_reliance_statistics[aggregation]["abs_average"][f"{group}_{label}_label"][key] for label in ['positive', 'negative']]))) for key in RELIANCE_KEYS}
                by_group_reliance_statistics[aggregation]["abs_overall"][f"{group}"] = {key: float(np.mean(np.abs(np.concatenate([reliance_scores[aggregation][f"{group}_{predicted_class}_prediction"][key] for predicted_class in ['positive', 'negative']])))) for key in RELIANCE_KEYS}
                #by_group_reliance_statistics[aggregation]["average"][f"{group}_by_prediction"] = {key: float(np.mean([by_group_reliance_statistics[aggregation]["average"][f"{group}_{predicted_class}_prediction"][key] for predicted_class in ['positive', 'negative']])) for key in RELIANCE_KEYS}
                #by_group_reliance_statistics[aggregation]["average"][f"{group}_by_label"] = {key: float(np.mean([by_group_reliance_statistics[aggregation]["average"][f"{group}_{label}_label"][key] for label in ['positive', 'negative']])) for key in RELIANCE_KEYS}
                by_group_reliance_statistics[aggregation]["overall"][f"{group}"] = {key: float(np.mean(np.concatenate([reliance_scores[aggregation][f"{group}_{predicted_class}_prediction"][key] for predicted_class in ['positive', 'negative']]))) for key in RELIANCE_KEYS}
            by_group_reliance_statistics[aggregation]["abs_overall"][f"average_positive_prediction"] = {key: float(np.mean(np.abs([by_group_reliance_statistics[aggregation]["abs_overall"][f"{group}_positive_prediction"][key] for group in groups]))) for key in RELIANCE_KEYS}
            by_group_reliance_statistics[aggregation]["abs_overall"][f"average_negative_prediction"] = {key: float(np.mean(np.abs([by_group_reliance_statistics[aggregation]["abs_overall"][f"{group}_negative_prediction"][key] for group in groups]))) for key in RELIANCE_KEYS}
            by_group_reliance_statistics[aggregation]["abs_overall"][f"average_positive_label"] = {key: float(np.mean(np.abs([by_group_reliance_statistics[aggregation]["abs_overall"][f"{group}_positive_label"][key] for group in groups]))) for key in RELIANCE_KEYS}
            by_group_reliance_statistics[aggregation]["abs_overall"][f"average_negative_label"] = {key: float(np.mean(np.abs([by_group_reliance_statistics[aggregation]["abs_overall"][f"{group}_negative_label"][key] for group in groups]))) for key in RELIANCE_KEYS}
            by_group_reliance_statistics[aggregation]["overall"][f"average_positive_prediction"] = {key: float(np.mean([by_group_reliance_statistics[aggregation]["overall"][f"{group}_positive_prediction"][key] for group in groups])) for key in RELIANCE_KEYS}
            by_group_reliance_statistics[aggregation]["overall"][f"average_negative_prediction"] = {key: float(np.mean([by_group_reliance_statistics[aggregation]["overall"][f"{group}_negative_prediction"][key] for group in groups])) for key in RELIANCE_KEYS}
            by_group_reliance_statistics[aggregation]["overall"][f"average_positive_label"] = {key: float(np.mean([by_group_reliance_statistics[aggregation]["overall"][f"{group}_positive_label"][key] for group in groups])) for key in RELIANCE_KEYS}
            by_group_reliance_statistics[aggregation]["overall"][f"average_negative_label"] = {key: float(np.mean([by_group_reliance_statistics[aggregation]["overall"][f"{group}_negative_label"][key] for group in groups])) for key in RELIANCE_KEYS}
                
            
            # compute mean across groups
            for cls in ['positive', 'negative']:
                #by_group_reliance_statistics[aggregation]["abs_average"][f'{cls}_prediction_average_over_groups'] = {key: float(np.mean(np.abs([by_group_reliance_statistics[aggregation]["abs_average"][f"{group}_{cls}_prediction"][key] for group in groups]))) for key in RELIANCE_KEYS}
                #by_group_reliance_statistics[aggregation]["abs_average"][f'{cls}_label_average_over_groups'] = {key: float(np.mean(np.abs([by_group_reliance_statistics[aggregation]["abs_average"][f"{group}_{cls}_label"][key] for group in groups]))) for key in RELIANCE_KEYS}
                by_group_reliance_statistics[aggregation]["abs_overall"][f'{cls}_prediction'] = {key: float(np.mean(np.abs(np.concatenate([reliance_scores[aggregation][f"{group}_{cls}_prediction"][key] for group in groups])))) for key in RELIANCE_KEYS}
                by_group_reliance_statistics[aggregation]["abs_overall"][f'{cls}_label'] = {key: float(np.mean(np.abs(np.concatenate([reliance_scores[aggregation][f"{group}_{cls}_label"][key] for group in groups])))) for key in RELIANCE_KEYS}
                #by_group_reliance_statistics[aggregation]["average"][f'{cls}_prediction_average_over_groups'] = {key: float(np.mean([by_group_reliance_statistics[aggregation]["average"][f"{group}_{cls}_prediction"][key] for group in groups])) for key in RELIANCE_KEYS}
                #by_group_reliance_statistics[aggregation]["average"][f'{cls}_label_average_over_groups'] = {key: float(np.mean([by_group_reliance_statistics[aggregation]["average"][f"{group}_{cls}_label"][key] for group in groups])) for key in RELIANCE_KEYS}
                by_group_reliance_statistics[aggregation]["overall"][f'{cls}_prediction'] = {key: float(np.mean(np.concatenate([reliance_scores[aggregation][f"{group}_{cls}_prediction"][key] for group in groups]))) for key in RELIANCE_KEYS}
                by_group_reliance_statistics[aggregation]["overall"][f'{cls}_label'] = {key: float(np.mean(np.concatenate([reliance_scores[aggregation][f"{group}_{cls}_label"][key] for group in groups]))) for key in RELIANCE_KEYS}    

            #by_group_reliance_statistics[aggregation]["abs_average"]["mean_by_prediction"] = {key: float(np.mean(np.abs([by_group_reliance_statistics[aggregation]["abs_average"][f"{group}_by_prediction"][key] for group in groups]))) for key in RELIANCE_KEYS}
            #by_group_reliance_statistics[aggregation]["abs_average"]["mean_by_label"] = {key: float(np.mean(np.abs([by_group_reliance_statistics[aggregation]["abs_average"][f"{group}_by_label"][key] for group in groups]))) for key in RELIANCE_KEYS}
            #by_group_reliance_statistics[aggregation]["abs_overall"]["mean_all"] = {key: float(np.mean(np.abs(np.concatenate([reliance_scores[aggregation][f"{group}_{predicted_class}_prediction"][key] for group in groups for predicted_class in ['positive', 'negative']])))) for key in RELIANCE_KEYS}
            #by_group_reliance_statistics[aggregation]["abs_average"]["mean_all"] = {key: float(np.mean(np.abs([by_group_reliance_statistics[aggregation]["abs_overall"][f"{group}_{predicted_class}_prediction"][key] for group in groups for predicted_class in ['positive', 'negative']]))) for key in RELIANCE_KEYS}
            #by_group_reliance_statistics[aggregation]["average"]["mean_by_prediction"] = {key: float(np.mean([by_group_reliance_statistics[aggregation]["average"][f"{group}_by_prediction"][key] for group in groups])) for key in RELIANCE_KEYS}
            #by_group_reliance_statistics[aggregation]["overall"]["mean_all"] = {key: float(np.mean(np.concatenate([reliance_scores[aggregation][f"{group}_{predicted_class}_prediction"][key] for group in groups for predicted_class in ['positive', 'negative']]))) for key in RELIANCE_KEYS}
            #by_group_reliance_statistics[aggregation]["average"]["mean_all"] = {key: float(np.mean([by_group_reliance_statistics[aggregation]["average"][f"{group}_by_label"][key] for group in groups])) for key in RELIANCE_KEYS}
            #by_group_reliance_statistics[aggregation]["average"]["mean_by_label"] = {key: float(np.mean([by_group_reliance_statistics[aggregation]["average"][f"{group}_by_label"][key] for group in groups])) for key in RELIANCE_KEYS}
            
            # summary statistics
            #by_group_reliance_statistics[aggregation]["abs_average"]["all_by_labels"] = {key: float(np.mean(np.abs([by_group_reliance_statistics[aggregation]["abs_average"][f"{group}_by_label"][key] for group in groups]))) for key in RELIANCE_KEYS}
            #by_group_reliance_statistics[aggregation]["abs_average"]["all_by_predictions"] = {key: float(np.mean(np.abs([by_group_reliance_statistics[aggregation]["abs_average"][f"{group}_by_prediction"][key] for group in groups]))) for key in RELIANCE_KEYS}
            #by_group_reliance_statistics[aggregation]["abs_average"]["all_by_groups_predictions"] = {key: float(np.mean(np.abs([by_group_reliance_statistics[aggregation]["abs_average"][f"{group}_{predicted_class}_prediction"][key] for group in groups for predicted_class in ['positive', 'negative']]))) for key in RELIANCE_KEYS}
            #by_group_reliance_statistics[aggregation]["abs_average"]["all_by_groups_labels"] = {key: float(np.mean(np.abs([by_group_reliance_statistics[aggregation]["abs_average"][f"{group}_{label}_label"][key] for group in groups for label in ['positive', 'negative']]))) for key in RELIANCE_KEYS}
            by_group_reliance_statistics[aggregation]["abs_overall"]["overall"] = {key: float(np.mean(np.abs(np.concatenate([reliance_scores[aggregation][f"{group}_{predicted_class}_prediction"][key] for group in groups for predicted_class in ['positive', 'negative']])))) for key in RELIANCE_KEYS}
            by_group_reliance_statistics[aggregation]["abs_overall"]["average"] = {key: float(np.mean(np.abs([by_group_reliance_statistics[aggregation]["abs_overall"][f"{group}"][key] for group in groups]))) for key in RELIANCE_KEYS}

            #by_group_reliance_statistics[aggregation]["abs_average"]["overall"] = {key: float(np.mean(np.abs(np.concatenate([reliance_scores[aggregation][f"{group}_{label}_label"][key] for group in groups for label in ['positive', 'negative']])))) for key in RELIANCE_KEYS}
            #by_group_reliance_statistics[aggregation]["average"]["all_by_labels"] = {key: float(np.mean([by_group_reliance_statistics[aggregation]["_average"][f"{group}_by_label"][key] for group in groups])) for key in RELIANCE_KEYS}
            #by_group_reliance_statistics[aggregation]["average"]["all_by_predictions"] = {key: float(np.mean([by_group_reliance_statistics[aggregation]["average"][f"{group}_by_prediction"][key] for group in groups])) for key in RELIANCE_KEYS}
            #by_group_reliance_statistics[aggregation]["average"]["all_by_groups_predictions"] = {key: float(np.mean(np.concatenate([reliance_scores[aggregation][f"{group}_{predicted_class}_prediction"][key] for group in groups for predicted_class in ['positive', 'negative']]))) for key in RELIANCE_KEYS}
            #by_group_reliance_statistics[aggregation]["average"]["all_by_groups_labels"] = {key: float(np.mean(np.concatenate([reliance_scores[aggregation][f"{group}_{label}_label"][key] for group in groups for label in ['positive', 'negative']]))) for key in RELIANCE_KEYS}
            by_group_reliance_statistics[aggregation]["overall"]["overall"] = {key: float(np.mean(np.concatenate([reliance_scores[aggregation][f"{group}_{predicted_class}_prediction"][key] for group in groups for predicted_class in ['positive', 'negative']]))) for key in RELIANCE_KEYS}
            by_group_reliance_statistics[aggregation]["overall"]["average"] = {key: float(np.mean([by_group_reliance_statistics[aggregation]["overall"][f"{group}"][key] for group in groups])) for key in RELIANCE_KEYS}
            
        # compute differences   
        for aggregation in reliance_scores.keys():
            if aggregation not in summary_reliance_statistics:
                summary_reliance_statistics[aggregation] = {"summary_statistics": {}, "sensitive_reliance_difference": {}}
                summary_reliance_statistics[aggregation]["summary_statistics"]["overall"] = by_group_reliance_statistics[aggregation]["overall"]["overall"]
                summary_reliance_statistics[aggregation]["summary_statistics"]["abs_overall"] = by_group_reliance_statistics[aggregation]["abs_overall"]["overall"]
                summary_reliance_statistics[aggregation]["summary_statistics"]["average"] = by_group_reliance_statistics[aggregation]["overall"]["average"]
                summary_reliance_statistics[aggregation]["summary_statistics"]["abs_average"] = by_group_reliance_statistics[aggregation]["abs_overall"]["average"]
                #summary_reliance_statistics[aggregation]["summary_statistics"]["average_over_labels"] = by_group_reliance_statistics[aggregation]["abs_average"]["all_by_labels"]
                #summary_reliance_statistics[aggregation]["summary_statistics"]["average_over_predictions"] = by_group_reliance_statistics[aggregation]["abs_average"]["all_by_predictions"]
                #summary_reliance_statistics[aggregation]["summary_statistics"]["average_over_groups_labels"] = by_group_reliance_statistics[aggregation]["abs_average"]["all_by_groups_labels"]
                #summary_reliance_statistics[aggregation]["summary_statistics"]["average_over_groups_predictions"] = by_group_reliance_statistics[aggregation]["abs_average"]["all_by_groups_predictions"]
            for group in groups:
                summary_reliance_statistics[aggregation]["sensitive_reliance_difference"][group] = {}
                #summary_reliance_statistics[aggregation]["sensitive_reliance_difference"][group]["average_positive_prediction"] = {key: by_group_reliance_statistics[aggregation]["average"][f"{group}_positive_prediction"][key] - by_group_reliance_statistics[aggregation]["average"][f"mean_positive_prediction"][key] for key in RELIANCE_KEYS}
                #summary_reliance_statistics[aggregation]["sensitive_reliance_difference"][group]["average_negative_prediction"] = {key: by_group_reliance_statistics[aggregation]["average"][f"{group}_negative_prediction"][key] - by_group_reliance_statistics[aggregation]["average"][f"mean_negative_prediction"][key] for key in RELIANCE_KEYS}
                #summary_reliance_statistics[aggregation]["sensitive_reliance_difference"][group]["average_positive_label"] = {key: by_group_reliance_statistics[aggregation]["average"][f"{group}_positive_label"][key] - by_group_reliance_statistics[aggregation]["average"][f"mean_positive_label"][key] for key in RELIANCE_KEYS}
                #summary_reliance_statistics[aggregation]["sensitive_reliance_difference"][group]["average_negative_label"] = {key: by_group_reliance_statistics[aggregation]["average"][f"{group}_negative_label"][key] - by_group_reliance_statistics[aggregation]["average"][f"mean_negative_label"][key] for key in RELIANCE_KEYS}
                #summary_reliance_statistics[aggregation]["sensitive_reliance_difference"][group]["average_by_prediction"] = {key: by_group_reliance_statistics[aggregation]["average"][f"{group}_by_prediction"][key] - by_group_reliance_statistics[aggregation]["average"][f"mean_by_prediction"][key] for key in RELIANCE_KEYS}
                #summary_reliance_statistics[aggregation]["sensitive_reliance_difference"][group]["average_by_label"] = {key: by_group_reliance_statistics[aggregation]["average"][f"{group}_by_label"][key] - by_group_reliance_statistics[aggregation]["average"][f"mean_by_label"][key] for key in RELIANCE_KEYS}
                summary_reliance_statistics[aggregation]["sensitive_reliance_difference"][group]["positive_prediction(average)"] = {key: by_group_reliance_statistics[aggregation]["overall"][f"{group}_positive_prediction"][key] - by_group_reliance_statistics[aggregation]["overall"]["average_positive_prediction"][key] for key in RELIANCE_KEYS}
                summary_reliance_statistics[aggregation]["sensitive_reliance_difference"][group]["negative_prediction(average)"] = {key: by_group_reliance_statistics[aggregation]["overall"][f"{group}_negative_prediction"][key] - by_group_reliance_statistics[aggregation]["overall"]["average_negative_prediction"][key] for key in RELIANCE_KEYS}
                summary_reliance_statistics[aggregation]["sensitive_reliance_difference"][group]["positive_label(average)"] = {key: by_group_reliance_statistics[aggregation]["overall"][f"{group}_positive_label"][key] - by_group_reliance_statistics[aggregation]["overall"]["average_positive_label"][key] for key in RELIANCE_KEYS}
                summary_reliance_statistics[aggregation]["sensitive_reliance_difference"][group]["negative_label(average)"] = {key: by_group_reliance_statistics[aggregation]["overall"][f"{group}_negative_label"][key] - by_group_reliance_statistics[aggregation]["overall"]["average_negative_label"][key] for key in RELIANCE_KEYS}
                summary_reliance_statistics[aggregation]["sensitive_reliance_difference"][group]["positive_prediction(overall)"] = {key: by_group_reliance_statistics[aggregation]["overall"][f"{group}_positive_prediction"][key] - by_group_reliance_statistics[aggregation]["overall"]["positive_prediction"][key] for key in RELIANCE_KEYS}
                summary_reliance_statistics[aggregation]["sensitive_reliance_difference"][group]["negative_prediction(overall)"] = {key: by_group_reliance_statistics[aggregation]["overall"][f"{group}_negative_prediction"][key] - by_group_reliance_statistics[aggregation]["overall"]["negative_prediction"][key] for key in RELIANCE_KEYS}
                summary_reliance_statistics[aggregation]["sensitive_reliance_difference"][group]["positive_label(overall)"] = {key: by_group_reliance_statistics[aggregation]["overall"][f"{group}_positive_label"][key] - by_group_reliance_statistics[aggregation]["overall"]["positive_label"][key] for key in RELIANCE_KEYS}
                summary_reliance_statistics[aggregation]["sensitive_reliance_difference"][group]["negative_label(overall)"] = {key: by_group_reliance_statistics[aggregation]["overall"][f"{group}_negative_label"][key] - by_group_reliance_statistics[aggregation]["overall"]["negative_label"][key] for key in RELIANCE_KEYS}
                summary_reliance_statistics[aggregation]["sensitive_reliance_difference"][group]["overall"] = {key: by_group_reliance_statistics[aggregation]["overall"][f"{group}"][key] - by_group_reliance_statistics[aggregation]["overall"]["overall"][key] for key in RELIANCE_KEYS}
                summary_reliance_statistics[aggregation]["sensitive_reliance_difference"][group]["average"] = {key: by_group_reliance_statistics[aggregation]["overall"][f"{group}"][key] - by_group_reliance_statistics[aggregation]["overall"]["average"][key] for key in RELIANCE_KEYS}
                #summary_reliance_statistics[aggregation]["sensitive_reliance_difference"][group]["abs_average_positive_prediction"] = {key: by_group_reliance_statistics[aggregation]["abs_average"][f"{group}_positive_prediction"][key] - by_group_reliance_statistics[aggregation]["abs_average"][f"mean_positive_prediction"][key] for key in RELIANCE_KEYS}
                #summary_reliance_statistics[aggregation]["sensitive_reliance_difference"][group]["abs_average_negative_prediction"] = {key: by_group_reliance_statistics[aggregation]["abs_average"][f"{group}_negative_prediction"][key] - by_group_reliance_statistics[aggregation]["abs_average"][f"mean_negative_prediction"][key] for key in RELIANCE_KEYS}
                #summary_reliance_statistics[aggregation]["sensitive_reliance_difference"][group]["abs_average_positive_label"] = {key: by_group_reliance_statistics[aggregation]["abs_average"][f"{group}_positive_label"][key] - by_group_reliance_statistics[aggregation]["abs_average"][f"mean_positive_label"][key] for key in RELIANCE_KEYS}
                #summary_reliance_statistics[aggregation]["sensitive_reliance_difference"][group]["abs_average_negative_label"] = {key: by_group_reliance_statistics[aggregation]["abs_average"][f"{group}_negative_label"][key] - by_group_reliance_statistics[aggregation]["abs_average"][f"mean_negative_label"][key] for key in RELIANCE_KEYS}
                #summary_reliance_statistics[aggregation]["sensitive_reliance_difference"][group]["abs_average_by_prediction"] = {key: by_group_reliance_statistics[aggregation]["abs_average"][f"{group}_by_prediction"][key] - by_group_reliance_statistics[aggregation]["abs_average"][f"mean_by_prediction"][key] for key in RELIANCE_KEYS}
                #summary_reliance_statistics[aggregation]["sensitive_reliance_difference"][group]["abs_average_by_label"] = {key: by_group_reliance_statistics[aggregation]["abs_average"][f"{group}_by_label"][key] - by_group_reliance_statistics[aggregation]["abs_average"][f"mean_by_label"][key] for key in RELIANCE_KEYS}
                summary_reliance_statistics[aggregation]["sensitive_reliance_difference"][group]["abs_positive_prediction(average)"] = {key: by_group_reliance_statistics[aggregation]["abs_overall"][f"{group}_positive_prediction"][key] - by_group_reliance_statistics[aggregation]["abs_overall"]["average_positive_prediction"][key] for key in RELIANCE_KEYS}
                summary_reliance_statistics[aggregation]["sensitive_reliance_difference"][group]["abs_negative_prediction(average)"] = {key: by_group_reliance_statistics[aggregation]["abs_overall"][f"{group}_negative_prediction"][key] - by_group_reliance_statistics[aggregation]["abs_overall"]["average_negative_prediction"][key] for key in RELIANCE_KEYS}
                summary_reliance_statistics[aggregation]["sensitive_reliance_difference"][group]["abs_positive_label(average)"] = {key: by_group_reliance_statistics[aggregation]["abs_overall"][f"{group}_positive_label"][key] - by_group_reliance_statistics[aggregation]["abs_overall"]["average_positive_label"][key] for key in RELIANCE_KEYS}
                summary_reliance_statistics[aggregation]["sensitive_reliance_difference"][group]["abs_negative_label(average)"] = {key: by_group_reliance_statistics[aggregation]["abs_overall"][f"{group}_negative_label"][key] - by_group_reliance_statistics[aggregation]["abs_overall"]["average_negative_label"][key] for key in RELIANCE_KEYS}
                summary_reliance_statistics[aggregation]["sensitive_reliance_difference"][group]["abs_positive_prediction(overall)"] = {key: by_group_reliance_statistics[aggregation]["abs_overall"][f"{group}_positive_prediction"][key] - by_group_reliance_statistics[aggregation]["abs_overall"]["positive_prediction"][key] for key in RELIANCE_KEYS}
                summary_reliance_statistics[aggregation]["sensitive_reliance_difference"][group]["abs_negative_prediction(overall)"] = {key: by_group_reliance_statistics[aggregation]["abs_overall"][f"{group}_negative_prediction"][key] - by_group_reliance_statistics[aggregation]["abs_overall"]["negative_prediction"][key] for key in RELIANCE_KEYS}
                summary_reliance_statistics[aggregation]["sensitive_reliance_difference"][group]["abs_positive_label(overall)"] = {key: by_group_reliance_statistics[aggregation]["abs_overall"][f"{group}_positive_label"][key] - by_group_reliance_statistics[aggregation]["abs_overall"]["positive_label"][key] for key in RELIANCE_KEYS}
                summary_reliance_statistics[aggregation]["sensitive_reliance_difference"][group]["abs_negative_label(overall)"] = {key: by_group_reliance_statistics[aggregation]["abs_overall"][f"{group}_negative_label"][key] - by_group_reliance_statistics[aggregation]["abs_overall"]["negative_label"][key] for key in RELIANCE_KEYS}
                summary_reliance_statistics[aggregation]["sensitive_reliance_difference"][group]["abs_overall"] = {key: by_group_reliance_statistics[aggregation]["abs_overall"][f"{group}"][key] - by_group_reliance_statistics[aggregation]["abs_overall"]["overall"][key] for key in RELIANCE_KEYS}
                summary_reliance_statistics[aggregation]["sensitive_reliance_difference"][group]["abs_average"] = {key: by_group_reliance_statistics[aggregation]["abs_overall"][f"{group}"][key] - by_group_reliance_statistics[aggregation]["abs_overall"]["average"][key] for key in RELIANCE_KEYS}

        
        # save results
        reliance_statistics_file = os.path.join(reliance_statistics_dir, f"{method}_{args.bias_type}_{args.split}_sensitive_reliance_statistics.json")
        if not os.path.exists(os.path.dirname(reliance_statistics_file)):
            os.makedirs(os.path.dirname(reliance_statistics_file))
        with open(reliance_statistics_file, 'w') as f:
            json.dump({"summary": summary_reliance_statistics, "by_group": by_group_reliance_statistics}, f, indent=4)
        print(f"Saved reliance statistics to {reliance_statistics_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='BERT Attribution with Captum')

    parser.add_argument('--results_dir', type=str, default='baseline_saliency_results/all_methods_1000_examples_512', help='Path to the saliency data')
    parser.add_argument('--split', type=str, default='test', help='Dataset split to use (e.g., train, test)')
    parser.add_argument('--methods', type=str, default=None, help='List of attribution methods to use separated by commas')
    parser.add_argument('--bias_type', type=str, default="race", choices=["race", "gender", "religion"])

    args = parser.parse_args()
    main(args)
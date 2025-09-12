import argparse
import numpy as np
import json
import os
from tqdm import tqdm
from utils.utils import EXPLANATION_METHODS
from utils.vocabulary import *
from scipy.stats import pearsonr, spearmanr

RELIANCE_KEYS = ["raw", "max", "len", "norm"]

# compute correlation between test and val results

def compute_correlation(val_diffs, test_diffs):
    test_values = np.array(list(test_diffs.values()))
    val_values = np.array(list(val_diffs.values()))
    # deal with inf and nan values by applying a mask to val values
    mask = np.isfinite(val_values)
    val_values = val_values[mask]
    # apply the same mask to test values
    test_values = test_values[mask]
    if len(val_values) < 2 or len(test_values) < 2:
        return 0, 1.0
    correlation, p = pearsonr(val_values, test_values)
    return correlation, p

# rank methods by who has the smallest difference in each metric
def rank_models(metric_diffs):
    sorted_models = sorted(metric_diffs.items(), key=lambda x: x[1])
    return [model for model, _ in sorted_models]

# compute rank correlation using Spearman's rank correlation

def compute_rank_correlation(test_diffs, val_diffs):
    test_ranks = rank_models(test_diffs)
    val_ranks = rank_models(val_diffs)
    correlation, p = spearmanr(test_ranks, val_ranks)
    return correlation, p

def load_fairness_results(paths, bias_type, models):
    groups = SOCIAL_GROUPS[bias_type]
    results = {}
    for model, path in zip(models, paths):
        if not os.path.exists(path):
            print(f"Fairness results for {model} not found at {path}")
            continue
        with open(path, 'r') as f:
            fairness_data = json.load(f)
        results[model] = {
            "accuracy": sum(abs(fairness_data["Group_Fairness"]["average"][group]["accuracy"]) for group in groups),
            "f1": sum(abs(fairness_data["Group_Fairness"]["average"][group]["f1"]) for group in groups),
            "fpr": sum(abs(fairness_data["Group_Fairness"]["average"][group]["fpr"]) for group in groups),
            "fnr": sum(abs(fairness_data["Group_Fairness"]["average"][group]["fnr"]) for group in groups),
            "individual_fairness": fairness_data["Individual_Fairness"]["overall"]["predicted_class"]["abs_average"]
        }
    return results

def compute_fairness_correlation(val_results, test_results):
    accuracy_corr, accuracy_p = compute_correlation(
        {model: results["accuracy"] for model, results in val_results.items()},
        {model: results["accuracy"] for model, results in test_results.items()},
    )
    accuracy_rank_corr, accuracy_rank_p = compute_rank_correlation(
        {model: results["accuracy"] for model, results in val_results.items()},
        {model: results["accuracy"] for model, results in test_results.items()},
    )
    f1_corr, f1_p = compute_correlation(
        {model: results["f1"] for model, results in val_results.items()},
        {model: results["f1"] for model, results in test_results.items()},
    )
    f1_rank_corr, f1_rank_p = compute_rank_correlation(
        {model: results["f1"] for model, results in val_results.items()},
        {model: results["f1"] for model, results in test_results.items()},
    )
    fpr_corr, fpr_p = compute_correlation(
        {model: results["fpr"] for model, results in val_results.items()},
        {model: results["fpr"] for model, results in test_results.items()},
    )
    fpr_rank_corr, fpr_rank_p = compute_rank_correlation(
        {model: results["fpr"] for model, results in val_results.items()},
        {model: results["fpr"] for model, results in test_results.items()},
    )
    fnr_corr, fnr_p = compute_correlation(
        {model: results["fnr"] for model, results in val_results.items()},
        {model: results["fnr"] for model, results in test_results.items()},
    )
    fnr_rank_corr, fnr_rank_p = compute_rank_correlation(
        {model: results["fnr"] for model, results in val_results.items()},
        {model: results["fnr"] for model, results in test_results.items()},
    )
    individual_fairness_corr, individual_fairness_p = compute_correlation(
        {model: results["individual_fairness"] for model, results in val_results.items()},
        {model: results["individual_fairness"] for model, results in test_results.items()},
    )
    individual_fairness_rank_corr, individual_fairness_rank_p = compute_rank_correlation(
        {model: results["individual_fairness"] for model, results in val_results.items()},
        {model: results["individual_fairness"] for model, results in test_results.items()},
    )
    return {
        "accuracy": (accuracy_corr, accuracy_p),
        "accuracy_rank": (accuracy_rank_corr, accuracy_rank_p),
        "f1": (f1_corr, f1_p),
        "f1_rank": (f1_rank_corr, f1_rank_p),
        "fpr": (fpr_corr, fpr_p),
        "fpr_rank": (fpr_rank_corr, fpr_rank_p),
        "fnr": (fnr_corr, fnr_p),
        "fnr_rank": (fnr_rank_corr, fnr_rank_p),
        "individual_fairness": (individual_fairness_corr, individual_fairness_p),
        "individual_fairness_rank": (individual_fairness_rank_corr, individual_fairness_rank_p)
    }

def load_reliance_results(dirs, methods, bias_type, models, split="val"):
    groups = SOCIAL_GROUPS[bias_type]
    results = {}
    for model, dir in zip(models, dirs):
        if not os.path.exists(dir):
            print(f"Reliance results for {model} not found at {dir}")
            continue
        results[model] = {}
        for method in methods:
            method_results = {}
            for group in groups:
                file_path = os.path.join(dir, f"{method}_{bias_type}_{split}_sensitive_reliance_statistics.json")
                if not os.path.exists(file_path):
                    print(f"Reliance results for {model}, {method}, {group} not found at {file_path}")
                    continue
                with open(file_path, 'r') as f:
                    data = json.load(f)["summary"]
                aggregations = data.keys()
                for aggregation in aggregations:
                    if aggregation not in method_results:
                        method_results[aggregation] = {}
                    method_results[aggregation]["overall"] = data[aggregation]["summary_statistics"]["abs_overall"]
                    method_results[aggregation]["reliance_diff"] = {key: sum([abs(data[aggregation]["sensitive_reliance_difference"][group]["abs_overall"][key]) for group in groups]) for key in RELIANCE_KEYS}
                    method_results[aggregation]["reliance_diff_positive_prediction"] = {key: sum([abs(data[aggregation]["sensitive_reliance_difference"][group]["abs_positive_prediction(average)"][key]) for group in groups]) for key in RELIANCE_KEYS}
                    method_results[aggregation]["reliance_diff_negative_prediction"] = {key: sum([abs(data[aggregation]["sensitive_reliance_difference"][group]["abs_negative_prediction(average)"][key]) for group in groups]) for key in RELIANCE_KEYS}
                    method_results[aggregation]["reliance_diff_positive_label"] = {key: sum([abs(data[aggregation]["sensitive_reliance_difference"][group]["abs_positive_label(average)"][key]) for group in groups]) for key in RELIANCE_KEYS}
                    method_results[aggregation]["reliance_diff_negative_label"] = {key: sum([abs(data[aggregation]["sensitive_reliance_difference"][group]["abs_negative_label(average)"][key]) for group in groups]) for key in RELIANCE_KEYS}
            results[model][method] = method_results

    return results

def compute_reliance_correlation(reliance_results, fairness_results):
    correlation_results = {}
    models = list(reliance_results.keys())
    for method in reliance_results[models[0]].keys():
        method_results = {}

        for aggregation in reliance_results[models[0]][method].keys():
            method_results[aggregation] = {"accuracy": {}, "accuracy_rank": {}, "f1": {}, "f1_rank": {}, "fpr": {}, "fpr_rank": {}, "fnr": {}, "fnr_rank": {}, "individual_fairness": {}, "individual_fairness_rank": {}}
            overall_list_dict = {key: [abs(reliance_results[model][method][aggregation]["overall"][key]) for model in models] for key in RELIANCE_KEYS}
            reliance_diff_list_dict = {key: [abs(reliance_results[model][method][aggregation]["reliance_diff"][key]) for model in models] for key in RELIANCE_KEYS}
            reliance_diff_positive_prediction_list_dict = {key: [abs(reliance_results[model][method][aggregation]["reliance_diff_positive_prediction"][key]) for model in models] for key in RELIANCE_KEYS}
            reliance_diff_negative_prediction_list_dict = {key: [abs(reliance_results[model][method][aggregation]["reliance_diff_negative_prediction"][key]) for model in models] for key in RELIANCE_KEYS}
            reliance_diff_positive_label_list_dict = {key: [abs(reliance_results[model][method][aggregation]["reliance_diff_positive_label"][key]) for model in models] for key in RELIANCE_KEYS}
            reliance_diff_negative_label_list_dict = {key: [abs(reliance_results[model][method][aggregation]["reliance_diff_negative_label"][key]) for model in models] for key in RELIANCE_KEYS}
            # Compute correlations with fairness metrics
            for key in RELIANCE_KEYS:
                # individual fairness reliance correlation
                individual_fairness_corr, individual_fairness_p = compute_correlation(
                    {model: overall for model, overall in zip(models, overall_list_dict[key])},
                    {model: fairness_results[model]["individual_fairness"] for model in models}
                )
                method_results[aggregation][f"individual_fairness"][key] = (individual_fairness_corr, individual_fairness_p)
                individual_fairness_rank_corr, individual_fairness_rank_p = compute_rank_correlation(
                    {model: overall for model, overall in zip(models, overall_list_dict[key])},
                    {model: fairness_results[model]["individual_fairness"] for model in models}
                )
                method_results[aggregation][f"individual_fairness_rank"][key] = (individual_fairness_rank_corr, individual_fairness_rank_p)

                # accuracy reliance correlation
                accuracy_corr, accuracy_p = compute_correlation(
                    {model: reliance_diff for model, reliance_diff in zip(models, reliance_diff_list_dict[key])},
                    {model: fairness_results[model]["accuracy"] for model in models}
                )
                method_results[aggregation][f"accuracy"][key] = (accuracy_corr, accuracy_p)
                accuracy_rank_corr, accuracy_rank_p = compute_rank_correlation(
                    {model: reliance_diff for model, reliance_diff in zip(models, reliance_diff_list_dict[key])},
                    {model: fairness_results[model]["accuracy"] for model in models}
                )
                method_results[aggregation][f"accuracy_rank"][key] = (accuracy_rank_corr, accuracy_rank_p)

                # f1 reliance correlation
                f1_corr, f1_p = compute_correlation(
                    {model: reliance_diff for model, reliance_diff in zip(models, reliance_diff_list_dict[key])},
                    {model: fairness_results[model]["f1"] for model in models}
                )
                method_results[aggregation][f"f1"][key] = (f1_corr, f1_p)
                f1_rank_corr, f1_rank_p = compute_rank_correlation(
                    {model: reliance_diff for model, reliance_diff in zip(models, reliance_diff_list_dict[key])},
                    {model: fairness_results[model]["f1"] for model in models}
                )
                method_results[aggregation][f"f1_rank"][key] = (f1_rank_corr, f1_rank_p)

                # fpr reliance correlation
                fpr_corr, fpr_p = compute_correlation(
                    {model: reliance_diff_positive_prediction for model, reliance_diff_positive_prediction in zip(models, reliance_diff_positive_prediction_list_dict[key])},
                    {model: fairness_results[model]["fpr"] for model in models}
                )
                method_results[aggregation][f"fpr"][key] = (fpr_corr, fpr_p)
                fpr_rank_corr, fpr_rank_p = compute_rank_correlation(
                    {model: reliance_diff_positive_prediction for model, reliance_diff_positive_prediction in zip(models, reliance_diff_positive_prediction_list_dict[key])},
                    {model: fairness_results[model]["fpr"] for model in models}
                )
                method_results[aggregation][f"fpr_rank"][key] = (fpr_rank_corr, fpr_rank_p)
                # fnr reliance correlation
                fnr_corr, fnr_p = compute_correlation(
                    {model: reliance_diff_negative_prediction for model, reliance_diff_negative_prediction in zip(models, reliance_diff_negative_prediction_list_dict[key])},
                    {model: fairness_results[model]["fnr"] for model in models}
                )
                method_results[aggregation][f"fnr"][key] = (fnr_corr, fnr_p)
                fnr_rank_corr, fnr_rank_p = compute_rank_correlation(
                    {model: reliance_diff_negative_prediction for model, reliance_diff_negative_prediction in zip(models, reliance_diff_negative_prediction_list_dict[key])},
                    {model: fairness_results[model]["fnr"] for model in models}
                )
                method_results[aggregation][f"fnr_rank"][key] = (fnr_rank_corr, fnr_rank_p)

        correlation_results[method] = method_results
    return correlation_results

                

    
def main(args):
    
    if args.methods is not None:
        methods = args.methods.replace(' ', '').split(",")
    else:
        methods = EXPLANATION_METHODS

    if args.data_type is None or args.data_type not in ["civil", "jigsaw"]:
        if "civil" in args.results_dir:
            data_type= "civil"
        elif "jigsaw" in args.results_dir:
            data_type = "jigsaw"
        else:
            raise ValueError(f"Unknown data type in results_dir {args.results_dir}. Please use 'civil' or 'jigsaw'.")
    else:
        data_type = args.data_type

    if args.num_val_examples == -1:
        num_val_examples = 200
    else:
        num_val_examples = args.num_val_examples
    
    if args.num_test_examples == -1:
        if args.bias_type == "religion":
            num_test_examples = 1000
        else:
            num_test_examples = 2000
    else:
        num_test_examples = args.num_test_examples

    if args.experiment_type == "encoder":
        debiasing_methods = ["no_debiasing", "group_balance", "group_class_balance", "cda", "dropout", "attention_entropy", "causal_debias"]
    elif args.experiment_type == "decoder":
        debiasing_methods = ["zero_shot", "few_shot", "fairness_imagination", "fairness_instruction"]
    else:
        raise ValueError(f"Unknown experiment type {args.experiment_type}. Please choose from 'encoder' or 'decoder'.")
    
    models = debiasing_methods
    if args.experiment_type == "encoder" and args.train_type == "single":
        val_dirs = [os.path.join(args.val_results_dir, f"{args.model_type}_{data_type}_{args.bias_type}_{args.bias_type}_val_{num_val_examples}_{args.val_seed}" if args.val_seed != -1 else f"{args.model_type}_{data_type}_{args.bias_type}_{args.bias_type}_val_{num_val_examples}", debiasing_method) for debiasing_method in debiasing_methods]
        test_dirs = [os.path.join(args.results_dir, f"{args.model_type}_{data_type}_{args.bias_type}_{args.bias_type}_test_{num_test_examples}_{args.test_seed}" if args.test_seed != -1 else f"{args.model_type}_{data_type}_{args.bias_type}_{args.bias_type}_test_{num_test_examples}", debiasing_method) for debiasing_method in debiasing_methods]
    elif args.experiment_type == "encoder" and args.train_type == "all":
        val_dirs = [os.path.join(args.val_results_dir, f"{args.model_type}_{data_type}_all_{args.bias_type}_val_{num_val_examples}_{args.val_seed}" if args.val_seed != -1 else f"{args.model_type}_{data_type}_all_{args.bias_type}_val_{num_val_examples}", debiasing_method) for debiasing_method in debiasing_methods]
        test_dirs = [os.path.join(args.results_dir, f"{args.model_type}_{data_type}_all_{args.bias_type}_test_{num_test_examples}_{args.test_seed}" if args.test_seed != -1 else f"{args.model_type}_{data_type}_all_{args.bias_type}_test_{num_test_examples}", debiasing_method) for debiasing_method in debiasing_methods]
    elif args.experiment_type == "decoder":
        val_dirs = [os.path.join(args.val_results_dir, f"{args.model_type}_{data_type}_{args.bias_type}_val_{num_val_examples}_{args.val_seed}" if args.val_seed != -1 else f"{args.model_type}_{data_type}_{args.bias_type}_val_{num_val_examples}", debiasing_method) for debiasing_method in debiasing_methods]
        test_dirs = [os.path.join(args.results_dir, f"{args.model_type}_{data_type}_{args.bias_type}_test_{num_test_examples}_{args.test_seed}" if args.test_seed != -1 else f"{args.model_type}_{data_type}_{args.bias_type}_test_{num_test_examples}", debiasing_method) for debiasing_method in debiasing_methods]        
    #models = [f"individual_{debiasing_method}" for debiasing_method in debiasing_methods] + [f"all_{debiasing_method}" for debiasing_method in debiasing_methods]
    #val_dirs = [os.path.join(args.val_results_dir, f"{args.model_type}_{data_type}_{args.bias_type}_{args.bias_type}_val_{num_val_examples}_{args.val_seed}" if args.val_seed != -1 else f"{args.model_type}_{data_type}_{args.bias_type}_{args.bias_type}_val_{num_val_examples}", debiasing_method) for debiasing_method in debiasing_methods]
    #val_dirs += [os.path.join(args.val_results_dir, f"{args.model_type}_{data_type}_all_{args.bias_type}_val_{num_val_examples}_{args.val_seed}" if args.val_seed != -1 else f"{args.model_type}_{data_type}_all_{args.bias_type}_val_{num_val_examples}", debiasing_method) for debiasing_method in debiasing_methods]
    #test_dirs = [os.path.join(args.results_dir, f"{args.model_type}_{data_type}_{args.bias_type}_{args.bias_type}_test_{num_test_examples}_{args.test_seed}" if args.test_seed != -1 else f"{args.model_type}_{data_type}_{args.bias_type}_{args.bias_type}_test_{num_test_examples}", debiasing_method) for debiasing_method in debiasing_methods]
    #test_dirs += [os.path.join(args.results_dir, f"{args.model_type}_{data_type}_all_{args.bias_type}_test_{num_test_examples}_{args.test_seed}" if args.test_seed != -1 else f"{args.model_type}_{data_type}_all_{args.bias_type}_test_{num_test_examples}", debiasing_method) for debiasing_method in debiasing_methods]

    # fairness_results
    val_fairness_paths = [os.path.join(val_path, "fairness", f"fairness_{args.bias_type}_val_summary_stats.json") for val_path in val_dirs]
    test_fairness_paths = [os.path.join(test_path, "fairness", f"fairness_{args.bias_type}_test_summary_stats.json") for test_path in test_dirs]

    # reliance_results
    val_reliance_dirs = [os.path.join(val_path, "reliance_statistics") for val_path in val_dirs]
    test_reliance_dirs = [os.path.join(test_path, "reliance_statistics") for test_path in test_dirs]

    if args.val_seed == -1 and args.test_seed == -1:
        if args.experiment_type == "encoder":
            correlation_results_dir = os.path.join(args.results_dir, "model_selection_correlation", f"{args.model_type}_{args.bias_type}_{args.train_type}_val_{num_val_examples}_test_{num_test_examples}")
        else:
            correlation_results_dir = os.path.join(args.results_dir, "model_selection_correlation", f"{args.model_type}_{args.bias_type}_val_{num_val_examples}_test_{num_test_examples}")
    elif args.val_seed == -1 and args.test_seed != -1:
        if args.experiment_type == "encoder":
            correlation_results_dir = os.path.join(args.results_dir, "model_selection_correlation", f"{args.model_type}_{args.bias_type}_{args.train_type}_val_{num_val_examples}_test_{num_test_examples}_seed_{args.test_seed}")
        else:
            correlation_results_dir = os.path.join(args.results_dir, "model_selection_correlation", f"{args.model_type}_{args.bias_type}_val_{num_val_examples}_test_{num_test_examples}_seed_{args.test_seed}")
    elif args.val_seed != -1 and args.test_seed == -1:
        if args.experiment_type == "encoder":
            correlation_results_dir = os.path.join(args.results_dir, "model_selection_correlation", f"{args.model_type}_{args.bias_type}_{args.train_type}_val_{num_val_examples}_seed_{args.val_seed}_test_{num_test_examples}")
        else:
            correlation_results_dir = os.path.join(args.results_dir, "model_selection_correlation", f"{args.model_type}_{args.bias_type}_val_{num_val_examples}_seed_{args.val_seed}_test_{num_test_examples}")
    else:
        if args.experiment_type == "encoder":
            correlation_results_dir = os.path.join(args.results_dir, "model_selection_correlation", f"{args.model_type}_{args.bias_type}_{args.train_type}_val_{num_val_examples}_seed_{args.val_seed}_test_{num_test_examples}_seed_{args.test_seed}")
        else:
            correlation_results_dir = os.path.join(args.results_dir, "model_selection_correlation", f"{args.model_type}_{args.bias_type}_val_{num_val_examples}_seed_{args.val_seed}_test_{num_test_examples}_seed_{args.test_seed}")

    if not os.path.exists(correlation_results_dir):
        os.makedirs(correlation_results_dir)

    
    # load fairness results    
    print(f"Loading fairness results for {args.model_type} {args.bias_type}...")
    val_fairness_results = load_fairness_results(val_fairness_paths, args.bias_type, models)
    test_fairness_results = load_fairness_results(test_fairness_paths, args.bias_type, models)
 
    # compute correlation for fairness results
    print(f"Computing fairness correlation for {args.model_type} {args.bias_type}...")
    fairness_correlation_results = compute_fairness_correlation(val_fairness_results, test_fairness_results)

    # load val and test reliance results
    print(f"Loading reliance results for {args.model_type} {args.bias_type}...")
    val_reliance_results = load_reliance_results(val_reliance_dirs, methods, args.bias_type, models, split="val")
    test_reliance_results = load_reliance_results(test_reliance_dirs, methods, args.bias_type, models, split="test")

    # compute correlation for reliance results
    print(f"Computing reliance correlation for {args.model_type} {args.bias_type}...")
    val_reliance_correlation_results = compute_reliance_correlation(val_reliance_results, test_fairness_results)
    test_reliance_correlation_results = compute_reliance_correlation(test_reliance_results, test_fairness_results)

    # save results
    with open(os.path.join(correlation_results_dir, "fairness_correlation_results.json"), 'w') as f:
        json.dump(fairness_correlation_results, f, indent=4)
    for method in val_reliance_correlation_results.keys():
        with open(os.path.join(correlation_results_dir, f"{method}_{args.bias_type}_val_reliance_correlation_results.json"), 'w') as f:
            json.dump(val_reliance_correlation_results[method], f, indent=4)
        with open(os.path.join(correlation_results_dir, f"{method}_{args.bias_type}_test_reliance_correlation_results.json"), 'w') as f:
            json.dump(test_reliance_correlation_results[method], f, indent=4)
    
    
    print(f"{args.model_type} {args.bias_type} model selection correlation results saved to {correlation_results_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='BERT Attribution with Captum')

    parser.add_argument('--results_dir', type=str, default='baseline_saliency_results/all_methods_1000_examples_512', help='Path to the saliency data')
    parser.add_argument('--val_results_dir', type=str, default='baseline_saliency_results/all_methods_1000_examples_512', help='Path to the saliency data')
    parser.add_argument('--experiment_type', type=str, default='encoder', choices=['encoder', 'decoder'], help='Type of model to use')
    parser.add_argument('--methods', type=str, default=None, help='List of attribution methods to use separated by commas')
    parser.add_argument('--bias_type', type=str, default="race", choices=["race", "gender", "religion"])
    parser.add_argument("--model_type", type=str, default="bert", choices=["bert", "roberta", "distilbert", "llama", "qwen"], help="Type of model to use")
    parser.add_argument("--num_val_examples", type=int, default=-1, help="Number of examples to use for validation")
    parser.add_argument("--num_test_examples", type=int, default=-1, help="Number of examples to use for testing")
    parser.add_argument("--test_seed", type=int, default=-1, help="Seed for test set shuffling")
    parser.add_argument("--val_seed", type=int, default=-1, help="Seed for validation set shuffling")
    parser.add_argument("--data_type", type=str, default=None, help="Type of data to use for model selection")
    parser.add_argument("--train_type", type=str, default="single", choices=["single", "all"],)
    args = parser.parse_args()
    main(args)
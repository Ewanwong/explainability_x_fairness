from utils.utils import set_random_seed, compute_metrics, apply_dataset_perturbation, to_builtin_number
from utils.dataset_utils import customized_load_dataset, customized_split_dataset, load_subsets_for_fairness_explainability
import argparse
import torch
from torch.utils.data import DataLoader, Subset
from transformers import AutoTokenizer, AutoConfig, AutoModelForSequenceClassification, AutoModelForCausalLM
import numpy as np
import json
import os
import random
from tqdm import tqdm

from utils.vocabulary import *
from utils.prompt import *


def make_predictions(model, tokenizer, batch, model_type, template, prompt):
    if model_type == "encoder":
        inputs = tokenizer(batch['text'], padding=True, truncation=True, max_length=args.max_seq_length, return_tensors='pt')
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        labels = batch['label'].numpy()
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            probs = torch.nn.functional.softmax(logits, dim=-1)
            predictions = torch.argmax(probs, dim=-1).cpu().numpy()
            confidences_class_0 = probs[:, 0].cpu().numpy()
            confidences_class_1 = probs[:, 1].cpu().numpy()
            confidences_predicted_class = probs[torch.arange(probs.size(0)), torch.argmax(probs, dim=-1)].cpu().numpy()
    elif model_type == "decoder":
        positive_token = "Yes"
        negative_token = "No"
        positive_token_id = tokenizer(positive_token, add_special_tokens=False)["input_ids"][0]
        negative_token_id = tokenizer(negative_token, add_special_tokens=False)["input_ids"][0]
        text = tokenizer.apply_chat_template(fill_in_template(template, prompt.replace("[TEST EXAMPLE]", batch['text'][0])),tokenize=False,add_generation_prompt=True, enable_thinking=False, date_string="2025-07-01")
        inputs = tokenizer(text, return_tensors='pt')
        device = model.get_input_embeddings().weight.device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        labels = batch['label'].numpy() if type(batch['label']) is not list else batch['label']
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits[:, -1, :]
            probs = torch.softmax(logits, dim=-1)
            predictions = 1 if probs[0, positive_token_id] > probs[0, negative_token_id] else 0
            confidences_class_0 = probs[0, negative_token_id].cpu().numpy()
            confidences_class_1 = probs[0, positive_token_id].cpu().numpy()
            confidences_predicted_class = probs[0, positive_token_id].cpu().numpy() if predictions == 1 else probs[0, negative_token_id].cpu().numpy()
            predictions = np.array([predictions])
            confidences_class_0 = np.array([confidences_class_0])
            confidences_class_1 = np.array([confidences_class_1])
            confidences_predicted_class = np.array([confidences_predicted_class])
    else:
        raise ValueError("Unsupported model type. Please use 'encoder' or 'decoder'.")
    return labels, predictions, confidences_class_0, confidences_class_1, confidences_predicted_class
    
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
    
    if args.prompt_type == "zero_shot":
        SYSTEM_PROMPT = construct_zero_shot_prompt(dataset, args.bias_type)
    elif args.prompt_type == "few_shot":
        SYSTEM_PROMPT = construct_few_shot_prompt(dataset, args.bias_type)
    elif args.prompt_type == "fairness_imagination":
        SYSTEM_PROMPT = construct_fairness_imagination_prompt(dataset, args.bias_type)
    elif args.prompt_type == "fairness_instruction":
        SYSTEM_PROMPT = construct_fairness_instruction_prompt(dataset, args.bias_type)
    else:
        raise ValueError(f"Unknown prompt type: {args.prompt_type}")

    # Load the model
    model_type = None
    if "bert" in args.model_dir.lower():
        model = AutoModelForSequenceClassification.from_pretrained(args.model_dir)
        model_type = "encoder"
    elif "llama" in args.model_dir.lower() or "qwen" in args.model_dir.lower():
        model = AutoModelForCausalLM.from_pretrained(args.model_dir, device_map="auto")
        args.batch_size = 1
        model_type = "decoder"
    model.eval()
    #model.to(device)

    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)

    # Load a dataset from HuggingFace datasets library
    print("Loading dataset...")
    
    target_datasets = load_subsets_for_fairness_explainability(args.dataset_name, args.split, args.split_ratio, args.bias_type, args.num_examples, shuffle=args.shuffle, seed=args.seed)


    # make predictions
    groups = SOCIAL_GROUPS[args.bias_type]
    predictions = {group: {} for group in groups} # predictions[group1][group2] = (list) predictions on subset of group1 after perturbing to group2 (group1==group2: results on the original group1 subset)
    labels = {} # labels[group] = (list) labels of the original group1 subset
    prediction_confidences = {group: {group2: {} for group2 in groups} for group in groups} # prediction_confidences[group][group2] = (dict) confidence scores of the class 0/class 1/predicted class on subset of group1 after perturbing to group2 (group1==group2: results on the original group1 subset)
    
    for group in groups:
        group_dataset = target_datasets[group]
        group_dataloader = DataLoader(group_dataset, batch_size=args.batch_size, shuffle=False)
        group_labels = []
        group_predictions = []
        group_confidences_class_0 = []
        group_confidences_class_1 = []
        group_confidences_predicted_class = []
        for batch in tqdm(group_dataloader, desc=f"Predicting {group}"):
            group_labels_batch, group_predictions_batch, group_confidences_class_0_batch, group_confidences_class_1_batch, group_confidences_predicted_class_batch = make_predictions(model, tokenizer, batch, model_type, template=TEMPLATE, prompt=SYSTEM_PROMPT)
            group_labels.extend(group_labels_batch)
            group_predictions.extend(group_predictions_batch)
            group_confidences_class_0.extend(group_confidences_class_0_batch)
            group_confidences_class_1.extend(group_confidences_class_1_batch)
            group_confidences_predicted_class.extend(group_confidences_predicted_class_batch)
                    
        predictions[group][group] = group_predictions
        labels[group] = group_labels
        prediction_confidences[group][group]['class_0'] = group_confidences_class_0
        prediction_confidences[group][group]['class_1'] = group_confidences_class_1
        prediction_confidences[group][group]['predicted_class'] = group_confidences_predicted_class


        # compute predictions for the perturbed datasets
        perturbed_group_datasets = apply_dataset_perturbation(group_dataset, group, args.bias_type)
        for perturbed_group in perturbed_group_datasets.keys():
            perturbed_group_dataset = perturbed_group_datasets[perturbed_group]
            perturbed_group_dataloader = DataLoader(perturbed_group_dataset, batch_size=args.batch_size, shuffle=False)
            perturbed_group_labels = []
            perturbed_group_predictions = []
            perturbed_group_confidences_class_0 = []
            perturbed_group_confidences_class_1 = []
            perturbed_group_confidences_predicted_class = []
            for batch in tqdm(perturbed_group_dataloader, desc=f"Predicting perturbed {perturbed_group}"):
                perturbed_group_labels_batch, perturbed_group_predictions_batch, perturbed_group_confidences_class_0_batch, perturbed_group_confidences_class_1_batch, perturbed_group_confidences_predicted_class_batch = make_predictions(model, tokenizer, batch, model_type, template=TEMPLATE, prompt=SYSTEM_PROMPT)
                perturbed_group_labels.extend(perturbed_group_labels_batch)
                perturbed_group_predictions.extend(perturbed_group_predictions_batch)
                perturbed_group_confidences_class_0.extend(perturbed_group_confidences_class_0_batch)
                perturbed_group_confidences_class_1.extend(perturbed_group_confidences_class_1_batch)
                perturbed_group_confidences_predicted_class.extend(perturbed_group_confidences_predicted_class_batch)
            predictions[group][perturbed_group] = perturbed_group_predictions
            prediction_confidences[group][perturbed_group]['class_0'] = perturbed_group_confidences_class_0
            prediction_confidences[group][perturbed_group]['class_1'] = perturbed_group_confidences_class_1
            prediction_confidences[group][perturbed_group]['predicted_class'] = perturbed_group_confidences_predicted_class

    # compute group fairness and individual fairness results
    # group-wise accuracy/f1/tnr/fpr/fnr/tpr
    metrics = {}
    for group in groups:
        group_metrics = compute_metrics(labels[group], predictions[group][group], num_classes=2)
        metrics[group] = {}
        metrics[group]["num_examples"] = len(labels[group])
        metrics[group]['accuracy'] = group_metrics['accuracy']
        metrics[group]['f1'] = group_metrics['f1']
        metrics[group]['tpr'] = group_metrics[f"class_1"]['tpr']
        metrics[group]['tnr'] = group_metrics[f"class_1"]['tnr']
        metrics[group]['fpr'] = group_metrics[f"class_1"]['fpr']
        metrics[group]['fnr'] = group_metrics[f"class_1"]['fnr']

    
    metrics["average"] = {}
    metrics["average"]["num_examples"] = sum([metrics[group]["num_examples"] for group in groups])
    for metric in ['accuracy', 'f1', 'tpr', 'tnr', 'fpr', 'fnr']:
        metrics["average"][metric] = np.mean([metrics[group][metric] for group in groups])

    metrics['overall'] = {}
    metrics["overall"]["num_examples"] = sum([metrics[group]["num_examples"] for group in groups])
    overall_labels = [label for group in groups for label in labels[group]]
    overall_predictions = [prediction for group in groups for prediction in predictions[group][group]]
    overall_metrics = compute_metrics(overall_labels, overall_predictions, num_classes=2)
    metrics['overall']['accuracy'] = overall_metrics['accuracy']
    metrics['overall']['f1'] = overall_metrics['f1']
    metrics['overall']['tpr'] = overall_metrics[f"class_1"]['tpr']
    metrics['overall']['tnr'] = overall_metrics[f"class_1"]['tnr']
    metrics['overall']['fpr'] = overall_metrics[f"class_1"]['fpr']
    metrics['overall']['fnr'] = overall_metrics[f"class_1"]['fnr']

    # group fairness: group metric - average metric
    group_fairness = {"average": {group: {} for group in groups}, "overall": {group: {} for group in groups}}
    for group in groups:
        for metric in ['accuracy', 'f1', 'tpr', 'tnr', 'fpr', 'fnr']:
            group_fairness["average"][group][metric] = metrics[group][metric] - metrics["average"][metric]
            group_fairness["overall"][group][metric] = metrics[group][metric] - metrics['overall'][metric]

    # individual fairness: prediction confidence - average prediction confidence across original and perturbed groups
    for group in groups:
        prediction_confidences[group]["average"] = {}
        prediction_confidences[group]["average"]["class_0"] = np.mean([prediction_confidences[group][perturbed_group]['class_0'] for perturbed_group in groups], axis=0)
        prediction_confidences[group]["average"]["class_1"] = np.mean([prediction_confidences[group][perturbed_group]['class_1'] for perturbed_group in groups], axis=0)
        predicted_classes = predictions[group][group]
        ## TODO: check whether this is correct
        prediction_confidences[group]["average"]["predicted_class"] = np.mean([[prediction_confidences[group][perturbed_group][f'class_{i}'][idx] for idx, i in enumerate(predicted_classes)] for perturbed_group in groups], axis=0)

    individual_fairness = {}
    individual_prediction_differences = {}
    for group in groups:
        individual_prediction_differences[group] = {}
        individual_prediction_differences[group]['class_0'] = prediction_confidences[group][group]['class_0'] - prediction_confidences[group]["average"]['class_0']
        individual_prediction_differences[group]['class_1'] = prediction_confidences[group][group]['class_1'] - prediction_confidences[group]["average"]['class_1']
        individual_prediction_differences[group]['predicted_class'] = prediction_confidences[group][group]['predicted_class'] - prediction_confidences[group]["average"]['predicted_class']
        individual_fairness[group] = {}
        for metric in ['class_0', 'class_1', 'predicted_class']:
            individual_fairness[group][metric] = {"average": np.mean(individual_prediction_differences[group][metric]), "abs_average": np.mean(np.abs(individual_prediction_differences[group][metric]))}
    individual_fairness["overall"] = {}
    for metric in ['class_0', 'class_1', 'predicted_class']:
        individual_fairness["overall"][metric] = {"average": np.mean([individual_fairness[group][metric]['average'] for group in groups]), "abs_average": np.mean([individual_fairness[group][metric]['abs_average'] for group in groups])}
    # convert all np float to float
    metrics = to_builtin_number(metrics)
    group_fairness = to_builtin_number(group_fairness)
    individual_fairness = to_builtin_number(individual_fairness)
     
    predictions = to_builtin_number(predictions)
    labels = to_builtin_number(labels)
    prediction_confidences = to_builtin_number(prediction_confidences)
    individual_prediction_differences = to_builtin_number(individual_prediction_differences)

    # Save the results to a JSON file
    output_dir = os.path.join(args.output_dir, "fairness")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    output_summary_file = os.path.join(output_dir, f'fairness_{args.bias_type}_{args.split}_summary_stats.json')
    with open(output_summary_file, 'w') as f:
        json.dump({"Metrics": metrics, "Group_Fairness": group_fairness, "Individual_Fairness": individual_fairness}, f, indent=4)
    print(f"\nFairness results saved to {output_summary_file}")
    # Save the predictions to a JSON file
    output_predictions_file = os.path.join(output_dir, f'fairness_{args.bias_type}_{args.split}_individual_stats.json')
    with open(output_predictions_file, 'w') as f:
        json.dump({"Predictions": predictions, "Labels": labels, "Prediction_Confidences": prediction_confidences, "Individual_Differences": individual_prediction_differences}, f, indent=4)
    print(f"\nPredictions saved to {output_predictions_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='BERT Attribution with Captum')

    parser.add_argument('--dataset_name', type=str, default='stanfordnlp/imdb', help='Name of the HuggingFace dataset to use') #fancyzhx/ag_news, stanfordnlp/imdb
    parser.add_argument('--split', type=str, default='test', help='Dataset split to use (e.g., train, test)')
    parser.add_argument('--split_ratio', type=str, default="0.8, 0.2",
                    help='Ratio to split the train set into train and validation sets')
    parser.add_argument('--model_dir', type=str, default='models/bert_base_imdb_512', help='Name of the pre-trained model')
    #parser.add_argument('--num_labels', type=int, default=2, help='Number of labels in the classification')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size for DataLoader')
    parser.add_argument('--max_seq_length', type=int, default=512, help='Maximum sequence length for tokenization')
    parser.add_argument('--num_examples', type=int, default=1000, help='Number of examples to process (-1 for all)')
    parser.add_argument('--output_dir', type=str, default='baseline_saliency_results/all_methods_1000_examples_512', help='Directory to save the output files')
    parser.add_argument('--shuffle', action='store_true', help='Whether to shuffle the dataset before processing')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--bias_type', type=str, default="race", choices=["race", "gender", "religion"])
    parser.add_argument('--prompt_type', type=str, default="zero_shot", choices=["zero_shot", "few_shot", "fairness_imagination", "fairness_instruction"], help='Type of prompt to use for the model')

    args = parser.parse_args()
    main(args)

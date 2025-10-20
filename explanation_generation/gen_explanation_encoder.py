from explainer.Explainer_Encoder import BcosExplainer, AttentionExplainer, GradientNPropabationExplainer, OcclusionExplainer, ShapleyValueExplainer, LimeExplainer
from utils.utils import set_random_seed
from utils.dataset_utils import customized_load_dataset, customized_split_dataset, load_subsets_for_fairness_explainability
import argparse
import torch
from torch.utils.data import DataLoader, Subset
from transformers import AutoTokenizer, AutoConfig, AutoModelForSequenceClassification

import numpy as np
import json
import os
import random
from tqdm import tqdm

from utils.vocabulary import *

EXPLANATION_METHODS_MAPPING = {
    "Bcos": BcosExplainer,
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
    "DecompX": None,  # DecompX has its own implementation
}

def main(args):

    # Set random seed for reproducibility
    set_random_seed(args.seed)
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the model
    model = AutoModelForSequenceClassification.from_pretrained(args.model_dir, output_attentions=True)
    model.eval()
    model.to(device)

    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
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
        if method == "DecompX":
            continue  # Skip DecompX here, handle it separately
        print(f"\nRunning {method} explainer...")
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
            if method == "IntegratedGradients" and args.batch_size > 4:
                # IntegratedGradients requires more memory, so we reduce the batch size
                batch_size = 4
            else:
                batch_size = args.batch_size
            explanation_results = explainer.explain_dataset(target_dataset, num_classes=args.num_labels, batch_size=batch_size, max_length=args.max_length, only_predicted_classes=args.only_predicted_classes)
            result = explanation_results

            # Save the results to a JSON file
            output_file = os.path.join(output_dir, f'{method}_{group}_{args.split}_explanations.json')
            with open(output_file, 'w') as f:
                json.dump(result, f, indent=4)
            print(f"\nAttribution results saved to {output_file}")

    if "DecompX" in attribution_methods:
        # delete the model to free up memory
        del model
        torch.cuda.empty_cache()
        print("\nRunning DecompX explainer...")
        from DecompX.src.decompx_utils import DecompXConfig
        from DecompX.src.modeling_bert import BertForSequenceClassification
        from DecompX.src.modeling_roberta import RobertaForSequenceClassification

        CONFIGS = {
            "DecompX":
                DecompXConfig(
                    include_biases=True,
                    bias_decomp_type="absdot",
                    include_LN1=True,
                    include_FFN=True,
                    FFN_approx_type="GeLU_ZO",
                    include_LN2=True,
                    aggregation="vector",
                    include_classifier_w_pooler=True,
                    tanh_approx_type="ZO",
                    output_all_layers=True,
                    output_attention=None,
                    output_res1=None,
                    output_LN1=None,
                    output_FFN=None,
                    output_res2=None,
                    output_encoder=None,
                    output_aggregated="norm",
                    output_pooler="norm",
                    output_classifier=True,
                ),
        }

        def batch_loader(dataset, batch_size, shuffle=False):
            if type(dataset) == dict:
                # get the length of the dataset
                length = len(dataset[list(dataset.keys())[0]])
                indices = list(range(length))
                if shuffle:
                    random.shuffle(indices)
                for i in range(0, length, batch_size):
                    batch_indices = indices[i:i+batch_size]
                    batch = {key: [dataset[key][j] for j in batch_indices] for key in dataset.keys()}
                    yield batch
            elif type(dataset) == list:
                length = len(dataset)
                indices = list(range(length))
                if shuffle:
                    random.shuffle(indices)
                for i in range(0, length, batch_size):
                    batch_indices = indices[i:i+batch_size]
                    batch = [dataset[j] for j in batch_indices]
                    yield batch
        
        MODEL = args.model_dir  # Only BERT or RoBERTa
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if "roberta" in MODEL:
            model = RobertaForSequenceClassification.from_pretrained(MODEL).to(device)
        elif "bert" in MODEL:
            model = BertForSequenceClassification.from_pretrained(MODEL).to(device)
        else:
            raise Exception(f"Not implented model: {MODEL}")
        
        output_dir = os.path.join(args.output_dir, "explanations")
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        for group in target_datasets.keys():
            target_dataset = target_datasets[group].to_dict()
            target_dataset['index'] = list(range(len(target_dataset['text'])))
            test_dataloader = batch_loader(target_dataset, args.batch_size, shuffle=False)
            decompx_results = []

            for batch in tqdm(test_dataloader):
                texts = batch['text']
                example_indices = batch['index']
                labels = batch['label']

                batch_size = len(texts)
            
                tokenized_sentence = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=args.max_length)
                tokenized_sentence = {k: v.to(device) for k, v in tokenized_sentence.items()}
                real_lengths = tokenized_sentence['attention_mask'].sum(dim=-1)


                with torch.no_grad():
                    model.eval()
                    logits, hidden_states, decompx_last_layer_outputs, decompx_all_layers_outputs = model(
                        **tokenized_sentence, 
                        output_attentions=False, 
                        return_dict=False, 
                        output_hidden_states=True, 
                        decompx_config=CONFIGS["DecompX"]
                    )

                predicted_classes = logits.argmax(dim=-1)
                predicted_confidence = torch.softmax(logits, dim=-1).max(dim=-1).values

                importance = np.array([g.squeeze().cpu().detach().numpy() for g in decompx_last_layer_outputs.classifier]).squeeze()  # (batch, seq_len, classes)
                if importance.ndim == 2:
                    importance = importance[np.newaxis, :]
                importance = [importance[j][:real_lengths[j], :] for j in range(len(importance))]
                #decompx_outputs["importance_last_layer_classifier"] = importance

                for i in range(batch_size):
                    tokens = tokenizer.convert_ids_to_tokens(tokenized_sentence["input_ids"][i][:real_lengths[i]])
                    decompx_results.append(
                        [
                            {
                                "index": example_indices[i],
                                "text": tokenizer.decode([t for t in tokenized_sentence["input_ids"][i][:real_lengths[i]] if not (t in tokenizer.all_special_ids and t != tokenizer.unk_token_id)], skip_special_tokens=False),
                                "true_label": labels[i],
                                "predicted_class": predicted_classes[i].item(),
                                "predicted_class_confidence": predicted_confidence[i].item(),
                                "target_class": predicted_classes[i].item(),
                                "target_class_confidence": predicted_confidence[i].item(),
                                "method": "DecompX",
                                "attribution": list(zip(tokens, importance[i][:, predicted_classes[i].item()].tolist())),
                            }
                        ]
                    )

            decompx_results = {"DecompX": decompx_results}

            # Save the results to a JSON file
            output_file = os.path.join(output_dir, f'DecompX_{group}_{args.split}_explanations.json')
            with open(output_file, 'w') as f:
                json.dump(decompx_results, f, indent=4)
            print(f"\nAttribution results saved to {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='BERT Attribution with Captum')

    parser.add_argument('--dataset_name', type=str, default='stanfordnlp/imdb', help='Name of the HuggingFace dataset to use') #fancyzhx/ag_news, stanfordnlp/imdb
    parser.add_argument('--split', type=str, default='test', help='Dataset split to use (e.g., train, test)')
    parser.add_argument('--split_ratio', type=str, default="0.8, 0.2",
                    help='Ratio to split the train set into train and validation sets')
    parser.add_argument('--model_dir', type=str, default='models/bert_base_imdb_512', help='Name of the pre-trained model')
    parser.add_argument('--num_labels', type=int, default=2, help='Number of labels in the classification')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size for DataLoader')
    parser.add_argument('--max_length', type=int, default=512, help='Maximum sequence length for tokenization')
    parser.add_argument('--baseline', type=str, default='pad', help='Baseline for the attribution methods, select from zero, mask, pad')    
    parser.add_argument('--num_examples', type=int, default=1000, help='Number of examples to process (-1 for all)')
    parser.add_argument('--methods', type=str, default=None, help='List of attribution methods to use separated by commas')
    parser.add_argument('--output_dir', type=str, default='baseline_saliency_results/all_methods_1000_examples_512', help='Directory to save the output files')
    parser.add_argument('--shuffle', action='store_true', help='Shuffle the dataset before processing')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--shap_n_samples', type=int, default=25, help='Number of samples for Shapley Value Sampling')
    parser.add_argument('--only_predicted_classes', action='store_true', help='Only explain the predicted class')
    parser.add_argument('--bias_type', type=str, default="race", choices=["race", "gender", "religion"])


    args = parser.parse_args()
    main(args)

import random
import re
import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, Subset
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import torch

from transformers import AutoTokenizer
from collections import defaultdict
import string

def aggregate_token_scores(scores, token_selection="max"):

    if len(scores) == 1:
        return scores[0]
    if token_selection == "sum":
        return np.sum(scores)
    elif token_selection == "mean":
        return np.mean(scores)
    elif token_selection == "max":
        return scores[np.argmax(np.abs(scores))]
    else:
        raise ValueError(f"Unrecognized token_selection method: {token_selection}")


def compute_reliance_score(sensitive_attribution, total_attribution, method="normalized", token_selection="max"):
    
    # TODO: make sure sensitive attribution scores are not empty
    if len(sensitive_attribution) == 0:
        # print("Sensitive attribution is empty")
        return 0.0
    sensitive_attribution_scores = np.array([aggregate_token_scores(attribution_score[1], token_selection) for attribution_score in sensitive_attribution])

    #[[token, score], [token, score]]
    total_attribution_scores = np.array([attribution_score[1] for attribution_score in total_attribution])

    # select the sensitive attribution score with the largest magnitude
    sensitive_attribution_score =  sensitive_attribution_scores[np.argmax(np.abs(sensitive_attribution_scores))]
    if method == "raw":
        return sensitive_attribution_score

    elif method == "max":
        scale_factor = np.max(np.abs(total_attribution_scores))
    elif method == "len":
        scale_factor = len(total_attribution_scores)
    elif method == "norm":
        scale_factor = np.linalg.norm(total_attribution_scores)
    else:
        raise ValueError("Method not recognized")
    normalized_sensitive_attribution_score = sensitive_attribution_score / scale_factor
    return normalized_sensitive_attribution_score

def extract_sensitive_attributions(explanations, sensitive_tokens, tokenizer):
    # special tokens for the model
    special_tokens = set(tokenizer.all_special_tokens)

    # 构建双版本 token 模式（带空格 / 不带空格）
    sensitive_token_patterns = []
    for word in sensitive_tokens:
        toks1 = tokenizer.tokenize(word, add_special_tokens=False)
        toks2 = tokenizer.tokenize(" " + word, add_special_tokens=False)
        
        sensitive_token_patterns.append(toks1)
        if toks1 != toks2:
            sensitive_token_patterns.append(toks2)

    results = []

    for explanations_per_sample in explanations:
        index = explanations_per_sample[0]["index"]
        predicted_class = explanations_per_sample[0]["predicted_class"]
        true_label = explanations_per_sample[0]["true_label"]
        example_result = {}
        example_result["index"] = index
        example_result["prediction"] = predicted_class
        example_result["label"] = true_label

        for expl_id in range(len(explanations_per_sample)):
            target_class = explanations_per_sample[expl_id]["target_class"]
            attribution_scores = explanations_per_sample[expl_id]["attribution"]
            tokens = [tok for tok, _ in attribution_scores]
            scores = [score for _, score in attribution_scores]
            total_attribution = list(zip(tokens, scores))

            # 3) 滑动窗口敏感 token 匹配
            sensitive_attr = []
            matched_token_positions = set()

            i = 0
            while i < len(tokens):
                if tokens[i] in special_tokens:
                    i += 1
                    continue
                found = False
                for tok_seq in sorted(sensitive_token_patterns, key=lambda x: -len(x)):
                    n = len(tok_seq)
                    if i + n <= len(tokens) and tokens[i:i+n] == tok_seq:
                        if any(j in matched_token_positions for j in range(i, i+n)):
                            continue  # 已经匹配，跳过
                        # word_scores = [scores[j] for j in range(i, i+n)]

                        sensitive_attr.append([tokens[i:i+n], scores[i:i+n]])

                        matched_token_positions.update(range(i, i+n))
                        i += n
                        found = True
                        break
                if not found:
                    i += 1

            # to print sentences without being matched -> may contain sensitive words not contained in the list
            # if len(sensitive_attr) == 0:
            #     print(f"\nNo sensitive tokens matched in sample index {index}, class {target_class}")
            #     print("Tokens:", " ".join(tokens))

            example_result[f"class_{target_class}"] = {
                "sensitive_attribution": sensitive_attr,
                "total_attribution": total_attribution
            }
        example_result["predicted_class"] = example_result[f"class_{predicted_class}"].copy()
        results.append(example_result)

    return results


def _get_sensitive_token_mask(input_ids, attention_mask, target_group, sensitive_patterns_dict, tokenizer, special_tokens):
    # input ids and attention masks are lists here, not tensors
    sensitive_token_patterns = sensitive_patterns_dict[target_group]
    sensitive_token_mask = [0] * len(attention_mask)
    tokens = [tokenizer.decode(input_id) for input_id in input_ids]
    matched_token_positions = set()
    real_length = sum(attention_mask)
    i = 0
    while i < real_length:
        if tokens[i] in special_tokens:
            i += 1
            continue
        found = False
        for tok_seq in sorted(sensitive_token_patterns, key=lambda x: -len(x)):
            n = len(tok_seq)
            if i + n <= len(tokens) and tokens[i:i+n] == tok_seq:
                if any(j in matched_token_positions for j in range(i, i+n)):
                    continue  
                sensitive_token_mask[i:i+n] = [1] * n
                matched_token_positions.update(range(i, i+n))
                i += n
                found = True
                break
        if not found:
            i += 1

    return sensitive_token_mask

def get_sensitive_token_mask(dataset, target_groups, tokenizer, sensitive_tokens_dict):
    all_input_ids = dataset["input_ids"]
    all_attention_mask = dataset["attention_mask"]
    all_groups = dataset["group"]

    special_tokens = set(tokenizer.all_special_tokens)
    sensitive_patterns_dict = {}
    bias_types = sensitive_tokens_dict.keys()
    for bias in bias_types:
        groups = sensitive_tokens_dict[bias].keys()
        for group in groups:
            if group not in target_groups:
                continue
            sensitive_tokens = sensitive_tokens_dict[bias][group]
            sensitive_token_patterns = []
            for word in sensitive_tokens:
                toks1 = tokenizer.tokenize(word, add_special_tokens=False)
                toks2 = tokenizer.tokenize(" " + word, add_special_tokens=False)
                
                sensitive_token_patterns.append(toks1)
                if toks1 != toks2:
                    sensitive_token_patterns.append(toks2)
            sensitive_patterns_dict[group] = sensitive_token_patterns
    
    all_sensitive_token_masks = []

    for input_ids, attention_mask, group in zip(all_input_ids, all_attention_mask, all_groups):
        sensitive_token_mask = _get_sensitive_token_mask(input_ids, attention_mask, group, sensitive_patterns_dict, tokenizer, special_tokens)
        all_sensitive_token_masks.append(sensitive_token_mask)

    # all_sensitive_token_masks = torch.stack(all_sensitive_token_masks, dim=0).numpy()
    dataset = dataset.add_column("sensitive_token_mask", all_sensitive_token_masks)
    return dataset



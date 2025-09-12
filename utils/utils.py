from utils.vocabulary import *
import random
import re
import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, Subset
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from datasets import Dataset
from collections import defaultdict
import json

import numpy as np
JIGSAW_DATASET_DIR_RAW = "/scratch/yifwang/datasets/jigsaw_raw"
JIGSAW_DATASET_DIR_PROCESSED = "/scratch/yifwang/datasets/jigsaw_processed"

EXPLANATION_METHODS = [
    "Bcos",
    "Attention",
    "Saliency",
    "DeepLift",
    #"GuidedBackprop",
    "InputXGradient",
    "IntegratedGradients",
    #"SIG",
    "Occlusion",
    #"ShapleyValue",
    "KernelShap",
    #"Lime",
]

def to_builtin_number(obj):
    """Recursively replace NumPy numbers/arrays with plain Python numbers/lists."""
    if isinstance(obj, dict):                           # ── dive into mappings
        return {k: to_builtin_number(v) for k, v in obj.items()}

    if isinstance(obj, (list, tuple, set)):             # ── dive into sequences
        return type(obj)(to_builtin_number(x) for x in obj)

    if isinstance(obj, np.ndarray):                     # ── whole array → list
        return [to_builtin_number(x) for x in obj.tolist()]

    if isinstance(obj, (np.floating, np.integer)):      # ── NumPy scalar → int/float
        return obj.item()                               # same as int() or float()

    # (optional) handle pandas objects, dataclasses, etc. here …

    return obj              

def compute_metrics(labels, predictions, num_classes=2):
    metrics_dict = {}
    accuracy = accuracy_score(labels, predictions)
    f1 = f1_score(labels, predictions, average='macro')

    metrics_dict["accuracy"] = accuracy
    metrics_dict["f1"] = f1

    cm = confusion_matrix(labels, predictions, labels=range(num_classes))
    total_samples = np.sum(cm)

    for i in range(num_classes):
        # True Positives
        TP = cm[i, i]
        # False Negatives
        FN = np.sum(cm[i, :]) - TP
        # False Positives
        FP = np.sum(cm[:, i]) - TP
        # True Negatives
        TN = total_samples - (TP + FP + FN)

        # TPR (Sensitivity, Recall) = TP / (TP + FN)
        tpr = TP / (TP + FN) if (TP + FN) > 0 else 0.0
        # FPR = FP / (FP + TN)
        fpr = FP / (FP + TN) if (FP + TN) > 0 else 0.0
        # TNR = TN / (TN + FP)
        tnr = TN / (TN + FP) if (TN + FP) > 0 else 0.0
        # FNR = FN / (FN + TP)
        fnr = FN / (FN + TP) if (FN + TP) > 0 else 0.0
        metrics_dict[f"class_{i}"] = {"tpr": tpr, "fpr": fpr, "tnr": tnr, "fnr": fnr}
    return metrics_dict


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    # Ensuring deterministic behavior
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def split_dataset(dataset, split_ratio):
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    split = int(np.floor(split_ratio * dataset_size))
    np.random.shuffle(indices)

    val_indices, test_indices = indices[:split], indices[split:]
    dataset = Subset(dataset, test_indices)
    return dataset

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

from nltk.stem import PorterStemmer
import re

stemmer = PorterStemmer()

def normalize_words(text):
    words = re.findall(r'\b\w+\b', text.lower())
    normalized = [stemmer.stem(w) for w in words]
    return set(normalized)

def filter_text(text, should_contain, should_not_contain):
    text_stems = normalize_words(text)
    should_contain_stem = set(stemmer.stem(w.lower()) for w in should_contain)
    should_not_contain_stem = set(stemmer.stem(w.lower()) for w in should_not_contain)

    if should_contain_stem and not any(word in text_stems for word in should_contain_stem):
        return False
    if any(word in text_stems for word in should_not_contain_stem):
        return False
    return True 


def preprocess_text_for_compound_matching(text, compound_words):
    """
    将 text 中形如 'word1 _ word2'（空格+下划线+空格）形式的词组合并成 'word1_word2'
    """
    for compound in compound_words:
        parts = compound.split("_")
        if len(parts) >= 2:
            # 只匹配空格 _ 空格结构，如 'male _ host'
            pattern = r'\b{}\b'.format(r'\s+_\s+'.join(map(re.escape, parts)))
            text = re.sub(pattern, compound, text, flags=re.IGNORECASE)
    return text


def perturb_example(text, perturbation_list):
    compound_words = [orig for orig, _ in perturbation_list if "_" in orig]
    text = preprocess_text_for_compound_matching(text, compound_words)

    perturbation_list = sorted(perturbation_list, key=lambda x: -len(x[0]))
    for orig, perturb in perturbation_list:
        pattern = r'(?<!\w){}(?!\w)'.format(re.escape(orig))
        text = re.sub(pattern, perturb, text)
    return text


def apply_dataset_perturbation(dataset, orig_group, bias_type=None):
    ## return a list of perturbed datasets
    if bias_type is None:
        # bias type is the key whose value contains the group
        bias_type = [key for key, value in SOCIAL_GROUPS.items() if orig_group in value][0]
    perturbed_groups = PERTURBATION_LIST[bias_type][orig_group].keys()
    perturbed_datasets = {}
    for group in perturbed_groups:
        perturbation_list = PERTURBATION_LIST[bias_type][orig_group][group]
        perturbed_dataset = dataset.to_dict().copy()
        perturbed_texts = [perturb_example(text, perturbation_list) for text in perturbed_dataset['text']]
        perturbed_dataset['text'] = perturbed_texts
        perturbed_datasets[group] = Dataset.from_dict(perturbed_dataset)
    return perturbed_datasets
    


from datasets import load_dataset, concatenate_datasets, DatasetDict, Dataset, load_from_disk
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, Subset
import numpy as np
from utils.vocabulary import *
from utils.utils import perturb_example, apply_dataset_perturbation, filter_text, JIGSAW_DATASET_DIR_PROCESSED
from tqdm import tqdm

def merge_dataset_list(dataset_list):
    """
    Merges a list of datasets (with splits) into a single dataset.
    """
    merged_dataset = DatasetDict({split: concatenate_datasets([ds[split] for ds in dataset_list]) for split in dataset_list[0].keys()})
    return merged_dataset

def customized_load_dataset(dataset_name, group="all"):
    if "civil_comments" in dataset_name:
        if group == "all":
            dataset_list = []
            all_groups = [target_group for target_groups in SOCIAL_GROUPS.values() for target_group in target_groups]
            for target_group in all_groups:
                dataset_list.append(load_dataset(dataset_name, target_group, trust_remote_code=True))
            dataset = merge_dataset_list(dataset_list)
        elif group == "jewish":
            dataset = load_dataset(dataset_name, "other_religions", trust_remote_code=True)
            religion_groups = SOCIAL_GROUPS["religion"]
            for split in dataset.keys():
                dataset[split] = Dataset.from_list([example for example in dataset[split] if filter_text(example['text'].lower(), SHOULD_CONTAIN["religion"]["jewish"], SHOULD_NOT_CONTAIN["religion"]["jewish"]) and not any(filter_text(example['text'].lower(), SHOULD_CONTAIN["religion"][other_group], SHOULD_NOT_CONTAIN["religion"][other_group]) for other_group in religion_groups if other_group != "jewish")])
        else:
            dataset = load_dataset(dataset_name, group, trust_remote_code=True)
    elif "jigsaw" in dataset_name:
        if group == "all":
            dataset_list = []
            all_groups = [target_group for target_groups in SOCIAL_GROUPS.values() for target_group in target_groups]
            for target_group in all_groups:
                dataset_list.append(load_from_disk(f"{JIGSAW_DATASET_DIR_PROCESSED}/{target_group}"))
            dataset = merge_dataset_list(dataset_list)
        else:
            dataset = load_from_disk(f"{JIGSAW_DATASET_DIR_PROCESSED}/{group}")
    else:
        raise ValueError("Dataset not supported")
    if "civil_comments" in dataset_name:
        dataset = dataset.map(lambda example: {"label": 1 if example['sub_split'] == 'toxic' else 0}, 
                      keep_in_memory=True)
        dataset = dataset.remove_columns(['sub_split', 'gold'])

    # lowercase the text
    dataset = dataset.map(lambda example: {"text": example['text'].lower()})
    return dataset
    
def customized_split_dataset(dataset, split_ratio):
    if split_ratio is not None:
        split_ratio = [float(r) for r in split_ratio.strip().split(",")]
    
    train_dataset = dataset['train']
    test_dataset = dataset['test']
    if 'val' in dataset:
        val_dataset = dataset['val']
    elif "validation" in dataset:
        val_dataset = dataset['validation']
    elif "dev" in dataset:
        val_dataset = dataset['dev']
    else:
        # Split the train dataset into train and validation sets
        train_dataset_size = len(train_dataset)
        indices = list(range(train_dataset_size))
        if len(split_ratio) == 1:
            split = int(np.floor(split_ratio[0] * train_dataset_size))
        elif len(split_ratio) == 2:
            ratio = split_ratio[0] / (split_ratio[0] + split_ratio[1])
            split = int(np.floor(ratio * train_dataset_size))
        else:
            raise ValueError("Invalid split ratio")
        np.random.shuffle(indices)
        if split < len(indices):
            train_indices, val_indices = indices[:split], indices[split:]
        else:
            train_indices, val_indices = indices, []
        if len(val_indices) == 0:
            val_dataset = Dataset.from_list([])
        else:
            val_dataset = train_dataset.select(val_indices)
        train_dataset = train_dataset.select(train_indices)
 
    return train_dataset, val_dataset, test_dataset

def customized_split_dataset_dict(dataset_dict, split_ratio):
    # add the source dataset name to each example
    for group in dataset_dict.keys():
        for split in dataset_dict[group].keys():
            dataset_dict[group][split] = dataset_dict[group][split].map(lambda example: {"source": group}, keep_in_memory=True)

    
    # merge the datasets
    merged_dataset = merge_dataset_list([dataset_dict[group] for group in dataset_dict.keys()])
    # split the merged dataset
    train_dataset, val_dataset, test_dataset = customized_split_dataset(merged_dataset, split_ratio)
    # split the datasets back to the original groups
    train_dataset_dict = {group: train_dataset.filter(lambda example: example['source'] == group) for group in dataset_dict.keys()}
    val_dataset_dict = {group: val_dataset.filter(lambda example: example['source'] == group) for group in dataset_dict.keys()}
    test_dataset_dict = {group: test_dataset.filter(lambda example: example['source'] == group) for group in dataset_dict.keys()}

    # remove the source column
    train_dataset_dict = {group: dataset.remove_columns(['source']) for group, dataset in train_dataset_dict.items()}
    val_dataset_dict = {group: dataset.remove_columns(['source']) for group, dataset in val_dataset_dict.items()}
    test_dataset_dict = {group: dataset.remove_columns(['source']) for group, dataset in test_dataset_dict.items()}
    return train_dataset_dict, val_dataset_dict, test_dataset_dict


def load_subsets_for_fairness_explainability(dataset_name, split="test", split_ratio=None, bias_type="race", num_samples=-1, shuffle=False, seed=42):
    groups = SOCIAL_GROUPS[bias_type]
    target_datasets = {}
    if "civil_comments" in dataset_name:
        for group in groups:
            if split == "test":
                complete_dataset = customized_load_dataset(dataset_name, group)[split]
                target_datasets[group] = Dataset.from_list([example for example in complete_dataset if filter_text(example['text'], SHOULD_CONTAIN[bias_type][group], SHOULD_NOT_CONTAIN[bias_type][group]) and not any(filter_text(example['text'], SHOULD_CONTAIN[bias_type][other_group], SHOULD_NOT_CONTAIN[bias_type][other_group]) for other_group in groups if other_group != group)])
            else:
                complete_dataset = customized_load_dataset(dataset_name, group)
                train_dataset, val_dataset, test_dataset = customized_split_dataset(complete_dataset, split_ratio)
                if split == "train":
                    target_datasets[group] = Dataset.from_list([example for example in train_dataset if filter_text(example['text'], SHOULD_CONTAIN[bias_type][group], SHOULD_NOT_CONTAIN[bias_type][group]) and not any(filter_text(example['text'], SHOULD_CONTAIN[bias_type][other_group], SHOULD_NOT_CONTAIN[bias_type][other_group]) for other_group in groups if other_group != group)])
                elif split == "val":
                    target_datasets[group] = Dataset.from_list([example for example in val_dataset if filter_text(example['text'], SHOULD_CONTAIN[bias_type][group], SHOULD_NOT_CONTAIN[bias_type][group]) and not any(filter_text(example['text'], SHOULD_CONTAIN[bias_type][other_group], SHOULD_NOT_CONTAIN[bias_type][other_group]) for other_group in groups if other_group != group)])
    elif "jigsaw" in dataset_name:
        for group in groups:
            complete_dataset = customized_load_dataset(dataset_name, group)
            train_dataset, val_dataset, test_dataset = customized_split_dataset(complete_dataset, split_ratio=None)
            if split == "train":
                target_datasets[group] = Dataset.from_list([example for example in train_dataset if filter_text(example['text'], SHOULD_CONTAIN[bias_type][group], SHOULD_NOT_CONTAIN[bias_type][group]) and not any(filter_text(example['text'], SHOULD_CONTAIN[bias_type][other_group], SHOULD_NOT_CONTAIN[bias_type][other_group]) for other_group in groups if other_group != group)])
            elif split == "val":
                target_datasets[group] = Dataset.from_list([example for example in val_dataset if filter_text(example['text'], SHOULD_CONTAIN[bias_type][group], SHOULD_NOT_CONTAIN[bias_type][group]) and not any(filter_text(example['text'], SHOULD_CONTAIN[bias_type][other_group], SHOULD_NOT_CONTAIN[bias_type][other_group]) for other_group in groups if other_group != group)])
            elif split == "test":
                target_datasets[group] = Dataset.from_list([example for example in test_dataset if filter_text(example['text'], SHOULD_CONTAIN[bias_type][group], SHOULD_NOT_CONTAIN[bias_type][group]) and not any(filter_text(example['text'], SHOULD_CONTAIN[bias_type][other_group], SHOULD_NOT_CONTAIN[bias_type][other_group]) for other_group in groups if other_group != group)])
    else:
        raise ValueError("Dataset not supported")

    if num_samples > 0:
        for group in groups:
            if len(target_datasets[group]) < num_samples:
                raise ValueError(f"Number of samples for group {group} exceeds dataset size.")
            if shuffle:
                target_datasets[group] = target_datasets[group].shuffle(seed=seed)
            target_datasets[group] = target_datasets[group].select(range(num_samples))
        
    return target_datasets


#####################################################################################################################

# given a dataset, perform: random sampling; balanced resampling; counterfactual augmentation

def filter_cda_example_dataset(group_dataset, target_group, bias_type):
    groups = SOCIAL_GROUPS[bias_type]
    cda_example_dataset = []
    for example in tqdm(group_dataset):
        should_contain = SHOULD_CONTAIN[bias_type][target_group]
        should_not_contain = SHOULD_NOT_CONTAIN[bias_type][target_group]
        if filter_text(example['text'], should_contain, should_not_contain) and not any(filter_text(example['text'], SHOULD_CONTAIN[bias_type][other_group], SHOULD_NOT_CONTAIN[bias_type][other_group]) for other_group in groups if other_group != target_group):
            cda_example_dataset.append(example)
    cda_example_dataset = Dataset.from_list(cda_example_dataset)
    return cda_example_dataset

def random_sampling(dataset, total_num_samples, seed=42):
    if total_num_samples > len(dataset):
        raise ValueError("Number of samples exceeds dataset size.")
    return dataset.shuffle(seed=seed).select(range(total_num_samples))

def random_sampling_dict(dataset_dict, total_num_samples, seed=42):
    groups = list(dataset_dict.keys())
    length_ratio = [len(dataset_dict[group]) for group in groups]
    total_length = sum(length_ratio)
    sampled_num_samples = [int(total_num_samples * (length / total_length)) for length in length_ratio]
    # make sure the total number of samples is equal to total_num_samples
    sampled_num_samples[-1] += total_num_samples - sum(sampled_num_samples)
    for i, group in enumerate(groups):
        if sampled_num_samples[i] > len(dataset_dict[group]):
            raise ValueError(f"Number of samples for group {group} exceeds dataset size. {group} subset size: {len(dataset_dict[group])}, requested samples: {sampled_num_samples[groups.index(group)]}")
    sampled_dataset_dict = {group: random_sampling(dataset_dict[group], sampled_num_samples[i], seed=seed) for i, group in enumerate(groups)}
    return sampled_dataset_dict
    

def group_balanced_resampling(dataset_dict, bias_type, num_relevant_samples, irrelevant_dataset=None, total_num_samples=-1, seed=42):
    groups = SOCIAL_GROUPS[bias_type]
    assert num_relevant_samples <= total_num_samples or total_num_samples<=0, "Total number of samples exceeds the dataset size."
    num_samples_each_group = num_relevant_samples // len(groups)
    #dataset_dict = split_by_filtering(dataset_dict, bias_type)
    # for each group, sample num_samples from the relevant dataset
    sampled_datasets = {}
    for group in groups:
        if len(dataset_dict[group]) < num_samples_each_group:
            raise ValueError(f"Number of samples for group {group} exceeds relevant dataset size. {group} subset size: {len(dataset_dict[group])}, requested samples: {num_samples_each_group}")
        sampled_datasets[group] = random_sampling(dataset_dict[group], num_samples_each_group, seed=seed)
    group_balanced_dataset = Dataset.from_list([example for group in groups for example in sampled_datasets[group]])
    # sample the remaining samples from the irrelevant dataset
    
    if total_num_samples > 0 and num_relevant_samples * len(groups) < total_num_samples:
        assert irrelevant_dataset is not None, "Irrelevant dataset is required for group balanced resampling."
        remaining_samples = total_num_samples - len(group_balanced_dataset)
        irrelevant_sampled_dataset = random_sampling(irrelevant_dataset, remaining_samples, seed=seed)
        group_balanced_dataset = concatenate_datasets([group_balanced_dataset, irrelevant_sampled_dataset])
    return group_balanced_dataset

def group_and_class_balanced_resampling(dataset_dict, bias_type, num_relevant_samples, irrelevant_dataset=None, total_num_samples=-1, seed=42):
    groups = SOCIAL_GROUPS[bias_type]
    assert num_relevant_samples <= total_num_samples or total_num_samples <= 0, "Total number of samples exceeds the dataset size."
    #dataset_dict = split_by_filtering(dataset_dict, bias_type)
    # for each group, sample num_samples from the relevant dataset
    overall_positive_ratio = sum([sum(dataset_dict[group]['label']) for group in groups]) / sum([len(dataset_dict[group]) for group in groups])
    num_positive_samples_each_group = int(num_relevant_samples * overall_positive_ratio / len(groups))
    num_negative_samples_each_group = num_relevant_samples // len(groups) - num_positive_samples_each_group
    sampled_datasets = {}
    for group in groups:
        positive_samples = Dataset.from_list([example for example in dataset_dict[group] if example['label'] == 1])
        if len(positive_samples) < num_positive_samples_each_group:
            raise ValueError(f"Number of positive samples for group {group} exceeds relevant dataset size. {group} positive subset size: {len(positive_samples)}, requested samples: {num_positive_samples_each_group}")
        negative_samples = Dataset.from_list([example for example in dataset_dict[group] if example['label'] == 0])
        if len(negative_samples) < num_negative_samples_each_group:
            raise ValueError(f"Number of negative samples for group {group} exceeds relevant dataset size. {group} negative subset size: {len(negative_samples)}, requested samples: {num_negative_samples_each_group}")
        # sample num_samples from the positive and negative samples
        sampled_positive = random_sampling(positive_samples, num_positive_samples_each_group, seed=seed)
        sampled_negative = random_sampling(negative_samples, num_negative_samples_each_group, seed=seed)
        sampled_datasets[group] = concatenate_datasets([sampled_positive, sampled_negative])
    group_and_class_balanced_dataset = Dataset.from_list([example for group in groups for example in sampled_datasets[group]])
    # sample the remaining samples from the irrelevant dataset
    
    if total_num_samples > 0 and num_relevant_samples < total_num_samples:
        assert irrelevant_dataset is not None, "Irrelevant dataset is required for group and class balanced resampling."
        remaining_samples = total_num_samples - len(group_and_class_balanced_dataset)
        remaining_samples_each_class = remaining_samples // 2
        positive_samples = Dataset.from_list([example for example in irrelevant_dataset if example['label'] == 1])
        negative_samples = Dataset.from_list([example for example in irrelevant_dataset if example['label'] == 0])
        sampled_positive = random_sampling(positive_samples, remaining_samples_each_class, seed=seed)
        sampled_negative = random_sampling(negative_samples, remaining_samples_each_class, seed=seed)
        group_and_class_balanced_dataset = concatenate_datasets([group_and_class_balanced_dataset, sampled_positive, sampled_negative])
    return group_and_class_balanced_dataset

def counterfactual_augmentation(dataset_dict, bias_type, num_relevant_samples, irrelevant_dataset=None, total_num_samples=-1, seed=42):
    groups = SOCIAL_GROUPS[bias_type]
    assert num_relevant_samples <= total_num_samples or total_num_samples <= 0, "Total number of samples exceeds the dataset size."
    num_cda_samples_per_group = num_relevant_samples // (len(groups) * len(groups))
    dataset_dict = {group: filter_cda_example_dataset(dataset_dict[group], group, bias_type) for group in groups} # filter the dataset that can be cda
    cda_dataset_dict = {}
    for group in groups:
        if len(dataset_dict[group]) < num_cda_samples_per_group:
            raise ValueError(f"Number of samples for group {group} exceeds relevant dataset size. {group} subset size (cda examples): {len(dataset_dict[group])}, requested samples: {num_cda_samples_per_group}")
        sampled_datasets = random_sampling(dataset_dict[group], num_cda_samples_per_group, seed=seed)
        perturbed_dataset_dict = apply_dataset_perturbation(sampled_datasets, orig_group=group, bias_type=bias_type)
        perturbed_dataset = concatenate_datasets([perturbed_dataset_dict[perturbed_group] for perturbed_group in perturbed_dataset_dict.keys()])
        cda_dataset_dict[group] = concatenate_datasets([sampled_datasets, perturbed_dataset])
    
    cda_dataset = concatenate_datasets([cda_dataset_dict[group] for group in groups])
    
    if total_num_samples > 0 and num_relevant_samples < total_num_samples:
        assert irrelevant_dataset is not None, "Irrelevant dataset is required for counterfactual augmentation."
        if len(irrelevant_dataset) < (total_num_samples - len(cda_dataset)):
            raise ValueError("Number of samples in the irrelevant dataset is less than the required number of samples.")
        # sample the remaining samples from the irrelevant dataset
        remaining_samples = total_num_samples - len(cda_dataset)
        irrelevant_sampled_dataset = random_sampling(irrelevant_dataset, remaining_samples, seed=seed)
        cda_dataset = concatenate_datasets([cda_dataset, irrelevant_sampled_dataset])
    return cda_dataset

def counterfactual_augmentation_dict(dataset_dict, bias_type, num_relevant_samples, total_num_samples=-1, seed=42):
    groups = SOCIAL_GROUPS[bias_type]
    assert num_relevant_samples <= total_num_samples or total_num_samples <= 0, "Total number of samples exceeds the dataset size."
    num_cda_samples_per_group = num_relevant_samples // (len(groups) * len(groups))
    dataset_dict = {group: filter_cda_example_dataset(dataset_dict[group], group, bias_type) for group in groups} # filter the dataset that can be cda
    cda_dataset_dict = {}
    for group in groups:
        if len(dataset_dict[group]) < num_cda_samples_per_group:
            raise ValueError(f"Number of samples for group {group} exceeds relevant dataset size. {group} subset size (cda examples): {len(dataset_dict[group])}, requested samples: {num_cda_samples_per_group}")
        sampled_datasets = random_sampling(dataset_dict[group], num_cda_samples_per_group, seed=seed)
        perturbed_dataset_dict = apply_dataset_perturbation(sampled_datasets, orig_group=group, bias_type=bias_type)
        perturbed_dataset = concatenate_datasets([perturbed_dataset_dict[perturbed_group] for perturbed_group in perturbed_dataset_dict.keys()])
        cda_dataset_dict[group] = concatenate_datasets([sampled_datasets, perturbed_dataset])
    return cda_dataset_dict


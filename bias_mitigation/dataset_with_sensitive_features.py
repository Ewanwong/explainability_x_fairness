from datasets import load_dataset, concatenate_datasets, DatasetDict, Dataset
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, Subset
import numpy as np
from utils.vocabulary import *
from utils.utils import perturb_example, apply_dataset_perturbation, filter_text
from tqdm import tqdm
from utils.utils import set_random_seed
from utils.dataset_utils import customized_load_dataset, customized_split_dataset_dict, random_sampling
from utils.reliance_utils import get_sensitive_token_mask
from transformers import AutoTokenizer
from typing import Union

class DatasetWithSensitiveFeatures(Dataset):
    def __init__(self, dataset: Dataset, model_name_or_path: str = "bert-base-uncased", split_ratio: str = "8, 2", bias_types: list = ["race"], num_examples_list: list = [800], seed=42, max_seq_length=512, tokenizer=None):

        if model_name_or_path is None and tokenizer is None:
            raise ValueError("Either model_name_or_path or tokenizer must be provided.")

        self.tokenizer = tokenizer if tokenizer is not None else AutoTokenizer.from_pretrained(model_name_or_path)
        assert len(bias_types) == len(num_examples_list), "bias_types and num_examples_list must have the same length."

        # load and split the dataset
        self.target_groups = []
        for bias_type in bias_types:
            self.target_groups += SOCIAL_GROUPS[bias_type]

        self.sensitive_tokens = SENSITIVE_TOKENS

        train_datasets = []
        val_datasets = []
        test_datasets = []
        for i, bias_type in enumerate(bias_types):
            dataset_dict = {}
            target_groups = SOCIAL_GROUPS[bias_type]
            num_examples = num_examples_list[i]
            for group in target_groups:
                dataset_dict[group] = customized_load_dataset(dataset, group)
            train_dataset_dict, val_dataset_dict, test_dataset_dict = customized_split_dataset_dict(dataset_dict, split_ratio)
            for group in target_groups:
                train_dataset_dict[group] = train_dataset_dict[group].add_column('group', [group] * len(train_dataset_dict[group]))
                val_dataset_dict[group] = val_dataset_dict[group].add_column('group', [group] * len(val_dataset_dict[group]))
                test_dataset_dict[group] = test_dataset_dict[group].add_column('group', [group] * len(test_dataset_dict[group]))
            val_dataset = concatenate_datasets([val_dataset_dict[group] for group in target_groups])
            test_dataset = concatenate_datasets([test_dataset_dict[group] for group in target_groups])
            train_dataset = random_sampling(concatenate_datasets([train_dataset_dict[group] for group in target_groups]), num_examples, seed=seed)

            train_datasets.append(train_dataset)
            val_datasets.append(val_dataset)
            test_datasets.append(test_dataset)

        self.train_dataset = concatenate_datasets(train_datasets)
        self.val_dataset = concatenate_datasets(val_datasets)
        self.test_dataset = concatenate_datasets(test_datasets)
        self.train_dataset = self.train_dataset.shuffle(seed=seed)
        self.val_dataset = self.val_dataset.shuffle(seed=seed)
        self.test_dataset = self.test_dataset.shuffle(seed=seed)

        # apply tokenization
        def tokenize_function(examples):
            return self.tokenizer(examples['text'],
                            padding='max_length',
                            truncation=True,
                            max_length=max_seq_length)
        
        self.train_dataset = self.train_dataset.map(tokenize_function, batched=True)
        self.val_dataset = self.val_dataset.map(tokenize_function, batched=True)
        self.test_dataset = self.test_dataset.map(tokenize_function, batched=True)

        # add sensitive_token_mask
        self.train_dataset = get_sensitive_token_mask(self.train_dataset, self.target_groups, self.tokenizer, self.sensitive_tokens)
        self.val_dataset = get_sensitive_token_mask(self.val_dataset, self.target_groups, self.tokenizer, self.sensitive_tokens)
        # self.test_dataset = get_sensitive_token_mask(self.test_dataset, self.target_groups, self.tokenizer, self.sensitive_tokens)
        self.train_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label', 'sensitive_token_mask', 'group'])
        self.train_dataset = self.train_dataset.rename_column('label', 'labels')
        self.val_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label', 'sensitive_token_mask', 'group'])
        self.val_dataset = self.val_dataset.rename_column('label', 'labels')
        # self.test_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label', 'sensitive_token_mask', 'group'])
        self.test_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label', 'group'])
        self.test_dataset = self.test_dataset.rename_column('label', 'labels')

    def get_train_dataset(self):
        return self.train_dataset
    
    def get_val_dataset(self):
        return self.val_dataset
    
    def get_test_dataset(self):
        return self.test_dataset
    
    def get_tokenizer(self):
        return self.tokenizer
    
    def get_train_dataloader(self, batch_size=32):
        return DataLoader(self.train_dataset, sampler=RandomSampler(self.train_dataset), batch_size=batch_size)

    def get_val_dataloader(self, batch_size=32):
        return DataLoader(self.val_dataset, sampler=SequentialSampler(self.val_dataset), batch_size=batch_size)

    def get_test_dataloader(self, batch_size=32):
        return DataLoader(self.test_dataset, sampler=SequentialSampler(self.test_dataset), batch_size=batch_size)
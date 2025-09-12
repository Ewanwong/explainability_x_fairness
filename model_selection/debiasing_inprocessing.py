import argparse
import json
import logging
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, Subset
from transformers import (AutoTokenizer, AutoConfig, AutoModelForSequenceClassification,
                          get_linear_schedule_with_warmup)
from torch.optim import AdamW
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
import random
import os
from tqdm import tqdm
from utils.utils import set_random_seed, perturb_example
from utils.dataset_utils import customized_load_dataset, random_sampling, random_sampling_dict, filter_cda_example_dataset, customized_split_dataset_dict
from utils.vocabulary import *
from datasets import concatenate_datasets
# from scipy.stats import wasserstein_distance
import shutil

def main():
    # Argument parser for hyperparameters
    parser = argparse.ArgumentParser(description="Fine-tune BERT for sequence classification")

    # Hyperparameters
    parser.add_argument('--model_name_or_path', type=str, default='bert-base-uncased',
                        help='Pre-trained model name or path')
    parser.add_argument('--dataset_name', type=str, default='fancyzhx/ag_news',
                        help='Dataset name (default: ag_news)')
    parser.add_argument('--num_labels', type=int, default=2,
                        help='Number of labels in the dataset')
    parser.add_argument('--output_dir', type=str, default='/local/yifwang/bcos_bert_base_agnews_512',
                        help='Directory to save the model')
    parser.add_argument('--max_seq_length', type=int, default=512,
                        help='Maximum input sequence length after tokenization')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size for training and evaluation')
    parser.add_argument('--learning_rate', type=float, default=3e-5,
                        help='Learning rate for the optimizer')
    parser.add_argument('--warmup_steps_or_ratio', type=float, default=0.1,
                        help='Number or ratio of warmup steps for the learning rate scheduler')
    parser.add_argument('--num_train_epochs', type=int, default=10,
                        help='Total number of training epochs')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for initialization')
    parser.add_argument('--early_stopping_patience', type=int, default=-1,
                        help='Number of epochs with no improvement after which training will be stopped')
    parser.add_argument('--log_file', type=str, default='training.log',
                        help='Path to the log file')
    parser.add_argument('--eval_steps', type=int, default=1000,
                        help='Evaluate the model every X training steps')
    parser.add_argument('--save_steps', type=int, default=1000,
                        help='Save the model every X training steps')
    parser.add_argument('--split_ratio', type=str, default="0.8, 0.2",
                    help='Ratio to split the train set into train and validation sets')
    parser.add_argument("--eval_metric", type=str, default="accuracy",)
    parser.add_argument('--debiasing_method', type=str, default="dropout", choices=["dropout", "causal_debias", "attention_entropy"],
                        help='Debiasing method to use (default: dropout)')
    parser.add_argument('--entropy_weight', type=float, default=0.01,
                        help='Weight for the entropy loss in attention entropy debiasing method')
    parser.add_argument('--hidden_dropout_prob', type=float, default=0.2,
                        help='Dropout probability for the hidden layers in the model')
    parser.add_argument('--attention_dropout_prob', type=float, default=0.15,
                        help='Dropout probability for the attention probabilities in the model')
    parser.add_argument('--causal_debias_weight', type=float, default=0.5,)
    parser.add_argument('--total_num_examples', type=str, default=-1, help='Total number of examples to use for training. If -1, use all available examples.')
    parser.add_argument('--bias_types', type=str, default="race", help='comma separated list of bias types to debias against')
    args = parser.parse_args()

    # create output directory if it doesn't exist
    save_dir = os.path.join(args.output_dir, f"{args.debiasing_method}")
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    log_file = os.path.join(save_dir, args.log_file)

    # Set up logging
    logging.basicConfig(
        filename=log_file,
        filemode='w',
        format='%(asctime)s - %(levelname)s - %(message)s',
        level=logging.INFO
    )
    # Also log to console
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)

    # Log the hyperparameters
    logging.info("Hyperparameters:")
    for arg in vars(args):
        logging.info(f"{arg}: {getattr(args, arg)}")

    # Set up the device for GPU usage if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    # Set seeds for reproducibility
    seed_val = args.seed
    set_random_seed(seed_val)

    # collect the list of bias types and numbers of training examples
    bias_types = args.bias_types.split(",")
    bias_types = [bias_type.strip() for bias_type in bias_types]  # remove any leading/trailing whitespace
    # Set up debiasing method
    
    total_num_examples_list = args.total_num_examples.split(",")
    total_num_examples_list = [int(num.strip()) for num in total_num_examples_list]
    assert any(num > 0 for num in total_num_examples_list), "Total number of examples must be greater than 0."
    assert len(total_num_examples_list) == len(bias_types), "Number of total examples must match the number of bias types."
    

    # Load the dataset
    logging.info(f"Loading {args.dataset_name} dataset...")
    train_datasets, val_datasets, test_datasets = [], [], []
    for i, bias_type in enumerate(bias_types):
        target_groups = SOCIAL_GROUPS[bias_type]
        # load the dataset by groups
        dataset_dict = {}
        for group in target_groups:
            dataset_dict[group] = customized_load_dataset(args.dataset_name, group)
        
        # split the dataset, each group in dataset_dict is a Dataset object, group information will be used for pre-processing
        train_dataset_dict, val_dataset_dict, test_dataset_dict = customized_split_dataset_dict(dataset_dict, args.split_ratio)
        # for val and test datasets, merge the datasets from all groups
        val_dataset = concatenate_datasets([val_dataset_dict[group] for group in target_groups])
        test_dataset = concatenate_datasets([test_dataset_dict[group] for group in target_groups])

        # train dataset is pre-processed based on the debiasing method
        total_num_examples = total_num_examples_list[i]
        if args.debiasing_method == "dropout" or args.debiasing_method == "attention_entropy":
            train_dataset = random_sampling(concatenate_datasets([train_dataset_dict[group] for group in target_groups]), total_num_examples, seed=seed_val)
        elif args.debiasing_method == "causal_debias":
            train_dataset_dict = {group: filter_cda_example_dataset(train_dataset_dict[group], target_group=group, bias_type=bias_type) for group in target_groups}
            train_dataset_dict = random_sampling_dict(train_dataset_dict, total_num_examples, seed=seed_val)
            # merge all datasets from all groups into a single dataset, with an additional 'group' column indicating the group of the example
            for group in target_groups:
                train_dataset_dict[group] = train_dataset_dict[group].add_column('group', [group] * len(train_dataset_dict[group]))
                train_dataset_dict[group] = train_dataset_dict[group].add_column('bias_type', [bias_type] * len(train_dataset_dict[group]))
            train_dataset = concatenate_datasets([train_dataset_dict[group] for group in target_groups])
            #train_dataset = random_sampling(train_dataset, total_num_examples, seed=seed_val)
        else:
            raise ValueError(f"Unknown debiasing method: {args.debiasing_method}")
        

                    
        train_datasets.append(train_dataset)
        val_datasets.append(val_dataset)
        test_datasets.append(test_dataset)
    
    # Merge the datasets from all bias types

    train_dataset = concatenate_datasets(train_datasets)

    val_dataset = concatenate_datasets(val_datasets)
    test_dataset = concatenate_datasets(test_datasets)
    len_train_dataset = len(train_dataset)
    logging.info(f"Train dataset size: {len_train_dataset}")
    # Initialize the tokenizer and model

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_name_or_path, num_labels=args.num_labels, output_attentions=True)
    if args.debiasing_method == "dropout":
        # Set the dropout rate for the model
        if hasattr(model.config, 'hidden_dropout_prob'):
            model.config.hidden_dropout_prob = args.hidden_dropout_prob
        else:
            model.config.dropout = args.hidden_dropout_prob
        if hasattr(model.config, 'attention_probs_dropout_prob'):
            model.config.attention_probs_dropout_prob = args.attention_dropout_prob
        else:
            model.config.attention_dropout = args.attention_dropout_prob
       
    logging.info(f"Model {args.model_name_or_path} loaded with {args.num_labels} labels.")
    model.to(device)

    # Tokenization function
    def tokenize_function(examples):

        return tokenizer(examples['text'],
                        padding='max_length',
                        truncation=True,
                        max_length=args.max_seq_length)

    # Apply tokenization to the datasets
    logging.info("Tokenizing datasets...")
    
    train_dataset = train_dataset.map(tokenize_function, batched=True)

    if args.debiasing_method != "causal_debias":
        train_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
    else:
        train_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label', 'text', 'group', 'bias_type'])
    train_dataset = train_dataset.rename_column('label', 'labels')
            
    val_dataset = val_dataset.map(tokenize_function, batched=True)
    test_dataset = test_dataset.map(tokenize_function, batched=True)

    
    val_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
    val_dataset = val_dataset.rename_column('label', 'labels')
    test_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
    test_dataset = test_dataset.rename_column('label', 'labels')

    train_dataloader = DataLoader(train_dataset, sampler=RandomSampler(train_dataset), batch_size=args.batch_size)

    validation_dataloader = DataLoader(val_dataset, sampler=SequentialSampler(val_dataset), batch_size=args.batch_size)
    test_dataloader = DataLoader(test_dataset, sampler=SequentialSampler(test_dataset), batch_size=args.batch_size)

    # Initialize the optimizer and learning rate scheduler
    optimizer = AdamW(model.parameters(), lr=args.learning_rate, eps=1e-8)
    total_steps = len(train_dataloader) * args.num_train_epochs
    if args.warmup_steps_or_ratio > 1.0:
        warmup_steps = args.warmup_steps_or_ratio
    else:
        warmup_steps = int(total_steps * args.warmup_steps_or_ratio)

    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps=warmup_steps,
                                                num_training_steps=total_steps)

    # Accuracy evaluation function
    def evaluate(model, dataloader, average='macro'):
        model.eval()
        predictions, true_labels = [], []

        for batch in dataloader:
            batch = {k: v.to(device) for k, v in batch.items() if k in ['input_ids', 'attention_mask', 'labels']}

            with torch.no_grad():
                outputs = model(input_ids=batch['input_ids'],
                                attention_mask=batch['attention_mask'],
                                labels=batch['labels'])

            logits = outputs.logits.detach().cpu().numpy()
            label_ids = batch['labels'].to('cpu').numpy()

            predictions.extend(np.argmax(logits, axis=1))
            true_labels.extend(label_ids)

        accuracy = accuracy_score(true_labels, predictions)
        f1 = f1_score(true_labels, predictions, average=average)
        model.train()
        return accuracy, f1

    # Early stopping parameters
    early_stopping_patience = args.early_stopping_patience if args.early_stopping_patience != -1 else np.inf
    best_metric = 0.0
    evaluations_no_improve = 0
    global_step = 0

    # Training loop
    for epoch_i in range(args.num_train_epochs):
        logging.info(f"\n======== Epoch {epoch_i + 1} / {args.num_train_epochs} ========")
        logging.info("Training...")
        total_loss = 0
        model.train()

        for step, batch in tqdm(enumerate(train_dataloader)):
            
            global_step += 1
            # move input_ids, attention_mask, and labels to the device, other fields stay on cpu
            input_batch = {k: v.to(device) for k, v in batch.items() if k in ['input_ids', 'attention_mask', 'labels']}

            optimizer.zero_grad()

            outputs = model(input_ids=input_batch['input_ids'],
                            attention_mask=input_batch['attention_mask'],
                            labels=input_batch['labels'])
            loss = outputs.loss
            total_loss += loss.item()

            if args.debiasing_method == "attention_entropy":
                # compute attention distribution for each layer average over all heads
                attentions = outputs.attentions
                attention_entropy = 0.0
                for layer_attention in attentions:
                    # average over all heads
                    layer_attention = layer_attention.mean(dim=1)
                    # compute entropy
                    layer_attention = layer_attention.softmax(dim=-1)
                    layer_entropy = -torch.sum(layer_attention * torch.log(layer_attention + 1e-10), dim=-1)
                    attention_entropy += layer_entropy.mean()
                attention_entropy /= len(attentions)
                # add attention entropy to the loss
                loss += attention_entropy * args.entropy_weight
                total_loss += attention_entropy.item() * args.entropy_weight
            elif args.debiasing_method == "causal_debias":
                
                # for each example in the batch, apply cda
                wasserstein_dists = []
                for i, orig_text in enumerate(batch['text']):
                    orig_group = batch['group'][i]
                    bias_type = batch['bias_type'][i]
                    
                    orig_logits = outputs.logits[i, :].unsqueeze(0)  # shape (1, num_labels)
                    perturbed_groups = PERTURBATION_LIST[bias_type][orig_group].keys()
                    # apply cda to the original text
                    cda_texts = []
                    for perturbed_group in perturbed_groups:
                        perturbed_text = perturb_example(orig_text, PERTURBATION_LIST[bias_type][orig_group][perturbed_group])
                        cda_texts.append(perturbed_text)
                    # compute the loss for all perturbed texts
                    cda_inputs = tokenizer(cda_texts, padding='max_length', truncation=True, max_length=args.max_seq_length, return_tensors='pt')
                    cda_inputs = {'input_ids': cda_inputs['input_ids'].to(device),
                                  'attention_mask': cda_inputs['attention_mask'].to(device),
                                  'labels': torch.tensor([batch['labels'][i]] * len(cda_texts), device=device)}

                    cda_outputs = model(**cda_inputs)
                    cda_logits = cda_outputs.logits
                    all_logits = torch.cat([orig_logits, cda_logits], dim=0).squeeze() # shape (num_perturbations + 1, num_labels)
                    mean_logits = all_logits.mean(dim=0)
                    # compute sum wasserstein distance between the each logits and the mean logits
                    wasserstein_dist = torch.sum(torch.abs(all_logits - mean_logits))
                    wasserstein_dists.append(wasserstein_dist)
                # compute the average wasserstein distance
                # print(f"Wasserstein distances: {wasserstein_dists}")
                wasserstein_dist_mean = torch.mean(torch.tensor(wasserstein_dists, device=device))
                wasserstein_dist_var = torch.var(torch.tensor(wasserstein_dists, device=device))
                # add the wasserstein distance to the loss
                loss += (wasserstein_dist_mean + wasserstein_dist_var) * args.causal_debias_weight
                total_loss += (wasserstein_dist_mean + wasserstein_dist_var).item() * args.causal_debias_weight
            
            loss.backward()

            #torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            # Evaluate the model at specified steps
            if args.eval_steps and global_step % args.eval_steps == 0:
                logging.info(f"\nStep {global_step}: running evaluation...")
                val_accuracy, val_f1 = evaluate(model, validation_dataloader)
                logging.info(f"Validation Accuracy at step {global_step}: {val_accuracy:.4f}")
                logging.info(f"Validation F1 Score at step {global_step}: {val_f1:.4f}")

                # Check for early stopping
                if (args.eval_metric == "accuracy" and val_accuracy > best_metric) or (args.eval_metric == "f1" and val_f1 > best_metric):
                    best_metric = val_accuracy if args.eval_metric == "accuracy" else val_f1
                    evaluations_no_improve = 0

                    # Save the best model using Hugging Face's save_pretrained
                    model.save_pretrained(save_dir)
                    tokenizer.save_pretrained(save_dir)
                    logging.info(f"Best model saved to {save_dir}")
                else:
                    evaluations_no_improve += 1
                    logging.info(f"No improvement in validation accuracy for {evaluations_no_improve} evaluation(s).")
                    if evaluations_no_improve >= early_stopping_patience:
                        logging.info("Early stopping triggered.")
                        break

            # Save the model at specified steps
            if args.save_steps and global_step % args.save_steps == 0:
                checkpoint_dir = os.path.join(save_dir, f'checkpoint-{global_step}')
                if not os.path.exists(checkpoint_dir):
                    os.makedirs(checkpoint_dir)
                model.save_pretrained(checkpoint_dir)
                tokenizer.save_pretrained(checkpoint_dir)
                logging.info(f"Model checkpoint saved at step {global_step} to {checkpoint_dir}")

        avg_train_loss = total_loss / len(train_dataloader)
        logging.info(f"Average training loss for epoch {epoch_i + 1}: {avg_train_loss:.4f}")

        # Evaluate at the end of each epoch if eval_steps is not set
        if not args.eval_steps:
            logging.info("Running Validation at the end of the epoch...")
            val_accuracy, val_f1 = evaluate(model, validation_dataloader)
            logging.info(f"Validation Accuracy: {val_accuracy:.4f}")
            logging.info(f"Validation F1 Score: {val_f1:.4f}")

            # Check for early stopping
            if (args.eval_metric == "accuracy" and val_accuracy > best_metric) or (args.eval_metric == "f1" and val_f1 > best_metric):
                best_metric = val_accuracy if args.eval_metric == "accuracy" else val_f1
                evaluations_no_improve = 0

                # Save the best model using Hugging Face's save_pretrained
                model.save_pretrained(save_dir)
                tokenizer.save_pretrained(save_dir)
                logging.info(f"Best model saved to {save_dir}")
            else:
                evaluations_no_improve += 1
                logging.info(f"No improvement in validation accuracy for {evaluations_no_improve} epoch(s).")
                if evaluations_no_improve >= early_stopping_patience:
                    logging.info("Early stopping triggered.")
                    break

        # Save the model at the end of each epoch if save_steps is not set
        if not args.save_steps:
            checkpoint_dir = os.path.join(save_dir, f'checkpoint-epoch-{epoch_i + 1}')
            if not os.path.exists(checkpoint_dir):
                os.makedirs(checkpoint_dir)
            model.save_pretrained(checkpoint_dir)
            tokenizer.save_pretrained(checkpoint_dir)
            logging.info(f"Model checkpoint saved at the end of epoch {epoch_i + 1} to {checkpoint_dir}")

        # Break the outer loop if early stopping is triggered during evaluation steps
        if evaluations_no_improve >= early_stopping_patience:
            break

    # Load the best model
    model = AutoModelForSequenceClassification.from_pretrained(save_dir, num_labels=args.num_labels)
    tokenizer = AutoTokenizer.from_pretrained(save_dir)
    model.to(device)

    # Test evaluation
    logging.info("\nRunning Test Evaluation...")
    test_accuracy, test_f1 = evaluate(model, test_dataloader)
    logging.info(f"Test Accuracy: {test_accuracy:.4f}")
    logging.info(f"Test F1 Score: {test_f1:.4f}")

    # only save the best model and delete all checkpoints
    for filename in os.listdir(save_dir):
        file_path = os.path.join(save_dir, filename)
        if os.path.isdir(file_path) and filename.startswith('checkpoint-'):
            logging.info(f"Deleting checkpoint {file_path}")
            shutil.rmtree(file_path)

if __name__ == '__main__':
    main()
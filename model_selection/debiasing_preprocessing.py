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
from utils.utils import set_random_seed
from utils.dataset_utils import merge_dataset_list, customized_load_dataset, random_sampling, group_balanced_resampling, group_and_class_balanced_resampling, counterfactual_augmentation, customized_split_dataset_dict
from utils.vocabulary import *
from datasets import concatenate_datasets
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

    parser.add_argument('--debiasing_method', type=str, default="no_debiasing", choices=["no_debiasing", "group_balance", "group_class_balance", "cda"],
                        help='Debiasing method to use (default: no_debiasing)')
    parser.add_argument('--total_num_examples', type=str, default=-1, help='Total number of examples to use for training. If -1, use all available examples.')
    #parser.add_argument('--debiased_dropout_rate', type=float, default=0.1, help='Dropout rate for debiasing dropout method')
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
        if args.debiasing_method == "group_balance":
            
            train_dataset = group_balanced_resampling(train_dataset_dict, bias_type=bias_type, num_relevant_samples=total_num_examples, seed=seed_val)
        elif args.debiasing_method == "group_class_balance":
            
            train_dataset = group_and_class_balanced_resampling(train_dataset_dict, bias_type=bias_type, num_relevant_samples=total_num_examples, seed=seed_val)
        elif args.debiasing_method == "cda":
            
            train_dataset = counterfactual_augmentation(train_dataset_dict, bias_type=bias_type, num_relevant_samples=total_num_examples, seed=seed_val)
        else:
            train_dataset = random_sampling(concatenate_datasets([train_dataset_dict[group] for group in target_groups]), total_num_examples, seed=seed_val)
        

        train_datasets.append(train_dataset)
        val_datasets.append(val_dataset)
        test_datasets.append(test_dataset)
    
    # Merge the datasets from all bias types
    train_dataset = concatenate_datasets(train_datasets)
    val_dataset = concatenate_datasets(val_datasets)
    test_dataset = concatenate_datasets(test_datasets)
    logging.info(f"Train dataset size: {len(train_dataset)}")
    # Initialize the tokenizer and model

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_name_or_path, num_labels=args.num_labels)
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
    val_dataset = val_dataset.map(tokenize_function, batched=True)
    test_dataset = test_dataset.map(tokenize_function, batched=True)

    train_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
    train_dataset = train_dataset.rename_column('label', 'labels')
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
            batch = {k: v.to(device) for k, v in batch.items()}

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
            batch = {k: v.to(device) for k, v in batch.items()}

            optimizer.zero_grad()

            outputs = model(input_ids=batch['input_ids'],
                            attention_mask=batch['attention_mask'],
                            labels=batch['labels'])
            loss = outputs.loss
            total_loss += loss.item()

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
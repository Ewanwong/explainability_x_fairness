import torch
from captum.attr import Saliency, DeepLift, GuidedBackprop, InputXGradient, IntegratedGradients, Occlusion, ShapleyValueSampling, DeepLiftShap, GradientShap, KernelShap, FeatureAblation 
from tint.attr import SequentialIntegratedGradients
from tqdm import tqdm
from utils.utils import batch_loader


class GPTEmbeddingModelWrapper(torch.nn.Module):
    def __init__(self, model):
        super(GPTEmbeddingModelWrapper, self).__init__()
        self.model = model

    def forward(self, embeddings, attention_mask=None):
        outputs = self.model(inputs_embeds=embeddings, attention_mask=attention_mask)
        logits = outputs.logits
        logits = logits[:, -1, :]  # Get the last token's logits
        return logits
    
class GPTEmbeddingModelProbWrapper(torch.nn.Module):
    def __init__(self, model):
        super(GPTEmbeddingModelProbWrapper, self).__init__()
        self.model = model

    def forward(self, embeddings, attention_mask=None):
        outputs = self.model(inputs_embeds=embeddings, attention_mask=attention_mask)
        logits = outputs.logits
        logits = logits[:, -1, :]  # Get the last token's logits
        probabilities = torch.softmax(logits, dim=-1)
        return probabilities
    
class GPTModelWrapper(torch.nn.Module):
    def __init__(self, model):
        super(GPTModelWrapper, self).__init__()
        self.model = model

    def forward(self, input_ids, attention_mask=None):       
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        logits = logits[:, -1, :]
        return logits
    
class GPTModelProbWrapper(torch.nn.Module):
    def __init__(self, model):
        super(GPTModelProbWrapper, self).__init__()
        self.model = model

    def forward(self, input_ids, attention_mask=None):       
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        logits = logits[:, -1, :]
        probabilities = torch.softmax(logits, dim=-1)
        return probabilities
    
def empty_all_cuda_caches(devices=None):
    """
    Empty the CUDA caching allocator on the given devices.
    If devices is None, clear all visible CUDA devices.
    """
    if not torch.cuda.is_available():
        return
    if devices is None:
        devices = range(torch.cuda.device_count())
    for d in devices:
        with torch.cuda.device(d):
            torch.cuda.empty_cache()

class BaseExplainer:

    def _explain(self):
        raise NotImplementedError
    
    def explain(self):
        raise NotImplementedError
    

    def explain_embeddings(self, prompts, labels, targets, raw_inputs, example_indices):
        assert len(prompts) == 1, "Only one prompt is supported for now"
        inputs = self.tokenizer(prompts, return_tensors='pt', padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        input_ids = inputs['input_ids']
        attention_mask = inputs['attention_mask']
        if "position_ids" in inputs:
            position_ids = inputs['position_ids']
        else:

            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 0)
            position_ids = position_ids.to(self.device)
        # if targets do not start with a white space, add a white space
        if targets is not None:
            target_ids = [self.tokenizer(target, return_tensors='pt', add_special_tokens=False)['input_ids'][:, :1] for target in targets]
            target_ids = torch.cat(target_ids, dim=0)
            target_ids = target_ids.to(self.device)
        else:
            target_ids = None

        if raw_inputs is not None:
            for i in range(len(raw_inputs)):
                if not raw_inputs[i].startswith(" "):
                    raw_inputs[i] = " "+raw_inputs[i]
            
            raw_input_ids = self.tokenizer(raw_inputs, return_tensors='pt', padding=True, add_special_tokens=False)['input_ids']
            raw_input_ids = raw_input_ids.to(self.device)
        else:
            raw_input_ids = None
        
        explanations = self._explain(input_ids=input_ids, attention_mask=attention_mask, position_ids=position_ids, labels=labels, target_ids=target_ids, raw_input_ids=raw_input_ids, example_indices=example_indices)
        return explanations


    def explain_tokens(self, prompts, labels, targets, raw_inputs, example_indices):
        assert len(prompts) == 1, "Only one prompt and one target is supported for now"
        inputs = self.tokenizer(prompts, return_tensors='pt', padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        input_ids = inputs['input_ids']
        attention_mask = inputs['attention_mask']
        # if targets do not start with a white space, add a white space
        if targets is not None:
            target_ids = [self.tokenizer(target, return_tensors='pt', add_special_tokens=False)['input_ids'][:, :1] for target in targets]
            target_ids = torch.cat(target_ids, dim=0)
            target_ids = target_ids.to(self.device)
        else:
            target_ids = None
        if raw_inputs is not None:
            for i in range(len(raw_inputs)):
                if not raw_inputs[i].startswith(" "):
                    raw_inputs[i] = " "+raw_inputs[i]
            
            raw_input_ids = self.tokenizer(raw_inputs, return_tensors='pt', padding=True, add_special_tokens=False)['input_ids']
            raw_input_ids = raw_input_ids.to(self.device)
        else:
            raw_input_ids = None

        explanations = self._explain(input_ids=input_ids, attention_mask=attention_mask, labels=labels, target_ids=target_ids, raw_input_ids=raw_input_ids, example_indices=example_indices)
        return explanations
    
    
    def explain_dataset(self, dataset):
        # if class_labels is not provided, then num_classes must be provided
        data_loader = batch_loader(dataset, batch_size=1, shuffle=False)
        saliency_results = {}
        for batch in tqdm(data_loader):
            prompts = batch['prompt']
            example_indices = batch['index']
            if 'target' in batch:
                targets = batch['target']
            else:
                targets = None
            if 'raw_input' in batch:
                raw_inputs = batch['raw_input']
            else:
                raw_inputs = None
            if 'label' in batch:
                labels = batch['label']
            else:
                labels = None
            explanations = self.explain(prompts=prompts, labels=labels, targets=targets, raw_inputs=raw_inputs, example_indices=example_indices)
            for key, value in explanations.items():
                if key not in saliency_results:
                    saliency_results[key] = []
                saliency_results[key].extend(value)
        return saliency_results
    

class BcosExplainer(BaseExplainer):
    def __init__(self, model, tokenizer):

        self.model = GPTEmbeddingModelWrapper(model)
        self.model.eval()
        #self.model.to(model.model.get_input_embeddings().weight.device)
        self.tokenizer = tokenizer
        self.device = model.model.get_input_embeddings().weight.device

        self.method = "Bcos_absolute"
        self.positive_token = "Yes"
        self.negative_token = "No"
        self.positive_token_id = self.tokenizer(self.positive_token, add_special_tokens=False)["input_ids"][0]
        self.negative_token_id = self.tokenizer(self.negative_token, add_special_tokens=False)["input_ids"][0]
    
    def _explain(self, input_ids, attention_mask, position_ids=None, labels=None, target_ids=None, raw_input_ids=None, example_indices=None):
        """
        if position_ids is None:
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 0)
        """
        batch_size = input_ids.shape[0]
        assert batch_size == 1, "Batch size must be 1 for now"

        if hasattr(self.model.model, "transformer") and hasattr(self.model.model.transformer, "wte"):
            wte = self.model.model.transformer.wte ## gpt model
        elif hasattr(self.model.model, "model") and hasattr(self.model.model.model, "embed_tokens"):
            wte = self.model.model.model.embed_tokens ## llama and qwen model
        else:
            raise ValueError("Model is not supported, cannot extract embeddings")
        embeddings = wte(input_ids) 
        embeddings.requires_grad_()

        # Get the model's predictions
        with torch.no_grad():
            outputs = self.model(embeddings, attention_mask=attention_mask)
        probabilities = torch.softmax(outputs, dim=-1)
        positive_prediction_probabilities = probabilities[:, self.positive_token_id]
        negative_prediction_probabilities = probabilities[:, self.negative_token_id]
        # get the predicted ids
        predicted_ids = torch.where(positive_prediction_probabilities > negative_prediction_probabilities, self.positive_token_id, self.negative_token_id).unsqueeze(1)
        
        if target_ids is None:
            # if target_ids is None, then use the predicted ids as target ids
            target_ids = predicted_ids
            
        # get the probability of the target token
        prediction_probabilities = probabilities[torch.arange(probabilities.shape[0]), predicted_ids].unsqueeze(1)

        all_saliency_ixg_L2_results = [[] for _ in range(batch_size)]
        all_saliency_ixg_mean_results = [[] for _ in range(batch_size)]

        for explained_target_ids in target_ids:
            explained_target_ids = explained_target_ids.unsqueeze(0)
            target_probabilities = probabilities[torch.arange(probabilities.shape[0]), explained_target_ids].unsqueeze(1)
            # activate explanation mode
            with self.model.model.explanation_mode():
                explainer_ixg = InputXGradient(self.model)
                attributions_ixg = explainer_ixg.attribute(
                    inputs=(embeddings),
                    target=explained_target_ids.squeeze(),
                    additional_forward_args=(attention_mask,)
                )

            attributions_ixg_all = attributions_ixg
            for i in range(batch_size):
                true_label = labels[i] if labels is not None else None
                if raw_input_ids is not None:
                    def find_sublist_indexes(full, sub):
                        n, m = len(full), len(sub)
                        for i in range(n - m + 1):
                            if full[i:i + m] == sub:
                                return list(range(i, i + m))
                        return []
                    raw_input_indexes = find_sublist_indexes(input_ids[i].detach().cpu().float().numpy().tolist(), raw_input_ids[i].detach().cpu().float().numpy().tolist())
                    if len(raw_input_indexes) == 0:
                        print(f"Warning: raw_input_ids not found in input_ids for example {example_indices[i]}, return the original input")
                        raw_input_ids = None
                tokens = self.tokenizer.convert_ids_to_tokens(input_ids[i].detach().cpu().float().numpy().tolist())
                target_token = self.tokenizer.convert_ids_to_tokens(explained_target_ids[i].detach().cpu().float().numpy().tolist())[0]                 
                prediction_token = self.tokenizer.convert_ids_to_tokens(predicted_ids[i].detach().cpu().float().numpy().tolist())[0]
                if prediction_token == "Yes":
                    predicted_class = 1
                elif prediction_token == "No":
                    predicted_class = 0
                else:
                    raise ValueError(f"Warning: predicted class {prediction_token} is not Yes or No")

                if target_token == "Yes":
                    target_class = 1
                elif target_token == "No":
                    target_class = 0
                else:
                    print(f"Warning: target class {target_token} is not Yes or No")
                    target_class = target_token
                # Compute saliency metrics for each token
                saliency_ixg_L2 = torch.norm(attributions_ixg_all[i:i+1], dim=-1, p=2).detach().cpu().float().numpy()[0]
                saliency_ixg_mean = attributions_ixg_all[i:i+1].mean(dim=-1).detach().cpu().float().numpy()[0]
                # Collect results for the current example and class
                # skip padding tokens
                # tokens = [token for token in tokens if token != self.tokenizer.pad_token]
                real_length = len(tokens)
                if raw_input_ids is not None:
                    raw_tokens = self.tokenizer.convert_ids_to_tokens(raw_input_ids[i].detach().cpu().float().numpy().tolist())
                    raw_token_ixg_L2 = [saliency_ixg_L2.tolist()[raw_input_index] for raw_input_index in raw_input_indexes]  
                    raw_token_ixg_mean = [saliency_ixg_mean.tolist()[raw_input_index] for raw_input_index in raw_input_indexes] 
                    raw_tokens = [token for token in raw_tokens if token != self.tokenizer.pad_token]
                    result_ixg_L2 = {
                        'index': example_indices[i],
                        'text': self.tokenizer.decode([t for t in raw_input_ids[i] if not (t in self.tokenizer.all_special_ids and t != self.tokenizer.unk_token_id)], skip_special_tokens=False),
                        'true_label': true_label,
                        'predicted_class': predicted_class,
                        'predicted_class_confidence': prediction_probabilities[i].item(),
                        'target_class': target_class,
                        'target_class_confidence': target_probabilities[i].item(),
                        'method': f"{self.method}_ixg_L2",
                        'attribution': list(zip(raw_tokens, raw_token_ixg_L2)),
                    }
                    result_ixg_mean = {
                        'index': example_indices[i],
                        'text': self.tokenizer.decode([t for t in raw_input_ids[i] if not (t in self.tokenizer.all_special_ids and t != self.tokenizer.unk_token_id)], skip_special_tokens=False),
                        'true_label': true_label,
                        'predicted_class': predicted_class,
                        'predicted_class_confidence': prediction_probabilities[i].item(),
                        'target_class': target_class,
                        'target_class_confidence': target_probabilities[i].item(),
                        'method': f"{self.method}_ixg_mean",
                        "attribution": list(zip(raw_tokens, raw_token_ixg_mean)),
                    }
                else:
                    result_ixg_L2 = {
                        'index': example_indices[i],
                        'text': self.tokenizer.decode([t for t in input_ids[i] if not (t in self.tokenizer.all_special_ids and t != self.tokenizer.unk_token_id)], skip_special_tokens=False),
                        'true_label': true_label,
                        'predicted_class': predicted_class,
                        'predicted_class_confidence': prediction_probabilities[i].item(),
                        'target_class': target_class,
                        'target_class_confidence': target_probabilities[i].item(),
                        'method': f"{self.method}_ixg_L2",
                        'attribution': list(zip(tokens, saliency_ixg_L2.tolist()[:real_length])),
                    }

                    result_ixg_mean = {
                        'index': example_indices[i],
                        'text': self.tokenizer.decode([t for t in input_ids[i] if not (t in self.tokenizer.all_special_ids and t != self.tokenizer.unk_token_id)], skip_special_tokens=False),
                        'true_label': true_label,
                        'predicted_class': predicted_class,
                        'predicted_class_confidence': prediction_probabilities[i].item(),
                        'target_class': target_class,
                        'target_class_confidence': target_probabilities[i].item(),
                        'method': f"{self.method}_ixg_mean",
                        "attribution": list(zip(tokens, saliency_ixg_mean.tolist()[:real_length])),
                    }
                all_saliency_ixg_L2_results[i].append(result_ixg_L2)
                all_saliency_ixg_mean_results[i].append(result_ixg_mean)

        saliency_results = {f"{self.method}_ixg_mean": all_saliency_ixg_mean_results}
        return saliency_results
    
    def explain(self, prompts, labels, targets, raw_inputs, example_indices):
        return self.explain_embeddings(prompts=prompts, labels=labels, targets=targets, raw_inputs=raw_inputs, example_indices=example_indices)
    

class AttentionExplainer(BaseExplainer):
    def __init__(self, model, tokenizer, method=None, baseline='zero'):
        # attention explainer can only explain the predicted classes
        self.model = model
        self.model.eval()
        self.tokenizer = tokenizer
        self.device = model.model.get_input_embeddings().weight.device
        self.positive_token = "Yes"
        self.negative_token = "No"
        self.positive_token_id = self.tokenizer(self.positive_token, add_special_tokens=False)["input_ids"][0]
        self.negative_token_id = self.tokenizer(self.negative_token, add_special_tokens=False)["input_ids"][0]

    def _explain(self, input_ids, attention_mask, labels, target_ids, raw_input_ids, example_indices=None):

        batch_size = input_ids.shape[0]
        assert batch_size == 1, "Batch size must be 1 for now"

        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)

        # compute the probability of the target token
        probabilities = torch.softmax(outputs.logits, dim=-1)[:, -1, :]
        positive_prediction_probabilities = probabilities[:, self.positive_token_id]
        negative_prediction_probabilities = probabilities[:, self.negative_token_id]
        # get the predicted ids
        predicted_ids = torch.where(positive_prediction_probabilities > negative_prediction_probabilities, self.positive_token_id, self.negative_token_id).unsqueeze(1)

        if target_ids is None:
            target_ids = predicted_ids
        # get the probability of the target token
        
        prediction_probabilities = probabilities[torch.arange(probabilities.shape[0]), predicted_ids].unsqueeze(1)

        attentions = outputs.attentions

        # Stack attentions over layers
        all_attentions = torch.stack(attentions)
        # Get sequence length and batch size
        seq_len = input_ids.shape[1]
        batch_size = input_ids.shape[0]

        # Expand attention mask to match attention shapes
        # Shape: (batch_size, 1, 1, seq_len)
        attention_mask_expanded = attention_mask.unsqueeze(1).unsqueeze(2)

        # Create a mask for attention weights
        # Shape: (batch_size, 1, seq_len, seq_len)
        attention_mask_matrix = attention_mask_expanded * attention_mask_expanded.transpose(-1, -2)

        # Mask out padding tokens in attention weights
        # We set the attention weights corresponding to padding tokens to zero
        all_attentions = all_attentions * attention_mask_matrix.unsqueeze(0)

        # Normalize the attention weights so that they sum to 1 over the real tokens
        # Sum over the last dimension (seq_len)
        attn_weights_sum = all_attentions.sum(dim=-1, keepdim=True) + 1e-9  # Add epsilon to avoid division by zero
        all_attentions = all_attentions / attn_weights_sum

        # Convert input IDs back to tokens
        tokens_batch = [self.tokenizer.convert_ids_to_tokens(ids) for ids in input_ids]

        # Average Attention
        # Average over heads
        avg_attn_heads = all_attentions.mean(dim=2)  # Shape: (num_layers, batch_size, seq_len, seq_len)
        # Average over layers
        avg_attn = avg_attn_heads.mean(dim=0)  # Shape: (batch_size, seq_len, seq_len)

        # Attention Rollout
        rollout = torch.eye(seq_len).unsqueeze(0).repeat(batch_size, 1, 1)
        rollout = rollout.to(self.device)  # Shape: (batch_size, seq_len, seq_len)
        for attn in avg_attn_heads:
            attn = attn + torch.eye(seq_len).unsqueeze(0).to(self.device) # Add identity for self-connections
            attn = attn / attn.sum(dim=-1, keepdim=True)  # Normalize rows
            rollout = torch.bmm(rollout, attn)  # Batch matrix multiplication
        roll_next_token_attn = rollout[:, -1, :]  # Shape: (batch_size, seq_len)
        
        
        # Attention Flow
        # Take maximum over heads
        attn_per_layer_max = all_attentions.max(dim=2)[0]  # Shape: (num_layers, batch_size, seq_len, seq_len)
        # Initialize cumulative attention starting from [CLS]
        cumulative_attn = torch.zeros(batch_size, seq_len)
        cumulative_attn = cumulative_attn.to(self.device)
        cumulative_attn[:, -1] = 1.0  # [CLS] token index is 0
        for attn in attn_per_layer_max:
            # attn shape: (batch_size, seq_len, seq_len)
            # cumulative_attn shape: (batch_size, seq_len)
            # Compute maximum attention flow to each token
            cumulative_attn = torch.max(cumulative_attn.unsqueeze(-1) * attn, dim=1)[0]
        flow_next_token_attn = cumulative_attn  # Shape: (batch_size, seq_len)

        # Extract attention from [CLS] token
        avg_next_token_attn = avg_attn[:, -1, :]  # Shape: (batch_size, seq_len)


        all_raw_attention_explanations = [[] for _ in range(batch_size)]
        all_attention_rollout_explanations = [[] for _ in range(batch_size)]
        all_attention_flow_explanations = [[] for _ in range(batch_size)]

        # For each example in the batch, print the attention scores
        for explained_target_ids in target_ids:
            explained_target_ids = explained_target_ids.unsqueeze(0)
            target_probabilities = probabilities[torch.arange(probabilities.shape[0]), explained_target_ids].unsqueeze(1)
            for i in range(batch_size):
                true_label = labels[i] if labels is not None else None
                if raw_input_ids is not None:
                    def find_sublist_indexes(full, sub):
                        n, m = len(full), len(sub)
                        for i in range(n - m + 1):
                            if full[i:i + m] == sub:
                                return list(range(i, i + m))
                        return []
                    raw_input_indexes = find_sublist_indexes(input_ids[i].detach().cpu().float().numpy().tolist(), raw_input_ids[i].detach().cpu().float().numpy().tolist())
                    if len(raw_input_indexes) == 0:
                        print(f"Warning: raw_input_ids not found in input_ids for example {example_indices[i]}, return the original input")
                        raw_input_ids = None
                
                tokens = tokens_batch[i]          
                valid_len = attention_mask[i].sum().item()  # Number of real tokens
                raw_attention_attribution = avg_next_token_attn[i][:int(valid_len)].cpu().float().numpy()
                attention_rollout_attribution = roll_next_token_attn[i][:int(valid_len)].cpu().float().numpy()
                attention_flow_attribution = flow_next_token_attn[i][:int(valid_len)].cpu().float().numpy()
                target_token = self.tokenizer.convert_ids_to_tokens(explained_target_ids[i].detach().cpu().float().numpy().tolist())[0]  
                prediction_token = self.tokenizer.convert_ids_to_tokens(predicted_ids[i].detach().cpu().float().numpy().tolist())[0]
                if prediction_token == "Yes":
                    predicted_class = 1
                elif prediction_token == "No":
                    predicted_class = 0
                else:
                    raise ValueError(f"Warning: predicted class {prediction_token} is not Yes or No")
                if target_token == "Yes":
                    target_class = 1
                elif target_token == "No":
                    target_class = 0
                else:
                    print(f"Warning: target class {target_token} is not Yes or No")
                    target_class = target_token
                if raw_input_ids is not None:
                    raw_tokens = self.tokenizer.convert_ids_to_tokens(raw_input_ids[i].detach().cpu().float().numpy().tolist())
                    raw_token_attention = [raw_attention_attribution.tolist()[raw_input_index] for raw_input_index in raw_input_indexes]  
                    raw_token_attention_rollout = [attention_rollout_attribution.tolist()[raw_input_index] for raw_input_index in raw_input_indexes]
                    raw_token_attention_flow = [attention_flow_attribution.tolist()[raw_input_index] for raw_input_index in raw_input_indexes]
                    raw_tokens = [token for token in raw_tokens if token != self.tokenizer.pad_token]
                    raw_attention_result = {
                    'index': example_indices[i],
                    'text': self.tokenizer.decode([t for t in raw_input_ids[i] if not (t in self.tokenizer.all_special_ids and t != self.tokenizer.unk_token_id)], skip_special_tokens=False),
                    'true_label': true_label,
                    'predicted_class': predicted_class,
                    'predicted_class_confidence': prediction_probabilities[i].item(),
                    'target_class': target_class,
                    'target_class_confidence': target_probabilities[i].item(),
                    'method': 'raw_attention',
                    'attribution': list(zip(raw_tokens, raw_token_attention)),
                    }
                    attention_rollout_result = {
                        'index': example_indices[i],
                        'text': self.tokenizer.decode([t for t in raw_input_ids[i] if not (t in self.tokenizer.all_special_ids and t != self.tokenizer.unk_token_id)], skip_special_tokens=False),
                        'true_label': true_label,
                        'predicted_class': predicted_class,
                        'predicted_class_confidence': prediction_probabilities[i].item(),
                        'target_class': target_class,
                        'target_class_confidence': target_probabilities[i].item(),
                        'method': 'attention_rollout',
                        'attribution': list(zip(raw_tokens, raw_token_attention_rollout)),
                    }
                    attention_flow_result = {
                        'index': example_indices[i],
                        'text': self.tokenizer.decode([t for t in raw_input_ids[i] if not (t in self.tokenizer.all_special_ids and t != self.tokenizer.unk_token_id)], skip_special_tokens=False),
                        'true_label': true_label,
                        'predicted_class': predicted_class,
                        'predicted_class_confidence': prediction_probabilities[i].item(),
                        'target_class': target_class,
                        'target_class_confidence': target_probabilities[i].item(),
                        'method': 'attention_flow',
                        'attribution': list(zip(raw_tokens, raw_token_attention_flow)),
                    }
                else:
                    raw_attention_result = {
                        'index': example_indices[i],
                        'text': self.tokenizer.decode([t for t in input_ids[i] if not (t in self.tokenizer.all_special_ids and t != self.tokenizer.unk_token_id)], skip_special_tokens=False),
                        'true_label': true_label,
                        'predicted_class': predicted_class,
                        'predicted_class_confidence': prediction_probabilities[i].item(),
                        'target_class': target_class,
                        'target_class_confidence': target_probabilities[i].item(),
                        'method': 'raw_attention',
                        'attribution': list(zip(tokens[:int(valid_len)], raw_attention_attribution.tolist())),
                    }    
                    attention_rollout_result = {
                        'index': example_indices[i],
                        'text': self.tokenizer.decode([t for t in input_ids[i] if not (t in self.tokenizer.all_special_ids and t != self.tokenizer.unk_token_id)], skip_special_tokens=False),
                        'true_label': true_label,
                        'predicted_class': predicted_class,
                        'predicted_class_confidence': prediction_probabilities[i].item(),
                        'target_class': target_class,
                        'target_class_confidence': target_probabilities[i].item(),
                        'method': 'attention_rollout',
                        'attribution': list(zip(tokens[:int(valid_len)], attention_rollout_attribution.tolist())),
                    }
                    attention_flow_result = {
                        'index': example_indices[i],
                        'text': self.tokenizer.decode([t for t in input_ids[i] if not (t in self.tokenizer.all_special_ids and t != self.tokenizer.unk_token_id)], skip_special_tokens=False),
                        'true_label': true_label,
                        'predicted_class': predicted_class,
                        'predicted_class_confidence': prediction_probabilities[i].item(),
                        'target_class': target_class,
                        'target_class_confidence': target_probabilities[i].item(),
                        'method': 'attention_flow',
                        'attribution': list(zip(tokens[:int(valid_len)], attention_flow_attribution.tolist())),
                    }             
                all_raw_attention_explanations[i].append(raw_attention_result)
                all_attention_rollout_explanations[i].append(attention_rollout_result)
                all_attention_flow_explanations[i].append(attention_flow_result)
        attention_explanations = {"raw_attention": all_raw_attention_explanations, "attention_rollout": all_attention_rollout_explanations, "attention_flow": all_attention_flow_explanations}
        return attention_explanations
    
    def explain(self, prompts, labels, targets, raw_inputs, example_indices):
        return self.explain_tokens(prompts=prompts, labels=labels, targets=targets, raw_inputs=raw_inputs, example_indices=example_indices)
    
    
class GradientNPropabationExplainer(BaseExplainer):
    def __init__(self, model, tokenizer, method='saliency', baseline='zero'):
        self.model = GPTEmbeddingModelWrapper(model)
        self.model.eval()
        #self.model.to(model.model.get_input_embeddings().weight.device)
        self.tokenizer = tokenizer
        self.positive_token = "Yes"
        self.negative_token = "No"
        self.positive_token_id = self.tokenizer(self.positive_token, add_special_tokens=False)["input_ids"][0]
        self.negative_token_id = self.tokenizer(self.negative_token, add_special_tokens=False)["input_ids"][0]

        self.method = method
        if method == 'Saliency':
            self.explainer = Saliency(self.model)
        elif method == 'InputXGradient':
            self.explainer = InputXGradient(self.model)
        elif method == 'IntegratedGradients':
            self.explainer = IntegratedGradients(self.model)
        elif method == 'DeepLift':
            self.explainer = DeepLift(self.model)
        elif method == 'GuidedBackprop':
            self.explainer = GuidedBackprop(self.model)
        elif method == 'SIG':
            self.explainer = SequentialIntegratedGradients(self.model)
        else:
            raise ValueError(f"Invalid method {method}")
        self.device = model.model.get_input_embeddings().weight.device
        if baseline == 'zero':
            self.baseline = None
        elif baseline == 'mask':
            self.baseline = self.tokenizer.mask_token_id
        elif baseline == 'pad':
            self.baseline = self.tokenizer.pad_token_id
        else:
            raise ValueError(f"Invalid baseline {baseline}")

    def _explain(self, input_ids, attention_mask, position_ids=None, labels=None, target_ids=None, raw_input_ids=None, example_indices=None):
        """
        if position_ids is None:
            #position_ids = torch.arange(input_ids.size(1), dtype=torch.long, device=self.device).unsqueeze(0).repeat(input_ids.size(0), 1)
            # generate according to attention mask, starting from the first non-padding token
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 0)
        """

        batch_size = input_ids.shape[0]
        assert batch_size == 1, "Batch size must be 1 for now"

        if hasattr(self.model.model, "transformer") and hasattr(self.model.model.transformer, "wte"):
            wte = self.model.model.transformer.wte ## gpt model
        elif hasattr(self.model.model, "model") and hasattr(self.model.model.model, "embed_tokens"):
            wte = self.model.model.model.embed_tokens ## llama model
        else:
            raise ValueError("Model is not supported, cannot extract embeddings")
        #wpe = self.model.model.transformer.wpe
        embeddings = wte(input_ids) 
        embeddings.requires_grad_()

        # Get the model's predictions
        with torch.no_grad():
            outputs = self.model(embeddings, attention_mask=attention_mask)
        probabilities = torch.softmax(outputs, dim=-1)
        positive_prediction_probabilities = probabilities[:, self.positive_token_id]
        negative_prediction_probabilities = probabilities[:, self.negative_token_id]
        # get the predicted ids
        predicted_ids = torch.where(positive_prediction_probabilities > negative_prediction_probabilities, self.positive_token_id, self.negative_token_id).unsqueeze(1)

        if target_ids is None:
            target_ids = predicted_ids
            #target_ids = target_ids.unsqueeze(-1)
        # get the probability of the target token
        prediction_probabilities = probabilities[torch.arange(probabilities.shape[0]), predicted_ids.squeeze(1)].unsqueeze(1) # shape: [batch_size, 1]
        
        
        all_saliency_L2_results = [[] for _ in range(batch_size)]
        all_saliency_mean_results = [[] for _ in range(batch_size)]
        
        
        # explain all targets
        for explained_target_ids in target_ids:
            explained_target_ids = explained_target_ids.unsqueeze(0)
            target_probabilities = probabilities[torch.arange(probabilities.shape[0]), explained_target_ids.squeeze(1)].unsqueeze(1) # shape: [batch_size, 1]
            if self.method == 'Saliency':
                attributions = self.explainer.attribute(
                    inputs=(embeddings),
                    target=explained_target_ids.squeeze(),
                    additional_forward_args=(attention_mask,),
                    abs=False,
                )
            elif self.method == 'IntegratedGradients' or self.method == 'DeepLift' or self.method == 'SIG':
                if self.baseline is not None:
                    token_baseline_ids = torch.ones_like(input_ids) * self.baseline 
                    baselines = wte(token_baseline_ids)
                else:
                    baselines = None
                if self.method == 'IntegratedGradients':
                    # try to generate with 20 steps, if get cuda ood error, then keep reducing the steps by 5
                    n_steps = 20
                    try:  
                        attributions = self.explainer.attribute(
                            inputs=(embeddings),
                            baselines=baselines,
                            target=explained_target_ids.squeeze(),
                            additional_forward_args=(attention_mask,),
                            n_steps=n_steps,
                        )
                    except RuntimeError as e:
                        # clear cache
                        empty_all_cuda_caches()
                        if 'out of memory' in str(e):
                            try:
                                attributions = self.explainer.attribute(
                                    inputs=(embeddings),
                                    baselines=baselines,
                                    target=explained_target_ids.squeeze(),
                                    additional_forward_args=(attention_mask,),
                                    n_steps=10,
                                )
                                print(f"Warning: CUDA out of memory, reduce n_steps to 10")
                            except RuntimeError as e:
                                # clear cache
                                empty_all_cuda_caches()
                                print(f"Warning: {e}, return zero attributions")
                                attributions = torch.ones_like(embeddings) * 1e-12
                            empty_all_cuda_caches()
                        else:
                            # clear cache   
                            empty_all_cuda_caches()
                            # generate a attribution with all 1e-12
                            print(f"Warning: {e}, return zero attributions")
                            attributions = torch.ones_like(embeddings) * 1e-12
                    

                else:
                    attributions = self.explainer.attribute(
                        inputs=(embeddings),
                        baselines=baselines,
                        target=explained_target_ids.squeeze(),
                        additional_forward_args=(attention_mask,),
                    )
            else:
                attributions = self.explainer.attribute(
                    inputs=(embeddings),
                    target=explained_target_ids.squeeze(),
                    additional_forward_args=(attention_mask,)
                )
    
            attributions_all = attributions
            # clear cache
            empty_all_cuda_caches()


            for i in range(batch_size):
                true_label = labels[i] if labels is not None else None
                # find the index of the raw_input_ids in the input_ids
                if raw_input_ids is not None:
                    def find_sublist_indexes(full, sub):
                        n, m = len(full), len(sub)
                        for i in range(n - m + 1):
                            if full[i:i + m] == sub:
                                return list(range(i, i + m))
                        return []
                    raw_input_indexes = find_sublist_indexes(input_ids[i].detach().cpu().float().numpy().tolist(), raw_input_ids[i].detach().cpu().float().numpy().tolist())
                    if len(raw_input_indexes) == 0:
                        print(f"Warning: raw_input_ids not found in input_ids for example {example_indices[i]}, return the original input")
                        raw_input_ids = None
                tokens = self.tokenizer.convert_ids_to_tokens(input_ids[i].detach().cpu().float().numpy().tolist())
                if raw_input_ids is not None:
                    raw_tokens = self.tokenizer.convert_ids_to_tokens(raw_input_ids[i].detach().cpu().float().numpy().tolist())
                target_token = self.tokenizer.convert_ids_to_tokens(explained_target_ids[i].detach().cpu().float().numpy().tolist())[0]                  
                prediction_token = self.tokenizer.convert_ids_to_tokens(predicted_ids[i].detach().cpu().float().numpy().tolist())[0]
                if prediction_token == "Yes":
                    predicted_class = 1
                elif prediction_token == "No":
                    predicted_class = 0
                else:
                    raise ValueError(f"Warning: predicted class {prediction_token} is not Yes or No")
                if target_token == "Yes":
                    target_class = 1
                elif target_token == "No":
                    target_class = 0
                else:
                    print(f"Warning: target class {target_token} is not Yes or No")
                    target_class = target_token
                # Compute saliency metrics for each token
                saliency_L2 = torch.norm(attributions_all[i:i+1], dim=-1, p=2).detach().cpu().float().numpy()[0]
                saliency_mean = attributions_all[i:i+1].mean(dim=-1).detach().cpu().float().numpy()[0]
                if raw_input_ids is not None:
                    raw_token_saliency_L2 = [saliency_L2.tolist()[raw_input_index] for raw_input_index in raw_input_indexes]  
                    raw_token_saliency_mean = [saliency_mean.tolist()[raw_input_index] for raw_input_index in raw_input_indexes] 
                    raw_tokens = [token for token in raw_tokens if token != self.tokenizer.pad_token]  
                # Collect results for the current example and class
                # skip padding tokens
                # tokens = [token for token in tokens if token != self.tokenizer.pad_token]
                 
                
                real_length = len(tokens)
                if raw_input_ids is not None:
                    result_L2 = {
                    'index': example_indices[i],
                    'text': self.tokenizer.decode([t for t in raw_input_ids[i] if not (t in self.tokenizer.all_special_ids and t != self.tokenizer.unk_token_id)], skip_special_tokens=False),
                    'true_label': true_label,
                    'predicted_class': predicted_class,
                    'predicted_class_confidence': prediction_probabilities[i].item(),
                    'target_class': target_class,
                    'target_class_confidence': target_probabilities[i].item(),
                    'method': f"{self.method}_L2",
                    'attribution': list(zip(raw_tokens, raw_token_saliency_L2)),
                }

                    result_mean = {
                        'index': example_indices[i],
                        'text': self.tokenizer.decode([t for t in raw_input_ids[i] if not (t in self.tokenizer.all_special_ids and t != self.tokenizer.unk_token_id)], skip_special_tokens=False),
                        'true_label': true_label,
                        'predicted_class': predicted_class,
                        'predicted_class_confidence': prediction_probabilities[i].item(),
                        'target_class': target_class,
                        'target_class_confidence': target_probabilities[i].item(),
                        'method': f"{self.method}_mean",
                        "attribution": list(zip(raw_tokens, raw_token_saliency_mean)),
                    }
                    all_saliency_L2_results[i].append(result_L2)
                    all_saliency_mean_results[i].append(result_mean)
                else:
                    result_L2 = {
                        'index': example_indices[i],
                        'text': self.tokenizer.decode([t for t in input_ids[i] if not (t in self.tokenizer.all_special_ids and t != self.tokenizer.unk_token_id)], skip_special_tokens=False),
                        'true_label': true_label,
                        'predicted_class': predicted_class,
                        'predicted_class_confidence': prediction_probabilities[i].item(),
                        'target_class': target_class,
                        'target_class_confidence': target_probabilities[i].item(),
                        'method': f"{self.method}_L2",
                        'attribution': list(zip(tokens, saliency_L2.tolist()[:real_length])),
                    }

                    result_mean = {
                        'index': example_indices[i],
                        'text': self.tokenizer.decode([t for t in input_ids[i] if not (t in self.tokenizer.all_special_ids and t != self.tokenizer.unk_token_id)], skip_special_tokens=False),
                        'true_label': true_label,
                        'predicted_class': predicted_class,
                        'predicted_class_confidence': prediction_probabilities[i].item(),
                        'target_class': target_class,
                        'target_class_confidence': target_probabilities[i].item(),
                        'method': f"{self.method}_mean",
                        "attribution": list(zip(tokens, saliency_mean.tolist()[:real_length])),
                    }
                    all_saliency_L2_results[i].append(result_L2)
                    all_saliency_mean_results[i].append(result_mean)
        saliency_results = {f"{self.method}_L2": all_saliency_L2_results, f"{self.method}_mean": all_saliency_mean_results}
        return saliency_results
    
    def explain(self, prompts, labels, targets, raw_inputs, example_indices):
        return self.explain_embeddings(prompts=prompts, labels=labels, targets=targets, raw_inputs=raw_inputs, example_indices=example_indices)
    
    
class OcclusionExplainer(BaseExplainer):
    def __init__(self, model, tokenizer, method='Occlusion', baseline='pad'):
        self.model = GPTModelProbWrapper(model)
        self.model.eval()
        #self.model.to(model.model.get_input_embeddings().weight.device)
        self.tokenizer = tokenizer
        # self.explainer = Occlusion(self.model)
        # we use feature ablation here because it supports ablating only specific tokens
        self.explainer = FeatureAblation(self.model)
        # Occlusion parameters
        self.sliding_window_size = (1,)  # Occlude one token at a time
        if baseline == 'zero':
            self.baseline = None
        elif baseline == 'mask':
            self.baseline = self.tokenizer.mask_token_id
        elif baseline == 'pad':
            self.baseline = self.tokenizer.pad_token_id
        else:
            raise ValueError(f"Invalid baseline {baseline}")
        
        self.positive_token = "Yes"
        self.negative_token = "No"
        self.positive_token_id = self.tokenizer(self.positive_token, add_special_tokens=False)["input_ids"][0]
        self.negative_token_id = self.tokenizer(self.negative_token, add_special_tokens=False)["input_ids"][0]

        self.stride = (1,)
        self.device = model.model.get_input_embeddings().weight.device

    def _explain(self, input_ids, attention_mask, labels=None, target_ids=None, raw_input_ids=None, example_indices=None):

        batch_size = input_ids.shape[0]
        assert batch_size == 1, "Batch size must be 1 for now"

        # Get the model's predictions
        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        probabilities = outputs
        positive_prediction_probabilities = probabilities[:, self.positive_token_id]
        negative_prediction_probabilities = probabilities[:, self.negative_token_id]
        # get the predicted ids
        predicted_ids = torch.where(positive_prediction_probabilities > negative_prediction_probabilities, self.positive_token_id, self.negative_token_id).unsqueeze(1)

        if target_ids is None:
            target_ids = predicted_ids
        # get the probability of the target token
        
        prediction_probabilities = probabilities[torch.arange(probabilities.shape[0]), predicted_ids].unsqueeze(1) # shape: [batch_size, 1]

        all_occlusion_results = [[] for _ in range(batch_size)]
        
        if raw_input_ids is not None:
            # find the index of the raw_input_ids in the input_ids
            def find_sublist_indexes(full, sub):
                n, m = len(full), len(sub)
                for i in range(n - m + 1):
                    if full[i:i + m] == sub:
                        return list(range(i, i + m))
                return []
            raw_input_indexes_list = [find_sublist_indexes(input_ids[i].detach().cpu().float().numpy().tolist(), raw_input_ids[i].detach().cpu().float().numpy().tolist()) for i in range(batch_size)]
            if any(len(indexes) == 0 for indexes in raw_input_indexes_list):
                print(f"Warning: raw_input_ids not found in input_ids for some examples, returning the original input")
                raw_input_ids = None
            if raw_input_ids is not None:
                feature_masks = torch.zeros(input_ids.shape, device=self.device, dtype=torch.int32)
                for i in range(batch_size):
                    for j, raw_input_pos in enumerate(raw_input_indexes_list[i]):
                        feature_masks[i, raw_input_pos] = j + 1
            else:
                feature_masks = None
        else:
            feature_masks = None

        for explained_target_ids in target_ids:
            explained_target_ids = explained_target_ids.unsqueeze(0)
            target_probabilities = probabilities[torch.arange(probabilities.shape[0]), explained_target_ids].unsqueeze(1)
            attributions = self.explainer.attribute(
                inputs=input_ids,
                feature_mask=feature_masks,
                baselines=self.baseline,
                target=explained_target_ids.squeeze(),
                additional_forward_args=(attention_mask,)
            )

            for i in range(batch_size):
                true_label = labels[i] if labels is not None else None
                if raw_input_ids is not None:
                    raw_input_indexes = raw_input_indexes_list[i]
                tokens = self.tokenizer.convert_ids_to_tokens(input_ids[i].detach().cpu().float().numpy().tolist())
                if raw_input_ids is not None:
                    raw_tokens = self.tokenizer.convert_ids_to_tokens(raw_input_ids[i].detach().cpu().float().numpy().tolist())
                target_token = self.tokenizer.convert_ids_to_tokens(explained_target_ids[i].detach().cpu().float().numpy().tolist())[0]
                prediction_token = self.tokenizer.convert_ids_to_tokens(predicted_ids[i].detach().cpu().float().numpy().tolist())[0]
                if prediction_token == "Yes":
                    predicted_class = 1
                elif prediction_token == "No":
                    predicted_class = 0
                else:
                    raise ValueError(f"Warning: predicted class {prediction_token} is not Yes or No")
                if target_token == "Yes":
                    target_class = 1
                elif target_token == "No":
                    target_class = 0
                else:
                    print(f"Warning: target class {target_token} is not Yes or No")
                    target_class = target_token
                attributions_i = attributions.detach().cpu().float().numpy()[i]  # Shape: [seq_len]
                # skip padding tokens
                # # tokens = [token for token in tokens if token != self.tokenizer.pad_token]
                if raw_input_ids is not None:
                    raw_token_attributions_i = [attributions_i.tolist()[raw_input_index] for raw_input_index in raw_input_indexes]  
                    raw_tokens = [token for token in raw_tokens if token != self.tokenizer.pad_token]
                real_length = len(tokens)
                # Collect results for the current example and class

                if raw_input_ids is not None:
                    result = {
                    'index': example_indices[i],
                    'text': self.tokenizer.decode([t for t in raw_input_ids[i] if not (t in self.tokenizer.all_special_ids and t != self.tokenizer.unk_token_id)], skip_special_tokens=False),
                    'true_label': true_label,
                    'predicted_class': predicted_class,
                    'predicted_class_confidence': prediction_probabilities[i].item(),
                    'target_class': target_class,
                    'target_class_confidence': target_probabilities[i].item(),
                    'method': 'Occlusion',
                    'attribution': list(zip(raw_tokens, raw_token_attributions_i)),
                }
                    all_occlusion_results[i].append(result)
                else:
                    result = {
                        'index': example_indices[i],
                        'text': self.tokenizer.decode([t for t in input_ids[i] if not (t in self.tokenizer.all_special_ids and t != self.tokenizer.unk_token_id)], skip_special_tokens=False),
                        'true_label': true_label,
                        'predicted_class': predicted_class,
                        'predicted_class_confidence': prediction_probabilities[i].item(),
                        'target_class': target_class,
                        'target_class_confidence': target_probabilities[i].item(),
                        'method': 'Occlusion',
                        'attribution': list(zip(tokens, attributions_i.tolist()[:real_length])),
                    }
                    all_occlusion_results[i].append(result)
        return {"Occlusion": all_occlusion_results}
    
    def explain(self, prompts, labels, targets, raw_inputs, example_indices):
        return self.explain_tokens(prompts=prompts, labels=labels, targets=targets, raw_inputs=raw_inputs, example_indices=example_indices)
    
    
class ShapleyValueExplainer(BaseExplainer):
    def __init__(self, model, tokenizer, method='ShapleyValue', baseline='pad', n_samples=25):
        self.model = GPTModelWrapper(model)
        self.model.eval()
        #self.model.to(model.model.get_input_embeddings().weight.device)
        self.tokenizer = tokenizer
        self.n_samples = n_samples
        self.method = method
        if method == 'ShapleyValue':
            self.explainer = ShapleyValueSampling(self.model)
        elif method == 'KernelShap':
            self.explainer = KernelShap(self.model)
        else:
            raise ValueError(f"Invalid method {method}")
        self.device = model.model.get_input_embeddings().weight.device
        if baseline == 'zero':
            self.baseline = None
        elif baseline == 'mask':
            self.baseline = self.tokenizer.mask_token_id
        elif baseline == 'pad':
            self.baseline = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.tokenizer.eos_token_id
        else:
            raise ValueError(f"Invalid baseline {baseline}")
        
        self.positive_token = "Yes"
        self.negative_token = "No"
        self.positive_token_id = self.tokenizer(self.positive_token, add_special_tokens=False)["input_ids"][0]
        self.negative_token_id = self.tokenizer(self.negative_token, add_special_tokens=False)["input_ids"][0]

    def _explain(self, input_ids, attention_mask, position_ids=None, labels=None, target_ids=None, raw_input_ids=None, example_indices=None):
        """
        if position_ids is None:
            #position_ids = torch.arange(input_ids.size(1), dtype=torch.long, device=self.device).unsqueeze(0).repeat(input_ids.size(0), 1)
            # generate according to attention mask, starting from the first non-padding token
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 0)
        """

        batch_size = input_ids.shape[0]
        assert batch_size == 1, "Batch size must be 1 for now"

        # Get the model's predictions
        with torch.no_grad():
            outputs = self.model(input_ids, attention_mask=attention_mask)
        probabilities = torch.softmax(outputs, dim=-1)
        positive_prediction_probabilities = probabilities[:, self.positive_token_id]
        negative_prediction_probabilities = probabilities[:, self.negative_token_id]
        # get the predicted ids
        predicted_ids = torch.where(positive_prediction_probabilities > negative_prediction_probabilities, self.positive_token_id, self.negative_token_id).unsqueeze(1)
        
        if target_ids is None:
            target_ids = predicted_ids
            #target_ids = target_ids.unsqueeze(-1)
        # get the probability of the target token
        prediction_probabilities = probabilities[torch.arange(probabilities.shape[0]), predicted_ids.squeeze(1)].unsqueeze(1) # shape: [batch_size, 1]
        all_shap_results = [[] for _ in range(batch_size)]  

        if raw_input_ids is not None:
            # find the index of the raw_input_ids in the input_ids
            def find_sublist_indexes(full, sub):
                n, m = len(full), len(sub)
                for i in range(n - m + 1):
                    if full[i:i + m] == sub:
                        return list(range(i, i + m))
                return []
            raw_input_indexes_list = [find_sublist_indexes(input_ids[i].detach().cpu().float().numpy().tolist(), raw_input_ids[i].detach().cpu().float().numpy().tolist()) for i in range(batch_size)]
            if any(len(indexes) == 0 for indexes in raw_input_indexes_list):
                print(f"Warning: raw_input_ids not found in input_ids for some examples, returning the original input")
                raw_input_ids = None
            if raw_input_ids is not None:
                feature_masks = torch.zeros(input_ids.shape, device=self.device, dtype=torch.int32)
                for i in range(batch_size):
                    for j, raw_input_pos in enumerate(raw_input_indexes_list[i]):
                        feature_masks[i, raw_input_pos] = j + 1
            else:
                feature_masks = None
        else:
            feature_masks = None

        # explain all targets
        for explained_target_ids in target_ids:
            explained_target_ids = explained_target_ids.unsqueeze(0)
            target_probabilities = probabilities[torch.arange(probabilities.shape[0]), explained_target_ids.squeeze(1)].unsqueeze(1) # shape: [batch_size, 1]
            attributions = self.explainer.attribute(
                inputs=input_ids,
                baselines=self.baseline,
                target=explained_target_ids.squeeze(),
                additional_forward_args=(attention_mask,),
                n_samples=self.n_samples,
                feature_mask=feature_masks
            )


            for i in range(batch_size):
                true_label = labels[i] if labels is not None else None
                # find the index of the raw_input_ids in the input_ids
                if raw_input_ids is not None:
                    raw_input_indexes = raw_input_indexes_list[i]
                    
                tokens = self.tokenizer.convert_ids_to_tokens(input_ids[i].detach().cpu().float().numpy().tolist())
                if raw_input_ids is not None:
                    raw_tokens = self.tokenizer.convert_ids_to_tokens(raw_input_ids[i].detach().cpu().float().numpy().tolist())
                target_token = self.tokenizer.convert_ids_to_tokens(explained_target_ids[i].detach().cpu().float().numpy().tolist())[0]                  
                prediction_token = self.tokenizer.convert_ids_to_tokens(predicted_ids[i].detach().cpu().float().numpy().tolist())[0]
                if prediction_token == "Yes":
                    predicted_class = 1
                elif prediction_token == "No":
                    predicted_class = 0
                else:
                    raise ValueError(f"Warning: predicted class {prediction_token} is not Yes or No")
                if target_token == "Yes":
                    target_class = 1
                elif target_token == "No":
                    target_class = 0
                else:
                    print(f"Warning: target class {target_token} is not Yes or No")
                    target_class = target_token
                # Compute saliency metrics for each token
                attribution_i = attributions.detach().cpu().float().numpy()[i]
                if raw_input_ids is not None:
                    raw_token_attribution_i = [attribution_i.tolist()[raw_input_index] for raw_input_index in raw_input_indexes]  
                    raw_tokens = [token for token in raw_tokens if token != self.tokenizer.pad_token]  
                # Collect results for the current example and class
                # skip padding tokens
                # tokens = [token for token in tokens if token != self.tokenizer.pad_token]
                 
                
                real_length = len(tokens)
                if raw_input_ids is not None:
                    result = {
                    'index': example_indices[i],
                    'text': self.tokenizer.decode([t for t in raw_input_ids[i] if not (t in self.tokenizer.all_special_ids and t != self.tokenizer.unk_token_id)], skip_special_tokens=False),
                    'true_label': true_label,
                    'predicted_class': predicted_class,
                    'predicted_class_confidence': prediction_probabilities[i].item(),
                    'target_class': target_class,
                    'target_class_confidence': target_probabilities[i].item(),
                    'method': self.method,
                    'attribution': list(zip(raw_tokens, raw_token_attribution_i)),
                }

                    
                    all_shap_results[i].append(result)
      
                else:
                    result = {
                        'index': example_indices[i],
                        'text': self.tokenizer.decode([t for t in input_ids[i] if not (t in self.tokenizer.all_special_ids and t != self.tokenizer.unk_token_id)], skip_special_tokens=False),
                        'true_label': true_label,
                        'predicted_class': predicted_class,
                        'predicted_class_confidence': prediction_probabilities[i].item(),
                        'target_class': target_class,
                        'target_class_confidence': target_probabilities[i].item(),
                        'method': self.method,
                        'attribution': list(zip(tokens, attribution_i.tolist()[:real_length])),
                    }

                    all_shap_results[i].append(result)
                    
        saliency_results = {self.method: all_shap_results}
        return saliency_results
    
    def explain(self, prompts, labels, targets, raw_inputs, example_indices):
        return self.explain_tokens(prompts=prompts, labels=labels, targets=targets, raw_inputs=raw_inputs, example_indices=example_indices)
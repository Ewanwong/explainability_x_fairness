# implement attention, attention rollout, attention flow, gradient, input x gradient, integrated gradients, deeplift, kernel shap explanations in a differientiable way

import torch

torch.backends.cuda.enable_mem_efficient_sdp(False)  # for computation of second-order gradients
torch.backends.cuda.enable_flash_sdp(False)  

def compute_attribution_loss(model, input_ids, attention_mask=None, token_type_ids=None, position_ids=None, sensitive_token_mask=None, target_classes=None, method="Saliency", aggregation="L1", baseline_token_id=None, steps=50):
    """
    Compute attribution loss using the specified method.
    
    Args:
        model: The model to explain.
        input_ids: Input token IDs.
        attention_mask: Attention mask for the input.
        token_type_ids: Token type IDs (optional).
        position_ids: Position IDs (optional).
        sensitive_token_mask: Mask for sensitive tokens.
        target_classes: Target classes for the input.
        method: Method to use for attribution ("Saliency", "InputXGradient", "IntegratedGradients", "raw_attention", "attention_flow", "attention_rollout", "Occlusion").
        aggregation: Aggregation method ('L1' or 'L2').
        baseline_token_id: Token ID to use as baseline for occlusion and integrated gradients.
        steps: Number of steps for integrated gradients.

    Returns:
        Attribution loss value.
    """
    model.eval()
    if method == "Saliency":
        return gradient_attr(model, input_ids, attention_mask, token_type_ids, position_ids, sensitive_token_mask, target_classes, aggregation)
    elif method == "InputXGradient":
        return input_x_gradient_attr(model, input_ids, attention_mask, token_type_ids, position_ids, sensitive_token_mask, target_classes, aggregation)
    elif method == "IntegratedGradients":
        return integrated_gradients_attr(model, input_ids, attention_mask, token_type_ids, position_ids, sensitive_token_mask, baseline_token_id, target_classes, aggregation, steps)
    elif method == "raw_attention":
        return raw_attention_attr(model, input_ids, attention_mask, token_type_ids, position_ids, sensitive_token_mask, target_classes, aggregation)
    elif method == "attention_rollout":
        return attention_rollout_attr(model, input_ids, attention_mask, token_type_ids, position_ids, sensitive_token_mask, target_classes, aggregation)
    elif method == "attention_flow":
        return attention_flow_attr(model, input_ids, attention_mask, token_type_ids, position_ids, sensitive_token_mask, target_classes, aggregation)
    elif method == "Occlusion":
        return occlusion_attr(model, input_ids, attention_mask, token_type_ids, position_ids, sensitive_token_mask, baseline_token_id=baseline_token_id,
                              target_classes=target_classes,
                              aggregation=aggregation)
    else:
        raise ValueError(f"Method {method} not supported")

def get_embeddings(model, input_ids, attention_mask=None, token_type_ids=None, position_ids=None):
    if token_type_ids is None:
        token_type_ids = torch.zeros_like(input_ids, dtype=torch.long)
    if position_ids is None:
        position_ids = torch.arange(input_ids.size(1), dtype=torch.long, device=input_ids.device).unsqueeze(0).expand(input_ids.size(0), -1)
    if hasattr(model, "distilbert"):
        embeddings = model.distilbert.embeddings(input_ids=input_ids)
    elif hasattr(model, "roberta"):
        embeddings = model.roberta.embeddings(input_ids=input_ids, position_ids=None, token_type_ids=token_type_ids)
    elif hasattr(model, "bert"):
        embeddings = model.bert.embeddings(input_ids=input_ids, position_ids=position_ids, token_type_ids=token_type_ids)
    else:
        raise ValueError("Model not supported")
    embeddings.requires_grad_(True)
    return embeddings

def model_forward(model, embeddings, attention_mask=None):

    head_mask = model.get_head_mask(None, model.config.num_hidden_layers)
    #head_mask = [None] * self.model.config.num_hidden_layers

    if hasattr(model, "distilbert"):
        encoder_outputs = model.distilbert.transformer(
            embeddings,
            attn_mask=attention_mask,
            head_mask=head_mask,
        )
        hidden_state = encoder_outputs[0]
        pooled_output = hidden_state[:, 0]
        pooled_output = model.pre_classifier(pooled_output)
        pooled_output = model.dropout(pooled_output) 
        logits = model.classifier(pooled_output)

    elif hasattr(model, "roberta"):
        extended_attention_mask = model.get_extended_attention_mask(
            attention_mask, embeddings.shape[:2],
        )

        encoder_outputs = model.roberta.encoder(
            embeddings,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
        )
        sequence_output = encoder_outputs[0]
        #sequence_output = self.model.roberta.pooler(sequence_output) if self.model.roberta.pooler is not None else None
        logits = model.classifier(sequence_output)

    elif hasattr(model, "bert"):
        extended_attention_mask = model.get_extended_attention_mask(
            attention_mask, embeddings.shape[:2], embeddings.device
        )

        encoder_outputs = model.bert.encoder(
            embeddings,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
        )
        sequence_output = encoder_outputs[0]
        pooled_output = model.bert.pooler(sequence_output) if model.bert.pooler is not None else None
        pooled_output = model.dropout(pooled_output)
        logits = model.classifier(pooled_output)
    else:
        raise ValueError("Model not supported")

    return logits

def gradient_attr(model, input_ids, attention_mask, token_type_ids, position_ids, sensitive_token_mask, target_classes, aggregation="L1"):
    embeddings = get_embeddings(model, input_ids, attention_mask, token_type_ids, position_ids)
    logits = model_forward(model, embeddings, attention_mask)
    probs = torch.softmax(logits, dim=-1)
    # print(probs)
    if target_classes is None:
        target_classes = torch.argmax(logits, dim=-1)
    target_score = logits[range(logits.size(0)), target_classes].sum()
    
    grads = torch.autograd.grad(
        outputs=target_score,
        inputs=embeddings,
        create_graph=True
    )[0]

    # aggregate gradients
    if aggregation == "L1":
        attr = grads.abs().sum(dim=-1)
    elif aggregation == "L2":
        attr = grads.pow(2).sum(dim=-1)
    
    # take only positions with sensitive token mask
    if sensitive_token_mask is None:
        sensitive_token_mask = attention_mask.clone().to(attention_mask.device)

    attr = attr * sensitive_token_mask

    # attr loss: mean over all sensitive token positions
    attr_loss = attr.sum() / sensitive_token_mask.sum().clamp(min=1.0)
    return attr_loss

def input_x_gradient_attr(model, input_ids, attention_mask, token_type_ids, position_ids, sensitive_token_mask, target_classes, aggregation="L1"):
    embeddings = get_embeddings(model, input_ids, attention_mask, token_type_ids, position_ids)
    logits = model_forward(model, embeddings, attention_mask)
    probs = torch.softmax(logits, dim=-1)
    # print(probs)
    if target_classes is None:
        target_classes = torch.argmax(logits, dim=-1)
    target_score = logits[range(logits.size(0)), target_classes].sum()
    
    grads = torch.autograd.grad(
        outputs=target_score,
        inputs=embeddings,
        create_graph=True
    )[0]

    # aggregate gradients
    if aggregation == "L1":
        attr = (embeddings * grads).abs().sum(dim=-1)
    elif aggregation == "L2":
        attr = (embeddings * grads).pow(2).sum(dim=-1)

    # take only positions with sensitive token mask
    if sensitive_token_mask is None:
        sensitive_token_mask = attention_mask.clone().to(attention_mask.device)
    attr = attr * sensitive_token_mask

    # attr loss: mean over all sensitive token positions
    attr_loss = attr.sum() / sensitive_token_mask.sum().clamp(min=1.0)
    return attr_loss

def integrated_gradients_attr(model, input_ids, attention_mask, token_type_ids, position_ids, sensitive_token_mask, baseline_token_id=None, target_classes=None, aggregation="L1", steps=50):
    embeddings = get_embeddings(model, input_ids, attention_mask, token_type_ids, position_ids)
    logits = model_forward(model, embeddings, attention_mask)
    if target_classes is None:
        target_classes = torch.argmax(logits, dim=-1)
    if baseline_token_id is not None:
        baseline_input_ids = torch.ones_like(input_ids) * baseline_token_id
        baseline_embeddings = get_embeddings(model, baseline_input_ids, attention_mask, token_type_ids, position_ids)
    else:
        baseline_embeddings = torch.zeros_like(embeddings).to(embeddings.device)

    # baseline_embeddings.requires_grad_(True)
    diff = (embeddings - baseline_embeddings).detach()
    total_grads = torch.zeros_like(embeddings)
    for step in range(1, steps+1):
        alpha = step  / steps
        scaled = baseline_embeddings + alpha * diff
        scaled.requires_grad_(True)
        intermediate_logits = model_forward(model, scaled, attention_mask)
        target_score = intermediate_logits[range(intermediate_logits.size(0)), target_classes].sum()
        grads = torch.autograd.grad(
            outputs=target_score,
            inputs=scaled,
            create_graph=True
        )[0]
        total_grads += grads
        del scaled  # Free memory
        del intermediate_logits  # Free memory
        del grads  # Free memory
        del target_score  # Free memory

    avg_grads = total_grads / steps
    attr = diff * avg_grads

    # aggregate gradients
    if aggregation == "L1":
        attr = attr.abs().sum(dim=-1)
    elif aggregation == "L2":
        attr = attr.pow(2).sum(dim=-1)

    # take only positions with sensitive token mask
    if sensitive_token_mask is None:
        sensitive_token_mask = attention_mask.clone().to(attention_mask.device)
    attr = attr * sensitive_token_mask

    # attr loss: mean over all sensitive token positions
    attr_loss = attr.sum() / sensitive_token_mask.sum().clamp(min=1.0)
    return attr_loss


def raw_attention_attr(model, input_ids, attention_mask, token_type_ids, position_ids, sensitive_token_mask, target_classes=None, aggregation="L1"):
    outputs = model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, position_ids=position_ids, output_attentions=True)
    attentions = outputs.attentions  # L x batch x heads x seq x seq
    all_attentions = torch.stack(attentions)  # L x batch x heads x seq x seq
    attention_mask_expanded = attention_mask.unsqueeze(1).unsqueeze(2)
    attention_mask_matrix = attention_mask_expanded * attention_mask_expanded.transpose(-1, -2)
    all_attentions = all_attentions * attention_mask_matrix.unsqueeze(0)
    # normalize attention weights
    attn_weights_sum = all_attentions.sum(dim=-1, keepdim=True) + 1e-9  # Add epsilon to avoid division by zero
    all_attentions = all_attentions / attn_weights_sum
    # mean over heads, mean over layers
    avg_attn_heads = all_attentions.mean(dim=2)
    avg_attn = avg_attn_heads.mean(dim=0)
    attr = avg_attn[:, 0, :]  # token contributions to [CLS] (position 0)

    # take only positions with sensitive token mask
    if sensitive_token_mask is None:
        sensitive_token_mask = attention_mask.clone().to(attention_mask.device)
    attr = attr * sensitive_token_mask

    # attr loss: mean over all sensitive token positions
    attr_loss = attr.sum() / sensitive_token_mask.sum().clamp(min=1.0)
    return attr_loss

def attention_rollout_attr(model, input_ids, attention_mask, token_type_ids, position_ids, sensitive_token_mask, target_classes=None, aggregation="L1"):
    outputs = model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, position_ids=position_ids, output_attentions=True)
    attentions = outputs.attentions  # L x batch x heads x seq x seq
    all_attentions = torch.stack(attentions)  # L x batch x heads x seq x seq
    attention_mask_expanded = attention_mask.unsqueeze(1).unsqueeze(2)
    attention_mask_matrix = attention_mask_expanded * attention_mask_expanded.transpose(-1, -2)
    all_attentions = all_attentions * attention_mask_matrix.unsqueeze(0)
    # normalize attention weights
    attn_weights_sum = all_attentions.sum(dim=-1, keepdim=True) + 1e-9  # Add epsilon to avoid division by zero
    all_attentions = all_attentions / attn_weights_sum
    # mean over heads, mean over layers
    avg_attn_heads = all_attentions.mean(dim=2)
    # avg_attn = avg_attn_heads.mean(dim=0)

    seq_len = input_ids.size(1)
    batch_size = input_ids.size(0)
    rollout = torch.eye(seq_len).unsqueeze(0).repeat(batch_size, 1, 1).to(input_ids.device)  # Shape: (batch_size, seq_len, seq_len)
    for attn in avg_attn_heads:
        attn = attn + torch.eye(seq_len).unsqueeze(0).to(input_ids.device)  # Add identity for self-connections
        attn = attn / attn.sum(dim=-1, keepdim=True)  # Normalize rows
        rollout = torch.bmm(rollout, attn)  # Batch matrix multiplication
    # token contributions to [CLS] (position 0)
    attr = rollout[:, 0, :]

    # take only positions with sensitive token mask
    if sensitive_token_mask is None:
        sensitive_token_mask = attention_mask.clone().to(attention_mask.device)
    attr = attr * sensitive_token_mask

    # attr loss: mean over all sensitive token positions
    attr_loss = attr.sum() / sensitive_token_mask.sum().clamp(min=1.0)
    return attr_loss

def attention_flow_attr(model, input_ids, attention_mask, token_type_ids, position_ids, sensitive_token_mask, target_classes=None, aggregation="L1"):
    outputs = model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, position_ids=position_ids, output_attentions=True)
    attentions = outputs.attentions  # L x batch x heads x seq x seq
    all_attentions = torch.stack(attentions)  # L x batch x heads x seq x seq
    attention_mask_expanded = attention_mask.unsqueeze(1).unsqueeze(2)
    attention_mask_matrix = attention_mask_expanded * attention_mask_expanded.transpose(-1, -2)
    all_attentions = all_attentions * attention_mask_matrix.unsqueeze(0)
    # normalize attention weights
    attn_weights_sum = all_attentions.sum(dim=-1, keepdim=True) + 1e-9  # Add epsilon to avoid division by zero
    all_attentions = all_attentions / attn_weights_sum
    # mean over heads, mean over layers
    # avg_attn_heads = all_attentions.mean(dim=2)
    # avg_attn = avg_attn_heads.mean(dim=0)

    seq_len = input_ids.size(1)
    batch_size = input_ids.size(0)
    attn_per_layer_max = all_attentions.max(dim=2)[0]  # Shape: (num_layers, batch_size, seq_len, seq_len)
    # Initialize cumulative attention starting from [CLS]
    cumulative_attn = torch.zeros(batch_size, seq_len).to(input_ids.device)
    cumulative_attn[:, 0] = 1.0  # [CLS] token index is 0
    for attn in attn_per_layer_max:
        # attn shape: (batch_size, seq_len, seq_len)
        # cumulative_attn shape: (batch_size, seq_len)
        # Compute maximum attention flow to each token
        cumulative_attn = torch.max(cumulative_attn.unsqueeze(-1) * attn, dim=1)[0]
    flow_cls_attn = cumulative_attn  # Shape: (batch_size, seq_len)
    attr = flow_cls_attn

    # take only positions with sensitive token mask
    if sensitive_token_mask is None:
        sensitive_token_mask = attention_mask.clone().to(attention_mask.device)
    attr = attr * sensitive_token_mask

    # attr loss: mean over all sensitive token positions
    attr_loss = attr.sum() / sensitive_token_mask.sum().clamp(min=1.0)
    return attr_loss

def occlusion_attr(model, input_ids, attention_mask, token_type_ids, position_ids, sensitive_token_mask, baseline_token_id=None, target_classes=None, aggregation="L1"):
    embeddings = get_embeddings(model, input_ids, attention_mask, token_type_ids, position_ids)
    logits = model_forward(model, embeddings, attention_mask)
    probs = torch.softmax(logits, dim=-1)
    if target_classes is None:
        target_classes = torch.argmax(logits, dim=-1)
    target_probs = probs[range(logits.size(0)), target_classes].sum()

    if sensitive_token_mask is None:
        sensitive_token_mask = attention_mask.clone().to(attention_mask.device)
    if baseline_token_id is not None:
        baseline_input_ids = input_ids.clone().to(input_ids.device)
        # replace sensitive token positions with occlusion token ids (same shape as input_ids)
        baseline_input_ids[sensitive_token_mask] = baseline_input_ids[sensitive_token_mask] * 0 + baseline_token_id
        baseline_embeddings = get_embeddings(model, baseline_input_ids, attention_mask, token_type_ids, position_ids)
    else:
        baseline_embeddings = embeddings.clone().to(embeddings.device)
        # replace sensitive token positions with zero embeddings (same shape as embeddings)
        baseline_embeddings[sensitive_token_mask] = 0.0
        
    baseline_embeddings.requires_grad_(True)

    occlusion_logits = model_forward(model, embeddings, attention_mask)
    occlusion_probs = torch.nn.functional.softmax(occlusion_logits, dim=-1)
    occlusion_target_probs = occlusion_probs[range(occlusion_probs.size(0)), target_classes]

    # calculate the difference in probabilities
    attr = target_probs - occlusion_target_probs
    attr_loss = attr.abs().sum()
    
    # compute num of instances in the batch with non-zero sensitive token mask
    num_instances = sensitive_token_mask.sum(dim=-1)
    num_instances = num_instances[num_instances > 0].numel()
    if num_instances > 0:
        attr_loss = attr_loss / num_instances
    else:
        attr_loss = torch.tensor(0.0, device=attr_loss.device)
    return attr_loss

def deeplift_attr(model, input_ids, attention_mask, token_type_ids, position_ids, sensitive_token_mask, baseline_token_id=None, target_classes=None, aggregation="L1"):
    # DeepLIFT is a method that requires a baseline input and is not easily differentiable.
    # This is a placeholder for the actual implementation.
    # In practice, you would use a library like `deeplift` to compute DeepLIFT.
    raise NotImplementedError("DeepLIFT is not implemented in this example.")

def kernel_shap_attr(model, embeddings, attention_mask, sensitive_token_mask, target_classes=None, num_samples=100):
    # Kernel SHAP is a complex method that requires sampling and is not easily differentiable.
    # This is a placeholder for the actual implementation.
    # In practice, you would use a library like `shap` to compute Kernel SHAP.
    raise NotImplementedError("Kernel SHAP is not implemented in this example.")


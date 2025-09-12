import torch
from captum.attr import Saliency, DeepLift, GuidedBackprop, InputXGradient, IntegratedGradients, Occlusion, ShapleyValueSampling, DeepLiftShap, GradientShap, KernelShap 

torch.backends.cuda.enable_mem_efficient_sdp(False)  # for computation of second-order gradients
torch.backends.cuda.enable_flash_sdp(False)  

def get_embeddings(model, input_ids, attention_mask=None, token_type_ids=None, position_ids=None):
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

def model_forward(model, input_ids, attention_mask=None, token_type_ids=None, position_ids=None):
    return model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, position_ids=position_ids).logits

def model_forward_embeddings(model, embeddings, attention_mask=None):

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

def _compute_attribution_loss_embeddings(model, 
                                         explainer, 
                                         input_ids, 
                                         attention_mask=None, 
                                         token_type_ids=None, 
                                         position_ids=None, 
                                         sensitive_token_mask=None, 
                                         target_class=None,
                                         explanation_method=None,
                                         aggregation="L1",
                                         baseline_token_id=None,
                                         n_steps=50,
                                         ):
    if sensitive_token_mask is None:
        sensitive_token_mask = attention_mask.clone().to(input_ids.device)
    if position_ids is None:
        position_ids = torch.arange(input_ids.size(1), dtype=torch.long, device=input_ids.device).unsqueeze(0).expand_as(input_ids)
    if token_type_ids is None:
        token_type_ids = torch.zeros_like(input_ids, dtype=torch.long, device=input_ids.device)
    embeddings = get_embeddings(model, input_ids, attention_mask, token_type_ids, position_ids)

    
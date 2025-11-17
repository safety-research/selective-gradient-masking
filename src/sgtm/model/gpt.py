from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
from transformers.cache_utils import Cache, DynamicCache
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    BaseModelOutputWithPastAndCrossAttentions,
    CausalLMOutputWithCrossAttentions,
    CausalLMOutputWithPast,
)
from transformers.models.gpt_neo.modeling_gpt_neo import (
    GPTNeoBlock,
    GPTNeoModel,
    GPTNeoForCausalLM,
    GPTNeoSelfAttention,
)

from sgtm.model.activation_masking import (
    GPTNeoMLPActivationMasking,
    GPTNeoSelfAttentionActivationMasking,
)
from sgtm.model.gradient_routing import GPTNeoMLPGradientRouting, GPTNeoSelfAttentionGradientRouting
from sgtm.model.parameter_masking import (
    GPTNeoMLPParameterMasking,
    GPTNeoSelfAttentionParameterMasking,
    GPTNeoMLPParameterMaskingNoProj,
    GPTNeoSelfAttentionParameterMaskingNoProj,
)
import logging


def _validate_mode_parameter(sgtm_mode):
    """Validate and normalize the sgtm mode parameter.

    Args:
        sgtm_mode: Can be "forget", "default", "retain", or legacy boolean values

    Returns:
        str: Normalized mode parameter

    Raises:
        ValueError: If mode parameter is invalid
    """
    if sgtm_mode is True:
        # Legacy compatibility: True -> "forget"
        return "forget"
    elif sgtm_mode is False or sgtm_mode is None:
        # Legacy compatibility: False/None -> "default"
        return "default"
    elif sgtm_mode in ["forget", "default", "retain"]:
        return sgtm_mode
    else:
        raise ValueError(f"Invalid sgtm mode parameter: {sgtm_mode}. Must be 'forget', 'default', 'retain', True, or False.")


class AttentionWrapper(nn.Module):
    def __init__(self, attention):
        super().__init__()
        self.attention = attention

    def forward(self, *args, **kwargs):
        if not hasattr(self.attention, "adjust_gradients"):
            kwargs.pop("sgtm_mode")
        return self.attention.forward(*args, **kwargs)

    def adjust_gradients(self, sgtm_mode="default"):
        if hasattr(self.attention, "adjust_gradients"):
            self.attention.adjust_gradients(sgtm_mode)

    def ablate(self, trainable=False):
        if hasattr(self.attention, "ablate"):
            self.attention.ablate(trainable=trainable)


class GPTNeoBlockSGTM(GPTNeoBlock):
    CLASSES = {
        "activation_masking": (GPTNeoSelfAttentionActivationMasking, GPTNeoMLPActivationMasking),
        "parameter_masking": (GPTNeoSelfAttentionParameterMasking, GPTNeoMLPParameterMasking),
        "parameter_masking_no_mlp_proj": (GPTNeoSelfAttentionParameterMasking, GPTNeoMLPParameterMaskingNoProj),
        "parameter_masking_no_attn_proj": (GPTNeoSelfAttentionParameterMaskingNoProj, GPTNeoMLPParameterMasking),
        "parameter_masking_no_proj": (GPTNeoSelfAttentionParameterMaskingNoProj, GPTNeoMLPParameterMaskingNoProj),
        "parameter_masking_no_proj_no_attn": (GPTNeoSelfAttention, GPTNeoMLPParameterMaskingNoProj),
        "gradient_routing": (GPTNeoSelfAttentionGradientRouting, GPTNeoMLPGradientRouting),
    }

    def __init__(self, config, layer_id=None):
        super().__init__(config, layer_id)
        hidden_size = config.hidden_size
        inner_dim = config.intermediate_size if config.intermediate_size is not None else 4 * hidden_size
        attention_type = config.attention_layers[layer_id]

        attn_class, mlp_class = self.CLASSES[config.masking_strategy]

        self.attn = AttentionWrapper(attn_class(config, attention_type, layer_id))
        self.mlp = mlp_class(inner_dim, config)

    def adjust_gradients(self, sgtm_mode="default"):
        self.attn.adjust_gradients(sgtm_mode)
        self.mlp.adjust_gradients(sgtm_mode)

    def ablate(self, trainable=False):
        self.attn.ablate(trainable=trainable)
        self.mlp.ablate(trainable=trainable)

    def forward(
        self,
        hidden_states,
        layer_past=None,
        attention_mask=None,
        head_mask=None,
        use_cache=False,
        output_attentions=False,
        cache_position=None,
        sgtm_mode="default",
    ):
        residual = hidden_states
        hidden_states = self.ln_1(hidden_states)
        attn_outputs = self.attn(
            hidden_states,
            layer_past=layer_past,
            attention_mask=attention_mask,
            head_mask=head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            cache_position=cache_position,
            sgtm_mode=sgtm_mode,
        )
        attn_output = attn_outputs[0]  # output_attn: a, present, (attentions)
        outputs = attn_outputs[1:]
        # residual connection
        hidden_states = attn_output + residual

        residual = hidden_states
        hidden_states = self.ln_2(hidden_states)
        feed_forward_hidden_states = self.mlp(hidden_states, sgtm_mode)
        # residual connection
        hidden_states = residual + feed_forward_hidden_states

        if use_cache:
            outputs = (hidden_states,) + outputs
        else:
            outputs = (hidden_states,) + outputs[1:]

        return outputs  # hidden_states, past_kv, attentions


class GPTNeoModelSGTM(GPTNeoModel):
    def __init__(self, config):
        super().__init__(config)
        self.masked_layers = []
        if hasattr(config, "masked_layers") and config.masked_layers is not None:
            self.masked_layers = list(config.masked_layers)
            blocks = []
            for i in range(config.num_layers):
                if i in self.masked_layers:
                    blocks.append(GPTNeoBlockSGTM(config, layer_id=i))
                else:
                    blocks.append(GPTNeoBlock(config, layer_id=i))
            self.h = nn.ModuleList(blocks)
        self.post_init()

    def adjust_gradients(self, sgtm_mode="default"):
        sgtm_mode = _validate_mode_parameter(sgtm_mode)
        for i in self.masked_layers:
            self.h[i].adjust_gradients(sgtm_mode)

        if getattr(self.config, "sgtm_mask_embeddings", False) is True and sgtm_mode == "forget":
            self.wte.weight.grad = torch.zeros_like(self.wte.weight.grad)
            self.wpe.weight.grad = torch.zeros_like(self.wpe.weight.grad)

    def ablate(self, trainable=False):
        for i in self.masked_layers:
            self.h[i].ablate(trainable=trainable)

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        past_key_values: Optional[Union[Cache, Tuple[torch.FloatTensor]]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        sgtm_mode="default",
    ) -> Union[Tuple[torch.Tensor], BaseModelOutputWithPastAndCrossAttentions]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.wte(input_ids)

        # kept for BC (non `Cache` `past_key_values` inputs)
        return_legacy_cache = False
        if use_cache and not isinstance(past_key_values, Cache):
            return_legacy_cache = True
            if past_key_values is None:
                past_key_values = DynamicCache()
            else:
                past_key_values = DynamicCache.from_legacy_cache(past_key_values)
                logging.warning_once(
                    "We detected that you are passing `past_key_values` as a tuple of tuples. This is deprecated and "
                    "will be removed in v4.47. Please convert your cache or use an appropriate `Cache` class "
                    "(https://huggingface.co/docs/transformers/kv_cache#legacy-cache-format)"
                )

        seq_length = inputs_embeds.shape[1]
        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(past_seen_tokens, past_seen_tokens + seq_length, device=inputs_embeds.device)

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        causal_mask = self._update_causal_mask(
            attention_mask, inputs_embeds, cache_position, past_key_values, output_attentions
        )

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x num_heads x N x N
        # head_mask has shape n_layer x batch x num_heads x N x N
        head_mask = self.get_head_mask(head_mask, self.config.num_layers)
        position_embeds = self.wpe(position_ids)
        hidden_states = inputs_embeds + position_embeds

        if token_type_ids is not None:
            token_type_ids = token_type_ids.view(-1, seq_length)
            token_type_embeds = self.wte(token_type_ids)
            hidden_states = hidden_states + token_type_embeds

        hidden_states = self.drop(hidden_states)
        output_shape = (-1, seq_length, hidden_states.size(-1))

        next_decoder_cache = None
        all_self_attentions = () if output_attentions else None
        all_hidden_states = () if output_hidden_states else None
        for i, block in enumerate(self.h):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            kwargs = {
                "hidden_states": hidden_states,
                "layer_past": past_key_values,
                "attention_mask": causal_mask,
                "head_mask": head_mask[i],
                "use_cache": use_cache,
                "output_attentions": output_attentions,
                "cache_position": cache_position,
            }

            if i in self.masked_layers:
                kwargs["sgtm_mode"] = _validate_mode_parameter(sgtm_mode)

            outputs = block(**kwargs)

            hidden_states = outputs[0]
            if use_cache:
                next_decoder_cache = outputs[1]

            if output_attentions:
                all_self_attentions = all_self_attentions + (outputs[2 if use_cache else 1],)

        hidden_states = self.ln_f(hidden_states)

        hidden_states = hidden_states.view(output_shape)
        # Add last hidden state
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        next_cache = next_decoder_cache if use_cache else None
        if return_legacy_cache:
            next_cache = next_cache.to_legacy_cache()

        if not return_dict:
            return tuple(
                v for v in [hidden_states, next_cache, all_hidden_states, all_self_attentions] if v is not None
            )

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
        )


class GPTNeoForCausalLMSGTM(GPTNeoForCausalLM):
    def __init__(self, config):
        super().__init__(config)
        self.transformer = GPTNeoModelSGTM(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=True)
        # Initialize lm_head bias to zeros
        nn.init.zeros_(self.lm_head.bias)
        self.post_init()

    def adjust_gradients(self, sgtm_mode="default"):
        sgtm_mode = _validate_mode_parameter(sgtm_mode)
        self.transformer.adjust_gradients(sgtm_mode)

    def ablate(self, trainable=False):
        self.transformer.ablate(trainable=trainable)
    
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        past_key_values: Optional[Union[Cache, Tuple[torch.FloatTensor]]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        sgtm_mode="default",
        **kwargs,
    ) -> Union[Tuple[torch.Tensor], CausalLMOutputWithCrossAttentions]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for language modeling. Note that the labels **are shifted** inside the model, i.e. you can set
            `labels = input_ids` Indices are selected in `[-100, 0, ..., config.vocab_size]` All labels set to `-100`
            are ignored (masked), the loss is only computed for labels in `[0, ..., config.vocab_size]`
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        transformer_outputs = self.transformer(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
            sgtm_mode=sgtm_mode,
        )
        hidden_states = transformer_outputs[0]

        lm_logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            # move labels to correct device to enable model parallelism
            labels = labels.to(lm_logits.device)
            # Compute loss in fp32 to match with mesh-tf version
            # https://github.com/EleutherAI/gpt-neo/blob/89ce74164da2fb16179106f54e2269b5da8db333/models/gpt2/gpt2.py#L179
            lm_logits = lm_logits.to(torch.float32)

            # Flatten the tokens
            loss = self.loss_function(
                lm_logits,
                labels,
                vocab_size=self.config.vocab_size,
                **kwargs,
            )

            lm_logits = lm_logits.to(hidden_states.dtype)
            loss = loss.to(hidden_states.dtype)

        if not return_dict:
            output = (lm_logits,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=lm_logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
        )

    def named_parameters_split(self, sgtm_split="default"):
        for param_name, param in self.named_parameters():
            if sgtm_split == "default":
                yield param_name, param

            if sgtm_split == "forget" and "forget" in param_name:
                yield param_name, param

            if sgtm_split == "retain" and "retain" in param_name:
                yield param_name, param

            # TODO: write tests for this
            if sgtm_split == "joint" and "retain" not in param_name and "forget" not in param_name:
                yield param_name, param

    def parameters_split(self, sgtm_split=True):
        for _, param in self.named_parameters_split(sgtm_split=sgtm_split):
            yield param

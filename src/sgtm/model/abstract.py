from abc import ABC, abstractmethod
from transformers.models.gpt_neo.modeling_gpt_neo import GPTNeoMLP, GPTNeoSelfAttention
from sgtm.model.split_linear import SplitLinearOut


class GPTNeoMLPAbstractSGTM(GPTNeoMLP, ABC):
    def __init__(self, intermediate_size, config):
        super().__init__(intermediate_size, config)

        self.retain_mlp_dim = None
        if hasattr(config, "retain_mlp_dim"):
            self.retain_mlp_dim = config.retain_mlp_dim

        self.split_masked_weights = None
        if hasattr(config, "split_masked_weights"):
            self.split_masked_weights = config.split_masked_weights
            if self.split_masked_weights and self.retain_mlp_dim is None:
                raise ValueError("retain_mlp_dim must be set when split_masked_weights is True")

            if self.split_masked_weights and self.retain_mlp_dim > 0:
                self.c_fc = SplitLinearOut(
                    in_features=config.hidden_size,
                    out_features=intermediate_size,
                    retain_dim=self.retain_mlp_dim,
                )

    def forward(self, hidden_states, sgtm_mode="default"):
        if sgtm_mode == "default" or self.retain_mlp_dim is None or self.retain_mlp_dim == 0:
            return super().forward(hidden_states)

        if sgtm_mode == "retain":
            return self.forward_retain(hidden_states)
        elif sgtm_mode == "forget":
            return self.forward_forget(hidden_states)
        else:
            raise ValueError(f"Invalid sgtm mode: {sgtm_mode}")

    def forward_retain(self, hidden_states):
        hidden_states = self.c_fc(hidden_states)
        hidden_states[:, :, self.retain_mlp_dim :] = 0
        hidden_states = self.act(hidden_states)
        hidden_states = self.c_proj(hidden_states)
        hidden_states = self.dropout(hidden_states)

        return hidden_states

    @abstractmethod
    def forward_forget(self, hidden_states):
        pass

    @abstractmethod
    def adjust_gradients(self, sgtm_mode="default"):
        pass

    @abstractmethod
    def ablate(self, trainable=False):
        pass


class GPTNeoSelfAttentionAbstractSGTM(GPTNeoSelfAttention, ABC):
    def __init__(self, config, attention_type, layer_id=None):
        super().__init__(config, attention_type, layer_id)

        self.retain_attn_heads = None
        if hasattr(config, "retain_attn_heads"):
            self.retain_attn_heads = config.retain_attn_heads
            self.retain_dim = int(self.head_dim * self.retain_attn_heads)

        self.split_masked_weights = None
        if hasattr(config, "split_masked_weights"):
            self.split_masked_weights = config.split_masked_weights
            if self.split_masked_weights and self.retain_attn_heads is None:
                raise ValueError("retain_attn_heads must be set when split_masked_weights is True")

            if self.split_masked_weights and self.retain_attn_heads > 0:
                self.k_proj = SplitLinearOut(self.embed_dim, self.embed_dim, retain_dim=self.retain_dim, bias=False)
                self.v_proj = SplitLinearOut(self.embed_dim, self.embed_dim, retain_dim=self.retain_dim, bias=False)
                self.q_proj = SplitLinearOut(self.embed_dim, self.embed_dim, retain_dim=self.retain_dim, bias=False)

    def forward(
        self,
        *args,
        sgtm_mode="default",
        **kwargs,
    ):
        if sgtm_mode == "default" or self.retain_attn_heads is None or self.retain_dim == 0:
            return super().forward(*args, **kwargs)

        if sgtm_mode == "retain":
            return self.forward_retain(*args, **kwargs)
        elif sgtm_mode == "forget":
            return self.forward_forget(*args, **kwargs)
        else:
            raise ValueError(f"Invalid sgtm_modee: {sgtm_mode}")

    def forward_retain(
        self,
        hidden_states,
        attention_mask=None,
        layer_past=None,
        head_mask=None,
        use_cache=False,
        output_attentions=False,
        cache_position=None,
    ):
        query = self.q_proj(hidden_states)
        key = self.k_proj(hidden_states)
        value = self.v_proj(hidden_states)

        query = self._split_heads(query, self.num_heads, self.head_dim)
        key = self._split_heads(key, self.num_heads, self.head_dim)
        value = self._split_heads(value, self.num_heads, self.head_dim)

        if layer_past is not None:
            cache_kwargs = {"cache_position": cache_position}
            key, value = layer_past.update(key, value, self.layer_id, cache_kwargs)

        attn_output, attn_weights = self._attn(query, key, value, attention_mask, head_mask)

        if self.retain_attn_heads is not None and self.retain_attn_heads > 0:
            attn_output[:, self.retain_attn_heads :, :, :] = 0

        attn_output = self._merge_heads(attn_output, self.num_heads, self.head_dim)
        attn_output = self.out_proj(attn_output)
        attn_output = self.resid_dropout(attn_output)

        outputs = (attn_output, layer_past)
        if output_attentions:
            outputs += (attn_weights,)

        return outputs  # a, past_kv, (attentions)

    @abstractmethod
    def forward_forget(
        self,
        hidden_states,
        attention_mask=None,
        layer_past=None,
        head_mask=None,
        use_cache=False,
        output_attentions=False,
        cache_position=None,
    ):
        pass

    @abstractmethod
    def adjust_gradients(self, sgtm_mode="default"):
        pass

    @abstractmethod
    def ablate(self, trainable=False):
        pass

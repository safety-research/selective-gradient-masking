import torch
import torch.nn.init
from sgtm.model.abstract import GPTNeoMLPAbstractSGTM, GPTNeoSelfAttentionAbstractSGTM


class GPTNeoMLPGradientRouting(GPTNeoMLPAbstractSGTM):
    """
    Gradient routing (Cloud et al., 2024) strategy implementation.

    This implementation detaches gradients for specific activations during the forward pass.
    It does not change the forward pass behavior, only modifies which gradients flow.

    Key properties:
    - Does not change the forward pass output
    - Prevents gradient flow for specific dimensions by detaching them
    - Relevant parameters should not be updated
    """

    def __init__(self, intermediate_size, config):
        super().__init__(intermediate_size, config)

    def forward_forget(self, hidden_states):
        intermediate_output = self.c_fc(hidden_states)
        intermediate_output[:, :, :self.retain_mlp_dim] = intermediate_output[:, :, :self.retain_mlp_dim].detach()

        hidden_states = self.act(intermediate_output)
        hidden_states = self.c_proj(hidden_states)
        hidden_states = self.dropout(hidden_states)

        return hidden_states

    def adjust_gradients(self, sgtm_mode="default"):
        pass

    def ablate(self, trainable=False):
        with torch.no_grad():
            if self.split_masked_weights:
                if trainable:
                    # Calculate std of masked weights to maintain same distribution
                    weight_std = self.c_fc.weight_retain.std().item()
                    bias_std = self.c_fc.bias_retain.std().item()
                    torch.nn.init.normal_(self.c_fc.weight_forget, 0, weight_std)
                    torch.nn.init.normal_(self.c_fc.bias_forget, 0, bias_std)
                else:
                    self.c_fc.weight_forget.data = torch.zeros_like(self.c_fc.weight_forget)
                    self.c_fc.bias_forget.data = torch.zeros_like(self.c_fc.bias_forget)
            else:
                if trainable:
                    # Calculate std of masked weights to maintain same distribution
                    weight_std = self.c_fc.weight[:self.retain_mlp_dim].std().item()
                    bias_std = self.c_fc.bias[:self.retain_mlp_dim].std().item()
                    torch.nn.init.normal_(self.c_fc.weight[self.retain_mlp_dim:], 0, weight_std)
                    torch.nn.init.normal_(self.c_fc.bias[self.retain_mlp_dim:], 0, bias_std)
                else:
                    self.c_fc.weight[self.retain_mlp_dim:] = 0
                    self.c_fc.bias[self.retain_mlp_dim:] = 0


class GPTNeoSelfAttentionGradientRouting(GPTNeoSelfAttentionAbstractSGTM):

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
            attn_output[:,:self.retain_attn_heads,:,:] = attn_output[:,:self.retain_attn_heads,:,:].detach()

        attn_output = self._merge_heads(attn_output, self.num_heads, self.head_dim)
        attn_output = self.out_proj(attn_output)
        attn_output = self.resid_dropout(attn_output)

        outputs = (attn_output, layer_past)
        if output_attentions:
            outputs += (attn_weights,)

        return outputs  # a, past_kv, (attentions)

    def adjust_gradients(self, sgtm_mode="default"):
        pass

    def ablate(self, trainable=False):
        with torch.no_grad():
            if self.split_masked_weights:
                if trainable:
                    # Calculate std of masked weights to maintain same distribution
                    q_std = self.q_proj.weight_retain.std().item()
                    k_std = self.k_proj.weight_retain.std().item()
                    v_std = self.v_proj.weight_retain.std().item()
                    torch.nn.init.normal_(self.q_proj.weight_forget, 0, q_std)
                    torch.nn.init.normal_(self.k_proj.weight_forget, 0, k_std)
                    torch.nn.init.normal_(self.v_proj.weight_forget, 0, v_std)
                else:
                    self.q_proj.weight_forget.data = torch.zeros_like(self.q_proj.weight_forget)
                    self.k_proj.weight_forget.data = torch.zeros_like(self.k_proj.weight_forget)
                    self.v_proj.weight_forget.data = torch.zeros_like(self.v_proj.weight_forget)
            else:
                if trainable:
                    # Calculate std of masked weights to maintain same distribution
                    q_std = self.q_proj.weight[:self.retain_dim].std().item()
                    k_std = self.k_proj.weight[:self.retain_dim].std().item()
                    v_std = self.v_proj.weight[:self.retain_dim].std().item()
                    torch.nn.init.normal_(self.q_proj.weight[self.retain_dim:], 0, q_std)
                    torch.nn.init.normal_(self.k_proj.weight[self.retain_dim:], 0, k_std)
                    torch.nn.init.normal_(self.v_proj.weight[self.retain_dim:], 0, v_std)
                else:
                    self.q_proj.weight[self.retain_dim:] = 0
                    self.k_proj.weight[self.retain_dim:] = 0
                    self.v_proj.weight[self.retain_dim:] = 0

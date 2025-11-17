import torch
import torch.nn.init
from sgtm.model.abstract import GPTNeoMLPAbstractSGTM, GPTNeoSelfAttentionAbstractSGTM


class GPTNeoMLPActivationMasking(GPTNeoMLPAbstractSGTM):
    """
    Activation masking strategy.
    
    This implementation zeros out a portion of the activations during the forward pass,
    which in turn affects which gradients flow during backpropagation.
    
    Key properties:
    - Changes the output of a forward pass
    - Changes what gets backpropagated
    - Relevant parameters should not be updated
    """
    
    def forward_forget(self, hidden_states):
        hidden_states = self.c_fc(hidden_states)
        hidden_states[:, :, :self.retain_mlp_dim] = 0
        hidden_states = self.act(hidden_states)
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
                    proj_std = self.c_proj.weight[:, :self.retain_mlp_dim].std().item()
                    torch.nn.init.normal_(self.c_fc.weight_forget, 0, weight_std)
                    torch.nn.init.normal_(self.c_fc.bias_forget, 0, bias_std)
                    torch.nn.init.normal_(self.c_proj.weight[:, self.retain_mlp_dim:], 0, proj_std)
                else:
                    self.c_fc.weight_forget.data = torch.zeros_like(self.c_fc.weight_forget)
                    self.c_fc.bias_forget.data = torch.zeros_like(self.c_fc.bias_forget)
                    self.c_proj.weight.data[:, self.retain_mlp_dim:] = 0
            else:
                if trainable:
                    # Calculate std of masked weights to maintain same distribution
                    weight_std = self.c_fc.weight[:self.retain_mlp_dim].std().item()
                    bias_std = self.c_fc.bias[:self.retain_mlp_dim].std().item()
                    proj_std = self.c_proj.weight[:, :self.retain_mlp_dim].std().item()
                    torch.nn.init.normal_(self.c_fc.weight[self.retain_mlp_dim:], 0, weight_std)
                    torch.nn.init.normal_(self.c_fc.bias[self.retain_mlp_dim:], 0, bias_std)
                    torch.nn.init.normal_(self.c_proj.weight[:, self.retain_mlp_dim:], 0, proj_std)
                else:
                    self.c_fc.weight.data[self.retain_mlp_dim:] = 0
                    self.c_fc.bias.data[self.retain_mlp_dim:] = 0
                    self.c_proj.weight.data[:, self.retain_mlp_dim:] = 0


class GPTNeoSelfAttentionActivationMasking(GPTNeoSelfAttentionAbstractSGTM):

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
            attn_output[:,:self.retain_attn_heads,:,:] = 0

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
        retain_dim = self.head_dim * self.retain_attn_heads

        with torch.no_grad():
            if self.split_masked_weights:
                if trainable:
                    # Calculate std of masked weights to maintain same distribution
                    q_std = self.q_proj.weight_retain.std().item()
                    k_std = self.k_proj.weight_retain.std().item()
                    v_std = self.v_proj.weight_retain.std().item()
                    out_std = self.out_proj.weight[:, :retain_dim].std().item()
                    torch.nn.init.normal_(self.q_proj.weight_forget, 0, q_std)
                    torch.nn.init.normal_(self.k_proj.weight_forget, 0, k_std)
                    torch.nn.init.normal_(self.v_proj.weight_forget, 0, v_std)
                    torch.nn.init.normal_(self.out_proj.weight[:, retain_dim:], 0, out_std)
                else:
                    self.q_proj.weight_forget.data = torch.zeros_like(self.q_proj.weight_forget)
                    self.k_proj.weight_forget.data = torch.zeros_like(self.k_proj.weight_forget)
                    self.v_proj.weight_forget.data = torch.zeros_like(self.v_proj.weight_forget)
                    self.out_proj.weight[:, retain_dim:] = 0
            else:
                if trainable:
                    # Calculate std of masked weights to maintain same distribution
                    q_std = self.q_proj.weight[:retain_dim].std().item()
                    k_std = self.k_proj.weight[:retain_dim].std().item()
                    v_std = self.v_proj.weight[:retain_dim].std().item()
                    out_std = self.out_proj.weight[:, :retain_dim].std().item()
                    torch.nn.init.normal_(self.q_proj.weight[retain_dim:], 0, q_std)
                    torch.nn.init.normal_(self.k_proj.weight[retain_dim:], 0, k_std)
                    torch.nn.init.normal_(self.v_proj.weight[retain_dim:], 0, v_std)
                    torch.nn.init.normal_(self.out_proj.weight[:, retain_dim:], 0, out_std)
                else:
                    self.q_proj.weight[retain_dim:] = 0
                    self.k_proj.weight[retain_dim:] = 0
                    self.v_proj.weight[retain_dim:] = 0
                    self.out_proj.weight[:, retain_dim:] = 0

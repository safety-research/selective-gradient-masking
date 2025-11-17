import torch
import torch.nn.init
from sgtm.model.abstract import GPTNeoMLPAbstractSGTM, GPTNeoSelfAttentionAbstractSGTM


class GPTNeoMLPParameterMasking(GPTNeoMLPAbstractSGTM):
    """
    Parameter masking strategy.

    This implementation doesn't modify forward or backward passes,
    but instead masks gradients of specific parameters before the optimizer step.

    Key properties:
    - Does not affect the forward pass
    - Does not affect the backward pass
    - Prevents selected parameters from being updated
    """

    def forward_forget(self, hidden_states):
        return super(GPTNeoMLPAbstractSGTM, self).forward(hidden_states)

    def adjust_gradients(self, sgtm_mode="default"):
        if sgtm_mode != "forget" or self.retain_mlp_dim is None or self.retain_mlp_dim <= 0:
            return

        if self.split_masked_weights:
            self.c_fc.weight_retain.grad = torch.zeros_like(self.c_fc.weight_retain.grad)
            self.c_fc.bias_retain.grad = torch.zeros_like(self.c_fc.bias_retain.grad)
            self.c_proj.weight.grad[:, : self.retain_mlp_dim] = 0
            self.c_proj.bias.grad = torch.zeros_like(self.c_proj.bias.grad)
        else:
            self.c_fc.weight.grad[: self.retain_mlp_dim, :] = 0
            self.c_fc.bias.grad[: self.retain_mlp_dim] = 0
            self.c_proj.weight.grad[:, : self.retain_mlp_dim] = 0
            self.c_proj.bias.grad = torch.zeros_like(self.c_proj.bias.grad)

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
                    self.c_proj.weight[:, self.retain_mlp_dim :] = 0
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
                    self.c_fc.weight[self.retain_mlp_dim :] = 0
                    self.c_fc.bias[self.retain_mlp_dim :] = 0
                    self.c_proj.weight[:, self.retain_mlp_dim :] = 0

class GPTNeoMLPParameterMaskingNoProj(GPTNeoMLPAbstractSGTM):
    """
    Parameter masking strategy.

    This implementation doesn't modify forward or backward passes,
    but instead masks gradients of specific parameters before the optimizer step.

    Key properties:
    - Does not affect the forward pass
    - Does not affect the backward pass
    - Prevents selected parameters from being updated
    """

    def forward_forget(self, hidden_states):
        return super(GPTNeoMLPAbstractSGTM, self).forward(hidden_states)

    def adjust_gradients(self, sgtm_mode="default"):
        if sgtm_mode != "forget" or self.retain_mlp_dim is None or self.retain_mlp_dim <= 0:
            return

        if self.split_masked_weights:
            self.c_fc.weight_retain.grad = torch.zeros_like(self.c_fc.weight_retain.grad)
            self.c_fc.bias_retain.grad = torch.zeros_like(self.c_fc.bias_retain.grad)
        else:
            self.c_fc.weight.grad[: self.retain_mlp_dim, :] = 0
            self.c_fc.bias.grad[: self.retain_mlp_dim] = 0

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
                    self.c_fc.weight[self.retain_mlp_dim :] = 0
                    self.c_fc.bias[self.retain_mlp_dim :] = 0


class GPTNeoSelfAttentionParameterMasking(GPTNeoSelfAttentionAbstractSGTM):

    def forward_forget(self, *args, **kwargs):
        return super(GPTNeoSelfAttentionAbstractSGTM, self).forward(*args, **kwargs)

    def adjust_gradients(self, sgtm_mode="default"):
        if sgtm_mode != "forget" or self.retain_attn_heads is None or self.retain_attn_heads <= 0:
            return
        
        if self.split_masked_weights:
            self.q_proj.weight_retain.grad = torch.zeros_like(self.q_proj.weight_retain.grad)
            self.k_proj.weight_retain.grad = torch.zeros_like(self.k_proj.weight_retain.grad)
            self.v_proj.weight_retain.grad = torch.zeros_like(self.v_proj.weight_retain.grad)
            self.out_proj.weight.grad[:, :self.retain_dim] = 0
            self.out_proj.bias.grad = torch.zeros_like(self.out_proj.bias.grad)
        else:
            self.q_proj.weight.grad[:self.retain_dim, :] = 0
            self.k_proj.weight.grad[:self.retain_dim, :] = 0
            self.v_proj.weight.grad[:self.retain_dim, :] = 0
            self.out_proj.weight.grad[:, :self.retain_dim] = 0
            self.out_proj.bias.grad = torch.zeros_like(self.out_proj.bias.grad)

    def ablate(self, trainable=False):
        with torch.no_grad():
            if self.split_masked_weights:
                if trainable:
                    # Calculate std of masked weights to maintain same distribution
                    q_std = self.q_proj.weight_retain.std().item()
                    k_std = self.k_proj.weight_retain.std().item()
                    v_std = self.v_proj.weight_retain.std().item()
                    out_std = self.out_proj.weight[:, :self.retain_dim].std().item()
                    torch.nn.init.normal_(self.q_proj.weight_forget, 0, q_std)
                    torch.nn.init.normal_(self.k_proj.weight_forget, 0, k_std)
                    torch.nn.init.normal_(self.v_proj.weight_forget, 0, v_std)
                    torch.nn.init.normal_(self.out_proj.weight[:, self.retain_dim:], 0, out_std)
                else:
                    self.q_proj.weight_forget.data = torch.zeros_like(self.q_proj.weight_forget)
                    self.k_proj.weight_forget.data = torch.zeros_like(self.k_proj.weight_forget)
                    self.v_proj.weight_forget.data = torch.zeros_like(self.v_proj.weight_forget)
                    self.out_proj.weight[:, self.retain_dim:] = 0
            else:
                if trainable:
                    # Calculate std of masked weights to maintain same distribution
                    q_std = self.q_proj.weight[:self.retain_dim].std().item()
                    k_std = self.k_proj.weight[:self.retain_dim].std().item()
                    v_std = self.v_proj.weight[:self.retain_dim].std().item()
                    out_std = self.out_proj.weight[:, :self.retain_dim].std().item()
                    torch.nn.init.normal_(self.q_proj.weight[self.retain_dim:], 0, q_std)
                    torch.nn.init.normal_(self.k_proj.weight[self.retain_dim:], 0, k_std)
                    torch.nn.init.normal_(self.v_proj.weight[self.retain_dim:], 0, v_std)
                    torch.nn.init.normal_(self.out_proj.weight[:, self.retain_dim:], 0, out_std)
                else:
                    self.q_proj.weight[self.retain_dim:] = 0
                    self.k_proj.weight[self.retain_dim:] = 0
                    self.v_proj.weight[self.retain_dim:] = 0
                    self.out_proj.weight[:, self.retain_dim:] = 0


class GPTNeoSelfAttentionParameterMaskingNoProj(GPTNeoSelfAttentionAbstractSGTM):
    def forward_forget(self, *args, **kwargs):
        return super(GPTNeoSelfAttentionAbstractSGTM, self).forward(*args, **kwargs)

    def adjust_gradients(self, sgtm_mode="default"):
        if sgtm_mode != "forget" or self.retain_attn_heads is None or self.retain_attn_heads <= 0:
            return

        if self.split_masked_weights:
            self.q_proj.weight_retain.grad = torch.zeros_like(self.q_proj.weight_retain.grad)
            self.k_proj.weight_retain.grad = torch.zeros_like(self.k_proj.weight_retain.grad)
            self.v_proj.weight_retain.grad = torch.zeros_like(self.v_proj.weight_retain.grad)
        else:
            self.q_proj.weight.grad[:self.retain_dim, :] = 0
            self.k_proj.weight.grad[:self.retain_dim, :] = 0
            self.v_proj.weight.grad[:self.retain_dim, :] = 0

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

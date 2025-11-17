import torch
import torch.nn as nn
from torch.nn import functional as F, init
import math
from torch.nn.parameter import Parameter


class SplitLinearOut(nn.Linear):

    def __init__(
        self,
        in_features: int,
        out_features: int,
        retain_dim: int,
        bias: bool = True,
        device=None,
        dtype=None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super(nn.Linear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.retain_dim = retain_dim
        
        self.weight_retain = Parameter(
            torch.empty((retain_dim, in_features), **factory_kwargs)
        )
        self.weight_forget = Parameter(
            torch.empty((out_features - retain_dim, in_features), **factory_kwargs)
        )

        if bias:
            self.bias_retain = Parameter(
                torch.empty((retain_dim), **factory_kwargs)
            )
            self.bias_forget = Parameter(
                torch.empty((out_features - retain_dim), **factory_kwargs)
            )
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()
    
    @property
    def weight(self):
        return torch.cat([self.weight_retain, self.weight_forget], dim=0)
    
    @property
    def bias(self):
        return torch.cat([self.bias_retain, self.bias_forget], dim=0)
            
    
    def reset_parameters(self) -> None:
        # Setting a=sqrt(5) in kaiming_uniform is the same as initializing with
        # uniform(-1/sqrt(in_features), 1/sqrt(in_features)). For details, see
        # https://github.com/pytorch/pytorch/issues/57109
        weight_placeholder = torch.empty_like(self.weight)
        init.kaiming_uniform_(weight_placeholder, a=math.sqrt(5))

        with torch.no_grad():
            self.weight_retain.copy_(weight_placeholder[:self.retain_dim])
            self.weight_forget.copy_(weight_placeholder[self.retain_dim:])

        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            bias_placeholder = torch.empty_like(self.bias)
            init.uniform_(bias_placeholder, -bound, bound)

            with torch.no_grad():
                self.bias_retain.copy_(bias_placeholder[:self.retain_dim])
                self.bias_forget.copy_(bias_placeholder[self.retain_dim:])

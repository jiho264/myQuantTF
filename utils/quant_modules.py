import torch, math
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.vision_transformer import Encoder, EncoderBlock, MLPBlock
from collections import OrderedDict
from torch import Tensor
from typing import Optional, Tuple


class QuantWeight(nn.Module):
    def __init__(self, orgModule: nn.Module, args_w):
        super().__init__()

        self.do_quant = False

        self.weight_fp = orgModule.weight.clone().detach()
        self.bias_fp = (
            orgModule.bias.clone().detach() if orgModule.bias is not None else None
        )

    def forward(self, x: Tensor) -> Tensor:
        return x


class QuantAct(nn.Module):
    def __init__(self, args_a):
        super().__init__()
        self.do_quant = False

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        return x


class QuantGELU(nn.Module):
    def __init__(self, args_gelu):
        super().__init__()
        self.do_quant = False

    def forward(self, input: torch.Tensor):
        return F.gelu(input)


class QuantSoftMax(nn.Module):
    def __init__(self, args_softmax):
        super().__init__()
        self.do_quant = False

    def forward(self, input: torch.Tensor, dim: int = -1):
        return F.softmax(input, dim=dim)


class QuantLayerNorm(nn.Module):
    def __init__(self, orgModule, args_ln):
        super().__init__()
        self.do_quant = False
        self.weight = orgModule.weight.clone().detach()
        self.bias = orgModule.bias.clone().detach()
        self.orgModule = orgModule

    def forward(self, input: torch.Tensor):
        return self.orgModule(input)


class QuantLinear(nn.Module):
    def __init__(self, orgModule, args_w, args_a):
        super().__init__()
        """forward function setting"""
        if isinstance(orgModule, nn.Conv2d):
            self.fwd_kwargs = dict(
                stride=orgModule.stride,
                padding=orgModule.padding,
                dilation=orgModule.dilation,
                groups=orgModule.groups,
            )
            self.fwd_func = F.conv2d
        else:
            self.fwd_kwargs = dict()
            self.fwd_func = F.linear

        self.weight = orgModule.weight.clone().detach()
        self.bias = (
            orgModule.bias.clone().detach() if orgModule.bias is not None else None
        )

        self.weightQuantizer = QuantWeight(orgModule=orgModule, args_w=args_w)
        self.activationQuantizer = QuantAct(args_a=args_a)

    def forward(self, input: torch.Tensor):
        _weight = self.weightQuantizer(self.weight)

        x = self.fwd_func(input, _weight, self.bias, **self.fwd_kwargs)

        return self.activationQuantizer(x)

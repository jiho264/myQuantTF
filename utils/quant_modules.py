import torch, math
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
from torch import Tensor
from typing import Optional, Tuple
from .quant_utils import AbsMaxQuantizer, MovAvgAbsMaxQuantizer


class QuantAct(nn.Module):
    def __init__(self, args_a):
        super().__init__()
        self.do_quant = False
        if args_a == {}:  # No quantization
            return
        if args_a is not None:
            self.activationQuantizer = MovAvgAbsMaxQuantizer(None, args_a)

    def forward(self, x, s_x=1):
        if self.do_quant:
            x = self.activationQuantizer(x)
            return x, s_x
        return x, s_x


class QuantLinearWithWeight(nn.Module):
    def __init__(self, orgModule, args_w):
        super().__init__()
        self.do_quant = False

        """forward function setting"""
        if isinstance(orgModule, nn.Conv2d):
            self.fwd_kwargs = dict(
                stride=orgModule.stride,
                padding=orgModule.padding,
                dilation=orgModule.dilation,
                groups=orgModule.groups,
            )
            self.fwd_func = F.conv2d
        elif isinstance(orgModule, nn.Linear):
            self.fwd_kwargs = dict()
            self.fwd_func = F.linear

        self.WEIGHT_FP = orgModule.weight.clone().detach()
        self.BIAS_FP = orgModule.bias.clone().detach()

        if args_w.get("scheme") in ["AbsMaxQuantizer"]:
            self.weightQuantizer = AbsMaxQuantizer(
                org_tensor=self.WEIGHT_FP, args=args_w
            )
            self.WEIGHT_INT = self.weightQuantizer(self.WEIGHT_FP)
            del self.weightQuantizer
        else:
            raise NotImplementedError

    def forward(self, input, s_in=1):
        if self.do_quant:
            _weight = self.WEIGHT_INT
            # _bias = self.BIAS_INT
            _bias = self.BIAS_FP
        else:
            _weight = self.WEIGHT_FP
            _bias = self.BIAS_FP

        x = self.fwd_func(input, _weight, _bias, **self.fwd_kwargs)
        return x, s_in


class IntGELU(nn.Module):
    def __init__(self, args_gelu):
        super().__init__()
        self.do_quant = False
        # if args_gelu.get("scheme") in ["I-ViT"]:

        #     pass
        # else:
        #     raise NotImplementedError

    def forward(self, input, s_in=1):
        return F.gelu(input), s_in


class IntSoftMax(nn.Module):
    def __init__(self, args_softmax):
        super().__init__()
        self.do_quant = False
        # if args_softmax.get("scheme") in ["I-ViT"]:
        #     self.N = 15
        #     self.M = 1
        #     pass
        # else:
        #     raise NotImplementedError

    def forward(self, input, s_in=1, dim: int = -1):
        return F.softmax(input, dim=dim), s_in


class IntLayerNorm(nn.Module):
    def __init__(self, orgModule, args_ln):
        super().__init__()
        self.do_quant = False
        # if args_ln.get("scheme") in ["I-ViT"]:
        #     pass
        # else:
        #     raise NotImplementedError
        self.weight = orgModule.weight.clone().detach()
        self.bias = orgModule.bias.clone().detach()
        self.orgModule = orgModule

    def forward(self, input, s_in=1):
        return self.orgModule(input), s_in

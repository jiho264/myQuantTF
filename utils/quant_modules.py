import torch, math
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
from torch import Tensor
from typing import Optional, Tuple
from .quant_utils import AbsMaxQuantizer, MovAvgAbsMaxQuantizer


class QuantWeight(nn.Module):
    def __init__(self, orgModule: nn.Module, args_w):
        super().__init__()
        self.do_quant = False

        self.WEIGHT_FP = orgModule.weight.clone().detach()

        if args_w.get("scheme") in ["AbsMaxQuantizer"]:
            self.weightQuantizer = AbsMaxQuantizer(
                org_tensor=self.WEIGHT_FP, args=args_w
            )
        else:
            raise NotImplementedError

        """현재는 FP도메인의 INT반환"""
        self.WEIGHT_INT = self.weightQuantizer(self.WEIGHT_FP)

    def get_weight(self):
        if self.do_quant:
            return self.WEIGHT_INT
        return self.WEIGHT_FP

    def forward(self, x: Tensor) -> Tensor:
        raise NotImplementedError


class QuantAct(nn.Module):
    def __init__(self, args_a):
        super().__init__()
        self.do_quant = False
        if args_a.get("scheme") in ["MovAvgAbsMaxQuantizer"]:
            self.actQuantizer = MovAvgAbsMaxQuantizer(org_tensor=None, args=args_a)
        else:
            raise NotImplementedError

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        if self.do_quant:
            return self.actQuantizer(x)
        return x


class QuantConv2d(nn.Module):
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

        self.weightQuantizer = QuantWeight(orgModule=orgModule, args_w=args_w)
        self.BIAS_FP = (
            orgModule.bias.clone().detach() if orgModule.bias is not None else None
        )
        self.activationQuantizer = QuantAct(args_a=args_a)

    def forward(self, input: torch.Tensor):
        _weight = self.weightQuantizer.get_weight()

        x = self.fwd_func(input, _weight, self.BIAS_FP, **self.fwd_kwargs)

        return self.activationQuantizer(x)


class QuantLinear(nn.Module):
    def __init__(self, orgModule, args_w, args_a):
        super().__init__()
        """forward function setting"""
        if isinstance(orgModule, nn.Linear):
            self.fwd_kwargs = dict()
            self.fwd_func = F.linear

        self.weightQuantizer = QuantWeight(orgModule=orgModule, args_w=args_w)
        self.BIAS_FP = (
            orgModule.bias.clone().detach() if orgModule.bias is not None else None
        )
        self.activationQuantizer = QuantAct(args_a=args_a)

    def forward(self, input: torch.Tensor):
        _weight = self.weightQuantizer.get_weight()

        x = self.fwd_func(input, _weight, self.BIAS_FP, **self.fwd_kwargs)

        return self.activationQuantizer(x)


class IntGELU(nn.Module):
    def __init__(self, args_gelu):
        super().__init__()
        self.do_quant = False
        # if args_gelu.get("scheme") in ["I-ViT"]:

        #     pass
        # else:
        #     raise NotImplementedError

    def forward(self, input: torch.Tensor):
        return F.gelu(input)


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

    def forward(self, input: torch.Tensor, dim: int = -1):
        return F.softmax(input, dim=dim)


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

    def forward(self, input: torch.Tensor):
        return self.orgModule(input)

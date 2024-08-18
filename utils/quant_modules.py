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
        if args_gelu.get("scheme") in ["I-ViT"]:

            pass
        else:
            raise NotImplementedError

    def forward(self, input: torch.Tensor):
        return F.gelu(input)


class IntSoftMax(nn.Module):
    def __init__(self, args_softmax):
        super().__init__()
        self.do_quant = False
        if args_softmax.get("scheme") in ["I-ViT"]:
            self.N = 15
            self.M = 1
            pass
        else:
            raise NotImplementedError

    def ShiftExp(self, I, S):
        I_p = I + (I / 2**1).round() - (I / 2**4).round()  # I * log2(e) = I * 1.442695
        I_0 = (1 / S).round()
        q = (I_p / (-I_0)).floor()  # Integer part
        r = -(I_p - q * (-I_0))  # Decimal part
        I_b = (-r / 2**1).round() + I_0  # Ep. 15
        I_exp = I_b * (self.N - q)  # Ep. 14
        S_exp = S / 2**self.N
        return I_exp, S_exp  # S_exp * I_exp = e**(S*I)

    def Shiftmax(self, I_in, S_in, k_out):
        I_delta = I_in - I_in.max()  # Ep. 12
        (I_exp, S_exp) = self.ShiftExp(I_delta, S_in)

        # Ep. 16
        I_out = (2**self.M / I_exp.sum(dim=-1).floor() * I_exp).round() / (
            2 ** (self.M - (k_out - 1))
        )
        S_out = 1 / (2 ** (k_out - 1))
        return I_out, S_out

    def forward(self, input: torch.Tensor, dim: int = -1):
        return F.softmax(input, dim=dim)


class IntLayerNorm(nn.Module):
    def __init__(self, orgModule, args_ln):
        super().__init__()
        self.do_quant = False
        if args_ln.get("scheme") in ["I-ViT"]:
            pass
        else:
            raise NotImplementedError
        self.weight = orgModule.weight.clone().detach()
        self.bias = orgModule.bias.clone().detach()
        self.orgModule = orgModule

    def forward(self, input: torch.Tensor):
        return self.orgModule(input)

import torch, math
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
from torch import Tensor
from typing import Optional, Tuple

# from .quant_utils import AbsMaxQuantizer, MovAvgAbsMaxQuantizer
from torch.autograd import Function


class QuantMatMul(nn.Module):
    """
    Class to quantize weights of given matmul layer
    """

    def __init__(self):
        super(QuantMatMul, self).__init__()

    def forward(self, A, pre_act_scaling_factor_A, B, pre_act_scaling_factor_B):
        A_int = A / pre_act_scaling_factor_A
        B_int = B / pre_act_scaling_factor_B
        act_scaling_factor = pre_act_scaling_factor_A * pre_act_scaling_factor_B
        return (A_int @ B_int) * act_scaling_factor, act_scaling_factor


class QuantAct(nn.Module):
    def __init__(self, args_a={}, which=None):
        super(QuantAct, self).__init__()
        assert (
            args_a.get("per_channel", True) == False
        ), "only per-tensor quantization is supported for Activation Quantization"

        if which == "cls_token":
            self.bit_width = 16
            print(f"Int Activation {self.bit_width} for {which}")
        elif which == "idAdd":
            self.bit_width = 16
            print(f"Int Activation {self.bit_width} for {which}")
        else:
            self.bit_width = args_a.get("bit_width", 8)
            print(f"Int Activation {self.bit_width}")

    def forward(self, x_hat, s_x=None, id_hat=None, s_id=None):
        with torch.no_grad():
            x_out = x_hat if id_hat == None else id_hat + x_hat

            s_out = x_out.abs().max() / (2 ** (self.bit_width - 1) - 1)

        if s_x == None:
            # input quantization
            out_int = (
                (x_hat / s_out)
                .round()
                .clamp(
                    -(2 ** (self.bit_width - 1)),
                    2 ** (self.bit_width - 1) - 1,
                )
            )
        else:
            x_int = (x_hat / s_x).round()
            new_scaler = s_x / s_out
            out_int = (
                (x_int * new_scaler)
                .round()
                .clamp(
                    -(2 ** (self.bit_width - 1)),
                    2 ** (self.bit_width - 1) - 1,
                )
            )

            if id_hat is not None:
                id_int = (id_hat / s_id).round()
                new_scaler = s_id / s_out
                id_int = (
                    (id_int * new_scaler)
                    .round()
                    .clamp(
                        -(2 ** (self.bit_width - 1)),
                        2 ** (self.bit_width - 1) - 1,
                    )
                )

                out_int += id_int

        out_hat = out_int * s_out.view(-1)
        return out_hat, s_out


class QuantLinearWithWeight(nn.Module):
    def __init__(self, orgModule, args_w):
        super().__init__()
        self.do_quant = False
        self.bit_width = args_w.get("bit_width", 8)

        assert (
            args_w.get("per_channel", False) == True
        ), "only per-channel quantization is supported for Weight Quantization"

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

        weight_fp32 = orgModule.weight.clone().detach()
        bias_fp32 = orgModule.bias.clone().detach()

        s_w = weight_fp32.view(weight_fp32.size(0), -1).abs().max(dim=1).values / (
            2 ** (self.bit_width - 1) - 1
        )

        _n_ch = len(weight_fp32.size())
        self.s_w = s_w.view(-1, *([1] * (_n_ch - 1)))

        self.w_int8 = (
            (weight_fp32.clone().detach() / self.s_w)
            .round()
            .clamp(-(2 ** (self.bit_width - 1)), 2 ** (self.bit_width - 1) - 1)
        )
        self.b_int8 = (bias_fp32.clone().detach() / self.s_w.squeeze()).round()

    def forward(self, x_hat, s_x):
        # # 여기 x의 bit repr는 이전 layer에서 몇으로 줄였느냐에 따라 다름

        s_a = self.s_w * s_x
        b_int32 = (self.b_int8 / s_x).round()

        if self.fwd_func == F.conv2d:
            s_x = s_x.view(1, -1, 1, 1)
            s_a = s_a.view(1, -1, 1, 1)

        elif self.fwd_func == F.linear:
            s_a = s_a.view(1, -1)

        x_int = (x_hat / s_x).round()

        a_int32 = self.fwd_func(x_int, self.w_int8, b_int32, **self.fwd_kwargs)
        a_hat = a_int32 * s_a

        return a_hat, s_a


class floor_ste(Function):
    """
    Straight-through Estimator(STE) for torch.floor()
    """

    @staticmethod
    def forward(ctx, x):
        return torch.floor(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.clone()


class round_ste(Function):
    """
    Straight-through Estimator(STE) for torch.round()
    """

    @staticmethod
    def forward(ctx, x):
        return torch.round(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.clone()


class IntGELU(nn.Module):
    def __init__(self, args_gelu):
        super().__init__()
        self.do_quant = False

        self.bit_width = args_gelu.get("bit_width", 8)

        self.n = 17  # sufficiently large integer
        # The minimum value for ensuring accuracy (varies depending on models)

        print(f"IntGELU bit: {self.bit_width}")

    def int_exp_shift(self, x_int, scaling_factor):
        x_int = x_int + floor_ste.apply(x_int / 2) - floor_ste.apply(x_int / 2**4)

        with torch.no_grad():
            x0_int = torch.floor(-1.0 / scaling_factor)
        x_int = torch.max(x_int, self.n * x0_int)

        q = floor_ste.apply(x_int / x0_int)
        r = x_int - x0_int * q
        exp_int = r / 2 - x0_int
        exp_int = torch.clamp(floor_ste.apply(exp_int * 2 ** (self.n - q)), min=0)
        scaling_factor = scaling_factor / 2**self.n

        return exp_int, scaling_factor

    def int_forward(self, x, scaling_factor=None):
        pre_x_int = x / scaling_factor
        scaling_factor_sig = scaling_factor * 1.702

        x_int_max, _ = pre_x_int.max(dim=-1, keepdim=True)
        x_int = pre_x_int - x_int_max

        exp_int, _ = self.int_exp_shift(x_int, scaling_factor_sig)  # e^(x-x_max)

        exp_int_max, _ = self.int_exp_shift(
            -x_int_max, scaling_factor_sig
        )  # e^(-x_max)
        exp_int_sum = exp_int + exp_int_max

        exp_int_sum.clamp_max_(2**31 - 1)
        factor = floor_ste.apply((2**31 - 1) / exp_int_sum)
        sigmoid_int = floor_ste.apply(exp_int * factor / 2 ** (31 - self.bit_width + 1))
        sigmoid_scaling_factor = torch.Tensor([1 / 2 ** (self.bit_width - 1)]).cuda()

        x_int = pre_x_int * sigmoid_int
        scaling_factor = scaling_factor * sigmoid_scaling_factor
        return x_int * scaling_factor, scaling_factor

    def forward(self, input, s_pre):
        if self.do_quant:
            return self.int_forward(input, s_pre)
        else:
            print("FP GELU")
            return F.gelu(input), s_pre


class IntSoftMax(nn.Module):
    def __init__(self, args_softmax):
        super().__init__()
        self.do_quant = False

        self.bit_width = args_softmax.get("bit_width", 8)

        self.n = 15  # sufficiently large integer
        # The minimum value for ensuring accuracy (varies depending on models)

        print(f"IntSoftMax {self.bit_width}")

    def int_exp_shift(self, x_int, scaling_factor):
        x_int = x_int + floor_ste.apply(x_int / 2) - floor_ste.apply(x_int / 2**4)

        with torch.no_grad():
            x0_int = torch.floor(-1.0 / scaling_factor)
        x_int = torch.max(x_int, self.n * x0_int)

        q = floor_ste.apply(x_int / x0_int)
        r = x_int - x0_int * q
        exp_int = r / 2 - x0_int
        exp_int = torch.clamp(floor_ste.apply(exp_int * 2 ** (self.n - q)), min=0)
        scaling_factor = scaling_factor / 2**self.n
        return exp_int, scaling_factor

    def int_forward(self, x, scaling_factor):
        x_int = x / scaling_factor
        x_int_max, _ = x_int.max(dim=-1, keepdim=True)
        x_int = x_int - x_int_max

        exp_int, _ = self.int_exp_shift(x_int, scaling_factor)
        exp_int_sum = exp_int.sum(dim=-1, keepdim=True)

        exp_int_sum.clamp_max_(2**31 - 1)
        factor = floor_ste.apply((2**31 - 1) / exp_int_sum)
        exp_int = floor_ste.apply(exp_int * factor / 2 ** (31 - self.bit_width + 1))
        scaling_factor = torch.Tensor([1 / 2 ** (self.bit_width - 1)]).cuda()

        return exp_int * scaling_factor, scaling_factor

    def forward(self, input, s_pre, dim: int = -1):
        if self.do_quant:
            return self.int_forward(input, s_pre)
        else:
            print("FP Softmax")
            """ 
            The result of softmax's range is [0, 1], 
            So we can return 1/127 for INT8 quantization.
            """
            return F.softmax(input, dim=dim), 1 / 127


class IntLayerNorm(nn.Module):
    def __init__(self, orgModule, args_ln):
        super().__init__()
        self.do_quant = False

        self.weight = orgModule.weight.clone().detach()
        self.bias = orgModule.bias.clone().detach()
        self.orgModule = orgModule

        self.dim_sqrt = None

    def int_forward(self, x, scaling_factor=None):
        if self.dim_sqrt is None:
            n = torch.tensor(x.shape[2], dtype=torch.float)
            self.dim_sqrt = torch.sqrt(n).cuda()

        # Normalization: computes mean and variance(std)
        x_int = x / scaling_factor
        mean_int = round_ste.apply(x_int.mean(axis=2, keepdim=True))
        y_int = x_int - mean_int
        y_sq_int = y_int**2
        var_int = torch.sum(y_sq_int, axis=2, keepdim=True)

        # Integer Iteration
        k = 2**16
        for _ in range(10):
            k_1 = floor_ste.apply((k + floor_ste.apply(var_int / k)) / 2)
            k = k_1
        std_int = k

        factor = floor_ste.apply((2**31 - 1) / std_int)
        y_int = floor_ste.apply(y_int * factor / 2)
        scaling_factor = self.dim_sqrt / 2**30

        # scaling and shifting
        bias = self.bias.data.detach() / (self.weight.data.detach())
        bias_int = floor_ste.apply(bias / scaling_factor)

        y_int = y_int + bias_int
        scaling_factor = scaling_factor * self.weight
        x = y_int * scaling_factor
        return x, scaling_factor

    def forward(self, input, s_pre):
        if self.do_quant:
            return self.int_forward(input, s_pre)
        else:
            print("FP LayerNorm")
            return self.orgModule(input), s_pre

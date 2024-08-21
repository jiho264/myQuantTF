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
        self.register_buffer("act_scaling_factor", torch.zeros(1))

    def fix(self):
        pass

    def unfix(self):
        pass

    def forward(self, A, pre_act_scaling_factor_A, B, pre_act_scaling_factor_B):
        A_int = A / pre_act_scaling_factor_A
        B_int = B / pre_act_scaling_factor_B
        act_scaling_factor = pre_act_scaling_factor_A * pre_act_scaling_factor_B
        self.act_scaling_factor = act_scaling_factor
        return (A_int @ B_int) * act_scaling_factor, act_scaling_factor


class QuantAct(nn.Module):
    def __init__(self, args_a={}):
        super(QuantAct, self).__init__()
        # print("qant!")

        self.activation_bit = args_a.get("bit_width", 8)

    def forward(self, x_hat, s_x=None, id_hat=None, s_id=None):
        with torch.no_grad():
            x_out = x_hat if id_hat == None else id_hat + x_hat

            s_out = x_out.abs().max() / (2 ** (self.activation_bit - 1) - 1)

        if s_x == None:
            # input quantization
            out_int = (
                (x_hat / s_out)
                .round()
                .clamp(
                    -(2 ** (self.activation_bit - 1)),
                    2 ** (self.activation_bit - 1) - 1,
                )
            )
        else:
            x_int = (x_hat / s_x).round()
            new_scaler = s_x / s_out
            out_int = (
                (x_int * new_scaler)
                .round()
                .clamp(
                    -(2 ** (self.activation_bit - 1)),
                    2 ** (self.activation_bit - 1) - 1,
                )
            )

            if id_hat is not None:
                id_int = (id_hat / s_id).round()
                new_scaler = s_id / s_out
                id_int = (
                    (id_int * new_scaler)
                    .round()
                    .clamp(
                        -(2 ** (self.activation_bit - 1)),
                        2 ** (self.activation_bit - 1) - 1,
                    )
                )

                out_int += id_int
        print("QuantAct", out_int.shape, s_out.shape)
        return out_int * s_out.view(-1), s_out


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

        self.bits = args_w.get("bit_width", 8)
        # only per-channel quantization is supported

        weight_fp32 = orgModule.weight.clone().detach()
        bias_fp32 = orgModule.bias.clone().detach()

        s_w = weight_fp32.view(weight_fp32.size(0), -1).abs().max(dim=1).values / (
            2 ** (self.bits - 1) - 1
        )

        self._n_ch = len(weight_fp32.size())
        self.s_w = s_w.view(-1, *([1] * (self._n_ch - 1)))

        self.w_int8 = (
            (weight_fp32.clone().detach() / self.s_w)
            .round()
            .clamp(-(2 ** (self.bits - 1)), 2 ** (self.bits - 1) - 1)
        )
        self.b_int8 = (bias_fp32.clone().detach() / self.s_w.squeeze()).round()

        # self.qact = QuantAct()

    def forward(self, x_hat, s_x):
        # 여기 x의 bit repr는 이전 layer에서 몇으로 줄였느냐에 따라 다름
        x_int = (x_hat / s_x).round()
        if s_x.dim() == 0:
            # 768 vs []
            s_a = self.s_w * s_x

        elif self.b_int8.shape[0] == s_x.shape[0] * 3:
            # 2304 vs 768
            s_x = torch.concat([s_x, s_x, s_x], dim=0)
            s_a = self.s_w.view(-1) * s_x
        elif self.b_int8.shape[0] == s_x.shape[0] * 4:
            # 3072 vs 768
            s_x = torch.concat([s_x, s_x, s_x, s_x], dim=0)
            s_a = self.s_w.view(-1) * s_x

        b_int32 = (self.b_int8 / s_x).round()
        # print(x_int.shape, self.w_int8.shape, b_int32.shape)

        a_int32 = self.fwd_func(x_int, self.w_int8, b_int32, **self.fwd_kwargs)

        if self.fwd_func == F.conv2d:
            # if self.s_w.dim() == 4:
            s_a = s_a.view(1, -1, 1, 1)
        # elif self.s_w.dim() == 2:
        elif self.fwd_func == F.linear:
            s_a = s_a.view(-1)

        a_hat = a_int32 * s_a
        return a_hat, s_a
        # return self.qact(a_hat, s_a)


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

        self.output_bit = args_gelu.get("output_bit", 8)

        self.n = 23  # sufficiently large integer
        # The minimum value for ensuring accuracy (varies depending on models)

        self.register_buffer("act_scaling_factor", torch.zeros(1))

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
        sigmoid_int = floor_ste.apply(
            exp_int * factor / 2 ** (31 - self.output_bit + 1)
        )
        sigmoid_scaling_factor = torch.Tensor([1 / 2 ** (self.output_bit - 1)]).cuda()

        x_int = pre_x_int * sigmoid_int
        scaling_factor = scaling_factor * sigmoid_scaling_factor
        self.act_scaling_factor = scaling_factor
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

        self.output_bit = args_softmax.get("output_bit", 16)

        self.n = 15  # sufficiently large integer
        # The minimum value for ensuring accuracy (varies depending on models)

        self.register_buffer("act_scaling_factor", torch.zeros(1))

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
        exp_int = floor_ste.apply(exp_int * factor / 2 ** (31 - self.output_bit + 1))
        scaling_factor = torch.Tensor([1 / 2 ** (self.output_bit - 1)]).cuda()

        self.act_scaling_factor = scaling_factor
        return exp_int * scaling_factor, scaling_factor

    def forward(self, input, s_pre, dim: int = -1):
        if self.do_quant:
            return self.int_forward(input, s_pre)
        else:
            print("FP Softmax")
            return F.softmax(input, dim=dim), s_pre


class IntLayerNorm(nn.Module):
    def __init__(self, orgModule, args_ln):
        super().__init__()
        self.do_quant = False

        self.weight = orgModule.weight.clone().detach()
        self.bias = orgModule.bias.clone().detach()
        self.orgModule = orgModule

        self.dim_sqrt = None
        self.register_buffer("norm_scaling_factor", torch.zeros(1))
        self.register_buffer("bias_integer", torch.zeros_like(self.bias))

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

        self.bias_integer = bias_int

        y_int = y_int + bias_int
        scaling_factor = scaling_factor * self.weight
        x = y_int * scaling_factor
        self.norm_scaling_factor = scaling_factor
        return x, scaling_factor

    def forward(self, input, s_pre):
        if self.do_quant:
            return self.int_forward(input, s_pre)
        else:
            print("FP LayerNorm")
            return self.orgModule(input), s_pre

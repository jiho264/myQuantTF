import torch, math
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
from torch import Tensor
from typing import Optional, Tuple

# from .quant_utils import AbsMaxQuantizer, MovAvgAbsMaxQuantizer
from torch.autograd import Function

import math
import numpy as np
from torch.autograd import Function, Variable
import torch
import bisect
from fractions import Fraction
import decimal
from decimal import Decimal
import time


def linear_quantize(input, scale, zero_point, is_weight):
    """
    Quantize single-precision input tensor to integers with the given scaling factor and zeropoint.
    Parameters:
    ----------
    input: single-precision input tensor to be quantized
    scale: scaling factor for quantization
    zero_pint: shift for quantization
    """

    # reshape scale and zeropoint for convolutional weights and activation
    if is_weight:
        if len(input.shape) == 4:
            scale = scale.view(-1, 1, 1, 1)
            zero_point = zero_point.view(-1, 1, 1, 1)
            # reshape scale and zeropoint for linear weights
        elif len(input.shape) == 2:
            scale = scale.view(-1, 1)
            zero_point = zero_point.view(-1, 1)
        else:
            scale = scale.view(-1)
            zero_point = zero_point.view(-1)
    else:
        if len(input.shape) == 2:
            scale = scale.view(1, -1)
            zero_point = zero_point.view(1, -1)
        elif len(input.shape) == 3:
            scale = scale.view(1, 1, -1)
            zero_point = zero_point.view(1, 1, -1)
        elif len(input.shape) == 4:
            scale = scale.view(1, -1, 1, 1)
            zero_point = zero_point.view(1, -1, 1, 1)
        else:
            raise NotImplementedError

    # quantized = float / scale + zero_point
    return torch.round(1.0 / scale * input + zero_point)


def symmetric_linear_quantization_params(num_bits, min_val, max_val):
    """
    Compute the scaling factor with the given quantization range for symmetric quantization.
    Parameters:
    ----------
    saturation_min: lower bound for quantization range
    saturation_max: upper bound for quantization range
    """
    # in this part, we do not need any gradient computation,
    # in order to enfore this, we put torch.no_grad()
    with torch.no_grad():
        n = 2 ** (num_bits - 1) - 1
        eps = torch.finfo(torch.float32).eps

        max_val = torch.max(-min_val, max_val)
        scale = max_val / float(n)
        scale.clamp_(eps)

    return scale


class SymmetricQuantFunction(Function):
    """
    Class to quantize the given floating-point values using symmetric quantization with given range and bitwidth.
    """

    @staticmethod
    def forward(ctx, x, k, specified_scale, is_weight):
        """
        x: floating point tensor to be quantized
        k: quantization bitwidth
        Note that the current implementation of SymmetricQuantFunction requires pre-calculated scaling factor.
        specified_scale: pre-calculated scaling factor for the tensor x
        """

        scale = specified_scale

        zero_point = torch.tensor(0.0).cuda()

        n = 2 ** (k - 1) - 1
        new_quant_x = linear_quantize(x, scale, zero_point, is_weight=is_weight)
        new_quant_x = torch.clamp(new_quant_x, -n - 1, n)

        ctx.scale = scale
        ctx.is_weight = is_weight
        return new_quant_x

    @staticmethod
    def backward(ctx, grad_output):

        scale = ctx.scale
        is_weight = ctx.is_weight
        if is_weight:
            if len(grad_output.shape) == 4:
                scale = scale.view(-1, 1, 1, 1)
            elif len(grad_output.shape) == 2:
                scale = scale.view(-1, 1)
            else:
                scale = scale.view(-1)
        else:
            if len(grad_output.shape) == 2:
                scale = scale.view(1, -1)
            elif len(grad_output.shape) == 3:
                scale = scale.view(1, 1, -1)
            elif len(grad_output.shape) == 4:
                scale = scale.view(1, -1, 1, 1)
            else:
                raise NotImplementedError
        return grad_output.clone() / scale, None, None, None


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


def batch_frexp(inputs, max_bit=31):
    """
    Decompose the scaling factor into mantissa and twos exponent.
    Parameters:
    ----------
    inputs: scaling factor
    return: (mantissa, exponent)
    """

    shape_of_input = inputs.size()

    # trans the input to be a 1-d tensor
    inputs = inputs.view(-1)

    output_m, output_e = np.frexp(inputs.cpu().numpy())
    tmp_m = []
    for m in output_m:
        int_m_shifted = int(
            Decimal(m * (2**max_bit)).quantize(
                Decimal("1"), rounding=decimal.ROUND_HALF_UP
            )
        )
        tmp_m.append(int_m_shifted)
    output_m = np.array(tmp_m)

    output_e = float(max_bit) - output_e

    return torch.from_numpy(output_m).cuda().view(shape_of_input), torch.from_numpy(
        output_e
    ).cuda().view(shape_of_input)


class fixedpoint_mul(Function):
    """
    Function to perform fixed-point arthmetic that can match integer arthmetic on hardware.
    Parameters:
    ----------
    pre_act: input tensor
    pre_act_scaling_factor: ithe scaling factor of the input tensor
    bit_num: quantization bitwidth
    quant_mode: The mode for quantization, 'symmetric' or 'asymmetric'
    z_scaling_factor: the scaling factor of the output tensor
    identity: identity tensor
    identity_scaling_factor: the scaling factor of the identity tensor
    """

    @staticmethod
    def forward(
        ctx,
        pre_act,
        pre_act_scaling_factor,
        bit_num,
        quant_mode,
        z_scaling_factor,
        identity=None,
        identity_scaling_factor=None,
    ):

        # TODO(Sehoon): May require other type of reshape
        if len(pre_act.shape) == 2:
            reshape = lambda x: x.view(1, -1)
        elif len(pre_act.shape) == 3:
            reshape = lambda x: x.view(1, 1, -1)
        elif len(pre_act.shape) == 4:
            reshape = lambda x: x.view(1, -1, 1, 1)
        else:
            raise NotImplementedError
        ctx.identity = identity

        if quant_mode == "symmetric":
            n = 2 ** (bit_num - 1) - 1
        else:
            n = 2**bit_num - 1

        with torch.no_grad():
            pre_act_scaling_factor = reshape(pre_act_scaling_factor)
            if identity is not None:
                identity_scaling_factor = reshape(identity_scaling_factor)

            ctx.z_scaling_factor = z_scaling_factor

            z_int = torch.round(pre_act / pre_act_scaling_factor)
            _A = pre_act_scaling_factor.type(torch.double)
            _B = (z_scaling_factor.type(torch.float)).type(torch.double)
            new_scale = _A / _B
            # print(new_scale)
            # exit()
            new_scale = reshape(new_scale)

            m, e = batch_frexp(new_scale)
            output = z_int.type(torch.double) * m.type(torch.double)
            output = torch.round(output / (2.0**e))

            if identity is not None:
                # needs addition of identity activation
                wx_int = torch.round(identity / identity_scaling_factor)

                _A = identity_scaling_factor.type(torch.double)
                _B = (z_scaling_factor.type(torch.float)).type(torch.double)
                new_scale = _A / _B
                new_scale = reshape(new_scale)

                m1, e1 = batch_frexp(new_scale)
                output1 = wx_int.type(torch.double) * m1.type(torch.double)
                output1 = torch.round(output1 / (2.0**e1))

                output = output1 + output

            if bit_num in [4, 8, 16, 32]:
                if quant_mode == "symmetric":
                    return torch.clamp(output.type(torch.float), -n - 1, n)
                else:
                    return torch.clamp(output.type(torch.float), 0, n)
            else:
                return output.type(torch.float)

    @staticmethod
    def backward(ctx, grad_output):
        identity_grad = None
        if ctx.identity is not None:
            identity_grad = grad_output.clone() / ctx.z_scaling_factor
        return (
            grad_output.clone() / ctx.z_scaling_factor,
            None,
            None,
            None,
            None,
            identity_grad,
            None,
        )


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

        self.activation_bit = args_a.get("bit_width", 16)

        self.running_stat = 16
        self.quant_mode = "symmetric"
        self.act_range_momentum = 0.95
        self.min_val = torch.zeros(1)
        self.max_val = torch.zeros(1)
        self.act_function = SymmetricQuantFunction.apply

    def forward(
        self,
        x,
        pre_act_scaling_factor=None,
        identity=None,
        identity_scaling_factor=None,
    ):
        # collect runnng stats
        with torch.no_grad():
            x_act = x if identity is None else identity + x
            if self.running_stat > 0:
                if len(x_act.shape) == 4:
                    x_act = x_act.permute(0, 2, 3, 1)
                v = x_act.reshape(-1, x_act.shape[-1])
                v = v.transpose(0, 1)

                cur_min = v.min(axis=1).values
                cur_max = v.max(axis=1).values
                if torch.eq(self.min_val, self.max_val).all():
                    self.min_val = cur_min
                    self.max_val = cur_max
                else:
                    self.min_val = self.min_val * self.act_range_momentum + cur_min * (
                        1 - self.act_range_momentum
                    )
                    self.max_val = self.max_val * self.act_range_momentum + cur_max * (
                        1 - self.act_range_momentum
                    )
                self.max_val = self.max_val.max()
                self.min_val = self.min_val.min()
                self.running_stat -= 1
                if self.running_stat == 0:
                    print("min_val", self.min_val, "max_val", self.max_val)

            self.act_scaling_factor = symmetric_linear_quantization_params(
                self.activation_bit, self.min_val, self.max_val
            )

        if pre_act_scaling_factor is None:
            # this is for the input quantization
            quant_act_int = self.act_function(
                x, self.activation_bit, self.act_scaling_factor, False
            )
        else:
            quant_act_int = fixedpoint_mul.apply(
                x,
                pre_act_scaling_factor,
                self.activation_bit,
                self.quant_mode,
                self.act_scaling_factor,
                identity,
                identity_scaling_factor,
            )

        correct_output_scale = self.act_scaling_factor.view(-1)
        return quant_act_int * correct_output_scale, self.act_scaling_factor

    # def forward(self, x_hat, s_x=None, id_hat=None, s_id=None):
    #     with torch.no_grad():
    #         x_out = x_hat if id_hat == None else id_hat + x_hat

    #         s_out = x_out.abs().max() / (2 ** (self.activation_bit - 1) - 1)

    #     if s_x == None:
    #         # input quantization
    #         out_int = (
    #             (x_hat / s_out)
    #             .round()
    #             .clamp(
    #                 -(2 ** (self.activation_bit - 1)),
    #                 2 ** (self.activation_bit - 1) - 1,
    #             )
    #         )
    #     else:
    #         x_int = (x_hat / s_x).round()
    #         new_scaler = s_x / s_out
    #         out_int = (
    #             (x_int * new_scaler)
    #             .round()
    #             .clamp(
    #                 -(2 ** (self.activation_bit - 1)),
    #                 2 ** (self.activation_bit - 1) - 1,
    #             )
    #         )

    #         if id_hat is not None:
    #             id_int = (id_hat / s_id).round()
    #             new_scaler = s_id / s_out
    #             id_int = (
    #                 (id_int * new_scaler)
    #                 .round()
    #                 .clamp(
    #                     -(2 ** (self.activation_bit - 1)),
    #                     2 ** (self.activation_bit - 1) - 1,
    #                 )
    #             )

    #             out_int += id_int

    #     out_hat = out_int * s_out.view(-1)
    #     print(out_hat.min(), out_hat.max())
    #     return out_hat, s_out


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

        self.per_channel = True
        self.weight = weight_fp32
        self.bias = bias_fp32
        self.quant_mode = "symmetric"
        self.weight_bit = self.bits
        self.bias_bit = 32
        self.weight_function = SymmetricQuantFunction.apply

    def fcforward(self, x, prev_act_scaling_factor=None):
        with torch.no_grad():
            w = self.weight
            if self.per_channel:
                v = w.reshape(w.shape[0], -1)
                cur_min = v.min(axis=1).values
                cur_max = v.max(axis=1).values
                self.min_val = cur_min
                self.max_val = cur_max
            else:
                raise Exception("For weight, we only support per_channel quantization.")

            self.fc_scaling_factor = symmetric_linear_quantization_params(
                self.weight_bit, self.min_val, self.max_val
            )

        self.weight_integer = self.weight_function(
            self.weight, self.weight_bit, self.fc_scaling_factor, True
        )

        bias_scaling_factor = self.fc_scaling_factor * prev_act_scaling_factor

        if self.bias is not None:
            self.bias_integer = self.weight_function(
                self.bias, self.bias_bit, bias_scaling_factor, True
            )
        else:
            self.bias_integer = None

        prev_act_scaling_factor = prev_act_scaling_factor.view(1, -1)
        x_int = x / prev_act_scaling_factor

        return (
            F.linear(x_int, weight=self.weight_integer, bias=self.bias_integer)
            * bias_scaling_factor,
            bias_scaling_factor,
        )

    def convforward(self, x, pre_act_scaling_factor=None):
        if self.quant_mode == "symmetric":
            self.weight_function = SymmetricQuantFunction.apply
        elif self.quant_mode == "asymmetric":
            raise NotImplementedError(
                "unsupported quant mode: {}".format(self.quant_mode)
            )
        else:
            raise ValueError("unknown quant mode: {}".format(self.quant_mode))

        with torch.no_grad():
            w = self.weight
            if self.per_channel:
                v = w.reshape(w.shape[0], -1)
                cur_min = v.min(axis=1).values
                cur_max = v.max(axis=1).values
                self.min_val = cur_min
                self.max_val = cur_max
            else:
                raise Exception("For weight, we only support per_channel quantization.")

            self.conv_scaling_factor = symmetric_linear_quantization_params(
                self.weight_bit, self.min_val, self.max_val
            )

        self.weight_integer = self.weight_function(
            self.weight, self.weight_bit, self.conv_scaling_factor, True
        )
        bias_scaling_factor = self.conv_scaling_factor * pre_act_scaling_factor
        self.bias_integer = self.weight_function(
            self.bias, self.bias_bit, bias_scaling_factor, True
        )

        pre_act_scaling_factor = pre_act_scaling_factor.view(1, -1, 1, 1)
        x_int = x / pre_act_scaling_factor
        correct_output_scale = bias_scaling_factor.view(1, -1, 1, 1)

        return (
            F.conv2d(x_int, self.weight_integer, self.bias_integer, **self.fwd_kwargs)
            * correct_output_scale,
            correct_output_scale,
        )

    def forward(self, x_hat, s_x):
        if self.fwd_func == F.conv2d:
            return self.convforward(x_hat, s_x)
        elif self.fwd_func == F.linear:
            return self.fcforward(x_hat, s_x)
        else:
            raise ValueError("unknown forward function: {}".format(self.fwd_func))
        # # 여기 x의 bit repr는 이전 layer에서 몇으로 줄였느냐에 따라 다름

        # s_a = self.s_w * s_x
        # b_int32 = (self.b_int8 / s_x).round()

        # if self.fwd_func == F.conv2d:
        #     s_x = s_x.view(1, -1, 1, 1)
        #     s_a = s_a.view(1, -1, 1, 1)

        # elif self.fwd_func == F.linear:
        #     s_a = s_a.view(1, -1)

        # x_int = (x_hat / s_x).round()

        # a_int32 = self.fwd_func(x_int, self.w_int8, b_int32, **self.fwd_kwargs)
        # a_hat = a_int32 * s_a

        # return a_hat, s_a


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

        self.output_bit = args_gelu.get("output_bit", 16)

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

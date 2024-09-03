import torch, math
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
from torch import Tensor
from typing import Optional, Tuple
import numpy as np
from torch.autograd import Function


def lp_loss(pred, tgt, p=2.0, reduction="none"):
    """
    loss function measured in L_p Norm
    """
    if reduction == "none":
        return (pred - tgt).abs().pow(p).sum(1).mean()
    else:
        return (pred - tgt).abs().pow(p).mean()


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


class QuantMatMul(nn.Module):
    """
    Class to quantize weights of given matmul layer
    """

    def __init__(self):
        super(QuantMatMul, self).__init__()

    def forward(self, A, pre_act_scaling_factor_A, B, pre_act_scaling_factor_B):
        # print(pre_act_scaling_factor_A, pre_act_scaling_factor_B)
        if pre_act_scaling_factor_B == 0:
            return A @ B, 0

        """ Verify INT 8 GEMM """
        A_int = round_ste.apply(A / pre_act_scaling_factor_A)
        if A_int.min() >= 0:
            # for Softmax @ V (u8s8)
            # assert (
            #     0 <= A_int.min() and A_int.max() <= 255
            # ), f"{A_int.min()} {A_int.max()}"
            pass
        else:
            # for Q @ K (s8s8)
            # assert (
            #     -128 <= A_int.min() and A_int.max() <= 127
            # ), f"{A_int.min()} {A_int.max()}"
            pass

        B_int = round_ste.apply(B / pre_act_scaling_factor_B).clamp(-128, 127)

        act_scaling_factor = pre_act_scaling_factor_A * pre_act_scaling_factor_B
        return (A_int @ B_int) * act_scaling_factor, act_scaling_factor


class QuantAct(nn.Module):
    def __init__(self, args_a={}, which=None):
        super(QuantAct, self).__init__()
        self.do_quant = False
        if args_a == {}:
            # do not quantize
            return

        assert (
            args_a.get("scheme") == "MovAvgAbsMaxQuantizer"
        ), "only MovAvgAbsMaxQuantizer scheme is supported for Activation Quantization"

        assert (
            args_a.get("per_channel", True) == False
        ), "only per-tensor quantization is supported for Activation Quantization"

        if which == "idAdd":
            self.bit_width = args_a.get("idadd_bit_width")
            # print(f"Int Activation {self.bit_width} for {which} for identity add")
        elif which == "softmax_act":
            # 24.08.26 INT9 == UINT8 for non negative values
            self.bit_width = args_a.get("bit_width") + 1
        else:
            self.bit_width = args_a.get("bit_width")
            # print(f"Int Activation {self.bit_width}")

        self.running_stat = args_a.get("batches")
        self.min_val = torch.zeros(1)
        self.max_val = torch.zeros(1)
        self.s_out = None
        self.momentum = args_a.get("momentum", 0.95)

    def _quantize(self, x_hat, s_x):
        x_int = round_ste.apply(x_hat / s_x)
        x_int_clamp = x_int.clamp(
            -(2 ** (self.bit_width - 1)),
            2 ** (self.bit_width - 1) - 1,
        )
        return x_int_clamp

    def _compute_scaler(self, x_min, x_max):
        x_abs_max = torch.max(x_max.abs(), x_min.abs())
        return x_abs_max / (2 ** (self.bit_width - 1) - 1)

    def forward(self, x_hat, s_x=None, id_hat=None, s_id=None):
        if self.do_quant:
            """
            [ ] Turn to INT Division ( b * 2^ -c )

            x_int = x_hat / s_x
            out_int = x_int * (s_x / s_out)

            id_int = id_hat / s_id
            out_int == id_int * (s_id / s_out)

            이렇게 계산하기 때문에, s_x와 s_id는 약분되어서 사라짐.
            다만 quantize 함수를 이용해서 구현할 수도 있음.
            동일하게 return은 INT이며, 반올림과정에서 사소한 오차 있을 수 있음.
            실제 int inference에서는 m*2**-c를 곱하는 식으로 나눠야하기 때문에 이 부분은 추후 수정 필요함.

            사실 QuantAct에서 하는 일은 이전 scaler가 무엇이든지 상관없고, 들어온 input x_hat의 min, max 값에 대해서만 scaler를 구하기 때문에, input의 shape가 어떻게 생겼고 말고 상관없이 스칼라 하나만 구하면됨.

            """

            with torch.no_grad():
                x_out = x_hat if id_hat == None else id_hat + x_hat

                if self.running_stat > 0:
                    if len(x_out.shape) == 4:
                        x_out = x_out.permute(0, 2, 3, 1)
                    v = x_out.reshape(-1, x_out.shape[-1])
                    v = v.transpose(0, 1)

                    cur_min = v.min(axis=1).values
                    cur_max = v.max(axis=1).values
                    if torch.eq(self.min_val, self.max_val).all():
                        self.min_val = cur_min
                        self.max_val = cur_max
                    else:
                        self.min_val = self.min_val * self.momentum + cur_min * (
                            1 - self.momentum
                        )
                        self.max_val = self.max_val * self.momentum + cur_max * (
                            1 - self.momentum
                        )
                    self.max_val = self.max_val.max()
                    self.min_val = self.min_val.min()

                    self.running_stat -= 1
                    if self.running_stat == 0:
                        s_out = self._compute_scaler(self.min_val, self.max_val)
                        self.s_out = nn.Parameter(s_out, requires_grad=True)

            if self.s_out == None:
                s_out = self._compute_scaler(self.min_val, self.max_val)
            else:
                s_out = self.s_out

            out_int = self._quantize(x_hat, s_out)

            if s_x is not None and id_hat is not None:
                out_int += self._quantize(id_hat, s_out)

            out_hat = out_int * s_out.view(-1)
            return out_hat, s_out
        else:

            x_out = x_hat if id_hat == None else id_hat + x_hat
            return x_out, 0  # when org inference


class QuantLinearWithWeight(nn.Module):
    def __init__(self, orgModule, args_w, args_a=None):
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

        self.W_FP32 = orgModule.weight.clone().detach()
        self.B_FP32 = orgModule.bias.clone().detach()

        if args_w == {}:
            # do not quantize
            return

        self.bit_width = args_w.get("bit_width")

        assert (
            args_w.get("scheme") == "AbsMaxQuantizer"
        ), "only AbsMaxQuantizer scheme is supported for Weight Quantization"

        assert (
            args_w.get("per_channel", False) == True
        ), "only per-channel quantization is supported for Weight Quantization"

        with torch.no_grad():
            s_w = self.W_FP32.view(self.W_FP32.size(0), -1).abs().max(dim=1).values / (
                2 ** (self.bit_width - 1) - 1
            )

            _n_ch = len(self.W_FP32.size())
            self.S_W = s_w.view(-1, *([1] * (_n_ch - 1)))

            self.W_INT8 = (
                (self.W_FP32.clone().detach() / self.S_W)
                .round()
                .clamp(-(2 ** (self.bit_width - 1)), 2 ** (self.bit_width - 1) - 1)
            )
            self.B_INT8 = (self.B_FP32.clone().detach() / self.S_W.squeeze()).round()

        """ Activation quantizer """
        if args_a is not None:
            # all weighted layer without head !!
            # the head's output is not needed to be quantized
            self.act_quant = QuantAct(args_a=args_a)
        else:
            self.act_quant = lambda a, b: (a, b)

        """ AdaRound """
        self.adaround_scheme = args_w.get("AdaRound", False)
        if self.adaround_scheme:
            self.zeta = 1.1  # fixed param for function h()
            self.gamma = -0.1  # fixed pamam for function h()
            self.lamda = 1  # lambda. fixed param for regularization function f()

            self.rouning_value = None

            # [1] init the v value. (h(v) == rounding value)
            self._v = None
            self._init_v()

    def _init_v(self) -> Tensor:
        # [1-1] compute the residual == initial h(v)
        with torch.no_grad():
            self.W_INT_FLOOR = (
                (self.W_FP32 / self.S_W)
                .floor()
                .clamp(-(2 ** (self.bit_width - 1)), 2 ** (self.bit_width - 1) - 1)
            )

            _residual = self.W_INT8 - self.W_INT_FLOOR

            assert torch.all(
                (_residual == 0) | (_residual == 1)
            ), "The residual must be {0, 1}."

            # [1-2] compute the v value using inverse h() function
            _v = -torch.log(
                (self.zeta - self.gamma) / (_residual - self.gamma) - 1
            )  # h^-1
            self._v = nn.Parameter(_v, requires_grad=True)
            assert (_residual - self._h()).abs().sum() == 0
            self.residual = _residual
            print("    Initiated the V")

    def _h(self) -> Tensor:
        # Rectified_sigmoid (strached sigmoid function)

        return torch.clamp(
            self._v.sigmoid() * (self.zeta - self.gamma) + self.gamma, 0, 1
        )

    def f_reg(self, beta=2.0) -> Tensor:
        # _regularization_term for determining the v
        return (1 - (2 * self._h() - 1).abs().pow(beta)).sum()

    def _get_w_int8(self) -> Tensor:
        if self.rouning_value == None:
            # return FP
            return (self.W_INT_FLOOR + self._h()).clamp(
                -(2 ** (self.bit_width - 1)), 2 ** (self.bit_width - 1) - 1
            )

        else:
            return (self.W_INT_FLOOR + self.rouning_value).clamp(
                -(2 ** (self.bit_width - 1)), 2 ** (self.bit_width - 1) - 1
            )

    def setRoundingValues(self):
        with torch.no_grad():
            FIXED_ROUNDING_VALUE = self._h().clone().detach()
            assert torch.all(
                (FIXED_ROUNDING_VALUE == 0) | (FIXED_ROUNDING_VALUE == 1)
            ), "The rounding value have to be {0, 1}."

            self.rouning_value = FIXED_ROUNDING_VALUE
            print("    Set the rounding value")

            if torch.equal(self.residual, self._h()):
                print("    - The rounding value is same with the residual.")

    def forward(self, x_hat, s_x):
        if self.do_quant and s_x == 0:
            """weight quantization only (ex. W8A32..)"""
            return (
                self.fwd_func(
                    x_hat, self.W_INT8 * self.S_W, self.B_FP32, **self.fwd_kwargs
                ),
                s_x,
            )
        elif self.do_quant:
            """weight and activation quantization (ex. W8A8..)"""
            s_a = self.S_W * s_x
            b_int32 = round_ste.apply(self.B_INT8 / s_x)

            if self.fwd_func == F.conv2d:
                s_x = s_x.view(1, -1, 1, 1)
                s_a = s_a.view(1, -1, 1, 1)

            elif self.fwd_func == F.linear:
                s_a = s_a.view(1, -1)

            x_int = round_ste.apply(x_hat / s_x)
            """ Verify INT 8 GEMM """
            assert (
                -128 <= x_int.min() and x_int.max() <= 127
            ), f"{x_int.min()} {x_int.max()}"

            if self.adaround_scheme:
                """if using AdaRound"""
                # usually using 4bit quantization
                a_int32 = self.fwd_func(
                    x_int, self._get_w_int8(), b_int32, **self.fwd_kwargs
                )
            else:
                a_int32 = self.fwd_func(x_int, self.W_INT8, b_int32, **self.fwd_kwargs)

            a_hat = a_int32 * s_a

            return self.act_quant(a_hat, s_a)
        else:
            """No quantization"""
            return (
                self.fwd_func(x_hat, self.W_FP32, self.B_FP32, **self.fwd_kwargs),
                s_x,
            )


def bits_required(x_int):
    # 절대값을 취하고 이진수 변환 후, 부호 비트 포함해서 비트 수 계산
    x_int = int(x_int.item())  # 텐서에서 파이썬 정수로 변환
    if x_int == 0:
        return 1  # 0은 1비트로 표현 가능
    return (
        len(bin(abs(x_int))) - 2 + 1
    )  # bin()은 '0b'로 시작하기 때문에 -2, 부호 비트 포함


class IntGELU(nn.Module):
    def __init__(self, args_gelu):
        super().__init__()
        self.do_quant = False
        if args_gelu == {}:
            # do not quantize
            return

        self.sigmoid_bit_width = args_gelu.get("sigmoid_bit_width")

        self.n = args_gelu.get("left_shift_for_exp", 23)
        # 23 is an I-ViT's default value
        # sufficiently large integer
        # The minimum value for ensuring accuracy (varies depending on models)

        print(
            f"IntGELU    | sigmoid bit: {self.sigmoid_bit_width}, exp Lshift: {self.n}"
        )

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
        # print(
        #     "gelu    max bit length: ",
        #     bits_required(exp_int.max()),
        #     torch.unique(exp_int).numel(),
        #     bits_required(exp_int_sum.max()),
        #     torch.unique(exp_int_sum).numel(),
        # )
        # print("")
        exp_int_sum.clamp_max_(2**31 - 1)
        factor = floor_ste.apply((2**31 - 1) / exp_int_sum)
        sigmoid_int = floor_ste.apply(
            exp_int * factor / 2 ** (31 - self.sigmoid_bit_width + 1)
        )
        sigmoid_scaling_factor = torch.Tensor(
            [1 / 2 ** (self.sigmoid_bit_width - 1)]
        ).cuda()
        assert sigmoid_int.min() >= 0

        x_int = pre_x_int * sigmoid_int
        scaling_factor = scaling_factor * sigmoid_scaling_factor
        return x_int * scaling_factor, scaling_factor

    def forward(self, input, s_pre):
        if self.do_quant:
            """Verify under INT8 input"""
            with torch.no_grad():
                _test_input_int = (input / s_pre).round()
                # assert -128 <= _test_input_int.min() and _test_input_int.max() <= 127

            return self.int_forward(input, s_pre)
        else:
            # print("FP GELU")
            return F.gelu(input), s_pre


class IntSoftMax(nn.Module):
    def __init__(self, args_softmax):
        super().__init__()
        self.do_quant = False
        if args_softmax == {}:
            # do not quantize
            return

        self.bit_width = args_softmax.get("bit_width")

        self.n = args_softmax.get("left_shift_for_exp", 15)
        # 15 is an I-ViT's default value
        # sufficiently large integer
        # The minimum value for ensuring accuracy (varies depending on models)

        print(f"IntSoftMax | output bit : {self.bit_width}, exp Lshift : {self.n}")

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
        # print(
        #     "softmax max bit length: ",
        #     bits_required(exp_int.max()),
        #     torch.unique(exp_int).numel(),
        #     bits_required(exp_int_sum.max()),
        #     torch.unique(exp_int_sum).numel(),
        # )

        exp_int_sum.clamp_max_(2**31 - 1)
        factor = floor_ste.apply((2**31 - 1) / exp_int_sum)
        exp_int = floor_ste.apply(exp_int * factor / 2 ** (31 - self.bit_width + 1))
        scaling_factor = torch.Tensor([1 / 2 ** (self.bit_width - 1)]).cuda()

        assert exp_int.min() >= 0
        # print("softmax out_int")
        # print(exp_int.min(), exp_int.max())
        return exp_int * scaling_factor, scaling_factor

    def forward(self, input, s_pre, dim: int = -1):
        if self.do_quant:
            """when other's non quantize"""
            if s_pre == 0:
                return self.int_forward(input, torch.tensor(1).to("cuda"))

            """Verify under INT8 input"""
            with torch.no_grad():
                _test_input_int = (input / s_pre).round()
                assert -128 <= _test_input_int.min() and _test_input_int.max() <= 127
            return self.int_forward(input, s_pre)
        else:
            # print("FP Softmax")
            return F.softmax(input, dim=dim), s_pre


class LogSqrt2Quantizer(nn.Module):
    def __init__(self, args_any):
        super().__init__()
        """ log sqrt 2 quantizer for attention map """
        self.do_quant = True
        if args_any == {}:
            # do not quantize
            return

        self.bit_width = args_any.get("act_quant_bit_width")
        self.n_levels = 2**self.bit_width

        self.inited = False

        # self._k = nn.Parameter(
        #     torch.tensor(0.0001, dtype=torch.float32), requires_grad=True
        # )
        # self._a = nn.Parameter(
        #     torch.tensor(
        #         [
        #             1.0000,
        #             0.7071,
        #             0.5000,
        #             0.3536,
        #             0.2500,
        #             0.1768,
        #             0.1250,
        #             0.0884,
        #             0.0625,
        #             0.0442,
        #             0.0313,
        #             0.0221,
        #             0.0156,
        #             0.0110,
        #             0.0078,
        #             0.0055,
        #         ]
        #     ),
        #     requires_grad=True,
        # )
        self.pi = torch.tensor(math.pi)

        print(f"Int log_sqrt_2 quantizer | output bit : {self.bit_width}")

        self.int_bias = None
        self.times = None
        self.int_minv = None
        self.int_maxv = None

    # def _f_map(self, x_q):
    #     """Dirac Delta Function"""
    #     assert 0 <= x_q.min() and x_q.max() <= self.n_levels - 1, f"{x_q.unique()}"

    #     def _dirac_delta(x, offset, pi, a):
    #         a = a[offset]
    #         ddf = (1 / (a.abs() * pi.sqrt())) * torch.exp(-(((x - offset) / a) ** 2))

    #         return ddf

    #     _sum = torch.zeros_like(x_q)

    #     for i in range(self.n_levels):
    #         _sum = _sum + _dirac_delta(x_q, i, self.pi, self._a)

    #     return _sum

    def int_huge_16bit_int_to_log(self, x, times):
        """ref from desmos graphs"""
        return -1 * x.log2().round() * 2**times, times

    def int_huge_16bit_log_to_int(self, x, times):
        base = torch.exp(torch.log(torch.tensor(2)) / 2**times)
        return base ** (-1 * x), base

    def tmp_uniform_quant_dequant(self, x_int, min=None, max=None):
        if min is None and max is None:
            # for searching min, max
            min = x_int.min()
            max = x_int.max()
        else:
            # for inference
            min = min
            max = max

        scale = (max - min) / (self.n_levels - 1)
        zp = torch.round(-min / scale)

        x_int_log_q_quant = torch.round(x_int / scale) + zp
        # print(x_int_log_q_quant.unique())
        x_int_log_q_quant.clamp_(0, self.n_levels - 1)
        x_int_log_q_deuant = torch.round((x_int_log_q_quant - zp) * scale)
        # 어차피 숫자 엄청 커서 라운딩해봤자 큰 차이 없음.
        # print(x_int_log_q_deuant.unique())

        return x_int_log_q_deuant, min, max

    def init_bias_16bit(self, input):
        x_int = input.clone().detach()
        best_times = -10
        best_score = 1e10
        best_bias = None
        best_min = None
        best_max = None
        dja = None
        tmp_base_for_indicate = None

        best_map = None
        for times in range(8, 19):
            for bias in [
                2,
                3,
                4,
                5,
                6,
                7,
                8,
                9,
                10,
                11,
                12,
                13,
                14,
                15,
                20,
                24,
                25,
                26,
                27,
                28,
                30,
                40,
                50,
                65,
                327,
                655,
                1310,
                1966,
                2621,
                3276,
                6553,
                13107,
                32768,
                65536,
            ]:
                # [1] add bias for remove Inf
                x_int = x_int + bias

                # [2] log quantization in huge domain
                x_int_log_q, _ = self.int_huge_16bit_int_to_log(x_int, times)
                # print(f"bias : {bias}, times : {times}")
                # print(f"[1] {x_int_log_q.unique()} {x_int_log_q.unique().numel()}")

                # [3] Uniform Quantization for log values
                # x_int_log_q_q_dq, min, max = self.tmp_uniform_quant_dequant(x_int_log_q)

                # [4] dequantization from log values to INT16
                x_int_log_dq, base = self.int_huge_16bit_log_to_int(x_int_log_q, times)

                x_int_log_dq = x_int_log_dq - bias
                x_int_log_dq = x_int_log_dq.clamp(0, 65535).round()
                # if x_int_log_dq.unique().numel() != self.n_levels:
                #     continue
                map = x_int_log_dq.unique()
                # print(f"[2] {x_int_log_dq.unique()} {x_int_log_dq.unique().numel()}")
                # print(torch.norm(x_int - x_int_log_dq, p=2))
                # print()
                # [5] calculate the score

                score = torch.norm(x_int - x_int_log_dq, p=2)
                if score < best_score:
                    best_score = score
                    best_bias = bias
                    best_min = min
                    best_max = max
                    self.times = times
                    best_map = map
                    dja = x_int_log_q.unique()
                    tmp_base_for_indicate = base
        print(
            f"best bias : {best_bias }, best times : {self.times} best score : {best_score}"
        )
        print(f"best min : {best_min}, best max : {best_max}")
        print(f"best result : {dja},{dja.numel()}, base: {tmp_base_for_indicate}")
        print()

        self.int_bias = best_bias
        # self.times = torch.exp(torch.log(torch.tensor(2)) / 2**best_times)
        self.int_minv = best_min
        self.int_maxv = best_max
        self.map = best_map
        print(self.map)
        return

    def _map_quantize(self, x_hat, s_x):
        # [1] add bias for remove Inf
        x_int = round_ste.apply(x_hat / s_x)
        print(torch.norm(x_int, p=2))
        x_int = x_int + self.int_bias

        # [2] log quantization in huge domain
        x_int_log_q, _ = self.int_huge_16bit_int_to_log(x_int, self.times)

        # [3] Uniform Quantization for log values
        x_int_log_q_q_dq, _, _ = self.tmp_uniform_quant_dequant(x_int_log_q)
        print("log -> quant ", x_int_log_q_q_dq.unique())

        unq = x_int_log_q_q_dq.unique()

        for lognum, idx in zip(unq, range(len(self.map))):
            i = (len(self.map)) - 1 - idx
            print(lognum, self.map[i])
            x_int_log_q_q_dq = torch.where(
                x_int_log_q_q_dq == lognum, self.map[i], x_int_log_q_q_dq
            )
        print(torch.norm(x_int_log_q_q_dq, p=2))
        out = x_int_log_q_q_dq

        return out * s_x, s_x
        # # [4] dequantization from log values to INT16
        # x_int_log_dq, base = self.int_huge_16bit_log_to_int(
        #     x_int_log_q_q_dq, self.times
        # )

        # x_int_log_dq = x_int_log_dq - self.int_bias

        # return x_int_log_dq * s_x, s_x

    def forward(self, x_hat: torch.Tensor, s_x: torch.Tensor):
        if self.do_quant:
            if s_x == 0:
                """when other's non quantize"""
                return x_hat, s_x

            """Verify under INT8 input"""
            assert 0 <= x_hat.min() and x_hat.max() <= 1, f"{x_hat.min()} {x_hat.max()}"

            if self.inited == False:

                self.init_bias_16bit(round_ste.apply(x_hat / s_x))

                self.inited = True
            dja, wns = self._map_quantize(x_hat, s_x)
            print(dja.min(), dja.max())
            print()
            print()
            print()
            print()
            return dja, wns
        else:
            return x_hat, s_x


class IntLayerNorm(nn.Module):
    def __init__(self, orgModule, args_ln):
        super().__init__()
        self.do_quant = False

        self.weight = orgModule.weight.clone().detach()
        self.bias = orgModule.bias.clone().detach()
        self.orgModule = orgModule
        if args_ln == {}:
            # do not quantize
            return
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
            # print("FP LayerNorm")
            return self.orgModule(input), s_pre

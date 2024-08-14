import torch, math
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.vision_transformer import Encoder, EncoderBlock, MLPBlock
from collections import OrderedDict
from torch import Tensor

from .quantizer import quantizerDict, StraightThrough


class QuantBase(nn.Module):
    def __init__(self):
        super().__init__()
        self.do_quant = False
        self.__inited = False
        self.Quantizer = StraightThrough()

    def init_quantizer(self, args: dict, org_tensor=None):
        if self.__inited == True:
            raise Exception("Weight quantizer is already inited.")

        if org_tensor != None:
            if args.get("scheme") in [
                "AdaRoundQuantizer",
                "MinMaxQuantizer",
                "AbsMaxQuantizer",
            ]:
                self.Quantizer = quantizerDict[args.get("scheme")](
                    org_tensor=org_tensor, args=args
                )
            else:
                raise Exception(
                    "Only MinMaxQuantizer is supported for weight quantization."
                )
        else:
            if args.get("scheme") in [
                "DynamicMinMaxQuantizer",
                # "MovingAvgAbsMaxQuantizer",
                "MovingAvgMinMaxQuantizer",
            ]:
                self.Quantizer = quantizerDict[args.get("scheme")](
                    org_tensor=None, args=args
                )
            else:
                raise Exception(
                    "Only DynamicMinMaxQuantizer, MovingAvgMinMaxQuantizer are supported for activation quantization."
                )
        self.__inited = True

    def forward(self, input: Tensor) -> Tensor:
        return self.Quantizer(input) if self.do_quant else input


class QuantLinearLayer(nn.Module):
    def __init__(self, orgModule):
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

        self.weightQuantizer = QuantBase()
        self.activationQuantizer = QuantBase()

    def forward(self, input: torch.Tensor):
        _weight = self.weightQuantizer(self.weight)

        x = self.fwd_func(input, _weight, self.bias, **self.fwd_kwargs)

        return self.activationQuantizer(x)


class QuantActOnly(nn.Module):
    def __init__(self):
        super().__init__()
        self.activationQuantizer = QuantBase()

    def forward(self, input: torch.Tensor):
        return self.activationQuantizer(input)


class QuantLayerNorm(nn.Module):
    def __init__(self, orgLayerNorm):
        super().__init__()

        self.normalized_shape = orgLayerNorm.normalized_shape
        self.weight = orgLayerNorm.weight.clone().detach()
        self.bias = orgLayerNorm.bias.clone().detach()
        self.eps = orgLayerNorm.eps

        self.using_int_ln = False
        self.int_ln_bit_width = None

    def forward(self, input: torch.Tensor):
        if self.using_int_ln:
            # with torch.device(input.device):
            #     """
            #     여기 input이 어차피 8비트인 FP라서, 32비트로하건 8비트로하건 어차피 최대 8비트 정밀도임.
            #     """
            #     # the X is quantized value but represent by FP domain.
            #     S = torch.max(torch.abs(input)) / (2 ** (self.int_ln_bit_width - 1) - 1)
            #     x_q = torch.clip(
            #         torch.round(input / S),
            #         -(2 ** (self.int_ln_bit_width - 1)),
            #         2 ** (self.int_ln_bit_width - 1) - 1,
            #     )
            #     # apply the temporary sysmetric quantization
            #     x_q, _S = int_LayerNorm(x_q, S, self.weight, self.bias, self.eps)
            #     x_fp = x_q * _S
            #     # dequant -> the quantized value represent by FP domain
            #     return x_fp
            return F.layer_norm(
                input, self.normalized_shape, self.weight, self.bias, self.eps
            )
        else:
            return F.layer_norm(
                input, self.normalized_shape, self.weight, self.bias, self.eps
            )

import torch, math
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.vision_transformer import Encoder, EncoderBlock, MLPBlock
from collections import OrderedDict
from torch import Tensor
from typing import Optional, Tuple

""" Quantizer """


class UniformSymmetricQuantizer(nn.Module):
    def __init__(self, org_tensor, args):
        super().__init__()
        """
        example... 
            args = {
                "scheme": "AbsMaxQuantizer",
                "bit_width": 8,
                "per_channel": False,
            }
        """
        bit_width = args.get("bit_width")
        assert bit_width in [4, 8, 16, 32]

        self._n_bits = int(bit_width)  # "INT8" -> 8
        self._repr_min, self._repr_max = None, None

        self.per_channel = args.get("per_channel") if args.get("per_channel") else False
        self._n_ch = len(org_tensor.size()) if self.per_channel else 1
        self._scaler = None
        # self._zero_point = None
        self.one_side_dist = None  # {"pos", "neg", "no"}

        if org_tensor is not None:
            self._define_repr_min_max(org_tensor)

    def _define_repr_min_max(self, input: Tensor):
        if self.one_side_dist is None:
            self.one_side_dist = (
                "pos" if input.min() >= 0.0 else "neg" if input.max() <= 0.0 else "no"
            )
        if self.one_side_dist != "no":
            self._repr_min = 0  # "UINT8" -> 0
            self._repr_max = 2 ** (self._n_bits) - 1  # "UINT8" -> 255
            # print(f"1D search with UINT{self._n_bits}", end=" - \n")
        else:  # 2-d search
            self._repr_min = -(2 ** (self._n_bits - 1))  # "INT8" -> -128
            self._repr_max = 2 ** (self._n_bits - 1) - 1  # "INT8" -> 127
            # print(f"2D search with INT{self._n_bits}", end=" - \n")

    def round_ste(self, input: torch.Tensor):
        """
        Implement Straight-Through Estimator for rounding operation.
        """
        return (input.round() - input).detach() + input

    def compute_qparams(self, _min, _max) -> None:
        # Origin tensor shape: [out_channel, in_channel, k, k]
        # per_ch Qparam shape: [n_channel, 1, 1, 1]

        scaler = (_max - _min) / (self._repr_max - self._repr_min)
        self._scaler = scaler.view(-1, *([1] * (self._n_ch - 1)))

        # _min = _min.view(-1, *([1] * (self._n_ch - 1)))
        # self._zero_point = -self.round_ste(_min / self._scaler) + self._repr_min

    def _quantize(self, input: Tensor) -> Tensor:
        return torch.clamp(
            self.round_ste(input / self._scaler),
            self._repr_min,
            self._repr_max,
        )

    def _dequantize(self, input: Tensor) -> Tensor:
        return input * self._scaler

    def forward(self, x: Tensor) -> Tensor:
        return self._dequantize(self._quantize(x))


class AbsMaxQuantizer(UniformSymmetricQuantizer):
    def __init__(self, org_tensor, args):
        """
        [ Absolute Maximum Quantization ]
        - When zero_point is zero, the quantization range is symmetric.
            (Uniform Symmetric Quantization)
        - range: [-max(abs(x)), max(abs(x))]
        """
        super(AbsMaxQuantizer, self).__init__(org_tensor, args)

        _AbsMax = None
        if self.per_channel == True:
            _AbsMax = org_tensor.view(org_tensor.size(0), -1).abs().max(dim=1).values
        else:
            _AbsMax = org_tensor.abs().max()

        if self.one_side_dist == "no":
            # if s8, scaler = 2 * org_weight.abs().max() / (127 - (-128))
            #               = org_weight.abs().max() - (-org_weight.abs().max()) / (127 - (-128))
            self.compute_qparams(-_AbsMax, _AbsMax)
            # will convert to (-max, max) -> (-128, 127)
        elif self.one_side_dist == "pos":
            # if u8, scaler = org_weight.abs().max() / (255 - 0)
            self.compute_qparams(torch.zeros_like(_AbsMax), _AbsMax)
            # will convert to (0, max) -> (0, 255)
        else:
            print("This distribution is unexpected. plaese check the input data.")
            # self.compute_qparams(_AbsMax, torch.zeros_like(_AbsMax))
            raise ValueError("Unknown distribution type.")


class MovAvgAbsMaxQuantizer(UniformSymmetricQuantizer):
    def __init__(self, org_tensor=None, args=None):
        """
        DESIGNED FOR PER-LAYER SCHEME.
        """
        assert org_tensor == None, "MovAvgAbsMaxQuantizer should not have org_tensor."
        super().__init__(org_tensor=None, args=args)

        # 0.9 is default value in the white paper.
        self._m = args.get("momentum") if args.get("momentum") else 0.9
        self._batches = args.get("batches") if args.get("batches") else 16
        self.per_channel = True

        self._ready_to_quantize = False
        self._computed_batches_cnt = 0
        self._threshold_batches = int((1 / (1 - self._m)))
        self.first_n_times_min = []
        self.first_n_times_max = []
        self._min, self._max = None, None

    def _update_moving_avg(self, input):
        with torch.no_grad():
            """calibration process"""
            if self._computed_batches_cnt < self._threshold_batches:
                ## >> save first n times min, max
                self._computed_batches_cnt += 1
                self.first_n_times_min.append(input.min())
                self.first_n_times_max.append(input.max())

            elif self._computed_batches_cnt == self._threshold_batches:
                ## >> compute the mean of first n times min, max
                self._computed_batches_cnt += 1
                assert len(self.first_n_times_min) == self._threshold_batches
                self._min = torch.tensor(self.first_n_times_min).mean()
                self._max = torch.tensor(self.first_n_times_max).mean()

            elif self._computed_batches_cnt > self._threshold_batches:
                ## >> update min, max with momentum ( moving average )
                self._computed_batches_cnt += 1
                self._min = self._min * self._m + (1 - self._m) * input.min()
                self._max = self._max * self._m + (1 - self._m) * input.max()

            """ determine the quantization parameter """
            if self._computed_batches_cnt == self._batches:
                if self._min < 0 and self._max > 0:
                    self.one_side_dist = "no"
                elif self._min >= 0:
                    self.one_side_dist = "pos"
                elif self._max <= 0:
                    self.one_side_dist = "neg"
                else:
                    raise ValueError("Unknown distribution type.")
                # one_side_dist is determined.
                # -> we can send the tmp value such as None.
                self._define_repr_min_max(None)

                # if 16 == 16, calibration is done.
                self.compute_qparams(self._min, self._max)
                self._ready_to_quantize = True

    def forward(self, input: Tensor) -> Tensor:
        if self._ready_to_quantize == False:
            self._update_moving_avg(input)
            return input
        else:
            return super().forward(input)

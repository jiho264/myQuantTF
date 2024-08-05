import torch, tqdm
from torch import Tensor


#################################################################################################
## 2. Quantized module
#################################################################################################

import torch.nn as nn
import torch.nn.functional as F


class UniformAffineQuantizer(nn.Module):
    def __init__(self, org_tensor, args):
        super(UniformAffineQuantizer, self).__init__()
        """
        example... 
            args = {
                "scheme": "AbsMaxQuantizer",
                "bit_width": 8,
            }
        """
        bit_width = args.get("bit_width")
        assert bit_width in [4, 8, 16, 32]

        self._n_bits = int(bit_width)  # "INT8" -> 8
        self._repr_min, self._repr_max = None, None

        # self.per_channel = args.get("per_channel") if args.get("per_channel") else False
        # always using per_channel quantization
        self.per_channel = True
        self._n_ch = len(org_tensor.size()) if self.per_channel else 1
        self._scaler = None
        self._zero_point = None
        self.one_side_dist = None  # {"pos", "neg", "no"}

        self._define_repr_min_max(org_tensor)

    def _define_repr_min_max(self, input: Tensor):
        if self.one_side_dist is None:
            self.one_side_dist = (
                "pos" if input.min() >= 0.0 else "neg" if input.max() <= 0.0 else "no"
            )
        if self.one_side_dist != "no":
            self._repr_min = 0  # "UINT8" -> 0
            self._repr_max = 2 ** (self._n_bits) - 1  # "UINT8" -> 255
            print(f"    1D search with UINT{self._n_bits}")
        else:  # 2-d search
            self._repr_min = -(2 ** (self._n_bits - 1))  # "INT8" -> -128
            self._repr_max = 2 ** (self._n_bits - 1) - 1  # "INT8" -> 127
            print(f"    2D search with INT{self._n_bits}")

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

        _min = _min.view(-1, *([1] * (self._n_ch - 1)))
        self._zero_point = -self.round_ste(_min / self._scaler) + self._repr_min

    def _quantize(self, input: Tensor) -> Tensor:
        return torch.clamp(
            self.round_ste(input / self._scaler) + self._zero_point,
            self._repr_min,
            self._repr_max,
        )

    def _dequantize(self, input: Tensor) -> Tensor:
        if torch.all(input == input.floor()).item():
            return (input - self._zero_point) * self._scaler
        else:
            """ verifing the quantization """
            raise ValueError("The input tensor is not quantized.")

    def forward(self, x: Tensor) -> Tensor:
        return self._dequantize(self._quantize(x))


class AbsMaxQuantizer(UniformAffineQuantizer):
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

        # if s8 or u8, zero_point = 0
        self.zero_point = torch.zeros_like(self._scaler)


class MinMaxQuantizer(UniformAffineQuantizer):
    def __init__(self, org_tensor, args):
        """
        [ Min-Max Quantization ]
        - The quantization range is asymmetric.
        - range: [min(x), max(x)]
        - https://pytorch.org/docs/stable/generated/torch.ao.quantization.observer.MinMaxObserver
        """
        super(MinMaxQuantizer, self).__init__(org_tensor, args)

        if self.per_channel == True:
            _min = org_tensor.view(org_tensor.size(0), -1).min(dim=1).values
            _max = org_tensor.view(org_tensor.size(0), -1).max(dim=1).values
        else:
            _min = org_tensor.min()
            _max = org_tensor.max()

        # if s8, scaler = (_max - _min) / (127 - (-128))
        # if s8, zero_point = -_min / scaler + (-128)

        # if u8, scaler = (_max - _min) / (255 - 0)
        # if u8, zero_point = -_min / scaler + 0

        # Always using same equation.
        self.compute_qparams(_min, _max)

class DynamicMinMaxQuantizer(UniformAffineQuantizer):
    def __init__(self, org_tensor, args):
        assert org_tensor == None, "DynamicMinMaxQuantizer should not have org_tensor."
        super().__init__(org_tensor=torch.randn((128, 197, 768)),args=args)
        # self.per_channel == always True

    def _define_repr_min_max(self, input: Tensor):
        if self.one_side_dist is None:
            self.one_side_dist = (
                "pos" if input.min() >= 0.0 else "neg" if input.max() <= 0.0 else "no"
            )
        if self.one_side_dist != "no":
            self._repr_min = 0  # "UINT8" -> 0
            self._repr_max = 2 ** (self._n_bits) - 1  # "UINT8" -> 255
            # print(f"    1D search with UINT{self._n_bits}")
        else:  # 2-d search
            self._repr_min = -(2 ** (self._n_bits - 1))  # "INT8" -> -128
            self._repr_max = 2 ** (self._n_bits - 1) - 1  # "INT8" -> 127
            # print(f"    2D search with INT{self._n_bits}")

    def forward(self, input):
        self._define_repr_min_max(input)
        self._n_ch = len(input.size()) if self.per_channel else 1

        if self.per_channel == True:
            _min = input.view(input.size(0), -1).min(dim=1).values
            _max = input.view(input.size(0), -1).max(dim=1).values
        else:
            _min = input.min()
            _max = input.max()
        self.compute_qparams(_min, _max)

        return super().forward(input)


quantizerDict = {
    "AbsMaxQuantizer": AbsMaxQuantizer,
    "MinMaxQuantizer": MinMaxQuantizer,
    "DynamicMinMaxQuantizer": DynamicMinMaxQuantizer,
    # "NormQuantizer": NormQuantizer,
    # "OrgNormQuantizerCode": OrgNormQuantizerCode,
}


class StraightThrough(nn.Module):
    def __int__(self):
        super().__init__()

    def forward(self, input):
        return input

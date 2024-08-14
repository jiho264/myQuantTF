"""

From I-ViT

Li, Zhikai, and Qingyi Gu. "I-vit: Integer-only quantization for efficient vision transformer inference." Proceedings of the IEEE/CVF International Conference on Computer Vision. 2023.


https://github.com/zkkli/I-ViT

"""

from .quant_modules import (
    QuantLinear,
    QuantAct,
    QuantConv2d,
    QuantMatMul,
    IntLayerNorm,
    IntSoftmax,
    IntGELU,
)

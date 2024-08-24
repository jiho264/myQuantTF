import torch, math
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.vision_transformer import Encoder, EncoderBlock, MLPBlock
from collections import OrderedDict
from torch import Tensor
from typing import Optional, Tuple

from .quant_blocks import QuantMLP, QuantMSA
from .quant_modules import (
    IntLayerNorm,
    QuantLinearWithWeight,
    QuantAct,
    IntGELU,
    IntSoftMax,
    QuantMatMul,
)


class QuantEncoderBlock(nn.Module):
    """Transformer encoder block."""

    def __init__(self, orgModule: EncoderBlock, **kwargs):
        super().__init__()
        self.num_heads = orgModule.num_heads

        # [1-1] nn.LayerNorm : LayerNorm((768,), eps=1e-06, elementwise_affine=True)
        self.ln_1 = IntLayerNorm(orgModule.ln_1, args_ln=kwargs["args_ln"])
        self.ln_1_act = QuantAct(args_a=kwargs["args_a"])
        # [1-2] nn.MultiheadAttention
        self.self_attention = QuantMSA(
            orgModule.self_attention,
            args_w=kwargs["args_w"],
            args_a=kwargs["args_a"],
            args_softmax=kwargs["args_softmax"],
        )
        # [1-3] ID Addition
        self.idAdd_1 = QuantAct(args_a=kwargs["args_a"], which="idAdd")

        # nn.Dropout : Dropout(p=0.0, inplace=False)
        self.dropout = orgModule.dropout

        # [2-1] nn.LayerNorm : LayerNorm((768,), eps=1e-06, elementwise_affine=True)
        self.ln_2 = IntLayerNorm(orgModule.ln_2, args_ln=kwargs["args_ln"])
        self.ln_2_act = QuantAct(args_a=kwargs["args_a"])
        # [2-2] nn.MLPBlock
        self.mlp = QuantMLP(
            orgModule.mlp,
            args_w=kwargs["args_w"],
            args_a=kwargs["args_a"],
            args_gelu=kwargs["args_gelu"],
        )
        # [2-3] ID Addition
        self.idAdd_2 = QuantAct(args_a=kwargs["args_a"], which="idAdd")

    def forward(self, x_hat, s_x):
        torch._assert(
            x_hat.dim() == 3,
            f"Expected (batch_size, seq_length, hidden_dim) got {x_hat.shape}",
        )

        x1, s_x1 = self.ln_1(x_hat, s_x)
        x1, s_x1 = self.ln_1_act(x1, s_x1)
        x1, s_x1 = self.self_attention(x1, s_x1)
        x1 = self.dropout(x1)
        x1, s_x1 = self.idAdd_1(x1, s_x1, x_hat, s_x)

        x2, s_x2 = self.ln_2(x1, s_x1)
        x2, s_x2 = self.ln_2_act(x2, s_x2)
        x2, s_x2 = self.mlp(x2, s_x2)
        x2, s_x2 = self.idAdd_2(x2, s_x2, x1, s_x1)

        return x2, s_x2


class QuantEncoder(nn.Module):
    def __init__(self, orgModule: Encoder, **kwargs):
        super().__init__()
        self.pos_embedding = orgModule.pos_embedding
        self.pos_embedding_act = QuantAct(args_a=kwargs["args_a"])

        self.idAdd_pos_cls_x = QuantAct(args_a=kwargs["args_a"], which="idAdd")

        self.dropout = orgModule.dropout
        self.layers = nn.ModuleList()

        for orgEncoderBlock in orgModule.layers:
            self.layers.append(QuantEncoderBlock(orgModule=orgEncoderBlock, **kwargs))
        self.ln = IntLayerNorm(orgModule.ln, args_ln=kwargs["args_ln"])

    def forward(self, x_hat, s_x):
        torch._assert(
            x_hat.dim() == 3,
            f"Expected (batch_size, seq_length, hidden_dim) got {x_hat.shape}",
        )

        pos, s_pos = self.pos_embedding_act(self.pos_embedding)

        x_hat, s_x = self.idAdd_pos_cls_x(x_hat, s_x, pos, s_pos)
        # This x_hat is not INT8.
        # Represent by higher bit such as INT16

        x_hat = self.dropout(x_hat)

        for layer in self.layers:
            x_hat, s_x = layer(x_hat, s_x)

        x_hat, s_x = self.ln(x_hat, s_x)

        return x_hat, s_x


class QuantViT(nn.Module):
    def __init__(
        self,
        orgViT,
        args_w: dict = {},
        args_a: dict = {},
        args_softmax: dict = {},
        args_ln: dict = {},
        args_gelu: dict = {},
    ):
        super().__init__()
        self.patch_size = orgViT.patch_size
        self.image_size = orgViT.image_size
        self.hidden_dim = orgViT.hidden_dim
        self.class_token = orgViT.class_token
        self.representation_size = orgViT.representation_size

        """ [1] define the quantized modules """
        self.input_act = QuantAct(args_a=args_a)

        self.conv_proj = QuantLinearWithWeight(
            orgModule=orgViT.conv_proj, args_w=args_w, args_a=args_a
        )

        self.encoder = QuantEncoder(
            orgModule=orgViT.encoder,
            args_w=args_w,
            args_a=args_a,
            args_ln=args_ln,
            args_softmax=args_softmax,
            args_gelu=args_gelu,
        )
        self.encoder_ln_act = QuantAct(args_a=args_a)

        heads_layers = []

        if self.representation_size is None:
            heads_layers.append(
                QuantLinearWithWeight(orgModule=orgViT.heads.head, args_w=args_w)
            )
        # else:
        #     heads_layers.append(nn.Linear(self.hidden_dim, self.representation_size))
        #     heads_layers.append(nn.Tanh())
        #     heads_layers.append(nn.Linear(self.representation_size, 1000))
        #     pass

        self.heads = nn.ModuleList(heads_layers)

        """ [2] Quant setting """
        self.vit_w_quant = False if args_w == {} else True
        self.vit_a_quant = False if args_a == {} else True
        self.vit_int_ln = False if args_ln == {} else True
        self.vit_int_gelu = False if args_gelu == {} else True
        self.vit_int_softmax = False if args_softmax == {} else True

    def set_quant_mode(self, for_w, for_a, for_int_ln, for_int_gelu, for_int_softmax):
        self.vit_w_quant = for_w
        self.vit_a_quant = for_a
        self.vit_int_ln = for_int_ln
        self.vit_int_gelu = for_int_gelu
        self.vit_int_softmax = for_int_softmax
        # print(
        #     f"Quant mode set to: {self.vit_w_quant}, {self.vit_a_quant}, {self.vit_int_ln}, {self.vit_int_gelu}, {self.vit_int_softmax}"
        # )

    def _confirm_quant_mode(self):
        for name, module in self.named_modules():
            if isinstance(module, QuantLinearWithWeight):
                module.do_quant = self.vit_w_quant
            elif isinstance(module, QuantAct):
                module.do_quant = self.vit_a_quant
            elif isinstance(module, IntLayerNorm):
                module.do_quant = self.vit_int_ln
            elif isinstance(module, IntGELU):
                module.do_quant = self.vit_int_gelu
            elif isinstance(module, IntSoftMax):
                module.do_quant = self.vit_int_softmax

    def _process_input(self, x_hat_int8, s_x):
        n, c, h, w = x_hat_int8.shape
        p = self.patch_size
        torch._assert(
            h == self.image_size,
            f"Wrong image height! Expected {self.image_size} but got {h}!",
        )
        torch._assert(
            w == self.image_size,
            f"Wrong image width! Expected {self.image_size} but got {w}!",
        )
        n_h = h // p
        n_w = w // p

        # (n, c, h, w) -> (n, hidden_dim, n_h, n_w)
        x_hat_int8, s_x = self.conv_proj(x_hat_int8, s_x)
        # (n, hidden_dim, n_h, n_w) -> (n, hidden_dim, (n_h * n_w))
        x_hat_int8 = x_hat_int8.reshape(n, self.hidden_dim, n_h * n_w)

        # (n, hidden_dim, (n_h * n_w)) -> (n, (n_h * n_w), hidden_dim)
        # The self attention layer expects inputs in the format (N, S, E)
        # where S is the source sequence length, N is the batch size, E is the
        # embedding dimension
        x_hat_int8 = x_hat_int8.permute(0, 2, 1)

        return x_hat_int8, s_x

    def forward(self, input: torch.Tensor):
        self._confirm_quant_mode()
        # QuantAct : 149
        # QuantLinearWithWeight : 50
        # IntLayerNorm : 25
        # IntSoftMax : 12
        # QuantMatMul : 24
        # IntGELU : 12

        """ [stem] """
        # Reshape and permute the input tensor
        x_hat_int8, s_x = self.input_act(input)
        x_hat_int8, s_x = self._process_input(x_hat_int8, s_x)
        n = x_hat_int8.shape[0]

        # Expand the class token to the full batch
        batch_class_token = self.class_token.expand(n, -1, -1)
        x_hat_int8 = torch.cat([batch_class_token, x_hat_int8], dim=1)

        """ [encoder blocks] """
        x_hat_int8, s_x = self.encoder(x_hat_int8, s_x)

        # Classifier "token" as used by standard language architectures
        x_hat_int8 = x_hat_int8[:, 0]
        x_hat_int8, s_x = self.encoder_ln_act(x_hat_int8, s_x)

        """ [heads] """
        for layer in self.heads:
            x_hat_int8, s_x = layer(x_hat_int8, s_x)

        return x_hat_int8


"""
=Title==============================================================================================
ViT-base 16

-info-----------------------------------------------------------------------------------------------
There are 25 LayerNorms in ViT-base 16

-Arch-----------------------------------------------------------------------------------------------
>>> input.shape: torch.Size([1, 3, 224, 224])
conv_proj
>>> output.shape: torch.Size([1, 197, 768])
[Encoder -> QuantEncoder] encoder
    encoder.dropout
    encoder.layers
        [EncoderBlock] encoder.layers.encoder_layer_0
        [ INPUT -> LN -> MSA -> DROPOUT -> LN -> MLP -> OUTPUT ]
            [LayerNorm] encoder.layers.encoder_layer_0.ln_1
                >>> output.shape: torch.Size([1, 197, 768])
                LayerNorm((768,), eps=1e-06, elementwise_affine=True)
            [MultiheadAttention] encoder.layers.encoder_layer_0.self_attention
                >>> input.shape: torch.Size([12, 197, 64]) # <- 12 heads
                encoder.layers.encoder_layer_0.self_attention.out_proj
                    (out_proj): NonDynamicallyQuantizableLinear(in_features=768, out_features=768, bias=True)
                >>> output.shape: torch.Size([1, 197, 768])
            [Dropout] encoder.layers.encoder_layer_0.dropout
                Dropout(p=0.0, inplace=False)
                >>> output.shape: torch.Size([1, 197, 768]) + residual additon
            [LayerNorm] encoder.layers.encoder_layer_0.ln_2
                LayerNorm((768,), eps=1e-06, elementwise_affine=True)
                >>> output.shape: torch.Size([1, 197, 768])
            [MLPBlock] encoder.layers.encoder_layer_0.mlp
                [Linear] encoder.layers.encoder_layer_0.mlp.0
                    (0): Linear(in_features=768, out_features=3072, bias=True)
                    >>> output.shape: torch.Size([1, 197, 3072])
                [GELU] encoder.layers.encoder_layer_0.mlp.1
                    (1): GELU(approximate='none')
                    >>> output.shape: torch.Size([1, 197, 3072])
                [Dropout] encoder.layers.encoder_layer_0.mlp.2
                    (2): Dropout(p=0.0, inplace=False)
                    >>> output.shape: torch.Size([1, 197, 3072])
                [Linear] encoder.layers.encoder_layer_0.mlp.3
                    (3): Linear(in_features=3072, out_features=768, bias=True)
                    >>> output.shape: torch.Size([1, 197, 768])
                [Dropout] encoder.layers.encoder_layer_0.mlp.4
                    (4): Dropout(p=0.0, inplace=False)
                    >>> output.shape: torch.Size([1, 197, 768])
            >>> output.shape: torch.Size([1, 197, 768]) + residual additon
            
        More 11 Encoders {1 ~ 11}...
        >>> output.shape: torch.Size([1, 197, 768])
            
    encoder.ln # <- 25 LayerNorms
        LayerNorm((768,), eps=1e-06, elementwise_affine=True)
        >>> output.shape: torch.Size([1, 197, 768])
heads
    [head] heads.head
        Linear(in_features=768, out_features=1000, bias=True)
        >>> output.shape: torch.Size([1, 1000])
        

w4a32 79.834 / my old code 79.834%
w4a8 79.638 / my old code 79.756% without input, cls, pos quantization


weiht quantizer : 50
- conv_proj : 1
- MSA * 2 : 24
- MLP * 2 : 24
- head : 1

activation quantizer : 76
- input_act : 1
- conv_proj_act : 1
- cls_token_act : 1
- pos_act : 1
- LN : 25 (2*24 + 1)
- LN ID add : 24 ( 2*12 )
- MSA * 4 : 48 ( inproj, qkv, softmax, outproj )
- MLP * 3 : 36 ( linear, gelu , linear)
- head : None

# 
"""
# Number of QuantAct layers: 137
# Number of QuantLinear layers: 49
# Number of QuantConv2d layers: 1
# Number of IntLayerNorm layers: 25
# Number of IntSoftmax layers: 12
# Number of QuantMatMul layers: 24
# Number of IntGELU layers: 12
# """
# quantLinaerWeightcnt = 0
# quantactcnt = 0
# quantgelucnt = 0
# quantsoftmaxcnt = 0
# quantlncnt = 0
# quantmmcnt = 0
# for name, module in self.named_modules():
#     if isinstance(module, QuantLinearWithWeight):
#         quantLinaerWeightcnt += 1
#     elif isinstance(module, QuantAct):
#         quantactcnt += 1
#         # print(module.activation_bit)
#     elif isinstance(module, IntLayerNorm):
#         quantlncnt += 1
#     elif isinstance(module, IntGELU):
#         quantgelucnt += 1
#     elif isinstance(module, IntSoftMax):
#         quantsoftmaxcnt += 1
#     elif isinstance(module, QuantMatMul):
#         quantmmcnt += 1
# print(f"QuantAct : {quantactcnt}")
# print(f"QuantLinearWithWeight : {quantLinaerWeightcnt}")
# print(f"IntLayerNorm : {quantlncnt}")
# print(f"IntSoftMax : {quantsoftmaxcnt}")
# print(f"QuantMatMul : {quantmmcnt}")
# print(f"IntGELU : {quantgelucnt}")
# exit()

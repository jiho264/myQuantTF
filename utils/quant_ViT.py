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
)


class QuantEncoderBlock(nn.Module):
    """Transformer encoder block."""

    def __init__(self, orgModule: EncoderBlock, **kwargs):
        super().__init__()
        self.num_heads = orgModule.num_heads

        ## nn.LayerNorm : LayerNorm((768,), eps=1e-06, elementwise_affine=True)
        self.ln_1 = IntLayerNorm(orgModule.ln_1, args_ln=kwargs["args_ln"])

        ## nn.MultiheadAttention
        self.self_attention = QuantMSA(
            orgModule.self_attention,
            args_w=kwargs["args_w"],
            args_a=kwargs["args_a"],
            args_softmax=kwargs["args_softmax"],
        )

        ## nn.Dropout : Dropout(p=0.0, inplace=False)
        self.dropout = orgModule.dropout

        ## nn.LayerNorm : LayerNorm((768,), eps=1e-06, elementwise_affine=True)
        self.ln_2 = IntLayerNorm(orgModule.ln_2, args_ln=kwargs["args_ln"])

        ## MLPBlock
        self.mlp = QuantMLP(
            orgModule.mlp,
            args_w=kwargs["args_w"],
            args_a=kwargs["args_a"],
            args_gelu=kwargs["args_gelu"],
        )

    def forward(self, x, s_x):
        torch._assert(
            x.dim() == 3,
            f"Expected (batch_size, seq_length, hidden_dim) got {x.shape}",
        )
        x1, s_x1 = self.ln_1(x, s_x)
        x1, s_x1 = self.self_attention(x1, s_x1)
        x1 = self.dropout(x1)
        x1 += x

        x2, s_x2 = self.ln_2(x1, s_x1)
        x2, s_x2 = self.mlp(x2, s_x2)
        x2 += x1
        return x2, s_x2


class QuantEncoder(nn.Module):
    def __init__(self, orgModule: Encoder, **kwargs):
        super().__init__()
        self.pos_embedding = orgModule.pos_embedding
        self.pos_act = QuantAct(args_a=kwargs["args_a"])
        self.dropout = orgModule.dropout
        self.layers = nn.ModuleList()

        for orgEncoderBlock in orgModule.layers:
            self.layers.append(QuantEncoderBlock(orgModule=orgEncoderBlock, **kwargs))
        self.ln = IntLayerNorm(orgModule.ln, args_ln=kwargs["args_ln"])

    def forward(self, x, s_x):
        torch._assert(
            x.dim() == 3,
            f"Expected (batch_size, seq_length, hidden_dim) got {x.shape}",
        )
        pos, s_pos = self.pos_act(self.pos_embedding)
        x = x + pos

        x = self.dropout(x)
        for layer in self.layers:
            x, s_x = layer(x, s_x)
        x, s_x = self.ln(x, s_x)
        return x, s_x


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
        print(f"Weight quantization parameter : {args_w}")
        print(f"Activation quantization parameter : {args_a}")
        print(f"Softmax quantization parameter : {args_softmax}")
        print(f"LayerNorm quantization parameter : {args_ln}")
        print(f"GELU quantization parameter : {args_gelu}")

        self.patch_size = orgViT.patch_size
        self.image_size = orgViT.image_size
        self.hidden_dim = orgViT.hidden_dim
        self.class_token = orgViT.class_token
        self.representation_size = orgViT.representation_size
        """ [1] define the quantized modules """
        self.input_act = QuantAct(args_a=args_a)
        self.conv_proj = QuantLinearWithWeight(
            orgModule=orgViT.conv_proj, args_w=args_w
        )
        self.conv_proj_act = QuantAct(args_a=args_a)
        self.cls_token_act = QuantAct(args_a=args_a)

        self.encoder = QuantEncoder(
            orgModule=orgViT.encoder,
            args_w=args_w,
            args_a=args_a,
            args_ln=args_ln,
            args_softmax=args_softmax,
            args_gelu=args_gelu,
        )
        heads_layers = []

        if self.representation_size is None:
            heads_layers.append(
                QuantLinearWithWeight(orgModule=orgViT.heads.head, args_w=args_w)
            )
        else:
            # heads_layers.append(nn.Linear(hidden_dim, representation_size))
            # heads_layers.append(nn.Tanh())
            # heads_layers.append(nn.Linear(representation_size, num_classes))
            pass

        self.heads = nn.ModuleList(heads_layers)

        self.vit_w_quant = False if args_w == {} else True
        self.vit_a_quant = False if args_a == {} else True
        self.vit_int_ln = False if args_ln == {} else True
        self.vit_int_gelu = False if args_gelu == {} else True
        self.vit_int_softmax = False if args_softmax == {} else True

    def _quant_switch(self):
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

    def _process_input(self, x, s_x):
        n, c, h, w = x.shape
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
        x, s_x = self.conv_proj(x, s_x)
        x, s_x = self.conv_proj_act(x, s_x)
        # (n, hidden_dim, n_h, n_w) -> (n, hidden_dim, (n_h * n_w))
        x = x.reshape(n, self.hidden_dim, n_h * n_w)

        # (n, hidden_dim, (n_h * n_w)) -> (n, (n_h * n_w), hidden_dim)
        # The self attention layer expects inputs in the format (N, S, E)
        # where S is the source sequence length, N is the batch size, E is the
        # embedding dimension
        x = x.permute(0, 2, 1)

        return x, s_x

    def forward(self, x: torch.Tensor):
        self._quant_switch()
        # Reshape and permute the input tensor
        x, s_x = self.input_act(x)
        x, s_x = self._process_input(x, s_x)
        n = x.shape[0]

        # Expand the class token to the full batch
        batch_class_token = self.class_token.expand(n, -1, -1)
        x = torch.cat([batch_class_token, x], dim=1)

        x, s_x = self.cls_token_act(x, s_x)

        x, s_x = self.encoder(x, s_x)

        # Classifier "token" as used by standard language architectures
        x = x[:, 0]

        for layer in self.heads:
            x, s_x = layer(x, s_x)

        return x


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
        

w4a32 79.834
w4a8 79.638
        
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
- MSA * 4 : 48
- MLP * 2 : 24
- head : None

=================================================================================================
"""

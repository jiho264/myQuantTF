import torch, math
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.vision_transformer import Encoder, EncoderBlock, MLPBlock

from collections import OrderedDict
from typing import Optional, Tuple

import torch
from torch import Tensor


from .quantizer import quantizerDict


class QuantBaseModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.w_quant_switch = False
        self.a_quant_switch = False
        self.w_inited = False
        self.a_inited = False
        self.weight_quantizer = None
        self.activation_quantizer = None

    def init_weight_quantizer(self, org_tensor, args):
        if self.w_inited == True:
            raise Exception("Weight quantizer is already inited.")
        self.weight_quantizer = quantizerDict[args.get("scheme")](
            org_tensor=org_tensor, args=args
        )
        self.w_inited = True

    def init_activation_quantizer(self, org_tensor, args):
        if self.a_inited == True:
            raise Exception("Activation quantizer is already inited.")
        self.activation_quantizer = quantizerDict[args.get("scheme")](
            org_tensor=org_tensor, args=args
        )
        self.a_inited = True

    def forward_weight_quantizer(self, input):
        if self.w_quant_switch == True:
            return self.weight_quantizer(input)
        else:
            return input

    def forward_activation_quantizer(self, input):
        if self.a_quant_switch == True:
            return self.activation_quantizer(input)
        else:
            return input


# class QuantMLPBlock(QuantBaseModule):
class QuantMLPBlock(nn.Module):
    def __init__(
        self,
        orgMLPBlock: MLPBlock,
    ):
        super().__init__()
        """
        MLPBlock(
            (0): Linear(in_features=768, out_features=3072, bias=True)
            (1): GELU(approximate='none')
            (2): Dropout(p=0.0, inplace=False)
            (3): Linear(in_features=3072, out_features=768, bias=True)
            (4): Dropout(p=0.0, inplace=False)
            )
        """
        self.linear_1 = QuantLayer(orgMLPBlock.get_submodule("0"))
        # print(self.linear_1.weight.shape)  # torch.Size([3072, 768])
        self.gelu = orgMLPBlock.get_submodule("1")
        self.dropout_1 = orgMLPBlock.get_submodule("2")
        self.linear_2 = QuantLayer(orgMLPBlock.get_submodule("3"))
        # print(self.linear_2.weight.shape)  # torch.Size([768, 3072])
        self.dropout_2 = orgMLPBlock.get_submodule("4")
        print("    - QuantMLPBlock is created")

    def forward(self, input: torch.Tensor):

        x = self.linear_1(input)
        x = self.gelu(x)
        x = self.dropout_1(x)
        x = self.linear_2(x)
        x = self.dropout_2(x)
        return x


class QuantMultiheadAttention(nn.MultiheadAttention, QuantBaseModule):
    def __init__(self, orgMultiheadAttention):
        super().__init__(
            orgMultiheadAttention.embed_dim,
            orgMultiheadAttention.num_heads,
            dropout=orgMultiheadAttention.dropout,
            batch_first=orgMultiheadAttention.batch_first,
        )
        self.__dict__.update(orgMultiheadAttention.__dict__)

        print("    - QuantMultiheadAttention is created")

        self.ORG_MSA_FORWARD_FUNC = torch._native_multi_head_attention

    def QuantMSA(self, *args, **kwargs):
        """
        return torch._native_multi_head_attention(
            [0] query, >> shape : torch.Size([batch, 197, 768])
            [1] key,   >> shape : torch.Size([batch, 197, 768])
            [2] value, >> shape : torch.Size([batch, 197, 768])
            [3] self.embed_dim, >> 768
            [4] self.num_heads, >> 12
            [5] self.in_proj_weight, >> shape : torch.Size([2304, 768])
            [6] self.in_proj_bias,   >> shape : torch.Size([2304])
            [7] self.out_proj.weight, >> shape : torch.Size([768, 768])
            [8] self.out_proj.bias,   >> shape : torch.Size([768])
            [9] merged_mask, >> None
            [10] need_weights, >> False
            [11] average_attn_weights, >> True
            [12] mask_type) >> None
        """
        query, key, value = args[0], args[1], args[2]

        bsz, tgt_len, embed_dim = query.shape
        _, src_len, _ = key.shape

        q, k, v = F._in_projection_packed(
            query, key, value, self.in_proj_weight, self.in_proj_bias
        )
        q = q.view(bsz, tgt_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(bsz, src_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(bsz, src_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Efficient implementation equivalent to the following:
        attn_weight = q @ k.transpose(-2, -1) / math.sqrt(q.size(-1))
        attn_weight = torch.softmax(attn_weight, dim=-1)

        attn_output = attn_weight @ v
        attn_output = (
            attn_output.permute(0, 2, 1, 3).contiguous().view(bsz, tgt_len, embed_dim)
        )
        attn_output = F.linear(attn_output, self.out_proj.weight, self.out_proj.bias)

        return attn_output, attn_weight
        """torch.Size([128, 197, 768]) torch.Size([128, 12, 197, 197])"""

    def forward(self, *args, **kwargs):
        """
        class MultiheadAttention.forward(
        self,
        [0] query: Tensor,
        [1] key: Tensor,
        [2] value: Tensor,
        [3] key_padding_mask: Optional[Tensor] = None,
        [4] need_weights: bool = True,
        [5] attn_mask: Optional[Tensor] = None,
        [6] average_attn_weights: bool = True,
        [7] is_causal : bool = False) -> Tuple[Tensor, Optional[Tensor]]:
        """

        torch._native_multi_head_attention = self.QuantMSA
        _result = super().forward(*args, **kwargs)
        torch._native_multi_head_attention = self.ORG_MSA_FORWARD_FUNC

        return _result
        """torch.Size([128, 197, 768]) torch.Size([128, 197, 197])"""
        """torch.Size([128, 197, 768]) torch.Size([128, 12, 197, 197])"""


# class QuantEncoderBlock(QuantBaseModule):
class QuantEncoderBlock(nn.Module):
    """Transformer encoder block."""

    def __init__(
        self,
        orgEncoderBlock: EncoderBlock,
    ):
        super().__init__()
        self.num_heads = orgEncoderBlock.num_heads

        # Attention block
        ## nn.LayerNorm : LayerNorm((768,), eps=1e-06, elementwise_affine=True)
        self.ln_1 = orgEncoderBlock.ln_1
        ## nn.MultiheadAttention
        self.self_attention = QuantMultiheadAttention(orgEncoderBlock.self_attention)
        # self.self_attention = orgEncoderBlock.self_attention
        ## nn.Dropout : Dropout(p=0.0, inplace=False)
        self.dropout = orgEncoderBlock.dropout

        # MLP block
        ## nn.LayerNorm : LayerNorm((768,), eps=1e-06, elementwise_affine=True)
        self.ln_2 = orgEncoderBlock.ln_2
        ## MLPBlock
        # self.mlp = orgEncoderBlock.mlp
        self.mlp = QuantMLPBlock(orgEncoderBlock.mlp)
        print("    QuantEncoderBlock is created")

    def forward(self, input: torch.Tensor):
        # print("B", end="")
        torch._assert(
            input.dim() == 3,
            f"Expected (batch_size, seq_length, hidden_dim) got {input.shape}",
        )
        x = self.ln_1(input)
        # x, _ = self.self_attention(x, x, x, need_weights=False)
        x, _ = self.self_attention(
            x, x, x, need_weights=True
        )  # @jiho264, modify for attention map
        x = self.dropout(x)
        x = x + input

        y = self.ln_2(x)
        y = self.mlp(y)
        return x + y


# class QuantEncoder(QuantBaseModule):
class QuantEncoder(nn.Module):
    def __init__(self, orgEncoder: Encoder):
        super().__init__()
        self.pos_embedding = orgEncoder.pos_embedding
        self.dropout = orgEncoder.dropout
        self.layers = orgEncoder.layers
        layers: OrderedDict[str, nn.Module] = OrderedDict()

        for orgEncoderBlock, i in zip(orgEncoder.layers, range(len(orgEncoder.layers))):
            layers[f"encoder_layer_{i}"] = QuantEncoderBlock(
                orgEncoderBlock=orgEncoderBlock
            )
        self.layers = nn.Sequential(layers)
        self.ln = orgEncoder.ln
        print("QuantEncoder is created")

    def forward(self, input: torch.Tensor):
        torch._assert(
            input.dim() == 3,
            f"Expected (batch_size, seq_length, hidden_dim) got {input.shape}",
        )
        input = input + self.pos_embedding
        return self.ln(self.layers(self.dropout(input)))


class QuantLayer(QuantBaseModule):
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
        print("  QuantLayer is created")

    def forward(self, input: torch.Tensor):
        _weight = self.forward_weight_quantizer(self.weight)

        x = self.fwd_func(input, _weight, self.bias, **self.fwd_kwargs)

        return self.forward_activation_quantizer(x)


# class QuantViT(QuantBaseModule):
class QuantViT(nn.Module):
    def __init__(self, orgViT, argsW: dict = {}, argsA: dict = {}):
        super().__init__()
        self.image_size = orgViT.image_size
        self.patch_size = orgViT.patch_size
        self.hidden_dim = orgViT.hidden_dim
        self.mlp_dim = orgViT.mlp_dim
        self.attention_dropout = orgViT.attention_dropout
        self.dropout = orgViT.dropout
        self.num_classes = orgViT.num_classes
        self.representation_size = orgViT.representation_size  # <- None
        self.norm_layer = orgViT.norm_layer
        self.seq_length = orgViT.seq_length

        self.class_token = orgViT.class_token
        self.conv_proj = QuantLayer(orgViT.conv_proj)
        self.encoder = QuantEncoder(orgViT.encoder)

        heads_layers: OrderedDict[str, nn.Module] = OrderedDict()
        if self.representation_size is None:
            heads_layers["head"] = QuantLayer(orgViT.heads.head)
        else:
            pass
            # heads_layers["pre_logits"] = nn.Linear(hidden_dim, representation_size)
            # heads_layers["act"] = nn.Tanh()
            # heads_layers["head"] = nn.Linear(representation_size, num_classes)

        self.heads = nn.Sequential(heads_layers)

        self.w_quant_switch_order, self.a_quant_switch_order = False, False
        self.argsW = argsW
        self.argsA = argsA

        print("QuantViT is created")

    def _process_input(self, x: torch.Tensor) -> torch.Tensor:
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
        x = self.conv_proj(x)
        # (n, hidden_dim, n_h, n_w) -> (n, hidden_dim, (n_h * n_w))
        x = x.reshape(n, self.hidden_dim, n_h * n_w)

        # (n, hidden_dim, (n_h * n_w)) -> (n, (n_h * n_w), hidden_dim)
        # The self attention layer expects inputs in the format (N, S, E)
        # where S is the source sequence length, N is the batch size, E is the
        # embedding dimension
        x = x.permute(0, 2, 1)

        return x

    def _quant_switch(self):
        for name, module in self.named_modules():
            if hasattr(module, "w_quant_switch"):
                module.w_quant_switch = self.w_quant_switch_order
                if module.w_inited == False and hasattr(module, "weight"):
                    print(name, end="")
                    module.init_weight_quantizer(module.weight, self.argsW)
            if hasattr(module, "a_quant_switch"):
                module.a_quant_switch = self.a_quant_switch_order
                if module.a_inited == False:
                    pass

                    # [ ] implement activation quantizer
                    # You have to prepare the activation values before quantization
                    # raise Exception("Activation quantizer is not implemented yet.")

    def forward(self, x: torch.Tensor):
        self._quant_switch()

        # Reshape and permute the input tensor
        x = self._process_input(x)
        n = x.shape[0]

        # Expand the class token to the full batch
        batch_class_token = self.class_token.expand(n, -1, -1)
        x = torch.cat([batch_class_token, x], dim=1)

        x = self.encoder(x)

        # Classifier "token" as used by standard language architectures
        x = x[:, 0]

        x = self.heads(x)

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
=================================================================================================
"""

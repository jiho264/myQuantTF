import torch, math
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.vision_transformer import Encoder, EncoderBlock, MLPBlock
from collections import OrderedDict
from torch import Tensor
from .quantizer import quantizerDict, StraightThrough

from .int_functions import int_GELU, int_Softmax


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


class QuantActOnly(nn.Module):
    def __init__(self):
        super().__init__()
        self.activationQuantizer = QuantBase()

    def forward(self, input: torch.Tensor):
        return self.activationQuantizer(input)


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


class QuantMLPBlock(nn.Module):
    def __init__(self, orgMLPBlock: MLPBlock):
        super().__init__()

        self.linear_1 = QuantLinearLayer(orgMLPBlock.get_submodule("0"))
        # Linear(in_features=768, out_features=3072, bias=True)

        self.gelu = orgMLPBlock.get_submodule("1")
        # GELU(approximate='none')
        self.using_int_gelu = False
        self.int_gelu_bit_width = None

        self.dropout_1 = orgMLPBlock.get_submodule("2")
        # Dropout(p=0.0, inplace=False)

        self.linear_2 = QuantLinearLayer(orgMLPBlock.get_submodule("3"))
        # Linear(in_features=3072, out_features=768, bias=True)

        self.dropout_2 = orgMLPBlock.get_submodule("4")
        # Dropout(p=0.0, inplace=False)

    def forward(self, input: torch.Tensor):

        x = self.linear_1(input)
        if (
            self.using_int_gelu
            and self.linear_1.activationQuantizer.Quantizer._ready_to_quantize
        ):
            # the X is quantized value but represent by FP domain.
            S = torch.max(torch.abs(x)) / (2 ** (self.int_gelu_bit_width - 1) - 1)
            x_q = torch.round(x / S)
            # apply the temporary sysmetric quantization
            x_q, _S = int_GELU(x_q, S)
            x = x_q * _S
            # dequant -> the quantized value represent by FP domain
        else:
            x = self.gelu(x)
        x = self.dropout_1(x)
        x = self.linear_2(x)
        x = self.dropout_2(x)
        return x


class QuantMultiheadAttention(nn.MultiheadAttention):
    def __init__(self, orgMultiheadAttention):
        super().__init__(
            orgMultiheadAttention.embed_dim,
            orgMultiheadAttention.num_heads,
            dropout=orgMultiheadAttention.dropout,
            batch_first=orgMultiheadAttention.batch_first,
        )
        self.__dict__.update(orgMultiheadAttention.__dict__)

        self.ORG_MSA_FORWARD_FUNC = torch._native_multi_head_attention

        OUT_PROJ = self.out_proj
        delattr(self, "out_proj")

        # make in_proj module
        in_proj = nn.Linear(
            in_features=self.in_proj_weight.shape[1],  # 768
            out_features=self.in_proj_weight.shape[0],  # 2304 = 768 * 3
            bias=True,
        )
        in_proj.weight = self.in_proj_weight
        in_proj.bias = self.in_proj_bias

        self.in_proj = QuantLinearLayer(in_proj)

        # [1] Q @ K
        # INT8 GEMM -> return INT32

        # [2] Softmax(Q @ K / sqrt(d_k))
        self.using_int_softmax = False
        self.int_softmax_bit_width = None

        # [3] Attn @ V
        self.attnOutActivationQuantizer = QuantActOnly()

        self.out_proj = QuantLinearLayer(OUT_PROJ)

        # print("    - QuantMultiheadAttention is created")

    def QuantMSA(self, *args, **kwargs):
        """
        return torch._native_multi_head_attention(
            args[0] : query, >> shape : torch.Size([batch, 197, 768])
            args[1] : key,   >> shape : torch.Size([batch, 197, 768])
            args[2] : value, >> shape : torch.Size([batch, 197, 768])
            args[3] : self.embed_dim, >> 768
            args[4] : self.num_heads, >> 12
            args[5] : self.in_proj_weight, >> shape : torch.Size([2304, 768])
            args[6] : self.in_proj_bias,   >> shape : torch.Size([2304])
            args[7] : self.out_proj.weight, >> shape : torch.Size([768, 768])
            args[8] : self.out_proj.bias,   >> shape : torch.Size([768])
            args[9] : merged_mask, >> None
            args[10]: need_weights, >> False
            args[11]: average_attn_weights, >> True
            args[12]: mask_type) >> None
        """
        query, key, value = args[0], args[0], args[0]
        ## >> torch.Size([128, 197, 768])

        """ INT GEMM -> return INT32 -> return INT8 """
        proj = self.in_proj(query)  # with W_qkv
        ## >> torch.Size([128, 197, 2304])
        # scaler.shape = ([2304, 1])

        proj = (
            proj.unflatten(-1, (3, query.size(-1)))
            .unsqueeze(0)
            .transpose(0, -2)
            .squeeze(-2)
            .contiguous()
        )  ## >> torch.Size([3, 128, 197, 768])

        bsz, tgt_len, embed_dim = query.shape
        _, src_len, _ = key.shape

        q = proj[0].view(bsz, tgt_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = proj[1].view(bsz, src_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = proj[2].view(bsz, src_len, self.num_heads, self.head_dim).transpose(1, 2)

        """ INT GEMM -> return INT32 """
        qk = q @ k.transpose(-2, -1) / math.sqrt(q.size(-1))
        ## >> torch.Size([128, 12, 197, 197])

        """ INT32 -> return log softmax INT8 """
        if (
            self.using_int_softmax
            and self.in_proj.activationQuantizer.Quantizer._ready_to_quantize
        ):
            # [ ] implement the INT8 softmax

            S = torch.max(torch.abs(qk)) / (2 ** (self.int_softmax_bit_width - 1) - 1)
            qk_q = torch.round(qk / S)
            # apply the temporary sysmetric quantization
            qk_q, _S = int_Softmax(qk_q, S)
            x = qk_q  # * _S
            attn_map = x

        else:
            attn_map = torch.softmax(qk, dim=-1)
        ## >> torch.Size([128, 12, 197, 197])

        """ INT GEMM -> return INT32 -> return INT8 """
        attn_output = self.attnOutActivationQuantizer(attn_map @ v)
        ## >> torch.Size([128, 12, 197, 64])

        attn_output = (
            attn_output.permute(0, 2, 1, 3).contiguous().view(bsz, tgt_len, embed_dim)
        )
        ## >> torch.Size([128, 197, 768])
        # torch.save(attn_output, "prac/attn_output.pt")
        # torch.save(attn_map, "prac/attn_map.pt")

        """ INT GEMM -> return INT32 -> return INT8 """
        attn_output = self.out_proj(attn_output)  # with W_out
        ## >> torch.Size([128, 197, 768])

        return attn_output, attn_map
        """torch.Size([128, 197, 768]) torch.Size([128, 12, 197, 197])"""

    def forward(self, *args, **kwargs):

        torch._native_multi_head_attention = self.QuantMSA
        out = super().forward(*args, **kwargs)
        torch._native_multi_head_attention = self.ORG_MSA_FORWARD_FUNC

        return out


class QuantLayerNorm(nn.Module):
    def __init__(self, orgLayerNorm):
        super().__init__()

        self.normalized_shape = orgLayerNorm.normalized_shape
        self.weight = orgLayerNorm.weight.clone().detach()
        self.bias = orgLayerNorm.bias.clone().detach()
        self.eps = orgLayerNorm.eps

        self.activationQuantizer = QuantBase()

    def forward(self, input: torch.Tensor):
        return self.activationQuantizer(
            F.layer_norm(input, self.normalized_shape, self.weight, self.bias, self.eps)
        )


class QuantEncoderBlock(nn.Module):
    """Transformer encoder block."""

    def __init__(self, orgEncoderBlock: EncoderBlock):
        super().__init__()
        self.num_heads = orgEncoderBlock.num_heads

        ## nn.LayerNorm : LayerNorm((768,), eps=1e-06, elementwise_affine=True)
        self.ln_1 = QuantLayerNorm(orgEncoderBlock.ln_1)

        ## nn.MultiheadAttention
        self.self_attention = QuantMultiheadAttention(orgEncoderBlock.self_attention)

        ## nn.Dropout : Dropout(p=0.0, inplace=False)
        self.dropout = orgEncoderBlock.dropout

        ## nn.LayerNorm : LayerNorm((768,), eps=1e-06, elementwise_affine=True)
        self.ln_2 = QuantLayerNorm(orgEncoderBlock.ln_2)

        ## MLPBlock
        self.mlp = QuantMLPBlock(orgEncoderBlock.mlp)

    def forward(self, input: torch.Tensor):
        torch._assert(
            input.dim() == 3,
            f"Expected (batch_size, seq_length, hidden_dim) got {input.shape}",
        )

        x = self.ln_1(input)
        x, _ = self.self_attention(x, x, x, need_weights=False)
        x = self.dropout(x)
        x = x + input

        y = self.ln_2(x)
        y = self.mlp(y)
        return x + y


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
        self.ln = QuantLayerNorm(orgEncoder.ln)

    def forward(self, input: torch.Tensor):
        torch._assert(
            input.dim() == 3,
            f"Expected (batch_size, seq_length, hidden_dim) got {input.shape}",
        )
        input = input + self.pos_embedding
        return self.ln(self.layers(self.dropout(input)))


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
        print(f"Attention quantization parameter : {args_softmax}")
        print(f"LayerNorm quantization parameter : {args_ln}")
        print(f"GELU quantization parameter : {args_gelu}")

        self.__dict__.update(orgViT.__dict__)
        self.conv_proj = QuantLinearLayer(orgViT.conv_proj)
        self.encoder = QuantEncoder(orgViT.encoder)

        heads_layers: OrderedDict[str, nn.Module] = OrderedDict()
        if self.representation_size is None:
            heads_layers["head"] = QuantLinearLayer(orgViT.heads.head)
        else:
            pass
            # heads_layers["pre_logits"] = nn.Linear(hidden_dim, representation_size)
            # heads_layers["act"] = nn.Tanh()
            # heads_layers["head"] = nn.Linear(representation_size, num_classes)

        self.heads = nn.Sequential(heads_layers)

        self.vit_w_quant = False if args_w == {} else True
        self.vit_a_quant = False if args_a == {} else True
        self.vit_ln_quant = False if args_ln == {} else True
        self.vit_int_gelu = False if args_gelu == {} else True
        self.vit_int_softmax = False if args_softmax == {} else True

        for name, module in self.named_modules():
            if isinstance(module, QuantLinearLayer):
                if args_w:
                    module.weightQuantizer.init_quantizer(args_w, module.weight)
                    print("[W]", name)
                if args_a:
                    if name != "heads.head":
                        module.activationQuantizer.init_quantizer(args_a)
                        print("[A]", name)
            elif isinstance(module, QuantActOnly):
                if args_a:
                    module.activationQuantizer.init_quantizer(args_a)
                    print("[A]", name)
            elif isinstance(module, QuantLayerNorm):
                if args_ln:
                    module.activationQuantizer.init_quantizer(args_ln)
                    print("[A]", name)
            elif isinstance(module, QuantMLPBlock):
                if args_gelu:
                    module.int_gelu_bit_width = args_gelu.get("bit_width")
                    print(f"[INT{module.int_gelu_bit_width} GELU]", name)
            elif isinstance(module, QuantMultiheadAttention):
                if args_softmax:
                    module.int_softmax_bit_width = args_softmax.get("bit_width")
                    print(f"[INT{module.int_softmax_bit_width} Softmax]", name)

        self.orgforward = orgViT.forward

    def _quant_switch(self):
        for name, module in self.named_modules():
            if isinstance(module, QuantLinearLayer):
                module.weightQuantizer.do_quant = self.vit_w_quant
                module.activationQuantizer.do_quant = self.vit_a_quant
            elif isinstance(module, QuantActOnly):
                module.activationQuantizer.do_quant = self.vit_a_quant
            elif isinstance(module, QuantLayerNorm):
                module.activationQuantizer.do_quant = self.vit_ln_quant
            elif isinstance(module, QuantMLPBlock):
                module.using_int_gelu = self.vit_int_gelu
            elif isinstance(module, QuantMultiheadAttention):
                module.using_int_softmax = self.vit_int_softmax

    def forward(self, x: torch.Tensor):
        self._quant_switch()
        return self.orgforward(x)


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

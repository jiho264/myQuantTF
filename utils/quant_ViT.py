import torch, math
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.vision_transformer import Encoder, EncoderBlock, MLPBlock
from collections import OrderedDict
from torch import Tensor
from .quantizer import quantizerDict


class QuantControlModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.do_w_quant = False
        self.do_a_quant = False
        self.w_inited = False
        self.a_inited = False
        self.weightQuantizer = None
        self.activationQuantizer = None

    def init_weight_quantizer(self, org_tensor: Tensor, args: dict):
        if self.w_inited == True:
            raise Exception("Weight quantizer is already inited.")

        if args["scheme"] in ["MinMaxQuantizer"]:
            # using only MinMaxQuantizer
            pass
        else:
            raise NotImplementedError

        self.weightQuantizer = quantizerDict[args.get("scheme")](
            org_tensor=org_tensor, args=args
        )
        self.w_inited = True

    def init_activation_quantizer(self, org_tensor: Tensor, args: dict):
        if self.a_inited == True:
            raise Exception("Activation quantizer is already inited.")

        if args["scheme"] in ["DynamicMinMaxQuantizer", "MovingAvgMinMaxQuantizer"]:
            # using only DynamicMinMaxQuantizer, MovingAvgMinMaxQuantizer
            pass
        else:
            raise NotImplementedError

        self.activationQuantizer = quantizerDict[args.get("scheme")](
            org_tensor=org_tensor, args=args
        )
        self.a_inited = True

    def forward_weight_quantizer(self, input: Tensor) -> Tensor:
        if self.do_w_quant == True:
            return self.weightQuantizer(input)
        else:
            return input

    def forward_activation_quantizer(self, input: Tensor) -> Tensor:
        if self.do_a_quant == True:
            return self.activationQuantizer(input)
        else:
            return input


class QuantLinearLayer(QuantControlModule):
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


class QuantAttnMap(QuantControlModule):
    def __init__(self):
        """
        Only for attention map

        [1] result of attention map
        attn_map = q @ k.transpose(-2, -1) / math.sqrt(q.size(-1))
        ## >> torch.Size([128, 12, 197, 197])

        [2] result of softmax
        attn_map = torch.softmax(attn_map, dim=-1)
        ## >> torch.Size([128, 12, 197, 197])

        [3] result of attention output
        attn_output = attn_map @ v
        ## >> torch.Size([128, 12, 197, 64])

        """
        super().__init__()

        # case [3]
        self.is_result_of_mm_with_attn_with_value = False

    def forward(self, input: torch.Tensor):
        ## debug for MovingAvgMinMaxQuantizer.
        ## check for working in the every result of Attn @ V 
        # if self.is_result_of_mm_with_attn_with_value == True:
        #     print(self.activationQuantizer._ready_to_quantize, end="")
        #     print(self.activationQuantizer._scaler)
        return self.forward_activation_quantizer(input)


class QuantLayerNorm(QuantControlModule):
    def __init__(self, orgLayerNorm):
        super().__init__()
        self.__dict__.update(orgLayerNorm.__dict__)

        print("    - QuantLayerNorm is created")

    def forward(self, input: torch.Tensor):
        return self.forward_activation_quantizer(
            F.layer_norm(input, self.normalized_shape, self.weight, self.bias, self.eps)
        )


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
        self.linear_1 = QuantLinearLayer(orgMLPBlock.get_submodule("0"))
        # print(self.linear_1.weight.shape)  # torch.Size([3072, 768])
        self.gelu = orgMLPBlock.get_submodule("1")
        self.dropout_1 = orgMLPBlock.get_submodule("2")
        self.linear_2 = QuantLinearLayer(orgMLPBlock.get_submodule("3"))
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
        self.attnMapActivationQuantizer = QuantAttnMap()
        # [2] Attn = Softmax()
        self.attnMapSoftmaxActivationQuantizer = QuantAttnMap()
        # [3] Attn @ V
        self.attnOutActivationQuantizer = QuantAttnMap()
        # using QuantAttnMap class but different purpose
        # only for activation quantization.
        self.attnOutActivationQuantizer.is_result_of_mm_with_attn_with_value = True

        self.out_proj = QuantLinearLayer(OUT_PROJ)

        print("    - QuantMultiheadAttention is created")

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
        attn_map = q @ k.transpose(-2, -1) / math.sqrt(q.size(-1))
        # torch.save(attn_map, "attn_map_qk.pt") # 여기 min, max가 -26, 27으로 관찰됨. 일단은
        attn_map = self.attnMapActivationQuantizer(attn_map)
        
        ## >> torch.Size([128, 12, 197, 197])

        """ INT32 -> return log softmax INT8 """
        # [ ] implement log softmax quantizer
        attn_map = torch.softmax(attn_map, dim=-1)  # 1D
        zeta = 1
        gamma = -0.0001
        attn_map = torch.clip((zeta - gamma) * attn_map + gamma, 0, 1)
        # torch.save(attn_map, "attn_map_softmax.pt")
        attn_map = self.attnMapSoftmaxActivationQuantizer(attn_map)
        ## >> torch.Size([128, 12, 197, 197])

        """ INT GEMM -> return INT32 -> return INT8 """
        attn_output = attn_map @ v  # 2D
        attn_map = self.attnOutActivationQuantizer(attn_output)
        ## >> torch.Size([128, 12, 197, 64])

        attn_output = (
            attn_output.permute(0, 2, 1, 3).contiguous().view(bsz, tgt_len, embed_dim)
        )
        ## >> torch.Size([128, 197, 768])

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
        # self.ln_1 = orgEncoderBlock.ln_1
        self.ln_1 = QuantLayerNorm(orgEncoderBlock.ln_1)
        ## nn.MultiheadAttention
        self.self_attention = QuantMultiheadAttention(orgEncoderBlock.self_attention)
        # self.self_attention = orgEncoderBlock.self_attention
        ## nn.Dropout : Dropout(p=0.0, inplace=False)
        self.dropout = orgEncoderBlock.dropout

        # MLP block
        ## nn.LayerNorm : LayerNorm((768,), eps=1e-06, elementwise_affine=True)
        # self.ln_2 = orgEncoderBlock.ln_2
        self.ln_2 = QuantLayerNorm(orgEncoderBlock.ln_2)
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
        # self.ln = orgEncoder.ln
        self.ln = QuantLayerNorm(orgEncoder.ln)
        print("QuantEncoder is created")

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
        args_attn: dict = {},
        args_ln: dict = {},
    ):
        super().__init__()
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

        self.args_attn = args_attn
        self.model_attn_quant = False if self.args_attn == {} else True

        self.args_ln = args_ln
        self.model_ln_quant = False if self.args_ln == {} else True

        self.args_w = args_w
        self.model_w_quant = False if self.args_w == {} else True
        self.model_a_quant = False
        # for init weight quantizer, temporarily do not using activation quantizer
        self._quant_switch()

        self.args_a = args_a
        self.model_a_quant = False if self.args_a == {} else True

        self.orgforward = orgViT.forward
        print("QuantViT is created")

    def _quant_switch(self):
        for name, module in self.named_modules():
            if isinstance(module, QuantLinearLayer):
                """The quantizer for Linear layer"""
                module.do_w_quant = self.model_w_quant
                module.do_a_quant = self.model_a_quant
                if self.model_w_quant:
                    if module.w_inited == False and hasattr(module, "weight"):
                        print(f"[Weight] {name}    ", end="")
                        module.init_weight_quantizer(module.weight, self.args_w)

                if self.model_a_quant:
                    if module.a_inited == False:
                        print(f"[Activation] {name}    ", end="")
                        module.init_activation_quantizer(None, self.args_a)

            elif isinstance(module, QuantAttnMap):
                if module.is_result_of_mm_with_attn_with_value == True:
                    """ The quantizer for Attention Map ( result of Attn @ V ) """
                    module.do_a_quant = self.model_a_quant
                    if self.model_a_quant:
                        if module.a_inited == False:
                            print(f"[Activation] {name}    ", end="")
                            module.init_activation_quantizer(None, self.args_a)
                else:
                    """ The quantizer for MultiheadAttention ( Softmax process ) """
                    module.do_a_quant = self.model_attn_quant
                    if self.model_attn_quant:
                        if module.a_inited == False:
                            print(name, "attention quantizer is inited.")
                            module.init_activation_quantizer(None, self.args_attn)
                            # [ ] implement activation quantizer
                            # You have to prepare the activation values before quantization
                            # raise Exception("Activation quantizer is not implemented yet.")

            elif isinstance(module, QuantLayerNorm):
                """ The quantizer for LayerNorm """
                module.do_a_quant = self.model_ln_quant
                if self.model_ln_quant:
                    if module.a_inited == False:
                        print(name, "LN quantizer is inited.")
                        module.init_activation_quantizer(None, self.args_attn)
                        # [ ] implement activation quantizer
                        # You have to prepare the activation values before quantization
                        # raise Exception("Activation quantizer is not implemented yet

    def forward(self, x: torch.Tensor):
        self._quant_switch()
        ## debug for MovingAvgMinMaxQuantizer. 
        ## check for working in the first conv_proj layer.
        # print(self.conv_proj.activationQuantizer._ready_to_quantize, end="")
        # print(self.conv_proj.activationQuantizer._scaler)
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

import torch, math
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.vision_transformer import Encoder, EncoderBlock, MLPBlock
from collections import OrderedDict
from torch import Tensor
from typing import Optional, Tuple

from .quant_modules import QuantWeight, QuantAct, QuantLinear, IntGELU, IntSoftMax


class QuantMLP(nn.Module):
    def __init__(self, orgModule: MLPBlock, args_w, args_a, args_gelu):
        super().__init__()

        self.linear_1 = QuantLinear(
            orgModule=orgModule.get_submodule("0"), args_w=args_w, args_a=args_a
        )

        # Linear(in_features=768, out_features=3072, bias=True)

        self.gelu = IntGELU(args_gelu=args_gelu)
        # GELU(approximate='none')

        self.dropout_1 = orgModule.get_submodule("2")
        # Dropout(p=0.0, inplace=False)

        self.linear_2 = QuantLinear(
            orgModule=orgModule.get_submodule("3"), args_w=args_w, args_a=args_a
        )
        # Linear(in_features=3072, out_features=768, bias=True)

        self.dropout_2 = orgModule.get_submodule("4")
        # Dropout(p=0.0, inplace=False)

    def forward(self, input: torch.Tensor):

        x = self.linear_1(input)
        x = self.gelu(x)
        x = self.dropout_1(x)
        x = self.linear_2(x)
        x = self.dropout_2(x)
        return x


class QuantMSA(nn.Module):
    def __init__(self, orgModule, args_w, args_a, args_softmax):
        super().__init__()
        assert orgModule.num_heads == 12
        assert orgModule.batch_first == True
        assert orgModule._qkv_same_embed_dim == True
        assert orgModule.add_zero_attn == False
        assert orgModule.dropout == 0.0

        self.embed_dim = orgModule.embed_dim
        self.num_heads = orgModule.num_heads
        self.head_dim = orgModule.head_dim

        # make in_proj module
        tmp_in_proj = nn.Linear(
            in_features=orgModule.in_proj_weight.shape[1],  # 768
            out_features=orgModule.in_proj_weight.shape[0],  # 2304 = 768 * 3
            bias=True,
        )
        tmp_in_proj.weight = orgModule.in_proj_weight
        tmp_in_proj.bias = orgModule.in_proj_bias

        # [1] get Q, K, V
        self.in_proj = QuantLinear(orgModule=tmp_in_proj, args_w=args_w, args_a=args_a)

        # [2] Q @ K / sqrt(d_k)
        # INT8 GEMM -> return INT32

        # [3] Softmax()
        self.softmax = IntSoftMax(args_softmax=args_softmax)

        # [4] Softmaxed @ V
        self.attn_map_act = QuantAct(args_a=args_a)

        # [5] Attn @ w_out
        self.out_proj = QuantLinear(
            orgModule=orgModule.out_proj, args_w=args_w, args_a=args_a
        )

    def forward(self, input: torch.Tensor, **kwargs):
        # query, key, value = input, input, input
        ## >> torch.Size([128, 197, 768])

        """[1] INT GEMM -> return INT32 -> return INT8"""
        proj = self.in_proj(input)
        ## >> torch.Size([128, 197, 2304])

        proj = (
            proj.unflatten(-1, (3, input.size(-1)))
            .unsqueeze(0)
            .transpose(0, -2)
            .squeeze(-2)
            .contiguous()
        )  ## >> torch.Size([3, 128, 197, 768])

        bsz, tgt_len, embed_dim = input.shape
        _, src_len, _ = input.shape

        q = proj[0].view(bsz, tgt_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = proj[1].view(bsz, src_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = proj[2].view(bsz, src_len, self.num_heads, self.head_dim).transpose(1, 2)

        """ [2] INT GEMM -> return INT32 """
        qk = q @ k.transpose(-2, -1) / math.sqrt(q.size(-1))
        ## >> torch.Size([128, 12, 197, 197])

        """ [3] INT32 -> return log softmax INT8 """
        attn_map = self.softmax(qk, dim=-1)
        ## >> torch.Size([128, 12, 197, 197])

        """ [4] INT GEMM -> return INT32 -> return INT8 """
        attn_output = self.attn_map_act(attn_map @ v)
        ## >> torch.Size([128, 12, 197, 64])

        attn_output = (
            attn_output.permute(0, 2, 1, 3).contiguous().view(bsz, tgt_len, embed_dim)
        )
        ## >> torch.Size([128, 197, 768])

        """ [5] INT GEMM -> return INT32 -> return INT8 """
        attn_output = self.out_proj(attn_output)
        ## >> torch.Size([128, 197, 768])

        return attn_output, attn_map

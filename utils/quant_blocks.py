import torch, math
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.vision_transformer import Encoder, EncoderBlock, MLPBlock
from collections import OrderedDict
from torch import Tensor
from typing import Optional, Tuple

from .quant_modules import (
    QuantAct,
    QuantLinearWithWeight,
    IntGELU,
    IntSoftMax,
    QuantMatMul,
)


class QuantMLP(nn.Module):
    def __init__(self, orgModule: MLPBlock, args_w, args_a, args_gelu):
        super().__init__()

        # [1]
        self.linear_1 = QuantLinearWithWeight(
            orgModule=orgModule.get_submodule("0"), args_w=args_w
        )
        self.linear_1_act = QuantAct(args_a=args_a)
        # Linear(in_features=768, out_features=3072, bias=True)

        # [2]
        self.gelu = IntGELU(args_gelu=args_gelu)
        self.gelu_act = QuantAct(args_a=args_a)
        # GELU(approximate='none')

        self.dropout_1 = orgModule.get_submodule("2")
        # Dropout(p=0.0, inplace=False)

        # [3]
        self.linear_2 = QuantLinearWithWeight(
            orgModule=orgModule.get_submodule("3"), args_w=args_w
        )
        self.linear_2_act = QuantAct(args_a=args_a)
        self.linear_2_act.activation_bit = 16
        # Linear(in_features=3072, out_features=768, bias=True)

        self.dropout_2 = orgModule.get_submodule("4")
        # Dropout(p=0.0, inplace=False)

    def forward(self, x, s_x):

        x, s_x = self.linear_1(x, s_x)
        x, s_x = self.linear_1_act(x, s_x)
        x, s_x = self.gelu(x, s_x)
        x, s_x = self.gelu_act(x, s_x)
        x = self.dropout_1(x)
        x, s_x = self.linear_2(x, s_x)
        x, s_x = self.linear_2_act(x, s_x)
        x = self.dropout_2(x)
        return x, s_x


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
        self.in_proj = QuantLinearWithWeight(orgModule=tmp_in_proj, args_w=args_w)
        self.in_proj_act = QuantAct(args_a=args_a)

        # [2] Q @ K / sqrt(d_k)
        # INT8 GEMM -> return INT32
        self.qk_mm = QuantMatMul()
        self.qk_act = QuantAct(args_a=args_a)

        # [3] Softmax()
        self.softmax = IntSoftMax(args_softmax=args_softmax)

        # [4] Softmaxed @ V
        self.attnout_mm = QuantMatMul()
        self.attnout_act = QuantAct(args_a=args_a)

        # [5] Attn @ w_out
        self.out_proj = QuantLinearWithWeight(
            orgModule=orgModule.out_proj, args_w=args_w
        )
        self.out_proj_act = QuantAct(args_a=args_a)
        self.out_proj_act.activation_bit = 16

    def forward(self, x, s_x):
        # query, key, value = input, input, input
        ## >> torch.Size([128, 197, 768])

        """[1] INT GEMM -> return INT32 -> return INT8"""
        qkv, s_qkv = self.in_proj(x, s_x)
        qkv, s_qkv = self.in_proj_act(qkv, s_qkv)
        ## >> torch.Size([128, 197, 2304])

        qkv = (
            qkv.unflatten(-1, (3, x.size(-1)))
            .unsqueeze(0)
            .transpose(0, -2)
            .squeeze(-2)
            .contiguous()
        )  ## >> torch.Size([3, 128, 197, 768])

        bsz, tgt_len, embed_dim = x.shape
        _, src_len, _ = x.shape

        q = qkv[0].view(bsz, tgt_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = qkv[1].view(bsz, src_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = qkv[2].view(bsz, src_len, self.num_heads, self.head_dim).transpose(1, 2)

        """ [2] INT GEMM -> return INT32 """

        qk, s_qk = self.qk_mm(q, s_qkv, k.transpose(-2, -1), s_qkv)
        qk /= math.sqrt(q.size(-1))
        s_qk /= math.sqrt(q.size(-1))
        qk, s_qk = self.qk_act(qk, s_qk)
        ## >> torch.Size([128, 12, 197, 197])

        """ [3] INT32 -> return log softmax INT8 """
        attn_map, s_amap = self.softmax(qk, s_qk, dim=-1)
        ## >> torch.Size([128, 12, 197, 197])

        """ [4] INT GEMM -> return INT32 -> return INT8 """

        attn_output, s_aout = self.attnout_mm(attn_map, s_amap, v, s_qkv)
        attn_output, s_aout = self.attnout_act(attn_output, s_aout)  # with s_qk
        ## >> torch.Size([128, 12, 197, 64])

        attn_output = (
            attn_output.permute(0, 2, 1, 3).contiguous().view(bsz, tgt_len, embed_dim)
        )
        ## >> torch.Size([128, 197, 768])

        """ [5] INT GEMM -> return INT32 -> return INT8 """
        attn_output, s_aout = self.out_proj(attn_output, s_aout)
        attn_output, s_aout = self.out_proj_act(attn_output, s_aout)
        ## >> torch.Size([128, 197, 768])

        return attn_output, s_aout

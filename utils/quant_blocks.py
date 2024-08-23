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

        # [1] Linear(in_features=768, out_features=3072, bias=True)
        self.linear_1 = QuantLinearWithWeight(
            orgModule=orgModule.get_submodule("0"), args_w=args_w, args_a=args_a
        )

        # [2] GELU(approximate='none')
        self.gelu = IntGELU(args_gelu=args_gelu)
        self.gelu_act = QuantAct(args_a=args_a)

        self.dropout_1 = orgModule.get_submodule("2")
        # Dropout(p=0.0, inplace=False)

        # [3] Linear(in_features=3072, out_features=768, bias=True)
        self.linear_2 = QuantLinearWithWeight(
            orgModule=orgModule.get_submodule("3"), args_w=args_w, args_a=args_a
        )

        self.dropout_2 = orgModule.get_submodule("4")
        # Dropout(p=0.0, inplace=False)

    def forward(self, x_hat_int8, s_x):
        # [1]
        x_hat_int8, s_x = self.linear_1(x_hat_int8, s_x)

        # [2]
        x_hat_int8, s_x = self.gelu(x_hat_int8, s_x)
        x_hat_int8, s_x = self.gelu_act(x_hat_int8, s_x)

        x_hat_int8 = self.dropout_1(x_hat_int8)

        # [3]
        x_hat_int8, s_x = self.linear_2(x_hat_int8, s_x)

        x_hat_int8 = self.dropout_2(x_hat_int8)

        return x_hat_int8, s_x


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
        self.d_k = self.head_dim**0.5

        # make in_proj module
        tmp_in_proj = nn.Linear(
            in_features=orgModule.in_proj_weight.shape[1],  # 768
            out_features=orgModule.in_proj_weight.shape[0],  # 2304 = 768 * 3
            bias=True,
        )
        tmp_in_proj.weight = orgModule.in_proj_weight
        tmp_in_proj.bias = orgModule.in_proj_bias

        # [1] get Q, K, V
        self.in_proj = QuantLinearWithWeight(
            orgModule=tmp_in_proj, args_w=args_w, args_a=args_a
        )

        # [2] Q @ K / sqrt(d_k)
        # INT8 GEMM -> return INT32
        self.qk_mm = QuantMatMul()
        self.qk_act = QuantAct(args_a=args_a)

        # [3] Softmax(16bit) -> return INT8
        self.softmax = IntSoftMax(args_softmax=args_softmax)
        self.softmax_act = QuantAct(args_a=args_a, which="softmax_act")

        # [4] Softmaxed @ V (INT8 GEMM -> return INT32 -> return INT8)
        self.attnout_mm = QuantMatMul()
        self.attnout_act = QuantAct(args_a=args_a)

        # [5] Attn @ w_out
        self.out_proj = QuantLinearWithWeight(
            orgModule=orgModule.out_proj, args_w=args_w, args_a=args_a
        )

    def forward(self, x_hat, s_x):
        # query, key, value = input, input, input
        ## >> torch.Size([128, 197, 768])

        """[1] INT GEMM -> return INT32 -> return INT8"""
        qkv_hat_int8, s_qkv = self.in_proj(x_hat, s_x)
        ## >> torch.Size([128, 197, 2304])

        qkv_hat_int8 = (
            qkv_hat_int8.unflatten(-1, (3, x_hat.size(-1)))
            .unsqueeze(0)
            .transpose(0, -2)
            .squeeze(-2)
            .contiguous()
        )  ## >> torch.Size([3, 128, 197, 768])

        bsz, tgt_len, embed_dim = x_hat.shape
        _, src_len, _ = x_hat.shape

        q_shape = (bsz, tgt_len, self.num_heads, self.head_dim)
        kv_shape = (bsz, src_len, self.num_heads, self.head_dim)

        q_hat_int8 = qkv_hat_int8[0].view(q_shape).transpose(1, 2)
        k_hat_int8 = qkv_hat_int8[1].view(kv_shape).transpose(1, 2)
        v_hat_int8 = qkv_hat_int8[2].view(kv_shape).transpose(1, 2)

        """ [2] INT GEMM -> Accumulated by INT32 -> return INT8 """
        qk_hat_int32, s_qk = self.qk_mm(
            q_hat_int8, s_qkv, k_hat_int8.transpose(-2, -1), s_qkv
        )
        qk_hat_int32 /= self.d_k  # divide by 8 when head_dim == 64
        s_qk = (
            s_qk / self.d_k if s_qk is not None else s_qk
        )  # divide by 8 when head_dim == 64
        qk_hat_int8, s_qk = self.qk_act(qk_hat_int32, s_qk)
        ## >> torch.Size([128, 12, 197, 197])

        """ [3] INT8 -> return log softmax INT16 -> return INT8 (UINT7) """
        attn_map_hat_int16, s_amap = self.softmax(qk_hat_int8, s_qk, dim=-1)
        attn_map_hat_int8, s_amap = self.softmax_act(attn_map_hat_int16, s_amap)
        ## >> torch.Size([128, 12, 197, 197])

        """ [4] INT GEMM -> return INT32 -> return INT8 """

        attn_out_hat_int32, s_aout = self.attnout_mm(
            attn_map_hat_int8, s_amap, v_hat_int8, s_qkv
        )
        attn_out_hat_int8, s_aout = self.attnout_act(attn_out_hat_int32, s_aout)
        ## >> torch.Size([128, 12, 197, 64])

        attn_out_hat_int8 = (
            attn_out_hat_int8.permute(0, 2, 1, 3)
            .contiguous()
            .view(bsz, tgt_len, embed_dim)
        )
        ## >> torch.Size([128, 197, 768])

        """ [5] INT GEMM -> return INT32 -> return INT8 """
        attn_out_hat_int8, s_aout = self.out_proj(attn_out_hat_int8, s_aout)
        ## >> torch.Size([128, 197, 768])

        return attn_out_hat_int8, s_aout

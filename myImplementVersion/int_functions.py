import torch, math
import torch.nn as nn
import torch.nn.functional as F


def second_order_polynomial(x_q, s_x, a, b, c):
    """a(x+b)^2 + c"""
    """Kim, Sehoon, et al. "I-bert: Integer-only bert quantization." International conference on machine learning. PMLR, 2021."""
    with torch.no_grad():
        # [ Can be pre-computed ]
        q_b = torch.floor(b / s_x)  # PRE-COMPUTED >> INT
        q_c = torch.floor(c / (a * s_x**2))  # PRE-COMPUTED >> INT
        s_out = a * s_x**2  # PRE-COMPUTED-FP

    q_out = (x_q + q_b).pow(2) + q_c

    return q_out, s_out


def int_ERF(x_q, s_x):
    with torch.no_grad():
        a, b, c = -0.2888, -1.769, 1
        a, b, c = torch.tensor(a), torch.tensor(b), torch.tensor(c)

    """ONLY INT OPERATION"""
    _q = torch.min(x_q.abs(), (-b / s_x))

    q_L, s_L = second_order_polynomial(_q, s_x, a, b, c)

    q_out = torch.sign(x_q) * q_L

    return q_out, s_L


def int_GELU(x_q, s_x):
    with torch.device(x_q.device):
        # input : (INT, PRE-COMPUTED-FP)
        q_erf, s_erf = int_ERF(x_q, s_x / math.sqrt(2))
        # output : (INT, PRE-COMPUTED-FP)

        q_1 = torch.floor(1 / s_erf)  # floor(1 / PRE-COMPUTED-FP) >> INT

        q_out = x_q * (q_erf + q_1)  # INT
        s_out = s_x * s_erf / 2  # PRE-COMPUTED-FP

        return q_out, s_out


# def int_EXP(x_q, s_x):
#     a, b, c = 0.3585, 1.353, 0.344
#     a, b, c = torch.tensor(a), torch.tensor(b), torch.tensor(c)

#     q_ln2 = torch.floor(0.693147 / s_x)
#     z = torch.floor(-x_q / q_ln2)
#     # print(f"z min : {z.min()}, z max : {z.max()}")
#     q_p = x_q + z * q_ln2
#     p = q_p * s_x
#     # print(f"p min : {p.min()}, p max : {p.max()}")
#     q_L, s_L = second_order_polynomial(q_p, s_x, a, b, c)

#     q_out = torch.bitwise_right_shift(q_L.to(torch.int32), z.to(torch.int32)).to(
#         torch.float32
#     )
#     # q_out = (q_L / 2**z).round()
#     s_out = s_L

#     return q_out, s_out


# def int_Softmax(x_q, s_x):
#     q_hat = x_q - x_q.max(dim=-1).values.unsqueeze(-1)
#     q_exp, s_exp = int_EXP(q_hat, s_x)
#     q_out = q_exp / q_exp.sum(dim=-1).unsqueeze(-1)
#     # FIXME not clear! output is not int8
#     return q_out, s_exp


# def int_SQRT(n):
#     if n == 0:
#         """You have to prevent division by zero"""
#         return 0
#     else:
#         _bit_len = torch.tensor(int(n).bit_length())
#         x = 2 ** (torch.ceil(_bit_len / 2))
#         while True:
#             x_next = torch.floor((x + torch.floor(n / x)) / 2)

#             if x_next >= x:
#                 return x.item()  # Return as a Python integer
#             else:
#                 x = x_next


# # Function to apply int_SQRT to each element in a tensor
## FIXME huge computation time required
# def apply_int_SQRT_to_tensor(tensor):
#     shape = tensor.shape
#     tensor = tensor.flatten()  # Flatten the tensor to 1D for easier iteration
#     result = torch.tensor([int_SQRT(val) for val in tensor], dtype=torch.int32)
#     return result.view(shape)  # Reshape the result back to the original shape


# def int_LayerNorm(x_q, s_x, weight, bias, eps):

#     mean_q = x_q.mean(dim=-1, keepdim=True).round()
#     var_q = x_q.var(dim=-1, keepdim=True, unbiased=False).round()

#     weight_q = (weight / s_x).round()
#     bias_q = (bias / s_x).round()
#     eps_q = (eps / s_x).round()

#     assert var_q.min() >= 0, "Negative variance detected"
#     assert eps_q.min() >= 0, "Negative epsilon detected"
#     div_by = apply_int_SQRT_to_tensor(var_q + eps_q)
#     div_by[div_by == 0] = 1  # Avoid division by zero

#     y_q = (x_q - mean_q) / div_by * weight_q + bias_q

#     return y_q, s_x

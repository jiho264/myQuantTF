import torch, time, argparse, timm
import torch.nn as nn

from .data_utils import save_inp_oup_data, get_train_samples, GetDataset, evaluate
from model.quant_ViT import QuantViT, QuantEncoderBlock
from model.quant_modules import QuantLinearWithWeight, QuantAct
from model.quant_blocks import QuantMLP, QuantMSA


def _decayed_beta(i, n_iter, _warmup=0.05):
    if i < n_iter * _warmup:
        return torch.tensor(0.0)
    else:
        # 0 ~ 1 when after 4k iter of 20k len
        decay = (i - n_iter * _warmup) / (n_iter * (1 - _warmup))
        _beta = 18 - decay * 18 + 2
        return _beta


def _adaround_for_a_module(model, module, cali_data, batch_size, lr, n_iter):
    model.eval()

    # [1] get {Origin FP output(A_fp_lth), Quantized input and output(X_q_lth, A_q_lth)}
    model.set_quant_mode(False, False, False, False, False)
    _, _, OUTPUT_FP, _ = save_inp_oup_data(model, module, cali_data, batch_size)
    model.set_quant_mode(True, True, True, True, True)
    INPUT_HAT, s_x_input, _, _ = save_inp_oup_data(model, module, cali_data, batch_size)
    print(f"    INPUT_FP : {INPUT_HAT.shape}")
    print(f"    OUTPUT_FP : {OUTPUT_FP.shape}")
    S_X_INPUT = s_x_input[0].to("cuda")
    assert (s_x_input.mean() - S_X_INPUT) < 1e-6, s_x_input
    assert S_X_INPUT.dim() == 0

    # > INPUT_FP : Tensor
    # > S_X_INPUT : Scalar
    # > OUTPUT_FP : Tensor

    # [2] Define the optimizer and loss function
    parameters_v = []
    parameters_s_a = []
    for name, sub_module in module.named_modules():
        if isinstance(sub_module, QuantLinearWithWeight):
            parameters_v.append(sub_module._v)
            print(f"    V   : {name}, {sub_module._v.shape}")
        """ NOW IS ONLY FOR WEIGHTS"""
        # elif isinstance(sub_module, QuantAct):
        #     parameters_s_a.append(sub_module.s_out)
        #     print(f"    s_a : {name}, {sub_module.s_out.shape}, {sub_module.s_out}")

    if parameters_v != []:
        optimizer_w = torch.optim.Adam(parameters_v, lr=1e-2)
        print(optimizer_w, n_iter)
    if parameters_s_a != []:
        optimizer_s_a = torch.optim.Adam(parameters_s_a, lr=4e-5)
        print(optimizer_s_a, n_iter)

    model.train()
    for i in range(1, n_iter + 1):

        idx = torch.randperm(INPUT_HAT.size(0))[:batch_size]

        if parameters_v:
            optimizer_w.zero_grad()
        if parameters_s_a:
            optimizer_s_a.zero_grad()

        """ Layer reconstruction loss"""

        batch_input_fp = INPUT_HAT[idx].to("cuda").requires_grad_(True)
        batch_output_fp = OUTPUT_FP[idx].to("cuda").requires_grad_(True)

        out_hat, _ = module.forward(batch_input_fp, S_X_INPUT)
        _mse = (batch_output_fp - out_hat).abs().pow(2).mean()  # MSE

        loss_sum = _mse.clone()

        _beta = _decayed_beta(i, n_iter)
        _reg_loss_sum = None
        for name, sub_module in module.named_modules():
            if isinstance(sub_module, QuantLinearWithWeight):
                _reg_loss = sub_module.f_reg(beta=_beta)
                if _reg_loss_sum is None:
                    _reg_loss_sum = sub_module.lamda * _reg_loss
                else:
                    _reg_loss_sum += sub_module.lamda * _reg_loss

                loss_sum += sub_module.lamda * _reg_loss

        loss_sum.backward()

        if parameters_v:
            optimizer_w.step()
        if parameters_s_a:
            optimizer_s_a.step()

        if i % 1000 == 0 or i == 1 or _beta == 20:
            print(
                f"Iter {i:5d} | Total loss: {loss_sum:.4f} (MSE:{_mse:.4f}, Reg:{_reg_loss_sum:.4f}) beta={_beta:.2f}"
            )
        if _beta > 0 and _reg_loss_sum < 0.00001:
            print(
                f"Iter {i:5d} | Total loss: {loss_sum:.4f} (MSE:{_mse:.4f}, Reg:{_reg_loss_sum:.4f}) beta={_beta:.2f}\n    Early stopped"
            )
            break

    torch.cuda.empty_cache()
    for name, sub_module in module.named_modules():
        if isinstance(sub_module, QuantLinearWithWeight):
            sub_module.setRoundingValues()
        elif isinstance(sub_module, QuantAct):
            print(f"    s_a : {name}, {sub_module.s_out.shape}, {sub_module.s_out}")
    print("")
    return None


def run_AdaRound(
    model, train_loader, scheme, num_samples=1024, batch_size=32, lr=0.01, n_iter=10000
):
    model.eval()

    assert scheme in ["PerLayer", "PerBlock", "PerEncoder", "PerModel"]

    total_module_cnt = 0

    cali_data = get_train_samples(train_loader, num_samples)

    module_num_cnt = 0
    if scheme == "PerLayer":
        for name, module in model.named_modules():
            if isinstance(module, QuantLinearWithWeight):
                total_module_cnt += 1
        for name, module in model.named_modules():
            if isinstance(module, QuantLinearWithWeight):
                module_num_cnt += 1
                print(f"[{module_num_cnt}/{total_module_cnt}] {name}")
                _adaround_for_a_module(model, module, cali_data, batch_size, lr, n_iter)

    elif scheme == "PerBlock":
        for name, module in model.named_modules():
            if isinstance(module, QuantLinearWithWeight) and name in [
                "conv_proj",
                "heads.0",
            ]:
                total_module_cnt += 1
            elif isinstance(module, QuantMLP) or isinstance(module, QuantMSA):
                total_module_cnt += 1
        for name, module in model.named_modules():
            if isinstance(module, QuantLinearWithWeight) and name in [
                "conv_proj",
                "heads.0",
            ]:
                module_num_cnt += 1
                print(f"[{module_num_cnt}/{total_module_cnt}] {name}")
                _adaround_for_a_module(model, module, cali_data, batch_size, lr, n_iter)
            elif isinstance(module, QuantMLP) or isinstance(module, QuantMSA):
                module_num_cnt += 1
                print(f"[{module_num_cnt}/{total_module_cnt}] {name}")
                _adaround_for_a_module(model, module, cali_data, batch_size, lr, n_iter)
    elif scheme == "PerEncoder":
        for name, module in model.named_modules():
            if isinstance(module, QuantLinearWithWeight) and name in [
                "conv_proj",
                "heads.0",
            ]:
                total_module_cnt += 1
            elif isinstance(module, QuantEncoderBlock):
                total_module_cnt += 1
        for name, module in model.named_modules():
            if isinstance(module, QuantLinearWithWeight) and name in [
                "conv_proj",
                "heads.0",
            ]:
                # [ ] input act, output of encoder's LN act 감안 안 되어있음.
                module_num_cnt += 1
                print(f"[{module_num_cnt}/{total_module_cnt}] {name}")
                _adaround_for_a_module(model, module, cali_data, batch_size, lr, n_iter)
            elif isinstance(module, QuantEncoderBlock):
                module_num_cnt += 1
                print(f"[{module_num_cnt}/{total_module_cnt}] {name}")
                _adaround_for_a_module(model, module, cali_data, batch_size, lr, n_iter)
    elif scheme == "PerModel":
        # [ ] implement PerModel adaround
        raise NotImplementedError
        # print(f"[1/1] Whole model")
        # _adaround_for_a_module(model, model, cali_data, batch_size, lr, n_iter)

    return None

import torch, time, argparse, timm
import torch.nn as nn

from .data_utils import save_inp_oup_data, get_train_samples
from model.quant_modules import LogSqrt2Quantizer


def _runner(model, module, cali_data, batch_size, lr, n_iter):
    model.eval()
    print(f"   Get calibration data for {module}")
    # [1] get {Origin FP output(A_fp_lth), Quantized input and output(X_q_lth, A_q_lth)}
    model.set_quant_mode(False, False, False, False, False)
    _, _, OUTPUT_FP, _ = save_inp_oup_data(model, module, cali_data, batch_size)
    model.set_quant_mode(True, True, True, True, True)
    INPUT_HAT, s_x_input, _, _ = save_inp_oup_data(model, module, cali_data, batch_size)
    assert 0 <= OUTPUT_FP.min() and OUTPUT_FP.max() <= 1
    assert 0 <= INPUT_HAT.min() and INPUT_HAT.max() <= 1
    print(f"    INPUT_FP : {INPUT_HAT.shape}")
    print(f"    OUTPUT_FP : {OUTPUT_FP.shape}")
    S_X_INPUT = s_x_input[0].to("cuda")
    assert s_x_input.unique().numel() == 1, s_x_input

    # > INPUT_FP : Tensor
    # > S_X_INPUT : Scalar
    # > OUTPUT_FP : Tensor

    # [2] Define the optimizer and loss function
    criterion = nn.MSELoss()
    # optimizer = torch.optim.Adam(module.params, lr=4e-5)
    # print(optimizer, n_iter)
    # print(optimizer.param_groups[0]["params"])

    model.train()
    for i in range(1, n_iter + 1):
        idx = torch.randperm(INPUT_HAT.size(0))[:batch_size]

        # if optimizer:
        #     optimizer.zero_grad()

        batch_input_fp = INPUT_HAT[idx].to("cuda").requires_grad_(True)
        batch_output_fp = OUTPUT_FP[idx].to("cuda").requires_grad_(True)

        # print(optimizer.param_groups[0]["params"])
        out_hat, _ = module._map_quantize(batch_input_fp, S_X_INPUT)
        module.inited = True
        break
        loss = criterion(out_hat, batch_output_fp)

        loss.backward()

        # if optimizer:
        #     optimizer.step()

        if i % 10 == 0 or i == 1:
            print(f"    {i} iter loss: {loss:.4f}")

            print()

    torch.cuda.empty_cache()
    module.inited = True
    return None


def run_learnable_log_quant(
    model, train_loader, num_samples=1024, batch_size=32, lr=0.01, n_iter=10000
):
    model.eval()

    total_module_cnt = 0
    print("Training model for softmax log quantizer...")
    cali_data = get_train_samples(train_loader, num_samples)

    module_num_cnt = 0

    for name, module in model.named_modules():
        if "softmax_act" in name and isinstance(module, LogSqrt2Quantizer):
            total_module_cnt += 1

    for name, module in model.named_modules():
        if "softmax_act" in name and isinstance(module, LogSqrt2Quantizer):
            module_num_cnt += 1
            print(f"[{module_num_cnt}/{total_module_cnt}] {name}")
            _runner(model, module, cali_data, batch_size, lr, n_iter)

    print("mapping Activation calibration is done.\n")

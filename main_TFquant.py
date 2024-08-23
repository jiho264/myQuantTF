import torch, time, argparse
import torch.nn as nn

from utils.data_utils import save_inp_oup_data, get_train_samples, GetDataset, evaluate
from utils.quant_ViT import QuantViT, QuantEncoderBlock
from utils.quant_modules import QuantLinearWithWeight, QuantAct
from utils.quant_blocks import QuantMLP, QuantMSA
import torchvision.models.vision_transformer as vision_transformer


def _decayed_beta(i, n_iter, _warmup=0.2):
    if i < n_iter * _warmup:
        return torch.tensor(0.0)
    else:
        # 0 ~ 1 when after 4k iter of 20k len
        decay = (i - n_iter * _warmup) / (n_iter * (1 - _warmup))
        _beta = 18 - decay * 18 + 2
        return _beta


"""
이거 input quant하는 8비트 스케일러도 학습시켜야함
근데 지금은 빠져있음
LSQ논문에서 어떻게 제시했는지 알아야 논문에 적을 수 있을듯

"""


def _adaround_for_a_module(model, module, cali_data, batch_size, lr, n_iter):
    model.eval()

    # [1] get {Origin FP output(A_fp_lth), Quantized input and output(X_q_lth, A_q_lth)}
    model.set_quant_mode(False, False, False, False, False)
    INPUT_FP, _, OUTPUT_FP, _ = save_inp_oup_data(model, module, cali_data, batch_size)
    model.set_quant_mode(True, True, True, True, True)
    _, s_x_input, _, _ = save_inp_oup_data(model, module, cali_data, batch_size)
    print(f" INPUT_FP : {INPUT_FP.shape}")
    print(f" OUTPUT_FP : {OUTPUT_FP.shape}")
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
            parameters_v.append(nn.Parameter(sub_module._v, requires_grad=True))
            print(f"    V   : {name}, {sub_module._v.shape}")
        if isinstance(sub_module, QuantAct):
            parameters_s_a.append(nn.Parameter(sub_module.s_out, requires_grad=True))
            print(f"    s_a : {name}, {sub_module.s_out.shape}")

    if parameters_v != []:
        optimizer_w = torch.optim.Adam(parameters_v, lr=1e-4)
        print(optimizer_w, n_iter)
    if parameters_s_a != []:
        optimizer_s_a = torch.optim.Adam(parameters_s_a, lr=4e-5)
        print(optimizer_s_a, n_iter)

    model.train()
    with torch.set_grad_enabled(True):
        for i in range(1, n_iter + 1):

            idx = torch.randperm(INPUT_FP.size(0))[:batch_size]

            if parameters_v:
                optimizer_w.zero_grad()
            if parameters_s_a:
                optimizer_s_a.zero_grad()

            """ Layer reconstruction loss"""

            batch_input_fp = (
                INPUT_FP[idx].clone().detach().to("cuda").requires_grad_(True)
            )
            batch_output_fp = (
                OUTPUT_FP[idx].clone().detach().to("cuda").requires_grad_(True)
            )

            a_hat, _ = module.forward(batch_input_fp, S_X_INPUT)
            _mse = (batch_output_fp - a_hat).abs().pow(2).mean()  # MSE

            loss_sum = _mse
            _reg_loss_sum = torch.tensor(0.0).to("cuda")
            for name, sub_module in module.named_modules():
                if isinstance(sub_module, QuantLinearWithWeight):
                    _beta = _decayed_beta(i, n_iter)
                    _reg_loss = sub_module.f_reg(beta=_beta)

                    loss_sum += sub_module.lamda * _reg_loss
                    _reg_loss_sum += _reg_loss

            loss_sum.backward()
            if parameters_v:
                optimizer_w.step()
            if parameters_s_a:
                optimizer_s_a.step()

            if i % 1000 == 0 or i == 1:
                print(
                    f"Iter {i:5d} | Total loss: {loss_sum:.4f} (MSE:{_mse:.4f}, Reg:{_reg_loss_sum:.4f}) beta={_beta:.2f}\n"
                )
            if _beta > 0 and _reg_loss < 0.00001:
                print(
                    f"Iter {i:5d} | Total loss: {loss_sum:.4f} (MSE:{_mse:.4f}, Reg:{_reg_loss_sum:.4f}) beta={_beta:.2f}\n    Early stopped\n"
                )
                break

    torch.cuda.empty_cache()
    for name, sub_module in module.named_modules():
        if isinstance(sub_module, QuantLinearWithWeight):
            sub_module.setRoundingValues()
    print("")
    return None


def run_AdaRound(
    model, train_loader, scheme, num_samples=1024, batch_size=32, lr=0.01, n_iter=1000
):
    model.eval()

    assert scheme in ["PerLayer", "PerBlock", "PerEncoder", "PerModel"]
    total_module_cnt = 0
    if scheme == "PerLayer":
        for name, module in model.named_modules():
            if isinstance(module, QuantLinearWithWeight):
                total_module_cnt += 1
    elif scheme == "PerBlock":
        for name, module in model.named_modules():
            if isinstance(module, QuantLinearWithWeight) and name in [
                "conv_proj",
                "heads.0",
            ]:
                total_module_cnt += 1
            elif isinstance(module, QuantMLP) or isinstance(module, QuantMSA):
                total_module_cnt += 1
    elif scheme == "PerEncoder":
        for name, module in model.named_modules():
            if isinstance(module, QuantLinearWithWeight) and name in [
                "conv_proj",
                "heads.0",
            ]:
                total_module_cnt += 1
            elif isinstance(module, QuantEncoderBlock):
                total_module_cnt += 1

    cali_data = get_train_samples(train_loader, num_samples)

    module_num_cnt = 0
    if scheme == "PerLayer":
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
                module_num_cnt += 1
                print(f"[{module_num_cnt}/{total_module_cnt}] {name}")
                _adaround_for_a_module(model, module, cali_data, batch_size, lr, n_iter)
            elif isinstance(module, QuantEncoderBlock):
                module_num_cnt += 1
                print(f"[{module_num_cnt}/{total_module_cnt}] {name}")
                _adaround_for_a_module(model, module, cali_data, batch_size, lr, n_iter)
    elif scheme == "PerModel":
        print(f"[1/1] Whole model")
        _adaround_for_a_module(model, model, cali_data, batch_size, lr, n_iter)

    print(module_num_cnt)
    return None


def main(main_args={}, args_w={}, args_a={}, args_softmax={}, args_ln={}, args_gelu={}):
    main_args = {
        "arch": "ViT_B_16",
        "batch_size": 128,
        "num_samples": 1024,
    }

    args_w = {"scheme": "AbsMaxQuantizer", "bit_width": 8, "per_channel": True}
    args_w.update({"AdaRound": "PerEncoder"})

    args_a = {
        "scheme": "MovAvgAbsMaxQuantizer",
        "bit_width": 8,
        "per_channel": False,
        # below values are default in the class
        "momentum": 0.95,
        "batches": 16,
    }
    args_gelu = {"bit_width": 8}
    args_softmax = {"bit_width": 16}
    args_ln = {
        "bit_width": 8
    }  # FIXME layer norm bit width is no matter. have to change another setting method

    calib_len = args_a.get("batches", 16)

    argses = [main_args, args_w, args_a, args_softmax, args_ln, args_gelu]
    names = ["main_args", "weight", "activation", "softmax", "layer_norm", "gelu"]
    for name, args in zip(names, argses):
        print(f"\n- {name} params:")
        for k, v in args.items():
            print(f"    - {k}: {v}")
    print(f"\n- Identity addition : INT16 (The input of each LayerNorm)")
    print(f"\n- Activation of Softmax(Q@K/d_K) (attn_map) : UINT8")

    if main_args["arch"] == "ViT_B_16":
        model = vision_transformer.vit_b_16(
            weights=vision_transformer.ViT_B_16_Weights.IMAGENET1K_V1
        )
    else:
        raise NotImplementedError

    model.eval().to("cuda")
    model = QuantViT(model, args_w, args_a, args_softmax, args_ln, args_gelu)

    model.eval().to("cuda")

    _batch_size = main_args["batch_size"]
    train_loader, test_loader = GetDataset(batch_size=_batch_size)

    """ 여기 지우고 돌리면 dynamic act quantization """
    if args_a != {}:
        """calibration for activation"""
        _, _ = evaluate(model, test_loader, calib_len, "cuda")
        print("Activation calibration is done.\n")

    if args_w.get("AdaRound", None):
        scheme = args_w.get("AdaRound")
        run_AdaRound(model, train_loader, scheme)
        print(f"AdaRound for {scheme} weights is done.")

    """ evaluation """
    _top1, _ = evaluate(model, test_loader, len(test_loader), "cuda")
    # _top1, _ = evaluate(model, test_loader, 1, "cuda")
    print(
        f"\n    Quantized model Evaluation accuracy on 50000 images, {_top1.avg:2.3f}%"
    )


if __name__ == "__main__":

    # parser = argparse.ArgumentParser(
    #     description="myAdaRound", formatter_class=argparse.ArgumentDefaultsHelpFormatter
    # )
    # parser.add_argument("--arch", default="ViT_B_16", type=str, help="architecture")
    # parser.add_argument("--seed", default=0, type=int, help="seed")
    # parser.add_argument(
    #     "--batch_size", default=128, type=int, help="batch size for evaluation"
    # )

    # parser.add_argument(
    #     "--num_samples",
    #     default=1024,
    #     type=int,
    #     help="number of samples for calibration",
    # )

    # """ weight quantization"""
    # parser.add_argument(
    #     "--scheme_w",
    #     default="MinMaxQuantizer",
    #     type=str,
    #     help="quantization scheme for weights",
    #     choices=quantizerDict,
    # )
    # parser.add_argument(
    #     "--dstDtypeW",
    #     default="INT8",
    #     type=str,
    #     help="destination data type",
    #     choices=["INT4", "INT8"],
    # )

    # parser.add_argument(
    #     "--paper",
    #     default="",
    #     type=str,
    #     help="paper name",
    #     choices=["AdaRound", "BRECQ", "PDquant"],
    # )

    # parser.add_argument(
    #     "--per_channel",
    #     action="store_false",
    #     help="per channel quantization for weights",
    # )
    # parser.add_argument(
    #     "--batch_size_AdaRound", default=32, type=int, help="batch size for AdaRound"
    # )
    # parser.add_argument(
    #     "--p", default=2.4, type=float, help="L_p norm for NormQuantizer"
    # )
    # parser.add_argument(
    #     "--lr", default=0.001, type=float, help="learning rate for AdaRound"
    # )

    # """ Activation quantization """
    # parser.add_argument(
    #     "--scheme_a",
    #     default="MinMaxQuantizer",
    #     type=str,
    #     help="quantization scheme for activations",
    #     choices=quantizerDict,
    # )
    # parser.add_argument(
    #     "--dstDtypeA",
    #     default="INT8",
    #     type=str,
    #     help="destination data type",
    #     choices=["INT4", "INT8", "FP32"],
    # )
    # parser.add_argument(
    #     "--head_stem_8bit",
    #     action="store_true",
    #     help="Head and stem are 8 bit. default is 4 bit (fully quantized)",
    # )

    # """ Setup """
    # args = parser.parse_args()

    # def _get_args_dict(args):
    #     """weight"""
    #     weight_quant_params = dict(
    #         scheme=args.scheme_w,
    #         dstDtype=args.dstDtypeW,
    #         per_channel=args.per_channel,
    #     )
    #     if args.scheme_w == "NormQuantizer":
    #         weight_quant_params.update(dict(p=args.p))

    #     """ activation """
    #     act_quant_params = {}
    #     if args.dstDtypeA != "FP32":
    #         if args.paper:
    #             act_quant_params = dict(
    #                 scheme=args.scheme_a,
    #                 dstDtype=args.dstDtypeA,
    #                 per_channel=False,  # activation quantization is always per layer
    #             )
    #             if args.scheme_a == "NormQuantizer":
    #                 act_quant_params.update(dict(p=args.p))

    #     """ Main """
    #     main_args = dict(
    #         arch=args.arch,
    #         batch_size=args.batch_size,
    #         num_samples=args.num_samples,
    #     )

    #     if args.paper:
    #         main_args.update(dict(batch_size_AdaRound=args.batch_size_AdaRound))
    #         main_args.update(dict(lr=args.lr))
    #         if args.paper == "AdaRound":
    #             weight_quant_params.update(dict(AdaRound=True))
    #         elif args.paper == "BRECQ":
    #             weight_quant_params.update(dict(BRECQ=True))
    #         elif args.paper == "PDquant":
    #             weight_quant_params.update(dict(PDquant=True))
    #         else:
    #             print("error")
    #             exit()
    #         weight_quant_params["per_channel"] = True
    #         args.per_channel = True  # always Per-CH when using AdaRound for weights
    #     if (
    #         args.head_stem_8bit
    #         and args.dstDtypeW == "INT4"
    #         and args.dstDtypeA == "INT4"
    #     ):
    #         main_args.update(dict(head_stem_8bit=True))

    #     return weight_quant_params, act_quant_params, main_args

    # weight_quant_params, act_quant_params, main_args = _get_args_dict(args=args)

    # """ case naming """
    # _case_name = f"{args.arch}_"
    # if args.paper:
    #     _case_name += args.paper + "_"
    # _case_name += args.scheme_w + "_"
    # if args.head_stem_8bit:
    #     _case_name += "head_stem_8bit_"
    # _case_name += "CH_" if args.per_channel else "Layer_"
    # _case_name += "W" + args.dstDtypeW[-1]
    # _case_name += "A" + "32" if args.dstDtypeA == "FP32" else "A" + args.dstDtypeA[-1]
    # if args.scheme_w == "NormQuantizer":
    #     _case_name += "_p" + str(args.p)
    # if args.paper:
    #     _case_name += "_RoundingLR" + str(args.lr)

    # print(f"\nCase: [ {_case_name} ]")
    # for k, v in main_args.items():
    #     print(f"    - {k}: {v}")
    # print(f"\n- weight params:")
    # for k, v in weight_quant_params.items():

    #     print(f"    - {k}: {v}")
    # print(f"\n- activation params:")
    # for k, v in act_quant_params.items():
    #     print(f"    - {k}: {v}")
    # print("")
    # seed_all(args.seed)

    STARTTIME = time.time()
    main()
    # main(weight_quant_params, act_quant_params, main_args)
    print(f"Total time: {time.time() - STARTTIME:.2f} sec")

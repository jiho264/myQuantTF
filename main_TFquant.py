import torch, time, argparse
import torch.nn as nn

from utils.data_utils import save_inp_oup_data, _get_train_samples, GetDataset, evaluate
from utils.quant_ViT import QuantViT
import torchvision.models.vision_transformer as vision_transformer


# def _computeAdaRoundValues(model, layer, cali_data, batch_size, lr, n_iter):
#     model.eval()
#     # [1] get {Origin FP output(A_fp_lth), Quantized input and output(X_q_lth, A_q_lth)}
#     A_fp_lth = layer.weightQuantizer.Quantizer.fp_outputs
#     X_q_lth, _ = save_inp_oup_data(model, layer, cali_data)

#     # [2] Define the optimizer and loss function
#     optimizer_w = torch.optim.Adam([layer.weightQuantizer.Quantizer._v], lr=lr)

#     print(optimizer_w, n_iter, layer.weightQuantizer.Quantizer._v.shape)

#     model.train()

#     for i in range(1, n_iter + 1):

#         idx = torch.randperm(X_q_lth.size(0))[:batch_size]

#         optimizer_w.zero_grad()

#         """ Layer reconstruction loss"""
#         _tmp_A_q_lth = layer.forward(X_q_lth[idx])
#         _mse = (A_fp_lth[idx] - _tmp_A_q_lth).abs().pow(2).mean()
#         _beta = layer.weightQuantizer.Quantizer._decayed_beta(i, n_iter)
#         _reg_loss = layer.weightQuantizer.Quantizer.f_reg(beta=_beta)

#         loss = _mse + layer.weightQuantizer.Quantizer.lamda * _reg_loss

#         loss.backward()
#         optimizer_w.step()

#         if i % 1000 == 0 or i == 1:
#             print(
#                 f"Iter {i:5d} | Total loss: {loss:.4f} (MSE:{_mse:.4f}, Reg:{_reg_loss:.4f}) beta={_beta:.2f}"
#             )
#         if _beta > 0 and _reg_loss < 0.00001:
#             print(
#                 f"Iter {i:5d} | Total loss: {loss:.4f} (MSE:{_mse:.4f}, Reg:{_reg_loss:.4f}) beta={_beta:.2f} ... Early stopping\n"
#             )
#             break

#     del layer.weightQuantizer.Quantizer.fp_outputs
#     del X_q_lth, A_fp_lth

#     layer.weightQuantizer.Quantizer.setRoundingValues()
#     return None


# def run_AdaRound(
#     model, train_loader, num_samples=1024, batch_size=32, lr=0.01, n_iter=25000
# ):

#     model.eval()

#     cali_data = _get_train_samples(train_loader, num_samples)

#     for name, module in model.named_modules():
#         if isinstance(module, QuantLinearLayer):
#             model.vit_w_quant = False
#             _, FP_OUTPUT = save_inp_oup_data(model, module, cali_data)
#             model.vit_w_quant = True
#             module.weightQuantizer.Quantizer.fp_outputs = FP_OUTPUT
#             print(name)
#             print("   FP_OUTPUT shape", FP_OUTPUT.shape)
#             _computeAdaRoundValues(model, module, cali_data, batch_size, lr, n_iter)
#             torch.cuda.empty_cache()
#     return None


def main(main_args={}, args_w={}, args_a={}, args_softmax={}, args_ln={}, args_gelu={}):
    main_args = {
        "arch": "ViT_B_16",
        "batch_size": 128,
        "num_samples": 1024,
    }

    args_w = {"scheme": "AbsMaxQuantizer", "bit_width": 8, "per_channel": True}
    # args_w.update({"scheme": "AdaRoundQuantizer"})
    args_a = {
        "scheme": "MovAvgAbsMaxQuantizer",
        "bit_width": 8,
        "per_channel": False,
        #  75.198%
        # below values are default in the class
        # "momentum": 0.9,
        # "batches": 16,
    }
    args_gelu = {"bit_width": 8}
    args_softmax = {"bit_width": 16}
    args_ln = {"bit_width": 8}

    argses = [main_args, args_w, args_a, args_softmax, args_ln, args_gelu]
    names = ["main_args", "weight", "activation", "softmax", "layer_norm", "gelu"]
    for name, args in zip(names, argses):
        print(f"\n- {name} params:")
        for k, v in args.items():
            print(f"    - {k}: {v}")

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

    # if args_a != {}:
    #     """ calibration for activation """
    #     _, _ = evaluate(model, test_loader, 16, "cuda")
    #     print("Calibration done \n")

    # if args_w.get("scheme") == "AdaRoundQuantizer":
    #     run_AdaRound(model, train_loader)

    """ evaluation """
    # _top1, _ = evaluate(model, test_loader, len(test_loader), "cuda")
    _top1, _ = evaluate(model, test_loader, 1, "cuda")
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

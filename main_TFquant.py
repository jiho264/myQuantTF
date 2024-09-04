import torch, time, argparse
import torch.nn as nn
import torchvision.models.vision_transformer as vision_transformer

from utils import *


def main(args_main={}, args_w={}, args_a={}, args_softmax={}, args_ln={}, args_gelu={}):
    """PREPARE"""
    args_main = {
        "arch": "ViT_B_16",
        "batch_size": 128,
        "num_samples": 1024,
    }

    args_w = {"scheme": "AbsMaxQuantizer", "bit_width": 4, "per_channel": True}
    # args_w.update({"AdaRound": "PerLayer"})

    """NOW IS ONLY FOR WEIGHTS"""

    args_a = {
        "scheme": "MovAvgAbsMaxQuantizer",
        "bit_width": 8,
        "idadd_bit_width": 18,
        "per_channel": False,
        # below values are default in the class
        "momentum": 0.95,
        "batches": 4,
        # "batches": 4,
    }
    args_gelu = {
        "sigmoid_bit_width": 8,
        "left_shift_for_exp": 23,
        # "act_quant_bit_width": 4,  # LogSqrt2Quantizer
        "act_quant_bit_width": 8,  # QuantAct
    }  # I-ViT default
    ## bit width : INT arithmetic으로 exp를 구하는 과정에서, e / (e + e.max)인 항이 있는데, 여기서 반환 값을 몇 비트로 펼칠 것인지 결정하는 숫자.

    args_softmax = {
        "bit_width": 17,  # UINT16 of softmax output
        "left_shift_for_exp": 15,
        "act_quant_bit_width": 4,  # LogSqrt2Quantizer
        # "act_quant_bit_width": 8,  # QuantAct
    }  # I-ViT default
    # # bit width : softmax의 out이 0~1인데, 이 값을 몇 비트에 펼쳐서 반환할 것인지 결정하는 숫자

    args_ln = {
        "using": True,
    }  # FIXME layer norm bit width is no matter. have to change another setting method

    calib_len = args_a.get("batches", 16)

    argses = [args_main, args_w, args_a, args_softmax, args_ln, args_gelu]
    names = ["main_args", "weight", "activation", "softmax", "layer_norm", "gelu"]
    for name, args in zip(names, argses):
        print(f"\n- {name} params:")

        for k, v in args.items():
            print(f"    - {k}: {v}")

        if name == "softmax" and args_softmax != {}:
            print(f"    - Activation of Softmax(Q@K/d_K) (attn_map) : UINT8")
        elif name == "activation" and args_a != {}:
            print(f"    - Identity addition : INT16 (The input of each LayerNorm)")
    print("")

    """ Quantization """
    if args_main["arch"] == "ViT_B_16":
        model = vision_transformer.vit_b_16(
            weights=vision_transformer.ViT_B_16_Weights.IMAGENET1K_V1
        )
    else:
        raise NotImplementedError

    model.eval().to("cuda")
    model = QuantViT(model, args_w, args_a, args_softmax, args_ln, args_gelu)

    model.eval().to("cuda")

    _batch_size = args_main["batch_size"]
    train_loader, test_loader = GetDataset(batch_size=_batch_size)

    """ 여기 지우고 돌리면 dynamic act quantization """
    print("Training model for calibration...")
    if args_a != {}:
        """calibration for activation"""
        _, _ = evaluate(model, test_loader, calib_len, "cuda")
        print("Activation calibration is done.\n")

    if args_w.get("AdaRound", None):
        scheme = args_w.get("AdaRound")
        run_AdaRound(model, train_loader, scheme)
        print(f"AdaRound for {scheme} weights is done.")

    """ evaluation """
    _top1, _top5 = evaluate(model, test_loader, len(test_loader), "cuda")
    # _top1, _top5 = evaluate(model, test_loader, 1, "cuda")
    print(
        f"\n    Quantized model Evaluation accuracy on 50000 images, {_top1.avg:2.3f}%, {_top5.avg:2.3f}%"
    )


if __name__ == "__main__":
    seed_all(0)
    STARTTIME = time.time()
    main()
    print(f"Total time: {time.time() - STARTTIME:.2f} sec")

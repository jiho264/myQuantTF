import torch, tqdm
from torch import Tensor


#################################################################################################
## 1. Prepare the dataset and utility functions
#################################################################################################


def seed_all(seed=0):
    # random.seed(seed)
    # os.environ["PYTHONHASHSEED"] = str(seed)
    # np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU..
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=":f"):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = "{name} {val" + self.fmt + "} ({avg" + self.fmt + "})"
        return fmtstr.format(**self.__dict__)


def GetDataset(batch_size=64):
    import torchvision
    import torchvision.transforms as transforms

    train_dataset = torchvision.datasets.ImageNet(
        root="data/ImageNet",
        split="train",
        transform=transforms.Compose(
            [
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        ),
    )
    test_dataset = torchvision.datasets.ImageNet(
        root="data/ImageNet",
        split="val",
        transform=transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        ),
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=8,
        pin_memory=True,
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=8,
        pin_memory=True,
    )

    return train_loader, test_loader


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def evaluate(model, data_loader, neval_batches, device):
    model.eval().to(device)
    top1 = AverageMeter("Acc@1", ":6.2f")
    top5 = AverageMeter("Acc@5", ":6.2f")
    cnt = 0
    with torch.no_grad():
        for image, target in tqdm.tqdm(data_loader):
            # for image, target in data_loader:
            image, target = image.to(device), target.to(device)
            output = model(image)
            # loss = criterion(output, target)
            cnt += 1
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            top1.update(acc1[0], image.size(0))
            top5.update(acc5[0], image.size(0))
            if cnt >= neval_batches:
                return top1, top5

    return top1, top5


#################################################################################################
## 2. Quantized module
#################################################################################################

import torch.nn as nn
import torch.nn.functional as F


class UniformAffineQuantizer(nn.Module):
    def __init__(self, org_tensor, args):
        super(UniformAffineQuantizer, self).__init__()
        """
        example... 
            args = {
                "scheme": "AbsMaxQuantizer",
                "dstDtype": "INT8",
            }
        """
        _DtypeStr = args.get("dstDtype")
        assert _DtypeStr in [
            "INT8",
            "INT4",
        ], f"Unknown quantization type: {_DtypeStr}. Only support INT8, INT4. \n \
            If using INT8 with output of the ReLU activation function, we will use UINT8 instead. \n \
            Therefore, please only determine the BIT WITDH."

        self._n_bits = int(_DtypeStr[-1])  # "INT8" -> 8
        self._repr_min, self._repr_max = None, None

        # self.per_channel = args.get("per_channel") if args.get("per_channel") else False
        # always using per_channel quantization
        self.per_channel = True
        self._n_ch = len(org_tensor.size()) if self.per_channel else 1
        self._scaler = None
        self._zero_point = None
        self.one_side_dist = None  # {"pos", "neg", "no"}

        self._define_repr_min_max(org_tensor)

    def _define_repr_min_max(self, input: Tensor):
        if self.one_side_dist is None:
            self.one_side_dist = (
                "pos" if input.min() >= 0.0 else "neg" if input.max() <= 0.0 else "no"
            )
        if self.one_side_dist != "no":
            self._repr_min = 0  # "UINT8" -> 0
            self._repr_max = 2 ** (self._n_bits) - 1  # "UINT8" -> 255
            print(f"    1D search with UINT{self._n_bits}")
        else:  # 2-d search
            self._repr_min = -(2 ** (self._n_bits - 1))  # "INT8" -> -128
            self._repr_max = 2 ** (self._n_bits - 1) - 1  # "INT8" -> 127
            print(f"    2D search with INT{self._n_bits}")

    def round_ste(self, input: torch.Tensor):
        """
        Implement Straight-Through Estimator for rounding operation.
        """
        return (input.round() - input).detach() + input

    def compute_qparams(self, _min, _max) -> None:
        # Origin tensor shape: [out_channel, in_channel, k, k]
        # per_ch Qparam shape: [n_channel, 1, 1, 1]

        scaler = (_max - _min) / (self._repr_max - self._repr_min)
        self._scaler = scaler.view(-1, *([1] * (self._n_ch - 1)))

        _min = _min.view(-1, *([1] * (self._n_ch - 1)))
        self._zero_point = -self.round_ste(_min / self._scaler) + self._repr_min

    def _quantize(self, input: Tensor) -> Tensor:
        return torch.clamp(
            self.round_ste(input / self._scaler) + self._zero_point,
            self._repr_min,
            self._repr_max,
        )

    def _dequantize(self, input: Tensor) -> Tensor:
        if torch.all(input == input.floor()).item():
            return (input - self._zero_point) * self._scaler
        else:
            """ verifing the quantization """
            raise ValueError("The input tensor is not quantized.")

    def forward(self, x: Tensor) -> Tensor:
        return self._dequantize(self._quantize(x))


class AbsMaxQuantizer(UniformAffineQuantizer):
    def __init__(self, org_tensor, args):
        """
        [ Absolute Maximum Quantization ]
        - When zero_point is zero, the quantization range is symmetric.
            (Uniform Symmetric Quantization)
        - range: [-max(abs(x)), max(abs(x))]
        """
        super(AbsMaxQuantizer, self).__init__(org_tensor, args)

        _AbsMax = None
        if self.per_channel == True:
            _AbsMax = org_tensor.view(org_tensor.size(0), -1).abs().max(dim=1).values
        else:
            _AbsMax = org_tensor.abs().max()

        if self.one_side_dist == "no":
            # if s8, scaler = 2 * org_weight.abs().max() / (127 - (-128))
            #               = org_weight.abs().max() - (-org_weight.abs().max()) / (127 - (-128))
            self.compute_qparams(-_AbsMax, _AbsMax)
            # will convert to (-max, max) -> (-128, 127)
        elif self.one_side_dist == "pos":
            # if u8, scaler = org_weight.abs().max() / (255 - 0)
            self.compute_qparams(torch.zeros_like(_AbsMax), _AbsMax)
            # will convert to (0, max) -> (0, 255)
        else:
            print("This distribution is unexpected. plaese check the input data.")
            # self.compute_qparams(_AbsMax, torch.zeros_like(_AbsMax))
            raise ValueError("Unknown distribution type.")

        # if s8 or u8, zero_point = 0
        self.zero_point = torch.zeros_like(self._scaler)


class MinMaxQuantizer(UniformAffineQuantizer):
    def __init__(self, org_tensor, args):
        """
        [ Min-Max Quantization ]
        - The quantization range is asymmetric.
        - range: [min(x), max(x)]
        - https://pytorch.org/docs/stable/generated/torch.ao.quantization.observer.MinMaxObserver
        """
        super(MinMaxQuantizer, self).__init__(org_tensor, args)

        if self.per_channel == True:
            _min = org_tensor.view(org_tensor.size(0), -1).min(dim=1).values
            _max = org_tensor.view(org_tensor.size(0), -1).max(dim=1).values
        else:
            _min = org_tensor.min()
            _max = org_tensor.max()

        # if s8, scaler = (_max - _min) / (127 - (-128))
        # if s8, zero_point = -_min / scaler + (-128)

        # if u8, scaler = (_max - _min) / (255 - 0)
        # if u8, zero_point = -_min / scaler + 0

        # Always using same equation.
        self.compute_qparams(_min, _max)

quantizerDict = {
    "AbsMaxQuantizer": AbsMaxQuantizer,
    "MinMaxQuantizer": MinMaxQuantizer,
    # "NormQuantizer": NormQuantizer,
    # "OrgNormQuantizerCode": OrgNormQuantizerCode,
}


class StraightThrough(nn.Module):
    def __int__(self):
        super().__init__()

    def forward(self, input):
        return input

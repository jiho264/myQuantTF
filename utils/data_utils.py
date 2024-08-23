import torch, tqdm
import torch.nn as nn
import random
import os
import numpy as np

#################################################################################################
## 1. Prepare the dataset and utility functions
#################################################################################################


def seed_all(seed=0):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
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


class StopForwardException(Exception):
    """
    Used to throw and catch an exception to stop traversing the graph
    """

    pass


class DataSaverHook:
    """
    Forward hook that stores the input and output of a block
    """

    def __init__(self, store_input=False, store_output=False, stop_forward=False):
        self.store_input = store_input
        self.store_output = store_output
        self.stop_forward = stop_forward

        self.input_store = None
        self.output_store = None

    def __call__(self, module, input_batch, output_batch):
        if self.store_input:
            self.input_store = input_batch
        if self.store_output:
            self.output_store = output_batch
        if self.stop_forward:
            raise StopForwardException


def get_train_samples(train_loader, num_samples):
    # calibration data loader code in AdaRound
    train_data = []
    for batch in train_loader:
        train_data.append(batch[0])
        if len(train_data) * batch[0].size(0) >= num_samples:
            break
    return torch.cat(train_data, dim=0)[:num_samples]


class GetLayerInpOut:
    def __init__(
        self,
        model,
        layer,
        device: torch.device = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        self.model = model
        self.layer = layer
        self.device = device
        self.data_saver = DataSaverHook(
            store_input=True, store_output=True, stop_forward=True
        )

    def __call__(self, model_input):
        self.model.eval()
        handle = self.layer.register_forward_hook(self.data_saver)
        with torch.no_grad():
            try:
                _ = self.model(model_input.to(self.device))
            except StopForwardException:
                pass

        handle.remove()
        self.model.train()

        """
        input : (x_hat, s_x)
        output : (x_hat, s_x)
        """

        for _tensor in [self.data_saver.input_store, self.data_saver.output_store]:
            if isinstance(_tensor, tuple):
                for t in _tensor:
                    if isinstance(t, torch.Tensor):
                        t = t.detach()
            else:
                if isinstance(t, torch.Tensor):
                    tensor = tensor.detach()

        return (
            self.data_saver.input_store,  # Tuple of (x_hat, s_x)
            self.data_saver.output_store,  # Tuple of (x_hat, s_x)
        )


def save_inp_oup_data(
    model,
    layer,
    cali_data: torch.Tensor,
    batch_size: int = 128,
    keep_gpu: bool = False,
):
    """
    Save input data and output data of a particular layer/block over calibration dataset.

    :param model: QuantModel
    :param layer: QuantModule or QuantBlock
    :param cali_data: calibration data set
    :param asym: if Ture, save quantized input and full precision output
    :param act_quant: use activation quantization
    :param batch_size: mini-batch size for calibration
    :param keep_gpu: put saved data on GPU for faster optimization
    :return: input and output data
    """
    device = next(model.parameters()).device
    get_inp_out = GetLayerInpOut(model, layer, device=device)
    cached_batches = []
    torch.cuda.empty_cache()

    for i in range(int(cali_data.size(0) / batch_size)):
        cur_inp, cur_out = get_inp_out(cali_data[i * batch_size : (i + 1) * batch_size])
        # if you lack VRAM, I am so sorry.
        # plz use the smaller batch size.
        cached_batches.append((cur_inp, cur_out))

    cached_inps_x_hat = torch.cat([x[0][0] for x in cached_batches])
    cached_inps_s_x = torch.tensor([x[0][1] for x in cached_batches])
    cached_outs_x_hat = torch.cat([x[1][0] for x in cached_batches])
    try:
        cached_outs_s_x = torch.tensor([x[1][1] for x in cached_batches])
    except:
        cached_outs_s_x = None

    torch.cuda.empty_cache()

    if keep_gpu is False:
        for _tensor in [
            cached_inps_x_hat,
            cached_inps_s_x,
            cached_outs_x_hat,
            cached_outs_s_x,
        ]:
            if isinstance(_tensor, torch.Tensor):
                _tensor = _tensor.cpu()

    return (cached_inps_x_hat, cached_inps_s_x, cached_outs_x_hat, cached_outs_s_x)

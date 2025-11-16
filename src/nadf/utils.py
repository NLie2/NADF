import os
from itertools import islice

import numpy as np
import torch
import torch.nn as nn
from art.attacks.evasion import ProjectedGradientDescent
from art.estimators.classification import PyTorchClassifier
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms

#! Most functions written by Melhi

STATS = {
    "cifar10": {"mean": [0.491, 0.482, 0.447], "std": [0.247, 0.243, 0.262]},
    "cifar100": {"mean": [0.5071, 0.4867, 0.4408], "std": [0.2675, 0.2565, 0.2761]},
    "mnist": {"mean": [0.1307], "std": [0.3081]},
    "bmnist": {"mean": [0.1307], "std": [0.3081]},
    "svhn": {"mean": [0.4377, 0.4438, 0.4728], "std": [0.1980, 0.2010, 0.1970]},
}

dataset_info = {
    "cifar10": {
        "num_samples": {"train": 45000, "val": 5000, "test": 10000},
        "img_size": 32 * 32 * 3,
        "num_classes": 10,
    },
    "mnist": {"num_samples": {"train": 54000, "val": 6000, "test": 10000}, "img_size": 28 * 28, "num_classes": 10},
}


# * DATA LOADING UITLS
class Normalize:
    def __init__(self, mean, std):
        if len(mean) > 1:
            raise NotImplementedError
        self.mean = mean[0]
        self.std = std[0]

    def __call__(self, tensor):
        # Scale tensor from [input_min, input_max] to [output_min, output_max]
        return (tensor - self.mean) / self.std

    # def __repr__(self):
    #     return self.__class__.__name__ + f'(input_range=({self.input_min}, {self.input_max}), output_range=({self.output_min}, {self.output_max}))'


# From ChatGPT
class IndexedDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

        # If dataset is a Subset, check its base dataset
        base = dataset.dataset if isinstance(dataset, torch.utils.data.Subset) else dataset

        if hasattr(base, "data"):
            self.data = base.data
        if hasattr(base, "targets"):
            self.targets = base.targets
        if hasattr(base, "classes"):
            self.classes = base.classes
        if hasattr(base, "class_to_idx"):
            self.class_to_idx = base.class_to_idx
        if hasattr(base, "n_classes"):
            self.n_classes = base.n_classes

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        x, y = self.dataset[idx]
        return idx, x, y


def get_data(args=None, return_biased=False, abtrain=False, **kwargs):
    # Override data path from environment variable if available
    data_path_override = os.getenv("DATA_PATH")
    if data_path_override and hasattr(args, "path"):
        args.path = data_path_override

    # Check if dataset should be downloaded (default to False to avoid SSL issues on cluster)
    download_dataset = os.getenv("DOWNLOAD_DATASET", "false").lower() in ["true", "1", "yes"]

    eval_loaders = {}
    if args.dataset == "cifar100":
        data_class = "CIFAR100"
        num_classes = 100
        input_dim = 32 * 32 * 3
        stats = {"mean": [0.5071, 0.4867, 0.4408], "std": [0.2675, 0.2565, 0.2761]}
    elif args.dataset == "mnist":
        data_class = "MNIST"
        num_classes = 10
        input_dim = 28 * 28
        stats = {"mean": [0.1307], "std": [0.3081]}

    elif args.dataset in ["binarymnist", "bmnist"]:
        data_class = "BinaryMNIST"
        num_classes = 2
        input_dim = 28 * 28
        stats = {"mean": [0.1307], "std": [0.3081]}

    elif "cifar10" in args.dataset:
        data_class = "CIFAR10"
        num_classes = 10
        input_dim = 32 * 32 * 3
        stats = {"mean": [0.491, 0.482, 0.447], "std": [0.247, 0.243, 0.262]}
    elif "svhn" in args.dataset:
        data_class = "SVHN"
        num_classes = 10
        input_dim = 32 * 32 * 3
        stats = {"mean": [0.4377, 0.4438, 0.4728], "std": [0.1980, 0.2010, 0.1970]}
    else:
        raise ValueError("unknown dataset")

    # TODO: These statistcis can cause a leak if CIFAR10, MNIST etc. use a validation set.
    trans = [
        transforms.ToTensor(),
        lambda t: t.type(torch.get_default_dtype()),
        Normalize(**stats) if "dcase" in args.dataset else transforms.Normalize(**stats),
    ]

    split = {"split": "test"} if "svhn" in args.dataset else {"train": False}

    # Set download=False to avoid SSL issues on cluster where data already exists
    # Can be overridden with DOWNLOAD_DATASET environment variable
    te_data = getattr(datasets, data_class)(
        root=args.path, download=download_dataset, transform=transforms.Compose(trans), **split
    )

    tr_data = None
    val_data = None
    split = {"split": "train"} if "svhn" in args.dataset else {"train": True}
    if getattr(args, "val_criterion", None) and args.val_ratio > 0 and args.val_ratio < 1:
        tr_full_data = getattr(datasets, data_class)(
            root=args.path, download=download_dataset, transform=transforms.Compose(trans), **split
        )

        tr_full_eval_data = getattr(datasets, data_class)(
            root=args.path, download=download_dataset, transform=transforms.Compose(trans), **split
        )

        val_full_data = getattr(datasets, data_class)(
            root=args.path, download=download_dataset, transform=transforms.Compose(trans), **split
        )

        dataset_length = len(tr_full_data)
        val_len = int(args.val_ratio * dataset_length)
        train_len = dataset_length - val_len

        generator = torch.Generator().manual_seed(args.seed)

        tr_indices_subset, val_indices_subset = torch.utils.data.random_split(
            val_full_data, [train_len, val_len], generator=generator
        )

        tr_data = Subset(tr_full_data, tr_indices_subset.indices)
        tr_eval_data = Subset(tr_full_eval_data, tr_indices_subset.indices)
        val_data = Subset(val_full_data, val_indices_subset.indices)
        val_data = IndexedDataset(val_data)
    else:
        tr_data = getattr(datasets, data_class)(
            root=args.path, download=download_dataset, transform=transforms.Compose(trans), **split
        )
        tr_eval_data = getattr(datasets, data_class)(
            root=args.path, download=download_dataset, transform=transforms.Compose(trans), **split
        )
    tr_data, tr_eval_data, te_data = IndexedDataset(tr_data), IndexedDataset(tr_eval_data), IndexedDataset(te_data)

    # Fallback to training data if eval data is not available
    if tr_eval_data is None:
        tr_eval_data = tr_data

        #     tra
    # get tr_loader for train/eval and te_loader for eval

    train_loader = DataLoader(
        dataset=tr_data,
        batch_size=args.batch_size_train,
        shuffle=True,
    )
    eval_loaders["train"] = DataLoader(
        dataset=tr_eval_data,
        batch_size=args.batch_size_eval,
        shuffle=True,
    )
    eval_loaders["test"] = DataLoader(
        dataset=te_data,
        batch_size=args.batch_size_eval,
        shuffle=True,
    )
    if getattr(args, "val_criterion", ""):
        eval_loaders["val"] = DataLoader(
            dataset=val_data,
            batch_size=args.batch_size_eval,
            shuffle=True,
        )

    return train_loader, eval_loaders, num_classes, input_dim


def get_samples_from_loader(data_loader, num_samples, seed=0, device=None, bias=False, get_idx=False):
    batch_size = data_loader.batch_size
    num_batches = num_samples // batch_size
    if num_samples % batch_size:
        num_batches += 1

    with torch.random.fork_rng():
        # Set a local seed
        torch.manual_seed(seed)
        data_loader_iter = iter(data_loader)
        samples = list(islice(data_loader_iter, num_batches))
    samples = samples[:num_samples]
    idx, images, labels = (
        torch.concatenate([sample[0] for sample in samples]),
        torch.vstack([sample[1] for sample in samples]),
        torch.concatenate([sample[2] for sample in samples]),
    )
    result = images, labels
    if get_idx:
        result = idx, *result
    if device:
        result = (res.to(device) for res in result)
    return result


# * MODEL UTILS
def get_activation(activation):
    """Get activation function by name."""
    if activation == "relu":
        return nn.ReLU()
    elif activation == "lerelu":
        return nn.LeakyReLU()
    elif activation == "lerelu01":
        return nn.LeakyReLU(negative_slope=0.1)
    elif activation == "tanh":
        return nn.Tanh()
    elif activation == "linear":
        return nn.Identity()
    elif activation == "sigmoid":
        return nn.Sigmoid()
    else:
        raise KeyError("Activation function not recognized.")


def get_model_details(
    folder,
    get_model=True,
    model_iter=-1,
    bias=True,
    get_hist=True,
    get_data_loaders=True,
    get_samples=True,
    split="test",
    num_samples=1000,
    layer_idx=-1,
    seed=None,
    device=None,
    repr_eval_mode=False,
):
    r = {}

    # Determine the device for map_location - ALWAYS use CPU if CUDA not available
    map_device = "cpu"  # Always use CPU to avoid CUDA issues

    # Load args with map_location
    r["args"] = args = torch.load(os.path.join(folder, "args.info"), map_location=map_device, weights_only=False)

    seed = seed if seed else args.seed
    device = device if device else "cpu"  # Force CPU if device not specified

    if get_hist:
        # Load history files with map_location
        r["eval_hist"] = torch.load(
            os.path.join(folder, "evaluation_history.hist"), map_location=map_device, weights_only=False
        )
        r["train_hist"] = torch.load(
            os.path.join(folder, "training_history.hist"), map_location=map_device, weights_only=False
        )

    if get_model:
        # Load model with map_location
        net_path = (
            os.path.join(folder, "net.pyT")
            if model_iter == -1
            else os.path.join(folder, f"training_iters/net_{model_iter}.pyT")
        )
        r["net"] = net = torch.load(net_path, map_location=map_device, weights_only=False)
        net.eval()

    if get_data_loaders:
        r["train_loader"], r["eval_loaders"], r["num_classes"], r["input_dim"] = get_data(
            args, return_biased=bias, abtrain=False
        )

    if get_samples:
        data_loader = r["eval_loaders"][split]
        idx, images, labels, *spurs = get_samples_from_loader(
            data_loader, num_samples, seed=seed, device=device, bias=bias, get_idx=True
        )
        spurs = spurs[0] if bias else None
        r["idx"], r["images"], r["labels"], r["spurs"] = idx, images, labels, spurs

    return r


# def get_model_details(folder, get_model=True, model_iter=-1, bias=True, get_hist=True, get_data_loaders=True, get_samples=True, split="test", num_samples=1000, layer_idx=-1, seed=None, device=None, repr_eval_mode=False):
#     r = {}
#     folder = folder + "/" if folder[-1] != "/" else folder
#     r["args"] = args = torch.load(folder + "args.info")
#     seed = args.seed if not seed else seed
#     device = args.device if not device else device
#     if get_hist:
#         r["eval_hist"] = torch.load(folder + "evaluation_history.hist")
#         r["train_hist"] = torch.load(folder + "training_history.hist")
#     if get_model:
#         r["net"] = net = torch.load(folder + "net.pyT") if model_iter == -1 else torch.load(folder + f"training_iters/net_{model_iter}.pyT")
#         net.eval()
#     if get_data_loaders:
#         r["train_loader"], r["eval_loaders"], r["num_classes"], r["input_dim"] = get_data(args, return_biased=bias, abtrain=False)
#     if get_samples:
#         data_loader = r["eval_loaders"][split]
#         idx, images, labels, *spurs = get_samples_from_loader(data_loader, num_samples, seed=seed, device=device, bias=bias, get_idx=True)
#         spurs = spurs[0] if bias else None
#         r["idx"], r["images"], r["labels"], r["spurs"] = idx, images, labels, spurs
#     return r


# * ATTACK UTILS (LOW LEVEL)
def generate_x_adv(
    net,
    x_orig,
    p,
    input_shape,
    num_classes,
    stats,
    crit,
    device,
    attack_eps=0.0314,
    attack_lr=0.05882,
    attacks_max_iter=10,
    seed=0,
    attacker="pgd",
    evaluation=False,
):
    """
    Generate adversarial examples using PGD attack.

    IMPORTANT: Output order is preserved - x_adv[i] is the adversarial version of x_orig[i].
    The ART library processes each image independently and maintains ordering.

    Returns:
        x_adv: adversarial examples (same order as x_orig)
        x_orig: original clean examples (returned for convenience)
    """
    # Note: Check out the pre- and post-attack transformations
    x_mean, x_std = (
        np.array(stats["mean"])[np.newaxis, :, np.newaxis, np.newaxis],
        np.array(stats["std"])[np.newaxis, :, np.newaxis, np.newaxis],
    )
    x = (x_orig * x_std) + x_mean
    with torch.random.fork_rng():
        torch.manual_seed(seed)
        net.eval()

        net_cls = PyTorchClassifier(
            net,
            loss=crit,
            input_shape=input_shape,
            nb_classes=num_classes,
            optimizer=None,
            preprocessing=(stats["mean"], stats["std"]),  # type: ignore
        )
        if attacker == "pgd":
            attacks = ProjectedGradientDescent(
                net_cls,
                norm={"2": 2, "inf": np.inf}[p],
                eps=attack_eps,
                eps_step=attack_lr,
                max_iter=30 if evaluation else attacks_max_iter,
                num_random_init=10 if evaluation else 1,
                verbose=False,
            )
        else:
            raise ValueError
        x_adv = attacks.generate(x=x.astype(np.float32))  # Pass NCHW
        x_adv = (x_adv - x_mean) / x_std
    return torch.tensor(x_adv).float().to(device), torch.tensor(x_orig).float().to(device)


def filter_successful_attacks(model, x_clean_batch, y_clean_batch, x_adv_batch, device, target_class=-1, verbose=False):
    """
    Filter adversarial examples to keep only successful attacks.

    IMPORTANT: x_clean_batch[i], y_clean_batch[i], and x_adv_batch[i] must correspond!
    - x_clean_batch[i]: original clean image
    - y_clean_batch[i]: true label for that image
    - x_adv_batch[i]: adversarial perturbation of x_clean_batch[i]

    Args:
        target_class: If != -1, only keep attacks that are predicted as this class
        verbose: If True, return detailed statistics

    Returns:
        successful_attack_idx: Boolean mask where True indicates successful attack
        stats: Dictionary with statistics (only if verbose=True, otherwise None)
    """
    # Defensive check: ensure batch sizes match
    assert len(x_clean_batch) == len(y_clean_batch) == len(x_adv_batch), (
        f"Batch size mismatch! clean: {len(x_clean_batch)}, labels: {len(y_clean_batch)}, adv: {len(x_adv_batch)}"
    )

    model.eval()
    with torch.no_grad():
        x_clean_batch, y_clean_batch, x_adv_batch = (
            x_clean_batch.to(device),
            y_clean_batch.to(device),
            x_adv_batch.to(device),
        )
        pred_clean = model(x_clean_batch).argmax(dim=1)
        pred_adv = model(x_adv_batch).argmax(dim=1)

        # Basic successful attack filter
        successful_attack_idx = (pred_clean == y_clean_batch) & (
            pred_adv != y_clean_batch
        )  # items that were correctly classified and are now misclassified

        # Additional target class filtering if specified
        if target_class != -1:
            target_class_idx = pred_adv == target_class
            successful_attack_idx = successful_attack_idx & target_class_idx

        # Only compute detailed stats if verbose
        stats = None
        if verbose:
            total = len(x_clean_batch)
            successful_attacks_basic = ((pred_clean == y_clean_batch) & (pred_adv != y_clean_batch)).sum().item()

            # Get successful attack mask before target class filtering
            successful_mask = (pred_clean == y_clean_batch) & (pred_adv != y_clean_batch)

            stats = {
                "total": total,
                "successful_attacks": successful_attacks_basic,
                "pred_adv_distribution(successful_only)": pred_adv[successful_mask].cpu(),
                "y_clean_distribution(successful_only)": y_clean_batch[successful_mask].cpu(),  # Add clean classes
            }

            if target_class != -1:
                successful_with_target = successful_attack_idx.sum().item()
                stats["target_class"] = target_class
                stats["successful_with_target"] = successful_with_target

    return successful_attack_idx.cpu(), stats


# * GENERAL UTILS
def npize(tensor):
    return tensor.cpu().detach().numpy()


def get_dual_norm(p):
    if p == 1:
        return np.inf
    return 1 / (1 - (1 / p))


class LRConv2d(nn.Module):
    """
    A Minimum Viable Product for a Low-Rank Convolutional Layer used for parameter reduction.

    This layer replaces a standard Conv2d with a sequence of two Conv2d layers:
    1. A spatial convolution that maps `in_channels` to an intermediate `rank` (r)
       using the original kernel size, stride, padding, dilation, and groups.
    2. A pointwise (1x1) convolution that maps from `rank` (r) to `out_channels`.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size,
        rank: int,
        stride=1,
        padding=0,
        dilation=1,
        groups: int = 1,
        bias: bool = True,
        conv1_bias: bool = False,
        padding_mode="zeros",
    ):  # Ensure padding_mode is an argument
        super().__init__()

        if not isinstance(rank, int) or rank <= 0:
            raise ValueError("Rank 'rank' must be a positive integer.")

        if groups > 1:
            if in_channels % groups != 0:
                raise ValueError("in_channels must be divisible by groups")
            if rank % groups != 0:
                raise ValueError("Rank 'rank' must be divisible by 'groups' if groups > 1.")

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.rank = rank
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.bias = bias
        self.conv1_bias = conv1_bias
        self.padding_mode_param = padding_mode  # Crucial: Store padding_mode

        # Convolution 1: Spatial convolution
        # Note: The internal nn.Conv2d will use its own padding_mode argument.
        # If you want LRConv2d's padding_mode to affect conv1, pass self.padding_mode_param here.
        # For now, it uses PyTorch's default 'zeros' unless padding_mode is explicitly passed to conv1.
        # The original nn.Conv2d's padding_mode is available in self.padding_mode_param.
        self.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=rank,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,  # This padding value is used
            dilation=dilation,
            groups=groups,
            bias=conv1_bias,
            padding_mode=self.padding_mode_param,  # Pass the stored padding_mode to the first conv layer
        )

        # Convolution 2: Pointwise (1x1) convolution
        self.conv2 = nn.Conv2d(
            in_channels=rank,
            out_channels=out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            groups=1,  # Typically 1 for channel mixing, could be self.groups if rank and out_channels are divisible
            bias=bias,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if hasattr(self, "conv1"):
            x = self.conv1(x)
            x = self.conv2(x)
        else:
            # HACK:
            x = self.lowrank_conv1(x)
            x = self.lowrank_conv2(x)
        return x

    def __repr__(self):
        s = (
            f"{self.__class__.__name__}({self.in_channels}, {self.out_channels}, "
            f"kernel_size={self.kernel_size}, rank={self.rank}, stride={self.stride}, "
            f"padding={self.padding}, dilation={self.dilation}, groups={self.groups}"
        )
        if not self.bias:
            s += f", bias={self.bias}"
        if self.conv1_bias:
            s += f", conv1_bias={self.conv1_bias}"

        # Robustly check for padding_mode_param and if it's not the default
        if hasattr(self, "padding_mode_param") and self.padding_mode_param != "zeros":
            s += f", padding_mode='{self.padding_mode_param}'"
        # If padding_mode_param is not set or is "zeros", it won't be added to the repr,
        # which is standard behavior for default parameters.

        s += ")"
        return s

from functools import partial

import torch
import torch.nn as nn

from nadf.target_models.resnet18 import BasicBlock, ModifiedResNet, ResNet18
from nadf.utils import get_activation


class MultiLayerNN(nn.Module):
    def __init__(
        self,
        input_dim=28 * 28,
        width=50,
        depth=2,
        num_classes=10,
        activation="relu",
        bias=False,
        get_learned_repr=False,
        top_k_activations=0,
        top_k_neurons=0,
        ablate_top_k_neurons=0,
        activation_exponent=0.0,
        dropout=0.0,
        **kwargs,
    ):
        assert depth >= 1
        super(MultiLayerNN, self).__init__()
        self.input_dim = input_dim
        self.width = width
        self.depth = depth
        self.num_classes = num_classes
        self.get_learned_repr = get_learned_repr
        self.top_k_activations = top_k_activations
        self.top_k_neurons = top_k_neurons
        self.ablate_top_k_neurons = ablate_top_k_neurons
        self.activation_exponent = activation_exponent
        self.dropout = dropout

        Linear = partial(nn.Linear, bias=bias)

        num_output_dims = 1 if kwargs.get("single_logit", False) else self.num_classes
        layers = []
        for i in range(depth - 1):
            layers.append(Linear(self.width if i > 0 else self.input_dim, self.width))
            layers.append(get_activation(activation))
            if self.dropout > 0:
                layers.append(nn.Dropout(p=self.dropout))

        if getattr(self, "depth", 2) > 1:
            # layers.append(nn.Linear(self.width, 64, bias=bias))
            # layers.append(get_activation(activation))
            # layers.append(nn.Linear(64, self.num_classes, bias=bias))
            layers.append(nn.Linear(self.width, num_output_dims, bias=bias))
        elif getattr(self, "depth", 2) == 1:
            layers.append(nn.Linear(self.input_dim, num_output_dims, bias=bias))
        else:
            raise ValueError
        self.fc = nn.Sequential(*layers)

    def forward(self, x):
        # Changed since x.view requires contigious input.
        # x = x.view(x.size(0), self.input_dim)
        x = x.reshape(x.size(0), self.input_dim)
        if getattr(self, "depth", 2) > 1:
            x = self.fc[:-1](x)
        if hasattr(self, "get_learned_repr") and self.get_learned_repr:
            self.learned_repr = x
        if hasattr(self, "top_k_activations") and self.top_k_activations > 0:
            _, indices = torch.topk(x, x.shape[1] - self.top_k_activations, dim=1, largest=False, sorted=False)
            x = x.scatter(1, indices, 0)
        if hasattr(self, "top_k_neurons") and self.top_k_neurons > 0:
            mask = torch.ones_like(x)
            mask[:, torch.argsort((x > 0).sum(0))[: x.shape[1] - self.top_k_neurons]] = 0
            x = x * mask
        if hasattr(self, "ablate_top_k_neurons") and self.ablate_top_k_neurons > 0:
            mask = torch.ones_like(x)
            mask[:, torch.argsort((x > 0).sum(0))[-self.ablate_top_k_neurons :]] = 0
            x = x * mask
        if hasattr(self, "activation_exponent") and self.activation_exponent > 0.0:
            x = x**self.activation_exponent

        x = self.fc[-1](x)
        return x


def get_model(
    model,
    device,
    input_dim,
    width,
    depth,
    num_classes,
    activation,
    bias,
    llnobias=False,
    img_channels=3,
    dropout=0.0,
    get_all_repr_norms=0,
    **kwargs,
):
    if kwargs.get("normalize_logits", False) and model != "resnet18":
        raise NotImplementedError
    if get_all_repr_norms:
        if ("resnet18" not in model) or ("pretrained" in model):
            raise NotImplementedError
    if kwargs["single_logit"] and model != "fcn":
        raise NotImplementedError
    if ("resnet18" in model) and ("pretrained" not in model):
        net = ResNet18(
            img_channels=img_channels,
            num_classes=num_classes,
            num_layers=18,
            block=BasicBlock,
            get_all_repr_norms=get_all_repr_norms,
            padding_mode=model.split("-")[1] if "-" in model and "repr" not in model else "zeros",
            width=width if "resnet18w" in model else 64,
            **kwargs,
        ).to(device)
        if "repr" in model:  # resnet18repr-12
            repr_dim = int(model.split("-")[1])
            net = ResNet18_Repr(
                net,
                num_classes,
                repr_dim,
                img_channels=img_channels,
                num_layers=18,
                block=BasicBlock,
                get_all_repr_norms=get_all_repr_norms,
                padding_mode="zeros",
                width=width if "resnet18w" in model else 64,
                **kwargs,
            ).to(device)
        if dropout > 1e-6:
            net = ModifiedResNet(net, -1, dropout=dropout)
        if llnobias:
            net.fc.bias = None
    elif "fcn" in model:
        net = MultiLayerNN(
            input_dim=input_dim,
            width=width,
            depth=depth,
            num_classes=num_classes,
            activation=activation,
            bias=bias,
            dropout=dropout,
            **kwargs,
        ).to(device)

        if kwargs.get("custom_init", False):
            # net.apply(init_weights)
            with torch.random.fork_rng():
                # Set a local seed
                torch.manual_seed(0)
                p_init = list(torch.load(kwargs["custom_init"]).parameters())[-1]
                p_shape, p_numel = p_init.shape, p_init.numel()
                p = list(net.parameters())[-1]
                p.data = torch.reshape(p_init.flatten()[torch.randperm(p_numel)], p_shape)
        if kwargs.get("custom_init", False):
            for i, param in enumerate(net.parameters()):
                # HACK
                if num_classes in param.shape:
                    param.requires_grad = False
                    param.data.fill_(1 / width)
        net = net.to(device)
    return net


class ResNet18_Repr(nn.Module):
    # from ChatGPT
    def __init__(self, base, num_classes, repr_dim, **kwargs):
        super().__init__()
        # base = ResNet18(**kwargs)
        # base = ResNet18(num_classes, repr_dim, img_channels=img_channels, num_layers=18, block=BasicBlock, get_all_repr_norms=get_all_repr_norms, padding_mode="zeros", width=width if "resnet18w" in model else 64, **kwargs)
        self.backbone = nn.Sequential(*(list(base.children())[:-2]))  # up to last conv
        self.avgpool = base.avgpool
        self.embed = nn.Linear(base.fc.in_features, repr_dim, bias=False)
        self.classifier = nn.Linear(repr_dim, num_classes, bias=True)

    def forward(self, x, repr=False):
        x = self.backbone(x)
        x = self.avgpool(x).flatten(1)  # 512-d
        z = self.embed(x)  # 128-d
        if repr:
            return z  # useful for contrastive, retrieval, etc.
        return self.classifier(z)

"""
MLP models for triplet learning, regression, and binary classification.
"""

import torch
import torch.nn as nn

from nadf.utils import get_activation


class TripletMLP(nn.Module):
    """
    A simple MLP to be trained with triplet loss. It learns a new embedding.
    """

    def __init__(self, input_dim, width=64, depth=3, output_dim=32, activation="relu", bias=True):
        super().__init__()
        self.input_dim = input_dim

        layers = []
        current_dim = input_dim
        for _i in range(depth - 1):
            layers.append(nn.Linear(current_dim, width, bias=bias))
            layers.append(get_activation(activation))
            current_dim = width

        layers.append(nn.Linear(width, output_dim, bias=bias))
        self.fc = nn.Sequential(*layers)

    def forward(self, x):
        return self.fc(x)


class RegressionMLP(TripletMLP):
    """
    A simple MLP to be trained with regression loss.
    """

    def __init__(self, input_dim, width=64, depth=3, activation="relu", bias=True):
        super().__init__(output_dim=1, input_dim=input_dim, width=width, depth=depth, activation=activation, bias=bias)

    #! Fix: Flatten the output
    def forward(self, x):
        output = super().forward(x)
        return output.flatten()


class BinaryClassificationMLP(RegressionMLP):
    """
    MLP for binary classification with BCE loss.
    """

    def forward(self, x):
        output = super().forward(x)
        return torch.sigmoid(output).flatten()  # Probabilities: 0 to 1

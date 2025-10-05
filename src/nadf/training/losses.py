"""
Custom loss functions for weighted training.
"""

import torch
import torch.nn as nn


class WeightedMSELoss(nn.Module):
    """
    Weighted Mean Squared Error Loss that can handle sample-wise weights.
    """

    def __init__(self, reduction="mean"):
        super().__init__()
        self.reduction = reduction

    def forward(self, predictions, targets, weights=None):
        """
        Args:
            predictions: model predictions
            targets: ground truth targets
            weights: sample weights (optional). If None, uses standard MSE.
        """
        mse = (predictions - targets) ** 2

        if weights is not None:
            # Apply weights to the squared errors
            weighted_mse = mse * weights
            if self.reduction == "mean":
                # Weighted average: sum(w_i * mse_i) / sum(w_i)
                return weighted_mse.sum() / weights.sum()
            elif self.reduction == "sum":
                return weighted_mse.sum()
            else:
                return weighted_mse
        else:
            # Standard MSE
            if self.reduction == "mean":
                return mse.mean()
            elif self.reduction == "sum":
                return mse.sum()
            else:
                return mse


class WeightedHuberLoss(nn.Module):
    """
    Weighted Huber Loss that can handle sample-wise weights.
    Huber loss is less sensitive to outliers than squared loss.
    """

    def __init__(self, delta=0.05):
        super().__init__()
        self.delta = delta

    def forward(self, pred, target, w):
        """
        Args:
            pred: model predictions
            target: ground truth targets
            w: sample weights
        """
        d = (pred - target).abs()
        quad = torch.clamp(d, max=self.delta)
        lin = d - quad
        loss = 0.5 * quad**2 + self.delta * lin
        return (w * loss).sum() / (w.sum() + 1e-12)

"""
Evaluation functions for different model types.
"""

import torch


def evaluate_regression_model(model, data_loader, loss_fn, device):
    """Evaluate regression model on data loader."""
    total_loss = 0
    with torch.no_grad():
        for anchors, distances, labels in data_loader:
            anchors, distances, labels = anchors.to(device), distances.to(device), labels.to(device)
            est_distances = model(anchors)
            loss = loss_fn(est_distances, distances)
            total_loss += loss.item()
        avg_loss = total_loss / len(data_loader)
    return avg_loss


def evaluate_regression_model_weighted(model, data_loader, loss_fn, device):
    """
    Evaluate regression model with weighted data loader but compute unweighted MSE for fair comparison.
    """
    total_loss = 0
    with torch.no_grad():
        for batch in data_loader:
            if len(batch) == 4:  # Weighted data: anchors, distances, labels, weights
                anchors, distances, labels, sample_weights = batch
                anchors, distances, labels = anchors.to(device), distances.to(device), labels.to(device)
                est_distances = model(anchors)
                # Use unweighted MSE for evaluation
                loss = torch.nn.functional.mse_loss(est_distances, distances)
            else:  # Standard data: anchors, distances, labels
                anchors, distances, labels = batch
                anchors, distances, labels = anchors.to(device), distances.to(device), labels.to(device)
                est_distances = model(anchors)
                loss = torch.nn.functional.mse_loss(est_distances, distances)
            total_loss += loss.item()
        avg_loss = total_loss / len(data_loader)
    return avg_loss


def evaluate_bce_model(model, data_loader, loss_fn, device):
    """Evaluate binary classification model."""
    total_loss = 0
    with torch.no_grad():
        for representations, labels in data_loader:  # Changed from anchors, distances, labels
            representations, labels = representations.to(device), labels.to(device)
            predictions = model(representations)  # Changed from est_distances
            loss = loss_fn(predictions, labels)  # Predictions vs binary labels
            total_loss += loss.item()

    avg_loss = total_loss / len(data_loader)
    return avg_loss

"""
Training loops and utilities for adversarial robustness models.

This module contains the training logic for regression and binary classification
models used in clean example upweighting experiments.
"""

import os
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import wandb
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, TensorDataset

from nadf.evaluation.metrics import evaluate_regression_model_weighted


def train_regression_model(
    model: nn.Module,
    data_loaders: Dict[str, DataLoader],
    loss_fn: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    num_epochs: int = 100,
    scheduler: Optional[ReduceLROnPlateau] = None,
    verbose: bool = True,
    log_every: int = 20,
    use_wandb: bool = False,
    early_stopping_patience: Optional[int] = None,  # Add this parameter
    early_stopping_min_delta: float = 0.0,  # Minimum change to qualify as improvement
) -> Dict[str, float]:
    """
    Train a regression model for distance prediction.

    Args:
        model: The regression model to train
        data_loaders: Dict with 'train', 'val', 'test' DataLoaders
        loss_fn: Loss function (e.g., WeightedMSELoss or WeightedHuberLoss)
        optimizer: Optimizer for training
        device: Device to train on
        num_epochs: Number of training epochs
        scheduler: Optional learning rate scheduler
        verbose: Whether to print training progress
        log_every: Print progress every N epochs
        use_wandb: Whether to log metrics to wandb
        early_stopping_patience: Number of epochs to wait before stopping if no improvement.
                                If None, early stopping is disabled.
        early_stopping_min_delta: Minimum change in validation loss to qualify as improvement

    Returns:
        Dict with final training metrics
    """
    best_val_loss = float("inf")
    final_metrics = {}
    epochs_without_improvement = 0
    best_model_state = None  # Store best model state

    for epoch in range(num_epochs):
        # Training phase
        model.train()
        total_train_loss = 0
        total_train_correct = 0
        total_train_samples = 0

        for anchors, distances, labels, sample_weights in data_loaders["train"]:
            anchors = anchors.to(device)
            distances = distances.to(device)
            labels = labels.to(device)
            sample_weights = sample_weights.to(device)

            # Forward pass
            est_distances = model(anchors)
            loss = loss_fn(est_distances, distances, sample_weights)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Track metrics
            total_train_loss += loss.item()

            # Calculate binary accuracy at threshold 0.1
            threshold = 0.1
            predicted_labels = (est_distances.flatten() < threshold).long()
            true_labels = (distances < 1e-6).long()
            correct = (predicted_labels == true_labels).sum().item()
            total_train_correct += correct
            total_train_samples += len(true_labels)

        # Calculate epoch metrics
        if epoch % log_every == 0:
            avg_train_loss = total_train_loss / len(data_loaders["train"])

        train_accuracy = total_train_correct / total_train_samples if total_train_samples > 0 else 0

        # Validation phase
        model.eval()
        avg_val_loss = evaluate_regression_model_weighted(model, data_loaders["val"], loss_fn, device)

        # Update learning rate
        if scheduler is not None:
            scheduler.step(avg_val_loss)

        # Test phase
        avg_test_loss = evaluate_regression_model_weighted(model, data_loaders["test"], loss_fn, device)

        # Track best model and early stopping
        improved = False
        if avg_val_loss < best_val_loss - early_stopping_min_delta:
            best_val_loss = avg_val_loss
            epochs_without_improvement = 0
            improved = True
            # Save best model state
            best_model_state = model.state_dict().copy()
        else:
            epochs_without_improvement += 1

        # Log to wandb
        if use_wandb:
            wandb.log(
                {
                    "epoch": epoch + 1,
                    "train_loss": avg_train_loss,
                    "train_accuracy": train_accuracy,
                    "val_loss": avg_val_loss,
                    "test_loss": avg_test_loss,
                    "learning_rate": optimizer.param_groups[0]["lr"],
                    "best_val_loss": best_val_loss,
                    "epochs_without_improvement": epochs_without_improvement,
                }
            )

        # Console logging
        if verbose and (epoch % log_every == 0 or epoch == num_epochs - 1):
            improvement_str = "✓" if improved else f"({epochs_without_improvement} no improvement)"
            print(
                f"Epoch {epoch + 1}/{num_epochs}, "
                f"Train Loss: {avg_train_loss:.4f}, "
                f"Train Acc: {train_accuracy:.4f}, "
                f"Val Loss: {avg_val_loss:.4f}, "
                f"Test Loss: {avg_test_loss:.4f} "
                f"{improvement_str}"
            )

        # Early stopping check
        if early_stopping_patience is not None and epochs_without_improvement >= early_stopping_patience:
            if verbose:
                print(f"\nEarly stopping triggered after {epoch + 1} epochs!")
                print(f"Best validation loss: {best_val_loss:.4f} at epoch {epoch + 1 - epochs_without_improvement}")
            break

    # Restore best model state
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        if verbose:
            print("Restored best model state based on validation loss.")

    final_metrics = {
        "final_train_loss": avg_train_loss,
        "final_train_accuracy": train_accuracy,
        "final_val_loss": avg_val_loss,
        "final_test_loss": avg_test_loss,
        "best_val_loss": best_val_loss,
        "epochs_trained": epoch + 1,
        "early_stopped": early_stopping_patience is not None and epochs_without_improvement >= early_stopping_patience,
    }

    if verbose:
        print("Training finished.")

    return final_metrics


def train_regression_model_without_early_stopping(
    model: nn.Module,
    data_loaders: Dict[str, DataLoader],
    loss_fn: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    num_epochs: int = 100,
    scheduler: Optional[ReduceLROnPlateau] = None,
    verbose: bool = True,
    log_every: int = 20,
    use_wandb: bool = False,
) -> Dict[str, float]:
    """
    Train a regression model for distance prediction.

    Args:
        model: The regression model to train
        data_loaders: Dict with 'train', 'val', 'test' DataLoaders
        loss_fn: Loss function (e.g., WeightedMSELoss or WeightedHuberLoss)
        optimizer: Optimizer for training
        device: Device to train on
        num_epochs: Number of training epochs
        scheduler: Optional learning rate scheduler
        verbose: Whether to print training progress
        log_every: Print progress every N epochs
        use_wandb: Whether to log metrics to wandb

    Returns:
        Dict with final training metrics
    """
    best_val_loss = float("inf")
    final_metrics = {}

    for epoch in range(num_epochs):
        # Training phase
        model.train()
        total_train_loss = 0
        total_train_correct = 0
        total_train_samples = 0

        for anchors, distances, labels, sample_weights in data_loaders["train"]:
            anchors = anchors.to(device)
            distances = distances.to(device)
            labels = labels.to(device)
            sample_weights = sample_weights.to(device)

            # Forward pass
            est_distances = model(anchors)
            loss = loss_fn(est_distances, distances, sample_weights)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Track metrics
            total_train_loss += loss.item()

            # Calculate binary accuracy at threshold 0.1
            threshold = 0.1
            predicted_labels = (est_distances.flatten() < threshold).long()
            true_labels = (distances < 1e-6).long()
            correct = (predicted_labels == true_labels).sum().item()
            total_train_correct += correct
            total_train_samples += len(true_labels)

        # Calculate epoch metrics
        if epoch % log_every == 0:
            avg_train_loss = total_train_loss / len(data_loaders["train"])

        train_accuracy = total_train_correct / total_train_samples if total_train_samples > 0 else 0

        # Validation phase
        model.eval()
        avg_val_loss = evaluate_regression_model_weighted(model, data_loaders["val"], loss_fn, device)

        # Update learning rate
        if scheduler is not None:
            scheduler.step(avg_val_loss)

        # Test phase
        avg_test_loss = evaluate_regression_model_weighted(model, data_loaders["test"], loss_fn, device)

        # Track best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss

        # Log to wandb
        if use_wandb:
            wandb.log(
                {
                    "epoch": epoch + 1,
                    "train_loss": avg_train_loss,
                    "train_accuracy": train_accuracy,
                    "val_loss": avg_val_loss,
                    "test_loss": avg_test_loss,
                    "learning_rate": optimizer.param_groups[0]["lr"],
                    "best_val_loss": best_val_loss,
                }
            )

        # Console logging
        if verbose and (epoch % log_every == 0 or epoch == num_epochs - 1):
            print(
                f"Epoch {epoch + 1}/{num_epochs}, "
                f"Train Loss: {avg_train_loss:.4f}, "
                f"Train Acc: {train_accuracy:.4f}, "
                f"Val Loss: {avg_val_loss:.4f}, "
                f"Test Loss: {avg_test_loss:.4f}"
            )

    final_metrics = {
        "final_train_loss": avg_train_loss,
        "final_train_accuracy": train_accuracy,
        "final_val_loss": avg_val_loss,
        "final_test_loss": avg_test_loss,
        "best_val_loss": best_val_loss,
    }

    if verbose:
        print("Training finished.")

    return final_metrics


def train_regression_model_old(
    model: nn.Module,
    data_loaders: Dict[str, DataLoader],
    loss_fn: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    num_epochs: int = 100,
    scheduler: Optional[ReduceLROnPlateau] = None,
    verbose: bool = True,
    log_every: int = 20,
) -> Dict[str, float]:
    """
    Train a regression model for distance prediction.

    Args:
        model: The regression model to train
        data_loaders: Dict with 'train', 'val', 'test' DataLoaders
        loss_fn: Loss function (e.g., WeightedMSELoss or WeightedHuberLoss)
        optimizer: Optimizer for training
        device: Device to train on
        num_epochs: Number of training epochs
        scheduler: Optional learning rate scheduler
        verbose: Whether to print training progress
        log_every: Print progress every N epochs

    Returns:
        Dict with final training metrics
    """

    best_val_loss = float("inf")
    final_metrics = {}

    for epoch in range(num_epochs):
        # Training phase
        model.train()
        total_train_loss = 0
        total_train_correct = 0
        total_train_samples = 0

        for batch_idx, (anchors, distances, labels, sample_weights) in enumerate(data_loaders["train"]):
            anchors = anchors.to(device)
            distances = distances.to(device)
            labels = labels.to(device)
            sample_weights = sample_weights.to(device)

            # Forward pass
            est_distances = model(anchors)
            loss = loss_fn(est_distances, distances, sample_weights)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Track metrics
            total_train_loss += loss.item()

            # Calculate binary accuracy at threshold 0.1
            if batch_idx % 10 == 0:
                threshold = 0.1
                predicted_labels = (est_distances.flatten() < threshold).long()
                true_labels = (distances < 1e-6).long()
                correct = (predicted_labels == true_labels).sum().item()
                total_train_correct += correct
                total_train_samples += len(true_labels)

        # Calculate epoch metrics
        avg_train_loss = total_train_loss / len(data_loaders["train"])
        train_accuracy = total_train_correct / total_train_samples if total_train_samples > 0 else 0

        # Validation phase
        model.eval()
        avg_val_loss = evaluate_regression_model_weighted(model, data_loaders["val"], loss_fn, device)

        # Update learning rate
        if scheduler is not None:
            scheduler.step(avg_val_loss)

        # Test phase
        avg_test_loss = evaluate_regression_model_weighted(model, data_loaders["test"], loss_fn, device)

        # Track best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss

        # Logging
        if verbose and (epoch % log_every == 0 or epoch == num_epochs - 1):
            print(
                f"Epoch {epoch + 1}/{num_epochs}, "
                f"Train Loss: {avg_train_loss:.4f}, "
                f"Train Acc: {train_accuracy:.4f}, "
                f"Val Loss: {avg_val_loss:.4f}, "
                f"Test Loss: {avg_test_loss:.4f}"
            )

    final_metrics = {
        "final_train_loss": avg_train_loss,
        "final_train_accuracy": train_accuracy,
        "final_val_loss": avg_val_loss,
        "final_test_loss": avg_test_loss,
        "best_val_loss": best_val_loss,
    }

    if verbose:
        print("Training finished.")

    return final_metrics


def train_bce_model(
    model: nn.Module,
    data_loaders: Dict[str, DataLoader],
    loss_fn: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    num_epochs: int = 100,
    scheduler: Optional[ReduceLROnPlateau] = None,
    verbose: bool = True,
    log_every: int = 20,
) -> Dict[str, float]:
    """
    Train a binary classification model.

    Args:
        model: The binary classification model to train
        data_loaders: Dict with 'train', 'val', 'test' DataLoaders
        loss_fn: Loss function (e.g., nn.BCELoss)
        optimizer: Optimizer for training
        device: Device to train on
        num_epochs: Number of training epochs
        scheduler: Optional learning rate scheduler
        verbose: Whether to print training progress
        log_every: Print progress every N epochs

    Returns:
        Dict with final training metrics
    """
    from utils import evaluate_bce_model

    best_val_loss = float("inf")
    final_metrics = {}

    for epoch in range(num_epochs):
        # Training phase
        model.train()
        total_train_loss = 0
        total_train_correct = 0
        total_train_samples = 0

        for representations, labels in data_loaders["train"]:
            representations = representations.to(device)
            labels = labels.to(device)

            # Forward pass
            predictions = model(representations)
            loss = loss_fn(predictions, labels)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Track metrics
            total_train_loss += loss.item()

            # Calculate accuracy
            predicted_classes = (predictions > 0.5).float()
            correct = (predicted_classes == labels).sum().item()
            total_train_correct += correct
            total_train_samples += len(labels)

        # Calculate epoch metrics
        avg_train_loss = total_train_loss / len(data_loaders["train"])
        train_accuracy = total_train_correct / total_train_samples if total_train_samples > 0 else 0

        # Validation phase
        model.eval()
        avg_val_loss = evaluate_bce_model(model, data_loaders["val"], loss_fn, device)

        # Update learning rate
        if scheduler is not None:
            scheduler.step(avg_val_loss)

        # Test phase
        avg_test_loss = evaluate_bce_model(model, data_loaders["test"], loss_fn, device)

        # Track best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss

        # Logging
        if verbose and (epoch % log_every == 0 or epoch == num_epochs - 1):
            print(
                f"Epoch {epoch + 1}/{num_epochs}, "
                f"Train Loss: {avg_train_loss:.4f}, "
                f"Train Acc: {train_accuracy:.4f}, "
                f"Val Loss: {avg_val_loss:.4f}, "
                f"Test Loss: {avg_test_loss:.4f}"
            )

    final_metrics = {
        "final_train_loss": avg_train_loss,
        "final_train_accuracy": train_accuracy,
        "final_val_loss": avg_val_loss,
        "final_test_loss": avg_test_loss,
        "best_val_loss": best_val_loss,
    }

    if verbose:
        print("Training finished.")

    return final_metrics


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    model_config: Dict[str, Any],
    training_config: Dict[str, Any],
    attack_config: Dict[str, Any],
    save_path: str,
) -> None:
    """
    Save a complete training checkpoint.

    Args:
        model: The trained model
        optimizer: The optimizer state
        model_config: Dict with model configuration (input_dim, depth, width, loss_type)
        training_config: Dict with training configuration and final metrics
        attack_config: Dict with attack configuration
        save_path: Path to save the checkpoint
    """
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "model_config": model_config,
        "training_config": training_config,
        "attack_config": attack_config,
    }

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(checkpoint, save_path)
    print(f"Complete checkpoint saved to {save_path}!")


def load_checkpoint(
    checkpoint_path: str,
    model_class: type,
    device: torch.device,
) -> Tuple[nn.Module, Dict[str, Any]]:
    """
    Load a model from a checkpoint.

    Args:
        checkpoint_path: Path to the checkpoint file
        model_class: The model class to instantiate (RegressionMLP or BinaryClassificationMLP)
        device: Device to load the model onto

    Returns:
        Tuple of (loaded model, checkpoint dict)
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(
            f"Checkpoint file not found at {checkpoint_path}. Please train the model first by setting retrain=True."
        )

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model_config = checkpoint["model_config"]

    # Instantiate model
    model = model_class(
        input_dim=model_config["input_dim"], depth=model_config["depth"], width=model_config["width"]
    ).to(device)

    # Load weights
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    print(f"Model loaded from {checkpoint_path}")

    return model, checkpoint


def create_data_loaders(
    datasets: Dict[str, Dict[str, torch.Tensor]],
    batch_size: int,
    loss_type: str,
    splits: list = ["train", "val", "test"],
) -> Dict[str, DataLoader]:
    """
    Create DataLoaders from dataset dictionaries.

    Args:
        datasets: Dict mapping split names to dataset dicts
        batch_size: Batch size for DataLoaders
        loss_type: Type of loss ("regression", "BCE", or "triplet")
        splits: List of split names to create loaders for

    Returns:
        Dict mapping split names to DataLoaders
    """
    data_loaders = {}

    for split in splits:
        data = datasets[split]

        if loss_type == "regression":
            dataset = TensorDataset(data["anchors"], data["distances"], data["labels"], data["sample_weights"])
        elif loss_type == "BCE":
            dataset = TensorDataset(data["representations"], data["labels"])
        elif loss_type == "triplet":
            dataset = TensorDataset(data["anchors"], data["positives"], data["negatives"])
        else:
            raise ValueError(f"Unknown loss_type: {loss_type}")

        # Shuffle only training data
        shuffle = split == "train"
        data_loaders[split] = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

    return data_loaders


def print_dataset_statistics(
    datasets: Dict[str, Dict[str, torch.Tensor]],
    loss_type: str,
) -> None:
    """
    Print detailed statistics about the datasets for debugging.

    Args:
        datasets: Dict mapping split names to dataset dicts
        loss_type: Type of loss ("regression" or "BCE")
    """
    if loss_type == "regression":
        print("=== DEBUGGING TARGET DISTANCES ===")

        for split in ["train", "val", "test"]:
            if split not in datasets:
                continue

            print(f"\n{split.upper()} SET:")
            distances = datasets[split]["distances"].cpu()
            weights = datasets[split]["sample_weights"].cpu()

            # Overall distance statistics
            print(f"  Distance - Min: {distances.min():.4f}, Max: {distances.max():.4f}")
            print(
                f"  Distance - Mean: {distances.mean():.4f}, Median: {distances.median():.4f}, Std: {distances.std():.4f}"
            )
            print(f"  Unique distance values: {torch.unique(distances).shape[0]}")

            # Clean vs adversarial breakdown
            clean_mask = distances == 0
            adv_mask = distances > 0

            num_clean = clean_mask.sum().item()
            num_adv = adv_mask.sum().item()
            total = len(distances)

            print(f"  Clean examples (distance=0): {num_clean}/{total} ({num_clean / total * 100:.1f}%)")
            print(f"  Adversarial examples (distance>0): {num_adv}/{total} ({num_adv / total * 100:.1f}%)")

            # Sample weights
            if num_clean > 0:
                clean_weight = weights[clean_mask][0].item()
                print(f"  Clean example weight: {clean_weight:.1f}")
            if num_adv > 0:
                adv_weight = weights[adv_mask][0].item()
                print(f"  Adversarial example weight: {adv_weight:.1f}")

            # Show first 10 examples
            print(f"  First 10 distances: {distances[:10].tolist()}")

        print("===================================")

    elif loss_type == "BCE":
        print("=== DEBUGGING BINARY LABELS ===")

        for split in ["train", "val", "test"]:
            if split not in datasets:
                continue

            print(f"\n{split.upper()} SET:")
            labels = datasets[split]["labels"].cpu()

            clean_mask = labels == 0
            adv_mask = labels == 1

            num_clean = clean_mask.sum().item()
            num_adv = adv_mask.sum().item()
            total = len(labels)

            print(f"  Clean examples (label=0): {num_clean}/{total} ({num_clean / total * 100:.1f}%)")
            print(f"  Adversarial examples (label=1): {num_adv}/{total} ({num_adv / total * 100:.1f}%)")
            print(f"  First 10 labels: {labels[:10].tolist()}")

        print("===================================")

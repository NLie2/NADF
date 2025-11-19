"""
Training pipeline for NADF probe models.

This module orchestrates the high-level training workflow and delegates
the actual training loops to the trainer module.
"""

import os
from typing import Any, Dict

import torch
import wandb
from torch.optim.lr_scheduler import ReduceLROnPlateau

from nadf.evaluation.metrics import evaluate_regression_model_weighted

# Import your existing utilities
from nadf.models.mlp import RegressionMLP
from nadf.training.losses import WeightedHuberLoss, WeightedMSELoss

# Import the new trainer functions
from nadf.training.trainer import (
    create_data_loaders,
    load_checkpoint,
    print_dataset_statistics,
    save_checkpoint,
    train_regression_model,
    train_regression_model_without_early_stopping,
)


def apply_augmentation(dataset: Dict, args: Any) -> Dict:
    """
    Apply data augmentation to clean examples.

    Args:
        dataset: Dict containing the dataset splits
        args: Command-line arguments with augmentation settings

    Returns:
        Dict with augmentation statistics
    """
    if args.augmentation == "none":
        return {}

    print(f"Applying {args.augmentation} augmentation with {args.num_augmentations}x copies...")

    # TODO: Implement your augmentation logic here
    # This is a placeholder - you'll have your own implementation
    augmentation_stats = {
        "augmentation_type": args.augmentation,
        "num_augmentations": args.num_augmentations,
        "original_size": len(dataset["train"]["z"]),
    }

    return augmentation_stats


def save_augmented_dataset(dataset: Dict, args: Any) -> None:
    """
    Save augmented dataset to disk.

    Args:
        dataset: Dict containing the augmented dataset
        args: Command-line arguments with save settings
    """
    # TODO: Implement your save logic
    print(f"Saving augmented dataset to {args.save_dir}...")


def train_probe_model(datasets: Dict, args: Any, use_wandb: bool = False) -> None:
    """
    Train a probe model on the prepared datasets.

    This is the high-level training function that orchestrates the workflow.
    It now uses the trainer module for the actual training loops.

    Args:
        datasets: Dict with 'train', 'val', 'test' dataset dictionaries
        args: Command-line arguments with training configuration
        use_wandb: Whether to log metrics to wandb
    """
    print("\n[4/4] Training probe model...")

    # Setup device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Get input dimension from data
    input_dim = datasets["train"]["anchors"].shape[1]

    # Initialize model based on type
    if args.model_type == "mlp":
        model = RegressionMLP(input_dim=input_dim, depth=args.depth, width=args.width, activation=args.activation).to(
            device
        )
    elif args.model_type == "transformer":
        # TODO: Add transformer model when implemented
        raise NotImplementedError("Transformer model not yet implemented")
    else:
        raise ValueError(f"Unknown model type: {args.model_type}")

    # Initialize loss function
    if args.loss == "mse":
        loss_fn = WeightedMSELoss()
    elif args.loss == "huber":
        loss_fn = WeightedHuberLoss(delta=args.huber_delta)
    else:
        raise ValueError(f"Unknown loss: {args.loss}")

    # Create data loaders using trainer helper
    data_loaders = create_data_loaders(datasets, batch_size=args.batch_size, loss_type="regression")

    # Print dataset statistics for debugging
    if args.verbose:
        print_dataset_statistics(datasets, loss_type="regression")

    # Setup optimizer and scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    scheduler = ReduceLROnPlateau(optimizer, mode="min", patience=10, factor=0.5)

    # Log model architecture to wandb
    if use_wandb:
        wandb.watch(model, log="all", log_freq=100)

    print("\nStarting training...")
    final_metrics = train_regression_model_without_early_stopping(
        model=model,
        data_loaders=data_loaders,
        loss_fn=loss_fn,
        optimizer=optimizer,
        device=device,
        num_epochs=args.epochs,
        scheduler=scheduler,
        verbose=True,
        log_every=20,
        use_wandb=use_wandb,  # Pass wandb flag to training loop
    )

    # Save checkpoint using trainer helper
    checkpoint_name = args.checkpoint_name or _generate_checkpoint_name(args)
    checkpoint_path = os.path.join(args.save_dir, checkpoint_name)

    save_checkpoint(
        model=model,
        optimizer=optimizer,
        model_config={
            "input_dim": input_dim,
            "depth": args.depth,
            "width": args.width,
            "activation": args.activation,
            "loss_type": args.loss,
            "model_type": args.model_type,
        },
        training_config={
            "learning_rate": args.lr,
            "weight_decay": args.weight_decay,
            "num_epochs": args.epochs,
            "batch_size": args.batch_size,
            "loss_function": args.loss,
            "huber_delta": args.huber_delta if args.loss == "huber" else None,
            "upweight_factor": args.upweight,
            "augmentation": args.augmentation,
            "num_augmentations": args.num_augmentations,
            **final_metrics,  # Include all final training metrics
        },
        attack_config={
            "target_class": args.target_class,
        },
        save_path=checkpoint_path,
    )

    # Log final metrics to wandb
    if use_wandb:
        wandb.log(
            {
                "final_train_loss": final_metrics["final_train_loss"],
                "final_train_accuracy": final_metrics["final_train_accuracy"],
                "final_val_loss": final_metrics["final_val_loss"],
                "final_test_loss": final_metrics["final_test_loss"],
                "best_val_loss": final_metrics["best_val_loss"],
            }
        )

        # Save model artifact
        artifact = wandb.Artifact(
            name=f"model-{wandb.run.id}",
            type="model",
            description=f"NADF probe model: {args.model_type} depth={args.depth} width={args.width}",
        )
        artifact.add_file(checkpoint_path)
        wandb.log_artifact(artifact)

    # Print final summary
    print(f"\n{'=' * 60}")
    print("Training Summary")
    print(f"{'=' * 60}")
    print(f"Final Train Loss: {final_metrics['final_train_loss']:.4f}")
    print(f"Final Train Accuracy: {final_metrics['final_train_accuracy']:.4f}")
    print(f"Final Val Loss: {final_metrics['final_val_loss']:.4f}")
    print(f"Final Test Loss: {final_metrics['final_test_loss']:.4f}")
    print(f"Best Val Loss: {final_metrics['best_val_loss']:.4f}")
    print(f"Model saved to: {checkpoint_path}")
    print(f"{'=' * 60}")

    return checkpoint_path, final_metrics


def train_probe_model_old(datasets: Dict, args: Any) -> None:
    """
    Train a probe model on the prepared datasets.

    This is the high-level training function that orchestrates the workflow.
    It now uses the trainer module for the actual training loops.

    Args:
        datasets: Dict with 'train', 'val', 'test' dataset dictionaries
        args: Command-line arguments with training configuration
    """
    print("\n[4/4] Training probe model...")

    # Setup device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Get input dimension from data
    input_dim = datasets["train"]["anchors"].shape[1]

    # Initialize model based on type
    if args.model_type == "mlp":
        model = RegressionMLP(input_dim=input_dim, depth=args.depth, width=args.width, activation=args.activation).to(
            device
        )
    elif args.model_type == "transformer":
        # TODO: Add transformer model when implemented
        raise NotImplementedError("Transformer model not yet implemented")
    else:
        raise ValueError(f"Unknown model type: {args.model_type}")

    # Initialize loss function
    if args.loss == "mse":
        loss_fn = WeightedMSELoss()
    elif args.loss == "huber":
        loss_fn = WeightedHuberLoss(delta=args.huber_delta)
    else:
        raise ValueError(f"Unknown loss: {args.loss}")

    # Create data loaders using trainer helper
    data_loaders = create_data_loaders(datasets, batch_size=args.batch_size, loss_type="regression")

    # Print dataset statistics for debugging
    if args.verbose:
        print_dataset_statistics(datasets, loss_type="regression")

    # Setup optimizer and scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    scheduler = ReduceLROnPlateau(optimizer, mode="min", patience=10, factor=0.5)

    # Print initial predictions for debugging
    # if args.verbose:
    #     print_initial_predictions(model, data_loaders["train"], "regression", device)

    # ========================================
    # THIS IS WHERE TRAINER.PY COMES IN
    # Instead of 50+ lines of training loops,
    # we call the abstracted training function
    # ========================================

    print("\nStarting training...")
    final_metrics = train_regression_model(
        model=model,
        data_loaders=data_loaders,
        loss_fn=loss_fn,
        optimizer=optimizer,
        device=device,
        num_epochs=args.epochs,
        scheduler=scheduler,
        verbose=True,
        log_every=20,
    )

    # Save checkpoint using trainer helper
    checkpoint_name = args.checkpoint_name or _generate_checkpoint_name(args)
    checkpoint_path = os.path.join(args.save_dir, checkpoint_name)

    save_checkpoint(
        model=model,
        optimizer=optimizer,
        model_config={
            "input_dim": input_dim,
            "depth": args.depth,
            "width": args.width,
            "activation": args.activation,
            "loss_type": "regression",
            "model_type": args.model_type,
        },
        training_config={
            "learning_rate": args.lr,
            "weight_decay": args.weight_decay,
            "num_epochs": args.epochs,
            "batch_size": args.batch_size,
            "loss_function": args.loss,
            "huber_delta": args.huber_delta if args.loss == "huber" else None,
            "upweight_factor": args.upweight,
            "augmentation": args.augmentation,
            "num_augmentations": args.num_augmentations,
            **final_metrics,  # Include all final training metrics
        },
        attack_config={
            "target_class": args.target_class,
            # Add other attack parameters as needed
        },
        save_path=checkpoint_path,
    )

    # Print final summary
    print(f"\n{'=' * 60}")
    print("Training Summary")
    print(f"{'=' * 60}")
    print(f"Final Train Loss: {final_metrics['final_train_loss']:.4f}")
    print(f"Final Train Accuracy: {final_metrics['final_train_accuracy']:.4f}")
    print(f"Final Val Loss: {final_metrics['final_val_loss']:.4f}")
    print(f"Final Test Loss: {final_metrics['final_test_loss']:.4f}")
    print(f"Best Val Loss: {final_metrics['best_val_loss']:.4f}")
    print(f"Model saved to: {checkpoint_path}")
    print(f"{'=' * 60}")


def _generate_checkpoint_name(args: Any) -> str:
    """
    Generate a descriptive checkpoint filename.

    Args:
        args: Command-line arguments

    Returns:
        Checkpoint filename
    """
    parts = [
        args.model_type,
        f"d{args.depth}w{args.width}",
        f"{args.loss}",
        f"up{args.upweight}x",
        f"target_class{args.target_class}",
    ]

    if args.augmentation != "none":
        parts.append(f"aug{args.augmentation}{args.num_augmentations}x")

    filename = "_".join(parts) + ".pth"
    return filename


# Additional helper functions for your pipeline
def evaluate_probe_model(model, datasets, args):
    """
    Evaluate a trained probe model.

    Args:
        model: Trained model
        datasets: Test datasets
        args: Configuration arguments

    Returns:
        Dict with evaluation metrics
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.eval()

    # Create test data loader
    data_loaders = create_data_loaders(datasets, batch_size=args.batch_size, loss_type="regression")

    # Initialize loss function
    if args.loss == "mse":
        loss_fn = WeightedMSELoss()
    elif args.loss == "huber":
        loss_fn = WeightedHuberLoss(delta=args.huber_delta)

    # Evaluate
    with torch.no_grad():
        test_loss = evaluate_regression_model_weighted(model, data_loaders["test"], loss_fn, device)

    print(f"Test Loss: {test_loss:.4f}")

    return {"test_loss": test_loss}


def load_probe_model(checkpoint_path: str, device: torch.device):
    """
    Load a trained probe model from checkpoint.

    Args:
        checkpoint_path: Path to checkpoint file
        device: Device to load model onto

    Returns:
        Loaded model and checkpoint dict
    """
    from nadf.models.mlp import RegressionMLP

    model, checkpoint = load_checkpoint(checkpoint_path, model_class=RegressionMLP, device=device)

    return model, checkpoint

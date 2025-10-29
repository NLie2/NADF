"""
Train NADF probe models.
"""

import argparse
import os

from dotenv import load_dotenv

from nadf.data.adversarial import load_or_create_dataset
from nadf.data.datasets import create_training_datasets
from nadf.training.pipeline import apply_augmentation, save_augmented_dataset, train_probe_model

# Load .env file
load_dotenv()


def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Train NADF probe")

    # Data
    parser.add_argument(
        "--data-folder",
        type=str,
        default=os.getenv("RESNET_MODEL_FOLDER"),
        help="Path to model folder with adversarial cache",
    )
    parser.add_argument(
        "--cache-folder",
        type=str,
        default="/rds/general/user/nk1924/home/nadf/data",
        help="Path to folder for storing adversarial examples cache",
    )
    parser.add_argument("--target-class", type=int, default=-1)
    parser.add_argument("--recreate-data", action="store_true")

    # Training
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--upweight", type=float, default=1.0, help="Clean example upweighting factor")

    # Model architecture
    parser.add_argument("--model-type", type=str, choices=["mlp", "transformer"], default="mlp", help="Model type")
    parser.add_argument("--depth", type=int, default=5, help="Model depth")
    parser.add_argument("--width", type=int, default=256, help="Model width")
    parser.add_argument("--activation", type=str, default="relu", help="Activation function")

    # Loss
    parser.add_argument("--loss", choices=["mse", "huber"], default="mse")
    parser.add_argument("--huber-delta", type=float, default=0.05)

    # Data Augmentation
    parser.add_argument(
        "--augmentation",
        type=str,
        choices=["none", "minimal", "standard", "strong"],
        default="none",
        help="Augmentation type for clean examples",
    )
    parser.add_argument(
        "--num-augmentations",
        type=int,
        default=1,
        help="Number of augmented copies per clean image",
    )

    # Output
    parser.add_argument("--save-dir", type=str, default="trained_models")
    parser.add_argument("--checkpoint-name", type=str, default=None)
    parser.add_argument("--verbose", action="store_true", help="Enable verbose debugging output")

    return parser.parse_args()


def print_header(args):
    """Print training configuration header."""
    print("=" * 60)
    print("NADF Probe Training")
    print("=" * 60)
    print(f"Model: {args.model_type}")
    print(f"Architecture: depth={args.depth}, width={args.width}")
    print(f"Upweighting: {args.upweight}x")
    print(f"Loss: {args.loss}")
    print(
        f"Augmentation: {args.augmentation} ({args.num_augmentations}x)"
        if args.augmentation != "none"
        else "Augmentation: none"
    )
    print("=" * 60)


def main():
    """Main training pipeline."""
    args = parse_arguments()
    print_header(args)

    # Load adversarial dataset
    print("\n[1/4] Loading adversarial dataset...")
    dataset = load_or_create_dataset(
        folder=args.data_folder,
        target_class=args.target_class,
        num_attacks_eps_coef=[(4, 0.25), (2, 0.5), (3, 1), (1, 2)],
        splits=["train", "val", "test"],
        recreate=args.recreate_data,
        data_folder=args.cache_folder,
    )

    # Apply augmentation if enabled
    augmentation_stats = apply_augmentation(dataset, args)

    # Create regression datasets
    datasets = create_training_datasets(dataset, args)

    # Save augmented dataset if applicable
    if args.augmentation != "none" and augmentation_stats:
        save_augmented_dataset(dataset, args)

    # Train model
    train_probe_model(datasets, args)

    print("\n✓ Training complete!")


if __name__ == "__main__":
    main()

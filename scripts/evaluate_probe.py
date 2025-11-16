"""
Training logic for NADF probes.
"""

import os

import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from nadf.evaluation.metrics import evaluate_regression_model_weighted
from nadf.models.mlp import RegressionMLP
from nadf.models.transformer import TransformerAttentionProbe  # When ready
from nadf.training.config import TrainingConfig
from nadf.training.losses import WeightedHuberLoss, WeightedMSELoss


class ProbeTrainer:
    """Trainer for NADF probe models."""

    def __init__(self, config: TrainingConfig):
        self.config = config
        self.device = torch.device(config.device if torch.cuda.is_available() else "cpu")

        # Set seed
        torch.manual_seed(config.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(config.seed)

        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.loss_fn = None

    def build_model(self, input_dim: int):
        """Build the probe model based on config."""
        if self.config.model_type == "mlp":
            self.model = RegressionMLP(
                input_dim=input_dim, width=self.config.width, depth=self.config.depth, activation=self.config.activation
            ).to(self.device)
        elif self.config.model_type == "transformer":
            self.model = TransformerAttentionProbe(
                input_dim=input_dim,
                num_heads=self.config.num_heads,
                hidden_dim=self.config.hidden_dim,
                depth=self.config.depth,
                dropout=self.config.dropout,
            ).to(self.device)
        else:
            raise ValueError(f"Unknown model type: {self.config.model_type}")

        # Build loss function
        if self.config.loss_function == "mse":
            self.loss_fn = WeightedMSELoss()
        elif self.config.loss_function == "huber":
            self.loss_fn = WeightedHuberLoss(delta=self.config.huber_delta)
        else:
            raise ValueError(f"Unknown loss function: {self.config.loss_function}")

        # Build optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=self.config.learning_rate, weight_decay=self.config.weight_decay
        )

        # Build scheduler
        self.scheduler = ReduceLROnPlateau(
            self.optimizer,
            "min",
            patience=self.config.scheduler_patience,
            factor=self.config.scheduler_factor,
            verbose=True,
        )

        num_params = sum(p.numel() for p in self.model.parameters())
        print(f"Built {self.config.model_type} model with {num_params:,} parameters")

    def train_epoch(self, train_loader):
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        total_correct = 0
        total_samples = 0

        for anchors, distances, labels, sample_weights in tqdm(train_loader, desc="Training"):
            anchors = anchors.to(self.device)
            distances = distances.to(self.device)
            sample_weights = sample_weights.to(self.device)

            # Forward pass
            pred_distances = self.model(anchors)
            loss = self.loss_fn(pred_distances, distances, sample_weights)

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # Metrics
            total_loss += loss.item()

            # Threshold accuracy (binary classification at threshold=0.1)
            threshold = 0.1
            pred_labels = (pred_distances < threshold).long()
            true_labels = (distances < 1e-6).long()
            total_correct += (pred_labels == true_labels).sum().item()
            total_samples += len(true_labels)

        avg_loss = total_loss / len(train_loader)
        accuracy = total_correct / total_samples if total_samples > 0 else 0

        return avg_loss, accuracy

    def train(self, datasets: dict):
        """
        Main training loop.

        Args:
            datasets: Dict with keys 'train', 'val', 'test'
                     Each contains {'anchors', 'distances', 'labels', 'sample_weights'}
        """
        # Build model
        input_dim = datasets["train"]["anchors"].shape[1]
        self.build_model(input_dim)

        # Create data loaders
        train_loader = DataLoader(
            TensorDataset(
                datasets["train"]["anchors"],
                datasets["train"]["distances"],
                datasets["train"]["labels"],
                datasets["train"]["sample_weights"],
            ),
            batch_size=self.config.batch_size,
            shuffle=True,
        )

        val_loader = DataLoader(
            TensorDataset(
                datasets["val"]["anchors"],
                datasets["val"]["distances"],
                datasets["val"]["labels"],
                datasets["val"]["sample_weights"],
            ),
            batch_size=self.config.batch_size,
            shuffle=False,
        )

        test_loader = DataLoader(
            TensorDataset(
                datasets["test"]["anchors"],
                datasets["test"]["distances"],
                datasets["test"]["labels"],
                datasets["test"]["sample_weights"],
            ),
            batch_size=self.config.batch_size,
            shuffle=False,
        )

        # Training loop
        best_val_loss = float("inf")

        print(f"\n{'=' * 60}")
        print("Starting Training")
        print(f"{'=' * 60}")

        for epoch in range(self.config.num_epochs):
            # Train
            train_loss, train_acc = self.train_epoch(train_loader)

            # Validate
            self.model.eval()
            val_loss = evaluate_regression_model_weighted(self.model, val_loader, self.loss_fn, self.device)
            test_loss = evaluate_regression_model_weighted(self.model, test_loader, self.loss_fn, self.device)

            # Scheduler step
            self.scheduler.step(val_loss)

            # Logging
            if (epoch + 1) % 20 == 0 or epoch == self.config.num_epochs - 1:
                print(f"\nEpoch {epoch + 1}/{self.config.num_epochs}")
                print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
                print(f"  Val Loss:   {val_loss:.4f}")
                print(f"  Test Loss:  {test_loss:.4f}")

            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self.save_checkpoint(epoch, val_loss, test_loss)

        print(f"\n{'=' * 60}")
        print(f"Training complete! Best val loss: {best_val_loss:.4f}")
        print(f"{'=' * 60}\n")

    def save_checkpoint(self, epoch: int, val_loss: float, test_loss: float):
        """Save model checkpoint."""
        os.makedirs(self.config.save_dir, exist_ok=True)

        if self.config.checkpoint_name:
            checkpoint_name = self.config.checkpoint_name
        else:
            checkpoint_name = (
                f"probe_{self.config.model_type}_{self.config.loss_function}_"
                f"d{self.config.depth}_w{self.config.width}_"
                f"upweight_{self.config.clean_upweight_factor:.1f}x.pth"
            )

        checkpoint_path = os.path.join(self.config.save_dir, checkpoint_name)

        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "val_loss": val_loss,
            "test_loss": test_loss,
            "config": self.config.__dict__,
        }

        torch.save(checkpoint, checkpoint_path)
        print(f"  Saved checkpoint: {checkpoint_path}")

    def load_checkpoint(self, checkpoint_path: str):
        """Load model from checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        # Rebuild model
        # Note: You need to know input_dim, might want to save it in checkpoint
        self.build_model(input_dim=checkpoint["config"]["input_dim"])

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        print(f"Loaded checkpoint from {checkpoint_path}")
        print(f"  Epoch: {checkpoint['epoch']}")
        print(f"  Val Loss: {checkpoint['val_loss']:.4f}")

        return checkpoint

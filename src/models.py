"""
Compatibility module for loading old model checkpoints.

This module provides backward compatibility for models saved with
the old 'models' package structure.
"""

# Import all model classes from their new locations
from nadf.target_models.classifier import ResNet18_Repr
from nadf.target_models.resnet18 import (
    BasicBlock,
    MaskedResNet,
    ModifiedResNet,
    ResNet18,
)

# Make them available at the module level for pickle compatibility
__all__ = [
    "BasicBlock",
    "ResNet18",
    "ResNet18_Repr",
    "MaskedResNet",
    "ModifiedResNet",
]

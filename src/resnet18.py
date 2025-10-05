"""
Compatibility module for loading old model checkpoints.

This module provides backward compatibility for models saved with
the old 'resnet18' module structure.
"""

# Import all classes from their new location
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
    "MaskedResNet",
    "ModifiedResNet",
]

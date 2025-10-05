"""
Compatibility module for loading old model checkpoints.

This module provides backward compatibility for models saved with
the old 'utils' module structure.
"""

# Import from the new location
from nadf.utils import *  # noqa: F401, F403

# Specifically import commonly used items
from nadf.utils import (
    STATS,
    LRConv2d,
    dataset_info,
    filter_successful_attacks,
    generate_x_adv,
    get_activation,
    get_data,
    get_dual_norm,
    get_model_details,
    npize,
)

__all__ = [
    "LRConv2d",
    "get_activation",
    "get_data",
    "get_model_details",
    "generate_x_adv",
    "filter_successful_attacks",
    "npize",
    "get_dual_norm",
    "STATS",
    "dataset_info",
]

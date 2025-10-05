"""
Main experiment functions for upweighting experiments.
"""


def run_upweight_experiments(
    num_attacks_eps_coef=None,
    clean_upweight_factors=None,
    learning_rate=1e-4,
    weight_decay=1e-4,
    depth=5,
    width=256,
    loss_type="regression",
    loss_function="mse",
    retrain=True,
    additional_spec="",
    recreate_dataset=False,
    folder=None,
    ignore_clean_in_eval=False,
    target_class=-1,
):
    """
    Run experiments with different clean upweight factors.

    Main training and evaluation pipeline for the upweighting experiments.
    """
    # TODO: Extract from train_mlp.py
    pass


def run_mlp_size_comparison(
    sizes,
    num_attacks_eps_coef=None,
    clean_upweight_factors=None,
    learning_rate=1e-4,
    weight_decay=1e-4,
    loss_type="regression",
    loss_function="mse",
    retrain=True,
    additional_spec="",
    recreate_dataset=False,
    folder=None,
    ignore_clean_in_eval=False,
    target_class=-1,
):
    """Compare multiple MLP sizes across upweight factors."""
    # TODO: Extract from train_mlp.py
    pass


def plot_from_saved_pickle(results_save_path, loss_type, loss_function, additional_spec, ignore_clean_in_eval=False):
    """Load saved results pickle and generate performance dashboard."""
    # TODO: Extract from train_mlp.py
    pass


def rebuild_size_comparison_from_pickles(sizes, loss_type, loss_function, additional_spec, clean_upweight_factors):
    """Rebuild size comparison plot from existing pickle files."""
    # TODO: Extract from train_mlp.py
    pass

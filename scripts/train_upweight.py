"""
CLI script for running upweighting experiments.
"""

from nadf.experiments.upweight import run_mlp_size_comparison

if __name__ == "__main__":
    # Example configuration
    run_mlp_size_comparison(
        sizes=[(7, 512), (7, 1024)],
        num_attacks_eps_coef=[(4, 0.25), (2, 0.5), (3, 1), (1, 2)],
        clean_upweight_factors=[1.0, 2.0, 3.0],
        learning_rate=1e-4,
        weight_decay=1e-4,
        loss_type="regression",
        loss_function="huber",
        retrain=True,
        additional_spec="test_run",
        recreate_dataset=False,
        ignore_clean_in_eval=False,
    )

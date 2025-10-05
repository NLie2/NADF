# src/nadf/data/adversarial.py
"""
Adversarial example generation and representation extraction.

This module handles the complete pipeline:
1. Load pre-trained model and clean data
2. Generate adversarial examples using PGD attacks
3. Extract representations for both clean and adversarial samples
4. Save/load cached datasets
"""

import os
import pickle

import torch
import torch.nn as nn
from tqdm import tqdm

from nadf.utils import filter_successful_attacks, generate_x_adv, get_model_details, npize


def load_or_create_dataset(
    folder,
    target_class=-1,
    num_attacks_eps_coef=None,
    splits=None,
    recreate=False,
):
    """
    Complete dataset creation pipeline.

    Returns:
        Tuple of dicts containing:
        - x, y, z: Combined clean + adversarial data
        - x_clean, y_clean, z_clean: Clean examples only
        - x_adv, y_adv, z_adv: Adversarial examples only
        - z_clean_reference, y_clean_reference: Full clean pool for nearest neighbor search
        - z_clean_reference_full, y_clean_reference_full: Unfiltered clean pool
    """
    if splits is None:
        splits = ["train", "val", "test"]
    if num_attacks_eps_coef is None:
        num_attacks_eps_coef = [(1, 1)]

    # Check for cached dataset
    cache_path = _get_cache_path(folder, target_class, num_attacks_eps_coef)
    if os.path.exists(cache_path) and not recreate:
        print(f"Loading cached dataset from {cache_path}")
        return _load_cached_dataset(cache_path)

    # Step 1: Load model and clean data
    model, clean_data, device = _load_model_and_data(folder, splits, target_class)

    # Step 2: Generate adversarial examples
    adv_data = _generate_adversarial_examples(model, clean_data, num_attacks_eps_coef, splits, device)

    # Step 3: Extract representations
    representations = _extract_representations(model, clean_data, adv_data, splits, device)

    # Step 4: Save to cache
    _save_to_cache(cache_path, representations)

    return representations


def _load_model_and_data(folder, splits, target_class):
    """Load pre-trained model and extract clean data."""
    from nadf.utils import dataset_info

    # Load model and args
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = torch.load(
        os.path.join(folder, "net.pyT"), map_location=device, weights_only=False
    )  # set weights_olny to flase since this is our own trusted model
    args = torch.load(os.path.join(folder, "args.info"), map_location=device, weights_only=False)
    net.eval()
    net = net.to(device)
    clean_data = {}

    for split in splits:
        print(f"  Loading {split} data...")

        # Get data for this split
        r = get_model_details(
            folder=folder,
            num_samples=dataset_info[args.dataset]["num_samples"][split],
            split=split,
            bias=False,
            get_hist=False,
        )

        # Filter by target class
        idx = (r["labels"] == target_class).cpu() if target_class != -1 else (r["labels"] > -1).cpu()

        # Store filtered clean data
        r_images_cpu = r["images"].cpu()
        r_labels_cpu = r["labels"].cpu()

        x_clean = r_images_cpu[idx].to(device)
        y_clean = r_labels_cpu[idx].to(device)

        # Extract representations for the full clean pool (for reference)
        with torch.no_grad():
            z_clean_pool = net(r["images"].to(device), repr=True).cpu()
            y_clean_pool = r["labels"]

        clean_data[split] = {
            "x_clean": x_clean,
            "y_clean": y_clean,
            "z_clean_pool": z_clean_pool,
            "y_clean_pool": y_clean_pool,
        }

        print(f"    {split}: {len(x_clean)} examples (filtered), {len(z_clean_pool)} in pool")

        # Clear GPU cache after processing each split
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return net, clean_data, device


def _extract_representations(model, clean_data, adv_data, splits, device):
    """Extract feature representations from model for clean and adversarial examples."""

    representations = {
        "x": {},
        "y": {},
        "z": {},
        "x_clean": {},
        "y_clean": {},
        "z_clean": {},
        "x_adv": {},
        "y_adv": {},
        "z_adv": {},
        "z_clean_reference": {},
        "y_clean_reference": {},
        "z_clean_reference_full": {},
        "y_clean_reference_full": {},
    }

    for split in splits:
        print(f"  Extracting representations for {split}...")

        x_clean = clean_data[split]["x_clean"]
        y_clean = clean_data[split]["y_clean"]
        x_combined = adv_data[split]["x_combined"]
        y_combined = adv_data[split]["y_combined"]
        x_adv = adv_data[split]["x_adv"]
        y_adv = adv_data[split]["y_adv"]

        # Extract representations for clean examples
        model.eval()
        with torch.no_grad():
            z_clean = model(x_clean, repr=True).cpu()

        # Extract representations for adversarial examples
        if len(x_adv) > 0:
            with torch.no_grad():
                z_adv = model(x_adv, repr=True).cpu()
        else:
            z_adv = torch.tensor([])

        # Extract representations for combined (clean + adv)
        with torch.no_grad():
            z_combined = model(x_combined, repr=True).cpu()

        # Store everything
        representations["x"][split] = x_combined.cpu()
        representations["y"][split] = y_combined.cpu()
        representations["z"][split] = z_combined

        representations["x_clean"][split] = x_clean.cpu()
        representations["y_clean"][split] = y_clean.cpu()
        representations["z_clean"][split] = z_clean

        representations["x_adv"][split] = x_adv.cpu() if len(x_adv) > 0 else torch.tensor([])
        representations["y_adv"][split] = y_adv.cpu() if len(y_adv) > 0 else torch.tensor([])
        representations["z_adv"][split] = z_adv

        # Reference pools (used for nearest neighbor search later)
        representations["z_clean_reference"][split] = clean_data[split]["z_clean_pool"]
        representations["y_clean_reference"][split] = clean_data[split]["y_clean_pool"]
        representations["z_clean_reference_full"][split] = clean_data[split]["z_clean_pool"]
        representations["y_clean_reference_full"][split] = clean_data[split]["y_clean_pool"]

        print(f"    {split}: z shape = {z_combined.shape}")

        # Clear GPU cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return representations


def _generate_adversarial_examples(model, clean_data, num_attacks_eps_coef, splits, device):
    """Generate PGD adversarial examples for each split."""
    from nadf.utils import STATS

    # Load args to get dataset info
    args = model.args if hasattr(model, "args") else None
    baseline_eps = 0.5
    baseline_attack_lr = 0.033  # You need to verify this value
    attack_p = "2"
    crit = nn.CrossEntropyLoss()

    adv_data = {}

    for split in splits:
        print(f"  Processing {split}...")

        x_clean = clean_data[split]["x_clean"]
        y_clean = clean_data[split]["y_clean"]

        batch_size = 512
        num_batches = (len(x_clean) + batch_size - 1) // batch_size

        x_adv_list = []
        y_adv_list = []

        for num_attacks, eps_coef in num_attacks_eps_coef:
            print(f"    Attacks: num={num_attacks}, eps_coef={eps_coef}")

            for i_attack in range(num_attacks):
                for i_batch in tqdm(range(num_batches), desc=f"      Attack {i_attack + 1}/{num_attacks}"):
                    start_idx = i_batch * batch_size
                    end_idx = min((i_batch + 1) * batch_size, len(x_clean))

                    x_clean_batch = x_clean[start_idx:end_idx]
                    y_clean_batch = y_clean[start_idx:end_idx]

                    if len(x_clean_batch) == 0:
                        continue

                    # Generate adversarial examples
                    new_x_adv_batch, _ = generate_x_adv(
                        model,
                        npize(x_clean_batch),
                        attack_p,
                        (3, 32, 32),  # CIFAR-10 image shape
                        10,  # num_classes for CIFAR-10
                        STATS["cifar10"],
                        crit,
                        device,
                        attack_eps=baseline_eps * eps_coef,
                        attack_lr=baseline_attack_lr * eps_coef,
                        attacks_max_iter=10,
                        seed=i_attack,
                        attacker="pgd",
                        evaluation=False,
                    )

                    # Filter successful attacks
                    idx = filter_successful_attacks(model, x_clean_batch, y_clean_batch, new_x_adv_batch, device)
                    successful_x_adv = new_x_adv_batch[idx]
                    successful_y_adv = y_clean_batch[idx]

                    if len(successful_x_adv) > 0:
                        x_adv_list.append(successful_x_adv)
                        y_adv_list.append(successful_y_adv)

        # Combine all adversarial examples for this split
        if x_adv_list:
            x_adv = torch.vstack(x_adv_list)
            y_adv = torch.cat(y_adv_list)
        else:
            x_adv = torch.tensor([])
            y_adv = torch.tensor([])

        # Combine clean and adversarial
        if len(x_adv) > 0:
            x_combined = torch.vstack([x_clean, x_adv])
            y_combined = torch.cat([y_clean, y_adv])
        else:
            x_combined = x_clean
            y_combined = y_clean

        adv_data[split] = {"x_adv": x_adv, "y_adv": y_adv, "x_combined": x_combined, "y_combined": y_combined}

        print(f"    {split}: {len(x_adv)} successful adversarial examples")

        # Clear GPU cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return adv_data


def _get_cache_path(folder, target_class, num_attacks_eps_coef):
    """
    Generate cache path for adversarial dataset.

    Args:
        folder: Model folder path
        target_class: Target class for attacks (-1 for untargeted)
        num_attacks_eps_coef: List of (num_attacks, eps_coef) tuples
    """
    base = os.path.join(folder, "adversarial_examples")
    # Create a unique identifier for this attack configuration
    attack_configs = "_".join([f"{n}x{ec}" for n, ec in num_attacks_eps_coef])
    attack_name = f"pgd_l2_{attack_configs}"
    return os.path.join(base, "attacks", attack_name, f"target_{target_class}.pkl")


def _load_cached_dataset(cache_path):
    """Load dataset from pickle file."""
    with open(cache_path, "rb") as f:
        return pickle.load(f)


def _save_to_cache(cache_path, data):
    """Save dataset to pickle file."""
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    with open(cache_path, "wb") as f:
        pickle.dump(data, f)
    print(f"Dataset saved to {cache_path}")

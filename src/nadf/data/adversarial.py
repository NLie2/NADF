# src/nadf/data/adversarial.py
"""
Adversarial example generation and representation extraction.

This module handles the complete pipeline:
1. Load pre-trained model and clean data
2. Generate adversarial examples using PGD attacks
3. Extract representations for both clean and adversarial samples
4. Save/load cached datasets
"""

import csv
import os
import pickle
import random

import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
from tqdm import tqdm

from nadf.utils import filter_successful_attacks, generate_x_adv, get_model_details, npize


def load_or_create_dataset(
    folder,
    target_class=-1,
    num_attacks_eps_coef=None,
    splits=None,
    recreate=False,
    verbose=False,
    augment=False,
):
    """
    Complete dataset creation pipeline.

    Args:
        verbose: If True, print detailed statistics about filtering

    Returns:
        Dict of dicts containing (all split by train/val/test):
        - x, y, z: clean + adversarial data with predictions
        - x_clean, y_clean, z_clean, pred_clean: Clean examples only with predictions
        - x_adv, y_adv, z_adv, pred_adv: Adversarial examples only with predictions
        - z_clean_pool, y_clean_pool: Full clean pool for nearest neighbor search

    Note: Prints clean accuracy, attack success rate, and adversarial accuracy for each split.
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
    model, clean_data, device = _load_model_and_data(folder, splits, target_class, augment)
    print("clean_data", clean_data.keys())

    # Step 2: Generate adversarial examples
    adv_data = _generate_adversarial_examples(
        model, clean_data, num_attacks_eps_coef, splits, device, target_class, verbose, "data/results"
    )
    print("adv_data", adv_data.keys(), adv_data["train"].keys())

    # Step 3: Extract representations
    representations = _extract_representations(model, clean_data, adv_data, splits, device, target_class, augment)
    print("representations", representations.keys())

    # Step 4: Save to cache
    _save_to_cache(cache_path, representations)

    return representations


def _generate_clean_augmented_data(x_clean, y_clean):
    """
    Generate multiple augmented versions of each clean image with corresponding labels.

    For each input image, creates:
    - 1 rotated version
    - 1 affine transformed version
    - 1 zoomed (cropped) version
    - 1 clipped (noisy) version

    Args:
        x_clean: Tensor of shape [B, C, H, W] with normalized pixel values
        y_clean: Tensor of shape [B] with labels

    Returns:
        x_augmented: Tensor of shape [B*4, C, H, W] containing all augmented versions
        y_augmented: Tensor of shape [B*4] with labels (each label repeated 4 times)
    """
    print("generate clean augmented data", x_clean.shape, x_clean.dtype, x_clean.min(), x_clean.max())

    augmented_images = []

    for img in x_clean:
        # Augmentation 1: Rotation (10 degrees)
        img_rotated = TF.rotate(img, angle=10)
        augmented_images.append(img_rotated)

        # Augmentation 2: Affine transformation (translation + scale)
        img_affine = TF.affine(img, angle=0, translate=(3, 3), scale=1.05, shear=0)
        augmented_images.append(img_affine)

        # Augmentation 3: Zoom (random crop and resize)
        crop_size = 28  # Crop to 28x28 then resize back to 32x32
        i = random.randint(0, 32 - crop_size)
        j = random.randint(0, 32 - crop_size)
        img_zoomed = TF.crop(img, i, j, crop_size, crop_size)
        img_zoomed = TF.resize(img_zoomed, [32, 32])
        augmented_images.append(img_zoomed)

        # Augmentation 4: Clipped/noisy version
        noise = torch.randn_like(img) * 0.02
        img_noisy = torch.clamp(img + noise, x_clean.min(), x_clean.max())
        augmented_images.append(img_noisy)

    # Stack all augmented images
    x_augmented = torch.stack(augmented_images)

    # Duplicate labels to match (each label repeated 4 times)
    y_augmented = y_clean.repeat_interleave(4)

    print(
        "augmented data", x_augmented.shape, y_augmented.shape, x_augmented.dtype, x_augmented.min(), x_augmented.max()
    )
    # Should print: torch.Size([1992, 3, 32, 32]) torch.Size([1992]) for 498 input images

    return x_augmented, y_augmented


def _load_model_and_data(folder, splits, target_class, augment=False):
    """Load pre-trained model and extract clean data."""
    from nadf.utils import dataset_info

    # Load model and args
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device", device)
    model = torch.load(
        os.path.join(folder, "net.pyT"), map_location=device, weights_only=False
    )  # set weights_olny to flase since this is our own trusted model
    args = torch.load(os.path.join(folder, "args.info"), map_location=device, weights_only=False)
    model.eval()
    model = model.to(device)
    clean_data = {}

    # Batch size for processing representations
    batch_size = 512  # Adjust based on your GPU memory

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
        x_clean_augmented = None
        y_clean_augmented = None
        z_clean_augmented = None
        if augment:
            x_clean_augmented, y_clean_augmented = _generate_clean_augmented_data(x_clean, y_clean)

        # Process z_clean in batches
        z_clean_list = []
        num_batches_clean = (len(x_clean) + batch_size - 1) // batch_size
        with torch.no_grad():
            for i in range(num_batches_clean):
                start_idx = i * batch_size
                end_idx = min((i + 1) * batch_size, len(x_clean))
                z_batch = model(x_clean[start_idx:end_idx], repr=True).cpu()
                z_clean_list.append(z_batch)
                # Clear GPU cache periodically
                if (i + 1) % 10 == 0 and torch.cuda.is_available():
                    torch.cuda.empty_cache()
        z_clean = torch.cat(z_clean_list)

        # Process augmented data separately with its own batching
        if augment:
            z_clean_augmented_list = []
            num_batches_augmented = (len(x_clean_augmented) + batch_size - 1) // batch_size
            with torch.no_grad():
                for i in range(num_batches_augmented):
                    start_idx = i * batch_size
                    end_idx = min((i + 1) * batch_size, len(x_clean_augmented))
                    z_batch_augmented = model(x_clean_augmented[start_idx:end_idx], repr=True).cpu()
                    z_clean_augmented_list.append(z_batch_augmented)
                    # Clear GPU cache periodically
                    if (i + 1) % 10 == 0 and torch.cuda.is_available():
                        torch.cuda.empty_cache()
            z_clean_augmented = torch.cat(z_clean_augmented_list)
        #! debug print("labels", y_clean) # these all show target class

        # Extract representations for the full clean pool (for reference) in batches
        z_clean_pool_list = []
        r_images_device = r["images"].to(device)
        num_batches_pool = (len(r_images_device) + batch_size - 1) // batch_size
        with torch.no_grad():
            for i in range(num_batches_pool):
                start_idx = i * batch_size
                end_idx = min((i + 1) * batch_size, len(r_images_device))
                z_batch = model(r_images_device[start_idx:end_idx], repr=True).cpu()
                z_clean_pool_list.append(z_batch)
                # Clear GPU cache periodically
                if (i + 1) % 10 == 0 and torch.cuda.is_available():
                    torch.cuda.empty_cache()
        z_clean_pool = torch.cat(z_clean_pool_list)

        # Move back to CPU and clean up GPU memory
        del r_images_device
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        y_clean_pool = r["labels"]
        x_clean_pool = r["images"].cpu()

        clean_data[split] = {
            "x_clean": x_clean.cpu(),  # Move to CPU to free GPU memory
            "y_clean": y_clean.cpu(),
            "x_clean_augmented": x_clean_augmented.cpu() if augment else None,
            "y_clean_augmented": y_clean_augmented.cpu() if augment else None,
            "z_clean": z_clean,
            "z_clean_augmented": z_clean_augmented if augment else None,
            "x_clean_pool": x_clean_pool,
            "z_clean_pool": z_clean_pool,
            "y_clean_pool": y_clean_pool,
        }

        print(f"    {split}: {len(x_clean)} examples (filtered), {len(z_clean_pool)} in pool")

        # Clear GPU cache after processing each split
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return model, clean_data, device


def _load_model_and_data_old(folder, splits, target_class):
    """Load pre-trained model and extract clean data."""
    from nadf.utils import dataset_info

    # Load model and args
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device", device)
    model = torch.load(
        os.path.join(folder, "net.pyT"), map_location=device, weights_only=False
    )  # set weights_olny to flase since this is our own trusted model
    args = torch.load(os.path.join(folder, "args.info"), map_location=device, weights_only=False)
    model.eval()
    model = model.to(device)
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
        z_clean = model(x_clean, repr=True).cpu()
        #! debug print("labels", y_clean) # these all show target class

        # Extract representations for the full clean pool (for reference)
        with torch.no_grad():
            z_clean_pool = model(r["images"].to(device), repr=True).cpu()
            y_clean_pool = r["labels"]
            x_clean_pool = r["images"].cpu()

        clean_data[split] = {
            "x_clean": x_clean,
            "y_clean": y_clean,
            "z_clean": z_clean,
            "x_clean_pool": x_clean_pool,
            "z_clean_pool": z_clean_pool,
            "y_clean_pool": y_clean_pool,
        }

        print(f"    {split}: {len(x_clean)} examples (filtered), {len(z_clean_pool)} in pool")

        # Clear GPU cache after processing each split
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return model, clean_data, device


def _extract_representations(model, clean_data, adv_data, splits, device, target_class, augment=False):
    """Extract feature representations and predictions from model for clean and adversarial examples.

    Returns representations dict with predictions (pred_clean, pred_adv) added.
    Also prints clean accuracy, attack success rate, and adversarial accuracy.
    """

    representations = {
        "x": {},
        "y": {},
        "z": {},
        "x_clean": {},
        "y_clean": {},
        "z_clean": {},
        "x_clean_augmented": {},
        "y_clean_augmented": {},
        "z_clean_augmented": {},
        "x_augmented": {},
        "y_augmented": {},
        "z_augmented": {},
        "x_adv": {},
        "y_adv": {},
        "z_adv": {},
        "z_clean_pool": {},
        "y_clean_pool": {},
        "eps_coef_adv": {},
        "eps_coef_combined": {},
        "pred_clean": {},
        "pred_adv": {},
        "pred_combined": {},
        "pred_clean_augmented": {},
        "pred_combined_augmented": {},
    }

    for split in splits:
        print(f"  Extracting representations for {split}...")

        x_clean = clean_data[split]["x_clean"]
        y_clean = clean_data[split]["y_clean"]
        z_clean = clean_data[split]["z_clean"]
        x_combined = adv_data[split]["x_combined"]
        y_combined = adv_data[split]["y_combined"]

        x_clean_augmented = clean_data[split]["x_clean_augmented"]
        y_clean_augmented = clean_data[split]["y_clean_augmented"]
        z_clean_augmented = clean_data[split]["z_clean_augmented"]
        x_combined_augmented = adv_data[split]["x_combined_augmented"]
        y_combined_augmented = adv_data[split]["y_combined_augmented"]

        x_adv = adv_data[split]["x_adv"]
        y_adv = adv_data[split]["y_adv"]

        pred_adv = adv_data[split]["pred_adv"]
        pred_clean = adv_data[split]["pred_clean"]
        pred_combined = adv_data[split]["pred_combined"]

        pred_clean_augmented = adv_data[split]["pred_clean_augmented"]
        pred_combined_augmented = adv_data[split]["pred_combined_augmented"]

        eps_coef_adv = adv_data[split]["eps_coef_adv"]
        eps_coef_combined = adv_data[split]["eps_coef_combined"]

        # Extract representations and predictions for clean examples
        model.eval()

        # Extract representations for adversarial examples
        if len(x_adv) > 0:
            with torch.no_grad():
                # extract representations for adversarial examples
                z_adv = model(x_adv.to(device), repr=True).cpu()
        else:
            z_adv = torch.tensor([])

        # Extract representations for combined (clean + adv)
        with torch.no_grad():
            z_combined = model(x_combined.to(device), repr=True).cpu()
            if augment and len(x_combined_augmented) > 0:
                z_combined_augmented = model(x_combined_augmented.to(device), repr=True).cpu()
            else:
                z_combined_augmented = torch.tensor([])

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Calculate clean accuracy (using pool labels since pred_clean is for the full pool)
        y_clean_pool = clean_data[split]["y_clean_pool"]
        clean_accuracy = (pred_clean == y_clean_pool.cpu()).float().mean().item()
        print(f"    Clean Accuracy: {clean_accuracy * 100:.2f}%")

        if augment and len(pred_clean_augmented) > 0:
            # calculate clean accuracy for augmented data
            clean_accuracy_augmented = (pred_clean_augmented == y_clean_augmented.cpu()).float().mean().item()
            print(
                f"    Clean Accuracy Augmented: {clean_accuracy_augmented * 100:.2f}% ({len(y_clean_augmented)} augmented samples)"
            )

        # Attack success rate and adversarial accuracy are now handled during generation
        if len(x_adv) > 0:
            print(f"    {len(x_adv)} adversarial examples (already filtered during generation)")
        else:
            print("    No adversarial examples generated")

        # Store everything
        representations["x"][split] = x_combined.cpu()
        representations["y"][split] = y_combined.cpu()
        representations["z"][split] = z_combined

        representations["x_clean"][split] = x_clean.cpu()
        representations["y_clean"][split] = y_clean.cpu()
        representations["z_clean"][split] = z_clean
        representations["pred_clean"][split] = pred_clean

        if augment and len(x_combined_augmented) > 0:
            representations["x_augmented"][split] = x_combined_augmented.cpu()
            representations["y_augmented"][split] = y_combined_augmented.cpu()
            representations["z_augmented"][split] = z_combined_augmented

            representations["x_clean_augmented"][split] = x_clean_augmented.cpu()
            representations["y_clean_augmented"][split] = y_clean_augmented.cpu()
            representations["z_clean_augmented"][split] = z_clean_augmented
            representations["pred_clean_augmented"][split] = pred_clean_augmented
            representations["pred_combined_augmented"][split] = pred_combined_augmented
        else:
            # Store empty tensors if augmentation is disabled or failed
            representations["x_augmented"][split] = torch.tensor([])
            representations["y_augmented"][split] = torch.tensor([])
            representations["z_augmented"][split] = torch.tensor([])
            representations["x_clean_augmented"][split] = torch.tensor([])
            representations["y_clean_augmented"][split] = torch.tensor([])
            representations["z_clean_augmented"][split] = torch.tensor([])
            representations["pred_clean_augmented"][split] = torch.tensor([])
            representations["pred_combined_augmented"][split] = torch.tensor([])

        representations["x_adv"][split] = x_adv.cpu() if len(x_adv) > 0 else torch.tensor([])
        representations["y_adv"][split] = y_adv.cpu() if len(y_adv) > 0 else torch.tensor([])
        representations["z_adv"][split] = z_adv
        representations["eps_coef_adv"][split] = eps_coef_adv.cpu() if len(eps_coef_adv) > 0 else torch.tensor([])
        representations["eps_coef_combined"][split] = eps_coef_combined.cpu()

        representations["pred_adv"][split] = pred_adv.cpu() if len(pred_adv) > 0 else torch.tensor([])
        representations["pred_combined"][split] = pred_combined.cpu()

        # Reference pools (used for nearest neighbor search later)
        representations["z_clean_pool"][split] = clean_data[split]["z_clean_pool"]
        representations["y_clean_pool"][split] = clean_data[split]["y_clean_pool"]

        num_adv = len(x_adv) if len(x_adv) > 0 else 0
        print(f"    {split}: z shape = {z_combined.shape}, {len(x_clean)} clean + {num_adv} adv examples")

        # Clear GPU cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return representations


def _generate_adversarial_examples(
    model, clean_data, num_attacks_eps_coef, splits, device, target_class=-1, verbose=False, folder=None, augment=False
):
    """Generate PGD adversarial examples for each split."""
    from nadf.utils import STATS

    # Load args to get dataset info (unused but kept for potential future use)
    # args = model.args if hasattr(model, "args") else None
    baseline_eps = 0.5
    baseline_attack_lr = 0.033  # You need to verify this value
    attack_p = "2"
    crit = nn.CrossEntropyLoss()

    adv_data = {}

    for split in splits:
        print(f"  Processing {split}...")

        x_clean = clean_data[split]["x_clean_pool"]
        y_clean = clean_data[split]["y_clean_pool"]
        # ! the reason we need pool here is because we need adv examples of what the model CLASSIFIES as target class, not the actual target class

        # Get augmented data from filtered subset (not pool)
        x_clean_filtered_augmented = clean_data[split]["x_clean_augmented"]
        y_clean_filtered_augmented = clean_data[split]["y_clean_augmented"]

        # Get predictions for clean examples
        print(f"    Getting predictions for {len(x_clean)} clean examples...")
        pred_clean = get_predictions(model, x_clean, device)

        # Get predictions for augmented filtered data (if it exists)
        if augment and x_clean_filtered_augmented is not None and len(x_clean_filtered_augmented) > 0:
            pred_clean_augmented = get_predictions(model, x_clean_filtered_augmented, device)
        else:
            pred_clean_augmented = torch.tensor([])
            x_clean_filtered_augmented = torch.tensor([])
            y_clean_filtered_augmented = torch.tensor([])

        batch_size = 512
        num_batches = (len(x_clean) + batch_size - 1) // batch_size

        x_adv_list = []
        y_adv_list = []
        eps_coef_list = []
        pred_adv_list = []

        # Track cumulative statistics across ALL attack configurations for this split
        split_cumulative_stats = None
        if verbose:
            print("verbose is True")
            split_cumulative_stats = {
                "total": 0,
                "successful_attacks": 0,
                "pred_adv_distribution(successful_only)": [],
                "y_clean_distribution(successful_only)": [],
            }
            if target_class != -1:
                split_cumulative_stats["successful_with_target"] = 0

        for num_attacks, eps_coef in num_attacks_eps_coef:
            print(f"    Attacks: num={num_attacks}, eps_coef={eps_coef}")

            # Track statistics for this attack configuration
            total_attempted = 0
            total_successful = 0

            # Track per-attack-config statistics only if verbose (for printing)
            if verbose:
                cumulative_stats = {
                    "total": 0,
                    "successful_attacks": 0,
                    "pred_adv_distribution(successful_only)": [],
                    "y_clean_distribution(successful_only)": [],
                }
                if target_class != -1:
                    cumulative_stats["successful_with_target"] = 0

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

                    # Get predictions for ALL adversarial examples (before filtering)
                    # This is needed to save metrics per attack strength before filtering
                    all_preds_batch = None
                    all_clean_classes_batch = None
                    pred_adv_all = None
                    if verbose:
                        model.eval()
                        with torch.no_grad():
                            pred_adv_all = model(new_x_adv_batch.to(device)).argmax(dim=1).cpu()
                            all_preds_batch = pred_adv_all
                            all_clean_classes_batch = y_clean_batch.cpu()

                    # Filter successful attacks
                    idx, stats = filter_successful_attacks(
                        model, x_clean_batch, y_clean_batch, new_x_adv_batch, device, target_class, verbose
                    )
                    successful_x_adv = new_x_adv_batch[idx].cpu()
                    successful_y_adv = y_clean_batch[idx].cpu()

                    num_successful = len(successful_x_adv)
                    successful_eps_coef = torch.full((num_successful,), eps_coef, dtype=torch.float32)

                    # all_pred_adv = stats["pred_adv_distribution(successful_only)"]
                    successful_pred_adv = pred_adv_all[idx]

                    # Track statistics
                    total_attempted += len(x_clean_batch)
                    total_successful += len(successful_x_adv)

                    # Accumulate statistics for this attack config (for printing)
                    if verbose and stats is not None:
                        cumulative_stats["total"] += stats["total"]
                        cumulative_stats["successful_attacks"] += stats["successful_attacks"]
                        cumulative_stats["pred_adv_distribution(successful_only)"].append(
                            stats["pred_adv_distribution(successful_only)"]
                        )
                        cumulative_stats["y_clean_distribution(successful_only)"].append(
                            stats["y_clean_distribution(successful_only)"]
                        )

                        if target_class != -1:
                            cumulative_stats["successful_with_target"] += stats["successful_with_target"]

                        # Also accumulate into split-wide stats
                        split_cumulative_stats["total"] += stats["total"]
                        split_cumulative_stats["successful_attacks"] += stats["successful_attacks"]
                        split_cumulative_stats["pred_adv_distribution(successful_only)"].append(
                            stats["pred_adv_distribution(successful_only)"]
                        )
                        split_cumulative_stats["y_clean_distribution(successful_only)"].append(
                            stats["y_clean_distribution(successful_only)"]
                        )

                        if target_class != -1:
                            split_cumulative_stats["successful_with_target"] += stats["successful_with_target"]

                    # Track ALL predictions (before filtering) for this attack config
                    if verbose and all_preds_batch is not None:
                        if "all_preds_distribution" not in cumulative_stats:
                            cumulative_stats["all_preds_distribution"] = []
                            cumulative_stats["all_y_clean_distribution"] = []
                        cumulative_stats["all_preds_distribution"].append(all_preds_batch)
                        cumulative_stats["all_y_clean_distribution"].append(all_clean_classes_batch)

                    if len(successful_x_adv) > 0:
                        x_adv_list.append(successful_x_adv)
                        y_adv_list.append(successful_y_adv)
                        eps_coef_list.append(successful_eps_coef)
                        pred_adv_list.append(successful_pred_adv)

            # Print summary statistics for this attack configuration
            success_rate = total_successful / total_attempted * 100 if total_attempted > 0 else 0
            print(f"      Summary: {total_successful}/{total_attempted} successful attacks ({success_rate:.2f}%)")

            # Print detailed statistics only if verbose (for this attack config)
            if verbose:
                successful = cumulative_stats["successful_attacks"]

                if target_class != -1 and successful > 0:
                    with_target = cumulative_stats["successful_with_target"]
                    print(f"        Successful attacks before target filtering: {successful}")
                    target_pct = with_target / successful * 100
                    print(f"        After target class {target_class} filtering: {with_target} ({target_pct:.1f}%)")

                # Show distribution of adversarial predictions (successful attacks only)
                all_preds = torch.cat(cumulative_stats["pred_adv_distribution(successful_only)"])
                pred_counts = all_preds.bincount(minlength=10)
                total_preds = len(all_preds)
                pred_percentages = (pred_counts / total_preds * 100).tolist()

                print(f"        Adversarial prediction distribution (counts): {pred_counts.tolist()}")
                print(f"        Adversarial prediction distribution (%): {[f'{p:.1f}' for p in pred_percentages]}")

            # Save CSV for this attack configuration (ALL predictions, before filtering)
            if (
                verbose
                and "all_preds_distribution" in cumulative_stats
                and len(cumulative_stats["all_preds_distribution"]) > 0
            ):
                all_clean_classes_all = torch.cat(cumulative_stats["all_y_clean_distribution"])
                all_preds_all = torch.cat(cumulative_stats["all_preds_distribution"])

                num_classes = 10  # CIFAR-10

                # Create and save 10x10 confusion matrix as CSV (all predictions, before filtering)
                confusion_matrix_all = torch.zeros(num_classes, num_classes, dtype=torch.long)
                for clean_class in range(num_classes):
                    mask = all_clean_classes_all == clean_class
                    if mask.sum() > 0:
                        clean_class_preds = all_preds_all[mask]
                        pred_counts = clean_class_preds.bincount(minlength=num_classes)
                        confusion_matrix_all[clean_class] = pred_counts

                # Save to CSV
                if folder is not None:
                    # Get workspace root (go up from src/nadf/data/adversarial.py)
                    current_file_dir = os.path.dirname(os.path.abspath(__file__))
                    workspace_root = os.path.dirname(os.path.dirname(os.path.dirname(current_file_dir)))

                    # Use folder parameter (relative to workspace root) or absolute path
                    if os.path.isabs(folder):
                        csv_dir = folder
                    else:
                        csv_dir = os.path.join(workspace_root, folder)

                    os.makedirs(csv_dir, exist_ok=True)

                    # Create filename with attack config (per attack strength)
                    attack_config_str = f"{num_attacks}x{eps_coef}"
                    csv_filename = f"prediction_distribution_target_{target_class}_split_{split}_attack_{attack_config_str}_all.csv"
                    csv_path = os.path.join(csv_dir, csv_filename)

                    # Write CSV with header row and index column
                    with open(csv_path, "w", newline="") as f:
                        writer = csv.writer(f)
                        # Header row: predicted class labels
                        header = ["actual_class"] + [f"predicted_class_{i}" for i in range(num_classes)]
                        writer.writerow(header)
                        # Data rows: one per clean class
                        for clean_class in range(num_classes):
                            row = [clean_class] + confusion_matrix_all[clean_class].tolist()
                            writer.writerow(row)

                    print(f"        Saved prediction distribution (all attacks) to {csv_path}")

        # After all attack configurations, save CSV for this split
        if (
            verbose
            and split_cumulative_stats is not None
            and len(split_cumulative_stats["pred_adv_distribution(successful_only)"]) > 0
        ):
            all_clean_classes = torch.cat(split_cumulative_stats["y_clean_distribution(successful_only)"])
            all_preds = torch.cat(split_cumulative_stats["pred_adv_distribution(successful_only)"])

            num_classes = 10  # CIFAR-10

            # Print overall statistics for this split
            print(f"\n    Overall statistics for {split} split:")
            successful = split_cumulative_stats["successful_attacks"]
            if target_class != -1 and successful > 0:
                with_target = split_cumulative_stats["successful_with_target"]
                print(f"      Successful attacks before target filtering: {successful}")
                target_pct = with_target / successful * 100
                print(f"      After target class {target_class} filtering: {with_target} ({target_pct:.1f}%)")

            # Show overall distribution
            pred_counts = all_preds.bincount(minlength=num_classes)
            total_preds = len(all_preds)
            pred_percentages = (pred_counts / total_preds * 100).tolist()
            print(f"      Overall adversarial prediction distribution (counts): {pred_counts.tolist()}")
            print(f"      Overall adversarial prediction distribution (%): {[f'{p:.1f}' for p in pred_percentages]}")

            # Create and save 10x10 confusion matrix as CSV
            # Rows = clean class (actual), Columns = predicted class
            confusion_matrix = torch.zeros(num_classes, num_classes, dtype=torch.long)
            for clean_class in range(num_classes):
                mask = all_clean_classes == clean_class
                if mask.sum() > 0:
                    clean_class_preds = all_preds[mask]
                    pred_counts = clean_class_preds.bincount(minlength=num_classes)
                    confusion_matrix[clean_class] = pred_counts

            # Save to CSV
            if folder is not None:
                # Get workspace root (go up from src/nadf/data/adversarial.py)
                current_file_dir = os.path.dirname(os.path.abspath(__file__))
                workspace_root = os.path.dirname(os.path.dirname(os.path.dirname(current_file_dir)))

                # Use folder parameter (relative to workspace root) or absolute path
                if os.path.isabs(folder):
                    csv_dir = folder
                else:
                    csv_dir = os.path.join(workspace_root, folder)

                os.makedirs(csv_dir, exist_ok=True)

                # Create filename without attack configs (one file per split)
                csv_filename = f"prediction_distribution_target_{target_class}_split_{split}.csv"
                csv_path = os.path.join(csv_dir, csv_filename)

                # Write CSV with header row and index column
                with open(csv_path, "w", newline="") as f:
                    writer = csv.writer(f)
                    # Header row: predicted class labels
                    header = ["actual_class"] + [f"predicted_class_{i}" for i in range(num_classes)]
                    writer.writerow(header)
                    # Data rows: one per clean class
                    for clean_class in range(num_classes):
                        row = [clean_class] + confusion_matrix[clean_class].tolist()
                        writer.writerow(row)

                print(f"      Saved prediction distribution to {csv_path}")

        # Combine all adversarial examples for this split
        if x_adv_list:
            x_adv = torch.vstack(x_adv_list)
            y_adv = torch.cat(y_adv_list)
            eps_coef_adv = torch.cat(eps_coef_list)
            pred_adv = torch.cat(pred_adv_list)
        else:
            x_adv = torch.tensor([])
            y_adv = torch.tensor([])
            eps_coef_adv = torch.tensor([])
            pred_adv = torch.tensor([])

        # Combine clean and adversarial (from pool)
        if len(x_adv) > 0:
            x_combined = torch.vstack([x_clean, x_adv])
            y_combined = torch.cat([y_clean, y_adv])
            eps_coef_clean = torch.zeros(len(x_clean), dtype=torch.float32)
            eps_coef_combined = torch.cat([eps_coef_clean, eps_coef_adv])
            pred_combined = torch.cat([pred_clean, pred_adv]) if len(pred_adv) > 0 else pred_clean

            # For augmented: combine augmented filtered clean + adv (adv is same for both)
            if augment and len(x_clean_filtered_augmented) > 0:
                x_combined_augmented = torch.vstack([x_clean_filtered_augmented, x_adv])
                y_combined_augmented = torch.cat([y_clean_filtered_augmented, y_adv])
                pred_combined_augmented = (
                    torch.cat([pred_clean_augmented, pred_adv]) if len(pred_adv) > 0 else pred_clean_augmented
                )
            else:
                x_combined_augmented = torch.tensor([])
                y_combined_augmented = torch.tensor([])
                pred_combined_augmented = torch.tensor([])

        else:
            x_combined = x_clean
            y_combined = y_clean
            eps_coef_combined = torch.zeros(len(x_clean), dtype=torch.float32)
            pred_combined = pred_clean

            # No adversarial examples, so augmented combined is just augmented clean
            if augment and len(x_clean_filtered_augmented) > 0:
                x_combined_augmented = x_clean_filtered_augmented
                y_combined_augmented = y_clean_filtered_augmented
                pred_combined_augmented = pred_clean_augmented
            else:
                x_combined_augmented = torch.tensor([])
                y_combined_augmented = torch.tensor([])
                pred_combined_augmented = torch.tensor([])

        adv_data[split] = {
            "x_adv": x_adv,
            "y_adv": y_adv,
            "x_clean_augmented": x_clean_filtered_augmented,
            "y_clean_augmented": y_clean_filtered_augmented,
            "eps_coef_adv": eps_coef_adv,
            "x_combined": x_combined,
            "y_combined": y_combined,
            "x_combined_augmented": x_combined_augmented,
            "y_combined_augmented": y_combined_augmented,
            "eps_coef_combined": eps_coef_combined,
            "pred_adv": pred_adv,
            "pred_clean": pred_clean,
            "pred_combined": pred_combined,
            "pred_clean_augmented": pred_clean_augmented,
            "pred_combined_augmented": pred_combined_augmented,
        }

        print(f"    {split}: {len(x_adv)} successful adversarial examples")

        # Clear GPU cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return adv_data


def get_predictions(model, x_data, device, batch_size=512):
    """
    Get model predictions for a dataset in batches.

    Args:
        model: The neural network model
        x_data: Input data tensor
        device: Device to run predictions on
        batch_size: Batch size for prediction

    Returns:
        torch.Tensor: Predictions (argmax of logits) on CPU
    """
    model.eval()
    pred_list = []

    with torch.no_grad():
        for i in range(0, len(x_data), batch_size):
            x_batch = x_data[i : i + batch_size].to(device)
            pred_batch = model(x_batch).argmax(dim=1).cpu()
            pred_list.append(pred_batch)

    if pred_list:
        return torch.cat(pred_list)
    else:
        return torch.tensor([])


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

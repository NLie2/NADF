"""
Dataset creation functions for triplet, regression, and binary classification tasks.
"""

import torch
import torch.nn.functional as functional


def find_pos_neg_samples_regression(
    anchor_z,
    anchor_y,
    z_clean_reference,
    y_clean_reference,
    z_clean_reference_full=None,
    y_clean_reference_full=None,
    distance_metric="euclidean",
    positive_strategy="nearest",
):
    """
    For a given anchor representation, finds a positive and a negative sample.
    Modified for regression task - allows self-matches (distance 0 for clean examples).

    Args:
        anchor_z (torch.Tensor): The representation of the anchor sample (1D tensor).
        anchor_y (torch.Tensor): The label of the anchor sample (scalar).
        z_clean_reference (torch.Tensor): Pool of clean representations for positive
            sampling.
        y_clean_reference (torch.Tensor): Labels for the clean representations.
        distance_metric (str): 'euclidean' or 'cosine'.
        z_clean_reference_full (torch.Tensor, optional): Full pool for negative
            sampling if different from z_clean_reference.
        y_clean_reference_full (torch.Tensor, optional): Full labels for negative
            sampling if different from y_clean_reference.
        positive_strategy (str): How to select positive sample. Options:
            - "nearest": Use nearest same-class example (original behavior)
            - "centroid": Use centroid (mean) of all same-class examples


    Returns:
        (torch.Tensor, torch.Tensor): The positive sample (z_p) and negative sample (z_n).
    """
    # Find all clean samples with the same label as the anchor
    positive_indices = (y_clean_reference == anchor_y).nonzero(as_tuple=True)[0]

    # Calculate distances to all potential positives
    potential_positives = z_clean_reference[positive_indices]

    if positive_strategy == "nearest":
        if distance_metric == "euclidean":
            dists = torch.cdist(anchor_z.unsqueeze(0), potential_positives, p=2.0).squeeze(0)
        elif distance_metric == "cosine":
            dists = 1 - functional.cosine_similarity(anchor_z.unsqueeze(0), potential_positives)
        else:
            raise ValueError(f"Unknown distance metric: {distance_metric}")

        # Find the closest positive sample (INCLUDING self-matches for regression task)
        closest_positive_idx = positive_indices[torch.argmin(dists)]
        z_p = z_clean_reference[closest_positive_idx]

    elif positive_strategy == "centroid":
        # New behavior: use centroid of all same-class examples
        centroid = potential_positives.mean(dim=0)
        z_p = centroid

    else:
        raise ValueError(f"Unknown positive_strategy: {positive_strategy}. Use 'nearest' or 'centroid'.")

    # Find all clean samples with a different label for negative sampling
    # Use full reference pool if provided, otherwise use the filtered one
    z_neg_pool = z_clean_reference_full if z_clean_reference_full is not None else z_clean_reference
    y_neg_pool = y_clean_reference_full if y_clean_reference_full is not None else y_clean_reference

    negative_indices = (y_neg_pool != anchor_y).nonzero(as_tuple=True)[0]

    if len(negative_indices) == 0:
        # Fallback: if no different labels available, use a random sample from the same class
        # This should rarely happen, but prevents crashes
        negative_indices = torch.arange(len(z_neg_pool))
        # print("Warning: No different class samples found for negative sampling. Using same class.")

    # Randomly select one negative sample
    random_negative_idx = negative_indices[torch.randint(0, len(negative_indices), (1,))]
    z_n = z_neg_pool[random_negative_idx]

    return z_p, z_n


def create_regression_dataset(
    z_clean,
    y_clean,
    z_adv,
    y_adv,
    distance_metric="euclidean",
    clean_upweight_factor=1.0,
    z_clean_pool=None,
    y_clean_pool=None,
    split="train",
    eps_coef_adv=None,
    pred_adv=None,
    pred_clean=None,
    pred_combined=None,
    positive_strategy="nearest",
):
    """
    Create regression dataset with distance targets and sample weights.

    Args:
        z_clean (torch.Tensor): Clean representations.
        y_clean (torch.Tensor): Labels for clean representations.
        z_adv (torch.Tensor): Adversarial representations.
        y_adv (torch.Tensor): Labels for adversarial representations.
        distance_metric (str): The distance metric for finding positives.
        clean_upweight_factor (float): Weight factor for clean examples (>1 upweights clean).
        z_clean_pool (torch.Tensor, optional): Full pool for negative sampling.
        y_clean_pool (torch.Tensor, optional): Full labels for negative sampling.

    Returns:
        tuple: (anchors, distances, labels, sample_weights)
    """
    # Normalize the clean pool once
    z_clean_norm = functional.normalize(z_clean, p=2, dim=1)

    # Normalize full reference pool if provided
    z_clean_pool_norm = None
    if z_clean_pool is not None:
        z_clean_pool_norm = functional.normalize(z_clean_pool, p=2, dim=1)

    # PRE-COMPUTE CENTROIDS for all classes (if using centroid strategy)
    centroids = {}
    if positive_strategy == "centroid":
        print("pre-computing centroids")
        pool_to_use = z_clean_pool_norm if z_clean_pool_norm is not None else z_clean_norm
        labels_to_use = y_clean_pool if y_clean_pool is not None else y_clean

        unique_classes = labels_to_use.unique()
        for cls in unique_classes:
            cls_mask = labels_to_use == cls
            if cls_mask.sum() > 0:
                centroids[cls.item()] = pool_to_use[cls_mask].mean(dim=0)
            else:
                print(f"Warning: No examples found for class {cls.item()}")

    # Process clean examples
    clean_anchors, clean_distances, clean_labels, clean_weights = [], [], [], []
    clean_eps_coefs = []
    clean_preds = []

    for i in range(len(z_clean)):
        anchor_z = functional.normalize(z_clean[i].unsqueeze(0), p=2, dim=1).squeeze(0)
        anchor_y = y_clean[i]

        # if positive_strategy == "centroid":
        #     # Use pre-computed centroid
        #     cls_key = anchor_y.item()
        #     if cls_key in centroids:
        #         z_p = centroids[cls_key]
        #         distance = torch.linalg.norm(anchor_z - z_p)
        #     else:
        #         print(f"Warning: No centroid for class {cls_key}")
        #         distance = torch.tensor(float("nan"))
        # else:
        #     # Use original nearest neighbor approach
        #     z_p, z_n = find_pos_neg_samples_regression(
        #         anchor_z,
        #         anchor_y,
        #         z_clean_pool_norm if z_clean_pool_norm is not None else z_clean_norm,
        #         y_clean_pool if y_clean_pool is not None else y_clean,
        #         z_clean_pool_norm,
        #         y_clean_pool,
        #         distance_metric,
        #         positive_strategy,x
        #     )
        #     distance = torch.linalg.norm(anchor_z - z_p)

        # Clean examples have distance 0 (matching to themselves)
        distance = torch.tensor(0.0)

        clean_anchors.append(anchor_z)
        clean_distances.append(distance)
        clean_labels.append(anchor_y)
        clean_weights.append(clean_upweight_factor)
        clean_eps_coefs.append(torch.tensor(0.0))

        # ADD THIS: Append prediction for this clean example
        if pred_clean is not None:
            clean_preds.append(pred_clean[i])
        else:
            clean_preds.append(torch.tensor(float("nan")))

    # Process adversarial examples - find distance to nearest clean example
    adv_anchors, adv_distances, adv_labels, adv_weights = [], [], [], []
    adv_eps_coefs = []
    adv_preds = []

    for i in range(len(z_adv)):
        anchor_z = functional.normalize(z_adv[i].unsqueeze(0), p=2, dim=1).squeeze(0)
        anchor_y = y_adv[i]

        if positive_strategy == "centroid":
            # Use pre-computed centroid
            cls_key = anchor_y.item()
            if cls_key in centroids:
                z_p = centroids[cls_key]
                distance = torch.linalg.norm(anchor_z - z_p)
            else:
                print(f"Warning: No centroid for class {cls_key}")
                distance = torch.tensor(float("nan"))
        else:
            z_p, z_n = find_pos_neg_samples_regression(
                anchor_z,
                anchor_y,
                z_clean_norm,
                y_clean,
                z_clean_pool_norm,
                y_clean_pool,
                distance_metric,
                positive_strategy,
            )

            distance = torch.linalg.norm(anchor_z - z_p)

        adv_anchors.append(anchor_z)
        adv_distances.append(distance)
        adv_labels.append(anchor_y)
        adv_weights.append(1.0)

        if eps_coef_adv is not None:
            adv_eps_coefs.append(eps_coef_adv[i])
        else:
            adv_eps_coefs.append(torch.tensor(float("nan")))  # Use NaN if not provided

        if pred_adv is not None:
            adv_preds.append(pred_adv[i])
        else:
            adv_preds.append(torch.tensor(float("nan")))

    # Stack clean and adversarial together
    all_anchors = clean_anchors + adv_anchors
    all_distances = clean_distances + adv_distances
    all_labels = clean_labels + adv_labels
    all_weights = clean_weights + adv_weights
    all_eps_coefs = clean_eps_coefs + adv_eps_coefs
    all_preds = clean_preds + adv_preds

    # output: anchors, distances, labels, weights, eps_coefs
    return (
        torch.stack(all_anchors),
        torch.stack(all_distances),
        torch.stack(all_labels),
        torch.tensor(all_weights),
        torch.tensor(all_eps_coefs),
        torch.stack(all_preds),
    )


def create_regression_dataset_old1(
    z_all,
    y_all,
    z_clean,
    y_clean,
    distance_metric="euclidean",
    clean_upweight_factor=1.0,
    z_clean_pool=None,
    y_clean_pool=None,
    split="train",
):
    """
    Create regression dataset with distance targets and sample weights.

    Args:
        z_all (torch.Tensor): All representations (clean + adversarial).
        y_all (torch.Tensor): Corresponding labels.
        z_clean (torch.Tensor): Pool of clean representations for positive sampling.
        y_clean(torch.Tensor): Labels for the clean representations.
        distance_metric (str): The distance metric for finding positives.
        clean_upweight_factor (float): Weight factor for clean examples (>1 upweights clean).
        z_clean_pool (torch.Tensor, optional): Full pool for negative sampling.
        y_clean_pool (torch.Tensor, optional): Full labels for negative sampling.

    Returns:
        tuple: (anchors, distances, labels, sample_weights)
    """
    anchors, distances, labels, sample_weights, is_clean_flags = [], [], [], [], []
    if len(z_all) == 0:
        return torch.tensor([]), torch.tensor([]), torch.tensor([]), torch.tensor([])

    # Normalize the clean pool once
    z_clean_norm = functional.normalize(z_clean, p=2, dim=1)
    print("normalized")

    for i in range(len(z_all)):
        # Normalize anchor the SAME way as pool
        anchor_z = functional.normalize(z_all[i].unsqueeze(0), p=2, dim=1).squeeze(0)
        anchor_y = y_all[i]

        # Use the normalized pool for finding matches
        # Normalize full reference pool if provided
        z_clean_pool_norm = None
        if z_clean_pool is not None:
            z_clean_pool_norm = functional.normalize(z_clean_pool, p=2, dim=1)

        z_p, z_n = find_pos_neg_samples_regression(
            anchor_z,
            anchor_y,
            z_clean_norm,
            y_clean,
            distance_metric,
            z_clean_pool_norm,
            y_clean_pool,
        )

        # Calculate distance to determine if this is a clean example
        distance = torch.linalg.norm(anchor_z - z_p)  # Normalized - Normalized = 0 for self-matches
        is_clean = distance < 1e-6  # Clean examples have distance ~0

        anchors.append(anchor_z)
        distances.append(distance)
        labels.append(anchor_y)
        is_clean_flags.append(is_clean)

        # Set sample weight: higher for clean examples if upweighting
        weight = clean_upweight_factor if is_clean else 1.0
        sample_weights.append(weight)

    return torch.stack(anchors), torch.stack(distances), torch.stack(labels), torch.tensor(sample_weights)


def create_regression_dataset_old(z_all, y_all, z_clean_pool, y_clean_pool, distance_metric="euclidean"):
    """
    Creates a triplet dataset (anchor, positive, negative) from representations.

    Args:
        z_all (torch.Tensor): All representations (clean + adversarial).
        y_all (torch.Tensor): Corresponding labels.
        z_clean_pool (torch.Tensor): Pool of clean representations to sample from.
        y_clean_pool (torch.Tensor): Labels for the clean pool.
        distance_metric (str): The distance metric for finding positives.

    Returns:
        (torch.Tensor, torch.Tensor, torch.Tensor): anchors, positives, negatives.
    """
    anchors, distances, labels = [], [], []

    if len(z_all) == 0:
        return torch.tensor([]), torch.tensor([]), torch.tensor([])

    # Normalize the clean pool once
    z_clean_pool = F.normalize(z_clean_pool, p=2, dim=1)

    for i in range(len(z_all)):
        anchor_z = F.normalize(z_all[i], p=2, dim=0)  # L2 normalize
        anchor_y = y_all[i]

        z_p, z_n = find_pos_neg_samples_regression(anchor_z, anchor_y, z_clean_pool, y_clean_pool, distance_metric)

        anchors.append(anchor_z)
        distances.append(torch.linalg.norm(anchor_z - z_p))  # Already normalized from the pool
        labels.append(anchor_y)  # Already normalized from the pool

    return torch.stack(anchors), torch.stack(distances), torch.stack(labels)


def create_binary_classification_dataset(z_clean, z_adv):
    """
    Create dataset for binary classification: clean (0) vs adversarial (1)

    Args:
        z_clean: representations of clean examples
        z_adv: representations of adversarial examples

    Returns:
        representations: input features (concatenated z representations)
        binary_labels: 0 for clean, 1 for adversarial
    """

    if len(z_adv) > 0:
        # Concatenate clean and adversarial representations
        representations = torch.vstack((z_clean, z_adv))

        # Create binary labels: 0 for clean, 1 for adversarial
        binary_labels = torch.cat(
            [
                torch.zeros(len(z_clean), dtype=torch.float32, device=z_clean.device),  # Clean = 0
                torch.ones(len(z_adv), dtype=torch.float32, device=z_adv.device),  # Adversarial = 1
            ]
        )
    else:
        # Only clean examples
        representations = z_clean
        binary_labels = torch.zeros(len(z_clean), dtype=torch.float32, device=z_clean.device)

    return representations, binary_labels


# * OLD TRIPLET
# def find_pos_neg_samples(anchor_z, anchor_y, z_clean_reference, y_clean_reference, distance_metric='euclidean'):
#     """
#     For a given anchor representation, finds a positive and a negative sample.

#     Args:
#         anchor_z (torch.Tensor): The representation of the anchor sample (1D tensor).
#         anchor_y (torch.Tensor): The label of the anchor sample (scalar).
#         z_clean_reference (torch.Tensor): Pool of all clean representations for the split.
#         y_clean_reference (torch.Tensor): Labels for the clean representations.
#         distance_metric (str): 'euclidean' or 'cosine'.

#     Returns:
#         (torch.Tensor, torch.Tensor): The positive sample (z_p) and negative sample (z_n).
#     """
#     # Find all clean samples with the same label as the anchor
#     positive_indices = (y_clean_reference == anchor_y).nonzero(as_tuple=True)[0]

#     # Exclude the anchor itself if it's from the clean pool (to avoid picking itself as the closest)
#     # This is a simple check; a more robust way would involve passing the anchor's index.
#     # For now, we assume if a perfect match is found, it's the anchor itself.

#     # Calculate distances to all potential positives
#     potential_positives = z_clean_reference[positive_indices]

#     if distance_metric == 'euclidean':
#         dists = torch.cdist(anchor_z.unsqueeze(0), potential_positives, p=2.0).squeeze(0)
#     elif distance_metric == 'cosine':
#         dists = 1 - functional.cosine_similarity(anchor_z.unsqueeze(0), potential_positives)
#     else:
#         raise ValueError(f"Unknown distance metric: {distance_metric}")

#     # Find the closest positive sample (that is not the anchor itself)
#     # If the anchor is in the pool, its distance will be 0.
#     # We sort distances and pick the first one > 0, or the first one if all are > 0.
#     sorted_dists, sorted_idx = torch.sort(dists)

#     # Find first non-zero distance if anchor might be in the pool
#     non_zero_idx = (sorted_dists > 1e-6).nonzero(as_tuple=True)[0]
#     if len(non_zero_idx) > 0:
#         closest_idx_in_sorted = non_zero_idx[0]
#     else:
#         # This case happens if the anchor is the only sample of its class, or all others are identical
#         # We just pick the first one (which might be the anchor itself)

#         closest_idx_in_sorted = 0

#     closest_positive_idx = positive_indices[sorted_idx[closest_idx_in_sorted]]
#     z_p = z_clean_reference[closest_positive_idx]

#     # Find all clean samples with a different label
#     negative_indices = (y_clean_reference != anchor_y).nonzero(as_tuple=True)[0]

#     # Randomly select one negative sample
#     random_negative_idx = negative_indices[torch.randint(0, len(negative_indices), (1,))]
#     z_n = z_clean_reference[random_negative_idx]

#     return z_p, z_n

# def create_triplet_dataset(z_all, y_all, z_clean_reference, y_clean_reference, distance_metric='euclidean'):
#     """
#     Creates a triplet dataset (anchor, positive, negative) from representations.

#     Args:
#         z_all (torch.Tensor): All representations (clean + adversarial).
#         y_all (torch.Tensor): Corresponding labels.
#         z_clean_reference (torch.Tensor): Pool of clean representations to sample from.
#         y_clean_reference (torch.Tensor): Labels for the clean pool.
#         distance_metric (str): The distance metric for finding positives.

#     Returns:
#         (torch.Tensor, torch.Tensor, torch.Tensor): The triplet dataset.
#     """
#     anchors, positives, negatives = [], [], []

#     if len(z_all) == 0:
#         return torch.tensor([]), torch.tensor([]), torch.tensor([])

#     # Normalize the clean pool once
#     z_clean_reference = F.normalize(z_clean_reference, p=2, dim=1)

#     for i in range(len(z_all)):
#         anchor_z = F.normalize(z_all[i], p=2, dim=0) # L2 normalize
#         anchor_y = y_all[i]

#         z_p, z_n = find_pos_neg_samples(anchor_z, anchor_y, z_clean_reference, y_clean_reference, distance_metric)

#         anchors.append(anchor_z)
#         positives.append(z_p) # Already normalized from the pool
#         negatives.append(z_n) # Already normalized from the pool

#     return torch.stack(anchors), torch.stack(positives), torch.stack(negatives)

import numpy as np
import torch
from itertools import product


def generate_masks(attr: np.ndarray, mask_size: int) -> torch.Tensor:
    """
    Generates mask for images with feature importance scores
    Args:
        attr: feature importance scores
        mask_size: number of pixels masked

    Returns:
        Mask hiding most important pixels

    """
    dataset_size, n_chanels, H, W = attr.shape
    attr = torch.from_numpy(np.sum(np.abs(attr), axis=1, keepdims=True))  # Sum the attribution over the channels
    masks = torch.ones(attr.shape)
    masks = masks.view(dataset_size, -1)  # Reshape to make it compatible with torch.topk
    top_pixels = torch.topk(attr.view(dataset_size, -1), mask_size)[1]
    for feature_id, example_id in product(range(mask_size), range(dataset_size)):
        masks[example_id, top_pixels[example_id, feature_id]] = 0
    masks = masks.view(dataset_size, 1, H, W)
    return masks


def generate_tseries_masks(attr: np.ndarray, mask_size: int) -> torch.Tensor:
    """
        Generates mask for time series with feature importance scores
        Args:
            attr: feature importance scores
            mask_size: number of time steps masked

        Returns:
            Mask hiding most important time steps

        """
    dataset_size, time_steps, n_chanels = attr.shape
    attr = torch.from_numpy(np.sum(np.abs(attr), axis=-1, keepdims=True))  # Sum the attribution over the channels
    masks = torch.ones(attr.shape)
    masks = masks.view(dataset_size, -1)  # Reshape to make it compatible with torch.topk
    top_time_steps = torch.topk(attr.view(dataset_size, -1), mask_size)[1]
    for feature_id, example_id in product(range(mask_size), range(dataset_size)):
        masks[example_id, top_time_steps[example_id, feature_id]] = 0
    masks = masks.view(dataset_size, time_steps, n_chanels)
    return masks

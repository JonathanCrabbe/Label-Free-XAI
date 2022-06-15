import numpy as np
import torch
from scipy.stats import entropy, spearmanr


def off_diagonal_sum(mat: np.ndarray) -> np.ndarray:
    """
    Computes the sum of of-diagonal matrix elements
    Args:
        mat: matrix

    Returns:
        sum of the off diagonal elements of mat
    """
    return np.sum(mat) - np.trace(mat)


def pearson_saliency(saliency: np.ndarray) -> np.ndarray:
    """
    Computes the average Pearson correlation between different saliency maps
    Args:
        saliency: saliency maps stacked together (indexed by the first tensor dimension)

    Returns:
        Pearson correlation between saliency maps
    """
    latent_dim = saliency.shape[1]
    corr = np.corrcoef(saliency.swapaxes(0, 1).reshape(latent_dim, -1))
    return off_diagonal_sum(corr) / (latent_dim * (latent_dim - 1))


def spearman_saliency(saliency: np.ndarray) -> np.ndarray:
    """
    Computes the average Spearman correlation between different saliency maps
    Args:
        saliency: saliency maps stacked together (indexed by the first tensor dimension)

    Returns:
        Spearman correlation between saliency maps
    """
    latent_dim = saliency.shape[1]
    corr = spearmanr(saliency.swapaxes(0, 1).reshape(latent_dim, -1), axis=1)[0]
    return off_diagonal_sum(corr) / (latent_dim * (latent_dim - 1))


def cos_saliency(saliency: np.ndarray) -> np.ndarray:
    """
    Computes the average cosine between different saliency maps
    Args:
        saliency: saliency maps stacked together (indexed by the first tensor dimension)

    Returns:
        cosine  between saliency maps
    """
    latent_dim = saliency.shape[1]
    cos_avg = np.ones((latent_dim, latent_dim))
    for dim1 in range(1, latent_dim):
        for dim2 in range(dim1):
            saliency_dim1, saliency_dim2 = saliency[:, dim1], saliency[:, dim2]
            normalization = np.sqrt(
                np.sum(saliency_dim1**2, axis=(-2, -1))
                * np.sum(saliency_dim2**2, axis=(-2, -1))
            )
            cos_dim1_dim2 = (
                np.sum(saliency_dim1 * saliency_dim2, axis=(-2, -1)) / normalization
            )
            cos_avg[dim1, dim2] = np.mean(cos_dim1_dim2)
            cos_avg[dim2, dim1] = cos_avg[dim1, dim2]
    return off_diagonal_sum(cos_avg) / (latent_dim * (latent_dim - 1))


def entropy_saliency(saliency: np.ndarray) -> np.ndarray:
    """
    Computes the entropy of different saliency maps
    Args:
        saliency: saliency maps stacked together (indexed by the first tensor dimension)

    Returns:
        Entropy between saliency maps
    """
    latent_dim = saliency.shape[1]
    saliency_reshaped = np.reshape(saliency.swapaxes(1, -1), (-1, latent_dim))
    salient_pixels = saliency_reshaped.sum(1) > 0
    saliency_filtered = saliency_reshaped[salient_pixels]
    entropy_ar = entropy(saliency_filtered, axis=1)
    return np.mean(entropy_ar)


def count_activated_neurons(saliency: np.ndarray) -> np.ndarray:
    """
    Count the average number of neurons sensitive to a feature
    Args:
        saliency: saliency maps stacked together (indexed by the first tensor dimension)

    Returns:
        Average number of neurons sensitive to a feature
    """
    latent_dim = saliency.shape[1]
    saliency_reshaped = np.reshape(saliency.swapaxes(1, -1), (-1, latent_dim))
    salient_pixels = saliency_reshaped.sum(1) > 0
    saliency_filtered = saliency_reshaped[salient_pixels]
    activated_neurons = (
        saliency_filtered / saliency_filtered.sum(1, keepdims=True) > 1 / latent_dim
    )
    n_activated_neurons = np.count_nonzero(activated_neurons, axis=1)
    return np.mean(n_activated_neurons)


def similarity_rates(
    example_importance: torch.Tensor,
    labels_subtrain: torch.Tensor,
    labels_test: torch.Tensor,
    n_top_list: list = [1, 2, 5, 10, 20, 30, 40, 50],
) -> tuple:
    """
    Computes the similarity rate metric (see paper)
    Args:
        example_importance: attribution for each example
        labels_subtrain: labels of the train examples
        labels_test: labels of the test examples
        n_top_list: number of training examples to consider per test example

    Returns:
        Similary rates of most and least important examples for each element in n_to_list
    """
    test_size, subtrain_size = example_importance.shape
    result_most = []
    result_least = []
    for n_top in n_top_list:
        most_important_examples = torch.topk(example_importance, k=n_top)[1]
        least_important_examples = torch.topk(
            example_importance, k=n_top, largest=False
        )[1]
        similarities_most = []
        similarities_least = []
        for n in range(test_size):
            most_important_labels = labels_subtrain[most_important_examples[n]]
            least_important_labels = labels_subtrain[least_important_examples[n]]
            similarities_most.append(
                torch.count_nonzero(most_important_labels == labels_test[n]).item()
                / n_top
            )
            similarities_least.append(
                torch.count_nonzero(least_important_labels == labels_test[n]).item()
                / n_top
            )
        result_most.append(np.mean(similarities_most))
        result_least.append(np.mean(similarities_least))
    return result_most, result_least


def compute_metrics(data: np.ndarray, metrics: callable) -> list:
    return [metric(data) for metric in metrics]


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name):
        self.name = name
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

import numpy as np
import torch
from scipy.stats import entropy, spearmanr


def off_diagonal_sum(mat: np.ndarray) -> np.ndarray:
    return np.sum(mat) - np.trace(mat)


def pearson_saliency(saliency: np.ndarray) -> np.ndarray:
    latent_dim = saliency.shape[1]
    corr = np.corrcoef(saliency.swapaxes(0, 1).reshape(latent_dim, -1))
    return off_diagonal_sum(corr) / (latent_dim * (latent_dim - 1))


def spearman_saliency(saliency: np.ndarray) -> np.ndarray:
    latent_dim = saliency.shape[1]
    corr = spearmanr(saliency.swapaxes(0, 1).reshape(latent_dim, -1), axis=1)[0]
    return off_diagonal_sum(corr) / (latent_dim * (latent_dim - 1))


def cos_saliency(saliency: np.ndarray) -> np.ndarray:
    latent_dim = saliency.shape[1]
    cos_avg = np.ones((latent_dim, latent_dim))
    for dim1 in range(1, latent_dim):
        for dim2 in range(dim1):
            saliency_dim1, saliency_dim2 = saliency[:, dim1], saliency[:, dim2]
            normalization = np.sqrt(np.sum(saliency_dim1**2, axis=(-2, -1)) *
                                    np.sum(saliency_dim2**2, axis=(-2, -1)))
            cos_dim1_dim2 = np.sum(saliency_dim1*saliency_dim2, axis=(-2, -1))/normalization
            cos_avg[dim1, dim2] = np.mean(cos_dim1_dim2)
            cos_avg[dim2, dim1] = cos_avg[dim1, dim2]
    return off_diagonal_sum(cos_avg) / (latent_dim * (latent_dim - 1))


def entropy_saliency(saliency: np.ndarray) -> np.ndarray:
    latent_dim = saliency.shape[1]
    saliency_reshaped = np.reshape(saliency.swapaxes(1, -1), (-1, latent_dim))
    salient_pixels = saliency_reshaped.sum(1) > 0
    saliency_filtered = saliency_reshaped[salient_pixels]
    entropy_ar = entropy(saliency_filtered, axis=1)
    return np.mean(entropy_ar)


def count_activated_neurons(saliency: np.ndarray) -> np.ndarray:
    latent_dim = saliency.shape[1]
    saliency_reshaped = np.reshape(saliency.swapaxes(1, -1), (-1, latent_dim))
    salient_pixels = saliency_reshaped.sum(1) > 0
    saliency_filtered = saliency_reshaped[salient_pixels]
    activated_neurons = saliency_filtered/saliency_filtered.sum(1, keepdims=True) > 1/latent_dim
    n_activated_neurons = np.count_nonzero(activated_neurons, axis=1)
    return np.mean(n_activated_neurons)


def similarity_rate(example_importance: torch.Tensor, labels_subtrain: torch.Tensor, labels_test: torch.Tensor,
                    num_top: int = 5) -> list:
    test_size, subtrain_size = example_importance.shape
    most_important_examples = torch.topk(example_importance, k=num_top)[1]
    similarity_rates = []
    for n in range(test_size):
        most_important_labels = labels_subtrain[most_important_examples[n]]
        similarity_rates.append(torch.count_nonzero(most_important_labels == labels_test[n]).item()
                                / num_top)
    return similarity_rates


def similarity_rates(example_importance: torch.Tensor, labels_subtrain: torch.Tensor, labels_test: torch.Tensor,
                     n_top_list: list = [1, 2, 5, 10, 20, 30, 40, 50]) -> tuple:
    test_size, subtrain_size = example_importance.shape
    result_most = []
    result_least = []
    for n_top in n_top_list:
        most_important_examples = torch.topk(example_importance, k=n_top)[1]
        least_important_examples = torch.topk(example_importance, k=n_top, largest=False)[1]
        similarities_most = []
        similarities_least = []
        for n in range(test_size):
            most_important_labels = labels_subtrain[most_important_examples[n]]
            least_important_labels = labels_subtrain[least_important_examples[n]]
            similarities_most.append(torch.count_nonzero(most_important_labels == labels_test[n]).item() / n_top)
            similarities_least.append(torch.count_nonzero(least_important_labels == labels_test[n]).item() / n_top)
        result_most.append(np.mean(similarities_most))
        result_least.append(np.mean(similarities_least))
    return result_most, result_least


def top_consistency(example_importances: torch.Tensor, num_top: int = 5) -> np.ndarray:
    n_tasks, n_test, n_train = example_importances.shape
    top_idx = torch.topk(torch.from_numpy(example_importances), num_top)[1]
    consistency_matrix = np.ones((n_tasks, n_tasks))
    for task1 in range(1, n_tasks):
        for task2 in range(task1):
            top_task1 = top_idx[task1].numpy()
            top_task2 = top_idx[task2].numpy()
            consistency_matrix[task1, task2] = np.sum(
                [len(np.intersect1d(top_task1[id_test], top_task2[id_test])) for id_test in range(n_test)]
            ) / (num_top*n_test)
            consistency_matrix[task2, task1] = consistency_matrix[task1, task2]
    return consistency_matrix


def compute_metrics(data: np.ndarray, metrics: callable) -> list:
    return [metric(data) for metric in metrics]


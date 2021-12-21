from captum.attr._utils.visualization import visualize_image_attr
import torch
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt


def plot_image_saliency(image: torch.Tensor, saliency: torch.Tensor):
    image_np = image.permute((1, 2, 0)).cpu().numpy()
    saliency_np = saliency.permute((1, 2, 0)).cpu().numpy()
    visualize_image_attr(saliency_np, image_np)


def plot_vae_saliencies(saliency: np.ndarray) -> plt.Figure:
    W = saliency.shape[-1]
    n_plots = len(saliency)
    dim_latent = saliency.shape[1]
    cblind_palette = sns.color_palette("colorblind")
    fig, axs = plt.subplots(ncols=dim_latent, nrows=n_plots, figsize=(4 * dim_latent, 4 * n_plots))
    for example_id in range(n_plots):
        max_saliency = np.max(saliency[example_id])
        for dim in range(dim_latent):
            sub_saliency = saliency[example_id, dim]
            ax = axs[example_id, dim]
            h = sns.heatmap(np.reshape(sub_saliency, (W, W)), linewidth=0, xticklabels=False, yticklabels=False,
                            ax=ax, cmap=sns.light_palette(cblind_palette[dim], as_cmap=True), cbar=True,
                            alpha=1, zorder=2, vmin=0, vmax=max_saliency)
    return fig

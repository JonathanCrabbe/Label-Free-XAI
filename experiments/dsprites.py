import os
import logging
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
import torch
import itertools
import csv
from captum.attr import GradientShap
from torch.utils.data import DataLoader, random_split
from explanations.features import attribute_individual_dim
from models.images import VAE, EncoderBurgess,\
    DecoderBurgess
from models.losses import BetaHLoss, BtcvaeLoss
from utils.metrics import off_diagonal_sum, cos_saliency, entropy_saliency,\
    count_activated_neurons, correlation_saliency, compute_metrics
from utils.datasets import DSprites
from utils.visualize import vae_box_plots, plot_vae_saliencies


def disvae_feature_importance(random_seed: int = 1, batch_size: int = 500, n_plots: int = 20, n_runs: int = 5,
                              dim_latent: int = 6, n_epochs: int = 100, beta_list: list = [1, 5, 10],
                              test_split=0.1) -> None:
    # Initialize seed and device
    np.random.seed(random_seed)
    torch.random.manual_seed(random_seed)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # Load dsprites
    W = 64
    img_size = (1, W, W)
    data_dir = Path.cwd() / "data/dsprites"
    dsprites_dataset = DSprites(str(data_dir))
    test_size = int(test_split * len(dsprites_dataset))
    train_size = len(dsprites_dataset) - test_size
    train_dataset, test_dataset = random_split(dsprites_dataset, [train_size, test_size])
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Create saving directory
    save_dir = Path.cwd() / "results/dsprites/vae"
    if not save_dir.exists():
        os.makedirs(save_dir)

    # Define the computed metrics and create a csv file with appropriate headers
    loss_list = [BetaHLoss(), BtcvaeLoss(is_mss=False, n_data=len(train_dataset))]
    metric_list = [correlation_saliency, cos_saliency, entropy_saliency, count_activated_neurons]
    metric_names = ["Correlation", "Cosine", "Entropy", "Active Neurons"]
    headers = ["Loss Type", "Beta"] + metric_names
    csv_path = save_dir / "metrics.csv"
    if not csv_path.is_file():
        logging.info(f"Creating metrics csv in {csv_path}")
        with open(csv_path, 'w') as csv_file:
            dw = csv.DictWriter(csv_file, delimiter=',', fieldnames=headers)
            dw.writeheader()

    for beta, loss, run in itertools.product(beta_list, loss_list, range(1, n_runs + 1)):
        # Initialize vaes
        encoder = EncoderBurgess(img_size, dim_latent)
        decoder = DecoderBurgess(img_size, dim_latent)
        loss.beta = beta
        name = f'{str(loss)}-vae_beta{beta}_run{run}'
        model = VAE(img_size, encoder, decoder, dim_latent, loss, name=name)
        logging.info(f"Now fitting {name}")
        model.fit(device, train_loader, test_loader, save_dir, n_epochs)
        model.load_state_dict(torch.load(save_dir / (name + ".pt")), strict=False)

        # Compute test-set saliency and associated metrics
        baseline_image = torch.zeros((1, 1, W, W), device=device)
        gradshap = GradientShap(encoder.mu)
        attributions = attribute_individual_dim(encoder.mu, dim_latent, test_loader, device, gradshap, baseline_image)
        metrics = compute_metrics(attributions, metric_list)
        results_str = '\t'.join([f'{metric_names[k]} {metrics[k]:.2g}' for k in range(len(metric_list))])
        logging.info(f"Model {name} \t {results_str}")

        # Save the metrics
        with open(csv_path, 'a', newline='') as csv_file:
            writer = csv.writer(csv_file, delimiter=',')
            writer.writerow([str(loss), beta] + metrics)

        # Plot a couple of examples
        plot_idx = [n for n in range(n_plots)]
        images_to_plot = [test_dataset[i][0].numpy().reshape(W, W) for i in plot_idx]
        fig = plot_vae_saliencies(images_to_plot, attributions[plot_idx])
        fig.savefig(save_dir / f"{name}.pdf")

    fig = vae_box_plots(pd.read_csv(csv_path), metric_names)
    fig.savefig(save_dir / 'metric_box_plots.pdf')


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    disvae_feature_importance()


"""
 baseline_image = torch.zeros((1, 1, W, W), device=device)
        gradshap = GradientShap(encoder.mu)
        attributions = []
        latents = []

        for image_batch, _ in test_loader:
            image_batch = image_batch.to(device)
            attributions_batch = []
            latents.append(encoder.mu(image_batch).detach().cpu().numpy())
            for dim in range(dim_latent):
                attribution = gradshap.attribute(image_batch, baseline_image, target=dim).detach().cpu().numpy()
                attributions_batch.append(np.reshape(attribution, (len(image_batch), 1, W, W)))
            attributions.append(np.concatenate(attributions_batch, axis=1))
        latents = np.concatenate(latents)
        attributions = np.concatenate(attributions)
        attributions = np.abs(np.expand_dims(latents, (2, 3)) * attributions)
        corr = np.corrcoef(attributions.swapaxes(0, 1).reshape(dim_latent, -1))
        metric = off_diagonal_sum(corr)/(dim_latent*(dim_latent-1))
        activated_avg, activated_std = count_activated_neurons(attributions)
        logging.info(f"Model  \t Beta {beta} \t Pearson Correlation {metric:.2g} \t"
                     f" Active Neurons {activated_avg:.2g} +/- {activated_std:.2g} ")
        cblind_palette = sns.color_palette("colorblind")
        fig, axs = plt.subplots(ncols=dim_latent, nrows=n_plots, figsize=(4*dim_latent, 4*n_plots))
        for example_id in range(n_plots):
            max_saliency = np.max(attributions[example_id])
            for dim in range(dim_latent):
                sub_saliency = attributions[example_id, dim, :, :]
                ax = axs[example_id, dim]
                h = sns.heatmap(np.reshape(sub_saliency, (W, W)), linewidth=0, xticklabels=False, yticklabels=False,
                                ax=ax, cmap=sns.light_palette(cblind_palette[dim], as_cmap=True), cbar=True,
                                alpha=1, zorder=2, vmin=0, vmax=max_saliency)

        plt.savefig(save_dir/f"tcvae_{beta}.pdf")

"""

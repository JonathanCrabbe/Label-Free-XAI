import argparse
import csv
import itertools
import logging
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from captum.attr import GradientShap
from explanations.features import attribute_individual_dim
from torch.utils.data import random_split

from lfxai.models.images import VAE, DecoderBurgess, EncoderBurgess
from lfxai.models.losses import BetaHLoss, BtcvaeLoss
from lfxai.utils.datasets import DSprites
from lfxai.utils.metrics import (
    compute_metrics,
    cos_saliency,
    count_activated_neurons,
    entropy_saliency,
    pearson_saliency,
    spearman_saliency,
)
from lfxai.utils.visualize import plot_vae_saliencies, vae_box_plots


def disvae_feature_importance(
    random_seed: int = 1,
    batch_size: int = 500,
    n_plots: int = 10,
    n_runs: int = 5,
    dim_latent: int = 6,
    n_epochs: int = 100,
    beta_list: list = [1, 5, 10],
    test_split=0.1,
) -> None:
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
    train_dataset, test_dataset = random_split(
        dsprites_dataset, [train_size, test_size]
    )
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size)
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False
    )

    # Create saving directory
    save_dir = Path.cwd() / "results/dsprites/vae"
    if not save_dir.exists():
        os.makedirs(save_dir)

    # Define the computed metrics and create a csv file with appropriate headers
    loss_list = [BetaHLoss(), BtcvaeLoss(is_mss=False, n_data=len(train_dataset))]
    metric_list = [
        pearson_saliency,
        spearman_saliency,
        cos_saliency,
        entropy_saliency,
        count_activated_neurons,
    ]
    metric_names = [
        "Pearson Correlation",
        "Spearman Correlation",
        "Cosine",
        "Entropy",
        "Active Neurons",
    ]
    headers = ["Loss Type", "Beta"] + metric_names
    csv_path = save_dir / "metrics.csv"
    if not csv_path.is_file():
        logging.info(f"Creating metrics csv in {csv_path}")
        with open(csv_path, "w") as csv_file:
            dw = csv.DictWriter(csv_file, delimiter=",", fieldnames=headers)
            dw.writeheader()

    for beta, loss, run in itertools.product(
        beta_list, loss_list, range(1, n_runs + 1)
    ):
        # Initialize vaes
        encoder = EncoderBurgess(img_size, dim_latent)
        decoder = DecoderBurgess(img_size, dim_latent)
        loss.beta = beta
        name = f"{str(loss)}-vae_beta{beta}_run{run}"
        model = VAE(img_size, encoder, decoder, dim_latent, loss, name=name)
        logging.info(f"Now fitting {name}")
        model.fit(device, train_loader, test_loader, save_dir, n_epochs)
        model.load_state_dict(torch.load(save_dir / (name + ".pt")), strict=False)

        # Compute test-set saliency and associated metrics
        baseline_image = torch.zeros((1, 1, W, W), device=device)
        gradshap = GradientShap(encoder.mu)
        attributions = attribute_individual_dim(
            encoder.mu, dim_latent, test_loader, device, gradshap, baseline_image
        )
        metrics = compute_metrics(attributions, metric_list)
        results_str = "\t".join(
            [f"{metric_names[k]} {metrics[k]:.2g}" for k in range(len(metric_list))]
        )
        logging.info(f"Model {name} \t {results_str}")

        # Save the metrics
        with open(csv_path, "a", newline="") as csv_file:
            writer = csv.writer(csv_file, delimiter=",")
            writer.writerow([str(loss), beta] + metrics)

        # Plot a couple of examples
        plot_idx = [n for n in range(n_plots)]
        images_to_plot = [test_dataset[i][0].numpy().reshape(W, W) for i in plot_idx]
        fig = plot_vae_saliencies(images_to_plot, attributions[plot_idx])
        fig.savefig(save_dir / f"{name}.pdf")
        plt.close(fig)

    fig = vae_box_plots(pd.read_csv(csv_path), metric_names)
    fig.savefig(save_dir / "metric_box_plots.pdf")
    plt.close(fig)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_runs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=500)
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()
    disvae_feature_importance(
        n_runs=args.n_runs, batch_size=args.batch_size, random_seed=args.seed
    )

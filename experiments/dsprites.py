import os
import logging
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torchvision
from captum.attr import GradientShap, DeepLift, IntegratedGradients, Saliency
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from scipy.stats import pearsonr, spearmanr
from explanations.features import AuxiliaryFunction, attribute_auxiliary
from models.images import EncoderMnist, DecoderMnist, ClassifierMnist, BetaVaeMnist, BetaTcVaeMnist,\
    train_denoiser_epoch, test_denoiser_epoch, train_classifier_epoch, test_classifier_epoch, VAE, EncoderBurgess,\
    DecoderBurgess
from models.losses import BetaHLoss, BtcvaeLoss
from utils.math import off_diagonal_sum, cos_saliency, entropy_saliency, count_activated_neurons
from utils.datasets import DSprites


def disvae_feature_importance(random_seed: int = 1, batch_size: int = 500, n_plots: int = 20,
                              dim_latent: int = 6, n_epochs: int = 100, beta_list: list = [1, 2, 5, 7, 10],
                              test_split=0.1) -> None:
    # Initialize seed and device
    np.random.seed(random_seed)
    torch.random.manual_seed(random_seed)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    loss_types = ["betaH",  "btcvae"]
    W = 64
    img_size = (1, W, W)

    # Load dsprites
    data_dir = Path.cwd() / "data/dsprites"
    dsprites_dataset = DSprites(str(data_dir))
    test_size = int(test_split * len(dsprites_dataset))
    train_size = len(dsprites_dataset) - test_size
    train_dataset, test_dataset = random_split(dsprites_dataset, [train_size, test_size])
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    for beta in beta_list:
        logging.info(f"Now working with beta {beta}")

        # Initialize vaes
        encoder = EncoderBurgess(img_size, dim_latent)
        decoder = DecoderBurgess(img_size, dim_latent)
        loss_f = BtcvaeLoss(beta, is_mss=False)
        model = VAE(img_size, encoder, decoder, dim_latent, loss_f)
        logging.info(f"Now fitting model")
        model.fit(device, train_loader, test_loader, n_epochs)

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
        save_dir = Path.cwd() / "results/dsprites/vae"
        if not save_dir.exists():
            os.makedirs(save_dir)
        plt.savefig(save_dir/f"tcvae_{beta}.pdf")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    disvae_feature_importance()
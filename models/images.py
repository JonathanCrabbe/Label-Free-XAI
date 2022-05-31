import torch.nn.functional as F
import torch
import numpy as np
import math
import logging
import pathlib
import json
import hydra
from torch import nn
from models.losses import BaseVAELoss
from tqdm import tqdm
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LambdaLR
from omegaconf import DictConfig
from utils.datasets import CIFAR10Pair
from utils.metrics import AverageMeter
from pathlib import Path


'''
 These models are adapted from 
 https://medium.com/dataseries/convolutional-autoencoder-in-pytorch-on-mnist-dataset-d65145c132ac
 https://github.com/smartgeometry-ucl/dl4g
 https://github.com/AntixK/PyTorch-VAE
 https://github.com/p3i0t/SimCLR-CIFAR10
'''


def init_vae(img_size, latent_dim, loss_f, name):
    """Return an instance of a VAE with encoder and decoder from `model_type`."""
    encoder = EncoderBurgess(img_size, latent_dim)
    decoder = DecoderBurgess(img_size, latent_dim)
    model = VAE(img_size, encoder, decoder, latent_dim, loss_f, name)
    return model


class EncoderMnist(nn.Module):
    def __init__(self, encoded_space_dim):
        super().__init__()
        self.encoder_cnn = nn.Sequential(
            nn.Conv2d(1, 8, 3, stride=2, padding=1),
            nn.ReLU(True),
            nn.Conv2d(8, 16, 3, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            nn.Conv2d(16, 32, 3, stride=2, padding=0),
            nn.ReLU(True)
        )
        self.flatten = nn.Flatten(start_dim=1)
        self.encoder_lin = nn.Sequential(
            nn.Linear(3 * 3 * 32, 128),
            nn.ReLU(True),
            nn.Linear(128, encoded_space_dim)
        )
        self.encoded_space_dim = encoded_space_dim

    def forward(self, x):
        x = self.encoder_cnn(x)
        x = self.flatten(x)
        x = self.encoder_lin(x)
        return x


class DecoderMnist(nn.Module):
    def __init__(self, encoded_space_dim):
        super().__init__()
        self.decoder_lin = nn.Sequential(
            nn.Linear(encoded_space_dim, 128),
            nn.ReLU(True),
            nn.Linear(128, 3 * 3 * 32),
            nn.ReLU(True)
        )
        self.unflatten = nn.Unflatten(dim=1, unflattened_size=(32, 3, 3))
        self.decoder_conv = nn.Sequential(
            nn.ConvTranspose2d(32, 16, 3, stride=2, output_padding=0),
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            nn.ConvTranspose2d(16, 8, 3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(True),
            nn.ConvTranspose2d(8, 1, 3, stride=2, padding=1, output_padding=1)
        )

    def forward(self, x):
        x = self.decoder_lin(x)
        x = self.unflatten(x)
        x = self.decoder_conv(x)
        x = torch.sigmoid(x)
        return x


class AutoEncoderMnist(nn.Module):

    def __init__(self,  encoder: EncoderMnist, decoder: DecoderMnist,
                 latent_dim: int, input_pert: callable, name: str = "model", loss_f: callable = nn.MSELoss()):
        """
        Class which defines model and forward pass.
        Parameters
        ----------
        img_size : tuple of ints
            Size of images. E.g. (1, 32, 32) or (3, 64, 64).
        """
        super(AutoEncoderMnist, self).__init__()
        self.latent_dim = latent_dim
        self.encoder = encoder
        self.decoder = decoder
        self.input_pert = input_pert
        self.name = name
        self.loss_f = loss_f
        self.checkpoints_files = []
        self.lr = None

    def forward(self, x):
        """
        Forward pass of model.
        Parameters
        ----------
        x : torch.Tensor
            Batch of data. Shape (batch_size, n_chan, height, width)
        """
        if self.training:
            x = self.input_pert(x)
        x = self.encoder(x)
        x = self.decoder(x)
        return x

    def train_epoch(self, device: torch.device, dataloader: torch.utils.data.DataLoader,
                    optimizer: torch.optim.Optimizer) -> np.ndarray:
        self.train()
        train_loss = []
        for image_batch, _ in tqdm(dataloader, unit="batch", leave=False):
            image_batch = image_batch.to(device)
            recon_batch = self.forward(image_batch)
            loss = self.loss_f(image_batch, recon_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss.append(loss.detach().cpu().numpy())
        return np.mean(train_loss)

    def test_epoch(self, device: torch.device, dataloader: torch.utils.data.DataLoader):
        self.eval()
        test_loss = []
        with torch.no_grad():
            for image_batch, _ in dataloader:
                image_batch = image_batch.to(device)
                pert_batch = self.input_pert(image_batch)
                recon_batch = self.forward(pert_batch)
                loss = self.loss_f(image_batch, recon_batch)
                test_loss.append(loss.cpu().numpy())
        return np.mean(test_loss)

    def fit(self, device: torch.device, train_loader: torch.utils.data.DataLoader,
            test_loader: torch.utils.data.DataLoader, save_dir: pathlib.Path,
            n_epoch: int = 30, patience: int = 10, checkpoint_interval: int = -1) -> None:
        self.to(device)
        self.lr = 1e-03
        optim = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=1e-05)
        waiting_epoch = 0
        best_test_loss = float("inf")
        for epoch in range(n_epoch):
            train_loss = self.train_epoch(device, train_loader, optim)
            test_loss = self.test_epoch(device, test_loader)
            logging.info(f'Epoch {epoch + 1}/{n_epoch} \t '
                         f'Train loss {train_loss:.3g} \t Test loss {test_loss:.3g} \t ')
            if test_loss >= best_test_loss:
                waiting_epoch += 1
                logging.info(f'No improvement over the best epoch \t Patience {waiting_epoch} / {patience}')
            else:
                logging.info(f'Saving the model in {save_dir}')
                self.cpu()
                self.save(save_dir)
                self.to(device)
                best_test_loss = test_loss.data
                waiting_epoch = 0
            if checkpoint_interval > 0 and epoch % checkpoint_interval == 0:
                n_checkpoint = 1 + epoch // checkpoint_interval
                logging.info(f'Saving checkpoint {n_checkpoint} in {save_dir}')
                path_to_checkpoint = save_dir / f"{self.name}_checkpoint{n_checkpoint}.pt"
                torch.save(self.state_dict(), path_to_checkpoint)
                self.checkpoints_files.append(path_to_checkpoint)
            if waiting_epoch == patience:
                logging.info(f'Early stopping activated')
                break

    def save(self, directory: pathlib.Path) -> None:
        """
        Save a model and corresponding metadata.
        Parameters
        ----------
        directory : pathlib.Path
            Path to the directory where to save the data.
        """
        model_name = self.name
        self.save_metadata(directory)
        path_to_model = directory / (model_name + ".pt")
        torch.save(self.state_dict(), path_to_model)

    def load_metadata(self, directory: pathlib.Path) -> dict:
        """Load the metadata of a training directory.
        Parameters
        ----------
        directory : pathlib.Path
            Path to folder where model is saved. For example './experiments/mnist'.
        """
        path_to_metadata = directory / (self.name + ".json")

        with open(path_to_metadata) as metadata_file:
            metadata = json.load(metadata_file)
        return metadata

    def save_metadata(self, directory: pathlib.Path, **kwargs) -> None:
        """Load the metadata of a training directory.
        Parameters
        ----------
        directory: string
            Path to folder where to save model. For example './experiments/mnist'.
        kwargs:
            Additional arguments to `json.dump`
        """
        path_to_metadata = directory / (self.name + ".json")
        metadata = {"latent_dim": self.latent_dim,
                    "name": self.name}
        with open(path_to_metadata, 'w') as f:
            json.dump(metadata, f, indent=4, sort_keys=True, **kwargs)


class ClassifierMnist(nn.Module):
    def __init__(self, encoder: EncoderMnist, latent_dim: int, name: str):
        super().__init__()
        self.encoder = encoder
        self.encoder_cnn = encoder.encoder_cnn
        self.flatten = nn.Flatten(start_dim=1)
        self.encoder_lin = encoder.encoder_lin
        self.lin_output = nn.Sequential(
            nn.Linear(latent_dim, 10),
            nn.Softmax(dim=-1)
        )
        self.encoded_space_dim = encoder.encoded_space_dim
        self.loss_f = nn.CrossEntropyLoss()
        self.latent_dim = latent_dim
        self.name = name

    def forward(self, x):
        x = self.encoder_cnn(x)
        x = self.flatten(x)
        x = self.encoder_lin(x)
        x = self.lin_output(x)
        return x

    def train_epoch(self, device: torch.device, dataloader: torch.utils.data.DataLoader,
                    optimizer: torch.optim.Optimizer) -> np.ndarray:
        self.train()
        train_loss = []
        for image_batch, label_batch in tqdm(dataloader, unit="batches", leave=False):
            image_batch = image_batch.to(device)
            label_batch = label_batch.to(device)
            loss = self.loss_f(self.forward(image_batch), label_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss.append(loss.detach().cpu().numpy())
        return np.mean(train_loss)

    def test_epoch(self, device: torch.device, dataloader: torch.utils.data.DataLoader):
        self.eval()
        test_loss = []
        test_acc = []
        with torch.no_grad():
            for image_batch, label_batch in dataloader:
                image_batch = image_batch.to(device)
                label_batch = label_batch.to(device)
                proba_batch = self.forward(image_batch)
                loss = self.loss_f(proba_batch, label_batch)
                pred_batch = torch.argmax(proba_batch, dim=-1)
                test_loss.append(loss.cpu().numpy())
                test_acc.append(torch.count_nonzero(pred_batch == label_batch).cpu().numpy()/len(image_batch))
        return np.mean(test_loss), np.mean(test_acc)

    def fit(self, device: torch.device, train_loader: torch.utils.data.DataLoader,
            test_loader: torch.utils.data.DataLoader, save_dir: pathlib.Path,
            n_epoch: int = 30, patience: int = 10) -> None:
        self.to(device)
        optim = torch.optim.Adam(self.parameters(), lr=1e-03, weight_decay=1e-05)
        waiting_epoch = 0
        best_test_loss = float("inf")
        for epoch in range(n_epoch):
            train_loss = self.train_epoch(device, train_loader, optim)
            test_loss, test_acc = self.test_epoch(device, test_loader)
            logging.info(f'Epoch {epoch + 1}/{n_epoch} \t '
                         f'Train loss {train_loss:.3g} \t Test loss {test_loss:.3g} \t Test accuracy {test_acc:.3g}')
            if test_loss >= best_test_loss:
                waiting_epoch += 1
                logging.info(f'No improvement over the best epoch \t Patience {waiting_epoch} / {patience}')
            else:
                logging.info(f'Saving the model in {save_dir}')
                self.cpu()
                self.save(save_dir)
                self.to(device)
                best_test_loss = test_loss.data
                waiting_epoch = 0
            if waiting_epoch == patience:
                logging.info(f'Early stopping activated')
                break

    def save(self, directory: pathlib.Path) -> None:
        """
        Save a model and corresponding metadata.
        Parameters
        ----------
        directory : pathlib.Path
            Path to the directory where to save the data.
        """
        model_name = self.name
        self.save_metadata(directory)
        path_to_model = directory / (model_name + ".pt")
        torch.save(self.state_dict(), path_to_model)

    def load_metadata(self, directory: pathlib.Path) -> dict:
        """Load the metadata of a training directory.
        Parameters
        ----------
        directory : pathlib.Path
            Path to folder where model is saved. For example './experiments/mnist'.
        """
        path_to_metadata = directory / (self.name + ".json")

        with open(path_to_metadata) as metadata_file:
            metadata = json.load(metadata_file)
        return metadata

    def save_metadata(self, directory: pathlib.Path, **kwargs) -> None:
        """Load the metadata of a training directory.
        Parameters
        ----------
        directory: string
            Path to folder where to save model. For example './experiments/mnist'.
        kwargs:
            Additional arguments to `json.dump`
        """
        path_to_metadata = directory / (self.name + ".json")
        metadata = {"latent_dim": self.latent_dim,
                    "name": self.name}
        with open(path_to_metadata, 'w') as f:
            json.dump(metadata, f, indent=4, sort_keys=True, **kwargs)


class VarEncoderMnist(nn.Module):
    def __init__(self, c: int = 64, latent_dims: int = 10):
        super(VarEncoderMnist, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=c, kernel_size=4, stride=2, padding=1)  # out: c x 14 x 14
        self.conv2 = nn.Conv2d(in_channels=c, out_channels=c * 2, kernel_size=4, stride=2, padding=1)  # out: c x 7 x 7
        self.fc_mu = nn.Linear(in_features=c * 2 * 7 * 7, out_features=latent_dims)
        self.fc_logvar = nn.Linear(in_features=c * 2 * 7 * 7, out_features=latent_dims)
        self.latent_dims = latent_dims

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(x.size(0), -1)  # flatten batch of multi-channel feature maps to a batch of feature vectors
        x_mu = self.fc_mu(x)
        x_logvar = self.fc_logvar(x)
        return x_mu, x_logvar

    def mu(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(x.size(0), -1)  # flatten batch of multi-channel feature maps to a batch of feature vectors
        x_mu = self.fc_mu(x)
        return x_mu


class VarDecoderMnist(nn.Module):
    def __init__(self, c: int = 64, latent_dims: int = 10):
        super(VarDecoderMnist, self).__init__()
        self.fc = nn.Linear(in_features=latent_dims, out_features=c * 2 * 7 * 7)
        self.conv2 = nn.ConvTranspose2d(in_channels=c * 2, out_channels=c, kernel_size=4, stride=2, padding=1)
        self.conv1 = nn.ConvTranspose2d(in_channels=c, out_channels=1, kernel_size=4, stride=2, padding=1)
        self.c = c

    def forward(self, x):
        x = self.fc(x)
        x = x.view(x.size(0), self.c * 2, 7,
                   7)  # unflatten batch of feature vectors to a batch of multi-channel feature maps
        x = F.relu(self.conv2(x))
        x = torch.sigmoid(
            self.conv1(x))  # last layer before output is sigmoid, since we are using BCE as reconstruction loss
        return x


class BetaVaeMnist(nn.Module):
    def __init__(self, latent_dims: int = 10, beta: int = 1):
        super(BetaVaeMnist, self).__init__()
        self.encoder = VarEncoderMnist(latent_dims=latent_dims)
        self.decoder = VarDecoderMnist(latent_dims=latent_dims)
        self.beta = beta

    def forward(self, x):
        latent_mu, latent_logvar = self.encoder(x)
        latent = self.latent_sample(latent_mu, latent_logvar)
        x_recon = self.decoder(latent)
        return x_recon, latent_mu, latent_logvar

    def latent_sample(self, mu, logvar):
        if self.training:
            # the reparameterization trick
            std = logvar.mul(0.5).exp_()
            eps = torch.empty_like(std).normal_()
            return eps.mul(std).add_(mu)
        else:
            return mu

    def loss(self, recon_x: torch.Tensor, x: torch.Tensor, mu: torch.Tensor, logvar: torch.Tensor,
             dataset_size: int) -> torch.Tensor:
        # recon_loss = F.mse_loss(recon_x, x)
        recon_loss = F.binary_cross_entropy(recon_x.view(-1, 784), x.view(-1, 784), reduction='mean')
        kld_weight = x.shape[0] / dataset_size
        kld_loss = torch.mean(-0.5 * torch.sum(1 + logvar - mu ** 2 - logvar.exp(), dim=1), dim=0)
        loss = recon_loss + self.beta * kld_weight * kld_loss

        return loss

    def train_epoch(self, device: torch.device, dataloader: torch.utils.data.DataLoader,
                    optimizer: torch.optim.Optimizer) -> np.ndarray:
        self.train()
        train_loss = []
        for image_batch, _ in dataloader:
            image_batch = image_batch.to(device)
            recon_batch, mu_batch, logvar_batch = self.forward(image_batch)
            loss = self.loss(recon_batch, image_batch, mu_batch, logvar_batch, dataset_size=len(dataloader.dataset))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss.append(loss.detach().cpu().numpy())
        return np.mean(train_loss)

    def test_epoch(self, device: torch.device, dataloader: torch.utils.data.DataLoader) -> np.ndarray:
        self.eval()
        test_loss = []
        with torch.no_grad():
            for image_batch, _ in dataloader:
                image_batch = image_batch.to(device)
                recon_batch, mu_batch, logvar_batch = self.forward(image_batch)
                loss = self.loss(recon_batch, image_batch, mu_batch, logvar_batch, dataset_size=len(dataloader.dataset))
                test_loss.append(loss.cpu().numpy())
        return np.mean(test_loss)

    def fit(self, device: torch.device, train_loader: torch.utils.data.DataLoader,
            test_loader: torch.utils.data.DataLoader, n_epoch: int = 30) -> None:
        self.to(device)
        optim = torch.optim.Adam(self.parameters(), lr=1e-03, weight_decay=1e-05)
        for epoch in range(n_epoch):
            train_loss = self.train_epoch(device, train_loader, optim)
            test_loss = self.test_epoch(device, test_loader)
            logging.info(f' Epoch {epoch + 1}/{n_epoch} \t '
                         f'Train loss {train_loss:.3g} \t Test loss {test_loss:.3g} \t ')


class BetaTcVaeMnist(nn.Module):
    def __init__(self, latent_dims: int = 10, beta: int = 1):
        super(BetaTcVaeMnist, self).__init__()
        self.encoder = VarEncoderMnist(latent_dims=latent_dims)
        self.decoder = VarDecoderMnist(latent_dims=latent_dims)
        self.beta = beta

    def forward(self, x):
        latent_mu, latent_logvar = self.encoder(x)
        latent = self.latent_sample(latent_mu, latent_logvar)
        x_recon = self.decoder(latent)
        return x_recon, latent_mu, latent_logvar, latent

    def latent_sample(self, mu, logvar):
        if self.training:
            # the reparameterization trick
            std = logvar.mul(0.5).exp_()
            eps = torch.empty_like(std).normal_()
            return eps.mul(std).add_(mu)
        else:
            return mu

    def loss(self, recon_x: torch.Tensor, x: torch.Tensor, mu: torch.Tensor, logvar: torch.Tensor,
             z: torch.Tensor, dataset_size: int) -> torch.Tensor:

        recon_loss = F.binary_cross_entropy(recon_x.view(-1, 784), x.view(-1, 784), reduction='mean')
        # recon_loss = F.mse_loss(recon_x, x, reduction='sum')
        log_q_zx = log_density_gaussian(z, mu, logvar).sum(dim=1)

        zeros = torch.zeros_like(z)
        log_p_z = log_density_gaussian(z, zeros, zeros).sum(dim=1)

        batch_size, latent_dim = z.shape
        mat_log_q_z = log_density_gaussian(z.view(batch_size, 1, latent_dim),
                                           mu.view(1, batch_size, latent_dim),
                                           logvar.view(1, batch_size, latent_dim))

        strat_weight = (dataset_size - batch_size + 1) / (dataset_size * (batch_size - 1))
        importance_weights = torch.Tensor(batch_size, batch_size).fill_(1 / (batch_size - 1)).to(x.device)
        importance_weights.view(-1)[::batch_size] = 1 / dataset_size
        importance_weights.view(-1)[1::batch_size] = strat_weight
        importance_weights[batch_size - 2, 0] = strat_weight
        log_importance_weights = importance_weights.log()

        mat_log_q_z += log_importance_weights.view(batch_size, batch_size, 1)

        log_q_z = torch.logsumexp(mat_log_q_z.sum(2), dim=1, keepdim=False)
        log_prod_q_z = torch.logsumexp(mat_log_q_z, dim=1, keepdim=False).sum(1)

        mi_loss = (log_q_zx - log_q_z).mean()
        tc_loss = (log_q_z - log_prod_q_z).mean()
        kld_loss = (log_prod_q_z - log_p_z).mean()

        loss = recon_loss / batch_size + mi_loss + self.beta * tc_loss + kld_loss
        return loss

    def train_epoch(self, device: torch.device, dataloader: torch.utils.data.DataLoader,
                    optimizer: torch.optim.Optimizer) -> np.ndarray:
        self.train()
        train_loss = []
        for image_batch, _ in dataloader:
            image_batch = image_batch.to(device)
            recon_batch, mu_batch, logvar_batch, latent_batch = self.forward(image_batch)
            loss = self.loss(recon_batch, image_batch, mu_batch, logvar_batch,
                             latent_batch, len(dataloader.dataset))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss.append(loss.detach().cpu().numpy())
        return np.mean(train_loss)

    def test_epoch(self, device: torch.device, dataloader: torch.utils.data.DataLoader):
        self.eval()
        test_loss = []
        with torch.no_grad():
            for image_batch, _ in dataloader:
                image_batch = image_batch.to(device)
                recon_batch, mu_batch, logvar_batch, latent_batch = self.forward(image_batch)
                loss = self.loss(recon_batch, image_batch, mu_batch,
                                 logvar_batch, latent_batch, len(dataloader.dataset))
                test_loss.append(loss.cpu().numpy())
        return np.mean(test_loss)

    def fit(self, device: torch.device, train_loader: torch.utils.data.DataLoader,
            test_loader: torch.utils.data.DataLoader, n_epoch: int = 30) -> None:
        self.to(device)
        optim = torch.optim.Adam(self.parameters(), lr=1e-03, weight_decay=1e-05)
        for epoch in range(n_epoch):
            train_loss = self.train_epoch(device, train_loader, optim)
            test_loss = self.test_epoch(device, test_loader)
            logging.info(f'Epoch {epoch + 1}/{n_epoch} \t '
                         f'Train loss {train_loss:.3g} \t Test loss {test_loss:.3g} \t ')


class EncoderBurgess(nn.Module):
    def __init__(self, img_size,
                 latent_dim=10):
        r"""Encoder of the model proposed in [1].
        Parameters
        ----------
        img_size : tuple of ints
            Size of images. E.g. (1, 32, 32) or (3, 64, 64).
        latent_dim : int
            Dimensionality of latent output.
        Model Architecture (transposed for decoder)
        ------------
        - 4 convolutional layers (each with 32 channels), (4 x 4 kernel), (stride of 2)
        - 2 fully connected layers (each of 256 units)
        - Latent distribution:
            - 1 fully connected layer of 20 units (log variance and mean for 10 Gaussians)
        References:
            [1] Burgess, Christopher P., et al. "Understanding disentangling in
            $\beta$-VAE." arXiv preprint arXiv:1804.03599 (2018).
        """
        super(EncoderBurgess, self).__init__()

        # Layer parameters
        hid_channels = 32
        kernel_size = 4
        hidden_dim = 256
        self.latent_dim = latent_dim
        self.img_size = img_size
        # Shape required to start transpose convs
        self.reshape = (hid_channels, kernel_size, kernel_size)
        n_chan = self.img_size[0]

        # Convolutional layers
        cnn_kwargs = dict(stride=2, padding=1)
        self.conv1 = nn.Conv2d(n_chan, hid_channels, kernel_size, **cnn_kwargs)
        self.conv2 = nn.Conv2d(hid_channels, hid_channels, kernel_size, **cnn_kwargs)
        self.conv3 = nn.Conv2d(hid_channels, hid_channels, kernel_size, **cnn_kwargs)

        # If input image is 64x64 do fourth convolution
        if self.img_size[1] == self.img_size[2] == 64:
            self.conv_64 = nn.Conv2d(hid_channels, hid_channels, kernel_size, **cnn_kwargs)

        # Fully connected layers
        self.lin1 = nn.Linear(np.product(self.reshape), hidden_dim)
        self.lin2 = nn.Linear(hidden_dim, hidden_dim)

        # Fully connected layers for mean and variance
        self.mu_logvar_gen = nn.Linear(hidden_dim, self.latent_dim * 2)

    def forward(self, x):
        batch_size = x.size(0)

        # Convolutional layers with ReLu activations
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        if self.img_size[1] == self.img_size[2] == 64:
            x = torch.relu(self.conv_64(x))

        # Fully connected layers with ReLu activations
        x = x.view((batch_size, -1))
        x = torch.relu(self.lin1(x))
        x = torch.relu(self.lin2(x))

        # Fully connected layer for log variance and mean
        # Log std-dev in paper (bear in mind)
        mu_logvar = self.mu_logvar_gen(x)
        mu, logvar = mu_logvar.view(-1, self.latent_dim, 2).unbind(-1)

        return mu, logvar

    def mu(self, x):
        return self.forward(x)[0]


class DecoderBurgess(nn.Module):
    def __init__(self, img_size,
                 latent_dim=10):
        r"""Decoder of the model proposed in [1].
        Parameters
        ----------
        img_size : tuple of ints
            Size of images. E.g. (1, 32, 32) or (3, 64, 64).
        latent_dim : int
            Dimensionality of latent output.
        Model Architecture (transposed for decoder)
        ------------
        - 4 convolutional layers (each with 32 channels), (4 x 4 kernel), (stride of 2)
        - 2 fully connected layers (each of 256 units)
        - Latent distribution:
            - 1 fully connected layer of 20 units (log variance and mean for 10 Gaussians)
        References:
            [1] Burgess, Christopher P., et al. "Understanding disentangling in
            $\beta$-VAE." arXiv preprint arXiv:1804.03599 (2018).
        """
        super(DecoderBurgess, self).__init__()

        # Layer parameters
        hid_channels = 32
        kernel_size = 4
        hidden_dim = 256
        self.img_size = img_size
        # Shape required to start transpose convs
        self.reshape = (hid_channels, kernel_size, kernel_size)
        n_chan = self.img_size[0]
        self.img_size = img_size

        # Fully connected layers
        self.lin1 = nn.Linear(latent_dim, hidden_dim)
        self.lin2 = nn.Linear(hidden_dim, hidden_dim)
        self.lin3 = nn.Linear(hidden_dim, np.product(self.reshape))

        # Convolutional layers
        cnn_kwargs = dict(stride=2, padding=1)
        # If input image is 64x64 do fourth convolution
        if self.img_size[1] == self.img_size[2] == 64:
            self.convT_64 = nn.ConvTranspose2d(hid_channels, hid_channels, kernel_size, **cnn_kwargs)

        self.convT1 = nn.ConvTranspose2d(hid_channels, hid_channels, kernel_size, **cnn_kwargs)
        self.convT2 = nn.ConvTranspose2d(hid_channels, hid_channels, kernel_size, **cnn_kwargs)
        self.convT3 = nn.ConvTranspose2d(hid_channels, n_chan, kernel_size, **cnn_kwargs)

    def forward(self, z):
        batch_size = z.size(0)

        # Fully connected layers with ReLu activations
        x = torch.relu(self.lin1(z))
        x = torch.relu(self.lin2(x))
        x = torch.relu(self.lin3(x))
        x = x.view(batch_size, *self.reshape)

        # Convolutional layers with ReLu activations
        if self.img_size[1] == self.img_size[2] == 64:
            x = torch.relu(self.convT_64(x))
        x = torch.relu(self.convT1(x))
        x = torch.relu(self.convT2(x))
        # Sigmoid activation for final conv layer
        x = torch.sigmoid(self.convT3(x))

        return x


class VAE(nn.Module):
    def __init__(self, img_size: tuple, encoder: EncoderBurgess, decoder: DecoderBurgess,
                 latent_dim: int, loss_f: BaseVAELoss, name: str = "model"):
        """
        Class which defines model and forward pass.
        Parameters
        ----------
        img_size : tuple of ints
            Size of images. E.g. (1, 32, 32) or (3, 64, 64).
        """
        super(VAE, self).__init__()
        self.latent_dim = latent_dim
        self.img_size = img_size
        self.num_pixels = self.img_size[1] * self.img_size[2]
        self.encoder = encoder
        self.decoder = decoder
        self.loss_f = loss_f
        self.name = name

    def reparameterize(self, mean, logvar):
        """
        Samples from a normal distribution using the reparameterization trick.
        Parameters
        ----------
        mean : torch.Tensor
            Mean of the normal distribution. Shape (batch_size, latent_dim)
        logvar : torch.Tensor
            Diagonal log variance of the normal distribution. Shape (batch_size,
            latent_dim)
        """
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mean + std * eps
        else:
            # Reconstruction mode
            return mean

    def forward(self, x):
        """
        Forward pass of model.
        Parameters
        ----------
        x : torch.Tensor
            Batch of data. Shape (batch_size, n_chan, height, width)
        """
        latent_dist = self.encoder(x)
        latent_sample = self.reparameterize(*latent_dist)
        reconstruct = self.decoder(latent_sample)
        return reconstruct, latent_dist, latent_sample

    def sample_latent(self, x):
        """
        Returns a sample from the latent distribution.
        Parameters
        ----------
        x : torch.Tensor
            Batch of data. Shape (batch_size, n_chan, height, width)
        """
        latent_dist = self.encoder(x)
        latent_sample = self.reparameterize(*latent_dist)
        return latent_sample

    def train_epoch(self, device: torch.device, dataloader: torch.utils.data.DataLoader,
                    optimizer: torch.optim.Optimizer) -> np.ndarray:
        self.train()
        train_loss = []
        for image_batch, _ in tqdm(dataloader, unit="batches", leave=False):
            image_batch = image_batch.to(device)
            recon_batch, latent_dist, latent_batch = self.forward(image_batch)
            loss = self.loss_f(image_batch, recon_batch, latent_dist,
                               is_train=True, storer=None, latent_sample=latent_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss.append(loss.detach().cpu().numpy())
        return np.mean(train_loss)

    def test_epoch(self, device: torch.device, dataloader: torch.utils.data.DataLoader):
        self.eval()
        test_loss = []
        with torch.no_grad():
            for image_batch, _ in dataloader:
                image_batch = image_batch.to(device)
                recon_batch, latent_dist, latent_batch = self.forward(image_batch)
                loss = self.loss_f(image_batch, recon_batch, latent_dist,
                                   is_train=True, storer=None, latent_sample=latent_batch)
                test_loss.append(loss.cpu().numpy())
        return np.mean(test_loss)

    def fit(self, device: torch.device, train_loader: torch.utils.data.DataLoader,
            test_loader: torch.utils.data.DataLoader, save_dir: pathlib.Path,
            n_epoch: int = 30, patience: int = 10) -> None:
        self.to(device)
        optim = torch.optim.Adam(self.parameters(), lr=1e-03, weight_decay=1e-05)
        waiting_epoch = 0
        best_test_loss = float("inf")
        for epoch in range(n_epoch):
            train_loss = self.train_epoch(device, train_loader, optim)
            test_loss = self.test_epoch(device, test_loader)
            logging.info(f'Epoch {epoch + 1}/{n_epoch} \t '
                         f'Train loss {train_loss:.3g} \t Test loss {test_loss:.3g} \t ')
            if test_loss >= best_test_loss:
                waiting_epoch += 1
                logging.info(f'No improvement over the best epoch \t Patience {waiting_epoch} / {patience}')
            else:
                logging.info(f'Saving the model in {save_dir}')
                self.cpu()
                self.save(save_dir)
                self.to(device)
                best_test_loss = test_loss.data
                waiting_epoch = 0
            if waiting_epoch == patience:
                logging.info(f'Early stopping activated')
                break

    def save(self, directory: pathlib.Path) -> None:
        """
        Save a model and corresponding metadata.
        Parameters
        ----------
        directory : pathlib.Path
            Path to the directory where to save the data.
        """
        model_name = self.name
        self.save_metadata(directory)
        path_to_model = directory / (model_name + ".pt")
        torch.save(self.state_dict(), path_to_model)


    def load_metadata(self, directory: pathlib.Path) -> dict:
        """Load the metadata of a training directory.
        Parameters
        ----------
        directory : pathlib.Path
            Path to folder where model is saved. For example './experiments/mnist'.
        """
        path_to_metadata = directory / (self.name + ".json")

        with open(path_to_metadata) as metadata_file:
            metadata = json.load(metadata_file)
        return metadata

    def save_metadata(self, directory: pathlib.Path, **kwargs) -> None:
        """Load the metadata of a training directory.
        Parameters
        ----------
        directory: string
            Path to folder where to save model. For example './experiments/mnist'.
        kwargs:
            Additional arguments to `json.dump`
        """
        path_to_metadata = directory / (self.name + ".json")
        metadata = {"latent_dim": self.latent_dim,
                    "img_size": self.img_size,
                    "num_pixels": self.num_pixels,
                    "name": self.name}
        with open(path_to_metadata, 'w') as f:
            json.dump(metadata, f, indent=4, sort_keys=True, **kwargs)


class ClassifierLatent(nn.Module):
    def __init__(self, latent_dims: int):
        super().__init__()
        self.encoder_cnn = nn.Sequential(
            nn.Conv2d(1, 8, 3, stride=2, padding=1),
            nn.ReLU(True),
            nn.Conv2d(8, 16, 3, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            nn.Conv2d(16, 32, 3, stride=2, padding=0),
            nn.ReLU(True)
        )
        self.flatten = nn.Flatten(start_dim=1)
        self.encoder_lin = nn.Sequential(
            nn.Linear(3 * 3 * 32, 128),
            nn.ReLU(True),
            nn.Linear(128, latent_dims)
        )

    def forward(self, x):
        x = self.encoder_cnn(x)
        x = self.flatten(x)
        x = self.encoder_lin(x)
        x = nn.Softmax(x)
        return x


class SimCLR(nn.Module):
    def __init__(self, base_encoder, projection_dim=128):
        super().__init__()
        self.encoder = base_encoder(pretrained=True)  # load model from torchvision.models without pretrained weights.
        self.feature_dim = self.encoder.fc.in_features

        # Customize for CIFAR10. Replace conv 7x7 with conv 3x3, and remove first max pooling.
        # See Section B.9 of SimCLR paper.
        self.encoder.conv1 = nn.Conv2d(3, 64, 3, 1, 1, bias=False)
        self.encoder.maxpool = nn.Identity()
        self.encoder.fc = nn.Identity()  # remove final fully connected layer.

        # Add MLP projection.
        self.projection_dim = projection_dim
        self.projector = nn.Sequential(nn.Linear(self.feature_dim, 2048),
                                       nn.ReLU(),
                                       nn.Linear(2048, projection_dim))

    def forward(self, x):
        feature = self.encoder(x)
        projection = self.projector(feature)
        return feature, projection

    @staticmethod
    def nt_xent(x, t=0.5):
        x = F.normalize(x, dim=1)
        x_scores = (x @ x.t()).clamp(min=1e-7)  # normalized cosine similarity scores
        x_scale = x_scores / t  # scale with temperature

        # (2N-1)-way softmax without the score of i-th entry itself.
        # Set the diagonals to be large negative values, which become zeros after softmax.
        x_scale = x_scale - torch.eye(x_scale.size(0)).to(x_scale.device) * 1e5

        # targets 2N elements.
        targets = torch.arange(x.size()[0])
        targets[::2] += 1  # target of 2k element is 2k+1
        targets[1::2] -= 1  # target of 2k+1 element is 2k
        return F.cross_entropy(x_scale, targets.long().to(x_scale.device))

    @staticmethod
    def get_lr(step, total_steps, lr_max, lr_min):
        """Compute learning rate according to cosine annealing schedule."""
        return lr_min + (lr_max - lr_min) * 0.5 * (1 + np.cos(step / total_steps * np.pi))

    # color distortion composed by color jittering and color dropping.
    @staticmethod
    def get_color_distortion(s=0.5):  # 0.5 for CIFAR10 by default
        # s is the strength of color distortion
        color_jitter = transforms.ColorJitter(0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s)
        rnd_color_jitter = transforms.RandomApply([color_jitter], p=0.8)
        rnd_gray = transforms.RandomGrayscale(p=0.2)
        color_distort = transforms.Compose([rnd_color_jitter, rnd_gray])
        return color_distort

    def fit(self, args: DictConfig, device: torch.device) -> None:
        logger = logging.getLogger(__name__)

        train_transform = transforms.Compose([transforms.RandomResizedCrop(32),
                                              transforms.RandomHorizontalFlip(p=0.5),
                                              self.get_color_distortion(s=0.5),
                                              transforms.ToTensor()])
        data_dir = hydra.utils.to_absolute_path(args.data_dir)  # get absolute path of data dir
        train_set = CIFAR10Pair(root=data_dir,
                                train=True,
                                transform=train_transform,
                                download=True)

        train_loader = DataLoader(train_set,
                                  batch_size=args.batch_size,
                                  shuffle=True,
                                  num_workers=args.workers,
                                  drop_last=True)

        optimizer = torch.optim.SGD(
            self.parameters(),
            args.learning_rate,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
            nesterov=True)

        # cosine annealing lr
        scheduler = LambdaLR(
            optimizer,
            lr_lambda=lambda step: self.get_lr(  # pylint: disable=g-long-lambda
                step,
                args.epochs * len(train_loader),
                args.learning_rate,  # lr_lambda computes multiplicative factor
                1e-3))

        # SimCLR training
        self.train()
        for epoch in range(1, args.epochs + 1):
            loss_meter = AverageMeter("SimCLR_loss")
            train_bar = tqdm(train_loader)
            for x, y in train_bar:
                sizes = x.size()
                x = x.view(sizes[0] * 2, sizes[2], sizes[3], sizes[4]).to(device)
                optimizer.zero_grad()
                feature, rep = self.forward(x)
                loss = self.nt_xent(rep, args.temperature)
                loss.backward()
                optimizer.step()
                scheduler.step()
                loss_meter.update(loss.item(), x.size(0))
                train_bar.set_description("Train epoch {}, SimCLR loss: {:.4f}".format(epoch, loss_meter.avg))

            # save checkpoint very log_interval epochs
            if epoch >= args.log_interval and epoch % args.log_interval == 0:
                logger.info("==> Save checkpoint. Train epoch {}, SimCLR loss: {:.4f}".format(epoch, loss_meter.avg))
                torch.save(self.state_dict(), Path.cwd()/f'models/simclr_{args.backbone}_epoch{epoch}.pt')


def log_density_gaussian(x: torch.Tensor, mu: torch.Tensor, logvar: torch.Tensor):
    """
    Computes the log pdf of the Gaussian with parameters mu and logvar at x
    :param x: (Tensor) Point at whichGaussian PDF is to be evaluated
    :param mu: (Tensor) Mean of the Gaussian distribution
    :param logvar: (Tensor) Log variance of the Gaussian distribution
    :return:
    """
    norm = - 0.5 * (math.log(2 * math.pi) + logvar)
    log_density = norm - 0.5 * ((x - mu) ** 2 * torch.exp(-logvar))
    return log_density



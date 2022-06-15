import logging
import pathlib

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

"""
Models inspired by:
 https://curiousily.com/posts/time-series-anomaly-detection-using-lstm-autoencoder-with-pytorch-in-python/
"""


class Encoder(nn.Module):
    def __init__(self, seq_len: int, n_features: int = 1, embedding_dim: int = 64):
        super(Encoder, self).__init__()
        self.seq_len, self.n_features = seq_len, n_features
        self.embedding_dim, self.hidden_dim = embedding_dim, 2 * embedding_dim
        self.rnn1 = nn.LSTM(
            input_size=n_features,
            hidden_size=self.hidden_dim,
            num_layers=1,
            batch_first=True,
        )
        self.rnn2 = nn.LSTM(
            input_size=self.hidden_dim,
            hidden_size=embedding_dim,
            num_layers=1,
            batch_first=True,
        )

    def forward(self, x):
        x, (_, _) = self.rnn1(x)
        x, (hidden_n, _) = self.rnn2(x)
        return hidden_n.reshape((-1, self.embedding_dim))


class Decoder(nn.Module):
    def __init__(self, seq_len: int, n_features: int = 1, input_dim: int = 64):
        super(Decoder, self).__init__()
        self.seq_len, self.input_dim = seq_len, input_dim
        self.hidden_dim, self.n_features = 2 * input_dim, n_features
        self.rnn1 = nn.LSTM(
            input_size=input_dim, hidden_size=input_dim, num_layers=1, batch_first=True
        )
        self.rnn2 = nn.LSTM(
            input_size=input_dim,
            hidden_size=self.hidden_dim,
            num_layers=1,
            batch_first=True,
        )
        self.output_layer = nn.Linear(self.hidden_dim, n_features)

    def forward(self, x):
        x = x.unsqueeze(1)
        x = x.repeat(1, self.seq_len, self.n_features)
        x = x.reshape((-1, self.seq_len, self.input_dim))
        x, (hidden_n, cell_n) = self.rnn1(x)
        x, (hidden_n, cell_n) = self.rnn2(x)
        x = x.reshape((-1, self.seq_len, self.hidden_dim))
        return self.output_layer(x)


class RecurrentAutoencoder(nn.Module):
    def __init__(
        self,
        seq_len: int,
        n_features: int,
        embedding_dim: int = 64,
        name: str = "model",
        loss_f: callable = nn.L1Loss(reduction="mean"),
    ):
        super(RecurrentAutoencoder, self).__init__()
        self.encoder = Encoder(seq_len, n_features, embedding_dim)
        self.decoder = Decoder(seq_len, n_features, embedding_dim)
        self.lr = None
        self.checkpoints_files = []
        self.name = name
        self.loss_f = loss_f

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

    def train_epoch(
        self,
        device: torch.device,
        dataloader: torch.utils.data.DataLoader,
        optimizer: torch.optim.Optimizer,
    ) -> np.ndarray:
        self.train()
        train_loss = []
        for input_batch, _ in tqdm(dataloader, unit="batch", leave=False):
            input_batch = input_batch.to(device)
            recon_batch = self.forward(input_batch)
            loss = self.loss_f(input_batch, recon_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss.append(loss.detach().cpu().numpy())
        return np.mean(train_loss)

    def test_epoch(self, device: torch.device, dataloader: torch.utils.data.DataLoader):
        self.eval()
        test_loss = []
        with torch.no_grad():
            for input_batch, _ in dataloader:
                input_batch = input_batch.to(device)
                recon_batch = self.forward(input_batch)
                loss = self.loss_f(input_batch, recon_batch)
                test_loss.append(loss.cpu().numpy())
        return np.mean(test_loss)

    def fit(
        self,
        device: torch.device,
        train_loader: torch.utils.data.DataLoader,
        test_loader: torch.utils.data.DataLoader,
        save_dir: pathlib.Path,
        n_epoch: int = 30,
        patience: int = 10,
        checkpoint_interval: int = -1,
    ) -> None:
        self.to(device)
        self.lr = 1e-03
        optim = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=1e-05)
        waiting_epoch = 0
        best_test_loss = float("inf")
        for epoch in range(n_epoch):
            train_loss = self.train_epoch(device, train_loader, optim)
            test_loss = self.test_epoch(device, test_loader)
            logging.info(
                f"Epoch {epoch + 1}/{n_epoch} \t "
                f"Train loss {train_loss:.3g} \t Test loss {test_loss:.3g} \t "
            )
            if test_loss >= best_test_loss:
                waiting_epoch += 1
                logging.info(
                    f"No improvement over the best epoch \t Patience {waiting_epoch} / {patience}"
                )
            else:
                logging.info(f"Saving the model in {save_dir}")
                self.cpu()
                self.save(save_dir)
                self.to(device)
                best_test_loss = test_loss.data
                waiting_epoch = 0
            if checkpoint_interval > 0 and epoch % checkpoint_interval == 0:
                n_checkpoint = 1 + epoch // checkpoint_interval
                logging.info(f"Saving checkpoint {n_checkpoint} in {save_dir}")
                path_to_checkpoint = (
                    save_dir / f"{self.name}_checkpoint{n_checkpoint}.pt"
                )
                torch.save(self.state_dict(), path_to_checkpoint)
                self.checkpoints_files.append(path_to_checkpoint)
            if waiting_epoch == patience:
                logging.info("Early stopping activated")
                break

    def save(self, directory: pathlib.Path) -> None:
        """
        Save a model and corresponding metadata.
        Parameters:
        ----------
        directory : pathlib.Path
            Path to the directory where to save the data.
        """
        model_name = self.name
        path_to_model = directory / (model_name + ".pt")
        torch.save(self.state_dict(), path_to_model)


class EncoderCNN(nn.Module):
    def __init__(self, encoded_space_dim):
        super().__init__()
        self.encoder_cnn = nn.Sequential(
            nn.Conv1d(1, 8, 3, stride=2, padding=1),
            nn.ReLU(True),
            nn.Conv1d(8, 16, 3, stride=2, padding=1),
            nn.BatchNorm1d(16),
            nn.ReLU(True),
            nn.Conv1d(16, 32, 3, stride=2, padding=0),
            nn.ReLU(True),
        )
        self.flatten = nn.Flatten(start_dim=1)
        self.encoder_lin = nn.Sequential(
            nn.Linear(544, 128), nn.ReLU(True), nn.Linear(128, encoded_space_dim)
        )
        self.encoded_space_dim = encoded_space_dim

    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.encoder_cnn(x)
        x = self.flatten(x)
        x = self.encoder_lin(x)
        return x


class DecoderCNN(nn.Module):
    def __init__(self, encoded_space_dim):
        super().__init__()
        self.decoder_lin = nn.Sequential(
            nn.Linear(encoded_space_dim, 128), nn.ReLU(True), nn.Linear(128, 544)
        )
        self.unflatten = nn.Unflatten(dim=1, unflattened_size=(32, 17))
        self.decoder_conv = nn.Sequential(
            nn.ConvTranspose1d(32, 16, 3, stride=2, output_padding=0),
            nn.BatchNorm1d(16),
            nn.ReLU(True),
            nn.ConvTranspose1d(16, 8, 3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm1d(8),
            nn.ReLU(True),
            nn.ConvTranspose1d(8, 1, 3, stride=2, padding=1, output_padding=1),
        )

    def forward(self, x):
        x = self.decoder_lin(x)
        x = self.unflatten(x)
        x = self.decoder_conv(x)
        return x


class AutoencoderCNN(nn.Module):
    def __init__(
        self,
        embedding_dim: int = 64,
        name: str = "model",
        loss_f: callable = nn.L1Loss(reduction="mean"),
    ):
        super(AutoencoderCNN, self).__init__()
        self.encoder = EncoderCNN(embedding_dim)
        self.decoder = DecoderCNN(embedding_dim)
        self.lr = None
        self.checkpoints_files = []
        self.name = name
        self.loss_f = loss_f

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

    def train_epoch(
        self,
        device: torch.device,
        dataloader: torch.utils.data.DataLoader,
        optimizer: torch.optim.Optimizer,
    ) -> np.ndarray:
        self.train()
        train_loss = []
        for input_batch, _ in tqdm(dataloader, unit="batch", leave=False):
            input_batch = input_batch.to(device)
            recon_batch = self.forward(input_batch)
            loss = self.loss_f(input_batch, recon_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss.append(loss.detach().cpu().numpy())
        return np.mean(train_loss)

    def test_epoch(self, device: torch.device, dataloader: torch.utils.data.DataLoader):
        self.eval()
        test_loss = []
        with torch.no_grad():
            for input_batch, _ in dataloader:
                input_batch = input_batch.to(device)
                recon_batch = self.forward(input_batch)
                loss = self.loss_f(input_batch, recon_batch)
                test_loss.append(loss.cpu().numpy())
        return np.mean(test_loss)

    def fit(
        self,
        device: torch.device,
        train_loader: torch.utils.data.DataLoader,
        test_loader: torch.utils.data.DataLoader,
        save_dir: pathlib.Path,
        n_epoch: int = 30,
        patience: int = 10,
        checkpoint_interval: int = -1,
    ) -> None:
        self.to(device)
        self.lr = 1e-03
        optim = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=1e-05)
        waiting_epoch = 0
        best_test_loss = float("inf")
        for epoch in range(n_epoch):
            train_loss = self.train_epoch(device, train_loader, optim)
            test_loss = self.test_epoch(device, test_loader)
            logging.info(
                f"Epoch {epoch + 1}/{n_epoch} \t "
                f"Train loss {train_loss:.3g} \t Test loss {test_loss:.3g} \t "
            )
            if test_loss >= best_test_loss:
                waiting_epoch += 1
                logging.info(
                    f"No improvement over the best epoch \t Patience {waiting_epoch} / {patience}"
                )
            else:
                logging.info(f"Saving the model in {save_dir}")
                self.cpu()
                self.save(save_dir)
                self.to(device)
                best_test_loss = test_loss.data
                waiting_epoch = 0
            if checkpoint_interval > 0 and epoch % checkpoint_interval == 0:
                n_checkpoint = 1 + epoch // checkpoint_interval
                logging.info(f"Saving checkpoint {n_checkpoint} in {save_dir}")
                path_to_checkpoint = (
                    save_dir / f"{self.name}_checkpoint{n_checkpoint}.pt"
                )
                torch.save(self.state_dict(), path_to_checkpoint)
                self.checkpoints_files.append(path_to_checkpoint)
            if waiting_epoch == patience:
                logging.info("Early stopping activated")
                break

    def save(self, directory: pathlib.Path) -> None:
        """
        Save a model and corresponding metadata.
        Parameters:
        ----------
        directory : pathlib.Path
            Path to the directory where to save the data.
        """
        model_name = self.name
        path_to_model = directory / (model_name + ".pt")
        torch.save(self.state_dict(), path_to_model)

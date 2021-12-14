from torch import nn
import torch.nn.functional as F
import torch
import numpy as np
import math

'''
 These models are adapted from 
 https://medium.com/dataseries/convolutional-autoencoder-in-pytorch-on-mnist-dataset-d65145c132ac
 https://github.com/smartgeometry-ucl/dl4g
 https://github.com/AntixK/PyTorch-VAE
'''


class EncoderMnist(nn.Module):
    def __init__(self, encoded_space_dim, fc2_input_dim):
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
    def __init__(self, encoded_space_dim, fc2_input_dim):
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


class ClassifierMnist(nn.Module):
    def __init__(self, encoder: EncoderMnist):
        super().__init__()
        self.encoder_cnn = encoder.encoder_cnn
        self.flatten = nn.Flatten(start_dim=1)
        self.encoder_lin = encoder.encoder_lin
        self.lin_output = nn.Sequential(
            nn.Linear(encoder.encoded_space_dim, 10),
            nn.Softmax()
        )
        self.encoded_space_dim = encoder.encoded_space_dim

    def forward(self, x):
        x = self.encoder_cnn(x)
        x = self.flatten(x)
        x = self.encoder_lin(x)
        x = self.lin_output(x)
        return x


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


class VariationalAutoencoderMnist(nn.Module):
    def __init__(self, latent_dims: int = 10):
        super(VariationalAutoencoderMnist, self).__init__()
        self.encoder = VarEncoderMnist(latent_dims=latent_dims)
        self.decoder = VarDecoderMnist(latent_dims=latent_dims)

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


def beta_vae_loss(recon_x: torch.Tensor, x: torch.Tensor, mu: torch.Tensor, logvar: torch.Tensor,
                  beta: float, dataset_size: int) -> torch.Tensor:

    #recon_loss = F.binary_cross_entropy(recon_x.view(-1, 784), x.view(-1, 784), reduction='sum')
    kld_weight = x.shape[0] / dataset_size
    recon_loss = F.mse_loss(recon_x, x)
    kld_loss = torch.mean(-0.5 * torch.sum(1 + logvar - mu ** 2 - logvar.exp(), dim = 1), dim = 0)
    loss = recon_loss + beta * kld_weight * kld_loss

    return loss


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


def beta_tcvae_loss_function(recon_x: torch.Tensor, x: torch.Tensor, mu: torch.Tensor, logvar: torch.Tensor,
                             z: torch.Tensor, beta: float, dataset_size: int) -> torch.Tensor:

    #recon_loss = F.binary_cross_entropy(recon_x.view(-1, 784), x.view(-1, 784), reduction='sum')
    recon_loss = F.mse_loss(recon_x, x, reduction='sum')
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

    loss = recon_loss / batch_size + mi_loss + beta * tc_loss + kld_loss
    return loss


def train_denoiser_epoch(encoder: EncoderMnist, decoder: DecoderMnist, device: torch.device,
                         dataloader: torch.utils.data.DataLoader, loss_fn: callable, optimizer: torch.optim.Optimizer,
                         noise_factor: float = 0.3):
    # Set train mode for both the encoder and the decoder
    encoder.train()
    decoder.train()
    train_loss = []
    # Iterate the dataloader (we do not need the label values, this is unsupervised learning)
    for image_batch, _ in dataloader:  # with "_" we just ignore the labels (the second element of the dataloader tuple)
        # Move tensor to the proper device
        image_noisy = image_batch + noise_factor * torch.randn(image_batch.shape)
        image_noisy = image_noisy.to(device)
        # Encode data
        encoded_data = encoder(image_noisy)
        # Decode data
        decoded_data = decoder(encoded_data)
        # Evaluate loss
        loss = loss_fn(decoded_data, image_batch.to(device))
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss.append(loss.detach().cpu().numpy())
    return np.mean(train_loss)


def test_denoiser_epoch(encoder, decoder, device, dataloader, loss_fn):
    # Set evaluation mode for encoder and decoder
    encoder.eval()
    decoder.eval()
    with torch.no_grad():  # No need to track the gradients
        # Define the lists to store the outputs for each batch
        conc_out = []
        conc_label = []
        for image_batch, _ in dataloader:
            # Move tensor to the proper device
            image_batch = image_batch.to(device)
            # Encode data
            encoded_data = encoder(image_batch)
            # Decode data
            decoded_data = decoder(encoded_data)
            conc_out.append(decoded_data.cpu())
            conc_label.append(image_batch.cpu())
        # Create a single tensor with all the values in the lists
        conc_out = torch.cat(conc_out)
        conc_label = torch.cat(conc_label)
        # Evaluate global loss
        val_loss = loss_fn(conc_out, conc_label)
    return val_loss.data


def train_classifier_epoch(classifier: ClassifierMnist, device: torch.device, dataloader: torch.utils.data.DataLoader,
                           loss_fn: callable, optimizer: torch.optim.Optimizer):
    # Set train mode for both the encoder and the decoder
    classifier.train()
    train_loss = []
    # Iterate the dataloader (we do not need the label values, this is unsupervised learning)
    for image_batch, label_batch in dataloader:
        # Move tensor to the proper device
        image_batch = image_batch.to(device)
        # Predict Probabilities
        proba_batch = classifier(image_batch)
        # Evaluate loss
        loss = loss_fn(proba_batch, label_batch.to(device))
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss.append(loss.detach().cpu().numpy())
    return np.mean(train_loss)


def test_classifier_epoch(classifier: ClassifierMnist, device: torch.device, dataloader: torch.utils.data.DataLoader,
                          loss_fn: callable):
    # Set evaluation mode for encoder and decoder
    classifier.eval()
    with torch.no_grad():  # No need to track the gradients
        # Define the lists to store the outputs for each batch
        conc_out = []
        conc_label = []
        for image_batch, label_batch in dataloader:
            # Move tensor to the proper device
            image_batch = image_batch.to(device)
            # Predict probabilites
            proba_batch = classifier(image_batch)
            conc_out.append(proba_batch.cpu())
            conc_label.append(label_batch.cpu())
        # Create a single tensor with all the values in the lists
        conc_out = torch.cat(conc_out)
        conc_label = torch.cat(conc_label)
        # Evaluate global loss
        val_loss = loss_fn(conc_out, conc_label)
    return val_loss.data


def train_beta_vae_epoch(vae: VariationalAutoencoderMnist, device: torch.device, dataloader: torch.utils.data.DataLoader,
                         optimizer: torch.optim.Optimizer, beta: float = 1, dataset_size: int = 60000):
    vae.train()
    train_loss = []
    for image_batch, _ in dataloader:
        image_batch = image_batch.to(device)
        recon_batch, mu_batch, logvar_batch, _ = vae(image_batch)
        loss = beta_vae_loss(recon_batch, image_batch, mu_batch, logvar_batch, beta, dataset_size)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss.append(loss.detach().cpu().numpy())
    return np.mean(train_loss)


def test_beta_vae_epoch(vae: VariationalAutoencoderMnist, device: torch.device, dataloader: torch.utils.data.DataLoader,
                        beta: float = 1, dataset_size: int = 10000):
    vae.eval()
    test_loss = []
    with torch.no_grad():
        for image_batch, _ in dataloader:
            image_batch = image_batch.to(device)
            recon_batch, mu_batch, logvar_batch, _ = vae(image_batch)
            loss = beta_vae_loss(recon_batch, image_batch, mu_batch, logvar_batch, beta, dataset_size)
            test_loss.append(loss.cpu().numpy())
    return np.mean(test_loss)


def train_beta_tcvae_epoch(vae: VariationalAutoencoderMnist, device: torch.device,
                           dataloader: torch.utils.data.DataLoader, optimizer: torch.optim.Optimizer,
                           beta: float = 1, dataset_size: int = 60000):
    vae.train()
    train_loss = []
    for image_batch, _ in dataloader:
        image_batch = image_batch.to(device)
        recon_batch, mu_batch, logvar_batch, latent_batch = vae(image_batch)
        loss = beta_tcvae_loss_function(recon_batch, image_batch, mu_batch, logvar_batch,
                                        latent_batch, beta, dataset_size)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss.append(loss.detach().cpu().numpy())
    return np.mean(train_loss)


def test_beta_tcvae_epoch(vae: VariationalAutoencoderMnist, device: torch.device,
                          dataloader: torch.utils.data.DataLoader, beta: float = 1, dataset_size: int = 10000):
    vae.eval()
    test_loss = []
    with torch.no_grad():
        for image_batch, _ in dataloader:
            image_batch = image_batch.to(device)
            recon_batch, mu_batch, logvar_batch, latent_batch = vae(image_batch)
            loss = beta_tcvae_loss_function(recon_batch, image_batch, mu_batch, logvar_batch,
                                            latent_batch, beta, dataset_size)
            test_loss.append(loss.cpu().numpy())
    return np.mean(test_loss)

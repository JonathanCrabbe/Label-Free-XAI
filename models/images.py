from torch import nn
import torch.nn.functional as F
import torch
import numpy as np

'''
 This code is adapted from 
 https://medium.com/dataseries/convolutional-autoencoder-in-pytorch-on-mnist-dataset-d65145c132ac
 and 
 
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
        return x_recon, latent_mu, latent_logvar

    def latent_sample(self, mu, logvar):
        if self.training:
            # the reparameterization trick
            std = logvar.mul(0.5).exp_()
            eps = torch.empty_like(std).normal_()
            return eps.mul(std).add_(mu)
        else:
            return mu


def vae_loss(recon_x, x, mu, logvar, beta):
    # recon_x is the probability of a multivariate Bernoulli distribution p.
    # -log(p(x)) is then the pixel-wise binary cross-entropy.
    # Averaging or not averaging the binary cross-entropy over all pixels here
    # is a subtle detail with big effect on training, since it changes the weight
    # we need to pick for the other loss term by several orders of magnitude.
    # Not averaging is the direct implementation of the negative log likelihood,
    # but averaging makes the weight of the other loss term independent of the image resolution.
    recon_loss = F.binary_cross_entropy(recon_x.view(-1, 784), x.view(-1, 784), reduction='sum')

    # KL-divergence between the prior distribution over latent vectors
    # (the one we are going to sample from when generating new images)
    # and the distribution estimated by the generator for the given image.
    kldivergence = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return recon_loss + beta * kldivergence


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


def train_vae_epoch(vae: VariationalAutoencoderMnist, device: torch.device, dataloader: torch.utils.data.DataLoader,
                    optimizer: torch.optim.Optimizer, beta: float = 1):
    vae.train()
    train_loss = []
    for image_batch, _ in dataloader:
        image_batch = image_batch.to(device)
        recon_batch, mu_batch, logvar_batch = vae(image_batch)
        loss = vae_loss(recon_batch, image_batch, mu_batch, logvar_batch, beta)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss.append(loss.detach().cpu().numpy())
    return np.mean(train_loss)


def test_vae_epoch(vae: VariationalAutoencoderMnist, device: torch.device, dataloader: torch.utils.data.DataLoader,
                   beta: float = 1):
    vae.eval()
    test_loss = []
    with torch.no_grad():
        for image_batch, _ in dataloader:
            image_batch = image_batch.to(device)
            recon_batch, mu_batch, logvar_batch = vae(image_batch)
            loss = vae_loss(recon_batch, image_batch, mu_batch, logvar_batch, beta)
            test_loss.append(loss.cpu().numpy())
    return np.mean(test_loss)

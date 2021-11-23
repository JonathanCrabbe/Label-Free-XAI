import torch
import torchvision
import numpy as np
from torchvision import transforms
from torch.utils.data import DataLoader, random_split
from models.images import EncoderMnist, DecoderMnist, train_epoch, test_epoch
from pathlib import Path
from captum.attr import IntegratedGradients
from captum.attr._utils.visualization import visualize_image_attr
from explanations.features import AuxiliaryFunction



def denoiser_mnist(random_seed: int = 0, batch_size: int = 256, dim_latent: int = 4, n_epochs: int = 30):
    # Initialize seed and device
    torch.random.manual_seed(random_seed)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # Load MNIST
    data_dir = Path.cwd() / "data/mnist"
    train_dataset = torchvision.datasets.MNIST(data_dir, train=True, download=True)
    test_dataset = torchvision.datasets.MNIST(data_dir, train=False, download=True)
    train_transform = transforms.Compose([transforms.ToTensor()])
    test_transform = transforms.Compose([transforms.ToTensor()])
    train_dataset.transform = train_transform
    test_dataset.transform = test_transform
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Initialize encoder and decoder
    encoder = EncoderMnist(encoded_space_dim=dim_latent, fc2_input_dim=128)
    decoder = DecoderMnist(encoded_space_dim=dim_latent, fc2_input_dim=128)
    encoder.to(device)
    decoder.to(device)

    # Initialize optimizer
    loss_fn = torch.nn.MSELoss()
    params_to_optimize = [
        {'params': encoder.parameters()},
        {'params': decoder.parameters()}
    ]
    optim = torch.optim.Adam(params_to_optimize, lr=1e-03, weight_decay=1e-05)

    # Train the denoising autoencoder
    loss_hist = {'train_loss': [], 'val_loss': []}
    attribution_deltas = []
    baseline_features = torch.zeros((1, 28, 28), device=device)
    prev_attribution = np.zeros((len(test_dataset), 1, 28, 28))
    current_attribution = np.zeros((len(test_dataset), 1, 28, 28))
    for epoch in range(n_epochs):
        train_loss = train_epoch(encoder, decoder, device, train_loader, loss_fn, optim)
        val_loss = test_epoch(encoder, decoder, device, test_loader, loss_fn)
        loss_hist['train_loss'].append(train_loss)
        loss_hist['val_loss'].append(val_loss)
        for n_batch, (test_images, _) in enumerate(test_loader):
            test_images = test_images.to(device)
            auxiliary_encoder = AuxiliaryFunction(encoder, test_images)
            ig_explainer = IntegratedGradients(auxiliary_encoder)
            attr_batch = ig_explainer.attribute(test_images, baselines=baseline_features,
                                                internal_batch_size=len(test_images))
            current_attribution[n_batch*batch_size:(n_batch*batch_size+len(test_images))] = attr_batch.cpu().numpy()
        attribution_delta = np.sum((current_attribution - prev_attribution)**2)
        print(f'\n Epoch {epoch + 1}/{n_epochs} \t train loss {train_loss:.3g} \t val loss {val_loss:.3g} \t '
              f'attribution delta {attribution_delta:.3g}')
        attribution_deltas.append(attribution_delta.data)
        prev_attribution = current_attribution



    '''
    test_images, _ = next(iter(test_loader))
    test_image = test_images[10:11].to(device)
    auxiliary_encoder = AuxiliaryFunction(encoder, test_image)
    ig_explainer = IntegratedGradients(auxiliary_encoder)
    baseline_image = torch.zeros(test_image.shape, device=device)
    ig_attr = ig_explainer.attribute(test_image, baseline_image)

    visualize_image_attr(torch.permute(ig_attr[0], (1, 2, 0)).cpu().numpy(),
                         torch.permute(test_image[0], (1, 2, 0)).cpu().numpy(),
                         "blended_heat_map")
    '''


if __name__ == "__main__":
    denoiser_mnist()

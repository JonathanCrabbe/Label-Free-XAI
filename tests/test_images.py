from pathlib import Path

import torch
from torch.nn import MSELoss
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from torchvision.datasets import MNIST

from lfxai.explanations.examples import SimplEx
from lfxai.models.images import AutoEncoderMnist, DecoderMnist, EncoderMnist
from lfxai.models.pretext import Identity


def test_sanity_images() -> None:
    # Select torch device
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # Load data
    data_dir = Path.cwd() / "data/mnist"
    train_dataset = MNIST(data_dir, train=True, download=True)
    test_dataset = MNIST(data_dir, train=False, download=True)
    train_dataset.transform = transforms.Compose([transforms.ToTensor()])
    test_dataset.transform = transforms.Compose([transforms.ToTensor()])
    test_loader = DataLoader(test_dataset, batch_size=100, shuffle=False)

    # Get a model
    encoder = EncoderMnist(encoded_space_dim=10)
    decoder = DecoderMnist(encoded_space_dim=10)
    model = AutoEncoderMnist(encoder, decoder, latent_dim=10, input_pert=Identity())
    model.to(device)

    # Get label-free example importance
    train_subset = Subset(
        train_dataset, indices=list(range(100))
    )  # Limit the number of training examples
    train_subloader = DataLoader(train_subset, batch_size=100)
    attr_method = SimplEx(model, loss_f=MSELoss())
    example_importance = attr_method.attribute_loader(
        device, train_subloader, test_loader
    )

    assert example_importance.shape == (10000, 100)

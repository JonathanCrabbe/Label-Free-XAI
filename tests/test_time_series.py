import shutil
from pathlib import Path

import torch
from torch.utils.data import DataLoader, RandomSampler, Subset, random_split

from lfxai.explanations.examples import (
    InfluenceFunctions,
    NearestNeighbours,
    SimplEx,
    TracIn,
)
from lfxai.models.time_series import RecurrentAutoencoder
from lfxai.utils.datasets import ECG5000

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
random_seed = 1


def remove_create_dir(folder: Path) -> Path:
    folder = Path(folder)
    try:
        shutil.rmtree(folder)
    except BaseException:
        pass
    folder.mkdir(parents=True, exist_ok=True)

    return folder


def test_ts_autoencoder() -> None:
    batch_size = 100
    dim_latent = 20
    n_epochs = 1
    val_proportion = 0.2
    save_dir = remove_create_dir("tmp")

    torch.random.manual_seed(random_seed)
    data_dir = Path.cwd() / "data/ecg5000"

    # Create training, validation and test data
    train_dataset = ECG5000(data_dir, True, random_seed)

    # Attribution Baseline
    val_length = int(len(train_dataset) * val_proportion)  # Size of validation set
    train_dataset, val_dataset = random_split(
        train_dataset, (len(train_dataset) - val_length, val_length)
    )
    train_loader = DataLoader(train_dataset, batch_size, True)
    val_loader = DataLoader(val_dataset, batch_size, True)
    time_steps = 140
    n_features = 1

    # Fit an autoencoder
    autoencoder = RecurrentAutoencoder(time_steps, n_features, dim_latent)
    autoencoder.fit(device, train_loader, val_loader, save_dir, n_epochs, patience=10)
    autoencoder.train()


def test_ts_sanity_example_importance() -> None:
    batch_size: int = 100
    dim_latent: int = 16
    n_epochs: int = 1
    subtrain_size: int = 10
    save_dir = remove_create_dir("tmp")

    # Initialize seed and device
    torch.random.manual_seed(random_seed)
    data_dir = Path.cwd() / "data/ecg5000"

    # Load dataset
    train_dataset = ECG5000(data_dir, experiment="examples")
    train_dataset, test_dataset = random_split(train_dataset, (4000, 1000))
    train_loader = DataLoader(train_dataset, batch_size, True)
    test_loader = DataLoader(test_dataset, batch_size, False)
    time_steps = 140
    n_features = 1

    # Train the denoising autoencoder
    autoencoder = RecurrentAutoencoder(time_steps, n_features, dim_latent)
    autoencoder.fit(
        device,
        train_loader,
        test_loader,
        save_dir,
        n_epochs,
    )

    # Prepare subset loaders for example-based explanation methods
    y_train = torch.tensor([train_dataset[k][1] for k in range(len(train_dataset))])
    idx_subtrain = [
        torch.nonzero(y_train == (n % 2))[n // 2].item() for n in range(subtrain_size)
    ]
    idx_subtest = torch.randperm(len(test_dataset))[:subtrain_size]
    train_subset = Subset(train_dataset, idx_subtrain)
    test_subset = Subset(test_dataset, idx_subtest)
    subtrain_loader = DataLoader(train_subset)
    subtest_loader = DataLoader(test_subset)

    recursion_depth = 2

    train_sampler = RandomSampler(
        train_dataset, replacement=True, num_samples=recursion_depth * batch_size
    )
    train_loader_replacement = DataLoader(
        train_dataset, batch_size, sampler=train_sampler
    )

    # Fitting explainers, computing the metric and saving everything
    autoencoder.train().to(device)
    l1_loss = torch.nn.L1Loss()
    explainer_list = [
        InfluenceFunctions(autoencoder, l1_loss, save_dir / "if_grads"),
        TracIn(autoencoder, l1_loss, save_dir / "tracin_grads"),
        SimplEx(autoencoder, l1_loss),
        NearestNeighbours(autoencoder, l1_loss),
    ]
    for explainer in explainer_list:
        if isinstance(explainer, InfluenceFunctions):
            with torch.backends.cudnn.flags(enabled=False):
                attribution = explainer.attribute_loader(
                    device,
                    subtrain_loader,
                    subtest_loader,
                    train_loader_replacement=train_loader_replacement,
                    recursion_depth=recursion_depth,
                )
        else:
            attribution = explainer.attribute_loader(
                device, subtrain_loader, subtest_loader
            )
        assert attribution is not None

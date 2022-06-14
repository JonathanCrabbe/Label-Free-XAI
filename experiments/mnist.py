import argparse
import csv
import itertools
import logging
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torchvision
from captum.attr import GradientShap, IntegratedGradients, Saliency
from scipy.stats import spearmanr
from torch.utils.data import DataLoader, RandomSampler, Subset
from torchvision import transforms

from lfxai.explanations.examples import (
    InfluenceFunctions,
    NearestNeighbours,
    SimplEx,
    TracIn,
)
from lfxai.explanations.features import attribute_auxiliary, attribute_individual_dim
from lfxai.models.images import (
    VAE,
    AutoEncoderMnist,
    ClassifierMnist,
    DecoderBurgess,
    DecoderMnist,
    EncoderBurgess,
    EncoderMnist,
)
from lfxai.models.losses import BetaHLoss, BtcvaeLoss
from lfxai.models.pretext import Identity, Mask, RandomNoise
from lfxai.utils.datasets import MaskedMNIST
from lfxai.utils.feature_attribution import generate_masks
from lfxai.utils.metrics import (
    compute_metrics,
    cos_saliency,
    count_activated_neurons,
    entropy_saliency,
    pearson_saliency,
    similarity_rates,
    spearman_saliency,
)
from lfxai.utils.visualize import (
    correlation_latex_table,
    plot_pretext_saliencies,
    plot_pretext_top_example,
    plot_vae_saliencies,
    vae_box_plots,
)


def consistency_feature_importance(
    random_seed: int = 1,
    batch_size: int = 200,
    dim_latent: int = 4,
    n_epochs: int = 100,
) -> None:
    # Initialize seed and device
    torch.random.manual_seed(random_seed)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    W = 28  # Image width = height
    pert_percentages = [5, 10, 20, 50, 80, 100]

    # Load MNIST
    data_dir = Path.cwd() / "data/mnist"
    train_dataset = torchvision.datasets.MNIST(data_dir, train=True, download=True)
    test_dataset = torchvision.datasets.MNIST(data_dir, train=False, download=True)
    train_transform = transforms.Compose([transforms.ToTensor()])
    test_transform = transforms.Compose([transforms.ToTensor()])
    train_dataset.transform = train_transform
    test_dataset.transform = test_transform
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size)
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False
    )

    # Initialize encoder, decoder and autoencoder wrapper
    pert = RandomNoise()
    encoder = EncoderMnist(encoded_space_dim=dim_latent)
    decoder = DecoderMnist(encoded_space_dim=dim_latent)
    autoencoder = AutoEncoderMnist(encoder, decoder, dim_latent, pert)
    encoder.to(device)
    decoder.to(device)

    # Train the denoising autoencoder
    save_dir = Path.cwd() / "results/mnist/consistency_features"
    if not save_dir.exists():
        os.makedirs(save_dir)
    autoencoder.fit(device, train_loader, test_loader, save_dir, n_epochs)
    autoencoder.load_state_dict(
        torch.load(save_dir / (autoencoder.name + ".pt")), strict=False
    )

    attr_methods = {
        "Gradient Shap": GradientShap,
        "Integrated Gradients": IntegratedGradients,
        "Saliency": Saliency,
        "Random": None,
    }
    results_data = []
    baseline_features = torch.zeros((1, 1, W, W)).to(
        device
    )  # Baseline image for attributions
    for method_name in attr_methods:
        logging.info(f"Computing feature importance with {method_name}")
        results_data.append([method_name, 0, 0])
        attr_method = attr_methods[method_name]
        if attr_method is not None:
            attr = attribute_auxiliary(
                encoder, test_loader, device, attr_method(encoder), baseline_features
            )
        else:
            np.random.seed(random_seed)
            attr = np.random.randn(len(test_dataset), 1, W, W)

        for pert_percentage in pert_percentages:
            logging.info(
                f"Perturbing {pert_percentage}% of the features with {method_name}"
            )
            mask_size = int(pert_percentage * W**2 / 100)
            masks = generate_masks(attr, mask_size)
            for batch_id, (images, _) in enumerate(test_loader):
                mask = masks[
                    batch_id * batch_size : batch_id * batch_size + len(images)
                ].to(device)
                images = images.to(device)
                original_reps = encoder(images)
                images = mask * images
                pert_reps = encoder(images)
                rep_shift = torch.mean(
                    torch.sum((original_reps - pert_reps) ** 2, dim=-1)
                ).item()
                results_data.append([method_name, pert_percentage, rep_shift])

    logging.info("Saving the plot")
    results_df = pd.DataFrame(
        results_data, columns=["Method", "% Perturbed Pixels", "Representation Shift"]
    )
    sns.set(font_scale=1.3)
    sns.set_style("white")
    sns.set_palette("colorblind")
    sns.lineplot(
        data=results_df, x="% Perturbed Pixels", y="Representation Shift", hue="Method"
    )
    plt.tight_layout()
    plt.savefig(save_dir / "mnist_consistency_features.pdf")
    plt.close()


def consistency_examples(
    random_seed: int = 1,
    batch_size: int = 200,
    dim_latent: int = 4,
    n_epochs: int = 100,
    subtrain_size: int = 1000,
) -> None:
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
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False
    )

    # Initialize encoder, decoder and autoencoder wrapper
    pert = RandomNoise()
    encoder = EncoderMnist(encoded_space_dim=dim_latent)
    decoder = DecoderMnist(encoded_space_dim=dim_latent)
    autoencoder = AutoEncoderMnist(encoder, decoder, dim_latent, pert)
    encoder.to(device)
    decoder.to(device)
    autoencoder.to(device)

    # Train the denoising autoencoder
    logging.info("Now fitting autoencoder")
    save_dir = Path.cwd() / "results/mnist/consistency_examples"
    if not save_dir.exists():
        os.makedirs(save_dir)
    autoencoder.fit(
        device, train_loader, test_loader, save_dir, n_epochs, checkpoint_interval=10
    )
    autoencoder.load_state_dict(
        torch.load(save_dir / (autoencoder.name + ".pt")), strict=False
    )
    autoencoder.train().to(device)

    idx_subtrain = [
        torch.nonzero(train_dataset.targets == (n % 10))[n // 10].item()
        for n in range(subtrain_size)
    ]
    idx_subtest = [
        torch.nonzero(test_dataset.targets == (n % 10))[n // 10].item()
        for n in range(subtrain_size)
    ]
    train_subset = Subset(train_dataset, idx_subtrain)
    test_subset = Subset(test_dataset, idx_subtest)
    subtrain_loader = DataLoader(train_subset)
    subtest_loader = DataLoader(test_subset)
    labels_subtrain = torch.cat([label for _, label in subtrain_loader])
    labels_subtest = torch.cat([label for _, label in subtest_loader])

    # Create a training set sampler with replacement for computing influence functions
    recursion_depth = 100
    train_sampler = RandomSampler(
        train_dataset, replacement=True, num_samples=recursion_depth * batch_size
    )
    train_loader_replacement = DataLoader(
        train_dataset, batch_size, sampler=train_sampler
    )

    # Fitting explainers, computing the metric and saving everything
    mse_loss = torch.nn.MSELoss()
    explainer_list = [
        InfluenceFunctions(autoencoder, mse_loss, save_dir / "if_grads"),
        TracIn(autoencoder, mse_loss, save_dir / "tracin_grads"),
        SimplEx(autoencoder, mse_loss),
        NearestNeighbours(autoencoder, mse_loss),
    ]
    frac_list = [0.05, 0.1, 0.2, 0.5, 0.7, 1.0]
    n_top_list = [int(frac * len(idx_subtrain)) for frac in frac_list]
    results_list = []
    for explainer in explainer_list:
        logging.info(f"Now fitting {explainer} explainer")
        attribution = explainer.attribute_loader(
            device,
            subtrain_loader,
            subtest_loader,
            train_loader_replacement=train_loader_replacement,
            recursion_depth=recursion_depth,
        )
        autoencoder.load_state_dict(
            torch.load(save_dir / (autoencoder.name + ".pt")), strict=False
        )
        sim_most, sim_least = similarity_rates(
            attribution, labels_subtrain, labels_subtest, n_top_list
        )
        results_list += [
            [str(explainer), "Most Important", 100 * frac, sim]
            for frac, sim in zip(frac_list, sim_most)
        ]
        results_list += [
            [str(explainer), "Least Important", 100 * frac, sim]
            for frac, sim in zip(frac_list, sim_least)
        ]
    results_df = pd.DataFrame(
        results_list,
        columns=[
            "Explainer",
            "Type of Examples",
            "% Examples Selected",
            "Similarity Rate",
        ],
    )
    logging.info(f"Saving results in {save_dir}")
    results_df.to_csv(save_dir / "metrics.csv")
    sns.lineplot(
        data=results_df,
        x="% Examples Selected",
        y="Similarity Rate",
        hue="Explainer",
        style="Type of Examples",
        palette="colorblind",
    )
    plt.savefig(save_dir / "similarity_rates.pdf")


def pretext_task_sensitivity(
    random_seed: int = 1,
    batch_size: int = 300,
    n_runs: int = 5,
    dim_latent: int = 4,
    n_epochs: int = 100,
    patience: int = 10,
    subtrain_size: int = 1000,
    n_plots: int = 10,
) -> None:
    # Initialize seed and device
    np.random.seed(random_seed)
    torch.random.manual_seed(random_seed)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    mse_loss = torch.nn.MSELoss()

    # Load MNIST
    W = 28
    data_dir = Path.cwd() / "data/mnist"
    train_dataset = torchvision.datasets.MNIST(data_dir, train=True, download=True)
    test_dataset = torchvision.datasets.MNIST(data_dir, train=False, download=True)
    train_transform = transforms.Compose([transforms.ToTensor()])
    test_transform = transforms.Compose([transforms.ToTensor()])
    train_dataset.transform = train_transform
    test_dataset.transform = test_transform
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size)
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False
    )
    X_train = train_dataset.data
    X_train = X_train.unsqueeze(1).float()
    X_test = test_dataset.data
    X_test = X_test.unsqueeze(1).float()
    idx_subtrain = [
        torch.nonzero(train_dataset.targets == (n % 10))[n // 10].item()
        for n in range(subtrain_size)
    ]

    # Create saving directory
    save_dir = Path.cwd() / "results/mnist/pretext"
    if not save_dir.exists():
        logging.info(f"Creating saving directory {save_dir}")
        os.makedirs(save_dir)

    # Define the computed metrics and create a csv file with appropriate headers
    pretext_list = [Identity(), RandomNoise(noise_level=0.3), Mask(mask_proportion=0.2)]
    headers = [str(pretext) for pretext in pretext_list] + [
        "Classification"
    ]  # Name of each task
    n_tasks = len(pretext_list) + 1
    feature_pearson = np.zeros((n_runs, n_tasks, n_tasks))
    feature_spearman = np.zeros((n_runs, n_tasks, n_tasks))
    example_pearson = np.zeros((n_runs, n_tasks, n_tasks))
    example_spearman = np.zeros((n_runs, n_tasks, n_tasks))

    for run in range(n_runs):
        feature_importance = []
        example_importance = []
        # Perform the experiment with several autoencoders trained on different pretext tasks.
        for pretext in pretext_list:
            # Create and fit an autoencoder for the pretext task
            name = f"{str(pretext)}-ae_run{run}"
            encoder = EncoderMnist(dim_latent)
            decoder = DecoderMnist(dim_latent)
            model = AutoEncoderMnist(encoder, decoder, dim_latent, pretext, name)
            logging.info(f"Now fitting {name}")
            model.fit(device, train_loader, test_loader, save_dir, n_epochs, patience)
            model.load_state_dict(torch.load(save_dir / (name + ".pt")), strict=False)
            # Compute feature importance
            logging.info("Computing feature importance")
            baseline_image = torch.zeros((1, 1, 28, 28), device=device)
            gradshap = GradientShap(encoder)
            feature_importance.append(
                np.abs(
                    np.expand_dims(
                        attribute_auxiliary(
                            encoder, test_loader, device, gradshap, baseline_image
                        ),
                        0,
                    )
                )
            )
            # Compute example importance
            logging.info("Computing example importance")
            dknn = NearestNeighbours(model.cpu(), X_train, mse_loss)
            example_importance.append(
                np.expand_dims(dknn.attribute(X_test, idx_subtrain).cpu().numpy(), 0)
            )

        # Create and fit a MNIST classifier
        name = f"Classifier_run{run}"
        encoder = EncoderMnist(dim_latent)
        classifier = ClassifierMnist(encoder, dim_latent, name)
        logging.info(f"Now fitting {name}")
        classifier.fit(device, train_loader, test_loader, save_dir, n_epochs, patience)
        classifier.load_state_dict(torch.load(save_dir / (name + ".pt")), strict=False)
        baseline_image = torch.zeros((1, 1, 28, 28), device=device)
        # Compute feature importance for the classifier
        logging.info("Computing feature importance")
        gradshap = GradientShap(encoder)
        feature_importance.append(
            np.abs(
                np.expand_dims(
                    attribute_auxiliary(
                        encoder, test_loader, device, gradshap, baseline_image
                    ),
                    0,
                )
            )
        )
        # Compute example importance for the classifier
        logging.info("Computing example importance")
        dknn = NearestNeighbours(classifier.cpu(), X_train, mse_loss)
        example_importance.append(
            np.expand_dims(dknn.attribute(X_test, idx_subtrain).cpu().numpy(), 0)
        )

        # Compute correlation between the saliency of different pretext tasks
        feature_importance = np.concatenate(feature_importance)
        feature_pearson[run] = np.corrcoef(feature_importance.reshape((n_tasks, -1)))
        feature_spearman[run] = spearmanr(
            feature_importance.reshape((n_tasks, -1)), axis=1
        )[0]
        example_importance = np.concatenate(example_importance)
        example_pearson[run] = np.corrcoef(example_importance.reshape((n_tasks, -1)))
        example_spearman[run] = spearmanr(
            example_importance.reshape((n_tasks, -1)), axis=1
        )[0]
        logging.info(
            f"Run {run} complete \n Feature Pearson \n {np.round(feature_pearson[run], decimals=2)}"
            f"\n Feature Spearman \n {np.round(feature_spearman[run], decimals=2)}"
            f"\n Example Pearson \n {np.round(example_pearson[run], decimals=2)}"
            f"\n Example Spearman \n {np.round(example_spearman[run], decimals=2)}"
        )

        # Plot a couple of examples
        idx_plot = [
            torch.nonzero(test_dataset.targets == (n % 10))[n // 10].item()
            for n in range(n_plots)
        ]
        test_images_to_plot = [X_test[i][0].numpy().reshape(W, W) for i in idx_plot]
        train_images_to_plot = [
            X_train[i][0].numpy().reshape(W, W) for i in idx_subtrain
        ]
        fig_features = plot_pretext_saliencies(
            test_images_to_plot, feature_importance[:, idx_plot, :, :, :], headers
        )
        fig_features.savefig(save_dir / f"saliency_maps_run{run}.pdf")
        plt.close(fig_features)
        fig_examples = plot_pretext_top_example(
            train_images_to_plot,
            test_images_to_plot,
            example_importance[:, idx_plot, :],
            headers,
        )
        fig_examples.savefig(save_dir / f"top_examples_run{run}.pdf")
        plt.close(fig_features)

    # Compute the avg and std for each metric
    feature_pearson_avg = np.round(np.mean(feature_pearson, axis=0), decimals=2)
    feature_pearson_std = np.round(np.std(feature_pearson, axis=0), decimals=2)
    feature_spearman_avg = np.round(np.mean(feature_spearman, axis=0), decimals=2)
    feature_spearman_std = np.round(np.std(feature_spearman, axis=0), decimals=2)
    example_pearson_avg = np.round(np.mean(example_pearson, axis=0), decimals=2)
    example_pearson_std = np.round(np.std(example_pearson, axis=0), decimals=2)
    example_spearman_avg = np.round(np.mean(example_spearman, axis=0), decimals=2)
    example_spearman_std = np.round(np.std(example_spearman, axis=0), decimals=2)

    # Format the metrics in Latex tables
    with open(save_dir / "tables.tex", "w") as f:
        for corr_avg, corr_std in zip(
            [
                feature_pearson_avg,
                feature_spearman_avg,
                example_pearson_avg,
                example_spearman_avg,
            ],
            [
                feature_pearson_std,
                feature_spearman_std,
                example_pearson_std,
                example_spearman_std,
            ],
        ):
            f.write(correlation_latex_table(corr_avg, corr_std, headers))
            f.write("\n")


def disvae_feature_importance(
    random_seed: int = 1,
    batch_size: int = 300,
    n_plots: int = 20,
    n_runs: int = 5,
    dim_latent: int = 3,
    n_epochs: int = 100,
    beta_list: list = [1, 5, 10],
) -> None:
    # Initialize seed and device
    np.random.seed(random_seed)
    torch.random.manual_seed(random_seed)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # Load MNIST
    W = 32
    img_size = (1, W, W)
    data_dir = Path.cwd() / "data/mnist"
    train_dataset = torchvision.datasets.MNIST(data_dir, train=True, download=True)
    test_dataset = torchvision.datasets.MNIST(data_dir, train=False, download=True)
    train_transform = transforms.Compose([transforms.Resize(W), transforms.ToTensor()])
    test_transform = transforms.Compose([transforms.Resize(W), transforms.ToTensor()])
    train_dataset.transform = train_transform
    test_dataset.transform = test_transform
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size)
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False
    )

    # Create saving directory
    save_dir = Path.cwd() / "results/mnist/vae"
    if not save_dir.exists():
        logging.info(f"Creating saving directory {save_dir}")
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
        plot_idx = [
            torch.nonzero(test_dataset.targets == (n % 10))[n // 10].item()
            for n in range(n_plots)
        ]
        images_to_plot = [test_dataset[i][0].numpy().reshape(W, W) for i in plot_idx]
        fig = plot_vae_saliencies(images_to_plot, attributions[plot_idx])
        fig.savefig(save_dir / f"{name}.pdf")
        plt.close(fig)

    fig = vae_box_plots(pd.read_csv(csv_path), metric_names)
    fig.savefig(save_dir / "metric_box_plots.pdf")
    plt.close(fig)


def roar_test(
    random_seed: int = 1,
    batch_size: int = 200,
    dim_latent: int = 4,
    n_epochs: int = 100,
) -> None:
    # Initialize seed and device
    logging.info("Welcome in the ROAR test experiments")
    torch.random.manual_seed(random_seed)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    remove_percentages = [10, 20, 50, 70, 100]

    # Load MNIST
    W = 28  # Image width = height
    data_dir = Path.cwd() / "data/mnist"
    train_dataset = torchvision.datasets.MNIST(data_dir, train=True, download=True)
    test_dataset = torchvision.datasets.MNIST(data_dir, train=False, download=True)
    train_transform = transforms.Compose([transforms.ToTensor()])
    test_transform = transforms.Compose([transforms.ToTensor()])
    train_dataset.transform = train_transform
    test_dataset.transform = test_transform
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False
    )
    save_dir = Path.cwd() / "results/mnist/roar_test"
    if not save_dir.exists():
        os.makedirs(save_dir)

    # Initialize encoder, decoder and autoencoder wrapper
    pert = Identity()
    encoder = EncoderMnist(encoded_space_dim=dim_latent)
    decoder = DecoderMnist(encoded_space_dim=dim_latent)
    autoencoder = AutoEncoderMnist(
        encoder, decoder, dim_latent, pert, name="model_initial"
    )
    autoencoder.save(save_dir)
    encoder.to(device)
    decoder.to(device)

    # Train the denoising autoencoder
    logging.info("Training the initial autoencoder")
    autoencoder = AutoEncoderMnist(encoder, decoder, dim_latent, pert, name="model")
    autoencoder.load_state_dict(torch.load(save_dir / "model_initial.pt"), strict=False)
    autoencoder.fit(device, train_loader, test_loader, save_dir, n_epochs)
    autoencoder.load_state_dict(
        torch.load(save_dir / (autoencoder.name + ".pt")), strict=False
    )
    original_test_performance = autoencoder.test_epoch(device, test_loader)

    # Create dictionaries to store feature importance and shift induced by perturbations
    explainer_dic = {
        "Gradient Shap": GradientShap(encoder),
        "Integrated Gradients": IntegratedGradients(encoder),
        "Random": None,
    }
    baseline_features = torch.zeros((1, 1, W, W)).to(
        device
    )  # Baseline image for attributions
    results_data = []

    for explainer_name in explainer_dic:
        logging.info(f"Computing feature importance with {explainer_name}")
        results_data.append([explainer_name, 0, original_test_performance])
        if explainer_dic[explainer_name] is not None:
            attr = attribute_auxiliary(
                encoder,
                train_loader,
                device,
                explainer_dic[explainer_name],
                baseline_features,
            )
        else:  # Random attribution
            attr = np.random.randn(len(train_dataset), 1, W, W)
        for remove_percentage in remove_percentages:
            mask_size = int(remove_percentage * (W**2) / 100)
            torch.random.manual_seed(random_seed)
            logging.info(
                f"Retraining an autoencoder with {remove_percentage}% pixels masked by {explainer_name}"
            )
            masks = generate_masks(attr, mask_size)
            masked_train_set = MaskedMNIST(data_dir, True, masks)
            masked_train_set.transform = train_transform
            masked_train_loader = DataLoader(
                masked_train_set, batch_size=batch_size, shuffle=True
            )
            encoder = EncoderMnist(encoded_space_dim=dim_latent)
            decoder = DecoderMnist(encoded_space_dim=dim_latent)
            autoencoder_name = f"model_{explainer_name}_mask{mask_size}"
            autoencoder = AutoEncoderMnist(
                encoder, decoder, dim_latent, pert, name=autoencoder_name
            )
            autoencoder.load_state_dict(
                torch.load(save_dir / "model_initial.pt"), strict=False
            )
            encoder.to(device)
            decoder.to(device)
            autoencoder.fit(
                device, masked_train_loader, test_loader, save_dir, n_epochs
            )
            autoencoder.load_state_dict(
                torch.load(save_dir / (autoencoder_name + ".pt")), strict=False
            )
            test_performance = autoencoder.test_epoch(device, test_loader)
            results_data.append([explainer_name, remove_percentage, test_performance])

    logging.info(f"Saving the plot in {str(save_dir)}")
    results_df = pd.DataFrame(
        results_data, columns=["Method", "% of features removed", "Test Loss"]
    )
    sns.set(font_scale=1.3)
    sns.set_style("white")
    sns.set_palette("colorblind")
    sns.lineplot(
        data=results_df, x="% of features removed", y="Test Loss", hue="Method"
    )
    plt.tight_layout()
    plt.savefig(save_dir / "roar.pdf")
    plt.close()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str, default="disvae")
    parser.add_argument("--n_runs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=300)
    parser.add_argument("--random_seed", type=int, default=1)
    args = parser.parse_args()
    if args.name == "disvae":
        disvae_feature_importance(
            n_runs=args.n_runs, batch_size=args.batch_size, random_seed=args.random_seed
        )
    elif args.name == "pretext":
        pretext_task_sensitivity(
            n_runs=args.n_runs, batch_size=args.batch_size, random_seed=args.random_seed
        )
    elif args.name == "consistency_features":
        consistency_feature_importance(
            batch_size=args.batch_size, random_seed=args.random_seed
        )
    elif args.name == "consistency_examples":
        consistency_examples(batch_size=args.batch_size, random_seed=args.random_seed)
    elif args.name == "roar_test":
        roar_test(batch_size=args.batch_size, random_seed=args.random_seed, n_epochs=10)
    else:
        raise ValueError("Invalid experiment name")

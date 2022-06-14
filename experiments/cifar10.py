import logging
import os
from pathlib import Path

import hydra
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from captum.attr import GradientShap, IntegratedGradients, Saliency
from models.images import SimCLR
from omegaconf import DictConfig
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import CIFAR10
from torchvision.transforms import GaussianBlur, ToTensor

from lfxai.explanations.examples import NearestNeighbours, SimplEx
from lfxai.explanations.features import attribute_auxiliary
from lfxai.utils.feature_attribution import generate_masks
from lfxai.utils.metrics import similarity_rates


def fit_model(args: DictConfig):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    # Prepare model
    torch.manual_seed(args.seed)
    assert args.backbone in ["resnet18", "resnet34"]
    base_encoder = eval(args.backbone)
    model = SimCLR(base_encoder, projection_dim=args.projection_dim).to(device)
    logging.info("Fitting SimCLR model")
    model.fit(args, device)


def consistency_feature_importance(args: DictConfig):
    torch.manual_seed(args.seed)
    save_dir = Path.cwd() / "consistency_features"
    if not save_dir.exists():
        os.makedirs(save_dir)
    model_path = Path.cwd() / f"models/simclr_{args.backbone}_epoch{args.epochs}.pt"
    # Fit a model if it does not exist yet
    if not model_path.exists():
        if not (Path.cwd() / "models").exists():
            os.makedirs(Path.cwd() / "models")
        fit_model(args)

    # Prepare the model
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    pert_percentages = [5, 10, 20, 50, 80, 100]
    perturbation = GaussianBlur(21, sigma=5).to(device)

    assert args.backbone in ["resnet18", "resnet34"]
    base_encoder = eval(args.backbone)
    model = SimCLR(base_encoder, projection_dim=args.projection_dim).to(device)
    logging.info(
        f"Base model: {args.backbone} - feature dim: {model.feature_dim} - projection dim {args.projection_dim}"
    )
    model.load_state_dict(torch.load(model_path), strict=False)

    # Compute feature importance
    W = 32
    test_batch_size = int(args.batch_size / 20)
    encoder = model.encoder
    data_dir = hydra.utils.to_absolute_path(args.data_dir)
    test_set = CIFAR10(data_dir, False, transform=ToTensor())
    test_loader = DataLoader(test_set, test_batch_size)
    attr_methods = {
        "Gradient Shap": GradientShap,
        "Integrated Gradients": IntegratedGradients,
        "Saliency": Saliency,
        "Random": None,
    }
    results_data = []
    for method_name in attr_methods:
        logging.info(f"Computing feature importance with {method_name}")
        results_data.append([method_name, 0, 0])
        attr_method = attr_methods[method_name]
        if attr_method is not None:
            attr = attribute_auxiliary(
                encoder, test_loader, device, attr_method(encoder), perturbation
            )
        else:
            np.random.seed(args.seed)
            attr = np.random.randn(len(test_set), 1, W, W)

        for pert_percentage in pert_percentages:
            logging.info(
                f"Perturbing {pert_percentage}% of the features with {method_name}"
            )
            mask_size = int(pert_percentage * W**2 / 100)
            masks = generate_masks(attr, mask_size)
            for batch_id, (images, _) in enumerate(test_loader):
                mask = masks[
                    batch_id * test_batch_size : batch_id * test_batch_size
                    + len(images)
                ].to(device)
                images = images.to(device)
                original_reps = encoder(images)
                images = mask * images + (1 - mask) * perturbation(images)
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
    plt.savefig(save_dir / "cifar10_consistency_features.pdf")
    plt.close()


def consistency_example_importance(args: DictConfig):
    torch.manual_seed(args.seed)
    save_dir = Path.cwd() / "consistency_examples"
    if not save_dir.exists():
        os.makedirs(save_dir)
    model_path = Path.cwd() / f"models/simclr_{args.backbone}_epoch{args.epochs}.pt"
    # Fit a model if it does not exist yet

    if not model_path.exists():
        if not (Path.cwd() / "models").exists():
            os.makedirs(Path.cwd() / "models")
        fit_model(args)

    # Prepare the model
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    assert args.backbone in ["resnet18", "resnet34"]
    base_encoder = eval(args.backbone)
    model = SimCLR(base_encoder, projection_dim=args.projection_dim).to(device)
    logging.info(
        f"Base model: {args.backbone} - feature dim: {model.feature_dim} - projection dim {args.projection_dim}"
    )
    model.load_state_dict(torch.load(model_path), strict=False)

    # Compute feature importance
    test_batch_size = int(args.batch_size / 20)
    data_dir = hydra.utils.to_absolute_path(args.data_dir)
    train_set = CIFAR10(data_dir, False, transform=ToTensor())
    train_indices = torch.randperm(len(train_set))[:1000]
    train_subset = Subset(train_set, train_indices)
    train_loader = DataLoader(train_subset, test_batch_size)
    labels_subtrain = torch.cat([labels for _, labels in train_loader])
    test_set = CIFAR10(data_dir, False, transform=ToTensor())
    test_indices = torch.randperm(len(test_set))[:1000]
    test_subset = Subset(test_set, test_indices)
    test_loader = DataLoader(test_subset, test_batch_size)
    labels_subtest = torch.cat([labels for _, labels in test_loader])
    attr_methods = {"SimplEx": SimplEx(model), "DKNN": NearestNeighbours(model)}
    results_data = []
    frac_list = [0.05, 0.1, 0.2, 0.5, 0.7, 1.0]
    n_top_list = [int(frac * len(train_subset)) for frac in frac_list]
    for method_name in attr_methods:
        logging.info(f"Computing feature importance with {method_name}")
        attr = attr_methods[method_name].attribute_loader(
            device=device, train_loader=train_loader, test_loader=test_loader
        )
        sim_most, sim_least = similarity_rates(
            attr, labels_subtrain, labels_subtest, n_top_list
        )
        results_data += [
            [method_name, "Most Important", 100 * frac, sim]
            for frac, sim in zip(frac_list, sim_most)
        ]
        results_data += [
            [method_name, "Least Important", 100 * frac, sim]
            for frac, sim in zip(frac_list, sim_least)
        ]
    results_df = pd.DataFrame(
        results_data,
        columns=[
            "Explainer",
            "Type of Examples",
            "% Examples Selected",
            "Similarity Rate",
        ],
    )
    results_df.to_csv(save_dir / "metrics.csv")
    pal = sns.color_palette("colorblind")[2:4]
    sns.lineplot(
        data=results_df,
        x="% Examples Selected",
        y="Similarity Rate",
        hue="Explainer",
        style="Type of Examples",
        palette=pal,
    )
    plt.savefig(save_dir / "cifar10_similarity_rates.pdf")


@hydra.main(config_name="simclr_config.yaml", config_path=str(Path.cwd()))
def main(args: DictConfig):
    if args.experiment_name == "consistency_features":
        consistency_feature_importance(args)
    elif args.experiment_name == "consistency_examples":
        consistency_example_importance(args)
    else:
        raise ValueError("Invalid experiment name")


if __name__ == "__main__":
    main()

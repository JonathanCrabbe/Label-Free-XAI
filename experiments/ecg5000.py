import os
import torch
import logging
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
from utils.datasets import ECG5000
from torch.utils.data import DataLoader, random_split
from models.time_series import RecurrentAutoencoder
from explanations.features import AuxiliaryFunction
from captum.attr import GradientShap, Saliency, IntegratedGradients, DeepLift


def consistency_feature_importance(random_seed: int = 1, batch_size: int = 50,
                                   dim_latent: int = 64, n_epochs: int = 150,
                                   val_proportion: float = .2) -> None:

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.random.manual_seed(random_seed)
    data_dir = Path.cwd() / "data/ecg5000"
    save_dir = Path.cwd() / "results/ecg5000/consistency_features"
    if not save_dir.exists():
        os.makedirs(save_dir)

    # Create training, validation and test data
    train_dataset = ECG5000(data_dir, True, random_seed)
    baseline_sequence = torch.mean(torch.stack(train_dataset.sequences), dim=0, keepdim=True).to(device)  # Attribution Baseline
    val_length = int(len(train_dataset)*val_proportion)  # Size of validation set
    train_dataset, val_dataset = random_split(train_dataset, (len(train_dataset)-val_length, val_length))
    train_loader = DataLoader(train_dataset, batch_size, True)
    val_loader = DataLoader(val_dataset, batch_size, True)
    test_dataset = ECG5000(data_dir, False, random_seed)
    test_loader = DataLoader(test_dataset, batch_size, False)
    time_steps = 140
    n_features = 1

    # Fit an autoencoder
    autoencoder = RecurrentAutoencoder(time_steps, n_features, dim_latent)
    autoencoder.fit(device, train_loader, val_loader, save_dir, n_epochs, patience=10)
    autoencoder.train()
    encoder = autoencoder.encoder
    autoencoder.load_state_dict(torch.load(save_dir / (autoencoder.name + ".pt")), strict=False)

    # Create dictionaries to store feature importance and shift induced by perturbations
    n_time_steps_pert = [1, 2, 5, 10, 15, 20, 50, 70, 100]
    attr_dic = {'Gradient Shap': np.zeros((len(test_dataset), 140, 1)),
                'Integrated Gradients': np.zeros((len(test_dataset), 140, 1)),
                'DeepLift': np.zeros((len(test_dataset), 140, 1)),
                'Saliency': np.zeros((len(test_dataset), 140, 1))
                }
    rep_shift_dic = {'Gradient Shap': np.zeros((len(n_time_steps_pert), len(test_loader))),
                     'Integrated Gradients': np.zeros((len(n_time_steps_pert), len(test_loader))),
                     'DeepLift': np.zeros((len(n_time_steps_pert), len(test_loader))),
                     'Saliency': np.zeros((len(n_time_steps_pert), len(test_loader))),
                     'Random': np.zeros((len(n_time_steps_pert), len(test_loader)))}

    logging.info("Fitting explainers.")
    for n_batch, (batch_sequences, _) in enumerate(test_loader):
        batch_sequences = batch_sequences.to(device)
        auxiliary_encoder = AuxiliaryFunction(encoder, batch_sequences)  # Parametrize the auxiliary encoder to the batch
        explainer_dic = {'Gradient Shap': GradientShap(auxiliary_encoder),
                         'Integrated Gradients': IntegratedGradients(auxiliary_encoder),
                         'DeepLift': DeepLift(auxiliary_encoder),
                         'Saliency': Saliency(auxiliary_encoder)
                         }
        for explainer in rep_shift_dic:
            if explainer in ["Gradient Shap", "Integrated Gradients", "DeepLift"]:
                attr_batch = explainer_dic[explainer].attribute(batch_sequences, baseline_sequence)
            elif explainer in ["Saliency"]:
                attr_batch = explainer_dic[explainer].attribute(batch_sequences)
            for pert_id, n_pert in enumerate(n_time_steps_pert):
                mask = torch.ones(batch_sequences.shape, device=device)
                if explainer in explainer_dic.keys():  # Perturb the most important pixels
                    top_time_steps = torch.topk(torch.abs(attr_batch).view(len(batch_sequences), -1), n_pert)[1]
                    for k in range(n_pert):
                        mask[:, top_time_steps[:, k], 0] = 0
                    attr_dic[explainer][n_batch * batch_size:(n_batch * batch_size + len(batch_sequences))] = \
                        attr_batch.detach().cpu().numpy()
                elif explainer == "Random":  # Perturb random pixels
                    for batch_id in range(len(batch_sequences)):
                        top_time_steps = torch.randperm(time_steps)[:n_pert]
                        for k in range(n_pert):
                            mask[batch_id, top_time_steps[k], 0] = 0
                batch_sequences_pert = mask * batch_sequences + (1-mask) * baseline_sequence
                # Compute the latent shift between perturbed and unperturbed images
                representation_shift = torch.sqrt(
                    torch.sum((encoder(batch_sequences_pert) - encoder(batch_sequences)) ** 2, -1))
                rep_shift_dic[explainer][pert_id, n_batch] = torch.mean(representation_shift).cpu().detach().numpy()

    # Plot the results
    sns.set_style("white")
    sns.set_palette("colorblind")
    for explainer in rep_shift_dic:
        plt.plot(n_time_steps_pert, rep_shift_dic[explainer].mean(axis=-1), label=explainer)
        plt.fill_between(n_time_steps_pert, rep_shift_dic[explainer].mean(axis=-1)-rep_shift_dic[explainer].std(axis=-1),
                         rep_shift_dic[explainer].mean(axis=-1) + rep_shift_dic[explainer].std(axis=-1), alpha=0.4)
    plt.xlabel("Number of Perturbed Time Steps")
    plt.ylabel("Latent Shift")
    plt.legend()
    plt.savefig(save_dir / "time_pert.pdf")




if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    consistency_feature_importance()
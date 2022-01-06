import os
import logging
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
import torch
import torchvision
import itertools
import argparse
import csv
from captum.attr import GradientShap, DeepLift, IntegratedGradients, Saliency
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from scipy.stats import  spearmanr
from explanations.features import AuxiliaryFunction, attribute_auxiliary, attribute_individual_dim
from explanations.examples import InfluenceFunctions, TracIn, SimplEx, NearestNeighbours
from models.images import EncoderMnist, DecoderMnist, ClassifierMnist, AutoEncoderMnist, VAE, EncoderBurgess, \
    DecoderBurgess
from models.losses import BetaHLoss, BtcvaeLoss
from models.pretext import Identity, RandomNoise, Mask
from utils.metrics import cos_saliency, entropy_saliency, similarity_rate, \
    count_activated_neurons, pearson_saliency, compute_metrics, spearman_saliency, top_consistency
from utils.visualize import plot_vae_saliencies, vae_box_plots, correlation_latex_table, plot_pretext_saliencies,\
plot_pretext_top_example


def consistency_feature_importance(random_seed: int = 1, batch_size: int = 200,
                                   dim_latent: int = 4, n_epochs: int = 100) -> None:
    # Initialize seed and device
    torch.random.manual_seed(random_seed)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    n_pixels_pert = [1, 2, 3, 4, 5, 10, 20, 30, 50]

    # Load MNIST
    W = 28  # Image width = height
    data_dir = Path.cwd() / "data/mnist"
    train_dataset = torchvision.datasets.MNIST(data_dir, train=True, download=True)
    test_dataset = torchvision.datasets.MNIST(data_dir, train=False, download=True)
    train_transform = transforms.Compose([transforms.ToTensor()])
    test_transform = transforms.Compose([transforms.ToTensor()])
    train_dataset.transform = train_transform
    test_dataset.transform = test_transform
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

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
    autoencoder.load_state_dict(torch.load(save_dir / (autoencoder.name + ".pt")), strict=False)

    # Create dictionaries to store feature importance and shift induced by perturbations
    attr_dic = {'Gradient Shap': np.zeros((len(test_dataset), 1, 28, 28)),
                'Integrated Gradients': np.zeros((len(test_dataset), 1, 28, 28)),
                'DeepLift': np.zeros((len(test_dataset), 1, 28, 28)),
                'Saliency': np.zeros((len(test_dataset), 1, 28, 28))
                }
    rep_shift_dic = {'Gradient Shap': np.zeros((len(n_pixels_pert), len(test_loader))),
                     'Integrated Gradients': np.zeros((len(n_pixels_pert), len(test_loader))),
                     'DeepLift': np.zeros((len(n_pixels_pert), len(test_loader))),
                     'Saliency': np.zeros((len(n_pixels_pert), len(test_loader))),
                     'Random': np.zeros((len(n_pixels_pert), len(test_loader)))}
    baseline_features = torch.zeros((1, 1, W, W)).to(device)  # Baseline image for attributions

    for n_batch, (batch_images, _) in enumerate(test_loader):
        batch_images = batch_images.to(device)
        auxiliary_encoder = AuxiliaryFunction(encoder, batch_images)  # Parametrize the auxiliary encoder to the batch
        explainer_dic = {'Gradient Shap': GradientShap(auxiliary_encoder),
                         'Integrated Gradients': IntegratedGradients(auxiliary_encoder),
                         'DeepLift': DeepLift(auxiliary_encoder),
                         'Saliency': Saliency(auxiliary_encoder)
                         }
        for explainer in rep_shift_dic:
            if explainer in ["Gradient Shap", "Integrated Gradients", "DeepLift"]:
                attr_batch = explainer_dic[explainer].attribute(batch_images, baseline_features)
            elif explainer in ["Saliency"]:
                attr_batch = explainer_dic[explainer].attribute(batch_images)
            for pert_id, n_pixels in enumerate(n_pixels_pert):
                mask = torch.ones(batch_images.shape, device=device)
                if explainer in explainer_dic.keys():  # Perturb the most important pixels
                    top_pixels = torch.topk(torch.abs(attr_batch).view(len(batch_images), -1), n_pixels)[1]
                    for k in range(n_pixels):
                        mask[:, 0, top_pixels[:, k] // W, top_pixels[:, k] % W] = 0
                    attr_dic[explainer][n_batch * batch_size:(n_batch * batch_size + len(batch_images))] = \
                        attr_batch.detach().cpu().numpy()
                elif explainer == "Random":  # Perturb random pixels
                    for batch_id in range(len(batch_images)):
                        top_pixels = torch.randperm(W ** 2)[:n_pixels]
                        for k in range(n_pixels):
                            mask[batch_id, 0, top_pixels[k] // W, top_pixels[k] % W] = 0
                batch_images_pert = mask * batch_images
                # Compute the latent shift between perturbed and unperturbed images
                representation_shift = torch.sqrt(
                    torch.sum((encoder(batch_images_pert) - encoder(batch_images)) ** 2, -1))
                rep_shift_dic[explainer][pert_id, n_batch] = torch.mean(representation_shift).cpu().detach().numpy()

    sns.set_style("white")
    sns.set_palette("colorblind")
    for explainer in rep_shift_dic:
        plt.plot(n_pixels_pert, rep_shift_dic[explainer].mean(axis=-1), label=explainer)
        plt.fill_between(n_pixels_pert, rep_shift_dic[explainer].mean(axis=-1) - rep_shift_dic[explainer].std(axis=-1),
                         rep_shift_dic[explainer].mean(axis=-1) + rep_shift_dic[explainer].std(axis=-1), alpha=0.4)
    plt.xlabel("Number of Perturbed Pixels")
    plt.ylabel("Latent Shift")
    plt.legend()
    plt.savefig(save_dir / "pixel_pert.pdf")


def consistency_examples(random_seed: int = 1, batch_size: int = 200, dim_latent: int = 4,
                         n_epochs: int = 100, subtrain_size: int = 1000) -> None:
    # Initialize seed and device
    torch.random.manual_seed(random_seed)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # Load MNIST
    W = 28  # Image width = height
    data_dir = Path.cwd() / "data/mnist"
    train_dataset = torchvision.datasets.MNIST(data_dir, train=True, download=True)
    test_dataset = torchvision.datasets.MNIST(data_dir, train=False, download=True)
    train_transform = transforms.Compose([transforms.ToTensor()])
    test_transform = transforms.Compose([transforms.ToTensor()])
    train_dataset.transform = train_transform
    test_dataset.transform = test_transform
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    X_train = train_dataset.data
    X_train = X_train.unsqueeze(1).float()
    X_test = test_dataset.data
    X_test = X_test.unsqueeze(1).float()

    # Initialize encoder, decoder and autoencoder wrapper
    pert = RandomNoise()
    encoder = EncoderMnist(encoded_space_dim=dim_latent)
    decoder = DecoderMnist(encoded_space_dim=dim_latent)
    autoencoder = AutoEncoderMnist(encoder, decoder, dim_latent, pert)
    encoder.to(device)
    decoder.to(device)
    autoencoder.to(device)

    # Train the denoising autoencoder
    save_dir = Path.cwd() / "results/mnist/consistency_examples"
    if not save_dir.exists():
        os.makedirs(save_dir)
    autoencoder.fit(device, train_loader, test_loader, save_dir, n_epochs, checkpoint_interval=10)
    autoencoder.load_state_dict(torch.load(save_dir / (autoencoder.name + ".pt")), strict=False)

    # Fitting explainers, computing the metric and saving everything
    autoencoder.train().cpu()
    mse_loss = torch.nn.MSELoss()
    explainer_list = [InfluenceFunctions(autoencoder, X_train, mse_loss),
                      TracIn(autoencoder, X_train, mse_loss),
                      SimplEx(autoencoder, X_train, mse_loss),
                      NearestNeighbours(autoencoder, X_train, mse_loss)]
    results_list = []
    idx_subtrain = [torch.nonzero(train_dataset.targets == (n % 10))[n // 10].item() for n in range(subtrain_size)]
    idx_subtest = [torch.nonzero(train_dataset.targets == (n % 10))[n // 10].item() for n in range(subtrain_size)]
    labels_subtrain = train_dataset.targets[idx_subtrain]
    labels_subtest = test_dataset.targets[idx_subtrain]
    for explainer in explainer_list:
        logging.info(f"Now fitting {explainer} exaplainer")
        attribution = explainer.attribute(X_test[idx_subtest], idx_subtrain, recursion_depth=100,
                                          learning_rate=autoencoder.lr)
        autoencoder.load_state_dict(torch.load(save_dir / (autoencoder.name + ".pt")), strict=False)
        similarity_rates = similarity_rate(attribution, labels_subtrain, labels_subtest)
        results_list += [[str(explainer), metric] for metric in similarity_rates]
    results_df = pd.DataFrame(results_list, columns=["Explainer", "Similarity Rate"])
    results_df.to_csv(save_dir/"metrics.csv")
    sns.boxplot(x="Explainer", y="Similarity Rate", data=results_df, palette="colorblind")
    plt.savefig(save_dir/"box_plots.pdf")


def pretext_task_sensitivity(random_seed: int = 1, batch_size: int = 300, n_runs: int = 5,
                             dim_latent: int = 4, n_epochs: int = 100, patience: int = 10,
                             subtrain_size: int = 100, n_plots: int = 10) -> None:
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
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    X_train = train_dataset.data
    X_train = X_train.unsqueeze(1).float()
    X_test = test_dataset.data
    X_test = X_test.unsqueeze(1).float()
    idx_subtrain = [torch.nonzero(train_dataset.targets == (n % 10))[n // 10].item() for n in range(subtrain_size)]

    # Create saving directory
    save_dir = Path.cwd() / "results/mnist/pretext"
    if not save_dir.exists():
        logging.info(f"Creating saving directory {save_dir}")
        os.makedirs(save_dir)

    # Define the computed metrics and create a csv file with appropriate headers
    pretext_list = [Identity(), RandomNoise(noise_level=0.3), Mask(mask_proportion=0.2)]
    headers = [str(pretext) for pretext in pretext_list] + ["Classification"]  # Name of each task
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
            name = f'{str(pretext)}-ae_run{run}'
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
            feature_importance.append(np.abs(np.expand_dims(attribute_auxiliary(encoder, test_loader, device,
                                                                          gradshap, baseline_image), 0)))
            # Compute example importance
            logging.info("Computing example importance")
            simplex = SimplEx(model.cpu(), X_train, mse_loss)
            example_importance.append(np.expand_dims(simplex.attribute(X_test, idx_subtrain).cpu().numpy(), 0))

        # Create and fit a MNIST classifier
        name = f'Classifier_run{run}'
        encoder = EncoderMnist(dim_latent)
        classifier = ClassifierMnist(encoder, dim_latent, name)
        logging.info(f"Now fitting {name}")
        classifier.fit(device, train_loader, test_loader, save_dir, n_epochs, patience)
        classifier.load_state_dict(torch.load(save_dir / (name + ".pt")), strict=False)
        baseline_image = torch.zeros((1, 1, 28, 28), device=device)
        # Compute feature importance for the classifier
        logging.info("Computing feature importance")
        gradshap = GradientShap(encoder)
        feature_importance.append(np.abs(np.expand_dims(attribute_auxiliary(encoder, test_loader, device,
                                                                      gradshap, baseline_image), 0)))
        # Compute example importance for the classifier
        logging.info("Computing example importance")
        simplex = SimplEx(classifier.cpu(), X_train, mse_loss)
        example_importance.append(np.expand_dims(simplex.attribute(X_test, idx_subtrain).cpu().numpy(), 0))

        # Compute correlation between the saliency of different pretext tasks
        feature_importance = np.concatenate(feature_importance)
        feature_pearson[run] = np.corrcoef(feature_importance.reshape((n_tasks, -1)))
        feature_spearman[run] = spearmanr(feature_importance.reshape((n_tasks, -1)), axis=1)[0]
        example_importance = np.concatenate(example_importance)
        example_pearson[run] = np.corrcoef(example_importance.reshape((n_tasks, -1)))
        example_spearman[run] = spearmanr(example_importance.reshape((n_tasks, -1)), axis=1)[0]
        logging.info(f'Run {run} complete \n Feature Pearson \n {np.round(feature_pearson[run], decimals=2)}'
                     f'\n Feature Spearman \n {np.round(feature_spearman[run], decimals=2)}'
                     f'\n Example Pearson \n {np.round(example_pearson[run], decimals=2)}'
                     f'\n Example Spearman \n {np.round(example_spearman[run], decimals=2)}')

        # Plot a couple of examples
        idx_plot = [torch.nonzero(test_dataset.targets == (n % 10))[n // 10].item() for n in range(n_plots)]
        test_images_to_plot = [X_test[i][0].numpy().reshape(W, W) for i in idx_plot]
        train_images_to_plot = [X_train[i][0].numpy().reshape(W, W) for i in idx_subtrain]
        fig_features = plot_pretext_saliencies(test_images_to_plot,
                                               feature_importance[:, idx_plot, :, :, :], headers)
        fig_features.savefig(save_dir / f"saliency_maps_run{run}.pdf")
        plt.close(fig_features)
        fig_examples = plot_pretext_top_example(train_images_to_plot, test_images_to_plot,
                                                example_importance[:, idx_plot, :], headers)
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
    with open(save_dir/'tables.tex', 'w') as f:
        for corr_avg, corr_std in zip(
                [feature_pearson_avg, feature_spearman_avg, example_pearson_avg, example_spearman_avg],
                [feature_pearson_std, feature_spearman_std, example_pearson_std, example_spearman_std]):
            f.write(correlation_latex_table(corr_avg, corr_std, headers))
            f.write("\n")


def disvae_feature_importance(random_seed: int = 1, batch_size: int = 300, n_plots: int = 20, n_runs: int = 5,
                              dim_latent: int = 3, n_epochs: int = 100, beta_list: list = [1, 5, 10]) -> None:
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
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Create saving directory
    save_dir = Path.cwd() / "results/mnist/vae"
    if not save_dir.exists():
        logging.info(f"Creating saving directory {save_dir}")
        os.makedirs(save_dir)

    # Define the computed metrics and create a csv file with appropriate headers
    loss_list = [BetaHLoss(), BtcvaeLoss(is_mss=False, n_data=len(train_dataset))]
    metric_list = [pearson_saliency, spearman_saliency, cos_saliency, entropy_saliency, count_activated_neurons]
    metric_names = ["Pearson Correlation", "Spearman Correlation", "Cosine", "Entropy", "Active Neurons"]
    headers = ["Loss Type", "Beta"] + metric_names
    csv_path = save_dir / "metrics.csv"
    if not csv_path.is_file():
        logging.info(f"Creating metrics csv in {csv_path}")
        with open(csv_path, 'w') as csv_file:
            dw = csv.DictWriter(csv_file, delimiter=',', fieldnames=headers)
            dw.writeheader()

    for beta, loss, run in itertools.product(beta_list, loss_list, range(1, n_runs+1)):
        # Initialize vaes
        encoder = EncoderBurgess(img_size, dim_latent)
        decoder = DecoderBurgess(img_size, dim_latent)
        loss.beta = beta
        name = f'{str(loss)}-vae_beta{beta}_run{run}'
        model = VAE(img_size, encoder, decoder, dim_latent, loss, name=name)
        logging.info(f"Now fitting {name}")
        model.fit(device, train_loader, test_loader, save_dir, n_epochs)
        model.load_state_dict(torch.load(save_dir / (name + ".pt")), strict=False)

        # Compute test-set saliency and associated metrics
        baseline_image = torch.zeros((1, 1, W, W), device=device)
        gradshap = GradientShap(encoder.mu)
        attributions = attribute_individual_dim(encoder.mu, dim_latent, test_loader, device, gradshap, baseline_image)
        metrics = compute_metrics(attributions, metric_list)
        results_str = '\t'.join([f'{metric_names[k]} {metrics[k]:.2g}' for k in range(len(metric_list))])
        logging.info(f"Model {name} \t {results_str}")

        # Save the metrics
        with open(csv_path, 'a', newline='') as csv_file:
            writer = csv.writer(csv_file, delimiter=',')
            writer.writerow([str(loss), beta] + metrics)

        # Plot a couple of examples
        plot_idx = [torch.nonzero(test_dataset.targets == (n % 10))[n // 10].item() for n in range(n_plots)]
        images_to_plot = [test_dataset[i][0].numpy().reshape(W, W) for i in plot_idx]
        fig = plot_vae_saliencies(images_to_plot,
                                  attributions[plot_idx])
        fig.savefig(save_dir / f"{name}.pdf")
        plt.close(fig)

    fig = vae_box_plots(pd.read_csv(csv_path), metric_names)
    fig.savefig(save_dir / 'metric_box_plots.pdf')
    plt.close(fig)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", type=str, default="disvae")
    parser.add_argument("-n", type=int, default=5)
    parser.add_argument("-b", type=int, default=300)
    parser.add_argument("-r", type=int, default=1)
    args = parser.parse_args()
    if args.e == "disvae":
        disvae_feature_importance(n_runs=args.n, batch_size=args.b, random_seed=args.r)
    elif args.e == "pretext":
        pretext_task_sensitivity(n_runs=args.n, batch_size=args.b, random_seed=args.r)
    elif args.e == "consfeatures":
        consistency_feature_importance(batch_size=args.b, random_seed=args.r)
    elif args.e == "consexamples":
        consistency_examples(batch_size=args.b, random_seed=args.r)
    else:
        raise ValueError("Invalid experiment name")

"""
def track_importance(random_seed: int = 1, batch_size: int = 200,
                     dim_latent: int = 50, n_epochs: int = 10,
                     pretrain_ratio: float = 0.9) -> None:
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
    pretrain_size = int(pretrain_ratio * len(train_dataset))
    pretrain_dataset, train_dataset = random_split(train_dataset, [pretrain_size, len(train_dataset) - pretrain_size])
    pretrain_loader = torch.utils.data.DataLoader(pretrain_dataset, batch_size=batch_size)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Initialize encoder and decoder
    encoder = EncoderMnist(encoded_space_dim=dim_latent)
    decoder = DecoderMnist(encoded_space_dim=dim_latent)
    encoder.to(device)
    decoder.to(device)

    # Compute the initial attribution
    grad_shap = GradientShap(forward_func=encoder)
    baseline_features = torch.zeros((1, 1, 28, 28), device=device)
    initial_attribution = attribute_auxiliary(encoder, test_loader, device, grad_shap, baseline_features)

    # Initialize pretrain optimizer
    loss_pretrain = torch.nn.MSELoss()
    params_pretrain = [
        {'params': encoder.parameters()},
        {'params': decoder.parameters()}
    ]
    optim_pretrain = torch.optim.Adam(params_pretrain, lr=1e-03, weight_decay=1e-05)

    # Train the denoising autoencoder
    print("\t Pretraining \t")
    pretrain_loss_hist = {'train_loss': [], 'val_loss': []}
    for epoch in range(n_epochs):
        train_loss = train_denoiser_epoch(encoder, decoder, device, pretrain_loader, loss_pretrain, optim_pretrain,
                                          noise_factor=0.3)
        val_loss = test_denoiser_epoch(encoder, decoder, device, test_loader, loss_pretrain)
        pretrain_loss_hist['train_loss'].append(train_loss)
        pretrain_loss_hist['val_loss'].append(val_loss)
        print(f'\n Epoch {epoch + 1}/{n_epochs} \t Train loss {train_loss:.3g} \t Val loss {val_loss:.3g} \t')

    # Compute the attribution after pretraining
    pretrain_attribution = attribute_auxiliary(encoder, test_loader, device, grad_shap, baseline_features)

    # Initialize classifier
    classifier = ClassifierMnist(encoder)
    classifier = classifier.to(device)

    # Initialize optimizer
    loss_train = torch.nn.CrossEntropyLoss()
    params_train = [
        {'params': classifier.parameters()},
    ]
    optim_train = torch.optim.Adam(params_train)

    # Train the classifier
    print("\t Training \t")
    train_loss_hist = {'train_loss': [], 'val_loss': []}
    for epoch in range(n_epochs):
        train_loss = train_classifier_epoch(classifier, device, train_loader, loss_train, optim_train)
        val_loss = test_classifier_epoch(classifier, device, test_loader, loss_train)
        train_loss_hist['train_loss'].append(train_loss)
        train_loss_hist['val_loss'].append(val_loss)
        print(f'\n Epoch {epoch + 1}/{n_epochs} \t Train loss {train_loss:.3g} \t Val loss {val_loss:.3g} \t')

    # Compute the attribution after training
    train_attribution = attribute_auxiliary(encoder, test_loader, device, grad_shap, baseline_features)

    # Compute the attribution deltas
    print(f'Initial vs Pretrained Attribution Delta: {np.mean(np.abs(pretrain_attribution - initial_attribution)):.3g}')
    print(f'Pretrained vs Trained Attribution Delta: {np.mean(np.abs(train_attribution - pretrain_attribution)):.3g}')

    '''
    ig_explainer = IntegratedGradients(auxiliary_encoder)
        dl_explainer = DeepLift(auxiliary_encoder)
        ig_attr_batch = ig_explainer.attribute(test_images, baselines=baseline_features)
        dl_attr_batch = dl_explainer.attribute(test_images, baselines=baseline_features).detach()
        ig_attribution[n_batch * batch_size:(n_batch * batch_size + len(test_images))] = ig_attr_batch.cpu().numpy()
        dl_attribution[n_batch * batch_size:(n_batch * batch_size + len(test_images))] = dl_attr_batch.cpu().numpy()
    
    #attribution_deltas.append(attribution_delta.data)
        #prev_attribution = current_attribution

    #attribution_delta = np.mean(np.abs(current_attribution - prev_attribution))
    
    #prev_attribution = np.zeros((len(test_dataset), 1, 28, 28))
        #current_attribution = np.zeros((len(test_dataset), 1, 28, 28))
    
    #plot_image_saliency(test_images[0], ig_attr_batch[0])
        #plot_image_saliency(test_images[0], dl_attr_batch[0])
        
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

"""
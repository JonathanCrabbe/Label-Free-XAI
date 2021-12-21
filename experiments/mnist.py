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
import csv
from captum.attr import GradientShap, DeepLift, IntegratedGradients, Saliency
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from scipy.stats import pearsonr, spearmanr
from explanations.features import AuxiliaryFunction, attribute_auxiliary, attribute_individual_dim
from models.images import EncoderMnist, DecoderMnist, ClassifierMnist, BetaVaeMnist, BetaTcVaeMnist, \
    train_denoiser_epoch, test_denoiser_epoch, train_classifier_epoch, test_classifier_epoch, VAE, EncoderBurgess, \
    DecoderBurgess
from models.losses import BetaHLoss, BtcvaeLoss
from utils.metrics import cos_saliency, entropy_saliency, \
    count_activated_neurons, correlation_saliency, compute_metrics
from utils.visualize import plot_vae_saliencies, vae_box_plots


def consistency_feature_importance(random_seed: int = 1, batch_size: int = 200,
                                   dim_latent: int = 4, n_epochs: int = 30) -> None:
    # Initialize seed and device
    torch.random.manual_seed(random_seed)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    n_pixels_pert = [1, 2, 3, 4, 5, 10, 20, 30, 50]

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
    baseline_features = torch.zeros((1, 1, 28, 28), device=device)

    for epoch in range(n_epochs):
        train_loss = train_denoiser_epoch(encoder, decoder, device, train_loader, loss_fn, optim)
        val_loss = test_denoiser_epoch(encoder, decoder, device, test_loader, loss_fn)
        loss_hist['train_loss'].append(train_loss)
        loss_hist['val_loss'].append(val_loss)

        print(f'\n Epoch {epoch + 1}/{n_epochs} \t Train loss {train_loss:.3g} \t Val loss {val_loss:.3g} \t ')

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

    for n_batch, (batch_images, _) in enumerate(test_loader):
        batch_images = batch_images.to(device)
        auxiliary_encoder = AuxiliaryFunction(encoder, batch_images)
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
                if explainer in explainer_dic.keys():
                    top_pixels = torch.topk(torch.abs(attr_batch).view(len(batch_images), -1), n_pixels)[1]
                    for k in range(n_pixels):
                        mask[:, 0, top_pixels[:, k] // 28, top_pixels[:, k] % 28] = 0
                    attr_dic[explainer][n_batch * batch_size:(n_batch * batch_size + len(batch_images))] = \
                        attr_batch.detach().cpu().numpy()
                elif explainer == "Random":
                    for batch_id in range(len(batch_images)):
                        top_pixels = torch.randperm(28 ** 2)[:n_pixels]
                        for k in range(n_pixels):
                            mask[batch_id, 0, top_pixels[k] // 28, top_pixels[k] % 28] = 0
                batch_images_pert = mask * batch_images
                representation_shift = torch.sqrt(
                    torch.sum((encoder(batch_images_pert) - encoder(batch_images)) ** 2, -1))
                rep_shift_dic[explainer][pert_id, n_batch] = torch.mean(representation_shift).cpu().detach().numpy()

    sns.set_style("white")
    sns.set_palette("colorblind")
    for explainer in rep_shift_dic:
        plt.plot(n_pixels_pert, rep_shift_dic[explainer].mean(axis=-1), label=explainer)
        plt.fill_between(n_pixels_pert, rep_shift_dic[explainer].mean(axis=-1) - rep_shift_dic[explainer].std(axis=-1),
                         rep_shift_dic[explainer].mean(axis=-1) + rep_shift_dic[explainer].std(axis=-1), alpha=0.4)
    plt.xlabel("Number of Pixels Perturbed")
    plt.ylabel("Latent Shift")
    plt.legend()
    results_dir = Path.cwd() / "results/mnist/consistency/"
    if not results_dir.exists():
        os.makedirs(results_dir)
    plt.savefig(results_dir / "pixel_pert.pdf")
    plt.show()


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
    encoder = EncoderMnist(encoded_space_dim=dim_latent, fc2_input_dim=128)
    decoder = DecoderMnist(encoded_space_dim=dim_latent, fc2_input_dim=128)
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


def vae_feature_importance(random_seed: int = 1, batch_size: int = 200,
                           dim_latent: int = 2, n_epochs: int = 1, beta_list: list = [1, 2, 5, 10]) -> None:
    # Initialize seed and device
    torch.random.manual_seed(random_seed)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model_names = [r"$\beta$-VAE", r"$\beta$-TCVAE"]

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

    for beta in beta_list:
        logging.info(f"Now working with beta = {beta}...")

        # Initialize vaes
        vae = BetaVaeMnist(dim_latent, beta).cpu()
        tcvae = BetaTcVaeMnist(dim_latent, beta).cpu()
        for id_model, model in enumerate([vae, tcvae]):
            logging.info(f"Now fitting model {model_names[id_model]}...")
            model.fit(device, train_loader, test_loader, n_epochs)

            test_images = next(iter(test_loader))[0].to(device)
            test_image = test_images[1:2]
            latent_rep = model.encoder.mu(test_image).detach().cpu().numpy().reshape(dim_latent)
            baseline_image = torch.zeros(test_image.shape, device=device)
            gradshap = GradientShap(model.encoder.mu)
            attributions = []
            W = 28
            cblind_palette = sns.color_palette("colorblind")
            fig = plt.figure(figsize=(10, 10))

            for image_batch, _ in test_loader:
                image_batch = image_batch.to(device)
                attributions_batch = []
                for dim in range(dim_latent):
                    attribution = gradshap.attribute(image_batch, baseline_image, target=dim).detach().cpu().numpy()
                    attributions_batch.append(np.reshape(attribution, (len(image_batch), 1, W, W)))
                attributions.append(np.concatenate(attributions_batch, axis=1))
            attributions = np.abs(np.concatenate(attributions))
            corr = spearmanr(attributions[:, 0, :, :].flatten(),
                             attributions[:, 1, :, :].flatten())[0]

            logging.info(f"Model {model_names[id_model]} \t Beta {beta} \t Saliency Correlation {corr:.2g} ")

            for dim in range(dim_latent):
                attribution = gradshap.attribute(test_image, baseline_image, target=dim).detach().cpu().numpy()
                saliency = np.abs(latent_rep[dim] * attribution)
                ax = fig.add_subplot(3, 3, dim + 1)
                h = sns.heatmap(np.reshape(saliency, (W, W)), linewidth=0, xticklabels=False, yticklabels=False, ax=ax,
                                cmap=sns.dark_palette(cblind_palette[dim], as_cmap=True), cbar=True, alpha=0.7,
                                zorder=2)
                h.imshow(np.reshape(test_image.cpu().numpy(), (W, W)), zorder=1,
                         cmap=sns.color_palette("dark:white", as_cmap=True))
            # plt.show()


def disvae_feature_importance(random_seed: int = 1, batch_size: int = 300, n_plots: int = 20, n_runs: int = 10,
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
    metric_list = [correlation_saliency, cos_saliency, entropy_saliency, count_activated_neurons]
    metric_names = ["Correlation", "Cosine", "Entropy", "Active Neurons"]
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
        fig = plot_vae_saliencies(attributions[plot_idx])
        fig.savefig(save_dir / f"{name}.pdf")

    fig = vae_box_plots(pd.read_csv(csv_path), metric_names)
    fig.savefig(save_dir / 'metric_box_plots.pdf')


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    disvae_feature_importance()

"""
        attributions = []
        latents = []
        for image_batch, _ in test_loader:
            image_batch = image_batch.to(device)
            attributions_batch = []
            latents.append(encoder.mu(image_batch).detach().cpu().numpy())
            for dim in range(dim_latent):
                attribution = gradshap.attribute(image_batch, baseline_image, target=dim).detach().cpu().numpy()
                attributions_batch.append(np.reshape(attribution, (len(image_batch), 1, W, W)))
            attributions.append(np.concatenate(attributions_batch, axis=1))
        latents = np.concatenate(latents)
        attributions = np.concatenate(attributions)
        attributions = np.abs(np.expand_dims(latents, (2, 3)) * attributions)
        
        
        
        cblind_palette = sns.color_palette("colorblind")
        fig, axs = plt.subplots(ncols=dim_latent, nrows=n_plots, figsize=(4 * dim_latent, 4 * n_plots))
        for example_id in range(n_plots):
            test_id = torch.nonzero(test_dataset.targets == (example_id % 10))[example_id // 10]
            max_saliency = np.max(attributions[test_id])
            for dim in range(dim_latent):
                sub_saliency = attributions[test_id, dim, :, :]
                ax = axs[example_id, dim]
                h = sns.heatmap(np.reshape(sub_saliency, (W, W)), linewidth=0, xticklabels=False, yticklabels=False,
                                ax=ax, cmap=sns.light_palette(cblind_palette[dim], as_cmap=True), cbar=True,
                                alpha=1, zorder=2, vmin=0, vmax=max_saliency)
"""

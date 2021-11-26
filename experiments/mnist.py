import torch
import torchvision
import numpy as np
from torchvision import transforms
from torch.utils.data import DataLoader, random_split
from models.images import EncoderMnist, DecoderMnist, train_epoch, test_epoch
from pathlib import Path
from captum.attr import GradientShap, DeepLift, IntegratedGradients, Saliency
from explanations.features import AuxiliaryFunction


def denoiser_mnist(random_seed: int = 0, batch_size: int = 200, dim_latent: int = 4, n_epochs: int = 5):
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
    attribution_deltas = []
    baseline_features = torch.zeros((1, 1, 28, 28), device=device)

    for epoch in range(n_epochs):
        train_loss = train_epoch(encoder, decoder, device, train_loader, loss_fn, optim)
        val_loss = test_epoch(encoder, decoder, device, test_loader, loss_fn)
        loss_hist['train_loss'].append(train_loss)
        loss_hist['val_loss'].append(val_loss)

        print(f'\n Epoch {epoch + 1}/{n_epochs} \t train loss {train_loss:.3g} \t val loss {val_loss:.3g} \t ')


    attr_dic = {'Gradient Shap': np.zeros((len(test_dataset), 1, 28, 28)),
                'Integrated Gradients': np.zeros((len(test_dataset), 1, 28, 28)),
                'DeepLift': np.zeros((len(test_dataset), 1, 28, 28)),
                'Saliency': np.zeros((len(test_dataset), 1, 28, 28))
                }
    rep_shift_dic = {'Gradient Shap': np.zeros((len(n_pixels_pert), len(test_loader))),
                     'Integrated Gradients': np.zeros((len(n_pixels_pert), len(test_loader))),
                     'DeepLift': np.zeros((len(n_pixels_pert), len(test_loader))),
                     'Saliency': np.zeros((len(n_pixels_pert), len(test_loader)))}

    for n_batch, (batch_images, _) in enumerate(test_loader):
        batch_images = batch_images.to(device)
        auxiliary_encoder = AuxiliaryFunction(encoder, batch_images)
        explainer_dic = {'Gradient Shap': GradientShap(auxiliary_encoder),
                         'Integrated Gradients': IntegratedGradients(auxiliary_encoder),
                         'DeepLift': DeepLift(auxiliary_encoder),
                         'Saliency': Saliency(auxiliary_encoder)
                         }
        for explainer in explainer_dic:
            if explainer in ["Gradient Shap", "Integrated Gradients", "DeepLift"]:
                attr_batch = explainer_dic[explainer].attribute(batch_images, baseline_features)
            else:
                attr_batch = explainer_dic[explainer].attribute(batch_images)
            for pert_id, n_pixels in enumerate(n_pixels_pert):
                top_pixels = torch.topk(torch.abs(attr_batch).view(len(batch_images), -1), n_pixels)[1]
                mask = torch.ones(batch_images.shape, device=device)
                for k in range(n_pixels):
                    mask[:, 0, top_pixels[:, k] // 28, top_pixels[:, k] % 28] = 0
                batch_images_pert = mask*batch_images
                representation_shift = torch.sqrt(torch.sum((encoder(batch_images_pert)-encoder(batch_images))**2, -1))
                rep_shift_dic[explainer][pert_id, n_batch] = torch.mean(representation_shift).cpu().detach().numpy()
            attr_dic[explainer][n_batch * batch_size:(n_batch * batch_size + len(batch_images))] = attr_batch.detach().cpu().numpy()
    print(rep_shift_dic["Integrated Gradients"].mean(axis=-1))
    print(rep_shift_dic["Gradient Shap"].mean(axis=-1))
    print(rep_shift_dic["DeepLift"].mean(axis=-1))
    print(rep_shift_dic["Saliency"].mean(axis=-1))


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


if __name__ == "__main__":
    denoiser_mnist()

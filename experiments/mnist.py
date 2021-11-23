import torch
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader, random_split
from models.images import EncoderMnist, DecoderMnist, train_epoch, test_epoch
from pathlib import Path
from captum.attr import IntegratedGradients
from captum.attr._utils.visualization import visualize_image_attr
from explanations.features import AuxiliaryFunction


def denoiser_mnist():
    data_dir = Path.cwd()/"data/mnist"

    train_dataset = torchvision.datasets.MNIST(data_dir, train=True, download=True)
    test_dataset = torchvision.datasets.MNIST(data_dir, train=False, download=True)

    train_transform = transforms.Compose([
    transforms.ToTensor(),
    ])

    test_transform = transforms.Compose([
    transforms.ToTensor(),
    ])

    train_dataset.transform = train_transform
    test_dataset.transform = test_transform

    m=len(train_dataset)

    train_data, val_data = random_split(train_dataset, [int(m-m*0.2), int(m*0.2)])
    batch_size=256

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size)
    valid_loader = torch.utils.data.DataLoader(val_data, batch_size=batch_size)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size,shuffle=True)


    ### Define the loss function
    loss_fn = torch.nn.MSELoss()

    ### Define an optimizer (both for the encoder and the decoder!)
    lr= 0.001

    ### Set the random seed for reproducible results
    torch.manual_seed(0)

    ### Initialize the two networks
    d = 4

    #model = Autoencoder(encoded_space_dim=encoded_space_dim)
    encoder = EncoderMnist(encoded_space_dim=d, fc2_input_dim=128)
    decoder = DecoderMnist(encoded_space_dim=d, fc2_input_dim=128)
    params_to_optimize = [
        {'params': encoder.parameters()},
        {'params': decoder.parameters()}
    ]

    optim = torch.optim.Adam(params_to_optimize, lr=lr, weight_decay=1e-05)

    # Check if the GPU is available
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(f'Selected device: {device}')

    # Move both the encoder and the decoder to the selected device
    encoder.to(device)
    decoder.to(device)

    num_epochs = 1
    diz_loss = {'train_loss':[],'val_loss':[]}
    for epoch in range(num_epochs):
       train_loss =train_epoch(encoder, decoder, device, train_loader, loss_fn, optim)
       val_loss = test_epoch(encoder, decoder, device, test_loader, loss_fn)
       print('\n EPOCH {}/{} \t train loss {} \t val loss {}'.format(epoch + 1, num_epochs,train_loss,val_loss))
       diz_loss['train_loss'].append(train_loss)
       diz_loss['val_loss'].append(val_loss)

    test_images, _ = next(iter(test_loader))
    test_image = test_images[10:11].to(device)
    auxiliary_encoder = AuxiliaryFunction(encoder, test_image)
    ig_explainer = IntegratedGradients(auxiliary_encoder)
    baseline_image = torch.zeros(test_image.shape, device=device)
    ig_attr = ig_explainer.attribute(test_image, baseline_image)

    visualize_image_attr(torch.permute(ig_attr[0], (1, 2, 0)).cpu().numpy(),
                         torch.permute(test_image[0], (1, 2, 0)).cpu().numpy(),
                         "blended_heat_map")


if __name__ == "__main__":
    denoiser_mnist()
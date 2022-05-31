import torch
import numpy as np
from captum.attr import Attribution, Saliency
from torch.nn import Module


class AuxiliaryFunction(Module):
    def __init__(self, black_box: Module, base_features: torch.Tensor) -> None:
        super().__init__()
        self.black_box = black_box
        self.base_features = base_features
        self.prediction = black_box(base_features)

    def forward(self, input_features: torch.Tensor) -> torch.Tensor:
        if len(self.prediction) == len(input_features):
            return torch.sum(self.prediction * self.black_box(input_features), dim=-1)
        elif len(input_features) % len(self.prediction) == 0:
            n_repeat = int(len(input_features) / len(self.prediction))
            return torch.sum(self.prediction.repeat(n_repeat, 1) * self.black_box(input_features), dim=-1)
        else:
            raise ValueError("The internal batch size should be a multiple of input_features.shape[0]")


def attribute_individual_dim(encoder: callable,  dim_latent: int, data_loader: torch.utils.data.DataLoader,
                             device: torch.device, attr_method: Attribution, baseline: torch.Tensor) -> np.ndarray:
    attributions = []
    latents = []
    for input_batch, _ in data_loader:
        input_batch = input_batch.to(device)
        attributions_batch = []
        latents.append(encoder(input_batch).detach().cpu().numpy())
        for dim in range(dim_latent):
            attribution = attr_method.attribute(input_batch, baseline, target=dim).detach().cpu().numpy()
            attributions_batch.append(attribution)
        attributions.append(np.concatenate(attributions_batch, axis=1))
    latents = np.concatenate(latents)
    attributions = np.concatenate(attributions)
    attributions = np.abs(np.expand_dims(latents, (2, 3)) * attributions)
    return attributions


def attribute_auxiliary(encoder: Module, data_loader: torch.utils.data.DataLoader,
                        device: torch.device, attr_method: Attribution, baseline=None) -> np.ndarray:
    attributions = []
    for inputs, _ in data_loader:
        inputs = inputs.to(device)
        auxiliary_encoder = AuxiliaryFunction(encoder, inputs)
        attr_method.forward_func = auxiliary_encoder
        if isinstance(attr_method, Saliency):
            attributions.append(attr_method.attribute(inputs).detach().cpu().numpy())
        else:
            if isinstance(baseline, torch.Tensor):
                attributions.append(attr_method.attribute(inputs, baseline).detach().cpu().numpy())
            elif isinstance(baseline, Module):
                baseline_inputs = baseline(inputs)
                attributions.append(attr_method.attribute(inputs, baseline_inputs).detach().cpu().numpy())
            else:
                attributions.append(attr_method.attribute(inputs).detach().cpu().numpy())
    return np.concatenate(attributions)
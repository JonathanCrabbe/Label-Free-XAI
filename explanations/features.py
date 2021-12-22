import torch
import numpy as np
from captum.attr import Attribution
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


class IntegratedGradients(Attribution):
    def __init__(self, forward_func: Module):
        super().__init__(forward_func=forward_func)

    def attribute(self, input: torch.Tensor, baselines: torch.Tensor, n_steps: int = 50) -> torch.Tensor:
        latent_reps = self.forward_func(input).detach()
        grads = torch.zeros(input.shape, device=input.device)
        x_input = input.clone().requires_grad_()
        for step in range(n_steps):
            weight = step/n_steps
            self.forward_func((1-weight)*baselines + weight*x_input).backward(gradient=latent_reps)
            grads += x_input.grad
            x_input.grad.data.zero_()
        int_grads = (input-baselines)*grads/n_steps
        return int_grads.cpu()


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
        if baseline is not None:
            attributions.append(attr_method.attribute(inputs, baseline).detach().cpu().numpy())
        else:
            attributions.append(attr_method.attribute(inputs).detach().cpu().numpy())
    return np.concatenate(attributions)
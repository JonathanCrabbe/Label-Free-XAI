import torch
from captum.attr import Attribution
from torch.nn import Module


class AuxiliaryFunction(Module):
    def __init__(self, black_box: Module, base_features: torch.Tensor) -> None:
        super().__init__()
        self.black_box = black_box
        self.base_features = base_features
        self.prediction = black_box(base_features)

    def forward(self, input_features: torch.Tensor):
        return torch.sum(self.prediction * self.black_box(input_features), dim=-1)


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




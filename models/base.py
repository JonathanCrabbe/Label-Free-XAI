import torch


class BlackBox(torch.nn.Module):
    def latent(self, x: torch.tensor) -> torch.tensor:
        pass

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


class FeatureImportance:
    def __init__(self, attr_method: Attribution, black_box: Module):
        self.attr_method = attr_method
        self.black_box = black_box

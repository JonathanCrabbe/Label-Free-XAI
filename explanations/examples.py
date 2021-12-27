import numpy as np
import torch
import torch.nn as nn
from utils.influence import hessian_vector_product, stack_torch_tensors


class InfluenceFunctions:
    def __init__(self, model: nn.Module, X_train: torch.Tensor):
        self.model = model
        self.X_train = X_train

    def attribute(self, X_test: torch.Tensor, train_idx: list, batch_size: int = 300,
                  damp: float = 1e-3, scale: float = 1000, recursion_depth: int =1000) -> list:
        """
        Code adapted from https://github.com/ahmedmalaa/torch-influence-functions/
        This function applies the stochastic estimation approach to evaluating influence function based on the power-series
        approximation of matrix inversion. Recall that the exact inverse Hessian H^-1 can be computed as follows:
        H^-1 = \sum^\infty_{i=0} (I - H) ^ j
        This series converges if all the eigen values of H are less than 1.

        Returns:
            return_grads: list of torch tensors, contains product of Hessian and v.
        """

        SUBSAMPLES = batch_size
        NUM_SAMPLES = model.X.shape[0]

        loss = [self.model.loss_fn(self.X_train[idx], self.model.predict(self.X_train[train_idx]))
                for idx in train_idx]

        grads = [stack_torch_tensors(torch.autograd.grad(loss[k], self.model.parameters(), create_graph=True)) for k in
                 range(len(train_idx))]

        IHVP_ = [grads[k].clone().detach() for k in range(len(train_idx))]

        for j in range(recursion_depth):
            sampled_idx = np.random.choice(list(range(NUM_SAMPLES)), SUBSAMPLES)[0]
            sampled_loss = model.loss_fn(self.X_train[sampled_idx], self.model.predict(self.X_train[sampled_idx]))
            IHVP_prev = [IHVP_[k].clone().detach() for k in range(len(train_idx))]
            hvps_ = [stack_torch_tensors(hessian_vector_product(sampled_loss, self.model, [IHVP_prev[k]])) for k in
                     range(len(train_idx))]

            IHVP_ = [g_ + (1 - damp) * ihvp_ - hvp_ / scale for (g_, ihvp_, hvp_) in zip(grads, IHVP_prev, hvps_)]

        return [IHVP_[k] / (scale * NUM_SAMPLES) for k in range(len(train_idx))]
    
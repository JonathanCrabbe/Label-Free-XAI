import numpy as np
import torch
import torch.nn as nn
import logging
from utils.influence import hessian_vector_product, stack_torch_tensors


class InfluenceFunctions:
    def __init__(self, model: nn.Module, X_train: torch.Tensor, loss_f: callable):
        self.model = model
        self.X_train = X_train
        self.loss_f = loss_f

    def attribute(self, X_test: torch.Tensor, train_idx: list, batch_size: int = 300,
                  damp: float = 1e-3, scale: float = 1000, recursion_depth: int = 1000) -> list:
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
        NUM_SAMPLES = self.X_train.shape[0]

        loss = [self.loss_f(self.X_train[idx:idx+1], self.model(self.X_train[idx:idx+1])) for idx in train_idx]

        grads = [stack_torch_tensors(torch.autograd.grad(loss[k], self.model.parameters(), create_graph=True)) for k in
                 range(len(train_idx))]

        IHVP_ = [grads[k].clone().detach() for k in range(len(train_idx))]

        for j in range(recursion_depth):
            sampled_idx = np.random.choice(list(range(NUM_SAMPLES)), SUBSAMPLES)[0]
            sampled_loss = model.loss_fn(self.X_train[sampled_idx], self.model(self.X_train[sampled_idx]))
            IHVP_prev = [IHVP_[k].clone().detach() for k in range(len(train_idx))]
            hvps_ = [stack_torch_tensors(hessian_vector_product(sampled_loss, self.model, [IHVP_prev[k]])) for k in
                     range(len(train_idx))]

            IHVP_ = [g_ + (1 - damp) * ihvp_ - hvp_ / scale for (g_, ihvp_, hvp_) in zip(grads, IHVP_prev, hvps_)]

        IHVP_ = [IHVP_[k] / (scale * NUM_SAMPLES) for k in range(len(train_idx))] # Rescale Hessian-Vector products
        IHVP_ = torch.stack(IHVP_, dim=0)  # Make a tensor of shape (len(train_idx), n_params)

        test_loss = [self.model.loss_fn(X_test[idx:idx+1], self.model(X_test[idx:idx+1])) for idx in range(len(X_test))]
        test_grads = [stack_torch_tensors(torch.autograd.grad(test_loss[k], self.model.parameters(), create_graph=True))
                      for k in range(len(X_test))]
        test_grads = torch.stack(test_grads, dim=0)
        logging.info(test_grads.shape)
        IF = torch.zeros(0)
        return IF

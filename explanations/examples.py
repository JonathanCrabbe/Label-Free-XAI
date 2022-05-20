import abc
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
import datetime
from tqdm import tqdm
from pathlib import Path
from utils.influence import hessian_vector_product, stack_torch_tensors, save_json, display_progress, grad_z, s_test
from torch.utils.data import DataLoader
from abc import ABC



class ExampleBasedExplainer(ABC):
    def __init__(self, model: nn.Module, X_train: torch.Tensor, loss_f: callable, **kwargs):
        self.model = model
        self.X_train = X_train
        self.loss_f = loss_f

    @abc.abstractmethod
    def attribute(self, X_test: torch.Tensor, train_idx: list, **kwargs) -> torch.Tensor:
        """

        Args:
            X_test:
            train_idx:
            **kwargs:

        Returns:

        """

    @abc.abstractmethod
    def __str__(self):
        """

        Returns: The name of the method

        """


class InfluenceFunctions(ExampleBasedExplainer, ABC):
    def __init__(self, model: nn.Module, X_train: torch.Tensor, loss_f: callable):
        super().__init__(model, X_train, loss_f)

    def attribute(self, X_test: torch.Tensor, train_idx: list, batch_size: int = 1, damp: float = 1e-3,
                  scale: float = 1000, recursion_depth: int = 100, **kwargs) -> torch.Tensor:
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

        loss = [self.loss_f(self.X_train[idx:idx+1], self.model(self.X_train[idx:idx+1]))
                for idx in train_idx]

        grads = [stack_torch_tensors(torch.autograd.grad(loss[k], self.model.encoder.parameters(),
                                                         create_graph=True)) for k in range(len(train_idx))]

        IHVP_ = [grads[k].detach().cpu().clone() for k in range(len(train_idx))]

        for _ in tqdm(range(recursion_depth), unit="recursion", leave=False):
            sampled_idx = np.random.choice(list(range(NUM_SAMPLES)), SUBSAMPLES)
            sampled_loss = self.loss_f(self.X_train[sampled_idx],
                                       self.model(self.X_train[sampled_idx]))
            IHVP_prev = [IHVP_[k].detach().clone() for k in range(len(train_idx))]
            hvps_ = [stack_torch_tensors(hessian_vector_product(sampled_loss, self.model, [IHVP_prev[k]]))
                     for k in tqdm(range(len(train_idx)), unit="example", leave=False)]
            IHVP_ = [g_ + (1 - damp) * ihvp_ - hvp_ / scale for (g_, ihvp_, hvp_) in zip(grads, IHVP_prev, hvps_)]

        IHVP_ = [IHVP_[k] / (scale * NUM_SAMPLES) for k in range(len(train_idx))]  # Rescale Hessian-Vector products
        IHVP_ = torch.stack(IHVP_, dim=0).reshape((len(train_idx), -1))  # Make a tensor (len(train_idx), n_params)

        test_loss = [self.loss_f(X_test[idx:idx+1], self.model(X_test[idx:idx+1]))
                     for idx in range(len(X_test))]
        test_grads = [stack_torch_tensors(torch.autograd.grad(test_loss[k], self.model.encoder.parameters(),
                                                              create_graph=True)) for k in range(len(X_test))]
        test_grads = torch.stack(test_grads, dim=0).reshape((len(X_test), -1))
        attribution = torch.einsum('ab,cb->ac', test_grads, IHVP_)
        return attribution

    def __str__(self):
        return "Influence Functions"


class TracIn(ExampleBasedExplainer, ABC):
    def __init__(self, model: nn.Module, X_train: torch.Tensor, loss_f: callable):
        super().__init__(model, X_train, loss_f)

    def attribute(self, X_test: torch.Tensor, train_idx: list, learning_rate: float = 1, **kwargs) -> torch.Tensor:
        attribution = torch.zeros(len(X_test), len(train_idx))
        for checkpoint_file in self.model.checkpoints_files:
            self.model.load_state_dict(torch.load(checkpoint_file), strict=False)
            train_loss = [self.loss_f(self.X_train[idx:idx + 1], self.model(self.X_train[idx:idx + 1]))
                          for idx in train_idx]
            train_grads = [stack_torch_tensors(torch.autograd.grad(train_loss[k], self.model.encoder.parameters(),
                                               create_graph=True)) for k in range(len(train_idx))]
            train_grads = torch.stack(train_grads, dim=0).reshape((len(train_idx), -1))
            test_loss = [self.loss_f(X_test[idx:idx + 1], self.model(X_test[idx:idx + 1]))
                         for idx in range(len(X_test))]
            test_grads = [stack_torch_tensors(torch.autograd.grad(test_loss[k], self.model.encoder.parameters(),
                                                                  create_graph=True)) for k in range(len(X_test))]
            test_grads = torch.stack(test_grads, dim=0).reshape((len(X_test), -1))
            attribution += torch.einsum('ab,cb->ac', test_grads, train_grads)
        return learning_rate*attribution

    def __str__(self):
        return "TracIn"


class SimplEx(ExampleBasedExplainer, ABC):
    def __init__(self, model: nn.Module, X_train: torch.Tensor = None, loss_f: callable = None):
        super().__init__(model, X_train, loss_f)

    def attribute(self, X_test: torch.Tensor, train_idx: list, learning_rate: float = 1,
                  batch_size: int = 50, **kwargs) -> torch.Tensor:
        attribution = torch.zeros(len(X_test), len(train_idx))
        train_representations = self.model.encoder(self.X_train[train_idx]).detach()
        n_batch = int(len(X_test)/batch_size)
        for n in tqdm(range(n_batch), unit="batch", leave=False):
            batch_features = X_test[n*batch_size:(n+1)*batch_size]
            batch_representations = self.model.encoder(batch_features).detach()
            attribution[n*batch_size:(n+1)*batch_size] = self.compute_weights(batch_representations,
                                                                              train_representations)
        return attribution

    def attribute_loader(self, device: torch.device, train_loader: DataLoader, test_loader: DataLoader,
                         batch_size: int = 50) -> torch.Tensor:
        H_train = []
        for x_train, _ in train_loader:
            H_train.append(self.model.encoder(x_train.to(device)).detach().cpu())
        H_train = torch.cat(H_train)
        H_test = []
        for x_test, _ in test_loader:
            H_test.append(self.model.encoder(x_test.to(device)).detach().cpu())
        H_test = torch.cat(H_test)
        attribution = torch.zeros(len(H_test), len(H_train))
        n_batch = int(len(H_test) / batch_size)
        for n in tqdm(range(n_batch), unit="batch", leave=False):
            h_test = H_test[n*batch_size:(n+1)*batch_size]
            attribution[n*batch_size:(n+1)*batch_size] = self.compute_weights(h_test, H_train)
        return attribution

    @staticmethod
    def compute_weights(batch_representations: torch.Tensor, train_representations: torch.Tensor,
                        n_epoch: int = 1000) -> torch.Tensor:
        preweights = torch.zeros((len(batch_representations), len(train_representations)), requires_grad=True)
        optimizer = torch.optim.Adam([preweights])
        for epoch in range(n_epoch):
            optimizer.zero_grad()
            weights = F.softmax(preweights, dim=-1)
            approx_representations = torch.einsum('ij,jk->ik', weights, train_representations)
            error = ((approx_representations - batch_representations) ** 2).sum()
            error.backward()
            optimizer.step()
        return torch.softmax(preweights, dim=-1).detach()

    def __str__(self):
        return "SimplEx"


class NearestNeighbours(ExampleBasedExplainer, ABC):
    def __init__(self, model: nn.Module, X_train: torch.Tensor = None, loss_f: callable = None):
        super().__init__(model, X_train, loss_f)

    def attribute(self, X_test: torch.Tensor, train_idx: list, batch_size: int = 500, **kwargs) -> torch.Tensor:
        attribution = torch.zeros(len(X_test), len(train_idx))
        train_representations = self.model.encoder(self.X_train[train_idx]).detach().unsqueeze(0)
        n_batch = int(len(X_test)/batch_size)
        for n in tqdm(range(n_batch), unit="batch", leave=False):
            batch_features = X_test[n*batch_size:(n+1)*batch_size]
            batch_representations = self.model.encoder(batch_features).detach().unsqueeze(1)
            attribution[n*batch_size:(n+1)*batch_size] =\
                1/torch.sum((batch_representations-train_representations)**2, dim=-1)
        return attribution

    def attribute_loader(self, device: torch.device, train_loader: DataLoader, test_loader: DataLoader,
                         batch_size: int = 50) -> torch.Tensor:
        H_train = []
        for x_train, _ in train_loader:
            H_train.append(self.model.encoder(x_train.to(device)).detach().cpu())
        H_train = torch.cat(H_train)
        H_test = []
        for x_test, _ in test_loader:
            H_test.append(self.model.encoder(x_test.to(device)).detach().cpu())
        H_test = torch.cat(H_test)
        attribution = torch.zeros(len(H_test), len(H_train))
        n_batch = int(len(H_test) / batch_size)
        for n in tqdm(range(n_batch), unit="batch", leave=False):
            h_test = H_test[n*batch_size:(n+1)*batch_size]
            attribution[n*batch_size:(n+1)*batch_size] = \
                1/torch.sum((h_test.unsqueeze(1) - H_train.unsqueeze(0))**2, dim=-1)
        return attribution

    def __str__(self):
        return "DKNN"


class InfluenceFunction:
    def __init__(self, model: nn.Module):
        self.model = model

    def calc_s_test_single(self, z_test, t_test, train_loader, gpu=-1,
                           damp=0.01, scale=25, recursion_depth=5000, r=1):
        """Calculates s_test for a single test image taking into account the whole
        training dataset. s_test = invHessian * nabla(Loss(test_img, model params))
        Arguments:
            z_test: test image
            t_test: test image label
            train_loader: pytorch dataloader, which can load the train data
            gpu: int, device id to use for GPU, -1 for CPU (default)
            damp: float, influence function damping factor
            scale: float, influence calculation scaling factor
            recursion_depth: int, number of recursions to perform during s_test
                calculation, increases accuracy. r*recursion_depth should equal the
                training dataset size.
            r: int, number of iterations of which to take the avg.
                of the h_estimate calculation; r*recursion_depth should equal the
                training dataset size.
        Returns:
            s_test_vec: torch tensor, contains s_test for a single test image"""
        s_test_vec_list = []
        for i in range(r):
            s_test_vec_list.append(s_test(z_test, t_test, self.model, train_loader,
                                          gpu=gpu, damp=damp, scale=scale,
                                          recursion_depth=recursion_depth))
            display_progress("Averaging r-times: ", i, r)

        s_test_vec = s_test_vec_list[0]
        for i in range(1, r):
            s_test_vec += s_test_vec_list[i]

        s_test_vec = [i / r for i in s_test_vec]

        return s_test_vec

    def calc_influence_single(self, train_loader, test_loader, test_id_num, gpu,
                              recursion_depth, r, s_test_vec=None,
                              time_logging=False):
        """Calculates the influences of all training data points on a single
        test dataset image.
        Arugments:
            train_loader: DataLoader, loads the training dataset
            test_loader: DataLoader, loads the test dataset
            test_id_num: int, id of the test sample for which to calculate the
                influence function
            gpu: int, identifies the gpu id, -1 for cpu
            recursion_depth: int, number of recursions to perform during s_test
                calculation, increases accuracy. r*recursion_depth should equal the
                training dataset size.
            r: int, number of iterations of which to take the avg.
                of the h_estimate calculation; r*recursion_depth should equal the
                training dataset size.
            s_test_vec: list of torch tensor, contains s_test vectors. If left
                empty it will also be calculated
        Returns:
            influence: list of float, influences of all training data samples
                for one test sample
            harmful: list of float, influences sorted by harmfulness
            helpful: list of float, influences sorted by helpfulness
            test_id_num: int, the number of the test dataset point
                the influence was calculated for"""
        # Calculate s_test vectors if not provided
        if not s_test_vec:
            z_test, t_test = test_loader.dataset[test_id_num]
            z_test = test_loader.collate_fn([z_test])
            t_test = test_loader.collate_fn([t_test])
            s_test_vec = self.calc_s_test_single(z_test, z_test, train_loader, gpu,
                                                 recursion_depth=recursion_depth, r=r)

        # Calculate the influence function
        train_dataset_size = len(train_loader.dataset)
        influences = []
        for i in range(train_dataset_size):
            z, _ = train_loader.dataset[i]
            z = train_loader.collate_fn([z])
            if time_logging:
                time_a = datetime.datetime.now()
            grad_z_vec = grad_z(z, z, self.model, gpu=gpu)
            if time_logging:
                time_b = datetime.datetime.now()
                time_delta = time_b - time_a
                logging.info(f"Time for grad_z iter:"
                             f" {time_delta.total_seconds() * 1000}")
            tmp_influence = -sum(
                [
                    ####################
                    # TODO: potential bottle neck, takes 17% execution time
                    # torch.sum(k * j).data.cpu().numpy()
                    ####################
                    torch.sum(k * j).data
                    for k, j in zip(grad_z_vec, s_test_vec)
                ]) / train_dataset_size
            influences.append(tmp_influence.cpu())
            display_progress("Calc. influence function: ", i, train_dataset_size)

        harmful = np.argsort(influences)
        helpful = harmful[::-1]

        return influences, harmful.tolist(), helpful.tolist(), test_id_num

    def attribute(self, device, train_loader, test_loader, outdir: Path,
                  recursion_depth: int = 1000, r: int = 1):

        for input_test, _ in test_loader:
            input_test = input_test.to(device)
            s_test(device, input_test, self.model, train_loader, recursion_depth=recursion_depth)

        """Calculates the influence function one test point at a time. Calcualtes
        the `s_test` and `grad_z` values on the fly and discards them afterwards.
        Arguments:
            recursion_depth:
            config: dict, contains the configuration from cli params

        outdir.mkdir(exist_ok=True, parents=True)

        # If calculating the influence for a subset of the whole dataset,
        # calculate it evenly for the same number of samples from all classes.
        # `test_start_index` is `False` when it hasn't been set by the user. It can
        # also be set to `0`.

        test_dataset_iter_len = len(test_loader.dataset)


        influences = {}
        # Main loop for calculating the influence function one test sample per
        # iteration.
        for j in range(test_dataset_iter_len):
            i = j

            start_time = time.time()
            influence, harmful, helpful, _ = self.calc_influence_single(
                train_loader, test_loader, test_id_num=i, gpu=gpu,
                recursion_depth=recursion_depth, r=r)
            end_time = time.time()

            ###########
            # Different from `influence` above
            ###########
            influences[str(i)] = {}
            _, label = test_loader.dataset[i]
            influences[str(i)]['label'] = label
            influences[str(i)]['num_in_dataset'] = j
            influences[str(i)]['time_calc_influence_s'] = end_time - start_time
            infl = [x.cpu().numpy().tolist() for x in influence]
            influences[str(i)]['influence'] = infl
            influences[str(i)]['harmful'] = harmful[:500]
            influences[str(i)]['helpful'] = helpful[:500]

            tmp_influences_path = outdir.joinpath(f"influence_results_tmp_"
                                                  f"_last-i_{i}.json")
            save_json(influences, tmp_influences_path)
            display_progress("Test samples processed: ", j, test_dataset_iter_len)

            logging.info(f"The results for this run are:")
            logging.info("Influences: ")
            logging.info(influence[:3])
            logging.info("Most harmful img IDs: ")
            logging.info(harmful[:3])
            logging.info("Most helpful img IDs: ")
            logging.info(helpful[:3])

        influences_path = outdir.joinpath(f"influence_results.json")
        save_json(influences, influences_path)
        return influences
        """




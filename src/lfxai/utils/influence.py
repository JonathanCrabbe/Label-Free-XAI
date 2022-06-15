"""
This code is adapted from https://github.com/nimarb/pytorch_influence_functions/
"""

import torch

mse = torch.nn.MSELoss()


def stack_torch_tensors(input_tensors):
    """
    Takes a list of tensors and stacks them into one tensor
    """

    unrolled = [input_tensors[k].reshape(-1, 1) for k in range(len(input_tensors))]

    return torch.cat(unrolled)


def get_numpy_parameters(model):
    """
    Recovers the parameters of a pytorch model in numpy format
    """

    params = []

    for param in model.parameters():
        params.append(param)

    return stack_torch_tensors(params).detach().numpy()


def hessian_vector_product(loss, model, v):
    """
    Multiplies the Hessians of the loss of a model with respect to its parameters by a vector v.
    Adapted from: https://github.com/kohpangwei/influence-release

    This function uses a backproplike approach to compute the product between the Hessian
    and another vector efficiently, which even works for large Hessians with O(p) compelxity for p parameters.

    Arguments:
        loss: scalar/tensor, for example the output of the loss function
        model: the model for which the Hessian of the loss is evaluated
        v: list of torch tensors, rnn.parameters(),
            will be multiplied with the Hessian
    Returns:
        return_grads: list of torch tensors, contains product of Hessian and v.
    """

    # First backprop
    first_grads = stack_torch_tensors(
        torch.autograd.grad(
            loss, model.encoder.parameters(), retain_graph=True, create_graph=True
        )
    )

    # Elementwise products
    elemwise_products = torch.dot(first_grads.flatten(), v.flatten())

    # Second backprop
    HVP_ = torch.autograd.grad(elemwise_products, model.encoder.parameters())

    return HVP_

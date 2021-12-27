"""
This code is adapted from https://github.com/ahmedmalaa/torch-influence-functions/
"""

import numpy as np
import torch


def stack_torch_tensors(input_tensors):
    '''
    Takes a list of tensors and stacks them into one tensor
    '''

    unrolled = [input_tensors[k].view(-1, 1) for k in range(len(input_tensors))]

    return torch.cat(unrolled)


def get_numpy_parameters(model):
    '''
    Recovers the parameters of a pytorch model in numpy format
    '''

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
        rnn: the model for which the Hessian of the loss is evaluated
        v: list of torch tensors, rnn.parameters(),
            will be multiplied with the Hessian
    Returns:
        return_grads: list of torch tensors, contains product of Hessian and v.
    """

    # First backprop
    first_grads = stack_torch_tensors(
        torch.autograd.grad(loss, model.parameters(), retain_graph=True, create_graph=True))

    # Elementwise products
    elemwise_products = 0

    for grad_elem, v_elem in zip(first_grads, v):
        elemwise_products += torch.sum(grad_elem * v_elem)

    # Second backprop
    HVP_ = torch.autograd.grad(elemwise_products, model.parameters(), create_graph=True)

    return HVP_


def perturb_model(model, perturb):
    """
    Perturbs the parameters of a model by a given vector of influences

    Arguments:
        model: a pytorch model with p parameters
        perturb: a tensors with size p designating the desired parameter-wise perturbation

    Returns:
        perturbed_model : a copy of the original model with perturbed parameters
    """


    params = []
    NUM_SAMPLES = model.X.shape[0]

    for param in model.parameters():
        params.append(param)

    param_ = stack_torch_tensors(params)
    new_param_ = param_ + perturb

    # copy all model attributes

    perturbed_model = type(model)()
    perturbed_model.__dict__.update(model.__dict__)
    perturbed_model.load_state_dict(model.state_dict())  # copy weights and stuff

    index = 0

    for param in perturbed_model.parameters():

        if len(param.data.shape) > 1:

            new_size = np.max((1, param.data.shape[0])) * np.max((1, param.data.shape[1]))
            param.data = new_param_[index: index + new_size].view(param.data.shape[0], param.data.shape[1])

        else:

            new_size = param.data.shape[0]
            param.data = np.squeeze(new_param_[index: index + new_size])

        index += new_size

    return perturbed_model

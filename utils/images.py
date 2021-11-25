from captum.attr._utils.visualization import visualize_image_attr
import torch


def plot_image_saliency(image: torch.Tensor, saliency: torch.Tensor):
    image_np = image.permute((1, 2, 0)).cpu().numpy()
    saliency_np = saliency.permute((1, 2, 0)).cpu().numpy()
    visualize_image_attr(saliency_np, image_np)
import torch
import hydra
import logging
from models.images import SimCLR
from omegaconf import DictConfig
from pathlib import Path
from torchvision.models import resnet18, resnet34


@hydra.main(config_name='simclr_config.yaml', config_path=str(Path.cwd()/'models'))
def consistency_feature_importance(args: DictConfig):
    # Prepare model
    torch.manual_seed(args.seed)
    assert args.backbone in ['resnet18', 'resnet34']
    base_encoder = eval(args.backbone)
    model = SimCLR(base_encoder, projection_dim=args.projection_dim).cuda()
    logging.info('Base model: {}'.format(args.backbone))
    logging.info('feature dim: {}, projection dim: {}'.format(model.feature_dim, args.projection_dim))
    model.fit(args)


if __name__ == '__main__':
    consistency_feature_importance()
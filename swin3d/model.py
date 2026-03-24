import torch.nn as nn
from torchvision.models.video import Swin3D_S_Weights, swin3d_s

from .config import TrainingConfig


def build_model(config: TrainingConfig) -> nn.Module:
    model = swin3d_s(weights=Swin3D_S_Weights.DEFAULT)
    for parameter in model.parameters():
        parameter.requires_grad = False
    model.head = nn.Linear(model.head.in_features, config.num_classes)
    return model

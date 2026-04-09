import torch.nn as nn
from torchvision.models.video import Swin3D_S_Weights, swin3d_s

from .config import TrainingConfig


def build_model(config: TrainingConfig) -> nn.Module:
    """Build Swin3D-S classification model with partial fine-tuning support.

    Unfreezing strategy (finetune_last_stage=True):
        - features[5]: last PatchMerging layer
        - features[6]: last Stage (2 SwinTransformerBlocks)
        - norm: LayerNorm
        - head: new classification head nn.Linear(768, num_classes)

    When finetune_last_stage=False, degrades to linear probing (head only).
    """
    model = swin3d_s(weights=Swin3D_S_Weights.DEFAULT)

    # 1. Freeze all parameters
    for parameter in model.parameters():
        parameter.requires_grad = False

    # 2. Unfreeze last stage: PatchMerging + SwinTransformerBlock
    if config.finetune_last_stage:
        # features[5] = last PatchMerging (dim 384 -> 768)
        for param in model.features[5].parameters():
            param.requires_grad = True
        # features[6] = last Stage (2 SwinTransformerBlocks)
        for param in model.features[6].parameters():
            param.requires_grad = True
        # norm layer
        for param in model.norm.parameters():
            param.requires_grad = True

    # 3. Replace classification head (newly created layer has requires_grad=True by default)
    model.head = nn.Linear(model.head.in_features, config.num_classes)

    return model


def get_param_groups(model: nn.Module, config: TrainingConfig) -> list[dict]:
    """Create parameter groups with differential learning rates.

    - Head parameters: use config.learning_rate
    - Backbone trainable parameters: use config.backbone_lr (typically 1/10 of head lr)
    """
    head_params = list(model.head.parameters())
    backbone_trainable_params = [
        param for name, param in model.named_parameters()
        if param.requires_grad and not name.startswith("head")
    ]

    param_groups = []
    if backbone_trainable_params:
        param_groups.append({"params": backbone_trainable_params, "lr": config.backbone_lr})
    if head_params:
        param_groups.append({"params": head_params, "lr": config.learning_rate})

    if not param_groups:
        raise ValueError("No trainable parameters found. Check fine-tuning configuration.")

    return param_groups


def count_parameters(model: nn.Module) -> dict:
    """Count model parameters."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = total - trainable
    return {
        "total": total,
        "trainable": trainable,
        "frozen": frozen,
    }

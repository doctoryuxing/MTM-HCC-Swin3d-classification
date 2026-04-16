import torch.nn as nn
from torchvision.models.video import R3D_18_Weights, Swin3D_S_Weights, r3d_18, swin3d_s

from .config import TrainingConfig


def _build_swin3d_model(config: TrainingConfig) -> nn.Module:
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


def _build_r3d18_model(config: TrainingConfig) -> nn.Module:
    """Build R3D-18 (3D ResNet-18) classification model with partial fine-tuning support.

    Unfreezing strategy (finetune_last_stage=True):
        - layer4: last residual stage
        - fc: new classification head nn.Linear(512, num_classes)

    When finetune_last_stage=False, degrades to linear probing (fc only).
    """
    model = r3d_18(weights=R3D_18_Weights.DEFAULT)

    # 1. Freeze all parameters
    for parameter in model.parameters():
        parameter.requires_grad = False

    # 2. Unfreeze last residual stage
    if config.finetune_last_stage:
        for param in model.layer4.parameters():
            param.requires_grad = True

    # 3. Replace classification head (newly created layer has requires_grad=True by default)
    model.fc = nn.Linear(model.fc.in_features, config.num_classes)

    return model


def build_model(config: TrainingConfig) -> nn.Module:
    """Build a classification model based on config.backbone_name.

    Supported backbones:
        - ``swin3d_s``: Swin3D-Small (default)
        - ``r3d_18``: 3D ResNet-18 (CNN)
    """
    if config.backbone_name == "r3d_18":
        return _build_r3d18_model(config)
    return _build_swin3d_model(config)


def _head_attr(model: nn.Module) -> str:
    """Return the name of the classification head attribute."""
    return "head" if hasattr(model, "head") else "fc"


def get_param_groups(model: nn.Module, config: TrainingConfig) -> list[dict]:
    """Create parameter groups with differential learning rates.

    - Head parameters: use config.learning_rate
    - Backbone trainable parameters: use config.backbone_lr (typically 1/10 of head lr)
    """
    attr = _head_attr(model)
    head_params = list(getattr(model, attr).parameters())
    backbone_trainable_params = [
        param for name, param in model.named_parameters()
        if param.requires_grad and not name.startswith(attr)
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

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    def __init__(self, gamma: float = 2.0, alpha=None, reduction: str = "mean") -> None:
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

        if isinstance(alpha, (float, int)):
            self.alpha = torch.tensor([alpha, 1 - alpha], dtype=torch.float32)
        elif isinstance(alpha, list):
            self.alpha = torch.tensor(alpha, dtype=torch.float32)

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probabilities = F.softmax(inputs, dim=1)
        log_probabilities = torch.log(probabilities.clamp_min(1e-8))
        log_p = log_probabilities.gather(1, targets.unsqueeze(1)).view(-1)
        p_t = probabilities.gather(1, targets.unsqueeze(1)).view(-1)
        loss = -((1 - p_t) ** self.gamma) * log_p

        if self.alpha is not None:
            alpha = self.alpha.to(inputs.device)
            alpha_t = alpha.gather(0, targets)
            loss = loss * alpha_t

        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss

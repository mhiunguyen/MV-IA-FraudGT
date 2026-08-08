"""Class-weighted binary focal loss for highly imbalanced fraud labels."""

import torch
import torch.nn.functional as F

from fraudGT.graphgym.config import cfg
from fraudGT.graphgym.register import register_loss


@register_loss('class_weighted_focal')
def class_weighted_focal(pred, true, epoch=None):
    if cfg.model.loss_fun != 'class_weighted_focal':
        return None

    true = true.float()
    bce = F.binary_cross_entropy_with_logits(pred, true, reduction='none')
    probability = torch.sigmoid(pred)
    p_t = probability * true + (1.0 - probability) * (1.0 - true)
    focal_factor = (1.0 - p_t).pow(cfg.mvia.focal_gamma)

    class_weights = torch.as_tensor(
        cfg.model.loss_fun_weight,
        dtype=pred.dtype,
        device=pred.device,
    )
    if class_weights.numel() != 2:
        raise ValueError('class_weighted_focal expects two class weights')
    sample_weights = class_weights[true.long()]

    loss = (sample_weights * focal_factor * bce).mean()
    return loss, probability

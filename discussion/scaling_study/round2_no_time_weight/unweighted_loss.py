"""Identical uncertainty-weighted task losses, with equal weight for all times."""
from contextlib import contextmanager
import torch


def unweighted_multi_task_loss(pred, target, log_sigma_mag, log_sigma_dis, log_sigma_azi, T=None):
    batch, steps, _ = pred.shape
    if T is not None and steps != T:
        raise ValueError(f'Expected {T} predictions, got {steps}')
    # Keep the original target shape handling and per-task reductions.
    if target.dim() == 3:
        target = target.squeeze(1)
    elif target.dim() > 2:
        target = target.view(batch, -1)
        if target.size(1) != 4:
            target = target[:, :4]
    truth = target.unsqueeze(1).expand(-1, steps, -1)
    raw_mag = ((pred[:, :, 0:1] - truth[:, :, 0:1]) ** 2).mean()
    raw_dis = ((pred[:, :, 1:2] - truth[:, :, 1:2]) ** 2).mean()
    raw_sin = ((pred[:, :, 2:3] - truth[:, :, 2:3]) ** 2).mean()
    raw_cos = ((pred[:, :, 3:4] - truth[:, :, 3:4]) ** 2).mean()
    raw_azi = (raw_sin + raw_cos) / 2.0
    sigmas = tuple(torch.exp(q) for q in (log_sigma_mag, log_sigma_dis, log_sigma_azi))
    losses = tuple(raw / (2 * sigma**2) + q for raw, sigma, q in
                   zip((raw_mag, raw_dis, raw_azi), sigmas,
                       (log_sigma_mag, log_sigma_dis, log_sigma_azi)))
    return sum(losses), *losses, sigmas


@contextmanager
def equal_time_loss(original):
    """Change only this process, including BOTH training and validation calls.

    Never edit the original trainer on disk; restore even if a check fails.
    """
    previous = original.multi_task_weighted_mse_loss
    original.multi_task_weighted_mse_loss = unweighted_multi_task_loss
    try:
        yield
    finally:
        original.multi_task_weighted_mse_loss = previous

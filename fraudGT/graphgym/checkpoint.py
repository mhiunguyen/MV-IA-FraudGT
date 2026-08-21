import glob
import os
import os.path as osp
from typing import Any, Dict, List, Optional, Union

import torch

from fraudGT.graphgym.config import cfg

MODEL_STATE = 'model_state'
OPTIMIZER_STATE = 'optimizer_state'
SCHEDULER_STATE = 'scheduler_state'
BEST_CKPT_NAME = 'best.ckpt'


def load_ckpt(
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    epoch: int = -1,
) -> int:
    r"""Loads the model checkpoint at a given epoch."""
    epoch = get_ckpt_epoch(epoch)
    path = get_ckpt_path(epoch)

    if not osp.exists(path):
        return 0

    # The checkpoint contains optimizer/scheduler Python objects in addition
    # to tensors. PyTorch 2.6+ defaults to weights_only=True, which is not the
    # correct mode for this trusted, locally generated training checkpoint.
    ckpt = torch.load(path, weights_only=False)
    model.load_state_dict(ckpt[MODEL_STATE])
    if optimizer is not None and OPTIMIZER_STATE in ckpt:
        optimizer.load_state_dict(ckpt[OPTIMIZER_STATE])
    if scheduler is not None and SCHEDULER_STATE in ckpt:
        scheduler.load_state_dict(ckpt[SCHEDULER_STATE])

    return epoch + 1


def save_ckpt(
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    epoch: int = 0,
):
    r"""Saves the model checkpoint at a given epoch."""
    ckpt: Dict[str, Any] = {}
    ckpt[MODEL_STATE] = model.state_dict()
    if optimizer is not None:
        ckpt[OPTIMIZER_STATE] = optimizer.state_dict()
    if scheduler is not None:
        ckpt[SCHEDULER_STATE] = scheduler.state_dict()

    os.makedirs(get_ckpt_dir(), exist_ok=True)
    torch.save(ckpt, get_ckpt_path(get_ckpt_epoch(epoch)))


def save_best_ckpt(
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    epoch: int = 0,
    metric_name: str = '',
    metric_value: Optional[float] = None,
):
    """Save the best validation checkpoint separately from resume state."""
    ckpt: Dict[str, Any] = {
        MODEL_STATE: model.state_dict(),
        'epoch': int(epoch),
        'metric_name': str(metric_name),
        'metric_value': None if metric_value is None else float(metric_value),
    }
    if optimizer is not None:
        ckpt[OPTIMIZER_STATE] = optimizer.state_dict()
    if scheduler is not None:
        ckpt[SCHEDULER_STATE] = scheduler.state_dict()
    os.makedirs(get_ckpt_dir(), exist_ok=True)
    torch.save(ckpt, osp.join(get_ckpt_dir(), BEST_CKPT_NAME))


def get_best_ckpt_metadata() -> Dict[str, Any]:
    """Read lightweight selection metadata from best.ckpt when it exists."""
    path = osp.join(get_ckpt_dir(), BEST_CKPT_NAME)
    if not osp.exists(path):
        return {}
    ckpt = torch.load(path, map_location='cpu', weights_only=False)
    return {
        'epoch': ckpt.get('epoch'),
        'metric_name': ckpt.get('metric_name'),
        'metric_value': ckpt.get('metric_value'),
    }


def remove_ckpt(epoch: int = -1):
    r"""Removes the model checkpoint at a given epoch."""
    os.remove(get_ckpt_path(get_ckpt_epoch(epoch)))


def clean_ckpt():
    r"""Removes all but the last model checkpoint."""
    for epoch in get_ckpt_epochs()[:-1]:
        os.remove(get_ckpt_path(epoch))


###############################################################################


def get_ckpt_dir() -> str:
    return osp.join(cfg.run_dir, 'ckpt')


def get_ckpt_path(epoch: Union[int, str]) -> str:
    return osp.join(get_ckpt_dir(), f'{epoch}.ckpt')


def get_ckpt_epochs() -> List[int]:
    paths = glob.glob(get_ckpt_path('*'))
    stems = [osp.basename(path).split('.')[0] for path in paths]
    return sorted(int(stem) for stem in stems if stem.isdigit())


def get_ckpt_epoch(epoch: int) -> int:
    if epoch < 0:
        epochs = get_ckpt_epochs()
        epoch = epochs[epoch] if len(epochs) > 0 else 0
    return epoch

import random

import numpy as np
from tqdm.auto import tqdm


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
    except Exception:
        pass


def make_epoch_bar(total_epochs: int, desc: str = "Training"):
    return tqdm(range(1, total_epochs + 1), total=total_epochs, desc=desc)


def log_epoch_summary(epoch: int, metrics: dict, prefix: str = "train"):
    total = metrics.get("total_loss", float("nan"))
    diff = metrics.get("diffusion", float("nan"))
    dur = metrics.get("duration", float("nan"))
    prior = metrics.get("prior", float("nan"))
    lr = metrics.get("lr", None)
    lr_text = f" | lr={lr:.2e}" if isinstance(lr, (float, int)) else ""
    print(
        f"[{prefix}] epoch={epoch:03d} total={total:.4f} diff={diff:.4f} dur={dur:.4f} prior={prior:.4f}{lr_text}",
        flush=True,
    )

"""Learning rate scheduler utilities."""

import math

import torch.optim as optim


class WarmupCosineScheduler(optim.lr_scheduler.LambdaLR):
    """Cosine annealing schedule with linear warm-up.

    Args:
        optimizer: Wrapped optimizer.
        warmup_steps: Number of linear warm-up steps.
        total_steps: Total training steps.
        min_lr_ratio: Minimum learning rate as a fraction of the initial lr.
    """

    def __init__(
        self,
        optimizer: optim.Optimizer,
        warmup_steps: int,
        total_steps: int,
        min_lr_ratio: float = 0.0,
    ):
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr_ratio = min_lr_ratio

        def lr_lambda(step: int) -> float:
            if step < warmup_steps:
                return step / max(1, warmup_steps)
            progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
            return max(
                min_lr_ratio,
                0.5 * (1.0 + math.cos(math.pi * progress)),
            )

        super().__init__(optimizer, lr_lambda)


class InverseSquareRootScheduler(optim.lr_scheduler.LambdaLR):
    """Inverse square-root schedule with warm-up (Transformer original).

    Args:
        optimizer: Wrapped optimizer.
        warmup_steps: Number of warm-up steps.
    """

    def __init__(self, optimizer: optim.Optimizer, warmup_steps: int = 4000):
        self.warmup_steps = warmup_steps

        def lr_lambda(step: int) -> float:
            step = max(1, step)
            return min(step ** (-0.5), step * warmup_steps ** (-1.5))

        super().__init__(optimizer, lr_lambda)


def get_scheduler(
    name: str,
    optimizer: optim.Optimizer,
    warmup_steps: int = 4000,
    total_steps: int = 100000,
    **kwargs,
):
    """Factory function for learning rate schedulers.

    Args:
        name: Scheduler name ("warmup_cosine", "inverse_sqrt", "none").
        optimizer: Wrapped optimizer.
        warmup_steps: Number of warm-up steps.
        total_steps: Total training steps (for cosine schedule).

    Returns:
        Learning rate scheduler or None.
    """
    if name == "warmup_cosine":
        return WarmupCosineScheduler(optimizer, warmup_steps, total_steps, **kwargs)
    elif name == "inverse_sqrt":
        return InverseSquareRootScheduler(optimizer, warmup_steps)
    elif name == "none":
        return None
    else:
        raise ValueError(f"Unknown scheduler: {name}")

"""Training and evaluation helper functions."""

import os

import torch
import torch.nn as nn
from tqdm import tqdm

from src.data.preprocessing import PAD_IDX


def train_one_epoch(
    model: nn.Module,
    dataloader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    clip_grad: float = 1.0,
    scheduler=None,
) -> float:
    """Train the model for one epoch.

    Args:
        model: The translation model.
        dataloader: Training DataLoader.
        optimizer: Optimiser instance.
        criterion: Loss function (e.g., CrossEntropyLoss).
        device: Torch device.
        clip_grad: Maximum gradient norm for clipping.
        scheduler: Optional learning-rate scheduler (step per batch).

    Returns:
        Average training loss for the epoch.
    """
    model.train()
    total_loss = 0.0
    n_batches = 0

    for src, tgt in tqdm(dataloader, desc="Training", leave=False):
        src, tgt = src.to(device), tgt.to(device)

        optimizer.zero_grad()

        # For Transformer: input is tgt[:, :-1], target is tgt[:, 1:]
        # For Seq2Seq: model handles internally, output aligns with tgt[:, 1:]
        output = model(src, tgt)

        # Reshape for loss computation
        if output.dim() == 3:
            # (batch, seq_len, vocab) -> (batch * seq_len, vocab)
            # Target: tgt[:, 1:] for both model types
            tgt_out = tgt[:, 1:]
            if output.size(1) == tgt.size(1):
                # Transformer outputs same length as tgt input
                tgt_out = tgt[:, 1:]
                output = output[:, :-1]

            output = output.contiguous().view(-1, output.size(-1))
            tgt_out = tgt_out.contiguous().view(-1)
        else:
            tgt_out = tgt[:, 1:].contiguous().view(-1)

        loss = criterion(output, tgt_out)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
        optimizer.step()

        if scheduler is not None:
            scheduler.step()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(n_batches, 1)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    dataloader,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    """Evaluate the model on a validation/test set.

    Args:
        model: The translation model.
        dataloader: Validation/test DataLoader.
        criterion: Loss function.
        device: Torch device.

    Returns:
        Average evaluation loss.
    """
    model.eval()
    total_loss = 0.0
    n_batches = 0

    for src, tgt in tqdm(dataloader, desc="Evaluating", leave=False):
        src, tgt = src.to(device), tgt.to(device)

        output = model(src, tgt)

        if output.dim() == 3:
            tgt_out = tgt[:, 1:]
            if output.size(1) == tgt.size(1):
                output = output[:, :-1]
            output = output.contiguous().view(-1, output.size(-1))
            tgt_out = tgt_out.contiguous().view(-1)
        else:
            tgt_out = tgt[:, 1:].contiguous().view(-1)

        loss = criterion(output, tgt_out)
        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(n_batches, 1)


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    loss: float,
    path: str,
    **extra,
) -> None:
    """Save a training checkpoint.

    Args:
        model: Model to save.
        optimizer: Optimizer to save.
        epoch: Current epoch number.
        loss: Current loss value.
        path: File path for the checkpoint.
        **extra: Additional items to store.
    """
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    state = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": loss,
    }
    state.update(extra)
    torch.save(state, path)


def load_checkpoint(
    path: str,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    device: torch.device | None = None,
) -> dict:
    """Load a training checkpoint.

    Args:
        path: Path to the checkpoint file.
        model: Model to load state into.
        optimizer: Optional optimizer to load state into.
        device: Device to map the checkpoint to.

    Returns:
        The full checkpoint dict (for access to epoch, loss, etc.).
    """
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return checkpoint

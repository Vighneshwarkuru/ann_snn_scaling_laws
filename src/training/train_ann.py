"""
ANN training loop with checkpointing.

Trains an ANN model and saves the best checkpoint.
If a checkpoint already exists, it is loaded directly (no re-training).
"""

import os
import time
import math
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Training helpers
# ---------------------------------------------------------------------------

def _cosine_schedule(optimizer: optim.Optimizer, epoch: int, total_epochs: int, lr: float):
    """Set LR to cosine annealing value for current epoch."""
    lr_t = lr * 0.5 * (1 + math.cos(math.pi * epoch / total_epochs))
    for pg in optimizer.param_groups:
        pg["lr"] = lr_t


def _step_schedule(optimizer: optim.Optimizer, epoch: int, milestones=(30, 40), gamma=0.1):
    if epoch in milestones:
        for pg in optimizer.param_groups:
            pg["lr"] *= gamma


def _accuracy(logits: torch.Tensor, targets: torch.Tensor) -> float:
    preds = logits.argmax(dim=1)
    return (preds == targets).float().mean().item()


# ---------------------------------------------------------------------------
# Core training function
# ---------------------------------------------------------------------------

def train_ann(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    *,
    epochs: int = 50,
    lr: float = 0.01,
    weight_decay: float = 5e-4,
    scheduler: str = "cosine",
    device: torch.device,
    checkpoint_path: Optional[str] = None,
    verbose: bool = True,
) -> float:
    """
    Train an ANN model.

    Returns:
        Best validation accuracy achieved.
    """
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(
        model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay
    )

    best_acc = 0.0
    best_state = None

    for epoch in range(epochs):
        # LR schedule
        if scheduler == "cosine":
            _cosine_schedule(optimizer, epoch, epochs, lr)
        elif scheduler == "step":
            _step_schedule(optimizer, epoch)

        # --- Train ---
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", leave=False, disable=not verbose)
        for imgs, labels in pbar:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = model(imgs)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * imgs.size(0)
            train_correct += (logits.argmax(1) == labels).sum().item()
            train_total += imgs.size(0)

        train_acc = train_correct / train_total

        # --- Validate ---
        val_acc = _eval_accuracy(model, val_loader, device)

        if val_acc > best_acc:
            best_acc = val_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if verbose:
            current_lr = optimizer.param_groups[0]["lr"]
            print(
                f"  Epoch {epoch+1:3d}/{epochs} | "
                f"Loss {train_loss/train_total:.4f} | "
                f"Train {train_acc*100:.2f}% | "
                f"Val {val_acc*100:.2f}% | "
                f"Best {best_acc*100:.2f}% | "
                f"LR {current_lr:.5f}"
            )

    # Load best weights
    if best_state is not None:
        model.load_state_dict(best_state)

    # Save checkpoint
    if checkpoint_path is not None:
        Path(checkpoint_path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "best_val_acc": best_acc,
            },
            checkpoint_path,
        )
        if verbose:
            print(f"  Checkpoint saved → {checkpoint_path}")

    return best_acc


def _eval_accuracy(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            logits = model(imgs)
            correct += (logits.argmax(1) == labels).sum().item()
            total += imgs.size(0)
    return correct / total


# ---------------------------------------------------------------------------
# Load-or-train convenience wrapper
# ---------------------------------------------------------------------------

def load_or_train_ann(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    checkpoint_path: str,
    *,
    epochs: int = 50,
    lr: float = 0.01,
    weight_decay: float = 5e-4,
    scheduler: str = "cosine",
    device: torch.device,
    verbose: bool = True,
) -> float:
    """
    If a checkpoint exists at `checkpoint_path`, load it.
    Otherwise train the model from scratch and save the checkpoint.

    Returns:
        Best validation accuracy.
    """
    if os.path.exists(checkpoint_path):
        ckpt = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        best_acc = ckpt.get("best_val_acc", 0.0)
        if verbose:
            print(f"  Loaded checkpoint from {checkpoint_path} (val acc: {best_acc*100:.2f}%)")
        return best_acc

    if verbose:
        print(f"  No checkpoint found at {checkpoint_path}. Training from scratch...")

    return train_ann(
        model,
        train_loader,
        val_loader,
        epochs=epochs,
        lr=lr,
        weight_decay=weight_decay,
        scheduler=scheduler,
        device=device,
        checkpoint_path=checkpoint_path,
        verbose=verbose,
    )

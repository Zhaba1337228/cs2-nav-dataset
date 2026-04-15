"""
Training stub: demonstrates how to load the dataset, create DataLoaders,
and run a minimal training loop. No actual model — just shows the data flow.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from training.dataloader_utils import create_dataloaders, demo_dataloader
from training.label_maps import LabelEncoder


def run_training_demo(
    train_manifest: str,
    val_manifest: str,
    dataset_root: str = "",
    batch_size: int = 8,
    history_len: int = 1,
    epochs: int = 2,
) -> None:
    """
    Minimal training loop demonstrating dataset usage.

    Shows:
    - How to create DataLoaders
    - What batch shapes look like
    - How to access target labels
    - How to compute dummy losses per action head
    """
    print("=" * 60)
    print("CS2 Navigation Dataset — Training Demo")
    print("=" * 60)

    # Create dataloaders
    train_loader, val_loader, train_ds, val_ds = create_dataloaders(
        train_manifest=train_manifest,
        val_manifest=val_manifest,
        dataset_root=dataset_root if dataset_root else None,
        history_len=history_len,
        batch_size=batch_size,
        num_workers=0,
        pin_memory=False,
    )

    print(f"\nDataset loaded:")
    print(f"  Train: {len(train_ds)} samples")
    print(f"  Val:   {len(val_ds)} samples")
    print(f"  Move classes: {train_ds.n_move_classes}")
    print(f"  Turn classes: {train_ds.n_turn_classes}")
    print(f"  History length: {history_len}")

    # Label encoder
    label_encoder = train_ds.label_encoder
    print(f"\nMove labels: {label_encoder.get_all_move_labels()}")
    print(f"Turn labels: {label_encoder.get_all_turn_labels()}")

    # Dummy "model" — just random predictions for demonstration
    print(f"\n{'=' * 60}")
    print(f"Running {epochs} demo epochs...")
    print(f"{'=' * 60}")

    for epoch in range(epochs):
        train_loss = _run_epoch(train_loader, epoch, training=True)
        val_loss = _run_epoch(val_loader, epoch, training=False)
        print(f"Epoch {epoch + 1}/{epochs} — Train loss: {train_loss:.4f}, Val loss: {val_loss:.4f}")

    print("\nDemo complete. The dataset and DataLoaders are ready for real training.")


def _run_epoch(loader: DataLoader, epoch: int, training: bool = True) -> float:
    """Run one epoch, computing dummy losses per action head."""
    total_loss = 0.0
    n_batches = 0

    mode = "TRAIN" if training else "VAL"

    for batch_idx, batch in enumerate(loader):
        images = batch["images"]       # (B, history, C, H, W)
        targets = batch["targets"]     # dict of target tensors

        B = images.shape[0]

        # Dummy predictions (random)
        move_logits = torch.randn(B, 9)       # 9 move classes
        turn_logits = torch.randn(B, 9)       # 9 turn classes
        jump_logits = torch.randn(B, 2)       # binary
        crouch_logits = torch.randn(B, 2)     # binary
        fire_logits = torch.randn(B, 2)       # binary

        # Dummy cross-entropy losses
        move_loss = torch.nn.functional.cross_entropy(
            move_logits, targets["action_move"].long()
        )
        turn_loss = torch.nn.functional.cross_entropy(
            turn_logits, targets["action_turn"].long()
        )
        jump_loss = torch.nn.functional.binary_cross_entropy_with_logits(
            jump_logits.squeeze(), targets["action_jump"].float()
        )
        crouch_loss = torch.nn.functional.binary_cross_entropy_with_logits(
            crouch_logits.squeeze(), targets["action_crouch"].float()
        )
        fire_loss = torch.nn.functional.binary_cross_entropy_with_logits(
            fire_logits.squeeze(), targets["action_fire"].float()
        )

        # Combined loss
        loss = move_loss + turn_loss + jump_loss + crouch_loss + fire_loss
        total_loss += loss.item()
        n_batches += 1

        if batch_idx == 0 and epoch == 0:
            print(f"\n  [{mode}] Batch 0:")
            print(f"    Images shape: {images.shape}")
            print(f"    Move targets: {targets['action_move'][:4].tolist()}")
            print(f"    Turn targets: {targets['action_turn'][:4].tolist()}")
            print(f"    Jump targets: {targets['action_jump'][:4].tolist()}")
            print(f"    Loss: {loss.item():.4f}")

    return total_loss / max(n_batches, 1)


def main():
    parser = argparse.ArgumentParser(description="Training demo for CS2 navigation dataset")
    parser.add_argument("--train-manifest", type=str, default="dataset/manifests/train_manifest.jsonl")
    parser.add_argument("--val-manifest", type=str, default="dataset/manifests/val_manifest.jsonl")
    parser.add_argument("--dataset-root", type=str, default="")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--history-len", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=2)
    args = parser.parse_args()

    run_training_demo(
        train_manifest=args.train_manifest,
        val_manifest=args.val_manifest,
        dataset_root=args.dataset_root,
        batch_size=args.batch_size,
        history_len=args.history_len,
        epochs=args.epochs,
    )


if __name__ == "__main__":
    main()

"""
DataLoader utilities for CS2 navigation dataset.
Provides helpers for building DataLoaders, collation, and transforms.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import DataLoader

from training.dataset import NavigationDataset
from training.label_maps import LabelEncoder


def create_dataloaders(
    train_manifest: str | Path,
    val_manifest: str | Path,
    dataset_root: Optional[str | Path] = None,
    history_len: int = 1,
    image_size: tuple[int, int] = (224, 224),
    batch_size: int = 32,
    num_workers: int = 4,
    pin_memory: bool = True,
    train_transform: Optional[object] = None,
    val_transform: Optional[object] = None,
    label_encoder: Optional[LabelEncoder] = None,
) -> tuple[DataLoader, DataLoader, NavigationDataset, NavigationDataset]:
    """
    Create train and validation DataLoaders.

    Returns:
        (train_loader, val_loader, train_dataset, val_dataset)
    """
    le = label_encoder or LabelEncoder()

    train_dataset = NavigationDataset(
        manifest_path=train_manifest,
        dataset_root=dataset_root,
        history_len=history_len,
        image_size=image_size,
        transform=train_transform,
        label_encoder=le,
    )

    val_dataset = NavigationDataset(
        manifest_path=val_manifest,
        dataset_root=dataset_root,
        history_len=history_len,
        image_size=image_size,
        transform=val_transform,
        label_encoder=le,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    return train_loader, val_loader, train_dataset, val_dataset


def collate_fn(batch: list[tuple]) -> dict:
    """
    Custom collate function that batches images, targets, and metadata.

    Args:
        batch: List of (images, target, metadata) tuples from Dataset.__getitem__

    Returns:
        Dict with 'images', 'targets', 'metadata' keys
    """
    images, targets, metadatas = zip(*batch)

    return {
        "images": torch.stack(images, dim=0),
        "targets": {k: torch.tensor([t[k] for t in targets]) for k in targets[0]},
        "metadata": list(metadatas),
    }


def get_default_transforms():
    """
    Return default train and validation transforms.
    Uses torchvision if available, otherwise returns None.
    """
    try:
        from torchvision import transforms

        train_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
        ])

        val_transform = transforms.Compose([
            transforms.ToTensor(),
        ])

        return train_transform, val_transform
    except ImportError:
        return None, None


def demo_dataloader(
    train_manifest: str | Path,
    val_manifest: str | Path,
    dataset_root: Optional[str | Path] = None,
    batch_size: int = 4,
    history_len: int = 1,
) -> None:
    """
    Demo function showing how to use the DataLoader.
    Prints batch shapes and sample targets.
    """
    train_loader, val_loader, train_ds, val_ds = create_dataloaders(
        train_manifest=train_manifest,
        val_manifest=val_manifest,
        dataset_root=dataset_root,
        history_len=history_len,
        batch_size=batch_size,
        num_workers=0,  # single-process for demo
        pin_memory=False,
    )

    print(f"Train dataset size: {len(train_ds)}")
    print(f"Val dataset size: {len(val_ds)}")
    print(f"Move classes: {train_ds.n_move_classes}")
    print(f"Turn classes: {train_ds.n_turn_classes}")

    # Show one batch
    batch = next(iter(train_loader))
    images = batch["images"]
    targets = batch["targets"]

    print(f"\nBatch images shape: {images.shape}")
    print(f"Expected: (batch_size={batch_size}, history_len={history_len}, C=3, H=224, W=224)")

    print("\nTarget keys and shapes:")
    for k, v in targets.items():
        print(f"  {k}: {v.shape} — sample values: {v[:4].tolist()}")

    print("\nSample metadata:")
    for meta in batch["metadata"][:2]:
        print(f"  {meta}")

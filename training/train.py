"""
Full training script for CS2 navigation model with multi-GPU support.
Optimized for 2x RTX 3090 setup with DDP, mixed precision, and gradient accumulation.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from torch.amp import autocast, GradScaler
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

from training.model import create_model
from training.dataset import NavigationDataset
from training.label_maps import LabelEncoder
from training.dataloader_utils import collate_fn

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class Trainer:
    """Main trainer class with DDP support."""

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[object],
        device: torch.device,
        config: dict,
        rank: int = 0,
        world_size: int = 1,
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.config = config
        self.rank = rank
        self.world_size = world_size
        self.is_main = rank == 0

        # Mixed precision
        self.use_amp = config.get("use_amp", True)
        self.scaler = GradScaler(device="cuda", enabled=self.use_amp) if self.use_amp else None

        # Gradient accumulation
        self.grad_accum_steps = config.get("grad_accum_steps", 1)
        self.max_grad_norm = float(config.get("max_grad_norm", 1.0))
        self.label_smoothing = float(config.get("label_smoothing", 0.05))
        self.mouse_loss = str(config.get("mouse_loss", "huber")).lower()
        self.mouse_huber_beta = float(config.get("mouse_huber_beta", 1.0))

        # Loss weights
        self.loss_weights = {
            "move": config.get("loss_weight_move", 1.0),
            "turn": config.get("loss_weight_turn", 1.0),
            "jump": config.get("loss_weight_jump", 0.5),
            "crouch": config.get("loss_weight_crouch", 0.5),
            "fire": config.get("loss_weight_fire", 0.5),
            "mouse": config.get("loss_weight_mouse", 0.3),
        }

        # Metrics tracking
        self.best_val_loss = float('inf')
        self.no_improve_epochs = 0
        self.early_stopping_patience = int(config.get("early_stopping_patience", 12))
        self.early_stopping_min_delta = float(config.get("early_stopping_min_delta", 1e-3))
        self.train_losses = []
        self.val_losses = []

        # Checkpointing
        self.checkpoint_dir = Path(config.get("checkpoint_dir", "checkpoints"))
        if self.is_main:
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def train_epoch(self, epoch: int) -> dict:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        losses_dict = {
            "move": 0.0,
            "turn": 0.0,
            "jump": 0.0,
            "crouch": 0.0,
            "fire": 0.0,
            "mouse": 0.0,
        }
        n_batches = 0

        self.optimizer.zero_grad()

        for batch_idx, batch in enumerate(self.train_loader):
            images = batch["images"].to(self.device)  # (B, T, C, H, W)
            targets = {k: v.to(self.device) for k, v in batch["targets"].items()}

            # Forward pass with mixed precision
            with autocast(device_type="cuda", enabled=self.use_amp):
                outputs = self.model(images)
                loss, loss_components = self._compute_loss(outputs, targets)
                loss = loss / self.grad_accum_steps

            # Backward pass
            if self.use_amp:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()

            # Gradient accumulation
            if (batch_idx + 1) % self.grad_accum_steps == 0:
                if self.use_amp:
                    if self.max_grad_norm > 0:
                        self.scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    if self.max_grad_norm > 0:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                    self.optimizer.step()
                self.optimizer.zero_grad()

            # Track losses
            total_loss += loss.item() * self.grad_accum_steps
            for k, v in loss_components.items():
                losses_dict[k] += v
            n_batches += 1

            # Logging
            if self.is_main and batch_idx % 50 == 0:
                logger.info(
                    f"Epoch {epoch} [{batch_idx}/{len(self.train_loader)}] "
                    f"Loss: {loss.item() * self.grad_accum_steps:.4f}"
                )

        # Handle leftover gradients if number of batches is not divisible by grad_accum_steps.
        if n_batches > 0 and (n_batches % self.grad_accum_steps) != 0:
            if self.use_amp:
                if self.max_grad_norm > 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                if self.max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                self.optimizer.step()
            self.optimizer.zero_grad()

        # Reduce sums across ranks to compute true global averages.
        if dist.is_available() and dist.is_initialized():
            stat = torch.tensor(
                [
                    total_loss,
                    losses_dict["move"],
                    losses_dict["turn"],
                    losses_dict["jump"],
                    losses_dict["crouch"],
                    losses_dict["fire"],
                    losses_dict["mouse"],
                    float(n_batches),
                ],
                device=self.device,
                dtype=torch.float64,
            )
            dist.all_reduce(stat, op=dist.ReduceOp.SUM)
            total_loss = float(stat[0].item())
            losses_dict["move"] = float(stat[1].item())
            losses_dict["turn"] = float(stat[2].item())
            losses_dict["jump"] = float(stat[3].item())
            losses_dict["crouch"] = float(stat[4].item())
            losses_dict["fire"] = float(stat[5].item())
            losses_dict["mouse"] = float(stat[6].item())
            n_batches = int(stat[7].item())

        # Average losses
        avg_loss = total_loss / max(n_batches, 1)
        for k in losses_dict:
            losses_dict[k] /= max(n_batches, 1)

        return {"total": avg_loss, **losses_dict}

    @torch.no_grad()
    def validate(self, epoch: int) -> dict:
        """Validate on validation set."""
        self.model.eval()
        total_loss = 0.0
        losses_dict = {
            "move": 0.0,
            "turn": 0.0,
            "jump": 0.0,
            "crouch": 0.0,
            "fire": 0.0,
            "mouse": 0.0,
        }
        n_batches = 0

        # Accuracy tracking
        move_correct = 0
        turn_correct = 0
        total_samples = 0

        for batch in self.val_loader:
            images = batch["images"].to(self.device)
            targets = {k: v.to(self.device) for k, v in batch["targets"].items()}

            with autocast(device_type="cuda", enabled=self.use_amp):
                outputs = self.model(images)
                loss, loss_components = self._compute_loss(outputs, targets)

            total_loss += loss.item()
            for k, v in loss_components.items():
                losses_dict[k] += v
            n_batches += 1

            # Compute accuracy
            move_pred = outputs["move"].argmax(dim=1)
            turn_pred = outputs["turn"].argmax(dim=1)
            move_correct += (move_pred == targets["action_move"]).sum().item()
            turn_correct += (turn_pred == targets["action_turn"]).sum().item()
            total_samples += images.size(0)

        # Aggregate across all ranks (each rank validates only its shard).
        if dist.is_available() and dist.is_initialized():
            stat = torch.tensor(
                [
                    total_loss,
                    losses_dict["move"],
                    losses_dict["turn"],
                    losses_dict["jump"],
                    losses_dict["crouch"],
                    losses_dict["fire"],
                    losses_dict["mouse"],
                    float(n_batches),
                    float(move_correct),
                    float(turn_correct),
                    float(total_samples),
                ],
                device=self.device,
                dtype=torch.float64,
            )
            dist.all_reduce(stat, op=dist.ReduceOp.SUM)
            total_loss = float(stat[0].item())
            losses_dict["move"] = float(stat[1].item())
            losses_dict["turn"] = float(stat[2].item())
            losses_dict["jump"] = float(stat[3].item())
            losses_dict["crouch"] = float(stat[4].item())
            losses_dict["fire"] = float(stat[5].item())
            losses_dict["mouse"] = float(stat[6].item())
            n_batches = int(stat[7].item())
            move_correct = int(stat[8].item())
            turn_correct = int(stat[9].item())
            total_samples = int(stat[10].item())

        # Average losses
        avg_loss = total_loss / max(n_batches, 1)
        for k in losses_dict:
            losses_dict[k] /= max(n_batches, 1)

        # Compute accuracies
        move_acc = move_correct / max(total_samples, 1)
        turn_acc = turn_correct / max(total_samples, 1)

        return {
            "total": avg_loss,
            "move_acc": move_acc,
            "turn_acc": turn_acc,
            **losses_dict,
        }

    def _compute_loss(self, outputs: dict, targets: dict) -> tuple[torch.Tensor, dict]:
        """Compute multi-head loss."""
        # Classification losses
        move_loss = nn.functional.cross_entropy(
            outputs["move"], targets["action_move"].long(), label_smoothing=self.label_smoothing
        )
        turn_loss = nn.functional.cross_entropy(
            outputs["turn"], targets["action_turn"].long(), label_smoothing=self.label_smoothing
        )

        # Binary losses
        jump_loss = nn.functional.binary_cross_entropy_with_logits(
            outputs["jump"], targets["action_jump"].float()
        )
        crouch_loss = nn.functional.binary_cross_entropy_with_logits(
            outputs["crouch"], targets["action_crouch"].float()
        )
        fire_loss = nn.functional.binary_cross_entropy_with_logits(
            outputs["fire"], targets["action_fire"].float()
        )

        # Mouse regression loss (MSE)
        if self.mouse_loss == "mse":
            mouse_dx_loss = nn.functional.mse_loss(outputs["mouse_dx"], targets["mouse_dx"].float())
            mouse_dy_loss = nn.functional.mse_loss(outputs["mouse_dy"], targets["mouse_dy"].float())
        else:
            mouse_dx_loss = nn.functional.smooth_l1_loss(
                outputs["mouse_dx"],
                targets["mouse_dx"].float(),
                beta=self.mouse_huber_beta,
            )
            mouse_dy_loss = nn.functional.smooth_l1_loss(
                outputs["mouse_dy"],
                targets["mouse_dy"].float(),
                beta=self.mouse_huber_beta,
            )
        mouse_loss = (mouse_dx_loss + mouse_dy_loss) / 2.0

        # Weighted sum
        total_loss = (
            self.loss_weights["move"] * move_loss +
            self.loss_weights["turn"] * turn_loss +
            self.loss_weights["jump"] * jump_loss +
            self.loss_weights["crouch"] * crouch_loss +
            self.loss_weights["fire"] * fire_loss +
            self.loss_weights["mouse"] * mouse_loss
        )

        loss_components = {
            "move": move_loss.item(),
            "turn": turn_loss.item(),
            "jump": jump_loss.item(),
            "crouch": crouch_loss.item(),
            "fire": fire_loss.item(),
            "mouse": mouse_loss.item(),
        }

        return total_loss, loss_components

    def save_checkpoint(self, epoch: int, is_best: bool = False) -> None:
        """Save model checkpoint."""
        if not self.is_main:
            return

        # Get model state (unwrap DDP if needed)
        model_state = self.model.module.state_dict() if hasattr(self.model, 'module') else self.model.state_dict()

        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model_state,
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict() if self.scheduler else None,
            "best_val_loss": self.best_val_loss,
            "config": self.config,
        }

        # Save latest
        latest_path = self.checkpoint_dir / "checkpoint_latest.pt"
        torch.save(checkpoint, latest_path)
        logger.info(f"Saved checkpoint to {latest_path}")

        # Save best
        if is_best:
            best_path = self.checkpoint_dir / "checkpoint_best.pt"
            torch.save(checkpoint, best_path)
            logger.info(f"Saved best checkpoint to {best_path}")

        # Save periodic
        if epoch % 10 == 0:
            epoch_path = self.checkpoint_dir / f"checkpoint_epoch_{epoch}.pt"
            torch.save(checkpoint, epoch_path)

    def train(self, num_epochs: int) -> None:
        """Main training loop."""
        logger.info(f"Starting training for {num_epochs} epochs")
        logger.info(f"Device: {self.device}, Rank: {self.rank}/{self.world_size}")

        for epoch in range(1, num_epochs + 1):
            # Set epoch for distributed sampler
            if hasattr(self.train_loader.sampler, 'set_epoch'):
                self.train_loader.sampler.set_epoch(epoch)

            # Train
            train_metrics = self.train_epoch(epoch)

            # Validate
            val_metrics = self.validate(epoch)

            # Step scheduler
            if self.scheduler is not None:
                self.scheduler.step()

            # Logging
            if self.is_main:
                logger.info(
                    f"Epoch {epoch}/{num_epochs} - "
                    f"Train Loss: {train_metrics['total']:.4f}, "
                    f"Val Loss: {val_metrics['total']:.4f}, "
                    f"Move Acc: {val_metrics['move_acc']:.3f}, "
                    f"Turn Acc: {val_metrics['turn_acc']:.3f}"
                )

                # Save checkpoint
                is_best = val_metrics['total'] < (self.best_val_loss - self.early_stopping_min_delta)
                if is_best:
                    self.best_val_loss = val_metrics['total']
                    self.no_improve_epochs = 0
                else:
                    self.no_improve_epochs += 1
                self.save_checkpoint(epoch, is_best=is_best)

                # Save metrics
                self.train_losses.append(train_metrics)
                self.val_losses.append(val_metrics)
                self._save_metrics()

            # Early stopping must be synchronized across ranks to avoid DDP deadlocks.
            stop_training = False
            if self.is_main and self.early_stopping_patience > 0:
                stop_training = self.no_improve_epochs >= self.early_stopping_patience
                if stop_training:
                    logger.info(
                        "Early stopping triggered at epoch %d (patience=%d, min_delta=%.6f)",
                        epoch,
                        self.early_stopping_patience,
                        self.early_stopping_min_delta,
                    )

            stop_tensor = torch.tensor(
                1 if stop_training else 0,
                device=self.device,
                dtype=torch.int32,
            )
            if dist.is_available() and dist.is_initialized():
                dist.broadcast(stop_tensor, src=0)
            if stop_tensor.item() == 1:
                break

        logger.info("Training complete!")

    def _save_metrics(self) -> None:
        """Save training metrics to JSON."""
        metrics = {
            "train": self.train_losses,
            "val": self.val_losses,
        }
        metrics_path = self.checkpoint_dir / "metrics.json"
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)


def setup_ddp(rank: int, world_size: int, master_addr: str, master_port: int) -> None:
    """Initialize distributed training."""
    os.environ['MASTER_ADDR'] = master_addr
    os.environ['MASTER_PORT'] = str(master_port)
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


def cleanup_ddp() -> None:
    """Cleanup distributed training."""
    dist.destroy_process_group()


def train_worker(rank: int, world_size: int, config: dict) -> None:
    """Worker function for distributed training."""
    # Setup DDP
    setup_ddp(
        rank,
        world_size,
        master_addr=config.get("master_addr", "127.0.0.1"),
        master_port=int(config.get("master_port", 12355)),
    )
    device = torch.device(f"cuda:{rank}")

    # Create datasets with distributed sampler
    label_encoder = LabelEncoder()

    train_dataset = NavigationDataset(
        manifest_path=config["train_manifest"],
        dataset_root=config.get("dataset_root"),
        history_len=config.get("history_len", 1),
        image_size=tuple(config.get("image_size", [224, 224])),
        label_encoder=label_encoder,
    )

    val_dataset = NavigationDataset(
        manifest_path=config["val_manifest"],
        dataset_root=config.get("dataset_root"),
        history_len=config.get("history_len", 1),
        image_size=tuple(config.get("image_size", [224, 224])),
        label_encoder=label_encoder,
    )

    train_sampler = DistributedSampler(
        train_dataset, num_replicas=world_size, rank=rank, shuffle=True
    )
    val_sampler = DistributedSampler(
        val_dataset, num_replicas=world_size, rank=rank, shuffle=False
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.get("batch_size", 32),
        sampler=train_sampler,
        num_workers=config.get("num_workers", 4),
        pin_memory=True,
        drop_last=True,
        collate_fn=collate_fn,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.get("batch_size", 32),
        sampler=val_sampler,
        num_workers=config.get("num_workers", 4),
        pin_memory=True,
        collate_fn=collate_fn,
    )

    # Create model
    model = create_model(
        backbone=config.get("backbone", "resnet18"),
        n_move_classes=train_dataset.n_move_classes,
        n_turn_classes=train_dataset.n_turn_classes,
        history_len=config.get("history_len", 1),
        use_temporal=config.get("use_temporal", False),
        pretrained=config.get("pretrained", True),
        dropout=config.get("dropout", 0.3),
    )
    model = model.to(device)
    model = DDP(model, device_ids=[rank])

    # Optimizer
    optimizer = AdamW(
        model.parameters(),
        lr=config.get("lr", 1e-4),
        weight_decay=config.get("weight_decay", 1e-4),
    )

    # Scheduler: warmup + cosine decay tends to be much stabler on noisy imitation datasets.
    total_epochs = int(config.get("epochs", 100))
    warmup_epochs = int(config.get("warmup_epochs", 5))
    warmup_epochs = max(0, min(warmup_epochs, max(total_epochs - 1, 0)))
    if warmup_epochs > 0:
        warmup = LinearLR(
            optimizer,
            start_factor=float(config.get("warmup_start_factor", 0.2)),
            end_factor=1.0,
            total_iters=warmup_epochs,
        )
        cosine = CosineAnnealingLR(
            optimizer,
            T_max=max(1, total_epochs - warmup_epochs),
            eta_min=config.get("lr_min", 1e-6),
        )
        scheduler = SequentialLR(
            optimizer,
            schedulers=[warmup, cosine],
            milestones=[warmup_epochs],
        )
    else:
        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=max(1, total_epochs),
            eta_min=config.get("lr_min", 1e-6),
        )

    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        config=config,
        rank=rank,
        world_size=world_size,
    )

    try:
        # Train
        trainer.train(num_epochs=config.get("epochs", 100))
    finally:
        # Cleanup
        cleanup_ddp()


def main():
    parser = argparse.ArgumentParser(description="Train CS2 navigation model")
    parser.add_argument("--train-manifest", type=str, required=True, help="Path to train manifest")
    parser.add_argument("--val-manifest", type=str, required=True, help="Path to val manifest")
    parser.add_argument("--dataset-root", type=str, default=None, help="Dataset root directory")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints", help="Checkpoint directory")

    # Model args
    parser.add_argument("--backbone", type=str, default="resnet18", choices=["resnet18", "resnet34", "resnet50", "efficientnet_b0", "efficientnet_b1"])
    parser.add_argument("--history-len", type=int, default=1, help="Sequence length")
    parser.add_argument("--use-temporal", action="store_true", help="Use LSTM for temporal modeling")
    parser.add_argument("--dropout", type=float, default=0.3)

    # Training args
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size per GPU")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-accum-steps", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--image-size", type=int, nargs=2, default=[224, 224])
    parser.add_argument("--max-grad-norm", type=float, default=1.0, help="Gradient clipping norm (<=0 disables)")
    parser.add_argument("--label-smoothing", type=float, default=0.05, help="Label smoothing for move/turn CE loss")
    parser.add_argument("--mouse-loss", type=str, default="huber", choices=["huber", "mse"], help="Mouse regression loss")
    parser.add_argument("--mouse-huber-beta", type=float, default=1.0, help="Huber beta for mouse loss")
    parser.add_argument("--warmup-epochs", type=int, default=5, help="Warmup epochs before cosine decay")
    parser.add_argument("--warmup-start-factor", type=float, default=0.2, help="Initial LR factor at warmup start")
    parser.add_argument("--early-stopping-patience", type=int, default=12, help="Stop if val loss does not improve")
    parser.add_argument("--early-stopping-min-delta", type=float, default=1e-3, help="Minimum val loss improvement")

    # Loss weights
    parser.add_argument("--loss-weight-move", type=float, default=1.0)
    parser.add_argument("--loss-weight-turn", type=float, default=1.0)
    parser.add_argument("--loss-weight-jump", type=float, default=0.5)
    parser.add_argument("--loss-weight-crouch", type=float, default=0.5)
    parser.add_argument("--loss-weight-fire", type=float, default=0.5)
    parser.add_argument("--loss-weight-mouse", type=float, default=0.3)

    # System args
    parser.add_argument("--no-amp", action="store_true", help="Disable mixed precision")
    parser.add_argument("--world-size", type=int, default=2, help="Number of GPUs")
    parser.add_argument("--master-addr", type=str, default="127.0.0.1", help="DDP master address")
    parser.add_argument("--master-port", type=int, default=12355, help="DDP master port")

    args = parser.parse_args()

    # Build config dict
    config = vars(args)
    config["use_amp"] = not args.no_amp

    # Check GPU availability
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available")

    available_gpus = torch.cuda.device_count()
    if available_gpus < config["world_size"]:
        logger.warning(f"Requested {config['world_size']} GPUs but only {available_gpus} available")
        config["world_size"] = available_gpus

    logger.info(f"Starting training with {config['world_size']} GPUs")
    logger.info(f"Config: {json.dumps(config, indent=2)}")

    # Launch distributed training
    if config["world_size"] > 1:
        torch.multiprocessing.spawn(
            train_worker,
            args=(config["world_size"], config),
            nprocs=config["world_size"],
            join=True,
        )
    else:
        # Single GPU training
        train_worker(0, 1, config)


if __name__ == "__main__":
    main()

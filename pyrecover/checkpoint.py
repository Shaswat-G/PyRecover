"""
Module 1: Distributed Checkpointing System

Provides functionality for saving and loading distributed model checkpoints in PyTorch DDP/FSDP environments.

1. Ensure that after saving/loading, all ranks synchronize (e.g., using dist.barrier()) to avoid inconsistencies.
2. Use torch.save() and torch.load() for saving/loading model states, optimizers, and schedulers.
3. Use torch.distributed.get_rank() to get the current rank and save/load checkpoints accordingly, only Rank 0 should save the checkpoint.
4. Consider atomic file operations to avoid partial checkpoints.
5. Can Store metadata (epoch, step, etc.) in the checkpoint for easier recovery.
6. Add error handling for file I/O and distributed communication.

"""

import os
from pathlib import Path
from typing import Dict, Optional, Union, Any, Tuple

import torch


def save_ckpt(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    lr_scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
    sampler = None,
    step: int = 0,
    epoch: Optional[int] = None,
    checkpoint_path: str = "/iopstores/scratch",
    max_keep: int = 3,
    verify: bool = True,
) -> str:
    """Save distributed model checkpoint. Assumes only rank0 is doing it.
     Also supports storing non-distributed checkpoints.

    Args:
        model: PyTorch model (potentially ddp wrapped)
        optimizer: PyTorch optimizer
        lr_scheduler: Optional learning rate scheduler
        sampler: Optional sampler in distributed setting store sampler state of distributed sampler
        step: Current step number
        epoch: Optional epoch number
        checkpoint_path: path to store checkpoint to
        max_keep: Maximum number of checkpoints to keep, if exceeded, oldest checkpoints will be deleted.
        verify: Whether to verify checkpoint with checksums

    Returns:
        Path to saved checkpoint
    """
    # prepare state dict
    state = {
        "epoch": epoch,
        "step": step,
        "model": model.module.state_dict(),  # use .module for DDP-wrapped model
        "optimizer": optimizer.state_dict(),
        "lr_scheduler": lr_scheduler.state_dict(),
    }
    if sampler and hasattr(sampler, 'set_state'):
        state["sampler_state"] = sampler.state_dict()
    torch.save(state, checkpoint_path)
    # generate checksum
    if verify:
        import hashlib
        with open(checkpoint_path, 'rb') as f:
            file_hash = hashlib.md5(f.read()).hexdigest()
        checksum_path = f"{checkpoint_path}.md5"
        with open(checksum_path, 'w') as f:
            f.write(file_hash)
        print(f"Generated checksum file: {checksum_path}")

    # delete old checkpoints
    if max_keep > 0:
        ckpt_base_path = Path(checkpoint_path).parent
        ckpt_files = [x.relative_to(ckpt_base_path) for x in ckpt_base_path.glob('**/*') if x.is_file()]
        ckpt_files.sort()
        if len(ckpt_files) > max_keep:
            for f in ckpt_files[:-max_keep]:
                os.remove(os.path.join(checkpoint_path, f))
    return checkpoint_path


def load_ckpt(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    lr_scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
    sampler = None,
    checkpoint_path: str = "latest",
    experiment_dir: str = "/iopstores/scratch",
    verify: bool = True,
) -> Tuple[int, int]:
    """Load distributed model checkpoint. Also loads non-distributed checkpoints.
    Doesn't move model to GPU. This must be done in main loop.

    Args:
        model: PyTorch model (possibly wrapped in DDP)
        optimizer: PyTorch optimizer
        checkpoint_path: Path to checkpoint
        experiment_dir: given in case checkpoint_path is 'latest'. In that case path is needed to find latest
        lr_scheduler: Optional learning rate scheduler
        sampler: Optional sampler in distributed setting store sampler state of distributed sampler
        verify: Whether to verify checkpoint with checksums

    Returns:
        Return epoch and step number loaded from checkpoint.
    """
    if checkpoint_path == "latest":
        checkpoint_path = get_latest_checkpoint(experiment_dir)
        if checkpoint_path is None:
            raise RuntimeError(f"No checkpoint found in {experiment_dir}")
    # verify checksum if needed
    if verify:
        import hashlib
        with open(checkpoint_path, 'rb') as f:
            file_hash = hashlib.md5(f.read()).hexdigest()
        checksum_path = f"{checkpoint_path}.md5"
        with open(checksum_path, 'r') as f:
            ckpt_hash = f.read()
        if ckpt_hash != file_hash:
            raise RuntimeError(f"Checksum mismatch for checkpoint {checkpoint_path}")
        print(f"Checksum verified for checkpoint {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    model.module.load_state_dict(checkpoint["model"])  # .module for DDP
    optimizer.load_state_dict(checkpoint["optimizer"])
    lr_scheduler.load_state_dict(checkpoint["scheduler"])

    epoch = checkpoint.get("epoch", 0)
    step = checkpoint.get("step", 0)

    if sampler and "sampler_state" in checkpoint:
        sampler.load_state_dict(checkpoint["sampler_state"])

    print(f"Checkpoint loaded from {checkpoint_path} (epoch {epoch}, step {step})")
    return epoch, step


def get_latest_checkpoint(checkpoint_dir: str) -> Optional[str]:
    """Get latest checkpoint in directory. Can be experiment checkpoint directory"""
    ckpt_base_path = Path(checkpoint_dir)
    ckpt_files = [x.relative_to(ckpt_base_path) for x in ckpt_base_path.glob('**/*') if x.is_file()]
    ckpt_files.sort()
    if len(ckpt_files) > 0:
        return os.path.join(checkpoint_dir, ckpt_files[-1])
    else:
        return None
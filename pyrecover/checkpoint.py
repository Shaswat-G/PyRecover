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
from typing import Optional, Tuple

import torch
import torch.distributed as dist


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
    is_distributed: bool = False,
    rank: int = 0,
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
    if is_distributed:
        dist.barrier()

    if rank == 0 or not is_distributed:
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
            ckpt_files = [x.relative_to(ckpt_base_path) for x in ckpt_base_path.glob('**/*.pt') if x.is_file()]
            ckpt_files.sort()
            if len(ckpt_files) > max_keep:
                for f in ckpt_files[:-max_keep]:
                    specific_ckpt_path = os.path.join(ckpt_base_path, f)
                    os.remove(specific_ckpt_path)
                    checksum_file = Path(specific_ckpt_path, ".md5")
                    if checksum_file.exists():
                        os.remove(checksum_file)
    if is_distributed:
        dist.barrier()

    return checkpoint_path


def load_ckpt(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    lr_scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
    sampler = None,
    checkpoint_path: str = "latest",
    experiment_dir: str = "/iopstores/scratch",
    verify: bool = True,
    is_distributed: bool = False,
    rank: int = 0,
) -> Tuple[int, int]:
    """
    Load distributed model checkpoint. Also loads non-distributed checkpoints.
    Doesn't move model to GPU. This must be done in main loop.

    Args:
        is_distributed: flag indicating whether distributed training is used.
        rank: rank of the process. Rank 0 is by default also if training local.
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

    checkpoint = None
    checkpoint_keys = None
    if rank == 0:
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
        # load checkpoint only rank 0
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        checkpoint_keys = list(checkpoint.keys())

        # Load model state dictionary
        if hasattr(model, 'module'):
            model.module.load_state_dict(checkpoint["model"])
        else:
            model.load_state_dict(checkpoint["model"])

        # Load optimizer state
        optimizer.load_state_dict(checkpoint["optimizer"])

        # Load lr_scheduler state if present
        if lr_scheduler is not None and "lr_scheduler" in checkpoint:
            lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])

        # Load sampler state if present
        if sampler and "sampler_state" in checkpoint:
            sampler.load_state_dict(checkpoint["sampler_state"])

        epoch = checkpoint.get("epoch", 0)
        step = checkpoint.get("step", 0)
    else:
        epoch = step = None

    if is_distributed:
        # Broadcast list of checkpoint keys for guards
        if rank == 0:
            key_list = checkpoint_keys
        else:
            key_list = None
        key_list = [key_list]
        dist.broadcast_object_list(key_list, src=0)
        checkpoint_keys = key_list[0]

        # Broadcast model weights
        for param in model.parameters():
            dist.broadcast(param.data, 0)

        # Broadcast epoch and step
        if rank == 0:
            epoch_step = torch.tensor([epoch, step], dtype=torch.long)
        else:
            epoch_step = torch.zeros(2, dtype=torch.long)
        dist.broadcast(epoch_step, 0)
        if rank != 0:
            epoch, step = epoch_step.tolist()
            print(f"Rank {rank}: Model parameters received via broadcast (epoch {epoch}, step {step})")

        # Broadcast and load lr_scheduler state if present
        if lr_scheduler is not None and "lr_scheduler" in checkpoint_keys:
            if rank == 0:
                sched_state = lr_scheduler.state_dict()
            else:
                sched_state = None
            sched_state = [sched_state]
            dist.broadcast_object_list(sched_state, src=0)
            if rank != 0:
                lr_scheduler.load_state_dict(sched_state[0])

        # Broadcast and load sampler state if present
        if sampler and "sampler_state" in checkpoint_keys:
            if rank == 0:
                sampler_state = sampler.state_dict()
            else:
                sampler_state = None
            sampler_state = [sampler_state]
            dist.broadcast_object_list(sampler_state, src=0)
            if rank != 0:
                sampler.load_state_dict(sampler_state[0])

        # Broadcast and load optimizer state
        if "optimizer" in checkpoint_keys:
            if rank == 0:
                opt_state = optimizer.state_dict()
            else:
                opt_state = None
            opt_state = [opt_state]
            dist.broadcast_object_list(opt_state, src=0)
            if rank != 0:
                optimizer.load_state_dict(opt_state[0])

        dist.barrier()

    print(f"Checkpoint loaded from {checkpoint_path} (epoch {epoch}, step {step})")
    return epoch, step


def get_latest_checkpoint(checkpoint_dir: str) -> Optional[str]:
    """Get latest checkpoint in directory. Can be experiment checkpoint directory"""
    ckpt_base_path = Path(checkpoint_dir)
    ckpt_files = [x.relative_to(ckpt_base_path) for x in ckpt_base_path.glob('**/*.pt') if x.is_file()]
    ckpt_files.sort()
    if len(ckpt_files) > 0:
        return os.path.join(checkpoint_dir, ckpt_files[-1])
    else:
        return None
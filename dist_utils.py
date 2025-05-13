"""
dist_utils.py: Distributed Training Utilities. Mostly related to process group and device setup as well as logging.
"""

import os
from typing import Tuple

import torch
from utils import logger

# --- DDP minimal setup ---
def is_distributed() -> bool:
    return "SLURM_PROCID" in os.environ and int(os.environ.get("SLURM_NTASKS", "1")) > 1


def get_rank() -> int:
    if is_distributed():
        import torch.distributed as dist

        return dist.get_rank()
    return 0


def is_rank_eq(rank: int) -> bool:
    return get_rank() == rank


def is_rank0() -> bool:
    return get_rank() == 0


def maybe_init_distributed() -> Tuple[int, int]:
    if is_distributed():
        import torch.distributed as dist

        rank = int(os.environ["SLURM_PROCID"])
        world_size = int(os.environ["SLURM_NTASKS"])
        local_rank = int(os.environ.get("SLURM_LOCALID", 0))
        os.environ["MASTER_ADDR"] = os.environ.get("MASTER_ADDR", "127.0.0.1")
        os.environ["MASTER_PORT"] = os.environ.get("MASTER_PORT", "29500")
        dist.init_process_group(
            backend="nccl",
            rank=rank,
            world_size=world_size,
        )
        torch.cuda.set_device(local_rank)
        print(
            f"[Rank {get_rank()}] world_size={world_size}, local_rank={local_rank}, pid={os.getpid()}",
            flush=True,
        )
        log_rank0(
                f"DDP initialized: world_size={world_size}, local_rank={local_rank}, rank={get_rank()}, pid={os.getpid()}"
            )
        return local_rank, world_size
    else:
        return 0, 1


def maybe_cleanup_distributed():
    if is_distributed():
        import torch.distributed as dist

        dist.destroy_process_group()
        log_rank0("DDP cleaned-up!")


# --- end DDP minimal setup ---

def log_rank(msg, rank):
    if is_rank_eq(rank):
        logger.info(msg)

def log_rank0(msg):
    log_rank(msg, 0)
"""
dist_utils.py: Distributed Training Utilities. Mostly related to process group and device setup as well as logging.
"""

import os
from typing import Tuple

import torch
from utils import logger

# --- DDP minimal setup ---
def is_distributed_slurm_env() -> bool:
    return "SLURM_PROCID" in os.environ and int(os.environ.get("SLURM_NTASKS", "1")) > 1

def is_distributed_activated() -> bool:
    return "DISTRIBUTED_RUN" in os.environ


def get_rank() -> int:
    if is_distributed_slurm_env() and is_distributed_activated():
        import torch.distributed as dist

        return dist.get_rank()
    return 0


def is_rank_eq(rank: int) -> bool:
    return get_rank() == rank


def is_rank0() -> bool:
    return get_rank() == 0


def maybe_init_distributed(activate_distributed: bool) -> Tuple[int, int]:
    if activate_distributed:
        os.environ["DISTRIBUTED_RUN"] = "1"
    if activate_distributed and is_distributed_slurm_env():
        print("--DETECTED DISTRIBUTED SETUP, DO DISTRIBUTED TRAINING!--")
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
    elif activate_distributed and not is_distributed_slurm_env():
        exit("Try running distributed training but environment is not setup for this!")
    else:
        print("--NOT RUNNING DISTRIBUTED TRAINING!--")
        return 0, 1


def maybe_cleanup_distributed():
    if is_distributed_activated() and is_distributed_slurm_env():
        import torch.distributed as dist
        # Wait for all processes to be finished and then destroy
        dist.barrier()
        dist.destroy_process_group()
        print("DDP cleaned-up!")


# --- end DDP minimal setup ---

def log_rank(msg, rank):
    if not is_distributed_activated() or is_rank_eq(rank):
        logger.info(msg)

def log_rank0(msg):
    log_rank(msg, 0)

def get_slurm_job_end_time_env() -> float:
    """Return SLURM_JOB_END_TIME as a float (UNIX timestamp), or None if not set or invalid."""
    val = os.environ.get("SLURM_JOB_END_TIME")
    if val is not None:
        try:
            return float(val)
        except ValueError:
            pass
    return None
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from dataset import CollatorForCLM, ParquetDataset
from dist_utils import (
    maybe_init_distributed,
    is_rank0,
    get_rank,
    maybe_cleanup_distributed,
    log_rank0,
    is_distributed_activated,
    get_slurm_job_end_time_env,  # import the helper
)
from model import Transformer, TransformerModelArgs
from utils import (
    build_lr_scheduler,
    get_args,
    get_num_params,
    get_num_flop_per_token,
    init_logger,
    PRECISION_STR_TO_DTYPE,
    set_default_dtype,
)
from pyrecover.checkpoint import (
    save_ckpt_vanilla,
    load_ckpt_vanilla,
    save_ckpt_distributed,
    load_ckpt_distributed,
)


def train(args):
    # Set up distributed training if activated and Slurm env set!
    local_rank, world_size = maybe_init_distributed(args.distributed)
    log_rank0(f"Experiment args: {args}")
    # Init
    device = torch.device(f"cuda:{local_rank}")
    model_dtype = PRECISION_STR_TO_DTYPE[args.model_dtype]

    # Set up DataLoader
    log_rank0("Setting up DataLoaders...")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name_or_path)
    train_ds = ParquetDataset(
        args.dataset,
        tokenizer,
        args.sequence_length,
        args.batch_size * args.training_steps,
    )
    # set batch size
    global_batch_size = int(args.batch_size)
    local_batch_size = max(global_batch_size // world_size, 1)
    log_rank0(
        f"Global batch size: {global_batch_size}\nLocal batch size: {local_batch_size}"
    )
    if world_size > 1:
        # Set Distributed Sampler for DDP training
        from torch.utils.data.distributed import DistributedSampler

        train_sampler = DistributedSampler(
            train_ds, num_replicas=world_size, rank=get_rank(), shuffle=True
        )
    else:
        train_sampler = None
    train_collator = CollatorForCLM(args.sequence_length, tokenizer.pad_token_id)
    train_dl = DataLoader(
        train_ds,
        batch_size=local_batch_size,
        collate_fn=train_collator,
        sampler=train_sampler if train_sampler is not None else None,
        shuffle=(train_sampler is None),
    )
    train_dl_iterator = iter(train_dl)

    # Set up Model
    log_rank0("Setting up Model with default config...")
    model_config = TransformerModelArgs(
        dim=4096,
        n_layers=32,
        n_heads=32,
        n_kv_heads=8,
        ffn_dim_multiplier=1.3,
        multiple_of=1024,
        rope_theta=500000,
        vocab_size=tokenizer.vocab_size,
        seq_len=args.sequence_length,
    )
    with set_default_dtype(model_dtype):
        model = Transformer(model_config).to(device)

    if args.compile:
        log_rank0("Using `torch.compile`")
        model = torch.compile(model, fullgraph=True)

    if world_size > 1:
        from torch.nn.parallel import DistributedDataParallel as DDP

        model = DDP(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=False,
        )

    model.train()

    # Build Optimizers & LR Scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, fused=args.fused_optimizer
    )
    lr_scheduler = build_lr_scheduler(optimizer, args.lr_warmup_steps)

    # Utils
    num_flop_per_token = get_num_flop_per_token(
        get_num_params(model, exclude_embedding=True),
        model_config,
    )

    ntokens_since_last_log = 0
    ntraining_tokens_since_last_log = 0
    time_last_log = time.perf_counter()

    # Setup checkpoint dir
    checkpoint_freq_steps = int(args.checkpoint_frequency)
    ckpt_path = Path(args.checkpoint_dir)
    if ckpt_path.exists() and not ckpt_path.is_dir():
        exit(f"Checkpoint dir {ckpt_path} exists as file already! Abort!")
    exp_ckpt_path = ckpt_path / args.experiment_name
    exp_ckpt_path.mkdir(parents=True, exist_ok=True)

    # Select checkpoint save/load functions based on args
    if args.use_torch_distributed_ckpt:
        log_rank0("Using torch.distributed.checkpoint for checkpointing")
        save_ckpt_fn = save_ckpt_distributed
        load_ckpt_fn = load_ckpt_distributed
    else:
        log_rank0("Using vanilla PyTorch checkpointing")
        save_ckpt_fn = save_ckpt_vanilla
        load_ckpt_fn = load_ckpt_vanilla

    # Initialize max_iter_time and max_ckpt_time from args
    max_iter_time = float(args.default_iter_time)
    max_ckpt_time = float(args.default_ckpt_time)
    buffer_time = 5 * max_iter_time + 1 * max_ckpt_time
    log_rank0(
        f"Initial max_iter_time: {max_iter_time}, max_ckpt_time: {max_ckpt_time}, buffer_time: {buffer_time}"
    )

    # Read SLURM job end time from environment
    job_end_time = get_slurm_job_end_time_env()
    log_rank0(f"SLURM_JOB_END_TIME: {job_end_time}")

    # load checkpoint if wanted
    train_step = 0
    epoch = 1
    if args.resume_from_checkpoint is not None:
        log_rank0(f"Try resume from checkpoint {args.resume_from_checkpoint}")
        # Measure checkpoint loading time
        checkpoint_load_start = time.perf_counter()
        epoch, train_step = load_ckpt_fn(
            model,
            optimizer,
            lr_scheduler,
            train_sampler,
            args.resume_from_checkpoint,
            experiment_dir=exp_ckpt_path,
            verify=args.verify_checkpoints,
            is_distributed=(world_size > 1),
            rank=get_rank(),
        )
        checkpoint_load_time = time.perf_counter() - checkpoint_load_start
        log_rank0(f"Checkpoint loading completed in {checkpoint_load_time:.2f} seconds")

    if is_distributed_activated():
        torch.distributed.barrier()

    should_stop = False  # Initialize the stop flag before the training loop

    log_rank0("Starting training!")
    while train_step < args.training_steps:
        train_step += 1

        # Time checker at the beginning of the loop (rank0 only)
        if is_rank0() and job_end_time is not None:
            now = time.time()
            time_left = job_end_time - now
            threshold_time = max_iter_time + max_ckpt_time + buffer_time
            if time_left < threshold_time:
                should_stop = True
                log_rank0(
                    f"[TIME CHECK] Remaining time ({time_left:.2f}s) < threshold ({threshold_time:.2f}s). should_stop set to True."
                )

        if should_stop:
            log_rank0("[TIME CHECK] Stopping training early and checkpointing due to time limit.")
            break

        iter_start = time.perf_counter()

        # Profiling
        if args.profile and args.profile_step_start == train_step:
            torch.cuda.cudart().cudaProfilerStart()
            torch.autograd.profiler.emit_nvtx(record_shapes=True).__enter__()

        if "train_sampler" in locals() and train_sampler is not None:
            train_sampler.set_epoch(epoch)

        # restart with next epoch if needed
        try:
            input_ids, labels = next(train_dl_iterator)
        except StopIteration as err:
            train_dl_iterator = iter(train_dl)
            epoch += 1

        # capture metrics
        ntokens_since_last_log += global_batch_size * args.sequence_length
        num_items_in_batch = labels.ne(-100).sum()
        ntraining_tokens_since_last_log += num_items_in_batch.to("cpu") * world_size

        # move inputs and labels to device
        input_ids = input_ids.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        logits = model(input_ids)
        loss = torch.nn.functional.cross_entropy(
            logits.flatten(0, 1).float(), labels.flatten(0, 1), reduction="sum"
        )
        loss = loss / num_items_in_batch
        del logits
        # In ddp setting, gradients are synced here
        loss.backward()

        # Clip gradients
        # clip_grad_norm_(model.parameters(), args.grad_max_norm)

        optimizer.step()
        lr_scheduler.step()

        # Logging
        if train_step == 1 or train_step % args.logging_frequency == 0:
            time_delta = time.perf_counter() - time_last_log
            # tokens per second per device, abbreviated as tps
            tps = ntokens_since_last_log / time_delta
            mfu = 100 * num_flop_per_token * tps / 989e12
            tflops = num_flop_per_token * tps / 1e12
            training_tps = ntraining_tokens_since_last_log / time_delta

            log_rank0(
                f"Epoch: {epoch} | Step: {train_step} | Loss: {loss.item():.2f} | Tokens per second: {tps:.2f} | Training tokens per second (%): {100*training_tps/tps:.2f} | MFU (%): {mfu:.2f} | TFLOPs: {tflops:.2f}"
            )
            ntokens_since_last_log = 0
            ntraining_tokens_since_last_log = 0
            time_last_log = time.perf_counter()

        # Track max_iter_time
        iter_time = time.perf_counter() - iter_start
        if iter_time > max_iter_time:
            max_iter_time = iter_time
            log_rank0(f"Updated max_iter_time: {max_iter_time}")
        buffer_time = 5 * max_iter_time + 1 * max_ckpt_time
        # Optionally log buffer_time for debugging
        log_rank0(f"Current buffer_time: {buffer_time}")

        # Checkpointing
        if checkpoint_freq_steps != -1 and train_step % checkpoint_freq_steps == 0:
            if args.use_torch_distributed_ckpt:
                # For distributed checkpointing, use directories instead of files
                specific_ckpt_path = exp_ckpt_path / f"ckpt_{train_step}"
            else:
                specific_ckpt_path = exp_ckpt_path / f"ckpt_{train_step}.pt"

            log_rank0(f"Saving checkpoint to {specific_ckpt_path}")
            checkpoint_store_start = time.perf_counter()
            save_ckpt_fn(
                model,
                optimizer,
                lr_scheduler,
                train_sampler,
                train_step,
                epoch,
                specific_ckpt_path,
                max_keep=args.max_kept_checkpoints,
                verify=args.verify_checkpoints,
                is_distributed=(world_size > 1),
                rank=get_rank(),
            )
            checkpoint_store_time = time.perf_counter() - checkpoint_store_start
            if checkpoint_store_time > max_ckpt_time:
                max_ckpt_time = checkpoint_store_time
                log_rank0(f"Updated max_ckpt_time: {max_ckpt_time}")
            log_rank0(
                f"Checkpoint store completed in {checkpoint_store_time:.2f} seconds"
            )

        # Profiling
        if args.profile and args.profile_step_end == train_step:
            torch.cuda.cudart().cudaProfilerStop()

    log_rank0("Training completed")
    maybe_cleanup_distributed()


if __name__ == "__main__":
    init_logger()
    args = get_args()
    train(args)

#!/usr/bin/env python
"""
check_weights_equality.py: loads two checkpoints and checks if the model weights in these two checkpoints are equal.
                           Checkpoints should be trained with the same hyperparameters and same seed.
"""

import argparse
import sys
import time
from typing import Dict

import torch
import torch.nn as nn

# Import checkpoint methods from pyrecover
from pyrecover.checkpoint import (
    load_ckpt_distributed,
)


# Define a simple dummy model for loading checkpoint state
class DummyModel(nn.Module):
    def __init__(self):
        super(DummyModel, self).__init__()
        self.dummy_param = nn.Parameter(torch.zeros(1))

    @property
    def device(self):
        return self.dummy_param.device

    def forward(self, x):
        return x


# Define a dummy optimizer
class DummyOptimizer:
    def __init__(self, model):
        self.model = model

    def load_state_dict(self, state_dict):
        pass

    def state_dict(self):
        return {}


# Define a dummy scheduler
class DummyScheduler:
    def __init__(self):
        pass

    def load_state_dict(self, state_dict):
        pass

    def state_dict(self):
        return {}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Check if model weights are equal between two checkpoints"
    )
    parser.add_argument("checkpoint1", type=str, help="Path to first checkpoint")
    parser.add_argument("checkpoint2", type=str, help="Path to second checkpoint")
    parser.add_argument(
        "--distributed",
        action="store_true",
        help="Use distributed checkpoint loading functions",
    )
    parser.add_argument(
        "--tolerance", type=float, default=1e-7, help="Tolerance for weight comparison"
    )
    parser.add_argument(
        "--verbose", action="store_true", help="Print detailed comparison results"
    )
    return parser.parse_args()


def extract_model_state_dict(
    checkpoint_path: str, distributed: bool
) -> Dict[str, torch.Tensor]:
    """
    Load a checkpoint and extract the model state dict using the appropriate loading function
    """
    # Create dummy objects needed for loading checkpoint
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dummy_model = DummyModel().to(device)
    dummy_optimizer = DummyOptimizer(dummy_model)
    dummy_scheduler = DummyScheduler()

    # Use the appropriate loading function based on checkpoint type
    if distributed:
        # For distributed checkpoints
        try:
            load_ckpt_distributed(
                model=dummy_model,
                optimizer=dummy_optimizer,
                lr_scheduler=dummy_scheduler,
                checkpoint_path=checkpoint_path,
                verify=False,
            )
        except Exception as e:
            print(f"Error loading distributed checkpoint: {e}")
            raise

        if hasattr(dummy_model, "module"):
            return dummy_model.module.state_dict()
        else:
            return dummy_model.state_dict()
    else:
        # For vanilla checkpoints
        try:
            # Use direct loading to extract just the model state dict without loading into model
            checkpoint = torch.load(checkpoint_path, map_location=device)
            return checkpoint["model"]
        except Exception as e:
            print(f"Error loading vanilla checkpoint: {e}")
            raise


def compare_weights(
    state_dict1: Dict[str, torch.Tensor],
    state_dict2: Dict[str, torch.Tensor],
    tolerance: float = 1e-7,
    verbose: bool = False,
) -> bool:
    """
    Compare two model state dictionaries to check if weights are equal within tolerance
    """
    print(f"Comparing weights with tolerance {tolerance}...")

    # First check if both state dicts have the same keys
    keys1 = set(state_dict1.keys())
    keys2 = set(state_dict2.keys())

    if keys1 != keys2:
        print("❌ State dictionaries have different keys:")
        if verbose:
            missing_in_2 = keys1 - keys2
            missing_in_1 = keys2 - keys1
            if missing_in_2:
                print(f"Keys in first checkpoint but not in second: {missing_in_2}")
            if missing_in_1:
                print(f"Keys in second checkpoint but not in first: {missing_in_1}")
        return False

    # Compare each parameter tensor
    all_equal = True
    max_diff = 0.0
    diff_param_count = 0
    total_params = len(keys1)

    for key in keys1:
        tensor1 = state_dict1[key]
        tensor2 = state_dict2[key]

        # Check shapes first
        if tensor1.shape != tensor2.shape:
            print(
                f"❌ Shape mismatch for parameter '{key}': {tensor1.shape} vs {tensor2.shape}"
            )
            all_equal = False
            diff_param_count += 1
            continue

        # Calculate absolute difference
        diff = torch.abs(tensor1 - tensor2)
        max_diff_param = diff.max().item()
        max_diff = max(max_diff, max_diff_param)

        # Check if difference exceeds tolerance
        if max_diff_param > tolerance:
            if verbose:
                print(
                    f"❌ Parameter '{key}' has difference {max_diff_param:.8e} > tolerance {tolerance:.8e}"
                )
            all_equal = False
            diff_param_count += 1

    # Report results
    if all_equal:
        print(
            f"✅ All {total_params} parameters are equal within tolerance {tolerance}"
        )
        print(f"Maximum difference: {max_diff:.8e}")
    else:
        print(
            f"❌ {diff_param_count} out of {total_params} parameters differ beyond tolerance {tolerance}"
        )
        print(f"Maximum difference: {max_diff:.8e}")

    return all_equal


def main():
    args = parse_args()

    start_time = time.time()

    print(f"Checkpoint 1: {args.checkpoint1}")
    print(f"Checkpoint 2: {args.checkpoint2}")
    print(
        f"Using {'distributed' if args.distributed else 'vanilla'} checkpoint loading"
    )

    try:
        # Load first checkpoint
        print(f"Loading first checkpoint from {args.checkpoint1}...")
        state_dict1 = extract_model_state_dict(args.checkpoint1, args.distributed)

        # Load second checkpoint
        print(f"Loading second checkpoint from {args.checkpoint2}...")
        state_dict2 = extract_model_state_dict(args.checkpoint2, args.distributed)

        # Compare weights
        result = compare_weights(
            state_dict1, state_dict2, tolerance=args.tolerance, verbose=args.verbose
        )

        elapsed_time = time.time() - start_time
        print(f"Time elapsed: {elapsed_time:.2f} seconds")

        # Exit with appropriate code (0 for equal, 1 for different)
        sys.exit(0 if result else 1)

    except Exception as e:
        print(f"Error during checkpoint comparison: {e}")
        sys.exit(2)


if __name__ == "__main__":
    main()

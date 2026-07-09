"""Train a scalar-field INR using the file-backed CPU samplers.

This script is a thin wrapper around :func:`train.train` that exercises the two
CPU-backed pysampler backends added in PLAN_1:

  * ``virtual_memory`` - mmap-based random access; voxels are paged in lazily
                         by the OS.  Best for reliable local SSD/NVMe.
  * ``out_of_core``    - explicit heap-resident block cache loaded via pread().
                         Safe on slow / unreliable storage (network mounts,
                         spinning HDDs that may time out); a transient I/O
                         error surfaces as a Python exception, never SIGBUS.

Both backends use "normalize-then-trilinear" semantics (matching InstantVNR's
samplers of the same names) and require an explicit ``range=[vmin, vmax]`` to
normalize voxel values into ``[0, 1]`` before interpolation.

Usage:
  # Train with the virtual_memory sampler
  python train_file_backed.py \
      --filename /path/to/volume.raw --dims 256 256 256 \
      --dtype float32 --range -1.0 1.0 --mode virtual_memory \
      --expname my_volume

  # Train with the out_of_core sampler
  python train_file_backed.py \
      --filename /path/to/volume.raw --dims 256 256 256 \
      --dtype uint16 --range 0 4095 --mode out_of_core

  # Train both back to back; outputs are suffixed with the backend name
  python train_file_backed.py \
      --filename /path/to/volume.raw --dims 256 256 256 \
      --dtype float32 --range -1.0 1.0 --mode both

Outputs (per backend):
  logs/<expname>__<backend>/run<N>/  TensorBoard event files
  outputs/<expname>__<backend>.pt    PyTorch state_dict
  outputs/<expname>__<backend>.bson  BSON scene file (model + macrocell)
"""

import faulthandler
import os
import sys
import argparse

import numpy as np

np.random.seed(0)

# faulthandler.enable() already installs handlers for SIGSEGV / SIGFPE /
# SIGABRT / SIGBUS / SIGILL, so a fatal signal will dump a Python stack trace
# to stderr before the process terminates -- handy for telling apart crashes
# in the sampler, tcnn, CUDA, or PyTorch when running on slow storage.
faulthandler.enable()

sys.path.insert(0, os.path.dirname(__file__))

from inrtoolkit import create_sampler
from train import train, _ParseDims


SAMPLER_BACKENDS = ("virtual_memory", "out_of_core")


# ---------------------------------------------------------------------------
# Sampler construction
# ---------------------------------------------------------------------------

def build_sampler(backend, args):
    """Create a file-backed sampler with the explicit range required by the backend."""
    if backend not in SAMPLER_BACKENDS:
        raise ValueError(
            f"unsupported file-backed backend: {backend!r}. "
            f"Expected one of: {', '.join(SAMPLER_BACKENDS)}"
        )

    return create_sampler(
        "structuredRegular", backend,
        dims          = args.dims,
        dtype         = args.dtype,
        n_channels    = 1,
        filename      = args.filename,
        offset        = args.offset,
        is_big_endian = False,  # file-backed samplers reject big-endian input
        range         = list(args.range),
    )


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def run(args, backend):
    """Run training for a single backend and return the experiment name used."""
    expname = f"{args.expname}__{backend}"

    print()
    print("=" * 72)
    print(f"[file-backed] training with backend = {backend}")
    print(f"[file-backed] expname = {expname}")
    print(f"[file-backed] dims = {args.dims}  dtype = {args.dtype}  range = {tuple(args.range)}")
    print(f"[file-backed] filename = {args.filename}  offset = {args.offset}")
    print("=" * 72)

    sampler = build_sampler(backend, args)
    train(expname, sampler, args.dims, args, output_dir=args.output_dir)
    return expname


def main():
    parser = argparse.ArgumentParser(
        description="Train a scalar-field INR using pysampler's file-backed CPU samplers",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Volume input
    parser.add_argument("--filename", required=True,
                        help="path to the raw volume file")
    parser.add_argument("--dims", nargs="+", default=[128, 128, 128], action=_ParseDims,
                        help='volume dims: "Dx Dy Dz" or "DxxDyxDz"')
    parser.add_argument("--dtype", default="float32",
                        choices=["float32", "uint8", "uint16", "int8", "int16",
                                 "uint32", "int32", "float64"],
                        help="voxel data type for the raw file")
    parser.add_argument("--range", type=float, nargs=2, required=True,
                        metavar=("VMIN", "VMAX"),
                        help="explicit normalization range (required by the file-backed samplers)")
    parser.add_argument("--offset", type=int, default=0,
                        help="byte offset into the raw file (skip a fixed-size header)")

    # Sampler selection
    parser.add_argument(
        "--mode",
        default="virtual_memory",
        choices=[*SAMPLER_BACKENDS, "both"],
        help='which file-backed sampler to use; "both" runs the two backends sequentially',
    )

    # Output
    parser.add_argument("--expname", default=None,
                        help="experiment-name stem (default: volume filename stem)")
    parser.add_argument("--output-dir", default="outputs",
                        help="directory for .pt and .bson outputs (created if missing)")

    # Training schedule
    parser.add_argument("--epochs", type=int, default=64,
                        help="training epochs")
    parser.add_argument("--batch-size", type=int, default=64 * 1024,
                        help="samples per training step")
    parser.add_argument("--lr", type=float, default=1e-2,
                        help="initial Adam learning rate")

    # Network architecture
    parser.add_argument("--n-levels", type=int, default=16,
                        help="number of hash-grid levels")
    parser.add_argument("--n-features", type=int, default=8,
                        help="features per hash level (input dim = n_levels * n_features)")
    parser.add_argument("--log2-hashmap-size", type=int, default=19,
                        help="log2 of hash table size per level")
    parser.add_argument("--n-neurons", type=int, default=64,
                        help="MLP hidden layer width (must be a multiple of 16)")
    parser.add_argument("--hidden-layers", type=int, default=4,
                        help="number of MLP hidden layers")

    args = parser.parse_args()

    if args.range[0] >= args.range[1]:
        parser.error(f"--range must satisfy VMIN < VMAX, got {tuple(args.range)}")

    args.expname = args.expname or os.path.splitext(os.path.basename(args.filename))[0]

    backends = SAMPLER_BACKENDS if args.mode == "both" else (args.mode,)
    completed = []
    for backend in backends:
        completed.append(run(args, backend))

    print()
    print("[file-backed] completed runs:")
    for expname in completed:
        print(f"  - {expname}: {args.output_dir}/{expname}.pt | {args.output_dir}/{expname}.bson")


if __name__ == "__main__":
    main()

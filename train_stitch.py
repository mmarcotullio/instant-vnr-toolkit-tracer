"""Train a scalar-field INR on an ExaStitch volume.

This script is a thin wrapper around :func:`train.train` for stitch datasets
loaded by pysampler's ExaStitch CUDA backend.

Usage:
  python train_stitch.py \\
      --umesh /path/to/dataset.umesh \\
      --grids /path/to/dataset.grids \\
      --scalar /path/to/dataset.scalar \\
      --dims 512 512 512 \\
      --expname my_stitch_volume

Outputs:
  logs/<expname>/run<N>/   TensorBoard event files
  outputs/<expname>.pt     PyTorch state_dict
  outputs/<expname>.bson   BSON scene file (model + macrocell)
"""

import argparse
import faulthandler
import os
import sys

import numpy as np

np.random.seed(0)

sys.path.insert(0, os.path.dirname(__file__))

from inrtoolkit import create_sampler
from train import DEVICE, train, _ParseDims


faulthandler.enable()


def _default_expname(args):
    for filename in (args.scalar, args.umesh, args.grids):
        if filename:
            return os.path.splitext(os.path.basename(filename))[0]
    return "stitch"


def build_sampler(args):
    """Create a CUDA ExaStitch sampler from the stitch dataset files."""
    return create_sampler(
        "exastitch", "cuda",
        umesh=args.umesh,
        grids=args.grids,
        scalar=args.scalar,
    )


def main():
    parser = argparse.ArgumentParser(
        description="Train a scalar-field INR on an ExaStitch volume",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Volume input. The binding accepts both names; pass an empty string for
    # whichever stitch representation the dataset does not provide.
    parser.add_argument("--umesh", default="",
                        help="path to the ExaStitch .umesh file")
    parser.add_argument("--grids", default="",
                        help="path to the ExaStitch .grids file")
    parser.add_argument("--scalar", required=True,
                        help="path to the ExaStitch .scalar file")
    parser.add_argument("--dims", nargs="+", required=True, action=_ParseDims,
                        help='training/export dims: "Dx Dy Dz" or "DxxDyxDz"')

    # Output
    parser.add_argument("--expname", default=None,
                        help="experiment name (default: scalar filename stem)")
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

    if not args.umesh and not args.grids:
        parser.error("at least one of --umesh or --grids is required")

    if DEVICE.type != "cuda":
        parser.error("ExaStitch training requires a CUDA device")

    expname = args.expname or _default_expname(args)

    print(f"[stitch] sampler input: umesh={args.umesh or '<none>'}  grids={args.grids or '<none>'}")
    print(f"[stitch] scalar={args.scalar}  export_dims={args.dims}  device=cuda")

    sampler = build_sampler(args)
    train(expname, sampler, args.dims, args, output_dir=args.output_dir)


if __name__ == "__main__":
    main()

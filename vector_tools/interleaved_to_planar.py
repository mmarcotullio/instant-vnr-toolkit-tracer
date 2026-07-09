"""Convert an interleaved multi-channel .raw file to planar layout.

Interleaved layout (e.g. uvwuvwuvw...):
    element order: [u0,v0,w0, u1,v1,w1, ..., uN,vN,wN]

Planar layout (required by the CUDA sampler with n_channels > 1):
    element order: [u0,u1,...,uN, v0,v1,...,vN, w0,w1,...,wN]

Usage:
    python interleaved_to_planar.py \\
        --filename velocity_interleaved.raw \\
        --dims 864 240 640 \\
        --dtype float32 \\
        --n-channels 3 \\
        --output velocity_planar.raw

    # Dims can also be encoded in the filename (e.g. 864x240x640):
    python interleaved_to_planar.py \\
        --filename 1atm_velo_3.3750E-04__864x240x640_float32.raw \\
        --dims 864 240 640 --dtype float32

The output file defaults to <stem>_planar.<ext> if --output is not given.
"""

import argparse
import os
import sys

import numpy as np


_DTYPE_MAP = {
    "float32": np.float32,
    "float64": np.float64,
    "float16": np.float16,
    "uint8":   np.uint8,
    "uint16":  np.uint16,
    "uint32":  np.uint32,
    "int8":    np.int8,
    "int16":   np.int16,
    "int32":   np.int32,
}


def interleaved_to_planar(filename, dims, dtype, n_channels, output):
    n_voxels = dims[0] * dims[1] * dims[2]
    expected_elements = n_voxels * n_channels
    expected_bytes = expected_elements * np.dtype(dtype).itemsize

    actual_bytes = os.path.getsize(filename)
    if actual_bytes != expected_bytes:
        raise ValueError(
            f"File size mismatch.\n"
            f"  Expected: {expected_bytes:,} bytes  "
            f"({dims[0]}×{dims[1]}×{dims[2]} × {n_channels} channels × {np.dtype(dtype).itemsize}B)\n"
            f"  Got:      {actual_bytes:,} bytes\n"
            "Check --dims, --dtype, and --n-channels."
        )

    print(f"Reading  {filename}  ({actual_bytes / 1e9:.2f} GB) …")
    data = np.fromfile(filename, dtype=dtype)

    # Interleaved: (N_voxels, n_channels) → planar: (n_channels, N_voxels) → flatten
    print(f"Converting  interleaved ({n_voxels:,} × {n_channels})  →  planar ({n_channels} × {n_voxels:,}) …")
    planar = data.reshape(n_voxels, n_channels).T.reshape(-1)

    print(f"Writing  {output}  …")
    planar.tofile(output)

    written_bytes = os.path.getsize(output)
    assert written_bytes == expected_bytes, f"Write size mismatch: {written_bytes} != {expected_bytes}"
    print(f"Done.  {written_bytes / 1e9:.2f} GB written to {output}")

    # Quick sanity: first voxel's channels should now appear at indices 0, N, 2N
    first_voxel_interleaved = data[:n_channels]
    first_voxel_planar = np.array([planar[c * n_voxels] for c in range(n_channels)])
    assert np.array_equal(first_voxel_interleaved, first_voxel_planar), \
        "Sanity check failed: first-voxel values do not match after conversion."
    print(f"Sanity check passed:  first voxel = {first_voxel_planar.tolist()}")


def main():
    parser = argparse.ArgumentParser(
        description="Convert an interleaved multi-channel .raw file to planar layout",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--filename", required=True,
                        help="path to the interleaved input .raw file")
    parser.add_argument("--dims", nargs=3, type=int, required=True,
                        metavar=("X", "Y", "Z"),
                        help="volume dimensions in x y z order")
    parser.add_argument("--dtype", default="float32",
                        choices=list(_DTYPE_MAP.keys()),
                        help="element data type")
    parser.add_argument("--n-channels", type=int, default=3,
                        help="number of interleaved channels (e.g. 3 for u/v/w)")
    parser.add_argument("--output", default=None,
                        help="output path (default: <stem>_planar.<ext>)")

    args = parser.parse_args()

    if args.n_channels < 2:
        parser.error("--n-channels must be >= 2 (single-channel files have no interleaving to convert)")

    dtype = _DTYPE_MAP[args.dtype]

    if args.output is None:
        stem, ext = os.path.splitext(args.filename)
        args.output = stem + "_planar" + ext

    if os.path.abspath(args.output) == os.path.abspath(args.filename):
        parser.error("--output must differ from --filename (in-place conversion is not supported)")

    interleaved_to_planar(args.filename, args.dims, dtype, args.n_channels, args.output)


if __name__ == "__main__":
    main()

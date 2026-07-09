"""Extract metadata from a planar multi-channel .raw volume and write to JSON.

Reads the raw binary file, computes per-channel and global statistics, and
writes a companion .json containing everything needed to train, convert, and
denormalize: dims, dtype, layout, value ranges, per-channel stats, and the
ready-to-paste --range-* flags for bson_to_vtk.py.

Usage
-----
  python raw_metadata.py \\
      --filename data/planar_raw/velocity_planar.raw \\
      --dims 864 240 640 \\
      --dtype float32 \\
      --n-channels 3

  # Custom output path:
  python raw_metadata.py \\
      --filename velocity_planar.raw --dims 864 240 640 \\
      --dtype float32 --n-channels 3 \\
      --output velocity_planar.json

  # With physical spacing and origin (stored in JSON, used by bson_to_vtk):
  python raw_metadata.py \\
      --filename velocity_planar.raw --dims 864 240 640 \\
      --dtype float32 --n-channels 3 \\
      --spacing 0.01 0.01 0.01 --origin 0.0 0.0 0.0

Flags
-----
  --filename    PATH         Input planar .raw file (required).
  --dims        X Y Z        Volume dimensions in x y z order (required).
  --dtype       STR          Element dtype: float32, float64, uint8, uint16, etc. (default: float32).
  --n-channels  N            Number of channels stored in planar order (default: 3).
  --channel-names NAME ...   Names for each channel (default: u v w for 3-ch, c0 c1 ... otherwise).
  --spacing     DX DY DZ     Physical voxel spacing (optional, stored in JSON).
  --origin      OX OY OZ     World-space origin (optional, stored in JSON).
  --output      PATH         Output JSON path (default: <filename_stem>.json).

Output JSON keys
----------------
  filename, dims, dtype, n_channels, layout, n_voxels, file_size_bytes,
  spacing, origin,
  channels[]: name, min, max, mean, std, percentile_1, percentile_99,
  speed: min, max, mean, std          (L2 magnitude; only for n_channels == 3)
  bson_to_vtk_flags                   (ready-to-paste --range-* string)
"""

import argparse
import json
import os

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


def compute_metadata(filename, dims, dtype, n_channels, channel_names, spacing, origin):
    n_voxels = dims[0] * dims[1] * dims[2]
    expected_bytes = n_voxels * n_channels * np.dtype(dtype).itemsize
    actual_bytes = os.path.getsize(filename)

    if actual_bytes != expected_bytes:
        raise ValueError(
            f"File size mismatch.\n"
            f"  Expected: {expected_bytes:,} bytes  "
            f"({dims[0]}×{dims[1]}×{dims[2]} × {n_channels} ch × {np.dtype(dtype).itemsize}B)\n"
            f"  Got:      {actual_bytes:,} bytes\n"
            "Check --dims, --dtype, and --n-channels."
        )

    print(f"Reading {filename}  ({actual_bytes / 1e9:.3f} GB) …")
    data = np.fromfile(filename, dtype=dtype).reshape(n_channels, n_voxels)

    channels = []
    for c in range(n_channels):
        ch = data[c].astype(np.float64)
        p1, p99 = np.percentile(ch, [1, 99])
        channels.append({
            "name":          channel_names[c],
            "min":           float(ch.min()),
            "max":           float(ch.max()),
            "mean":          float(ch.mean()),
            "std":           float(ch.std()),
            "percentile_1":  float(p1),
            "percentile_99": float(p99),
        })
        print(f"  {channel_names[c]:4s}  min={ch.min():.6f}  max={ch.max():.6f}  "
              f"mean={ch.mean():.6f}  std={ch.std():.6f}")

    meta = {
        "filename":       os.path.abspath(filename),
        "dims":           {"x": dims[0], "y": dims[1], "z": dims[2]},
        "dtype":          np.dtype(dtype).name,
        "n_channels":     n_channels,
        "layout":         "planar",
        "n_voxels":       n_voxels,
        "file_size_bytes": actual_bytes,
        "channels":       channels,
    }

    if spacing is not None:
        meta["spacing"] = {"x": spacing[0], "y": spacing[1], "z": spacing[2]}
    if origin is not None:
        meta["origin"] = {"x": origin[0], "y": origin[1], "z": origin[2]}

    # Speed (L2 magnitude) stats — only meaningful for 3-channel velocity fields.
    if n_channels == 3:
        speed = np.sqrt(data[0].astype(np.float64)**2 +
                        data[1].astype(np.float64)**2 +
                        data[2].astype(np.float64)**2)
        meta["speed"] = {
            "min":  float(speed.min()),
            "max":  float(speed.max()),
            "mean": float(speed.mean()),
            "std":  float(speed.std()),
        }
        print(f"  speed min={speed.min():.6f}  max={speed.max():.6f}  "
              f"mean={speed.mean():.6f}  std={speed.std():.6f}")

    # Ready-to-paste flags for bson_to_vtk.py.
    range_flags = " ".join(
        f"--range-{ch['name']} {ch['min']:.6f} {ch['max']:.6f}"
        for ch in channels
    )
    meta["bson_to_vtk_flags"] = range_flags

    return meta


def main():
    parser = argparse.ArgumentParser(
        description="Extract metadata and per-channel stats from a planar .raw volume.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--filename", required=True,
                        help="path to the planar .raw file")
    parser.add_argument("--dims", nargs=3, type=int, required=True,
                        metavar=("X", "Y", "Z"),
                        help="volume dimensions in x y z order")
    parser.add_argument("--dtype", default="float32",
                        choices=list(_DTYPE_MAP.keys()),
                        help="element data type")
    parser.add_argument("--n-channels", type=int, default=3,
                        help="number of channels stored in planar order")
    parser.add_argument("--channel-names", nargs="+", default=None,
                        metavar="NAME",
                        help="names for each channel (default: u v w for 3-ch, c0 c1 ... otherwise)")
    parser.add_argument("--spacing", nargs=3, type=float, default=None,
                        metavar=("DX", "DY", "DZ"),
                        help="physical voxel spacing (stored in JSON if provided)")
    parser.add_argument("--origin", nargs=3, type=float, default=None,
                        metavar=("OX", "OY", "OZ"),
                        help="world-space origin (stored in JSON if provided)")
    parser.add_argument("--output", default=None,
                        help="output JSON path (default: <filename_stem>.json)")

    args = parser.parse_args()

    # Resolve channel names.
    if args.channel_names is not None:
        if len(args.channel_names) != args.n_channels:
            parser.error(
                f"--channel-names requires exactly {args.n_channels} names, "
                f"got {len(args.channel_names)}"
            )
        channel_names = args.channel_names
    elif args.n_channels == 3:
        channel_names = ["u", "v", "w"]
    else:
        channel_names = [f"c{i}" for i in range(args.n_channels)]

    output = args.output or os.path.splitext(args.filename)[0] + ".json"

    dtype = _DTYPE_MAP[args.dtype]
    meta = compute_metadata(
        args.filename, args.dims, dtype, args.n_channels,
        channel_names, args.spacing, args.origin,
    )

    with open(output, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\nMetadata written to {output}")
    print(f"\nbson_to_vtk flags:\n  {meta['bson_to_vtk_flags']}")


if __name__ == "__main__":
    main()

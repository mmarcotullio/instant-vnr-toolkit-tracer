"""Reconstruct a 3-D velocity vector field from trained INR .bson files and export as VTK.

Runs batched inference over a dense coordinate grid, assembles the (u, v, w) components
into a single point-data array, and writes a VTK ImageData file readable by ParaView.

Two model sources are supported (mutually exclusive):

  --model-vector   A single .bson trained with --n-channels 3 (one network, 3 outputs).
  --model-u/v/w    Three separate scalar .bson files, one per velocity component.

Usage
-----
# Single 3-channel vector model:
  python bson_to_vtk.py \\
      --model-vector outputs/velocity.bson \\
      --dims 864 240 640 \\
      --range-u -0.65 0.76 --range-v -0.66 0.60 --range-w -0.60 0.62 \\
      --output velocity.vtk

# Three separate scalar models:
  python bson_to_vtk.py \\
      --model-u outputs/output_u.bson \\
      --model-v outputs/output_v.bson \\
      --model-w outputs/output_w.bson \\
      --dims 864 240 640 \\
      --output velocity.vtk

# Set physical spacing and origin:
  python bson_to_vtk.py --model-vector vel.bson --dims 864 240 640 \\
      --spacing 0.01 0.01 0.01 --origin 0.0 0.0 0.0 --output vel.vtk

# Set bounding box instead of spacing/origin (overrides both):
  python bson_to_vtk.py --model-vector vel.bson --dims 864 240 640 \\
      --bounds 0 8.64 0 2.40 0 6.40 --output vel.vtk

# Limit GPU memory use by reducing inference chunk size:
  python bson_to_vtk.py --model-vector vel.bson --dims 864 240 640 \\
      --chunk 262144 --output vel.vtk

Flags
-----
  --model-vector PATH             Single 3-channel .bson (trained with --n-channels 3).
  --model-u/v/w  PATH             Three scalar .bson files for u, v, w components.
  --dims         X Y Z            Grid dimensions in X Y Z order (required).
  --output       PATH             Output .vtk path (default: reconstructed.vtk).
  --spacing      DX DY DZ         Physical voxel spacing in X Y Z (default: 1 1 1).
  --origin       OX OY OZ         World-space origin in X Y Z (default: 0 0 0).
  --bounds  XMIN XMAX YMIN YMAX ZMIN ZMAX
                                  Bounding box in X Y Z order; overrides --spacing and
                                  --origin.  Matches vtk_to_raw JSON "bounds" field.
  --chunk        N                Inference batch size in points (default: 2^20 ≈ 1M).
  --range-u      MIN MAX          Denormalization range for u (output in [0,1] if omitted).
  --range-v      MIN MAX          Denormalization range for v.
  --range-w      MIN MAX          Denormalization range for w.

Notes
-----
  - --dims and --bounds are both in X Y Z order.  ParaView's Information panel shows
    extents in reversed Z Y X order — so if ParaView shows [a, b, c], X=c, Y=b, Z=a.
  - The CUDA sampler normalizes each channel to [0, 1] during training using its
    per-channel min/max.  Pass --range-u/v/w to restore physical velocity units.
    Without these flags the output VTK will contain normalized [0, 1] values.
  - For the three-scalar path, ranges may be read from the .bson if stored there;
    --range-u/v/w overrides any stored value.
"""

import argparse

import bson
import numpy as np
import pyvista as pv
import torch
import tinycudann as tcnn


def load_model(bson_path: str) -> tuple[tcnn.NetworkWithInputEncoding, tuple[float, float] | None]:
    """Reconstruct a TCNN model from a .bson scene file and load its weights.

    Returns (model, value_range) where value_range is (vmin, vmax) if stored in
    the scene, or None if not present (caller must supply the range manually).
    n_output_dims is read from the scene file (defaults to 1 for legacy files).
    """
    with open(bson_path, "rb") as f:
        scene = bson.loads(f.read())

    n_output_dims = scene.get("model", {}).get("n_output_dims", 1)

    model = tcnn.NetworkWithInputEncoding(
        n_input_dims=3,
        n_output_dims=n_output_dims,
        encoding_config=scene["model"]["encoding"],
        network_config=scene["model"]["network"],
    )

    # Weights were saved as fp16 bytes; TCNN Python API stores params as fp32.
    raw = scene["parameters"]["params_binary"]
    params = torch.from_numpy(np.frombuffer(raw, dtype=np.float16).copy()).float().cuda()
    model.load_state_dict({"params": params})

    vrange = scene.get("volume", {}).get("value_range")
    if vrange is not None:
        vrange = (float(vrange["min"]), float(vrange["max"]))
    return model, vrange


def make_coords(dims: list[int]) -> torch.Tensor:
    """Build a dense (N, 3) coordinate tensor in VTK order (x varies fastest).

    Coordinates are normalized to [0, (dim-1)/dim] per axis, matching the
    convention used during training (vscale = 1/dim applied to integer indices).
    """
    W, H, D = dims
    vscale = torch.tensor([1.0 / W, 1.0 / H, 1.0 / D], dtype=torch.float32)

    # meshgrid with indexing="ij" over (z, y, x) → reshape gives x-fastest order
    zz, yy, xx = torch.meshgrid(
        torch.arange(D, dtype=torch.float32),
        torch.arange(H, dtype=torch.float32),
        torch.arange(W, dtype=torch.float32),
        indexing="ij",
    )
    coords = torch.stack([xx, yy, zz], dim=-1).reshape(-1, 3) * vscale
    return coords


def query_model(
    model: tcnn.NetworkWithInputEncoding,
    coords: torch.Tensor,
    value_range: tuple[float, float] | None = None,
    chunk: int = 1 << 20,
) -> np.ndarray:
    """Run batched inference and return a float32 numpy array of shape (N, 1).

    The CUDA structured sampler normalizes float32 values to [0, 1] during
    training using (v - vmin) / (vmax - vmin). If value_range=(vmin, vmax) is
    given, the output is mapped back to original units.
    """
    parts = []
    with torch.no_grad():
        for i in range(0, coords.shape[0], chunk):
            batch = coords[i : i + chunk].cuda()
            parts.append(model(batch).float().cpu())
    out = torch.cat(parts, dim=0).numpy()
    if value_range is not None:
        vmin, vmax = value_range
        out = out * (vmax - vmin) + vmin
    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Reconstruct a velocity vector field from three INR .bson files and save as VTK."
    )
    parser.add_argument("--model-u", default=None, help="Path to u.bson (x-velocity, scalar model)")
    parser.add_argument("--model-v", default=None, help="Path to v.bson (y-velocity, scalar model)")
    parser.add_argument("--model-w", default=None, help="Path to w.bson (z-velocity, scalar model)")
    parser.add_argument("--model-vector", default=None,
                        help="Path to a single 3-channel vector .bson (mutually exclusive with --model-u/v/w)")
    parser.add_argument(
        "--dims", nargs=3, type=int, required=True, metavar=("X", "Y", "Z"),
        help="Grid dimensions in X Y Z order. Note: ParaView Information panel shows extents in reversed Z Y X order.",
    )
    parser.add_argument("--output", default="reconstructed.vtk", help="Output .vtk path")
    parser.add_argument(
        "--spacing", nargs=3, type=float, default=None, metavar=("DX", "DY", "DZ"),
        help="Physical voxel spacing in x y z (default: 1.0 1.0 1.0)",
    )
    parser.add_argument(
        "--origin", nargs=3, type=float, default=None, metavar=("OX", "OY", "OZ"),
        help="World-space origin (default: 0.0 0.0 0.0)",
    )
    parser.add_argument(
        "--bounds", nargs=6, type=float, default=None,
        metavar=("XMIN", "XMAX", "YMIN", "YMAX", "ZMIN", "ZMAX"),
        help="World-space bounding box in X Y Z order; overrides --spacing and --origin. "
             "Matches the 'bounds' field in vtk_to_raw JSON output (xmin xmax ymin ymax zmin zmax).",
    )
    parser.add_argument(
        "--chunk", type=int, default=1 << 20,
        help="Inference batch size in points (default: 2^20 ≈ 1M)",
    )
    parser.add_argument(
        "--range-u", nargs=2, type=float, default=None, metavar=("UMIN", "UMAX"),
        help="Value range [vmin vmax] for u component (required if not stored in .bson)",
    )
    parser.add_argument(
        "--range-v", nargs=2, type=float, default=None, metavar=("VMIN", "VMAX"),
        help="Value range [vmin vmax] for v component",
    )
    parser.add_argument(
        "--range-w", nargs=2, type=float, default=None, metavar=("WMIN", "WMAX"),
        help="Value range [vmin vmax] for w component",
    )
    args = parser.parse_args()

    # Validate model source: exactly one of (--model-vector) or (--model-u/v/w) must be given.
    using_vector = args.model_vector is not None
    using_scalar = any(x is not None for x in (args.model_u, args.model_v, args.model_w))
    if using_vector and using_scalar:
        parser.error("--model-vector and --model-u/v/w are mutually exclusive.")
    if not using_vector and not using_scalar:
        parser.error("Provide either --model-vector or all of --model-u, --model-v, --model-w.")
    if using_scalar and not all(x is not None for x in (args.model_u, args.model_v, args.model_w)):
        parser.error("--model-u, --model-v, and --model-w must all be provided together.")

    dims = args.dims
    Nx, Ny, Nz = dims
    N = Nx * Ny * Nz
    print(f"Grid: X={Nx} × Y={Ny} × Z={Nz} = {N:,} points")

    def resolve_range(cli_range, bson_range, name):
        if cli_range is not None:
            return tuple(cli_range)
        if bson_range is not None:
            return bson_range
        print(f"[warn] no value range for {name}; output will be in [0,1] (pass --range-{name} vmin vmax)")
        return None

    print("Generating coordinate grid …")
    coords = make_coords(dims)

    if using_vector:
        # Single 3-channel model trained with --n-channels 3.
        print("Loading vector model …")
        model_vec, _ = load_model(args.model_vector)
        if model_vec.n_output_dims != 3:
            parser.error(
                f"--model-vector expects a model with n_output_dims=3, "
                f"got {model_vec.n_output_dims}. Use --model-u/v/w for scalar models."
            )
        print("Querying vector model …")
        raw = query_model(model_vec, coords, value_range=None, chunk=args.chunk)  # (N, 3) in [0,1]
        u = raw[:, 0:1]
        v = raw[:, 1:2]
        w = raw[:, 2:3]
        # Denormalize each component independently if ranges are provided.
        for arr, cli_range, name in ((u, args.range_u, "u"), (v, args.range_v, "v"), (w, args.range_w, "w")):
            r = resolve_range(cli_range, None, name)
            if r is not None:
                arr[:] = arr * (r[1] - r[0]) + r[0]
    else:
        # Three separate scalar models (one per component).
        print("Loading models …")
        model_u, bson_range_u = load_model(args.model_u)
        model_v, bson_range_v = load_model(args.model_v)
        model_w, bson_range_w = load_model(args.model_w)

        range_u = resolve_range(args.range_u, bson_range_u, "u")
        range_v = resolve_range(args.range_v, bson_range_v, "v")
        range_w = resolve_range(args.range_w, bson_range_w, "w")

        print("Querying u …")
        u = query_model(model_u, coords, range_u, args.chunk)
        print("Querying v …")
        v = query_model(model_v, coords, range_v, args.chunk)
        print("Querying w …")
        w = query_model(model_w, coords, range_w, args.chunk)

    velocity = np.concatenate([u, v, w], axis=1).astype(np.float32)  # (N, 3)

    print(f"Exporting to {args.output} …")
    mesh = pv.ImageData()
    mesh.dimensions = dims          # (Nx, Ny, Nz) in X, Y, Z order
    if args.bounds is not None:
        xmin, xmax, ymin, ymax, zmin, zmax = args.bounds
        mesh.origin = (xmin, ymin, zmin)
        mesh.spacing = (
            (xmax - xmin) / (Nx - 1),
            (ymax - ymin) / (Ny - 1),
            (zmax - zmin) / (Nz - 1),
        )
        print(f"  Bounds  X=[{xmin}, {xmax}]  Y=[{ymin}, {ymax}]  Z=[{zmin}, {zmax}]")
        print(f"  Spacing dx={mesh.spacing[0]:.6g}  dy={mesh.spacing[1]:.6g}  dz={mesh.spacing[2]:.6g}")
    else:
        if args.spacing is not None:
            mesh.spacing = args.spacing
        if args.origin is not None:
            mesh.origin = args.origin
        if args.spacing is None and args.origin is None:
            print("  [warn] no --bounds/--spacing/--origin specified; VTK will use unit spacing at origin")
    mesh.point_data["velocity"] = velocity
    mesh.save(args.output)

    print("Done.")


if __name__ == "__main__":
    main()

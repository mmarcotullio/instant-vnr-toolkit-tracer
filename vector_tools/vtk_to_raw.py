"""
vtk_to_raw.py — Convert a structured VTK/VTI vector field to binary .raw files
               for fast loading during Implicit Neural Representation training.

Usage
-----
# Save as a single interleaved (channels-last) file:
  python vtk_to_raw.py --input data.vtk --output ./processed/velocity.raw \
                       --array-name velocity --mode combined

# Save as a single planar (channels-first) file:
  python vtk_to_raw.py --input data.vtk --output ./processed/velocity.raw \
                       --array-name velocity --mode planar

# Save as three separate per-component files (u/v/w):
  python vtk_to_raw.py --input data.vtk --output ./processed/velocity.raw \
                       --array-name velocity --mode split

Output (combined mode)
    processed/velocity.raw   — float32, shape (Nz, Ny, Nx, 3) interleaved binary dump
    processed/velocity.json  — companion metadata  [layout: "interleaved"]

Output (planar mode)
    processed/velocity.raw   — float32, shape (3, Nz, Ny, Nx) planar binary dump
    processed/velocity.json  — companion metadata  [layout: "planar"]

Output (split mode)  — stem derived from --output path
    processed/velocity_u.raw / velocity_v.raw / velocity_w.raw
    processed/velocity_u.json / velocity_v.json / velocity_w.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pyvista as pv


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _load_mesh(input_path: Path) -> pv.DataSet:
    """Read a VTK/VTI file and unwrap MultiBlock to the first live block."""
    mesh = pv.read(str(input_path))
    if isinstance(mesh, pv.MultiBlock):
        for i in range(mesh.n_blocks):
            block = mesh[i]
            if block is not None:
                return block
        raise ValueError(
            f"'{input_path.name}' is a MultiBlock dataset with no non-empty blocks."
        )
    return mesh


def _extract_vector_array(mesh: pv.DataSet, array_name: str, input_name: str) -> np.ndarray:
    """
    Return the (N, 3) float32 point-data array for *array_name*.
    Prints available array names and exits on any mismatch.
    """
    if array_name not in mesh.point_data:
        in_cell = array_name in mesh.cell_data
        print(
            f"Error: array '{array_name}' not found in point_data of '{input_name}'.",
            file=sys.stderr,
        )
        if in_cell:
            print(
                f"  Found in cell_data instead. Convert with:\n"
                f"    mesh = mesh.cell_data_to_point_data()\n"
                f"    mesh.save('fixed.vti')",
                file=sys.stderr,
            )
        print(f"\nAvailable arrays in this file:", file=sys.stderr)
        print(f"  point_data : {mesh.array_names}", file=sys.stderr)
        sys.exit(1)

    raw = mesh.point_data[array_name]
    if raw.ndim != 2 or raw.shape[1] != 3:
        print(
            f"Error: '{array_name}' has shape {raw.shape}; expected (N, 3) for a 3-D vector field.",
            file=sys.stderr,
        )
        sys.exit(1)

    return raw.astype(np.float32)


def _grid_info(mesh: pv.DataSet) -> dict:
    """Extract grid dimensions, spacing, and origin from an ImageData or RectilinearGrid."""
    Nx, Ny, Nz = mesh.dimensions

    if isinstance(mesh, pv.ImageData):
        dx, dy, dz = mesh.spacing
        ox, oy, oz = mesh.origin
    elif isinstance(mesh, pv.RectilinearGrid):
        dx = float(np.mean(np.diff(mesh.x))) if len(mesh.x) > 1 else 1.0
        dy = float(np.mean(np.diff(mesh.y))) if len(mesh.y) > 1 else 1.0
        dz = float(np.mean(np.diff(mesh.z))) if len(mesh.z) > 1 else 1.0
        ox, oy, oz = float(mesh.x[0]), float(mesh.y[0]), float(mesh.z[0])
    else:
        bounds = mesh.bounds
        ox, oy, oz = float(bounds[0]), float(bounds[2]), float(bounds[4])
        dx = (float(bounds[1]) - ox) / max(Nx - 1, 1)
        dy = (float(bounds[3]) - oy) / max(Ny - 1, 1)
        dz = (float(bounds[5]) - oz) / max(Nz - 1, 1)

    return {
        "Nx": Nx, "Ny": Ny, "Nz": Nz,
        "dx": float(dx), "dy": float(dy), "dz": float(dz),
        "ox": float(ox), "oy": float(oy), "oz": float(oz),
    }


def _write_raw_and_json(
    array: np.ndarray,
    out_path: Path,
    grid: dict,
    num_channels: int,
    arr_zyxc: np.ndarray,
    layout: str = "interleaved",
) -> None:
    """Dump *array* to a .raw file and write the companion .json metadata."""
    array.tofile(str(out_path))

    meta = {
        "grid_dimensions": {"Nx": grid["Nx"], "Ny": grid["Ny"], "Nz": grid["Nz"]},
        "spacing":         {"dx": grid["dx"], "dy": grid["dy"], "dz": grid["dz"]},
        "origin":          {"x":  grid["ox"], "y":  grid["oy"], "z":  grid["oz"]},
        "bounds": {
            "xmin": grid["ox"],
            "xmax": grid["ox"] + (grid["Nx"] - 1) * grid["dx"],
            "ymin": grid["oy"],
            "ymax": grid["oy"] + (grid["Ny"] - 1) * grid["dy"],
            "zmin": grid["oz"],
            "zmax": grid["oz"] + (grid["Nz"] - 1) * grid["dz"],
        },
        "value_ranges": {
            "u": {"min": float(arr_zyxc[..., 0].min()), "max": float(arr_zyxc[..., 0].max())},
            "v": {"min": float(arr_zyxc[..., 1].min()), "max": float(arr_zyxc[..., 1].max())},
            "w": {"min": float(arr_zyxc[..., 2].min()), "max": float(arr_zyxc[..., 2].max())},
        },
        "num_channels":    num_channels,
        "layout":          layout,
        "dtype":           "float32",
    }
    json_path = out_path.with_suffix(".json")
    json_path.write_text(json.dumps(meta, indent=2))
    print(f"  Wrote {out_path}  ({array.nbytes / 1024**2:.2f} MB)")
    print(f"  Wrote {json_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Mode implementations
# ─────────────────────────────────────────────────────────────────────────────

def save_combined(
    arr_zyxc: np.ndarray,
    grid: dict,
    out_path: Path,
) -> None:
    """Write a single interleaved (Nz, Ny, Nx, 3) .raw file."""
    _write_raw_and_json(arr_zyxc, out_path, grid, num_channels=3, arr_zyxc=arr_zyxc, layout="interleaved")


def save_planar(
    arr_zyxc: np.ndarray,
    grid: dict,
    out_path: Path,
) -> None:
    """Write a single planar (3, Nz, Ny, Nx) .raw file — all U, then all V, then all W."""
    arr_czyx = np.ascontiguousarray(arr_zyxc.transpose(3, 0, 1, 2))
    _write_raw_and_json(arr_czyx, out_path, grid, num_channels=3, arr_zyxc=arr_zyxc, layout="planar")


def save_split(
    arr_zyxc: np.ndarray,
    grid: dict,
    out_path: Path,
) -> None:
    """Write three separate (Nz, Ny, Nx) .raw files for U, V, and W."""
    stem = out_path.stem
    for idx, component in enumerate(("u", "v", "w")):
        channel = np.ascontiguousarray(arr_zyxc[..., idx])  # (Nz, Ny, Nx)
        comp_path = out_path.with_name(f"{stem}_{component}{out_path.suffix}")
        _write_raw_and_json(channel, comp_path, grid, num_channels=1, arr_zyxc=arr_zyxc)


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Convert a structured VTK/VTI vector field to binary .raw files.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  %(prog)s --input data.vtk --output ./out/velocity.raw --array-name velocity --mode combined\n"
            "  %(prog)s --input tornado3d.vtk --output ./out/tornado.raw --array-name Velocity --mode split"
        ),
    )
    p.add_argument("--input",      required=True,  metavar="FILE",  help="Path to input .vtk/.vti file")
    p.add_argument("--output",     required=True,  metavar="PATH",
                   help="Output file path (e.g. out/velocity.raw). For split mode, _u/_v/_w are appended to the stem.")
    p.add_argument("--array-name", required=True,  metavar="NAME",  help="Point-data array name to extract (e.g. 'velocity')")
    p.add_argument(
        "--mode",
        choices=("combined", "planar", "split"),
        default="combined",
        help=(
            "'combined': one interleaved (Nz,Ny,Nx,3) file  |  "
            "'planar': one planar (3,Nz,Ny,Nx) file  |  "
            "'split': three (Nz,Ny,Nx) files  [default: combined]"
        ),
    )
    return p


def main() -> None:
    args = build_parser().parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: input file not found: {input_path}", file=sys.stderr)
        sys.exit(1)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # ── Load ──────────────────────────────────────────────────────────────────
    print(f"Reading {input_path} …")
    mesh = _load_mesh(input_path)
    print(f"  Dataset type : {type(mesh).__name__}")
    print(f"  Dimensions   : {mesh.dimensions}")
    print(f"  Arrays       : {mesh.array_names}")

    # ── Extract and cast ──────────────────────────────────────────────────────
    flat_f32 = _extract_vector_array(mesh, args.array_name, input_path.name)  # (N, 3) float32

    grid = _grid_info(mesh)
    Nx, Ny, Nz = grid["Nx"], grid["Ny"], grid["Nz"]

    # Reshape to (Nz, Ny, Nx, 3) — VTK X-fastest C-order matches this directly
    arr_zyxc = flat_f32.reshape(Nz, Ny, Nx, 3)

    print(f"\nArray shape (Nz, Ny, Nx, 3) : {arr_zyxc.shape}")
    print(f"Dimensions (X, Y, Z)  : {Nx} × {Ny} × {Nz}")
    print(f"  (ParaView Information panel shows these in reversed Z Y X order)")
    print(f"Spacing   : dx={grid['dx']:.6g}, dy={grid['dy']:.6g}, dz={grid['dz']:.6g}")
    print(f"Origin    : ({grid['ox']:.6g}, {grid['oy']:.6g}, {grid['oz']:.6g})")
    xmax = grid["ox"] + (Nx - 1) * grid["dx"]
    ymax = grid["oy"] + (Ny - 1) * grid["dy"]
    zmax = grid["oz"] + (Nz - 1) * grid["dz"]
    print(f"Bounds    : X=[{grid['ox']:.6g}, {xmax:.6g}]  Y=[{grid['oy']:.6g}, {ymax:.6g}]  Z=[{grid['oz']:.6g}, {zmax:.6g}]")
    print(f"  bson_to_vtk --bounds {grid['ox']:.6g} {xmax:.6g} {grid['oy']:.6g} {ymax:.6g} {grid['oz']:.6g} {zmax:.6g}")
    print(f"\nSaving in '{args.mode}' mode to {out_path}")

    # ── Write ─────────────────────────────────────────────────────────────────
    if args.mode == "combined":
        save_combined(arr_zyxc, grid, out_path)
    elif args.mode == "planar":
        save_planar(arr_zyxc, grid, out_path)
    else:
        save_split(arr_zyxc, grid, out_path)

    print("\nDone.")


if __name__ == "__main__":
    main()

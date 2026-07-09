"""Render a trained INR scene file using the instantvnr 3D volume renderer.

Two modes:

  batch       -- headless render to a PNG (no display required, good for scripts)
  interactive -- real-time OpenGL viewer for interactive exploration (requires a display)

Usage:
  python render.py --scene my_vol.bson --mode batch
  python render.py --scene my_vol.bson --mode batch --output renders/out.png --spp 64
  python render.py --scene my_vol.bson --mode interactive
  python render.py --scene my_vol.bson --mode interactive --rendering-mode 1
Rendering modes (--rendering-mode):
  0  ray marching, no shading         (default)
  1  ray marching, gradient shading
  2  ray marching, single-shade heuristic (deferred shadow)
  3  ray marching, GT shadow (simple volume only, expensive)
  4  path tracing

Shadow mapping via neural network is auto-inferred: if --shadowfield is provided,
the renderer automatically uses it for illumination regardless of mode.

The .bson scene file is self-contained: model architecture, fp16 weights, and the
macrocell acceleration structure are all embedded — no original volume required.
"""

import argparse
import os
import sys


def _validate_paths(args):
    if not os.path.isfile(args.scene):
        print(f"[error] scene file not found: {args.scene}", file=sys.stderr)
        sys.exit(1)
    if args.tfn and not os.path.isfile(args.tfn):
        print(f"[error] transfer function file not found: {args.tfn}", file=sys.stderr)
        sys.exit(1)


def _run_batch(args):
    from instantvnr.apps import run_batch

    ret = run_batch(
        neural_volume=args.scene,
        output=args.output,
        width=args.width,
        height=args.height,
        spp=args.spp,
        warmup=args.warmup,
        rendering_mode=args.rendering_mode,
        sampling_rate=args.sampling_rate,
        density_scale=args.density_scale,
        camera_from=args.camera_from,
        camera_at=args.camera_at,
        camera_up=args.camera_up,
        tfn=args.tfn,
    )
    if ret == 0:
        print(f"[info] saved: {args.output}")
    return ret


def _run_interactive(args):
    from instantvnr.apps import run_int_single

    return run_int_single(
        neural_volume=args.scene,
        rendering_mode=args.rendering_mode,
        sampling_rate=args.sampling_rate,
        density_scale=args.density_scale,
        camera_from=args.camera_from,
        camera_at=args.camera_at,
        camera_up=args.camera_up,
        tfn=args.tfn,
    )


def main():
    parser = argparse.ArgumentParser(
        description="Render a trained INR scene (.bson) using the instantvnr 3D renderer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument("--scene", required=True,
                        help="path to .bson scene file produced by train.py")
    parser.add_argument("--mode", choices=["batch", "interactive"], default="batch",
                        help="batch: headless PNG output; interactive: OpenGL viewer")

    # batch-only options
    batch = parser.add_argument_group("batch mode")
    batch.add_argument("--output", default="output.png",
                       help="output PNG path")
    batch.add_argument("--width", type=int, default=1024,
                       help="image width in pixels")
    batch.add_argument("--height", type=int, default=1024,
                       help="image height in pixels")
    batch.add_argument("--spp", type=int, default=1,
                       help="samples per pixel (higher = less noise, slower)")
    batch.add_argument("--warmup", type=int, default=5,
                       help="warm-up frames before capture (lets the renderer stabilise)")

    # shared options
    shared = parser.add_argument_group("shared")
    shared.add_argument("--rendering-mode", type=int, default=0,
                        help="rendering method: 0=ray march, 1=gradient shading, "
                             "2=single-shade heuristic, 3=GT shadow, 4=path tracing")
    shared.add_argument("--sampling-rate", type=float, default=1.0,
                        help="ray marching step size multiplier (higher = denser, slower)")
    shared.add_argument("--density-scale", type=float, default=1.0,
                        help="density scale multiplier applied to transfer function")
    shared.add_argument("--camera-from", nargs=3, type=float, metavar=("X", "Y", "Z"),
                        default=None, help="camera position in world space")
    shared.add_argument("--camera-at", nargs=3, type=float, metavar=("X", "Y", "Z"),
                        default=None, help="camera look-at point in world space")
    shared.add_argument("--camera-up", nargs=3, type=float, metavar=("X", "Y", "Z"),
                        default=None, help="camera up vector")
    shared.add_argument("--tfn", default=None,
                        help="path to a transfer function file")

    args = parser.parse_args()
    _validate_paths(args)

    try:
        if args.mode == "batch":
            sys.exit(_run_batch(args))
        else:
            sys.exit(_run_interactive(args))
    except ImportError as e:
        print(f"[error] instantvnr-py not installed: {e}", file=sys.stderr)
        print("[hint]  Build it: cd packages/instantvnr && SM=<your_sm> ./setup_venv.sh",
              file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()

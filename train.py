"""Train a scalar or vector-field INR on a volume file (raw binary or OpenVDB).

The workflow:
  1. Create a volume sampler (pysampler) to stream random (coord, value) batches
  2. Build a tiny-cuda-nn hash-grid encoding + fully-fused MLP via INR_TCNN
  3. Train with Adam + fp16 AMP and a two-phase StepLR schedule
  4. Export model weights and macrocell acceleration to a self-contained .bson

Usage:
  # Structured raw volume (scalar)
  python train.py --filename /path/to/volume.raw --dims 256 256 256
  python train.py --filename /path/to/volume.raw --dims 256x256x256 --expname my_exp

  # Planar 3-channel vector field (u,v,w)
  python train.py --filename /path/to/velocity.raw --dims 864 240 640 --n-channels 3

  # Streamline training with trajectory loss (requires --n-channels 3)
  python train.py --filename /path/to/velocity.raw --dims 128 128 128 --n-channels 3 \\
                  --train-traces

  # Streamline training with explicit integration endpoint
  python train.py --filename /path/to/velocity.raw --dims 128 128 128 --n-channels 3 \\
                  --train-traces --trace-tmax 0.05

  # Streamline training tuning all trace flags
  python train.py --filename /path/to/velocity.raw --dims 128 128 128 --n-channels 3 \\
                  --train-traces --trace-resolution 2.0 --trace-batch 64 --trace-tmax 0.1

  # OpenVDB volume (grid name defaults to 'density')
  python train.py --volume-type openvdb --filename /path/to/volume.vdb \\
                  --field density --dims 256 256 256 --expname my_exp

  # Common hyperparameter override
  python train.py --filename /path/to/volume.raw --dims 256 256 256 \\
                  --epochs 128 --n-neurons 64 --n-levels 16

Outputs:
  logs/<expname>/run<N>/   TensorBoard event files (view with: tensorboard --logdir logs/)
  outputs/<expname>.pt     PyTorch state_dict  (useful for fine-tuning or inspection)
  outputs/<expname>.bson   BSON scene file     (model config + fp16 weights + macrocell)

Notes:
  --n-channels expects the raw file to be in PLANAR layout: all U components
  first, then all V, then all W.  Interleaved layout (uvwuvw...) is not
  supported and will produce incorrect results.

  --train-traces adds a trajectory loss alongside pointwise MSE.  Ground-truth
  streamlines are pre-computed once from the discrete velocity field before
  training begins.  The integration endpoint --trace-tmax is auto-derived from
  the mean normalized velocity magnitude (~0.2/mean_speed) if not specified;
  this keeps particle trajectories within the [0,1]^3 domain for most fields.
  Override with --trace-tmax when you need a specific integration length/reach.
"""

import os
import sys
import time
import argparse

import numpy as np

np.random.seed(0)

import contextlib

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.profiler import profile, record_function, ProfilerActivity, schedule as prof_schedule, tensorboard_trace_handler
from tqdm import trange

sys.path.insert(0, os.path.dirname(__file__))

from inrtoolkit import INR_TCNN, default_device, create_logger, create_inr_scene, mse2psnr
from inrtoolkit import create_sampler, sample_volume
from inrtoolkit.training import autocast, gradscaler

try:
    from tracer import INRVectorField, DiscreteGridVectorField, trace_streamlines
    _TRACER_AVAILABLE = True
except ImportError:
    _TRACER_AVAILABLE = False

DEVICE = default_device()

_GT_POOL_SIZE     = 1024  # number of pre-computed GT streamlines in the seed pool
_TRAJ_LOSS_WEIGHT = 0.1   # weight λ on the trajectory loss term: total_loss = point_loss + λ * traj_loss
_WARMUP_EPOCHS    = 5     # default epochs of pointwise-only training before ODE loss activates


def build_model(args, n_output_dims=1):
    """Construct the INR: tiny-cuda-nn hash-grid encoding + fully-fused MLP."""
    return INR_TCNN(
        n_output_dims=n_output_dims,
        n_levels=args.n_levels,
        n_features_per_level=args.n_features,
        log2_hashmap_size=args.log2_hashmap_size,
        base_resolution=4,
        per_level_scale=1.5,
        n_hidden_layers=args.hidden_layers,
        n_neurons=args.n_neurons,
        activation="ReLU",
        output_activation="None",
    )


def sample_training_batch(sampler, batchsize):
    """Return one training batch as device tensors: coords (N,3), targets (N,C)."""
    # sample_volume moves tensors only when backend device differs from DEVICE.
    coords, targets = sample_volume(sampler, batchsize, target_device=DEVICE)
    return coords, targets


def train(expname, sampler, dims, args, output_dir=".", gt_seeds=None, gt_trajs=None):
    n_channels = sampler.n_channels()
    batchsize  = args.batch_size
    numvoxels  = dims[0] * dims[1] * dims[2]
    # Each epoch covers roughly one full pass over the volume
    steps_per_epoch = max(1, (numvoxels + batchsize - 1) // batchsize)
    total_steps = steps_per_epoch * args.epochs

    logger = create_logger("logs", expname)

    model = build_model(args, n_output_dims=n_channels)
    model.to(DEVICE)
    print(model)
    print(f"[info] parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Streamline-aware trajectory loss setup
    use_traj = (
        n_channels == 3
        and getattr(args, "train_traces", False)
        and gt_trajs is not None
        and _TRACER_AVAILABLE
    )
    if use_traj:
        N_pool       = gt_seeds.shape[0]
        gt_seeds_dev = gt_seeds.to(DEVICE)
        gt_trajs_dev = gt_trajs.to(DEVICE)
        t_span       = torch.linspace(0, args.trace_tmax, args.trace_steps, device=DEVICE)
        inr_field    = INRVectorField(model)
        warmup_epochs = getattr(args, "trace_warmup_epochs", _WARMUP_EPOCHS)
        warmup_steps = steps_per_epoch * warmup_epochs
        _adj_str = "adjoint" if getattr(args, "trace_adjoint", True) else "standard"
        print(f"[info] trajectory loss enabled: pool={N_pool}  "
              f"trace-steps={args.trace_steps}  trace-batch={args.trace_batch}  "
              f"t-max={args.trace_tmax:.4f}  warmup={warmup_epochs} epochs ({warmup_steps} steps)  "
              f"backward={_adj_str}")

    if use_traj:
        # eps=1e-15: required for TCNN fp16 weights where gradient scales are ~1e-4 to 1e-7
        optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-15)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=0)
    else:
        optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-8)
        scheduler = optim.lr_scheduler.StepLR(
            optimizer, step_size=max(1, total_steps // 2), gamma=0.5
        )
    scaler = gradscaler()

    do_profile = getattr(args, "profile", False)
    if do_profile:
        prof_dir = os.path.join("logs", expname, "profiler")
        prof_ctx = profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            schedule=prof_schedule(wait=165, warmup=3, active=5, repeat=1),
            on_trace_ready=tensorboard_trace_handler(prof_dir),
            record_shapes=True,
            profile_memory=True,
            with_stack=True,
        )
        print(f"[info] profiler enabled → {prof_dir}")
    else:
        prof_ctx = contextlib.nullcontext()

    progress = trange(1, total_steps + 1)
    t0 = time.time()

    with prof_ctx as prof:
        for step in progress:
            optimizer.zero_grad()

            with record_function("sampling"):
                coords, targets = sample_training_batch(sampler, batchsize)

            with autocast():
                with record_function("forward"):
                    preds = model(coords).float()
                    if n_channels == 1:
                        preds   = preds.squeeze(1)
                        targets = targets.squeeze(1)
                    l2 = F.mse_loss(preds, targets)
                    point_loss = l2 if use_traj else 0.5 * F.l1_loss(preds, targets) + 0.5 * l2

            if use_traj and step > warmup_steps:
                with record_function("trajectory"):
                    idx         = torch.randint(0, N_pool, (args.trace_batch,))
                    batch_seeds = gt_seeds_dev[idx]           # (B, 3)
                    batch_gt    = gt_trajs_dev[:, idx, :]     # (T, B, 3)
                    _use_adj    = getattr(args, "trace_adjoint", True)
                    pred_trajs  = trace_streamlines(inr_field, batch_seeds, t_span, adjoint=_use_adj)  # (T, B, 3)
                    traj_loss   = F.mse_loss(pred_trajs, batch_gt)
                    total_loss  = point_loss + _TRAJ_LOSS_WEIGHT * traj_loss
            else:
                traj_loss  = None
                total_loss = point_loss

            psnr = mse2psnr(l2.detach())

            logger.add_scalar("train/loss",      total_loss, step, new_style=True)
            logger.add_scalar("train/point_mse", l2,         step, new_style=True)
            logger.add_scalar("train/PSNR",      psnr,       step, new_style=True)
            logger.add_scalar("train/lr",
                              scheduler.get_last_lr()[0] if step > 1 else args.lr,
                              step, new_style=True)
            if use_traj and traj_loss is not None:
                logger.add_scalar("train/traj_mse", traj_loss, step, new_style=True)

            with record_function("backward"):
                scaler.scale(total_loss).backward()
                if use_traj:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            scheduler.step()

            if use_traj and traj_loss is not None:
                progress.set_postfix_str(
                    f"pt:{l2:.4f}  tr:{traj_loss:.4f}  psnr:{psnr:.1f}dB", refresh=True
                )
            else:
                progress.set_postfix_str(f"loss:{total_loss:.5f}  psnr:{psnr:.2f}dB", refresh=True)

            if do_profile:
                prof.step()

    if do_profile:
        print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=25))

    elapsed = time.time() - t0
    print(f"[info] training complete in {elapsed:.1f}s  ({total_steps} steps)")

    os.makedirs(output_dir, exist_ok=True)

    pt_path = os.path.join(output_dir, f"{expname}.pt")
    torch.save(model.state_dict(), pt_path)
    print(f"[info] saved model:  {pt_path}")

    try:
        bson_path = os.path.join(output_dir, f"{expname}.bson")
        create_inr_scene(bson_path, dims, model)
        print(f"[info] saved scene:  {bson_path}")
    except Exception as e:
        print(f"[warn] BSON export skipped: {e}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

class _ParseDims(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        if len(values) == 1 and "x" in values[0].lower():
            dims = [int(d) for d in values[0].lower().split("x")]
        else:
            dims = [int(v) for v in values]
        if len(dims) != 3:
            parser.error(f"--dims requires exactly 3 values, got {len(dims)}")
        setattr(namespace, self.dest, dims)


def main():
    parser = argparse.ArgumentParser(
        description="Train a scalar or vector-field INR on structured raw or OpenVDB volumes",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Volume input
    parser.add_argument("--volume-type", default="structuredRegular",
                        choices=["structuredRegular", "openvdb"],
                        help='volume backend: "structuredRegular" (raw) or "openvdb" (.vdb via OpenVKL)',
    )
    parser.add_argument("--filename", required=True,
                        help="path to the volume file (.raw for structuredRegular, .vdb for openvdb)")
    parser.add_argument("--field", default="density",
                        help="OpenVDB grid name (used only when --volume-type openvdb)",
    )
    parser.add_argument("--dims", nargs="+", default=[128, 128, 128], action=_ParseDims,
                        help='training/export dims: "128 128 128" or "128x128x128" (for openvdb, this is not loader input dims)')
    parser.add_argument("--dtype", default="float32",
                        choices=["float32", "float16", "uint8", "uint16"],
                        help="voxel data type for structuredRegular raw files (ignored for openvdb)")
    parser.add_argument("--n-channels", type=int, default=1,
                        help="number of channels in the raw file and output dims of the network "
                             "(1=scalar, 3=vector u/v/w). Raw file must be in planar layout.")

    # Output
    parser.add_argument("--expname", default=None,
                        help="experiment name (default: volume filename stem)")
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
                        help="features per hash level (total input dim = n_levels × n_features)")
    parser.add_argument("--log2-hashmap-size", type=int, default=19,
                        help="log2 of hash table size per level (controls memory vs. quality)")
    parser.add_argument("--n-neurons", type=int, default=64,
                        help="MLP hidden layer width (must be a multiple of 16)")
    parser.add_argument("--hidden-layers", type=int, default=4,
                        help="number of MLP hidden layers")

    # Streamline-aware training (requires --n-channels 3)
    parser.add_argument("--train-traces", action="store_true",
                        help="activate trajectory loss alongside pointwise MSE (requires --n-channels 3)")
    parser.add_argument("--trace-warmup-epochs", type=int, default=_WARMUP_EPOCHS,
                        help="epochs of pointwise-only training before ODE trajectory loss activates; "
                             "increase for large datasets where each epoch covers more of the volume")
    parser.add_argument("--trace-resolution", type=float, default=1.0,
                        help="target integration steps per voxel (CFL multiplier); higher = more "
                             "steps and more accurate integration, lower = fewer steps and faster")
    parser.add_argument("--trace-batch", type=int, default=_GT_POOL_SIZE,
                        help="streamlines traced per training iteration; "
                             "default: full GT pool for maximum gradient coverage at no extra cost")
    parser.add_argument("--trace-tmax", type=float, default=None,
                        help="ODE integration endpoint; default: auto-derived as "
                             "0.2/mean_speed from the normalized velocity field, "
                             "clamped to [0.01, 1.0]")
    parser.add_argument("--trace-adjoint", action=argparse.BooleanOptionalAction, default=False,
                        help="use adjoint sensitivity method for ODE backward pass: O(1) GPU memory "
                             "vs O(trace_steps) for standard backprop. Gradients are identical for "
                             "rk4. Disabled by default: profiling showed 2.5x slower with no memory "
                             "benefit for typical trajectory sizes.")

    # Profiling
    parser.add_argument("--profile", action="store_true",
                        help="enable PyTorch profiler; traces are written to logs/<expname>/profiler/ "
                             "and a summary table is printed after training "
                             "(view with: pip install torch-tb-profiler && tensorboard --logdir logs/)")

    args = parser.parse_args()
    expname = args.expname or os.path.splitext(os.path.basename(args.filename))[0]

    if args.volume_type == "openvdb":
        sampler = create_sampler(
            "openvdb", "openvkl",
            filename=args.filename, field=args.field,
        )
        if args.dtype != "float32":
            print("[warn] --dtype is ignored when --volume-type=openvdb")
        print(f"[info] sampler ready: type=openvdb field={args.field} device=openvkl  export_dims={args.dims}")
    else:
        sampler = create_sampler(
            "structuredRegular", str(DEVICE),
            dims=args.dims, dtype=args.dtype, n_channels=args.n_channels, filename=args.filename,
        )
        print(f"[info] sampler ready: type=structuredRegular dims={args.dims} dtype={args.dtype} "
              f"n_channels={args.n_channels} device={DEVICE}")

    # GT streamline pre-computation for streamline-aware training
    gt_seeds = gt_trajs = None
    if args.n_channels == 3 and args.train_traces and args.volume_type == "structuredRegular":
        if not _TRACER_AVAILABLE:
            raise ImportError(
                "--train-traces requires torchdiffeq.  Install it with: pip install torchdiffeq"
            )
        print("[info] pre-computing GT streamlines …")
        Nx, Ny, Nz = args.dims[0], args.dims[1], args.dims[2]
        _dtype_map  = {"float32": np.float32, "float16": np.float16,
                       "uint8": np.uint8, "uint16": np.uint16}
        raw = np.fromfile(args.filename, dtype=_dtype_map[args.dtype]).astype(np.float32, copy=False)
        raw = raw.reshape(3, Nz, Ny, Nx)
        # Per-channel normalize to [0, 1] — mirrors the C++ CUDA sampler's vmin/vmax scaling.
        for c in range(3):
            vmin, vmax = raw[c].min(), raw[c].max()
            raw[c] = (raw[c] - vmin) / max(vmax - vmin, 1e-8)
        vel_tensor = torch.from_numpy(raw)                # (3, Nz, Ny, Nx) float32, CPU
        gt_field   = DiscreteGridVectorField(vel_tensor)  # CPU, no grad
        # Stratified sampling: divide [0,1]^3 into a grid, one seed per cell.
        # Guarantees uniform domain coverage vs. random clustering.
        _strat_n   = round(_GT_POOL_SIZE ** (1/3))        # cells per axis (~10 for 1024)
        _offsets   = torch.rand(_strat_n**3, 3)
        _idx       = torch.arange(_strat_n**3)
        _iz        = _idx // (_strat_n * _strat_n)
        _iy        = (_idx % (_strat_n * _strat_n)) // _strat_n
        _ix        = _idx % _strat_n
        _grid      = torch.stack([_ix, _iy, _iz], dim=1).float()
        gt_seeds   = (_grid + _offsets) / _strat_n        # (N_pool, 3) in [0, 1]
        mean_speed = float(np.linalg.norm(raw, axis=0).mean())
        if args.trace_tmax is None:
            args.trace_tmax = float(np.clip(0.2 / max(mean_speed, 0.01), 0.01, 1.0))
            print(f"[info] auto trace_tmax={args.trace_tmax:.4f}  "
                  f"(mean_speed={mean_speed:.4f}; override with --trace-tmax)")
        # CFL-based step count: dt = voxel_size / (resolution × mean_speed)
        voxel_size = 1.0 / max(args.dims)
        dt = voxel_size / (args.trace_resolution * max(mean_speed, 1e-6))
        args.trace_steps = max(10, min(int(args.trace_tmax / dt), 100))
        print(f"[info] dynamic tracer: tmax={args.trace_tmax:.4f}  dt={dt:.5f}  steps={args.trace_steps}")
        t_span_cpu = torch.linspace(0, args.trace_tmax, args.trace_steps)
        with torch.no_grad():
            gt_trajs = trace_streamlines(gt_field, gt_seeds, t_span_cpu)  # (T, N_pool, 3)
        print(f"[info] GT pool ready: {_GT_POOL_SIZE} seeds × {args.trace_steps} steps "
              f"→ {tuple(gt_trajs.shape)}")
        del raw, vel_tensor, gt_field  # free memory before training starts

    train(expname, sampler, args.dims, args, output_dir=args.output_dir,
          gt_seeds=gt_seeds, gt_trajs=gt_trajs)


if __name__ == "__main__":
    main()

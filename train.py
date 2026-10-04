"""Train a scalar or vector-field INR on a volume file (raw binary or OpenVDB).

The workflow:
  1. Create a volume sampler (pysampler) to stream random (coord, value) batches
  2. Build a tiny-cuda-nn hash-grid encoding + fully-fused MLP via INR_TCNN
  3. Train with Adam + fp16 AMP and a two-phase StepLR schedule
  4. Export model weights and macrocell acceleration to a self-contained .bson

Usage:
  # Structured raw volume (scalar)
  python train.py --filename /path/to/volume.raw --dims 256 256 256 --expname my_exp

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
                  --train-traces --trace-steps 20 --trace-batch 64 --trace-tmax 0.1

  # Velocity-correction (VC) fine-tune: a brief extra pass on an already-converged
  # no-trace checkpoint, rather than training with trajectory loss from scratch
  python train.py --filename /path/to/velocity.raw --dims 128 128 128 --n-channels 3 \\
                  --train-traces --trace-velocity-correction \\
                  --init-checkpoint outputs/my_notrace_best.pt --epochs 3 --lr 1e-4

  # OpenVDB volume (grid name defaults to 'density')
  python train.py --volume-type openvdb --filename /path/to/volume.vdb \\
                  --field density --dims 256 256 256 --expname my_exp

  # Common hyperparameter override
  python train.py --filename /path/to/volume.raw --dims 256 256 256 \\
                  --epochs 128 --n-neurons 64 --n-levels 16

Outputs:
  logs/<expname>/run<N>/       TensorBoard event files (view with: tensorboard --logdir logs/)
  outputs/<expname>.pt         PyTorch state_dict, final step  (useful for fine-tuning or inspection)
  outputs/<expname>.bson       BSON scene file, final step     (model config + fp16 weights + macrocell)
  outputs/<expname>_best.pt    PyTorch state_dict, best step (lowest EMA-smoothed total_loss seen
                                during training -- guards against a late destabilization leaving
                                the final step worse than an earlier point)
  outputs/<expname>_best.bson  BSON scene file for the same best step

Notes:
  --n-channels expects the raw file to be in PLANAR layout: all U components
  first, then all V, then all W.  Interleaved layout (uvwuvw...) is not
  supported and will produce incorrect results.

  --train-traces adds a trajectory loss alongside pointwise MSE.  Ground-truth
  streamlines are pre-computed once from the discrete velocity field before
  training begins. Velocities are rescaled by a single global scalar (max
  speed over the whole volume) rather than normalized per-channel, since
  per-channel independent min-max normalization preserves each channel's own
  shape but not the 3D direction of the combined vector. A shared scalar
  keeps direction exact.  --trace-tmax (the ODE integration endpoint) should be
  dataset-dependent for best results. It's a required value, not a default: see
  docs/recorded_hyperparameters.md for validated values on known datasets, or
  run tools/calibrate_hyperparams.py to derive one for a new dataset.

  --trace-velocity-correction (independent of --train-traces and can be used
  alone or combined with it) replaces the plain trajectory loss with an
  explicit per-anchor correction: traces the model's own current rollout from 
  GT seeds (on-policy, not the GT path itself), uses the differentiable tracer 
  to compute how the velocity at each drifted anchor should change to reduce 
  the resulting downstream position error, and regresses toward that corrected 
  target as an ordinary supervised loss. Intended to be applied as a quick 
  fine-tune on an already-converged no-trace model (see the Usage example above)
  rather than trained from scratch. --trace-vc-max-correction-frac defaults to 
  0.05, which has held up well across all datasets tested. See 
  docs/recorded_hyperparameters.md if you need to verify it for a new one.
"""

import os
import sys
import time
import argparse

import numpy as np

import torch
import torch.nn.functional as F
import torch.optim as optim
from tqdm import trange

sys.path.insert(0, os.path.dirname(__file__))

from inrtoolkit import INR_TCNN, default_device, create_logger, create_inr_scene, mse2psnr
from inrtoolkit import create_sampler, sample_volume
from inrtoolkit.training import autocast, gradscaler

try:
    from tracer import (INRVectorField, DiscreteGridVectorField, trace_streamlines_until_exit,
                         generate_gt_seed_pool,
                         refresh_on_policy_rollout, sample_on_policy_anchors,
                         compute_velocity_correction)
    _TRACER_AVAILABLE = True
except ImportError:
    _TRACER_AVAILABLE = False

DEVICE = default_device()

# Normalized [0,1]^3 domain (not world coords), used by both GT pre-compute and the live tracer.
_ODE_BOUNDS_UNIT = (0.0, 1.0, 0.0, 1.0, 0.0, 1.0)

_GT_POOL_STRAT_N_REF     = 10    # strat_n at the reference resolution
_GT_POOL_STRAT_N_REF_DIM = 128   # reference grid resolution (matches --dims default)
_GT_POOL_STRAT_N_MIN     = 6     # floor: minimum spatial coverage for tiny volumes (N_pool=216)
_GT_POOL_STRAT_N_MAX     = 32    # cap: bounds CPU pre-compute + worst-case trace_batch (N_pool=32768)
_TRACE_BATCH_CAP         = 1024  # default per-step streamline batch cap, independent of pool size
_TRAJ_LOSS_WEIGHT_DEFAULT = 0.1  # default --trace-loss-weight (λ): total_loss = point_loss + λ * traj_loss
_WARMUP_EPOCHS    = 5     # default epochs of pointwise-only training before ODE loss activates
_TRACE_STEPS_DEFAULT = 10 # testing has shown that more steps didn't improve accuracy, only slowed training

# grad_clip_norm and trace_vc_max_correction_frac are fixed defaults rather than auto-calibrated
# see docs/recorded_hyperparameters.md and tools/calibrate_hyperparams.py
# if you need to re-derive either for a dataset that looks qualitatively different.
_GRAD_CLIP_NORM_DEFAULT = 2.0
_VC_MAX_CORRECTION_FRAC_DEFAULT = 0.05
_VC_LONG_HORIZON_MULTIPLIER_DEFAULT = 15
_VC_K_DEFAULT = 3


def _compute_gt_pool_strat_n(dims, override=None):
    """Cells per axis for the stratified GT seed grid, scaled to grid resolution."""
    if override is not None:
        return override
    scaled = _GT_POOL_STRAT_N_REF * (max(dims) / _GT_POOL_STRAT_N_REF_DIM)
    return int(np.clip(round(scaled), _GT_POOL_STRAT_N_MIN, _GT_POOL_STRAT_N_MAX))


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


def train(expname, sampler, dims, args, output_dir=".", gt_seeds=None, gt_trajs=None,
          value_ranges=None, velocity_scale=None,
          vc_max_correction_frac=None, vc_long_pool=None):
    n_channels = sampler.n_channels()
    batchsize  = args.batch_size
    numvoxels  = dims[0] * dims[1] * dims[2]
    # Each epoch covers roughly one full pass over the volume
    steps_per_epoch = max(1, (numvoxels + batchsize - 1) // batchsize)
    total_steps = steps_per_epoch * args.epochs

    logger = create_logger("logs", expname)

    model = build_model(args, n_output_dims=n_channels)
    if getattr(args, "init_checkpoint", None):
        # Fine-tune an existing checkpoint instead of random init -- e.g. applying
        # --trace-velocity-correction as a brief extra pass on a no-trace checkpoint.
        # Architecture flags must match the checkpoint; a mismatch surfaces only as
        # torch's own shape-mismatch error, not a clear message.
        state_dict = torch.load(args.init_checkpoint, map_location="cpu")
        model.load_state_dict(state_dict)
        print(f"[info] loaded initial weights from {args.init_checkpoint}")
    model.to(DEVICE)
    param_count = sum(p.numel() for p in model.parameters())
    print(model)
    print(f"[info] parameters: {param_count:,}")

    # Streamline-aware trajectory loss setup. Either --train-traces (plain closed-loop
    # trajectory loss) or --trace-velocity-correction (VC) activates this -- they're
    # independent mechanisms (VC's own training step never calls the plain trace-loss
    # code, see the if/elif below) that happen to share this warmup/INRVectorField setup.
    use_traj = (
        n_channels == 3
        and (getattr(args, "train_traces", False) or getattr(args, "trace_velocity_correction", False))
        and (gt_trajs is not None or vc_long_pool is not None)
        and _TRACER_AVAILABLE
    )
    warmup_epochs = 0
    if use_traj:
        if gt_trajs is not None:
            N_pool       = gt_seeds.shape[0]
            gt_seeds_dev = gt_seeds.to(DEVICE)
            gt_trajs_dev = gt_trajs.to(DEVICE)
            t_span       = torch.linspace(0, args.trace_tmax, args.trace_steps, device=DEVICE)
            print(f"[info] trajectory loss enabled: pool={N_pool}  "
                  f"trace-steps={args.trace_steps}  trace-batch={args.trace_batch}  "
                  f"t-max={args.trace_tmax:.4f}")
        inr_field    = INRVectorField(model, value_ranges=value_ranges, velocity_scale=velocity_scale)
        inr_field.to(DEVICE)
        warmup_epochs = getattr(args, "trace_warmup_epochs", _WARMUP_EPOCHS)
        warmup_steps = steps_per_epoch * warmup_epochs
        print(f"[info] warmup={warmup_epochs} epochs ({warmup_steps} steps) before "
              f"trajectory-based training activates")

    # Velocity correction (VC): replaces the plain trajectory loss entirely 
    vc_active = use_traj and getattr(args, "trace_velocity_correction", False) and vc_long_pool is not None
    if vc_active:
        vc_pool_seeds, vc_pool_trajs, vc_pool_exit = vc_long_pool
        vc_pool_seeds = vc_pool_seeds.to(DEVICE)
        vc_pool_trajs = vc_pool_trajs.to(DEVICE)
        vc_pool_exit = vc_pool_exit.to(DEVICE)
        vc_T_pool = vc_pool_trajs.shape[0]
        vc_k = args.trace_vc_k
        vc_dt = (args.trace_tmax * args.trace_vc_long_horizon_multiplier) / (vc_T_pool - 1)
        vc_valid_t_max = (vc_pool_exit - vc_k).clamp(min=0)
        vc_valid_seed_idx = torch.nonzero(vc_pool_exit >= vc_k, as_tuple=True)[0]
        vc_refresh_steps = (args.trace_vc_refresh_steps if args.trace_vc_refresh_steps is not None
                            else int(np.clip(round(total_steps * 0.01), 10, 100)))
        vc_batch = max(256, args.batch_size // 32)
        vc_self_rollout = None
        print(f"[info] [VC] velocity correction active: max_correction_frac="
              f"{vc_max_correction_frac:.4g}  K={vc_k}  refresh_steps={vc_refresh_steps}  "
              f"batch={vc_batch}  anchors={len(vc_valid_seed_idx)}/{vc_pool_seeds.shape[0]} seeds "
              f"have >= {vc_k} in-domain steps")

    # Optimizer/scheduler/clipping held identical regardless of use_traj, so a --train-traces
    # run and its no-trace baseline differ only in the trajectory loss term.
    optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-8)
    scheduler = optim.lr_scheduler.StepLR(
        optimizer, step_size=max(1, total_steps // 2), gamma=0.5
    )
    scaler = gradscaler()

    traj_started    = False
    traj_start_step = None

    grad_clip_norm = args.grad_clip_norm

    # Track the best-seen checkpoint alongside the final one -- guards against a late
    # destabilization leaving the only saved model worse than an earlier point in training.
    # "Best" is judged on an EMA of total_loss rather than the raw per-step value, since a  
    # single batch is noisy.
    _BEST_EMA_DECAY = 0.98
    best_ema_loss = None
    best_loss     = float("inf")
    best_step     = None
    best_state    = None

    progress = trange(1, total_steps + 1)
    t0 = time.time()

    for step in progress:
        optimizer.zero_grad()

        # sample_volume moves tensors only when backend device differs from DEVICE.
        coords, targets = sample_volume(sampler, batchsize, target_device=DEVICE)

        with autocast():
            preds = model(coords).float()
            if n_channels == 1:
                preds   = preds.squeeze(1)
                targets = targets.squeeze(1)
            l2 = F.mse_loss(preds, targets)
            point_loss = 0.5 * F.l1_loss(preds, targets) + 0.5 * l2

        if use_traj:
            traj_active = step > warmup_steps
            if traj_active and not traj_started:
                traj_started, traj_start_step = True, step
        else:
            traj_active = False

        if traj_active and vc_active:
            # Relative to traj_start_step, not absolute `step` -- otherwise, unless
            # warmup_steps is a multiple of vc_refresh_steps, this never fires on the
            # first active step and vc_self_rollout stays None.
            if (step - traj_start_step) % vc_refresh_steps == 0:
                vc_self_rollout = refresh_on_policy_rollout(
                    inr_field, vc_pool_seeds, torch.linspace(
                        0, vc_dt * (vc_T_pool - 1), vc_T_pool, device=DEVICE),
                    _ODE_BOUNDS_UNIT)

            vc_x0, vc_x_target = sample_on_policy_anchors(
                vc_self_rollout, vc_pool_trajs, vc_valid_seed_idx, vc_valid_t_max,
                vc_T_pool, vc_k, vc_batch)
            vc_target_v0 = compute_velocity_correction(
                inr_field, vc_x0, vc_x_target, vc_dt, vc_k, vc_max_correction_frac)
            vc_v0_pred = inr_field(None, vc_x0, clamp=False)
            traj_loss = F.mse_loss(vc_v0_pred, vc_target_v0)

            total_loss = point_loss + traj_loss
        elif traj_active:
            idx = torch.randint(0, N_pool, (args.trace_batch,))

            t_span_active = t_span
            batch_seeds   = gt_seeds_dev[idx]                   # (B, 3)
            batch_gt      = gt_trajs_dev[:, idx, :]

            # Freeze the live prediction at its own exit point too 
            # otherwise a streamline keeps querying the model's less-accurate boundary
            # region, a source of noisy gradient.
            pred_trajs, _ = trace_streamlines_until_exit(
                inr_field, batch_seeds, t_span_active, _ODE_BOUNDS_UNIT)

            traj_loss = F.mse_loss(pred_trajs, batch_gt)

            total_loss = point_loss + args.trace_loss_weight * traj_loss
        else:
            traj_loss  = None
            total_loss = point_loss

        psnr = mse2psnr(l2.detach())

        _loss_val = float(total_loss.detach())
        best_ema_loss = (_loss_val if best_ema_loss is None
                          else _BEST_EMA_DECAY * best_ema_loss + (1 - _BEST_EMA_DECAY) * _loss_val)
        if best_ema_loss < best_loss:
            best_loss  = best_ema_loss
            best_step  = step
            best_state = {k: v.detach().clone().cpu() for k, v in model.state_dict().items()}

        logger.add_scalar("train/loss",      total_loss, step, new_style=True)
        logger.add_scalar("train/point_mse", l2,         step, new_style=True)
        logger.add_scalar("train/PSNR",      psnr,       step, new_style=True)
        logger.add_scalar("train/lr",
                          (scheduler.get_last_lr()[0] if step > 1 else args.lr),
                          step, new_style=True)
        if use_traj and traj_loss is not None:
            logger.add_scalar("train/traj_mse", traj_loss, step, new_style=True)

        scaler.scale(total_loss).backward()
        # Unscale before clipping so clip_grad_norm_ operates on true-magnitude
        # gradients, not the AMP loss-scale-inflated ones.
        scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        logger.add_scalar("train/grad_norm", grad_norm, step, new_style=True)

        if use_traj and traj_loss is not None:
            progress.set_postfix_str(
                f"pt:{l2:.4f}  tr:{traj_loss:.4f}  psnr:{psnr:.1f}dB", refresh=True
            )
        else:
            progress.set_postfix_str(f"loss:{total_loss:.5f}  psnr:{psnr:.2f}dB", refresh=True)

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

    if best_state is not None:
        print(f"[info] best checkpoint: step {best_step}/{total_steps}  "
              f"(smoothed total_loss {best_loss:.6g}, vs {best_ema_loss:.6g} at the final step)")
        best_pt_path = os.path.join(output_dir, f"{expname}_best.pt")
        torch.save(best_state, best_pt_path)
        print(f"[info] saved best model:  {best_pt_path}")

        try:
            model.load_state_dict({k: v.to(DEVICE) for k, v in best_state.items()})
            best_bson_path = os.path.join(output_dir, f"{expname}_best.bson")
            create_inr_scene(best_bson_path, dims, model)
            print(f"[info] saved best scene:  {best_bson_path}")
        except Exception as e:
            print(f"[warn] best-checkpoint BSON export skipped: {e}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

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
    parser.add_argument("--dims", nargs=3, type=int, default=[128, 128, 128],
                        metavar=("NX", "NY", "NZ"),
                        help="training/export dims (for openvdb, this is not loader input dims)")
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
    parser.add_argument("--init-checkpoint", default=None,
                        help="load an existing model's state_dict (a .pt file, e.g. an "
                             "earlier run's *_best.pt) as the starting point instead of "
                             "random init -- for fine-tuning an already-converged model (e.g. "
                             "applying --trace-velocity-correction as a brief extra pass on "
                             "top of a no-trace checkpoint) rather than training from scratch. "
                             "The checkpoint's architecture (n-levels/n-features/"
                             "log2-hashmap-size/n-neurons/hidden-layers) must match this run's.")

    # Training schedule
    parser.add_argument("--epochs", type=int, default=64,
                        help="training epochs")
    parser.add_argument("--batch-size", type=int, default=64 * 1024,
                        help="samples per training step")
    parser.add_argument("--lr", type=float, default=1e-2,
                        help="initial Adam learning rate")
    parser.add_argument("--grad-clip-norm", type=float, default=_GRAD_CLIP_NORM_DEFAULT,
                        help="clip the total gradient norm (torch.nn.utils.clip_grad_norm_) "
                             "to this value before the optimizer step. The default matches "
                             "what held up across every dataset tested (see "
                             "docs/recorded_hyperparameters.md); raise it if you see "
                             "legitimate gradients being clipped away on a new dataset.")

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
    parser.add_argument("--trace-steps", type=int, default=_TRACE_STEPS_DEFAULT,
                        help="ODE integration steps per streamline for both GT pre-computation "
                             "and the live differentiable tracer; more steps = more accurate "
                             "integration but slower. Empirically, going well above the default "
                             "did not improve quality, only cost")
    parser.add_argument("--gt-pool-strat-n", type=int, default=None,
                        help="cells per axis for the stratified GT seed pool "
                             "(N_pool = this^3); default: auto-scaled from --dims, "
                             f"clamped to [{_GT_POOL_STRAT_N_MIN}, {_GT_POOL_STRAT_N_MAX}]")
    parser.add_argument("--trace-batch", type=int, default=None,
                        help="streamlines traced per training iteration; "
                             f"default: min(GT pool size, {_TRACE_BATCH_CAP}) for good gradient "
                             "coverage without runaway per-step ODE cost as the pool grows; "
                             "override manually (e.g. 16-64) if you have VRAM to spare or to spend")
    parser.add_argument("--trace-tmax", type=float, default=None,
                        help="ODE integration endpoint. Required when --train-traces is set -- "
                             "this is dataset-dependent (varies ~4x across the datasets this was "
                             "validated on), so there's no universal default. See "
                             "docs/recorded_hyperparameters.md for validated values on known "
                             "datasets, or run tools/calibrate_hyperparams.py for a new one.")
    parser.add_argument("--trace-loss-weight", type=float, default=_TRAJ_LOSS_WEIGHT_DEFAULT,
                        help="weight lambda on the trajectory loss term: "
                             "total_loss = point_loss + lambda * traj_loss")

    # Velocity correction: replaces the plain closed-loop trajectory loss
    # with an explicit, tracer-proven local correction
    parser.add_argument("--trace-velocity-correction", action="store_true",
                        help="[VC] replace the plain closed-loop trajectory loss with an "
                             "explicit velocity correction: trace the model's own current "
                             "rollout from GT seeds (on-policy), use the differentiable tracer "
                             "to compute exactly how the velocity at the model's OWN drifted "
                             "position should change to reduce the resulting downstream "
                             "position error, and regress toward that corrected target as an "
                             "ordinary supervised loss. Independent of --train-traces -- can "
                             "be used alone or combined with it.")
    parser.add_argument("--trace-vc-max-correction-frac", type=float,
                        default=_VC_MAX_CORRECTION_FRAC_DEFAULT,
                        help="[VC] cap on each correction's size, as a fraction of that "
                             "sample's own |v0|. The default held up across "
                             "every dataset tested (see docs/recorded_hyperparameters.md); "
                             "run tools/calibrate_hyperparams.py if a new dataset's flow "
                             "looks qualitatively different.")
    parser.add_argument("--trace-vc-refresh-steps", type=int, default=None,
                        help="[VC] steps between on-policy self-rollout refreshes (default: "
                             "clamp(round(total_steps*0.01), 10, 100))")
    parser.add_argument("--trace-vc-long-horizon-multiplier", type=int,
                        default=_VC_LONG_HORIZON_MULTIPLIER_DEFAULT,
                        help="[VC] the GT pool used for on-policy anchors spans this many times "
                             "--trace-tmax/--trace-steps (at the same fine dt, not a coarser one "
                             "over the same duration) -- the plain trace loss's own short "
                             "horizon shows negligible on-policy drift for an already-accurate "
                             "model, so a longer horizon is needed for anchors to reflect "
                             "genuinely accumulated error")
    parser.add_argument("--trace-vc-k", type=int, default=_VC_K_DEFAULT,
                        help="[VC] local correction lookahead, in GT-pool steps: how far forward "
                             "the differentiable tracer integrates from an anchor to measure the "
                             "endpoint error used to derive that anchor's correction")

    # Reproducibility
    parser.add_argument("--seed", type=int, default=0,
                        help="seeds numpy/torch (CPU+CUDA): controls GT streamline seed-pool jitter "
                             "and trace-batch sampling.")

    args = parser.parse_args()
    if (args.train_traces or args.trace_velocity_correction) and args.trace_tmax is None:
        parser.error("--trace-tmax is required when --train-traces or --trace-velocity-"
                      "correction is set (it's dataset-dependent -- see "
                      "docs/recorded_hyperparameters.md for validated values, or run "
                      "tools/calibrate_hyperparams.py to derive one for a new dataset)")
    expname = args.expname or os.path.splitext(os.path.basename(args.filename))[0]

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

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

    # GT streamline pre-computation for streamline-aware training. --train-traces and
    # --trace-velocity-correction are independent mechanisms that both need this same 
    # GT field, so either one triggers it.
    gt_seeds = gt_trajs = gt_exit_step = value_ranges = velocity_scale = None
    vc_max_correction_frac = vc_long_pool = None
    if (args.n_channels == 3 and (args.train_traces or args.trace_velocity_correction)
            and args.volume_type == "structuredRegular"):
        if not _TRACER_AVAILABLE:
            raise ImportError(
                "--train-traces/--trace-velocity-correction require torchdiffeq.  "
                "Install it with: pip install torchdiffeq"
            )
        print("[info] pre-computing GT streamlines …")
        Nx, Ny, Nz = args.dims[0], args.dims[1], args.dims[2]
        _dtype_map  = {"float32": np.float32, "float16": np.float16,
                       "uint8": np.uint8, "uint16": np.uint16}
        raw = np.fromfile(args.filename, dtype=_dtype_map[args.dtype]).astype(np.float32, copy=False)
        raw = raw.reshape(3, Nz, Ny, Nx)
        # value_ranges lets INRVectorField invert the model's per-channel-[0,1] output back to
        # physical units; `raw` itself stays in physical units. velocity_scale (a single shared
        # scalar) normalizes for tracing instead of per-channel min-max, which would corrupt
        # direction.
        value_ranges = [(float(raw[c].min()), float(raw[c].max())) for c in range(3)]
        velocity_scale = float(np.linalg.norm(raw, axis=0).max())
        vel_tensor = torch.from_numpy(raw / velocity_scale)  # (3, Nz, Ny, Nx) float32, CPU
        gt_field   = DiscreteGridVectorField(vel_tensor)  # CPU, no grad
        _strat_n   = _compute_gt_pool_strat_n(args.dims, override=args.gt_pool_strat_n)
        print(f"[info] auto GT pool: strat_n={_strat_n} (dims={args.dims}) → "
              f"{_strat_n**3} seeds; override with --gt-pool-strat-n")

        if args.trace_batch is None:
            args.trace_batch = min(_strat_n**3, _TRACE_BATCH_CAP)

        if args.train_traces:
            # Freeze each streamline at its own domain-exit point rather than continuing
            # through DiscreteGridVectorField's border-padded (physically meaningless)
            # velocity -- left unfrozen this is a source of large trace-loss
            # gradients late in training.
            gt_seeds, gt_trajs, gt_exit_step = generate_gt_seed_pool(
                gt_field, _strat_n, args.trace_tmax, args.trace_steps, bounds=_ODE_BOUNDS_UNIT)
            _exit_frac = float((gt_exit_step < args.trace_steps - 1).float().mean())
            print(f"[info] GT pool ready: {gt_seeds.shape[0]} seeds × {args.trace_steps} steps "
                  f"→ {tuple(gt_trajs.shape)}  trace_batch={args.trace_batch}  "
                  f"({_exit_frac:.0%} of seeds exit [0,1]^3 before t-max)")

        if args.trace_velocity_correction:
            vc_max_correction_frac = args.trace_vc_max_correction_frac

            _vc_mult = args.trace_vc_long_horizon_multiplier
            vc_long_pool = generate_gt_seed_pool(
                gt_field, _strat_n, args.trace_tmax * _vc_mult, args.trace_steps * _vc_mult,
                bounds=_ODE_BOUNDS_UNIT)
            print(f"[info] [VC] on-policy long-horizon pool: {vc_long_pool[0].shape[0]} seeds × "
                  f"{args.trace_steps * _vc_mult} steps (x{_vc_mult} the base trace horizon)")

        del raw, vel_tensor, gt_field  # the CPU copies are no longer needed either way

    train(expname, sampler, args.dims, args, output_dir=args.output_dir,
          gt_seeds=gt_seeds, gt_trajs=gt_trajs,
          value_ranges=value_ranges, velocity_scale=velocity_scale,
          vc_max_correction_frac=vc_max_correction_frac, vc_long_pool=vc_long_pool)


if __name__ == "__main__":
    main()

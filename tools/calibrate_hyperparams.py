"""Offline derivation of two trajectory-loss hyperparameters train.py takes as plain,
explicit values: --trace-tmax and --trace-vc-max-correction-frac.

Usage (derive both for a new dataset):
  python tools/calibrate_hyperparams.py \\
      --filename /path/to/velocity.raw --dims 128 128 128 --trace-steps 10

Output: printed recommended --trace-tmax and --trace-vc-max-correction-frac values
"""

import argparse

import numpy as np
import torch
import torch.nn.functional as F

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tracer import DiscreteGridVectorField, trace_streamlines_until_exit, stratified_seeds


def finite_difference_jacobian(velocity_fn, x: torch.Tensor, h: float) -> torch.Tensor:
    """Central finite-difference estimate of d(velocity_fn(x))/dx.

    `velocity_fn` is a plain `(x) -> (N,3)` callable. Uses finite differences instead of
    autograd since tinycudann's CUDA kernels don't implement double-backward.

    Returns (N,3,3): result[:, i, j] = d(velocity_fn output channel i)/d(x_j).
    """
    cols = []
    for j in range(3):
        offset = torch.zeros_like(x)
        offset[:, j] = h
        plus = velocity_fn((x + offset).clamp(0.0, 1.0))
        minus = velocity_fn((x - offset).clamp(0.0, 1.0))
        cols.append((plus - minus) / (2 * h))
    return torch.stack(cols, dim=-1)


def cfl_max_timestep(gt_field: "DiscreteGridVectorField", seeds: torch.Tensor,
                      theta_max: float = 0.1, h: float = None,
                      safe_percentile: float = 5.0) -> float:
    """Curvature-limited safe RK4 timestep, derived from the ground-truth field at the
    seed pool.

    Streamline curvature kappa = |v x a| / |v|^3, with acceleration a = (v.grad)v =
    J @ v. Bounding a single RK4 step's turn angle dtheta ~= kappa*|v|*dt to
    `theta_max` gives a per-point safe step dt_max_i = theta_max/(kappa_i*|v_i|).

    Uses a *low* percentile (default 5th) of dt_max_i, since dt_max_i is smallest
    exactly where curvature is highest -- a high percentile would pick a step too
    large for the points it's meant to protect.

    --trace-tmax = trace_steps * this value.
    """
    h = h if h is not None else 1.0 / 256
    with torch.no_grad():
        v = gt_field(None, seeds)
        J = finite_difference_jacobian(lambda x: gt_field(None, x), seeds, h)  # (N,3,3)
        a = torch.einsum("nij,nj->ni", J, v)
        speed = v.norm(dim=-1)
        kappa = torch.cross(v, a, dim=-1).norm(dim=-1) / speed.clamp(min=1e-8).pow(3)
        dt_max_i = theta_max / (kappa * speed).clamp(min=1e-8)
    return float(dt_max_i.quantile(safe_percentile / 100.0))


def measure_lagrangian_divergence_character(gt_field: "DiscreteGridVectorField", seeds: torch.Tensor,
                                             tmax: float, steps: int, delta: float = 1e-3,
                                             bounds: tuple = (0., 1., 0., 1., 0., 1.)) -> dict:
    """Characterizes how Lagrangian-chaotic a flow is over its own (tmax, steps)
    horizon: traces `seeds` and a `delta`-perturbed copy, checks whether their
    separation keeps growing through the whole horizon (sustained divergence) or
    flattens out partway (saturating).

    Deterministic given the field and seeds (perturbation direction uses a fixed seed).

    Returns:
        late_over_early_growth_ratio: second-half growth rate / first-half growth rate.
            ~1 => sustained; near 0 or negative => saturating.
        saturates: bool, `late_over_early_growth_ratio < 0.5`.
        log_separation: (steps,) mean log-separation at each t_span index, for diagnostics.
    """
    device = seeds.device
    t_span = torch.linspace(0, tmax, steps, device=device)
    _gen = torch.Generator(device=device).manual_seed(0)
    direction = F.normalize(torch.randn(seeds.shape, generator=_gen, device=device), dim=-1)
    seeds_perturbed = seeds + delta * direction
    with torch.no_grad():
        traj_a, _ = trace_streamlines_until_exit(gt_field, seeds, t_span, bounds)
        traj_b, _ = trace_streamlines_until_exit(gt_field, seeds_perturbed, t_span, bounds)
        sep = (traj_a - traj_b).norm(dim=-1)  # (steps, N)
        mean_log_sep = torch.log(sep.clamp(min=1e-12)).mean(dim=-1)  # (steps,)

    mid = steps // 2
    early_dt = float(t_span[mid] - t_span[0])
    late_dt = float(t_span[-1] - t_span[mid])
    early_rate = float(mean_log_sep[mid] - mean_log_sep[0]) / max(early_dt, 1e-12)
    late_rate = float(mean_log_sep[-1] - mean_log_sep[mid]) / max(late_dt, 1e-12)
    ratio = late_rate / early_rate if abs(early_rate) > 1e-8 else 0.0
    return {
        "late_over_early_growth_ratio": ratio,
        "saturates": ratio < 0.5,
        "log_separation": mean_log_sep,
    }


_VC_REF_GROWTH_RATIO = 1.2566
_VC_ANCHOR_FRAC = 0.05
_VC_MAX_CORRECTION_FRAC_MIN = 0.02
_VC_MAX_CORRECTION_FRAC_MAX = 0.08


def calibrate_vc_max_correction_frac(growth_ratio: float) -> float:
    """Derive a suggested --trace-vc-max-correction-frac from the GT field's Lagrangian
    divergence-growth ratio: a more saturating flow (smaller ratio) has less chaos
    to fight, so gets a larger correction; a sustained-divergence flow gets a smaller one.

    Use this function if a new dataset's flow looks qualitatively different (e.g. much more 
    saturating or much more chaotic).
    """
    return float(np.clip(
        _VC_ANCHOR_FRAC * (_VC_REF_GROWTH_RATIO / max(growth_ratio, 1e-6)),
        _VC_MAX_CORRECTION_FRAC_MIN, _VC_MAX_CORRECTION_FRAC_MAX))


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--filename", required=True, help="planar raw velocity field (3-channel, u/v/w)")
    parser.add_argument("--dims", nargs=3, type=int, required=True, metavar=("NX", "NY", "NZ"),
                        help="grid dimensions of the velocity field")
    parser.add_argument("--dtype", default="float32",
                        choices=["float32", "float16", "uint8", "uint16"],
                        help="voxel data type of the raw file")
    parser.add_argument("--trace-steps", type=int, default=10,
                        help="ODE integration steps per streamline -- should match the "
                             "--trace-steps you intend to pass to train.py")
    parser.add_argument("--strat-n", type=int, default=10,
                        help="seed-pool grid resolution per axis (strat_n^3 total seeds)")
    parser.add_argument("--trace-cfl-theta-max", type=float, default=0.1,
                        help="max angular deflection (radians) per RK4 step allowed by the "
                             "CFL-style --trace-tmax derivation")
    parser.add_argument("--trace-cfl-percentile", type=float, default=5.0,
                        help="percentile (of the per-point safe-dt distribution) used for the "
                             "CFL-style derivation -- must be LOW (conservative); see "
                             "cfl_max_timestep's docstring")
    args = parser.parse_args()

    dtype_map = {"float32": np.float32, "float16": np.float16, "uint8": np.uint8, "uint16": np.uint16}
    Nx, Ny, Nz = args.dims
    raw = np.fromfile(args.filename, dtype=dtype_map[args.dtype]).astype(np.float32, copy=False)
    raw = raw.reshape(3, Nz, Ny, Nx)
    velocity_scale = float(np.linalg.norm(raw, axis=0).max())
    vel_tensor = torch.from_numpy(raw / velocity_scale)
    gt_field = DiscreteGridVectorField(vel_tensor)
    seeds = stratified_seeds(args.strat_n)

    safe_dt = cfl_max_timestep(gt_field, seeds, theta_max=args.trace_cfl_theta_max,
                                safe_percentile=args.trace_cfl_percentile)
    trace_tmax = args.trace_steps * safe_dt
    print(f"recommended --trace-tmax {trace_tmax:.4f}  "
          f"(safe_dt={safe_dt:.4f} at p{args.trace_cfl_percentile:g} of "
          f"theta_max={args.trace_cfl_theta_max:g}-bounded per-point steps, "
          f"x --trace-steps {args.trace_steps})")

    divergence = measure_lagrangian_divergence_character(gt_field, seeds, trace_tmax, args.trace_steps)
    growth_ratio = divergence["late_over_early_growth_ratio"]
    vc_frac = calibrate_vc_max_correction_frac(growth_ratio)
    print(f"recommended --trace-vc-max-correction-frac {vc_frac:.4f}  "
          f"(divergence growth_ratio={growth_ratio:.4g}; default 0.05 is probably fine "
          f"unless this is far from ~1.0-1.5)")


if __name__ == "__main__":
    main()

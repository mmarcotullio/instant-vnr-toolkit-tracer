"""Differentiable streamline tracer for steady 3D vector fields."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchdiffeq import odeint, odeint_adjoint


class DiscreteGridVectorField(nn.Module):
    """Trilinear interpolation on a (3, Nz, Ny, Nx) velocity grid.

    Coords in [0,1]^3, converted to grid_sample's [-1,1] internally.
    """

    def __init__(
        self,
        velocity: torch.Tensor,
        padding_mode: str = "border",
    ) -> None:
        super().__init__()

        if velocity.ndim != 4 or velocity.shape[0] != 3:
            raise ValueError(
                f"velocity must have shape (3, Nz, Ny, Nx); got {tuple(velocity.shape)}"
            )

        # Buffer, not parameter -- moves with .to() but not optimized
        self.register_buffer("_velocity", velocity.float().unsqueeze(0))  # (1,3,Nz,Ny,Nx)
        self.padding_mode = padding_mode

    def forward(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """(N,3) positions in [0,1] → (N,3) interpolated velocity."""
        N = x.shape[0]
        coords_norm = x * 2.0 - 1.0
        grid = coords_norm.view(1, 1, 1, N, 3)
        sampled = F.grid_sample(
            self._velocity,
            grid,
            mode="bilinear",  # trilinear for 5-D
            padding_mode=self.padding_mode,
            align_corners=True,
        )
        return sampled.view(3, N).T


class INRVectorField(nn.Module):
    """Wraps an INR_TCNN (n_output_dims=3) as an ODE right-hand-side.

    Clamps positions to [0,1] before querying (border padding equivalent).

    Rescales by one shared scalar across all 3 channels rather than independent
    per-channel min-max, since a uniform scalar preserves the vector's 3D
    direction and per-channel normalization doesn't.

    A raw output of 0 denormalizes to the value range's minimum, not zero
    velocity. An untrained network emits a large constant drift before
    learning anything, which can blow up the trace loss within tens of steps
    if unbounded. `max_speed_multiple` clamps the velocity's *norm* (not its
    components, to keep direction exact) before it's fed to the ODE.
    """

    def __init__(self, model: nn.Module, value_ranges=None, velocity_scale=None,
                 max_speed_multiple=2.0) -> None:
        super().__init__()
        self.model = model
        if value_ranges is not None:
            mins = torch.tensor([lo for lo, hi in value_ranges], dtype=torch.float32)
            spans = torch.tensor([hi - lo for lo, hi in value_ranges], dtype=torch.float32)
            self.register_buffer("_range_min", mins)
            self.register_buffer("_range_span", spans)
        else:
            self._range_min = None
            self._range_span = None
        self.velocity_scale = velocity_scale
        self.max_speed_multiple = max_speed_multiple

    def forward(self, t: torch.Tensor, x: torch.Tensor, clamp: bool = True) -> torch.Tensor:
        """(N,3) positions in [0,1] → (N,3) predicted velocity (see class docstring).

        `clamp=False` skips the norm-clamp, needed for finite-difference Jacobian
        estimation (a clamp would flatten the local response). ODE integration
        always uses clamp=True.
        """
        raw = self.model(x.clamp(0.0, 1.0))
        if self._range_min is None:
            return raw
        physical = raw * self._range_span + self._range_min
        scaled = physical / self.velocity_scale
        if not clamp:
            return scaled
        # Cap norm, not components, to preserve exact direction.
        norm = scaled.norm(dim=-1, keepdim=True)
        factor = (self.max_speed_multiple / norm).clamp(max=1.0)
        return scaled * factor


def trace_streamlines(
    vector_field: nn.Module,
    initial_positions: torch.Tensor,
    t_span: torch.Tensor,
    method: str = "rk4",
    adjoint: bool = False,
    **odeint_kwargs,
) -> torch.Tensor:
    """Integrate a vector field to produce differentiable particle trajectories.

    Args:
        vector_field:      forward(t, x) ODE, coords in [0,1]
        initial_positions: (N,3) seed positions
        t_span:            (T,) time points for output
        method:            'rk4', 'dopri5', 'euler', etc.
        adjoint:           use adjoint method for O(1) memory backward

    Returns: (T, N, 3) trajectories
    """
    if adjoint:
        return odeint_adjoint(
            vector_field,
            initial_positions,
            t_span,
            method=method,
            adjoint_params=tuple(vector_field.parameters()),
            **odeint_kwargs,
        )
    return odeint(
        vector_field,
        initial_positions,
        t_span,
        method=method,
        **odeint_kwargs,
    )


def trace_streamlines_until_exit(
    vector_field: nn.Module,
    initial_positions: torch.Tensor,
    t_span: torch.Tensor,
    bounds: tuple,
    method: str = "rk4",
    adjoint: bool = False,
    **odeint_kwargs,
) -> tuple:
    """Like trace_streamlines, but each streamline freezes in place the first time
    it leaves `bounds`, instead of continuing through border-padded velocity outside
    the domain. Steps one t_span interval at a time so a freeze mask can be applied
    between steps.

    Args:
        bounds: (xmin, xmax, ymin, ymax, zmin, zmax), same units as initial_positions

    Returns:
        trajectories: (T, N, 3), held at each seed's last in-bounds point from
            its exit step onward
        exit_step: (N,) index into t_span of each seed's exit point (T-1 if it
            never left `bounds`)
    """
    xmin, xmax, ymin, ymax, zmin, zmax = bounds
    lo = initial_positions.new_tensor([xmin, ymin, zmin])
    hi = initial_positions.new_tensor([xmax, ymax, zmax])

    T = t_span.shape[0]
    N = initial_positions.shape[0]
    trajectories = initial_positions.new_empty(T, N, 3)
    trajectories[0] = initial_positions
    exit_step = torch.full((N,), T - 1, dtype=torch.long, device=initial_positions.device)
    exited = torch.zeros(N, dtype=torch.bool, device=initial_positions.device)

    current = initial_positions
    for i in range(1, T):
        stepped = trace_streamlines(
            vector_field, current, t_span[i - 1:i + 1], method=method, adjoint=adjoint, **odeint_kwargs
        )[-1]
        current = torch.where(exited.unsqueeze(-1), current, stepped)
        newly_exited = (~exited) & ((current < lo) | (current > hi)).any(dim=-1)
        exit_step[newly_exited] = i
        exited = exited | newly_exited
        trajectories[i] = current

    return trajectories, exit_step


def stratified_seeds(strat_n: int, device=None) -> torch.Tensor:
    """One random-jittered seed per cell of a strat_n^3 grid over [0,1]^3 -- guarantees
    uniform domain coverage vs. random clustering. Returns (strat_n^3, 3)."""
    device = device if device is not None else "cpu"
    offsets = torch.rand(strat_n**3, 3, device=device)
    idx = torch.arange(strat_n**3, device=device)
    iz = idx // (strat_n * strat_n)
    iy = (idx % (strat_n * strat_n)) // strat_n
    ix = idx % strat_n
    grid = torch.stack([ix, iy, iz], dim=1).float()
    return (grid + offsets) / strat_n


def generate_gt_seed_pool(gt_field: "DiscreteGridVectorField", strat_n: int, trace_tmax: float,
                           trace_steps: int, bounds: tuple = (0., 1., 0., 1., 0., 1.),
                           device=None) -> tuple:
    """Fresh stratified-random seed pool + GT trajectories (frozen at domain exit).

    Returns (gt_seeds, gt_trajs, gt_exit_step):
        gt_seeds: (strat_n^3, 3)
        gt_trajs: (trace_steps, strat_n^3, 3)
        gt_exit_step: (strat_n^3,) index into t_span of each seed's domain-exit step
            (trace_steps - 1 if it never exits)
    """
    device = device if device is not None else next(gt_field.buffers()).device
    gt_seeds = stratified_seeds(strat_n, device=device)
    t_span = torch.linspace(0, trace_tmax, trace_steps, device=device)
    with torch.no_grad():
        gt_trajs, gt_exit_step = trace_streamlines_until_exit(gt_field, gt_seeds, t_span, bounds)
    return gt_seeds, gt_trajs, gt_exit_step


def refresh_on_policy_rollout(inr_field, pool_seeds: torch.Tensor, t_span: torch.Tensor,
                               bounds: tuple) -> torch.Tensor:
    """Traces the model's own current dynamics from the GT seed pool. Called
    periodically during VC training so on-policy anchors reflect up-to-date drift.
    """
    with torch.no_grad():
        self_traj, _ = trace_streamlines_until_exit(inr_field, pool_seeds, t_span, bounds)
    return self_traj


def sample_on_policy_anchors(self_rollout_trajs: torch.Tensor, pool_trajs: torch.Tensor,
                              valid_seed_idx: torch.Tensor, valid_t_max: torch.Tensor,
                              T_POOL: int, K: int, batch_size: int) -> tuple:
    """Samples VC correction anchors: x0 is where the model's own rollout actually
    is at a random t_start (real compounded drift). x_target is the true position K steps later.

    valid_seed_idx/valid_t_max restrict sampling to seeds with >= K in-domain steps
    remaining from t_start, so x_target is always a genuine GT point.
    """
    device = self_rollout_trajs.device
    sidx = valid_seed_idx[torch.randint(0, len(valid_seed_idx), (batch_size,), device=device)]
    t_max = valid_t_max[sidx].float()
    t_start = (torch.rand(batch_size, device=device) * (t_max + 1)).long().clamp(max=T_POOL - 1 - K)
    x0 = self_rollout_trajs[t_start, sidx, :]
    x_target = pool_trajs[t_start + K, sidx, :]
    return x0, x_target


def compute_velocity_correction(inr_field, x0: torch.Tensor, x_target: torch.Tensor,
                                 dt: float, K: int, max_correction_frac: float) -> torch.Tensor:
    """Derives one velocity-correction target per anchor point (see train.py's
    --trace-velocity-correction).

    Evaluates v0 at x0, then asks: which direction would make a K-step
    forward-Euler rollout from x0 land closer to x_target? Answered by
    backpropagating the rollout's endpoint error into a detached copy of v0 (one
    `torch.autograd.grad` call, no `create_graph`). Isolated from model
    parameters, used only to compute a target for an ordinary regression loss
    elsewhere, not backpropped through directly.

    The correction is normalized to a FIXED size (`max_correction_frac * |v0|`) in
    the gradient's direction, not the raw gradient magnitude, keeping it
    dataset-scale-invariant (an anchor already on the true path yields zero
    correction).

    Returns target_v0 (B, 3), fully detached.
    """
    with torch.enable_grad():
        v0 = inr_field(None, x0, clamp=False)
        v0_leaf = v0.detach().clone().requires_grad_(True)
        x = x0 + dt * v0_leaf
        for _ in range(1, K):
            v = inr_field(None, x, clamp=False)
            x = x + dt * v
        rollout_loss = F.mse_loss(x, x_target)
        grad_v0 = torch.autograd.grad(rollout_loss, v0_leaf)[0]

    v0_norm = v0.detach().norm(dim=-1, keepdim=True).clamp_min(1e-8)
    correction_norm = grad_v0.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    direction = -grad_v0 / correction_norm
    correction = direction * max_correction_frac * v0_norm
    return v0.detach() + correction

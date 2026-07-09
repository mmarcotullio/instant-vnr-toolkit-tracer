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

        # Buffer, not parameter — moves with .to() but not optimized
        self.register_buffer("_velocity", velocity.float().unsqueeze(0))  # (1,3,Nz,Ny,Nx)
        self.padding_mode = padding_mode

    def forward(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """(N,3) positions in [0,1] → (N,3) interpolated velocity."""
        N = x.shape[0]
        coords_norm = x * 2.0 - 1.0                          # [0,1] → [-1,1]
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
    """

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """(N,3) positions in [0,1] → (N,3) predicted velocity."""
        return self.model(x.clamp(0.0, 1.0))


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

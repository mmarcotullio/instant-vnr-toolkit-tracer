"""Utility functions: device selection, logging, metrics, model export."""

import os
import json

import numpy as np
import torch


def default_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        return torch.device("xpu")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def mse2psnr(mse, data_range=1.0):
    if torch.is_tensor(mse):
        return -10.0 * torch.log10(mse / (data_range ** 2))
    return -10.0 * np.log10(mse / (data_range ** 2))


def batchify(fn, chunk, device=None):
    """Wrap fn to process inputs in chunks (avoids OOM on large grids)."""
    if chunk is None:
        return fn
    if device is None:
        device = default_device()

    def _batched(inputs):
        N = inputs.shape[0]
        return torch.cat([
            fn(inputs[i: min(i + chunk, N)].to(device))
            for i in range(0, N, chunk)
        ], dim=0)

    return _batched


def create_logger(basedir, expname):
    from torch.utils.tensorboard import SummaryWriter
    import shutil

    exp_dir = os.path.join(basedir, expname)
    os.makedirs(exp_dir, exist_ok=True)

    runs = sorted(f for f in os.listdir(exp_dir) if f.startswith("run"))
    run_id = int(runs[-1][3:]) + 1 if runs else 0
    model_dir = os.path.join(exp_dir, f"run{run_id:05d}")

    logger = SummaryWriter(model_dir)
    logger.model_dir = model_dir
    return logger


def compute_macrocell(vdims, model, mcsize_mip=4):
    device = default_device()
    mcsize = 1 << mcsize_mip
    mcdims = [(vdims[i] + mcsize - 1) // mcsize for i in range(3)]
    vscale = torch.tensor([1 / d for d in vdims], dtype=torch.float32, device=device)
    mcrange = torch.zeros(mcdims[2], mcdims[1], mcdims[0], 2)

    def make_grid(dx, dy, dz):
        x, y, z = torch.meshgrid(
            torch.linspace(0, dx - 1, dx, dtype=torch.int32, device=device),
            torch.linspace(0, dy - 1, dy, dtype=torch.int32, device=device),
            torch.linspace(0, dz - 1, dz, dtype=torch.int32, device=device),
            indexing="ij",
        )
        return torch.stack((x, y, z), dim=3).permute(2, 1, 0, 3).reshape(-1, 3)

    grid = make_grid(mcsize + 1, mcsize + 1, mcsize + 1).float() - 0.5
    n = grid.shape[0]

    for iz in range(mcdims[2]):
        m = mcdims[1] * mcdims[0]
        o = make_grid(mcdims[0], mcdims[1], 1).float() + torch.tensor([0, 0, iz], dtype=torch.float32, device=device)
        x = grid.repeat(m, 1).reshape(m, n, 3) + o.unsqueeze(1) * mcsize
        x = x.reshape(-1, 3)
        with torch.no_grad():
            y = model((x * vscale).clamp(0, 1)).float()
            # Vector field: reduce (N,C) → (N,) using L2 speed = sqrt(u²+v²+w²)
            # so the macrocell stores a meaningful scalar occupancy metric.
            if y.shape[-1] > 1:
                y = y.norm(dim=-1)
            else:
                y = y.squeeze(-1)
            y = y.reshape(m, n)
        vmin, _ = y.min(dim=1)
        vmax, _ = y.max(dim=1)
        mcrange[iz, :, :, 0] = vmin.reshape(mcdims[1], mcdims[0])
        mcrange[iz, :, :, 1] = vmax.reshape(mcdims[1], mcdims[0])

    mcrange[..., 0] -= 1
    mcrange[..., 1] += 1
    return {
        "volumedims": vdims,
        "dims": mcdims,
        "spacings": [mcsize / vdims[i] for i in range(3)],
        "value_ranges": mcrange.cpu().numpy(),
    }


def create_inr_scene(fn, dims, model):
    """Export trained INR to BSON format for C++ renderer consumption."""
    import bson

    mc = compute_macrocell(dims, model)
    state = model.state_dict()

    params_binary = state["params"].half().cpu().numpy().tobytes()

    root = bson.dumps({
        "macrocell": {
            "data": mc["value_ranges"].tobytes(),
            "dims": {"x": mc["dims"][0], "y": mc["dims"][1], "z": mc["dims"][2]},
            "groundtruth": False,
            "spacings": {"x": mc["spacings"][0], "y": mc["spacings"][1], "z": mc["spacings"][2]},
        },
        "model": {
            "encoding": model.encoding_config,
            "loss": {"otype": "L1"},
            "network": model.network_config,
            "n_output_dims": int(model.n_output_dims),
        },
        "parameters": {"params_binary": params_binary},
        "volume": {"dims": {"x": dims[0], "y": dims[1], "z": dims[2]}},
    })

    with open(fn, "wb") as f:
        f.write(root)

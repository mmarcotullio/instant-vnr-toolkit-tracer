"""End-to-end training tests: synthetic volume → INR_TCNN → BSON.

Verifies:
  - model converges on a smooth analytical volume (PSNR > 25 dB)
  - exported .bson contains all required keys with correct shapes
  - parameter count in .bson matches the live model
"""

import os
import tempfile

import numpy as np
import pytest
import torch
import torch.nn.functional as F

try:
    import pysampler as _pysampler  # noqa: F401
    _HAS_PYSAMPLER = True
except ImportError:
    _HAS_PYSAMPLER = False

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or not _HAS_PYSAMPLER,
    reason="requires a CUDA GPU and pysampler",
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_smooth_volume(dims):
    """Gaussian blob at the volume centre, pre-normalised to [0, 1].

    Pre-normalising means pysampler's (v - vmin)/(vmax - vmin) is identity,
    so ground-truth values at query coords equal the raw volume values.
    """
    Dx, Dy, Dz = dims
    xs = np.linspace(0, 1, Dx, dtype=np.float32)
    ys = np.linspace(0, 1, Dy, dtype=np.float32)
    zs = np.linspace(0, 1, Dz, dtype=np.float32)
    xg, yg, zg = np.meshgrid(xs, ys, zs, indexing="ij")
    vol = np.exp(-((xg - 0.5)**2 + (yg - 0.5)**2 + (zg - 0.5)**2) / 0.1)
    return ((vol - vol.min()) / (vol.max() - vol.min())).astype(np.float32)


# ---------------------------------------------------------------------------
# Module-scoped fixture: one training run shared across all tests
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def trained(tmp_path_factory):
    """Train a tiny INR on a 16³ synthetic volume.  Returns a dict with:
      model, dims, sampler, raw_path, bson_path
    """
    from inrtoolkit import INR_TCNN, default_device, create_inr_scene
    from inrtoolkit import create_sampler, sample_volume
    from inrtoolkit.training import autocast, gradscaler

    dims = [16, 16, 16]
    vol  = _make_smooth_volume(dims)

    tmp      = tmp_path_factory.mktemp("train_test")
    raw_path = str(tmp / "sphere.raw")
    vol.tofile(raw_path)

    device  = default_device()
    sampler = create_sampler(
        "structuredRegular", str(device),
        dims=dims, dtype="float32", n_channels=1, filename=raw_path,
    )

    # Small network — enough to fit a smooth 16³ volume quickly
    model = INR_TCNN(
        n_levels=8, n_features_per_level=4, log2_hashmap_size=14,
        n_neurons=32, n_hidden_layers=2,
    )
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    scaler    = gradscaler()
    batchsize = 4096
    # 128 steps is more than enough to converge on a smooth 16³ volume
    for _ in range(128):
        optimizer.zero_grad()
        coords, targets = sample_volume(sampler, batchsize)
        with autocast():
            preds = model(coords).float().squeeze(1)
            loss  = F.mse_loss(preds, targets.squeeze(1))
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

    bson_path = str(tmp / "sphere.bson")
    create_inr_scene(bson_path, dims, model)

    return dict(model=model, dims=dims, sampler=sampler,
                raw_path=raw_path, bson_path=bson_path, device=device)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_training_psnr(trained):
    """Model must achieve ≥ 25 dB PSNR on random interior query points."""
    from inrtoolkit import mse2psnr
    from inrtoolkit.sampler import decode
    from inrtoolkit.training import autocast

    model   = trained["model"]
    sampler = trained["sampler"]
    device  = trained["device"]

    rng       = np.random.default_rng(42)
    coords_np = rng.uniform(0.05, 0.95, (2048, 3)).astype(np.float32)
    coords_t  = torch.from_numpy(coords_np).to(device)

    # Ground truth via sampler decode (same normalization as training)
    gt = decode(sampler, coords_t).squeeze(1)

    model.eval()
    with torch.no_grad(), autocast():
        preds = model(coords_t).float().squeeze(1)

    psnr = mse2psnr(F.mse_loss(preds, gt).item())
    print(f"\n[test] PSNR = {psnr:.2f} dB")
    assert psnr > 25.0, f"expected PSNR > 25 dB, got {psnr:.2f} dB"


def test_bson_file_exists(trained):
    """create_inr_scene must produce a non-empty file."""
    bson_path = trained["bson_path"]
    assert os.path.exists(bson_path), ".bson file not created"
    assert os.path.getsize(bson_path) > 0, ".bson file is empty"


def test_bson_schema(trained):
    """Exported .bson must have all required top-level keys with correct types."""
    import bson

    with open(trained["bson_path"], "rb") as f:
        scene = bson.loads(f.read())

    for key in ("model", "parameters", "volume", "macrocell"):
        assert key in scene, f"missing key '{key}'"

    enc = scene["model"]["encoding"]
    assert enc["otype"] == "HashGrid", "encoding type mismatch"
    assert "n_levels" in enc

    vd = scene["volume"]["dims"]
    assert [vd["x"], vd["y"], vd["z"]] == trained["dims"]


def test_bson_param_count(trained):
    """Parameter binary in .bson must match the live model's parameter count."""
    import bson

    with open(trained["bson_path"], "rb") as f:
        scene = bson.loads(f.read())

    params_bytes = scene["parameters"]["params_binary"]
    n_bson = len(np.frombuffer(params_bytes, dtype=np.float16))
    n_live = sum(p.numel() for p in trained["model"].parameters())
    assert n_bson == n_live, f"param count mismatch: bson={n_bson}, model={n_live}"

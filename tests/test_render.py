"""Smoke tests for render.py: batch mode via instantvnr.apps.run_batch.

Requires:
  - CUDA GPU
  - pysampler (to build the test .bson)
  - instantvnr-py (to run the renderer)

Skipped automatically if either package is missing.
"""

import os
import shutil
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn.functional as F

try:
    import pysampler as _pysampler  # noqa: F401
    _HAS_PYSAMPLER = True
except ImportError:
    _HAS_PYSAMPLER = False

try:
    import instantvnr as _instantvnr  # noqa: F401
    _HAS_INSTANTVNR = True
except ImportError:
    _HAS_INSTANTVNR = False

_REQUIRES_GPU_AND_SAMPLER = pytest.mark.skipif(
    not torch.cuda.is_available() or not _HAS_PYSAMPLER,
    reason="requires CUDA GPU and pysampler",
)
_REQUIRES_INSTANTVNR = pytest.mark.skipif(
    not _HAS_INSTANTVNR,
    reason="instantvnr-py not installed — build with: cd packages/instantvnr && ./setup_venv.sh",
)

_GOLDEN_DIR = Path(__file__).parent / "golden"
_BATCH_RENDER_GOLDEN = _GOLDEN_DIR / "batch_render_sincos_128.png"


def _make_sincos_volume(dims):
    """Return a smooth deterministic scalar field in [0, 1]."""
    dx, dy, dz = dims
    x = np.linspace(0.0, 1.0, dx, dtype=np.float32)
    y = np.linspace(0.0, 1.0, dy, dtype=np.float32)
    z = np.linspace(0.0, 1.0, dz, dtype=np.float32)
    x_grid, y_grid, z_grid = np.meshgrid(x, y, z, indexing="ij")

    volume = (
        0.5
        + 0.25 * np.sin(2.0 * np.pi * x_grid)
        + 0.15 * np.cos(2.0 * np.pi * y_grid)
        + 0.10 * np.sin(4.0 * np.pi * z_grid)
    )
    volume = (volume - volume.min()) / (volume.max() - volume.min())
    return volume.astype(np.float32).transpose(2, 1, 0)  # (Dz, Dy, Dx)


def _read_png_dimensions(path: str):
    """Return PNG (width, height) from the IHDR chunk."""
    with open(path, "rb") as handle:
        header = handle.read(24)

    png_signature = b"\x89PNG\r\n\x1a\n"
    assert header[:8] == png_signature, "output file is not a PNG"
    assert header[12:16] == b"IHDR", "PNG header missing IHDR chunk"

    width = int.from_bytes(header[16:20], byteorder="big", signed=False)
    height = int.from_bytes(header[20:24], byteorder="big", signed=False)
    return width, height


def _read_image(path: str):
    """Read a PNG into float32 RGB/RGBA values in [0, 1]."""
    import matplotlib.image as mpimg

    image = mpimg.imread(path)
    if image.dtype == np.uint8:
        image = image.astype(np.float32) / 255.0
    return image.astype(np.float32)


def _assert_matches_golden(output: str, golden: Path):
    """Compare a rendered PNG with its golden image, with small GPU tolerance."""
    if os.environ.get("VNR_UPDATE_GOLDEN") == "1":
        golden.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(output, golden)
        pytest.skip(f"updated golden image: {golden}")

    if not golden.exists():
        pytest.skip(
            f"missing golden image: {golden}. "
            "Run with VNR_UPDATE_GOLDEN=1 to create it."
        )

    actual = _read_image(output)
    expected = _read_image(str(golden))
    assert actual.shape == expected.shape, (
        f"golden image shape mismatch: actual={actual.shape}, expected={expected.shape}"
    )

    diff = np.abs(actual - expected)
    assert float(diff.mean()) <= 1.0 / 255.0, (
        f"render differs from golden: mean abs diff={float(diff.mean()):.6f}"
    )
    assert float(diff.max()) <= 8.0 / 255.0, (
        f"render differs from golden: max abs diff={float(diff.max()):.6f}"
    )


# ---------------------------------------------------------------------------
# Fixture: a minimal .bson (a few training steps; quality does not matter)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def minimal_bson(tmp_path_factory):
    """Produce a .bson from a deterministic smooth volume with minimal training."""
    from inrtoolkit import INR_TCNN, default_device, create_inr_scene
    from inrtoolkit import create_sampler, sample_volume
    from inrtoolkit.training import autocast, gradscaler

    dims   = [16, 16, 16]
    vol    = _make_sincos_volume(dims)
    tmp    = tmp_path_factory.mktemp("render_test")
    raw    = str(tmp / "vol.raw")
    vol.tofile(raw)

    device  = default_device()
    sampler = create_sampler(
        "structuredRegular", str(device),
        dims=dims, dtype="float32", n_channels=1, filename=raw,
    )

    model = INR_TCNN(n_levels=4, n_features_per_level=2, log2_hashmap_size=12,
                     n_neurons=16, n_hidden_layers=1)
    model.to(device)

    opt    = torch.optim.Adam(model.parameters(), lr=1e-2)
    scaler = gradscaler()

    for _ in range(32):
        opt.zero_grad()
        coords, targets = sample_volume(sampler, 512)
        with autocast():
            preds = model(coords).float().squeeze(1)
            loss  = F.mse_loss(preds, targets.squeeze(1))
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()

    bson = str(tmp / "test.bson")
    create_inr_scene(bson, dims, model)
    return bson, tmp


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@_REQUIRES_GPU_AND_SAMPLER
@_REQUIRES_INSTANTVNR
def test_batch_render_creates_file(minimal_bson):
    """run_batch must exit 0 and create a valid PNG with requested dimensions."""
    from instantvnr.apps import run_batch

    scene, tmp = minimal_bson
    output = str(tmp / "out.png")

    ret = run_batch(
        neural_volume=scene,
        output=output,
        width=128,
        height=128,
        spp=1,
        warmup=5,
        rendering_mode=0,
        camera_from=[0, 0, -30],
        camera_at=[0, 0, 0],
        camera_up=[0, 1, 0],
    )

    assert ret == 0, f"vnr_batch exited with code {ret}"
    assert os.path.exists(output), "output PNG was not created"
    assert os.path.getsize(output) > 0, "output PNG is empty"
    assert _read_png_dimensions(output) == (128, 128), "output PNG dimensions mismatch"
    _assert_matches_golden(output, _BATCH_RENDER_GOLDEN)


@_REQUIRES_GPU_AND_SAMPLER
@_REQUIRES_INSTANTVNR
def test_batch_render_higher_spp(minimal_bson):
    """run_batch with spp > 1 must also succeed."""
    from instantvnr.apps import run_batch

    scene, tmp = minimal_bson
    output = str(tmp / "out_spp4.png")

    ret = run_batch(
        neural_volume=scene,
        output=output,
        width=64,
        height=64,
        spp=4,
        warmup=2,
        rendering_mode=0,
        camera_from=[0, 0, -30],
        camera_at=[0, 0, 0],
        camera_up=[0, 1, 0],
    )

    assert ret == 0, f"vnr_batch exited with code {ret}"
    assert os.path.exists(output)


@_REQUIRES_GPU_AND_SAMPLER
@_REQUIRES_INSTANTVNR
def test_run_batch_api_surface(minimal_bson):
    """run_batch must raise ValueError when both volume arguments are given."""
    from instantvnr.apps import run_batch

    scene, _ = minimal_bson
    with pytest.raises(ValueError, match="exactly one"):
        run_batch(simple_volume="dummy.raw", neural_volume=scene)

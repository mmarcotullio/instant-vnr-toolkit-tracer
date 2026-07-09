"""
Tests for the file-backed CPU samplers ('virtual_memory' and 'out_of_core').

Both backends:
  - read raw little-endian voxel data from a single open file (mmap for
    `virtual_memory`; pread() into a heap-resident block cache for
    `out_of_core`).
  - normalize per voxel using `range=[vmin, vmax]` (clamped to [0, 1])
  - then trilinearly interpolate the normalized values
    (the same "normalize-before-interpolate" semantics as InstantVNR's
    `VirtualMemorySampler` and `OutOfCoreSampler`).

`pysampler.sample()` and `pysampler.decode()` for these backends take HOST
pointers (this matches the OpenVKL CPU backend), so the tests exchange
numpy arrays via `arr.ctypes.data` and do not require a CUDA-capable GPU.
"""
import os
import tempfile

import numpy as np
import pytest

import pysampler


# ---------------------------------------------------------------------------
# Reference implementation
# ---------------------------------------------------------------------------

def _vm_reference(volume: np.ndarray, coords: np.ndarray, vmin: float, vmax: float) -> np.ndarray:
    """Software reference matching the file-backed sampler's
    'normalize-before-interpolate' behavior.

    Parameters
    ----------
    volume : float ndarray, shape (Dz, Dy, Dx)
    coords : float32 ndarray, shape (N, 3) — sample positions in [0, 1]^3
    vmin, vmax : float — explicit normalization range
    """
    Dz, Dy, Dx = volume.shape

    # CUDA tex3D address formula: f = u*N - 0.5, clamped to [0, N-1]
    fx = np.clip(coords[:, 0] * Dx - 0.5, 0.0, Dx - 1.0)
    fy = np.clip(coords[:, 1] * Dy - 0.5, 0.0, Dy - 1.0)
    fz = np.clip(coords[:, 2] * Dz - 0.5, 0.0, Dz - 1.0)

    ix0 = np.floor(fx).astype(np.int32)
    iy0 = np.floor(fy).astype(np.int32)
    iz0 = np.floor(fz).astype(np.int32)
    ix1 = np.minimum(ix0 + 1, Dx - 1)
    iy1 = np.minimum(iy0 + 1, Dy - 1)
    iz1 = np.minimum(iz0 + 1, Dz - 1)

    tx = (fx - ix0).astype(np.float32)
    ty = (fy - iy0).astype(np.float32)
    tz = (fz - iz0).astype(np.float32)

    scale = 1.0 / (vmax - vmin)

    def fetch_normalized(iz, iy, ix):
        v = volume[iz, iy, ix].astype(np.float32)
        return np.clip((v - vmin) * scale, 0.0, 1.0)

    c000 = fetch_normalized(iz0, iy0, ix0)
    c100 = fetch_normalized(iz0, iy0, ix1)
    c010 = fetch_normalized(iz0, iy1, ix0)
    c110 = fetch_normalized(iz0, iy1, ix1)
    c001 = fetch_normalized(iz1, iy0, ix0)
    c101 = fetch_normalized(iz1, iy0, ix1)
    c011 = fetch_normalized(iz1, iy1, ix0)
    c111 = fetch_normalized(iz1, iy1, ix1)

    c00 = c000 * (1 - tx) + c100 * tx
    c10 = c010 * (1 - tx) + c110 * tx
    c01 = c001 * (1 - tx) + c101 * tx
    c11 = c011 * (1 - tx) + c111 * tx
    c0  = c00  * (1 - ty) + c10  * ty
    c1  = c01  * (1 - ty) + c11  * ty
    return c0 * (1 - tz) + c1 * tz


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_volume(volume: np.ndarray, prefix: bytes = b"") -> str:
    """Write a raw volume (with optional file-prefix bytes) to a temp file."""
    fd, fname = tempfile.mkstemp(suffix=".raw")
    try:
        with os.fdopen(fd, "wb") as f:
            if prefix:
                f.write(prefix)
            volume.tofile(f)
    except Exception:
        os.unlink(fname)
        raise
    return fname


def _decode_host(sampler, coords_np: np.ndarray, n_channels: int = 1) -> np.ndarray:
    """Run pysampler.decode using host buffers (numpy arrays)."""
    coords = np.ascontiguousarray(coords_np, dtype=np.float32)
    values = np.zeros((n_channels, coords.shape[0]), dtype=np.float32)
    pysampler.decode(sampler, coords.ctypes.data, values.ctypes.data, coords.shape[0])
    return coords, values


def _sample_host(sampler, count: int, n_channels: int = 1) -> np.ndarray:
    """Run pysampler.sample using host buffers (numpy arrays)."""
    coords = np.zeros((count, 3), dtype=np.float32)
    values = np.zeros((n_channels, count), dtype=np.float32)
    pysampler.sample(sampler, coords.ctypes.data, values.ctypes.data, count)
    return coords, values


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("device", ["virtual_memory", "out_of_core"])
def test_decode_float_matches_reference(device):
    """decode() must match the normalize-before-interpolate reference."""
    rng = np.random.default_rng(2026)
    Dx, Dy, Dz = 24, 32, 40
    volume = rng.uniform(-1.0, 1.0, (Dz, Dy, Dx)).astype(np.float32)
    vmin, vmax = float(volume.min()) - 1e-3, float(volume.max()) + 1e-3
    fname = _write_volume(volume)
    try:
        sampler = pysampler.create_sampler(
            "structuredRegular", device,
            filename=fname, dims=[Dx, Dy, Dz], dtype="float32",
            range=[vmin, vmax],
        )
        assert sampler.n_channels() == 1

        N = 1024
        coords_np = rng.uniform(0.05, 0.95, (N, 3)).astype(np.float32)
        coords, values = _decode_host(sampler, coords_np)

        # Coords must be untouched by decode().
        np.testing.assert_array_equal(coords, coords_np)

        # Values must match the software reference within float epsilon.
        ref = _vm_reference(volume, coords_np, vmin, vmax).astype(np.float32)
        np.testing.assert_allclose(values[0], ref, atol=2e-5, rtol=0)
    finally:
        os.unlink(fname)


@pytest.mark.parametrize("device", ["virtual_memory", "out_of_core"])
def test_decode_uint16_with_explicit_range(device):
    """uint16 volumes must honor the user-provided range."""
    rng = np.random.default_rng(31337)
    Dx, Dy, Dz = 16, 16, 16
    volume = rng.integers(low=200, high=2000, size=(Dz, Dy, Dx), dtype=np.uint16)
    vmin, vmax = 200.0, 2000.0
    fname = _write_volume(volume)
    try:
        sampler = pysampler.create_sampler(
            "structuredRegular", device,
            filename=fname, dims=[Dx, Dy, Dz], dtype="uint16",
            range=[vmin, vmax],
        )
        N = 512
        coords_np = rng.uniform(0.1, 0.9, (N, 3)).astype(np.float32)
        _, values = _decode_host(sampler, coords_np)
        ref = _vm_reference(volume, coords_np, vmin, vmax).astype(np.float32)
        np.testing.assert_allclose(values[0], ref, atol=2e-5, rtol=0)
        # Sanity: values must lie in [0, 1] after normalization.
        assert values.min() >= 0.0 and values.max() <= 1.0
    finally:
        os.unlink(fname)


@pytest.mark.parametrize("device", ["virtual_memory", "out_of_core"])
def test_decode_with_file_offset(device):
    """A non-zero `offset` must skip a leading header without affecting decoded values."""
    rng = np.random.default_rng(7)
    Dx, Dy, Dz = 16, 24, 16
    volume = rng.uniform(0, 1, (Dz, Dy, Dx)).astype(np.float32)
    vmin, vmax = -1.0, 2.0  # explicit range so it does not depend on data
    header = b"\x00" * 256
    fname = _write_volume(volume, prefix=header)
    try:
        sampler = pysampler.create_sampler(
            "structuredRegular", device,
            filename=fname, dims=[Dx, Dy, Dz], dtype="float32",
            range=[vmin, vmax], offset=len(header),
        )
        N = 256
        coords_np = rng.uniform(0.05, 0.95, (N, 3)).astype(np.float32)
        _, values = _decode_host(sampler, coords_np)
        ref = _vm_reference(volume, coords_np, vmin, vmax).astype(np.float32)
        np.testing.assert_allclose(values[0], ref, atol=2e-5, rtol=0)
    finally:
        os.unlink(fname)


@pytest.mark.parametrize("device", ["virtual_memory", "out_of_core"])
def test_decode_clamps_to_unit_interval(device):
    """Voxels outside the explicit range must be clamped to [0, 1] per voxel."""
    rng = np.random.default_rng(11)
    Dx, Dy, Dz = 8, 8, 8
    volume = rng.uniform(-5.0, 5.0, (Dz, Dy, Dx)).astype(np.float32)
    # Intentionally narrow range so many voxels clamp.
    vmin, vmax = -1.0, 1.0
    fname = _write_volume(volume)
    try:
        sampler = pysampler.create_sampler(
            "structuredRegular", device,
            filename=fname, dims=[Dx, Dy, Dz], dtype="float32",
            range=[vmin, vmax],
        )
        N = 128
        coords_np = rng.uniform(0.0, 1.0, (N, 3)).astype(np.float32)
        _, values = _decode_host(sampler, coords_np)
        assert values.min() >= 0.0
        assert values.max() <= 1.0
        ref = _vm_reference(volume, coords_np, vmin, vmax).astype(np.float32)
        np.testing.assert_allclose(values[0], ref, atol=2e-5, rtol=0)
    finally:
        os.unlink(fname)


@pytest.mark.parametrize("device", ["virtual_memory", "out_of_core"])
def test_sample_smoke(device):
    """sample() must populate both coords and values, with values in [0, 1]."""
    rng = np.random.default_rng(99)
    Dx, Dy, Dz = 16, 16, 16
    volume = rng.uniform(0.0, 1.0, (Dz, Dy, Dx)).astype(np.float32)
    fname = _write_volume(volume)
    try:
        sampler = pysampler.create_sampler(
            "structuredRegular", device,
            filename=fname, dims=[Dx, Dy, Dz], dtype="float32",
            range=[0.0, 1.0],
        )
        N = 256
        coords, values = _sample_host(sampler, N)
        # Coords must lie in [0, 1]^3.
        assert coords.min() >= 0.0 and coords.max() <= 1.0, "coords out of [0, 1]^3"
        # Values must lie in [0, 1] (already pre-normalized in the sampler).
        assert values.min() >= 0.0 and values.max() <= 1.0
        # values must agree with the reference at the sampled coordinates.
        ref = _vm_reference(volume, coords, 0.0, 1.0).astype(np.float32)
        np.testing.assert_allclose(values[0], ref, atol=2e-5, rtol=0)
    finally:
        os.unlink(fname)


@pytest.mark.parametrize("device", ["virtual_memory", "out_of_core"])
def test_missing_range_raises(device):
    """File-backed samplers must raise if no range is provided."""
    rng = np.random.default_rng(0)
    Dx, Dy, Dz = 4, 4, 4
    volume = rng.uniform(0, 1, (Dz, Dy, Dx)).astype(np.float32)
    fname = _write_volume(volume)
    try:
        with pytest.raises(Exception, match="range"):
            pysampler.create_sampler(
                "structuredRegular", device,
                filename=fname, dims=[Dx, Dy, Dz], dtype="float32",
            )
    finally:
        os.unlink(fname)


@pytest.mark.parametrize("device", ["virtual_memory", "out_of_core"])
def test_big_endian_rejected(device):
    """Big-endian input is not supported by the file-backed samplers."""
    rng = np.random.default_rng(0)
    Dx, Dy, Dz = 4, 4, 4
    volume = rng.uniform(0, 1, (Dz, Dy, Dx)).astype(np.float32)
    fname = _write_volume(volume)
    try:
        with pytest.raises(Exception, match=r"big[\s-]?endian"):
            pysampler.create_sampler(
                "structuredRegular", device,
                filename=fname, dims=[Dx, Dy, Dz], dtype="float32",
                range=[0.0, 1.0], is_big_endian=True,
            )
    finally:
        os.unlink(fname)


@pytest.mark.parametrize("device", ["virtual_memory", "out_of_core"])
def test_multichannel_rejected(device):
    """File-backed samplers currently support only single-channel volumes."""
    rng = np.random.default_rng(0)
    Dx, Dy, Dz = 4, 4, 4
    volume = rng.uniform(0, 1, (Dz, Dy, Dx)).astype(np.float32)
    fname = _write_volume(volume)
    try:
        with pytest.raises(Exception, match="n_channels"):
            pysampler.create_sampler(
                "structuredRegular", device,
                filename=fname, dims=[Dx, Dy, Dz], dtype="float32",
                range=[0.0, 1.0], n_channels=2,
            )
    finally:
        os.unlink(fname)


def test_decode_consistency_between_backends():
    """`virtual_memory` and `out_of_core` must produce identical `decode()` outputs."""
    rng = np.random.default_rng(2024)
    Dx, Dy, Dz = 20, 20, 20
    volume = rng.uniform(-1.0, 1.0, (Dz, Dy, Dx)).astype(np.float32)
    vmin, vmax = float(volume.min()) - 1e-3, float(volume.max()) + 1e-3
    fname = _write_volume(volume)
    try:
        s_vm = pysampler.create_sampler(
            "structuredRegular", "virtual_memory",
            filename=fname, dims=[Dx, Dy, Dz], dtype="float32",
            range=[vmin, vmax],
        )
        s_ooc = pysampler.create_sampler(
            "structuredRegular", "out_of_core",
            filename=fname, dims=[Dx, Dy, Dz], dtype="float32",
            range=[vmin, vmax],
        )

        N = 256
        coords_np = rng.uniform(0.05, 0.95, (N, 3)).astype(np.float32)
        _, vm_values  = _decode_host(s_vm,  coords_np)
        _, ooc_values = _decode_host(s_ooc, coords_np)
        np.testing.assert_allclose(vm_values, ooc_values, atol=1e-6, rtol=0)
    finally:
        os.unlink(fname)

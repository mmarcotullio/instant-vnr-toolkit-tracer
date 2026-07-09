"""
Smoke + correctness tests for the OpenVDB backend (device='openvkl').

Unlike test_sampler.py (which exercises the CUDA backend), this file tests the
CPU-side OpenVKL/VDB code path, so we use **host-resident numpy buffers**
(passed via array.ctypes.data) rather than CUDA tensors.

Test asset
----------
The asset is downloaded on first run from the OpenVDB website's Git-LFS mirror
and cached at  $XDG_CACHE_HOME/pysampler/test/bunny_cloud.vdb  (default:
~/.cache/pysampler/test/, overridable with PYSAMPLER_TEST_CACHE).  The file
is ~73 MB (Stanford bunny as a fog volume; single FloatGrid named "density").

Notes on value ranges
---------------------
The OpenVKL backend normalises samples using (val - vmin) / (vmax - vmin),
where vmin/vmax come from the **active** voxels only
(openvdb::tools::extrema over cbeginValueOn).  For sparse fog grids like
bunny_cloud, vmin > 0 while the volume's background value is 0, so points
outside the active region normalise to a small negative number.  We therefore
only assert finiteness + variation, not strict [0, 1] containment.
"""
import os
import socket
import urllib.error
import urllib.request

import numpy as np
import pytest
import pysampler


# ---------------------------------------------------------------------------
# Asset download
# ---------------------------------------------------------------------------

# Git-LFS-resolved URL — `raw.githubusercontent.com` returns the ~130-byte
# pointer file, but `media.githubusercontent.com/media/...` proxies through
# LFS and returns the real binary.
_VDB_URL   = ("https://media.githubusercontent.com/media/"
              "AcademySoftwareFoundation/openvdb-website/master/"
              "download/models/bunny_cloud.vdb")
_VDB_NAME  = "bunny_cloud.vdb"
_VDB_FIELD = "density"
# Anything smaller than this means LFS resolution failed and we got the pointer
# file or a truncated download.
_MIN_BYTES = 1 * 1024 * 1024


def _cache_dir() -> str:
    """Return the cache directory, honouring XDG and PYSAMPLER_TEST_CACHE.

    Resolution order:
      1. $PYSAMPLER_TEST_CACHE        — explicit override
      2. $XDG_CACHE_HOME/pysampler/test
      3. ~/.cache/pysampler/test      — XDG default
    """
    explicit = os.environ.get("PYSAMPLER_TEST_CACHE")
    if explicit:
        return explicit
    xdg = os.environ.get("XDG_CACHE_HOME") or os.path.expanduser("~/.cache")
    return os.path.join(xdg, "pysampler", "test")


def _ensure_vdb_file() -> str:
    """Return a path to a cached bunny_cloud.vdb, downloading on first use.

    Raises pytest.skip if the file is unavailable (offline, LFS hiccup, etc.).
    """
    cache_dir = _cache_dir()
    os.makedirs(cache_dir, exist_ok=True)
    dest = os.path.join(cache_dir, _VDB_NAME)

    if os.path.exists(dest) and os.path.getsize(dest) >= _MIN_BYTES:
        return dest

    # Download to a temporary file and atomically rename so a failed download
    # doesn't poison the cache for the next run.
    tmp = dest + ".part"
    try:
        urllib.request.urlretrieve(_VDB_URL, tmp)
    except (urllib.error.URLError, socket.timeout, TimeoutError, OSError) as e:
        if os.path.exists(tmp):
            os.unlink(tmp)
        pytest.skip(f"could not download {_VDB_URL}: {e}")

    if os.path.getsize(tmp) < _MIN_BYTES:
        os.unlink(tmp)
        pytest.skip(
            f"download from {_VDB_URL} returned only "
            f"{os.path.getsize(tmp) if os.path.exists(tmp) else 0} bytes "
            "(likely an LFS pointer file, not the real .vdb)"
        )

    os.replace(tmp, dest)
    return dest


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def vdb_file():
    """Path to bunny_cloud.vdb in the test cache."""
    return _ensure_vdb_file()


@pytest.fixture(scope="session")
def vdb_sampler(vdb_file):
    """Cached sampler — built once per session so we only parse the .vdb once."""
    try:
        return pysampler.create_sampler(
            "openvdb", "openvkl",
            filename=vdb_file, field=_VDB_FIELD,
        )
    except RuntimeError as e:
        # Raised by sampler_openvkl.cpp when the build was configured without
        # ENABLE_OPENVDB.  Skip cleanly so the rest of the suite still runs.
        pytest.skip(f"OpenVDB sampler unavailable: {e}")


def _empty_buffers(n: int, n_channels: int = 1):
    """Allocate (coords, values) numpy arrays in the layout pysampler expects."""
    coords = np.zeros((n, 3),         dtype=np.float32)
    # values are written as (n_channels, count) row-major — see py_sample()
    values = np.zeros((n_channels, n), dtype=np.float32)
    return coords, values


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_openvdb_open_and_n_channels(vdb_sampler):
    """create_sampler() succeeds on a real .vdb and reports a single channel."""
    assert vdb_sampler.n_channels() == 1


def test_openvdb_decode_finite_at_random_points(vdb_sampler):
    """decode() at uniformly random points in [0, 1]^3 produces finite floats."""
    rng = np.random.default_rng(0)
    N = 4096

    coords, values = _empty_buffers(N)
    coords[:] = rng.uniform(0.0, 1.0, (N, 3)).astype(np.float32)

    pysampler.decode(vdb_sampler, coords.ctypes.data, values.ctypes.data, N)

    assert np.isfinite(values).all(), "decode produced NaN / inf"


def test_openvdb_decode_is_deterministic(vdb_sampler):
    """The same coords sampled twice must yield byte-identical values."""
    rng = np.random.default_rng(42)
    N = 1024

    coords, v1 = _empty_buffers(N)
    coords[:] = rng.uniform(0.05, 0.95, (N, 3)).astype(np.float32)

    pysampler.decode(vdb_sampler, coords.ctypes.data, v1.ctypes.data, N)

    _, v2 = _empty_buffers(N)
    pysampler.decode(vdb_sampler, coords.ctypes.data, v2.ctypes.data, N)

    np.testing.assert_array_equal(v1, v2)


def test_openvdb_decode_hits_active_voxels(vdb_sampler):
    """A dense uniform sweep must hit some active (positive-density) voxels."""
    rng = np.random.default_rng(123)
    N = 32_768

    coords, values = _empty_buffers(N)
    coords[:] = rng.uniform(0.0, 1.0, (N, 3)).astype(np.float32)

    pysampler.decode(vdb_sampler, coords.ctypes.data, values.ctypes.data, N)

    # bunny_cloud is a sparse fog grid — most random points land outside the
    # bunny.  Inside, normalised density is non-trivial (> 1e-3 say).  If we
    # never see anything above that, the sampler is silently returning the
    # background value everywhere, which would mean we're not actually reading
    # the VDB grid.
    n_inside = int((values > 1e-3).sum())
    assert n_inside > 0, "no active voxels hit — VDB data not being sampled"

    # Variation across the volume is also a quick sanity check.
    assert values.std() > 0.0, "all samples returned the same value"


def test_openvdb_sample_generates_coords_in_unit_cube(vdb_sampler):
    """sample() must populate coords with random points in [0, 1]^3 + values."""
    N = 4096

    coords, values = _empty_buffers(N)
    # Seed buffers with a sentinel so we can confirm sample() actually wrote.
    coords.fill(-99.0)
    values.fill(-99.0)

    pysampler.sample(vdb_sampler, coords.ctypes.data, values.ctypes.data, N)

    assert (coords >= 0.0).all() and (coords <= 1.0).all(), \
        "sample() produced coords outside [0, 1]^3"
    assert np.isfinite(values).all(), "sample() produced NaN / inf"
    assert (values != -99.0).any(), "sample() did not write to the values buffer"

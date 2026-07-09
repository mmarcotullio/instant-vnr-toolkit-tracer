"""Verify that all shared libraries in the installed instantvnr package can
resolve their runtime dependencies.

This catches problems like:
- Absolute build-cache paths baked into DT_NEEDED (the uv ephemeral dir bug)
- Missing libraries that should have been co-installed (e.g. libtinycudann)
- Broken LD_LIBRARY_PATH setup in _runtime_env()
"""

from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _so_dir() -> Path:
    """Return the installed package directory containing .so files."""
    import instantvnr
    return Path(instantvnr.__file__).parent


def _runtime_env() -> dict[str, str]:
    """Replicate the LD_LIBRARY_PATH that apps.py sets for subprocesses."""
    from instantvnr.apps import _runtime_env as _env
    return _env()


def _collect_shared_objects() -> list[Path]:
    """Return all .so files and ELF executables in the package directory."""
    pkg = _so_dir()
    results = []
    for p in sorted(pkg.iterdir()):
        if p.suffix == ".so" or (p.is_file() and os.access(p, os.X_OK) and not p.suffix):
            results.append(p)
    return results


_NOT_FOUND_RE = re.compile(r"^\s*(\S+)\s+=>\s+not found", re.MULTILINE)


def _ldd_missing(path: Path, env: dict[str, str]) -> list[str]:
    """Run ldd on a binary and return a list of library names marked 'not found'."""
    try:
        result = subprocess.run(
            ["ldd", str(path)],
            capture_output=True, text=True, timeout=15,
            env=env,
        )
    except FileNotFoundError:
        pytest.skip("ldd not available")
    return _NOT_FOUND_RE.findall(result.stdout + result.stderr)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestRuntimeLibs:
    """Check that every shared library / executable resolves all NEEDED deps."""

    @pytest.fixture(scope="class")
    def env(self) -> dict[str, str]:
        return _runtime_env()

    @pytest.fixture(scope="class")
    def shared_objects(self) -> list[Path]:
        libs = _collect_shared_objects()
        if not libs:
            pytest.skip("no shared libraries found in installed package")
        return libs

    def test_package_has_shared_objects(self, shared_objects: list[Path]):
        assert len(shared_objects) > 0

    def test_libtinycudann_exists(self):
        matches = list(_so_dir().glob("libtinycudann*"))
        assert matches, (
            f"libtinycudann*.so not found in {_so_dir()}; "
            "FindTCNN.cmake should copy it into CMAKE_LIBRARY_OUTPUT_DIRECTORY"
        )

    def test_no_ephemeral_paths_in_needed(self, shared_objects: list[Path]):
        """Ensure no DT_NEEDED entry contains a build-cache or temp path."""
        ephemeral_patterns = [".cache/uv/", ".tmp", "/tmp/"]
        for so in shared_objects:
            try:
                result = subprocess.run(
                    ["readelf", "-d", str(so)],
                    capture_output=True, text=True, timeout=15,
                )
            except FileNotFoundError:
                pytest.skip("readelf not available")
            for line in result.stdout.splitlines():
                if "(NEEDED)" not in line:
                    continue
                for pat in ephemeral_patterns:
                    assert pat not in line, (
                        f"{so.name} has ephemeral path in DT_NEEDED: {line.strip()}"
                    )

    def test_all_libs_resolve(self, shared_objects: list[Path], env: dict[str, str]):
        """ldd should report zero 'not found' entries for every .so / executable."""
        failures: list[str] = []
        for so in shared_objects:
            missing = _ldd_missing(so, env)
            if missing:
                failures.append(f"{so.name}: {missing}")
        assert not failures, "Unresolved shared libraries:\n" + "\n".join(failures)

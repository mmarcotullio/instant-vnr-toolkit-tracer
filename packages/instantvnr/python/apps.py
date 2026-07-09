"""
Python launchers for the three compiled VNR executables:

  vnr_batch      -- headless batch renderer (no display required)
  vnr_int_single -- interactive single-volume viewer  (requires OpenGL)
  vnr_int_dual   -- interactive dual-volume viewer with live training (requires OpenGL)

Each function builds the argv list and exec()s the binary.  When used as a
console script the function is called with sys.argv; when imported the caller
can pass kwargs directly.

Executable search order:
  1. <this package's directory>   (co-installed by CMake)
  2. PATH
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Optional, Sequence


def _runtime_env() -> dict[str, str]:
    """Return a copy of os.environ with LD_LIBRARY_PATH augmented so that
    standalone executables can find libtorch, libpython, and our own .so files."""
    env = os.environ.copy()
    extra_dirs: list[str] = []

    # libpython (libpython3.XX.so)
    import sysconfig
    _libdir = sysconfig.get_config_var("LIBDIR")
    if _libdir:
        extra_dirs.append(_libdir)

    # torch libs (libtorch.so, libc10.so, …)
    try:
        import torch
        extra_dirs.append(os.path.join(os.path.dirname(torch.__file__), "lib"))
    except ImportError:
        pass

    # co-installed shared libs (libivnr.so, libtinycudann.so, …)
    extra_dirs.append(str(Path(__file__).parent))

    existing = env.get("LD_LIBRARY_PATH", "")
    env["LD_LIBRARY_PATH"] = ":".join(extra_dirs + ([existing] if existing else []))
    return env


def _find_exe(name: str) -> str:
    here = Path(__file__).parent
    candidate = here / name
    if candidate.exists():
        return str(candidate)
    found = shutil.which(name)
    if found:
        return found
    raise FileNotFoundError(
        f"{name!r} not found in {here} or PATH. "
        "Build the package first: cd packages/instantvnr && ./setup_venv.sh"
    )


def _parse_vec3(values) -> tuple[float, float, float]:
    """Parse user input into a float vec3.

    Accepted forms:
    - comma-separated string: "x,y,z" (also supports "(x,y,z)" / "[x,y,z]")
    - Python sequence of 3 numbers: [x, y, z], (x, y, z)
    - numpy array (shape (3,) or any shape with 3 elements)
    - torch tensor (CPU/GPU, shape (3,) or any shape with 3 elements)
    """
    # String path: "x,y,z" (plus optional wrappers)
    if isinstance(values, str):
        text = values.strip()
        if text and text[0] in "([{" and text[-1] in ")]}":
            text = text[1:-1].strip()
        parts = [p.strip() for p in text.split(",")] if "," in text else text.split()
        parts = [p for p in parts if p]
        if len(parts) != 3:
            raise ValueError(f"expected vec3 string with 3 values, got {values!r}")
        return float(parts[0]), float(parts[1]), float(parts[2])

    # Tensor/ndarray-like path.
    if hasattr(values, "reshape") and hasattr(values, "tolist"):
        array_like = values
        if hasattr(array_like, "detach"):
            array_like = array_like.detach()
        if hasattr(array_like, "cpu"):
            array_like = array_like.cpu()
        flattened = array_like.reshape(-1).tolist()
        if len(flattened) != 3:
            raise ValueError(f"expected array/tensor with 3 elements, got {values!r}")
        return float(flattened[0]), float(flattened[1]), float(flattened[2])

    # Generic sequence path
    if isinstance(values, Sequence) and not isinstance(values, (str, bytes)):
        if len(values) != 3:
            raise ValueError(f"expected 3 floats, got {values!r}")
        return float(values[0]), float(values[1]), float(values[2])

    raise ValueError(
        "expected vec3 input as comma string, 3-item sequence, numpy array, or torch tensor; "
        f"got {type(values).__name__}: {values!r}"
    )


def _vec3_as_triplet_args(values) -> list[str]:
    """Encode vec3 as three CLI arguments: ['x', 'y', 'z']."""
    x, y, z = _parse_vec3(values)
    return [str(x), str(y), str(z)]


def _vec3_as_single_arg(values) -> list[str]:
    """Encode vec3 as one CLI argument: ['x,y,z'].

    Needed by apps that use args.hxx ValueFlag<vec3f, Vec3fReader>.
    """
    x, y, z = _parse_vec3(values)
    return [f"{x},{y},{z}"]


# ---------------------------------------------------------------------------
# vnr_batch
# ---------------------------------------------------------------------------

def run_batch(
    *,
    simple_volume: Optional[str] = None,
    neural_volume: Optional[str] = None,
    output: str = "output.png",
    width: int = 1024,
    height: int = 1024,
    spp: int = 1,
    warmup: int = 5,
    rendering_mode: int = 0,
    sampling_rate: float = 1.0,
    density_scale: float = 1.0,
    camera_from: Optional[Sequence[float]] = None,
    camera_at: Optional[Sequence[float]] = None,
    camera_up: Optional[Sequence[float]] = None,
    tfn: Optional[str] = None,
    scalarfield: Optional[str] = None,
    shadowfield: Optional[str] = None,
    photonfield: Optional[str] = None,
    nerf: Optional[str] = None,
    param_time: Optional[float] = None,
    param_theta: Optional[float] = None,
    param_phi: Optional[float] = None,
    denoise: bool = False,
    extra_args: Sequence[str] = (),
) -> int:
    """Run vnr_batch (headless renderer). Exactly one of simple_volume / neural_volume required."""
    if (simple_volume is None) == (neural_volume is None):
        raise ValueError("specify exactly one of simple_volume or neural_volume")

    cmd = [_find_exe("vnr_batch")]

    if simple_volume is not None:
        cmd += ["--simple-volume", simple_volume]
    else:
        cmd += ["--neural-volume", neural_volume]

    cmd += ["--output", output]
    cmd += ["--width", str(width), "--height", str(height)]
    cmd += ["--spp", str(spp), "--warmup", str(warmup)]
    cmd += ["--rendering-mode", str(rendering_mode)]
    cmd += ["--sampling-rate", str(sampling_rate)]
    cmd += ["--density-scale", str(density_scale)]

    if camera_from is not None:
        cmd += ["--camera-from"] + _vec3_as_single_arg(camera_from)
    if camera_at is not None:
        cmd += ["--camera-at"] + _vec3_as_single_arg(camera_at)
    if camera_up is not None:
        cmd += ["--camera-up"] + _vec3_as_single_arg(camera_up)

    if tfn is not None:
        cmd += ["--tfn", tfn]
    if scalarfield is not None:
        cmd += ["--scalarfield", scalarfield]
    if shadowfield is not None:
        cmd += ["--shadowfield", shadowfield]
    if photonfield is not None:
        cmd += ["--photonfield", photonfield]
    if nerf is not None:
        cmd += ["--nerf", nerf]

    if param_time is not None:
        cmd += ["--param-time", str(param_time)]
    if param_theta is not None:
        cmd += ["--param-theta", str(param_theta)]
    if param_phi is not None:
        cmd += ["--param-phi", str(param_phi)]

    if denoise:
        cmd += ["--denoise"]

    cmd += list(extra_args)

    return subprocess.call(cmd, env=_runtime_env())


def _batch_main() -> None:
    sys.exit(subprocess.call([_find_exe("vnr_batch")] + sys.argv[1:], env=_runtime_env()))


# ---------------------------------------------------------------------------
# vnr_int_single
# ---------------------------------------------------------------------------

def run_int_single(
    *,
    simple_volume: Optional[str] = None,
    neural_volume: Optional[str] = None,
    rendering_mode: int = 0,
    sampling_rate: float = 1.0,
    density_scale: float = 1.0,
    camera_from: Optional[Sequence[float]] = None,
    camera_at: Optional[Sequence[float]] = None,
    camera_up: Optional[Sequence[float]] = None,
    force_camera: bool = False,
    tfn: Optional[str] = None,
    scalarfield: Optional[str] = None,
    shadowfield: Optional[str] = None,
    photonfield: Optional[str] = None,
    nerf: Optional[str] = None,
    max_num_frames: int = 0,
    report_rendering_fps: bool = False,
    extra_args: Sequence[str] = (),
) -> int:
    """Run vnr_int_single (interactive GL viewer). Exactly one of simple_volume / neural_volume required."""
    if (simple_volume is None) == (neural_volume is None):
        raise ValueError("specify exactly one of simple_volume or neural_volume")

    cmd = [_find_exe("vnr_int_single")]

    if simple_volume is not None:
        cmd += ["--simple-volume", simple_volume]
    else:
        cmd += ["--neural-volume", neural_volume]

    cmd += ["--rendering-mode", str(rendering_mode)]
    cmd += ["--sampling-rate", str(sampling_rate)]
    cmd += ["--density-scale", str(density_scale)]

    if camera_from is not None:
        cmd += ["--camera-from"] + _vec3_as_single_arg(camera_from)
    if camera_at is not None:
        cmd += ["--camera-at"] + _vec3_as_single_arg(camera_at)
    if camera_up is not None:
        cmd += ["--camera-up"] + _vec3_as_single_arg(camera_up)

    if force_camera:
        cmd += ["--force-camera"]
    if tfn is not None:
        cmd += ["--tfn", tfn]
    if scalarfield is not None:
        cmd += ["--scalarfield", scalarfield]
    if shadowfield is not None:
        cmd += ["--shadowfield", shadowfield]
    if photonfield is not None:
        cmd += ["--photonfield", photonfield]
    if nerf is not None:
        cmd += ["--nerf", nerf]
    if max_num_frames > 0:
        cmd += ["--max-num-frames", str(max_num_frames)]
    if report_rendering_fps:
        cmd += ["--report-rendering-fps"]

    cmd += list(extra_args)

    return subprocess.call(cmd, env=_runtime_env())


def _int_single_main() -> None:
    sys.exit(subprocess.call([_find_exe("vnr_int_single")] + sys.argv[1:], env=_runtime_env()))


# ---------------------------------------------------------------------------
# vnr_int_dual
# ---------------------------------------------------------------------------

def run_int_dual(
    *,
    volume: str,
    network: str,
    resume: Optional[str] = None,
    rendering_mode: int = 0,
    training_mode: str = "GPU",
    sampling_rate: float = 1.0,
    density_scale: float = 1.0,
    camera_from: Optional[Sequence[float]] = None,
    camera_at: Optional[Sequence[float]] = None,
    camera_up: Optional[Sequence[float]] = None,
    max_frames: int = 0,
    max_steps: int = 0,
    groundtruth_macrocell: bool = False,
    save_reference_volume: bool = False,
    pause_training: bool = False,
    pause_reference: bool = False,
    pause_inference: bool = False,
    quiet: bool = False,
    summary: bool = False,
    report_macrocell_quality: bool = False,
    report_rendering_fps: bool = False,
    report: Optional[str] = None,
    extra_args: Sequence[str] = (),
) -> int:
    """Run vnr_int_dual (interactive dual-view: reference + live INR training)."""
    cmd = [_find_exe("vnr_int_dual")]
    cmd += ["--volume", volume, "--network", network]

    if resume is not None:
        cmd += ["--resume", resume]

    cmd += ["--rendering-mode", str(rendering_mode)]
    cmd += ["--training-mode", training_mode]
    cmd += ["--sampling-rate", str(sampling_rate)]
    cmd += ["--density-scale", str(density_scale)]

    if camera_from is not None:
        cmd += ["--camera-from"] + _vec3_as_triplet_args(camera_from)
    if camera_at is not None:
        cmd += ["--camera-at"] + _vec3_as_triplet_args(camera_at)
    if camera_up is not None:
        cmd += ["--camera-up"] + _vec3_as_triplet_args(camera_up)

    if max_frames > 0:
        cmd += ["--max-frames", str(max_frames)]
    if max_steps > 0:
        cmd += ["--max-steps", str(max_steps)]

    if groundtruth_macrocell:
        cmd += ["--groundtruth-macrocell"]
    if save_reference_volume:
        cmd += ["--save-reference-volume"]
    if pause_training:
        cmd += ["--pause-training"]
    if pause_reference:
        cmd += ["--pause-reference"]
    if pause_inference:
        cmd += ["--pause-inference"]
    if quiet:
        cmd += ["--quiet"]
    if summary:
        cmd += ["--summary"]
    if report_macrocell_quality:
        cmd += ["--report-macrocell-quality"]
    if report_rendering_fps:
        cmd += ["--report-rendering-fps"]
    if report is not None:
        cmd += ["--report", report]

    cmd += list(extra_args)

    return subprocess.call(cmd, env=_runtime_env())


def _int_dual_main() -> None:
    sys.exit(subprocess.call([_find_exe("vnr_int_dual")] + sys.argv[1:], env=_runtime_env()))

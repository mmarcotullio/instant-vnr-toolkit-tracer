# Instant VNR Toolkit

**Instant Volumetric Neural Representation** — compress a 3-D scientific volume into a tiny neural
network, then reconstruct and render it at interactive frame rates without keeping the original data
in memory.

> **Paper:** [Interactive Volume Visualization Via Multi-Resolution Hash Encoding Based Neural
> Representation](https://wilsoncernwq.github.io/publication/tvcg2022-instant-vnr)  
> *Qi Wu, David Bauer, Michael J. Doyle, Kwan-Liu Ma*  
> IEEE Transactions on Visualization and Computer Graphics (TVCG), 2023  
> [arXiv:2207.11620](https://arxiv.org/abs/2207.11620) · DOI: [10.1109/TVCG.2023.3293121](https://doi.org/10.1109/TVCG.2023.3293121)

---

## Table of Contents

- [Background: What problem does this solve?](#background-what-problem-does-this-solve)
- [How it works](#how-it-works)
- [Repository layout](#repository-layout)
- [Requirements](#requirements)
- [Installation](#installation)
- [Quick start](#quick-start)
- [Training in depth](#training-in-depth)
- [Rendering in depth](#rendering-in-depth)
- [Interactive applications](#interactive-applications)
- [Python API](#python-api)
- [Scene file format (.bson)](#scene-file-format-bson)
- [Citation](#citation)

---

## Background: What problem does this solve?

### What is a scientific volume?

A **volume** in scientific computing is a 3-D grid of scalar (or vector) values.  Think of a CT
scan, a fluid-simulation output, or a weather model: every voxel holds a measurement such as
density, temperature, or pressure.  A 512³ float32 volume already takes **512 MB**; petascale
simulations produce datasets measured in **terabytes**.

Storing, transferring, and rendering these volumes directly is expensive.  Traditional compression
(gzip, JPEG 2000, etc.) trades reconstruction quality for file size but still requires decoding the
full grid before any rendering can happen.

### What is an Implicit Neural Representation (INR)?

An **Implicit Neural Representation** replaces the voxel grid with a neural network **f** that
maps a 3-D coordinate directly to the scalar value at that point:

```
f : (x, y, z) ──► scalar value
```

The network *implicitly* encodes the volume in its weights.  Because modern neural networks are
very compact, you can store an entire volume in a few hundred kilobytes — 10–1000× smaller than
the raw data — and query any point on demand at inference time.

### The challenge: speed

Classical INRs (e.g. NeRF-style MLPs) are too slow for interactive rendering.  A naive MLP must
evaluate thousands of layers for every voxel sample, making real-time frame rates impractical.

---

## How it works

This toolkit implements the method from our TVCG 2023 paper.  The key insight is to combine two
ideas:

### 1. Multi-Resolution Hash Grid Encoding

Instead of sending raw (x, y, z) coordinates into an MLP, we first look them up in a set of
**multi-resolution hash tables** (borrowed from [Instant-NGP](https://github.com/NVlabs/instant-ngp)).

```
coordinate (x, y, z)
        │
        ▼  hash lookup at L resolution levels
  ┌─────────────────────────────────────┐
  │  level 1 (coarse)  │ 2-D features   │
  │  level 2           │ 2-D features   │
  │  …                 │ …              │
  │  level L (fine)    │ 2-D features   │
  └─────────────────────────────────────┘
        │
        ▼  concatenate → feature vector of length  L × F
```

Each level has its own hash table of size 2^`log2_hashmap_size`.  At a given level, the 8 grid
corners surrounding (x, y, z) are looked up in the table and trilinearly interpolated to produce a
small feature vector of `F` numbers.  Features from all levels are concatenated, giving the MLP a
rich, multi-scale description of the local neighbourhood.

Because the hash tables are dense and aligned to GPU cache lines, lookups are extremely fast.

### 2. Fully-Fused MLP (tiny-cuda-nn)

The concatenated feature vector feeds into a small, **fully-fused MLP** from the
[tiny-cuda-nn](https://github.com/NVlabs/tiny-cuda-nn) library.  "Fully fused" means the entire
forward (and backward) pass fits inside shared memory on a single CUDA thread block, eliminating
global-memory traffic between layers and achieving throughputs orders of magnitude higher than a
standard PyTorch MLP.

### End-to-end pipeline

```
┌─────────────────────────────────────────────────────────────────────┐
│  TRAINING                                                           │
│                                                                     │
│  Volume file ──► pysampler ──► random (coord, value) batches        │
│                                    │                                │
│                              [Hash grid] ──► [Fused MLP] ──► pred   │
│                                                      │              │
│                                               L1+L2 loss ◄── value  │
│                                                      │              │
│                                              Adam optimizer         │
│                                                      │              │
│                                        .pt weights + .bson scene    │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│  RENDERING                                                          │
│                                                                     │
│  .bson scene ──► load model config + fp16 params + macrocell        │
│                                    │                                │
│           for each sample point (x,y,z):                            │
│                [Hash grid] ──► [Fused MLP] ──► scalar value         │
│                                                       │             │
│                              ray marching / path tracing ──► PNG    │
└─────────────────────────────────────────────────────────────────────┘
```

The **macrocell acceleration structure** divides the volume into a coarse grid of bounding boxes.
Each cell records the min/max value range of the INR in that region.  During ray marching, cells
whose value range falls entirely below the opacity transfer function can be skipped, dramatically
reducing the number of network evaluations per frame.

---

## Repository layout

```
instant-vnr-toolkit/
├── train.py                 Entry point: train an INR from raw/OpenVDB volume files
├── train_stitch.py          Entry point: train an INR from ExaStitch volume files
├── render.py                Entry point: render slice images from a .bson scene
├── pyproject.toml           Root package (inrtoolkit + console scripts)
│
├── inrtoolkit/              Pure-Python library — no C++ required
│   ├── networks.py          INR_TCNN factory (wraps tiny-cuda-nn)
│   ├── sampler.py           Python shim over pysampler (or numpy fallback)
│   ├── training.py          autocast / GradScaler helpers
│   └── utils.py             BSON export, PSNR metric, macrocell builder
│
├── packages/
│   ├── sampler/             pysampler — GPU volume sampler
│   │   ├── csrc/            CUDA kernels (structuredRegular, OpenVDB, ExaBrick/ExaStitch)
│   │   └── setup_venv.sh    One-shot build script
│   │
│   └── instantvnr/          C++ renderer and pybind11 bindings
│       ├── ext/core/        TcnnNetwork, SirenNetwork, volume management
│       ├── ext/device/      Ray marching, path tracing, shadow mapping
│       ├── ext/apps/        vnr_batch, vnr_int_single, vnr_int_dual
│       ├── ovr/             Scene graph, image I/O (librendercommon)
│       ├── python/          instantvnr Python package (__init__.py, apps.py)
│       └── setup_venv.sh    One-shot build script
│
└── references/
    ├── instantvnr/          Original C++ implementation from the paper
    └── inr-research/        PyTorch research prototype
```

---

## Requirements

| Component | Minimum | Notes |
|-----------|---------|-------|
| NVIDIA GPU | Compute capability ≥ 7.0 (Volta) | sm_70 = V100, sm_75 = RTX 2000, sm_86 = RTX 3000 |
| CUDA Toolkit | 11.8 | 12.x recommended; 12.8 required for Blackwell (sm_120) |
| Python | 3.11 | Managed by uv inside each venv |
| CMake | 3.24+ | Required for FetchContent and CUDA language support |
| GCC / G++ | 11+ | Must be compatible with your CUDA toolkit version |
| ISPC | 1.30.0 | Installed automatically by `setup_venv.sh` |

> **Tip for new users:** Run `nvidia-smi` to find your GPU model, then look up its compute
> capability at [developer.nvidia.com/cuda-gpus](https://developer.nvidia.com/cuda-gpus).

---

## Installation

Most users only need the repo-root install commands below. Choose the smallest option that
matches what you want to do.

### Recommended: install from the repo root

```bash
# From the repo root
uv venv .venv --python 3.11
source .venv/bin/activate

# Choose one:
uv pip install -e ".[sampler]"   # training
# or
uv pip install -e ".[gpu]"       # training + C++ 3-D rendering
```

`.[gpu]` includes everything in `.[sampler]`. The first GPU build usually takes **5–20 minutes**.

If you want a one-command full-stack bootstrap, `./setup_venv.sh` creates `.venv`, selects a
matching PyTorch build, and installs `.[gpu]` for you.

### Minimal install (pure Python only)

```bash
# From the repo root
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

This installs `inrtoolkit` plus the `vnr-train` / `vnr-train-file-backed` /
`vnr-train-stitch` / `vnr-render` entry points, but it does **not**
build `pysampler` or the C++ renderer. Use this option if you only need the Python library or
2-D slice rendering from an existing `.bson` scene. Training on raw volumes still requires
`.[sampler]`.

### Manual / advanced builds

Use the package-level build scripts only if you want separate virtual environments for the C++
packages or want their helper scripts to configure the build for you.

```bash
# GPU volume sampler (required for training)
cd packages/sampler
./setup_venv.sh
source .venv/bin/activate
pip install -e ../../
```

```bash
# C++ renderer and viewers (needed for vnr-batch / vnr-int-*)
cd packages/instantvnr
SM=86 ./setup_venv.sh    # optional; omit SM to auto-detect
source .venv/bin/activate
pip install -e ../../
```

These scripts auto-detect GPU/CUDA settings when possible; the sampler build also installs ISPC
automatically. Use them when you are working directly on `packages/sampler` or
`packages/instantvnr`, or when you prefer to keep those builds isolated from the root
environment.

---

## Quick start

### Train on a structured raw volume

```bash
# From an environment with `.[sampler]` or `.[gpu]` installed
vnr-train \
    --filename /path/to/volume.raw \
    --dims 256 256 256 \
    --dtype float32 \
    --expname my_volume
```

### Train on an OpenVDB volume

```bash
# OpenVDB support is enabled when `packages/sampler` is built with ENABLE_OPENVDB=ON (default)
vnr-train \
    --volume-type openvdb \
    --filename /path/to/volume.vdb \
    --field density \
    --dims 256 256 256 \
    --expname my_vdb
```

For `--volume-type openvdb`, `--dims` controls training schedule and exported scene dimensions;
the OpenVDB grid is loaded from `--filename` + `--field`, and `--dtype` is ignored.
`vnr-train` always samples through `inrtoolkit.sample_volume(...)` and moves batches to the
model device automatically, so no OpenVDB-specific host-buffer code is needed in user scripts.

### Train on an ExaStitch volume

```bash
# Requires packages/sampler to be built with ENABLE_WITCHER=ON
vnr-train-stitch \
    --umesh /path/to/dataset.umesh \
    --grids /path/to/dataset.grids \
    --scalar /path/to/dataset.scalar \
    --dims 512 512 512 \
    --expname my_stitch_volume
```

For ExaStitch, `--dims` controls the training schedule and exported scene dimensions. The sampler
loads the stitch dataset from `--umesh` / `--grids` plus `--scalar`; at least one of `--umesh` or
`--grids` must be provided.

After training you will find:

| File | Contents |
|------|----------|
| `outputs/my_volume.pt` | PyTorch state dict with fp16 network parameters |
| `outputs/my_volume.bson` | Self-contained scene file (model config + params + macrocell) |
| `logs/my_volume/run00000/` | TensorBoard event files |

Unless you override `--output-dir`, use `outputs/my_volume.bson` as the scene path
for render/view commands below.

### Render 2-D slice images

```bash
# Works after any install option above
vnr-render --scene my_volume.bson --output renders/

# Extra options:
vnr-render --scene my_volume.bson \
           --axis z \
           --n-slices 9 \
           --colormap inferno \
           --width 512 --height 512 \
           --output renders/
```

Output: one PNG per slice position, e.g. `renders/z_slice_0p50.png`.

### Render headless (3-D ray marching; requires `.[gpu]` or the manual `packages/instantvnr` build)

```bash
# From an environment with `.[gpu]` installed
vnr-batch \
    --neural-volume my_volume.bson \
    --rendering-mode 1 \
    --spp 16 \
    --output render_3d.png
```

### View interactively (requires `.[gpu]` plus OpenGL)

```bash
# Single-volume interactive viewer
vnr-int-single --neural-volume my_volume.bson --rendering-mode 1

# Dual-view: reference volume on the left, live INR on the right
vnr-int-dual --volume scene.json --network example-model.json
```

---

## Training in depth

### Network architecture

`INR_TCNN` instantiates a `tcnn.NetworkWithInputEncoding` — a single object that fuses the hash
grid encoder and the MLP into one CUDA kernel.  The default hyperparameters in `train.py` are:

| Parameter | Default | Meaning |
|-----------|---------|---------|
| `n_levels` | 16 | Number of hash grid resolution levels |
| `n_features_per_level` | 8 | Feature vector width per level |
| `log2_hashmap_size` | 19 | Each table has 2¹⁹ = 524 288 entries |
| `base_resolution` | 4 | Coarsest grid has 4³ cells |
| `per_level_scale` | 1.5 | Each level is 1.5× finer than the previous |
| `n_hidden_layers` | 4 | MLP depth |
| `n_neurons` | 16 | MLP width (neurons per hidden layer) |

The combined model size is approximately **2–4 MB** for these defaults.  Increasing
`log2_hashmap_size` or `n_levels` raises quality at the cost of more memory.

### Loss function

Training minimises a blend of L1 and L2 loss:

```
loss = 0.5 × L1(pred, target) + 0.5 × L2(pred, target)
```

L2 penalises large errors strongly; L1 is more robust to outlier voxel values.  The blend
coefficient `alpha = 0.5` is hard-coded in `train.py`.

### Optimizer

Adam with learning rate **1e-2**, halved after `total_steps / 2` steps via a StepLR scheduler.
Mixed precision (fp16 forward pass, fp32 accumulation) is enabled via `torch.cuda.amp.autocast`.

### Training epochs and batch size

The number of training steps scales automatically with the volume size:

```
total_steps = ceil(num_voxels / 65536) × 64_epochs
```

A 256³ volume (~16 M voxels) runs for roughly **16 000 steps**.  Each step samples
**65 536 random coordinate–value pairs** from the volume using `pysampler`.

### Monitoring training with TensorBoard

```bash
tensorboard --logdir logs/
```

Metrics logged every step: `train/loss`, `train/L1`, `train/L2`, `train/PSNR`,
`train/learning_rate`.

---

## Rendering in depth

### 2-D slice rendering (`render.py`)

`render.py` generates orthographic cross-sections of the volume by evaluating the INR on a 2-D
grid of coordinates and saving the result as a coloured PNG.  It uses `matplotlib` for
colourmapping and does not require any C++ libraries — the PyTorch fallback model is sufficient.

### 3-D volumetric rendering (`vnr_batch`, `vnr_int_single`, `vnr_int_dual`)

The C++ renderer implements several ray-casting algorithms.  Select one with `--rendering-mode`:

| Mode | Name | Description |
|------|------|-------------|
| 0 | Ray Marching (No Shading) | Simple emission–absorption, no lighting |
| 1 | Ray Marching (Gradient Shading) | Phong shading from finite-difference gradients |
| 2 | Ray Marching (Single-Shot Heuristic) | Gradient shading with a single sample per ray |
| 3 | Path Tracing | Global illumination via Monte Carlo sampling |
| 4 | Shadow Map (Ground Truth) | Shadow mapping from the original volume |
| 5 | Shadow Map (NN) | Shadow mapping using a learned shadow field |
| 6 | Shadow Map (Dual NN) | Dual-field shadow mapping |

Modes 5 and 6 require a separately trained shadow-field network passed via `--shadowfield`.

### Macrocell acceleration

When a `.bson` scene is loaded, the macrocell grid is decoded and uploaded to the GPU.  Before
evaluating the INR along a ray, each grid cell is tested: if the cell's value range lies entirely
below the current opacity threshold, the entire cell is skipped.  This typically eliminates 80–95%
of network evaluations in practice.

---

## Interactive applications

### `vnr_int_single` — single-volume viewer

Opens an OpenGL window showing the current volume.  Controls:

- **Left-drag** — orbit camera
- **Right-drag / scroll** — zoom
- **ImGui panel** — change rendering mode, sampling rate, density scale, transfer function, HINR
  time/theta/phi parameters, enable denoiser
- **Save Screen** — export current frame as PNG

### `vnr_int_dual` — dual-view training monitor

The left half shows the reference (ground-truth) volume; the right half shows the live INR being
trained in a background thread.  Useful for watching the representation converge in real time.

Additional flags for `vnr_int_dual`:

| Flag | Description |
|------|-------------|
| `--network` | JSON network config (see `ext/apps/example-model.json`) |
| `--resume` | Path to a checkpoint `.json` to continue training from |
| `--max-steps` | Stop training after N steps |
| `--pause-training` | Start with training paused |
| `--quiet` | Suppress per-step log output |
| `--report` | Write timing/quality CSV to this file |

Example:

```bash
vnr-int-dual \
    --volume data/configs/scene_vorts1.json \
    --network packages/instantvnr/ext/apps/example-model.json \
    --rendering-mode 5
```

---

## Python API

Import `inrtoolkit` for the core building blocks:

```python
from inrtoolkit import INR_TCNN, create_inr_scene, default_device
from inrtoolkit.sampler import create_sampler, sample

# Create and train a model
device = default_device()            # "cuda" if available
model = INR_TCNN(n_levels=16, n_features_per_level=8, n_hidden_layers=4, n_neurons=16)
model.to(device)

sampler = create_sampler(
    "structuredRegular", str(device),
    dims=[256, 256, 256], dtype="float32",
    filename="/path/to/volume.raw", n_channels=1,
)

# Sample a batch
coords, values = sample(sampler, count=65536)
# coords: (65536, 3) float32 backend-native tensor in [0, 1]³
# values: (65536, 1) float32 backend-native tensor in [0, 1]
#         - CUDA sampler   -> CUDA tensors
#         - OpenVKL sampler -> CPU tensors
# Optional: request tensors on a specific torch device.
# Moves only when backend-native device differs.
coords, values = sample(sampler, count=65536, target_device=device)

# Export scene
create_inr_scene("output.bson", [256, 256, 256], model)
```

### C++ app launchers (`instantvnr.apps`)

After installing `instantvnr-py`:

```python
from instantvnr.apps import run_batch, run_int_single, run_int_dual

# Headless render
run_batch(
    neural_volume="my_volume.bson",
    output="out.png",
    rendering_mode=1,
    spp=32,
    width=1920, height=1080,
    camera_from=[0, 0, -500],
    camera_at=[0, 0, 0],
    camera_up=[0, 1, 0],
)

# Interactive dual view
run_int_dual(
    volume="scene.json",
    network="example-model.json",
    max_steps=100_000,
)
```

### Supported volume types

| `type` string | Device | Description |
|---------------|--------|-------------|
| `"structuredRegular"` | `"cuda"` or `"openvkl"` or `"virtual_memory"` or `"out_of_core"` | Raw binary grid (most common) |
| `"openvdb"` | `"openvkl"` | OpenVDB `.vdb` files |
| `"vtkm"` | `"openvkl"` | VTK-m structured mesh files |
| `"exastitch"` | `"cuda"` | ExaStitch AMR volumes |
| `"exabrick"` | `"cuda"` | ExaBrick AMR volumes |

Required kwargs for `structuredRegular`: `dims=[Dx,Dy,Dz]`, `dtype` (e.g. `"float32"`),
optionally `filename` (`.raw` file path), `spacing`, `n_channels`. The
`"virtual_memory"` and `"out_of_core"` devices additionally require
`range=[vmin, vmax]` (per-voxel normalize-then-trilinear).

Sampler tensor placement follows sampler backend:
- `device="cuda"`: `sample()` / `decode()` return CUDA tensors.
- `device="openvkl"`, `"virtual_memory"`, `"out_of_core"`: `sample()` /
  `decode()` return CPU tensors.

#### File-backed CPU samplers (`virtual_memory`, `out_of_core`)

For volumes that don't fit comfortably in GPU memory, two CPU-backed sampling
modes are available against raw structured files:

- **`virtual_memory`** uses `mmap()` + lazy OS paging. Best for reliable local
  SSD/NVMe; transient I/O failures during a page fault raise SIGBUS (fatal).
- **`out_of_core`** keeps a fixed-size heap-resident block cache populated via
  `pread()`. Safe on slow / unreliable storage (network mounts, slow HDDs);
  I/O errors surface as Python exceptions, never SIGBUS. Cache geometry is
  tunable via `VNR_NUM_BLOCKS` / `VNR_NUM_CONCURRENT_BLOCKS`.

A small training driver, `train_file_backed.py`, exposes both backends via a
`--mode {virtual_memory,out_of_core,both}` switch:

```bash
vnr-train-file-backed \
    --filename /path/to/volume.raw --dims 256 256 256 \
    --dtype uint16 --range 0 4095 \
    --mode out_of_core --expname my_volume
```

A complementary diagnostic script lives at `tests/bench_file_backed_sampler.py`
(it sits next to the pytest tests but is not a pytest test -- run it directly):

```bash
python tests/bench_file_backed_sampler.py \
    --filename /path/to/volume.raw --dims 256 256 256 \
    --dtype uint16 --range 0 4095 \
    --mode out_of_core --steps 32000
```

It runs the sampler in a tight loop with no model attached, so if training
ever crashes you can use it to isolate sampler bugs from GPU / tcnn /
PyTorch issues.

See `packages/sampler/README.md` for the full per-backend reference.

---

## Scene file format (.bson)

A `.bson` file is a self-contained binary scene that embeds everything needed for rendering without
the original volume.  It is written by `create_inr_scene()` during training and read by
`render.py` and the C++ renderer at load time.

Top-level keys:

| Key | Type | Contents |
|-----|------|----------|
| `model.encoding` | dict | Hash-grid encoding configuration |
| `model.network` | dict | MLP configuration |
| `model.loss` | dict | Loss configuration (used by C++ trainer) |
| `parameters.params_binary` | bytes | Raw fp16 network parameters |
| `macrocell.data` | bytes | Per-cell `[vmin, vmax]` pairs (float32) |
| `macrocell.dims` | dict | `{x, y, z}` number of macrocells |
| `macrocell.spacings` | dict | Macrocell size in normalised coordinates |
| `volume.dims` | dict | `{x, y, z}` original voxel dimensions |

Loaded in Python via the `bson` module:

```python
import bson
with open("my_volume.bson", "rb") as f:
    scene = bson.loads(f.read())

enc_cfg   = scene["model"]["encoding"]
params_np = np.frombuffer(scene["parameters"]["params_binary"], dtype=np.float16)
dims      = [scene["volume"]["dims"][k] for k in ("x", "y", "z")]
```

---

## Citation

If you use this toolkit in your research, please cite:

```bibtex
@article{wu2022instant,
  author  = {Wu, Qi and Bauer, David and Doyle, Michael J. and Ma, Kwan-Liu},
  journal = {IEEE Transactions on Visualization and Computer Graphics},
  title   = {Interactive Volume Visualization Via Multi-Resolution Hash Encoding Based Neural Representation},
  year    = {2023},
  pages   = {1--14},
  doi     = {10.1109/TVCG.2023.3293121}
}
```

Project page: <https://wilsoncernwq.github.io/publication/tvcg2022-instant-vnr>

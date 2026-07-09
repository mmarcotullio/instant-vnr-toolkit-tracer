#!/usr/bin/env bash
# Set up the Python environment for pysampler.
#
# Usage:
#   ./setup_venv.sh                       # auto-detect GPU and CUDA version
#   ./setup_venv.sh --clean               # remove .venv and uv cache, then exit
#   VENV_DIR=~/myenv ./setup_venv.sh      # custom venv location
#   SM=86 ./setup_venv.sh                 # override GPU arch (skips nvidia-smi)
#   ENABLE_OPENVDB=OFF ./setup_venv.sh    # skip OpenVDB (.vdb file) support
#
# What it does:
#   1. Detects GPU compute capability and CUDA toolkit version
#   2. Picks the right PyTorch wheel index (stable or nightly for Blackwell)
#   3. Creates a uv venv and installs ISPC
#   4. Runs `uv pip install .[test]` which builds the C++ extension via scikit-build-core
#      (ENABLE_OPENVDB=ON pulls and builds OpenVDB v12.1.1 from source on first run)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${VENV_DIR:-.venv}"

# ── clean mode ───────────────────────────────────────────────────────────────
if [[ "${1:-}" == "--clean" ]]; then
    echo "[clean] removing venv: $VENV_DIR"
    rm -rf "$VENV_DIR"
    proj_name=$(grep -m1 '^name' "$SCRIPT_DIR/pyproject.toml" | sed 's/.*"\(.*\)".*/\1/')
    echo "[clean] clearing uv cache of $proj_name"
    uv cache clean "$proj_name"
    echo "[clean] done"
    exit 0
fi

export UV_VENV_CLEAR=1

# ── check uv ─────────────────────────────────────────────────────────────────
if ! command -v uv &>/dev/null; then
    echo "[error] uv not found — install from https://docs.astral.sh/uv/"
    exit 1
fi

# ── detect GPU SM ─────────────────────────────────────────────────────────────
# Allow SM env var override (e.g. SM=86 ./setup_venv.sh).
if [[ -n "${SM:-}" ]]; then
    echo "[info] Using SM=$SM from environment"
elif command -v nvidia-smi &>/dev/null; then
    SM=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d '.')
    echo "[info] Detected GPU sm_$SM"
else
    echo "[warn] nvidia-smi not found — set SM=<arch> (e.g. SM=86) to specify GPU arch"
    echo "[warn] cmake will auto-detect arch via 'native'; PyTorch stable will be used"
    SM="90"
fi

# ── detect CUDA toolkit ───────────────────────────────────────────────────────
if   command -v nvcc &>/dev/null;       then CUDA_HOME="$(realpath "$(dirname "$(command -v nvcc)")/..")"
elif [[ -x /usr/local/cuda/bin/nvcc ]]; then CUDA_HOME="/usr/local/cuda"
else
    echo "[error] nvcc not found — CUDA toolkit is required (CPU builds are not supported)" >&2
    exit 1
fi

cuda_tag() {
    local ver mm
    ver=$("$CUDA_HOME/bin/nvcc" --version | grep -oP 'release \K[0-9]+\.[0-9]+' | head -1)
    mm="${ver%%.*}${ver##*.}"
    if   [[ $mm -ge 128 ]]; then echo "cu128"
    elif [[ $mm -ge 124 ]]; then echo "cu124"
    elif [[ $mm -ge 121 ]]; then echo "cu121"
    else                         echo "cu118"; fi
}

TAG=$(cuda_tag)
echo "[info] GPU sm_$SM  |  CUDA tag: $TAG  |  CUDA_HOME: $CUDA_HOME"

# ── pick PyTorch index URL ─────────────────────────────────────────────────────
# sm_120+ = Blackwell (RTX 5090, etc.) — not yet in stable PyTorch, needs nightly.
if [[ -n "$SM" ]] && [[ "$SM" -ge 120 ]]; then
    TORCH_INDEX="https://download.pytorch.org/whl/nightly/cu128"
    echo "[info] Blackwell GPU (sm_$SM) — using PyTorch nightly"
else
    TORCH_INDEX="https://download.pytorch.org/whl/$TAG"
fi

# ── CUDA paths ────────────────────────────────────────────────────────────────
export PATH="$CUDA_HOME/bin${PATH:+:$PATH}"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
echo "[info] CUDA bin/lib64 added to PATH/LD_LIBRARY_PATH for this session"

# ── OpenVDB ───────────────────────────────────────────────────────────────────
# Built from source by cmake/dep_openvdb.cmake; CMake emits its own warnings
# for missing optional deps (Blosc).  Set ENABLE_OPENVDB=OFF to skip.
ENABLE_OPENVDB="${ENABLE_OPENVDB:-ON}"

# ── create venv ───────────────────────────────────────────────────────────────
uv venv "$VENV_DIR" --python 3.11
source "$VENV_DIR/bin/activate"

# ── ISPC ─────────────────────────────────────────────────────────────────────
# Required by the OpenVKL CPU device in the C++ bindings build.
# Installed into venv/bin so it's on PATH whenever the venv is active.
ISPC_VERSION="1.30.0"
ISPC_BIN="$VENV_DIR/bin/ispc"
if [[ ! -x "$ISPC_BIN" ]]; then
    echo "[info] installing ISPC v$ISPC_VERSION into venv..."
    ISPC_TMP=$(mktemp -d)
    curl -sL "https://github.com/ispc/ispc/releases/download/v${ISPC_VERSION}/ispc-v${ISPC_VERSION}-linux.tar.gz" \
        | tar -xz -C "$ISPC_TMP" --strip-components=1
    cp "$ISPC_TMP/bin/ispc" "$ISPC_BIN"
    chmod +x "$ISPC_BIN"
    rm -rf "$ISPC_TMP"
    echo "[info] ISPC installed: $("$ISPC_BIN" --version 2>&1 | head -1)"
else
    echo "[info] ISPC already present: $("$ISPC_BIN" --version 2>&1 | head -1)"
fi

# ── install pysampler + deps ─────────────────────────────────────────────────
# CMAKE_ARGS is forwarded by scikit-build-core to the underlying cmake call;
# this is how non-built-in CMake variables (ENABLE_*) reach our CMakeLists.
echo "[info] Building pysampler..."
CMAKE_ARGS="-DENABLE_OPENVDB=$ENABLE_OPENVDB" \
CMAKE_CUDA_ARCHITECTURES="$SM" UV_INDEX="pytorch=$TORCH_INDEX" \
    uv pip install -v .[test]

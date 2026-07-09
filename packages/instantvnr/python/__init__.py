"""instantvnr — Instant Volume Neural Representation C++ bindings and tools."""

__version__ = "0.1.0"

# libtorch.so, libc10.so, etc. live inside the torch package.  Importing torch
# early causes Python to dlopen them, making the symbols available to our own
# shared libraries (libivnr.so, libdevice_hyperinr.so, vnr_network.so, …)
# which link against libtorch at build time.
import torch as _torch  # noqa: F401


def _so_dir():
    """Directory containing the compiled .so extensions and executables."""
    from pathlib import Path
    return Path(__file__).parent


def _load_vnr_network():
    """Import the vnr_network pybind11 extension from this package."""
    import importlib
    return importlib.import_module(".vnr_network", __name__)

"""Thin Python wrapper around the ``pysampler`` C++ extension.

The extension works with raw pointers, so this module handles tensor placement:

- ``device="cuda"`` samplers use CUDA tensors.
- ``device="openvkl"``, ``device="virtual_memory"``, and ``device="out_of_core"``
  samplers use CPU tensors (the extension treats their buffers as host pointers).

The public API stays small: ``create_sampler(...)``, ``sample(...)``,
and ``decode(...)``.
"""

from weakref import WeakKeyDictionary

import torch

try:
    import pysampler as _pysampler_extension
except ImportError as import_error:
    raise ImportError(
        "pysampler is required. Build/install it first (for example with "
        "`uv pip install -e \".[sampler]\"` from the repo root)."
    ) from import_error

# Preferred mapping: sampler object -> backend, automatically cleaned when the
# sampler object is garbage-collected.
_sampler_backend_by_object = WeakKeyDictionary()
# Fallback for extension objects that do not support weak references.
_sampler_backend_by_object_id = {}

_BACKEND_TO_TORCH_DEVICE = {
    "cuda":           torch.device("cuda"),
    "openvkl":        torch.device("cpu"),
    "virtual_memory": torch.device("cpu"),
    "out_of_core":    torch.device("cpu"),
}


def _get_sampler_backend(sampler) -> str:
    """Return the remembered backend for a sampler, defaulting to CUDA.

    Defaulting to CUDA keeps compatibility with sampler objects that were
    created outside this wrapper.
    """
    backend_name = None
    try:
        backend_name = _sampler_backend_by_object.get(sampler)
    except TypeError:
        pass
    if backend_name is None:
        backend_name = _sampler_backend_by_object_id.get(id(sampler))
    return backend_name or "cuda"


def create_sampler(volume_type: str, device: str = "cuda", **kwargs):
    """Create and return a C++ sampler object.

    Args:
        volume_type: One of ``structuredRegular``, ``openvdb``, ``vtkm``,
            ``exastitch``, or ``exabrick``.
        device: Backend string forwarded to C++. Supported values:

            - ``cuda`` -- GPU sampler with CUDA tensors.
            - ``openvkl`` -- CPU sampler via OpenVKL.
            - ``virtual_memory`` -- CPU sampler that mmap()s the raw volume
              file and reads voxels lazily through the OS page cache.
              Requires an explicit ``range=[vmin, vmax]`` kwarg.
            - ``out_of_core`` -- CPU sampler with an explicit heap-resident
              block cache populated via ``pread()``; safe on slow / unreliable
              storage. Also requires an explicit ``range=[vmin, vmax]`` kwarg.
        **kwargs: Volume-specific fields (for example ``dims``, ``dtype``,
            ``filename``, ``field``, ``range``).
    """
    backend_name = str(device).lower()
    if backend_name not in _BACKEND_TO_TORCH_DEVICE:
        valid_values = ", ".join(sorted(_BACKEND_TO_TORCH_DEVICE))
        raise ValueError(
            f"unsupported sampler backend: {device!r}. "
            f"Expected one of: {valid_values}."
        )

    sampler = _pysampler_extension.create_sampler(volume_type, backend_name, **kwargs)
    try:
        _sampler_backend_by_object[sampler] = backend_name
    except TypeError:
        # Some extension objects may not support weak references.
        _sampler_backend_by_object_id[id(sampler)] = backend_name
    return sampler


def sample(sampler, count: int, target_device=None):
    """Sample random coordinates and values from the volume.

    Args:
        sampler: Sampler object returned by :func:`create_sampler`.
        count: Number of random samples to generate.
        target_device: Optional torch device. If provided, output tensors are
            moved to this device only when needed.

    Returns:
        A tuple ``(coordinates, values)`` where:
        - ``coordinates`` has shape ``(count, 3)``
        - ``values`` has shape ``(count, n_channels)``
        - by default, both tensors are on the sampler's backend-native device
          (CUDA for ``cuda``; CPU for ``openvkl`` / ``virtual_memory`` /
          ``out_of_core``)
        - if ``target_device`` is set, tensors are returned on that device.
    """
    backend_name = _get_sampler_backend(sampler)
    backend_native_device = _BACKEND_TO_TORCH_DEVICE[backend_name]
    channel_count = sampler.n_channels()

    # The extension writes values in (n_channels, count) layout.
    coordinate_buffer = torch.empty(
        count, 3, dtype=torch.float32, device=backend_native_device
    )
    value_buffer = torch.empty(
        channel_count, count, dtype=torch.float32, device=backend_native_device
    )

    _pysampler_extension.sample(
        sampler, coordinate_buffer.data_ptr(), value_buffer.data_ptr(), count
    )
    sampled_values = value_buffer.transpose(0, 1).contiguous()

    if target_device is not None:
        requested_device = torch.device(target_device)
        if coordinate_buffer.device != requested_device:
            coordinate_buffer = coordinate_buffer.to(requested_device)
        if sampled_values.device != requested_device:
            sampled_values = sampled_values.to(requested_device)

    return coordinate_buffer, sampled_values


def decode(sampler, coordinates: torch.Tensor):
    """Decode values at user-provided coordinates.

    Args:
        sampler: Sampler object returned by :func:`create_sampler`.
        coordinates: Tensor with shape ``(N, 3)`` in normalized coordinates.
            Input is converted to backend-native contiguous ``float32`` before
            calling the C++ extension.

    Returns:
        Tensor with shape ``(N, n_channels)`` on the sampler's backend-native
        device.
    """
    backend_name = _get_sampler_backend(sampler)
    backend_native_device = _BACKEND_TO_TORCH_DEVICE[backend_name]
    prepared_coordinates = coordinates.to(
        device=backend_native_device, dtype=torch.float32
    ).contiguous()
    sample_count = prepared_coordinates.shape[0]
    channel_count = sampler.n_channels()

    value_buffer = torch.empty(
        channel_count,
        sample_count,
        dtype=torch.float32,
        device=backend_native_device,
    )
    _pysampler_extension.decode(
        sampler,
        prepared_coordinates.data_ptr(),
        value_buffer.data_ptr(),
        sample_count,
    )
    decoded_values = value_buffer.transpose(0, 1).contiguous()
    return decoded_values

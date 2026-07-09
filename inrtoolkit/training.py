"""Mixed-precision training helpers."""

import torch


def _amp_autocast(device_type: str, dtype):
    """Return an autocast context manager for a given device type."""
    if hasattr(torch, "amp") and hasattr(torch.amp, "autocast"):
        return torch.amp.autocast(device_type, dtype=dtype)

    # Backward compatibility for older torch versions.
    if device_type == "cuda":
        return torch.cuda.amp.autocast(dtype=dtype)
    if device_type == "xpu" and hasattr(torch, "xpu"):
        return torch.xpu.amp.autocast(enabled=True, dtype=dtype)
    return torch.cpu.amp.autocast(dtype=dtype)


def autocast():
    if torch.cuda.is_available():
        return _amp_autocast("cuda", torch.float16)
    elif hasattr(torch, "xpu") and torch.xpu.is_available():
        return _amp_autocast("xpu", torch.bfloat16)
    else:
        return _amp_autocast("cpu", torch.float16)


def gradscaler():
    if torch.cuda.is_available():
        if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
            return torch.amp.GradScaler("cuda")
        return torch.cuda.amp.GradScaler()
    else:
        class _NoopScaler:
            def scale(self, loss): return loss
            def step(self, optimizer): optimizer.step()
            def update(self): pass
        return _NoopScaler()

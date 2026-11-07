"""Soft torch and lpips dependency for ``NaturalnessLoss.lpips``.

``import makeup`` succeeds without either package. ``require_lpips()``
raises ``ImportError`` with an install hint when one is missing.
Install with ``pip install torch lpips``.
"""

from __future__ import annotations

try:
    import torch
except ImportError:  # soft dependency
    torch = None

try:
    import lpips
except ImportError:  # soft dependency
    lpips = None


def require_lpips():
    """Return ``(torch, lpips)``, or raise if either package is missing."""
    missing = []
    if torch is None:
        missing.append("torch")
    if lpips is None:
        missing.append("lpips")
    if missing:
        names = " and ".join(missing)
        verb = "is" if len(missing) == 1 else "are"
        raise ImportError(
            f"{names} {verb} required for NaturalnessLoss.lpips. "
            f"Install with: pip install {' '.join(missing)}"
        )
    return torch, lpips

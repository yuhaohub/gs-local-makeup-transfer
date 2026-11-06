"""Soft NumPy dependency for ``makeup``.

``import makeup`` succeeds without NumPy. Numeric methods call
``require_numpy()`` and raise ``ImportError`` with an install hint when it
is missing. Install with ``pip install numpy``.
"""

from __future__ import annotations

try:
    import numpy as np
except ImportError:  # soft dependency
    np = None


def require_numpy():
    """Return the NumPy module, or raise if it is not installed."""
    if np is None:
        raise ImportError(
            "numpy is required for makeup mask algebra, residual projection, "
            "and the smoothness term. Install with: pip install numpy"
        )
    return np

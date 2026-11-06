"""RGB or DC residual ``δ`` with ``supp(δ) ⊆ M`` and ``||δ||∞ ≤ ε``.

``project`` and ``apply`` are NumPy implementations. They do not search for
``δ``. NumPy is a soft import; see ``makeup/README.md``.
"""

from __future__ import annotations

from enum import Enum
from typing import Union

from makeup._numpy import require_numpy
from makeup.mask import ArrayLike, SemanticMask, as_mask_array

MaskLike = Union[ArrayLike, SemanticMask]


class ResidualKind(str, Enum):
    """Where ``δ`` lives."""

    RGB = "rgb"
    """Image-space color residual, shape ``(H, W, 3)``."""

    DC = "dc"
    """Per-Gaussian diffuse (0th-order) color residual, shape ``(N, 3)``."""


class MakeupResidual:
    """Feasible residual: support inside ``M``, infinity norm at most ``ε``.

    ``ε`` is required and has no default. This type does not search for ``δ``.
    """

    def __init__(self, epsilon: float, kind: Union[ResidualKind, str] = ResidualKind.RGB) -> None:
        """Record the bound and the residual domain.

        Args:
            epsilon: bound ``ε ≥ 0``. Entries of ``δ`` must lie in ``[-ε, ε]``.
            kind: ``"rgb"`` or ``"dc"``.

        Raises:
            ValueError: if ``epsilon`` is negative.
        """
        if epsilon < 0:
            raise ValueError("epsilon must be >= 0")
        self.epsilon = float(epsilon)
        self.kind = ResidualKind(kind)

    def project(self, delta: ArrayLike, mask: MaskLike) -> ArrayLike:
        """Project a candidate onto ``{δ : supp(δ) ⊆ M, ||δ||∞ ≤ ε}``.

        Where ``M`` is 0, the result is 0. Elsewhere the candidate is clipped
        to ``[-ε, ε]`` and multiplied by ``M`` clipped to ``[0, 1]``, so a
        soft edge attenuates ``δ`` and ``|δ|`` cannot exceed ``ε``.

        Args:
            delta: candidate residual, shape ``(H, W, 3)`` (RGB) or ``(N, 3)`` (DC).
            mask: ``M`` broadcastable over the spatial axes, or a ``SemanticMask``
                (its ``union``).

        Returns:
            ``δ`` with the same shape as ``delta``.

        Raises:
            ValueError: ``mask`` is not broadcastable to ``delta``.
            ImportError: NumPy is not installed.
        """
        np = require_numpy()
        values = np.asarray(delta, dtype=float)
        weight = _broadcast_weight(as_mask_array(mask), values.shape)
        clipped = np.clip(values, -self.epsilon, self.epsilon)
        return clipped * weight

    def apply(
        self,
        base: ArrayLike,
        delta: ArrayLike,
        mask: MaskLike,
        clip_rgb: bool = True,
    ) -> ArrayLike:
        """Return ``base + δ`` after ``δ`` is projected onto the feasible set.

        Args:
            base: RGB image ``(H, W, 3)`` or per-Gaussian DC color ``(N, 3)``.
            delta: residual, same shape as ``base``.
            mask: semantic mask ``M``.
            clip_rgb: when True (default) and ``kind`` is RGB, clip the sum
                to ``[0, 1]``. DC residuals are left unclipped.

        Returns:
            Edited ``base``, same shape as ``base``.

        Raises:
            ValueError: ``base`` and the projected residual differ in shape.
            ImportError: NumPy is not installed.
        """
        np = require_numpy()
        base_arr = np.asarray(base, dtype=float)
        projected = np.asarray(self.project(delta, mask), dtype=float)
        if base_arr.shape != projected.shape:
            raise ValueError(
                f"base shape {base_arr.shape} != projected residual shape {projected.shape}"
            )
        edited = base_arr + projected
        if clip_rgb and self.kind is ResidualKind.RGB:
            edited = np.clip(edited, 0.0, 1.0)
        return edited


def _broadcast_weight(mask: ArrayLike, shape: tuple) -> ArrayLike:
    """Clip ``mask`` to ``[0, 1]`` and broadcast it to ``shape``."""
    np = require_numpy()
    weight = np.clip(np.asarray(mask, dtype=float), 0.0, 1.0)
    if weight.shape == shape:
        return weight
    if weight.ndim + 1 == len(shape) and shape[:-1] == weight.shape:
        return weight.reshape(weight.shape + (1,))
    try:
        return np.broadcast_to(weight, shape)
    except ValueError as exc:
        raise ValueError(
            f"mask shape {weight.shape} is not broadcastable to residual shape {shape}"
        ) from exc


def project_residual(
    delta: ArrayLike,
    mask: MaskLike,
    epsilon: float,
    kind: Union[ResidualKind, str] = ResidualKind.RGB,
) -> ArrayLike:
    """Project ``delta`` onto ``supp(δ) ⊆ M`` and ``||δ||∞ ≤ epsilon``.

    Args:
        delta: candidate residual, ``(H, W, 3)`` or ``(N, 3)``.
        mask: semantic mask ``M``.
        epsilon: infinity-norm bound ``ε ≥ 0``.
        kind: ``"rgb"`` or ``"dc"``. Projection itself does not use ``kind``.

    Returns:
        Feasible residual ``δ``, same shape as ``delta``.
    """
    return MakeupResidual(epsilon, kind=kind).project(delta, mask)


def apply_residual(
    base: ArrayLike,
    delta: ArrayLike,
    mask: MaskLike,
    epsilon: float,
    clip_rgb: bool = True,
    kind: Union[ResidualKind, str] = ResidualKind.RGB,
) -> ArrayLike:
    """Add a residual that has been restricted to ``M`` and to ``[-ε, ε]``.

    Args:
        base: RGB image or per-Gaussian DC color.
        delta: residual, same shape as ``base``.
        mask: semantic mask ``M``.
        epsilon: infinity-norm bound ``ε ≥ 0``.
        clip_rgb: clip an RGB sum to ``[0, 1]`` when True (default).
        kind: ``"rgb"`` or ``"dc"``. DC sums are not clipped to ``[0, 1]``.

    Returns:
        Edited base, same shape as ``base``.
    """
    return MakeupResidual(epsilon, kind=kind).apply(base, delta, mask, clip_rgb=clip_rgb)

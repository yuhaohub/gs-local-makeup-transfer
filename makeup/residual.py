"""RGB or DC residual ``δ`` with ``supp(δ) ⊆ M`` and ``||δ||∞ ≤ ε``.

Interface only. Projection and application are not implemented.
"""

from __future__ import annotations

from enum import Enum
from typing import Union

from makeup.mask import ArrayLike, SemanticMask

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

        TODO: project and apply still raise ``NotImplementedError``.
        """
        if epsilon < 0:
            raise ValueError("epsilon must be >= 0")
        self.epsilon = float(epsilon)
        self.kind = ResidualKind(kind)

    def project(self, delta: ArrayLike, mask: MaskLike) -> ArrayLike:
        """Project a candidate onto ``{δ : supp(δ) ⊆ M, ||δ||∞ ≤ ε}``.

        Args:
            delta: candidate residual, shape ``(H, W, 3)`` (RGB) or ``(N, 3)`` (DC).
            mask: ``M`` broadcastable over the spatial axes, or a ``SemanticMask``
                (use its union).

        Returns:
            ``δ`` with the same shape as ``delta``. Outside ``M``, ``δ`` is 0.
            Every entry lies in ``[-ε, ε]``.

        Raises:
            NotImplementedError: projection is not implemented.
        """
        del delta, mask
        raise NotImplementedError("MakeupResidual.project is not implemented")

    def apply(self, base: ArrayLike, delta: ArrayLike, mask: MaskLike) -> ArrayLike:
        """Return ``base + δ`` after ``δ`` is projected onto the feasible set.

        Args:
            base: RGB image ``(H, W, 3)`` or per-Gaussian DC color ``(N, 3)``.
            delta: residual, same shape as ``base``.
            mask: semantic mask ``M``.

        Returns:
            Edited ``base``, same shape as ``base``.

        Raises:
            NotImplementedError: application is not implemented.
        """
        del base, delta, mask
        raise NotImplementedError("MakeupResidual.apply is not implemented")


def project_residual(delta: ArrayLike, mask: MaskLike, epsilon: float) -> ArrayLike:
    """Project ``delta`` onto ``supp(δ) ⊆ M`` and ``||δ||∞ ≤ epsilon``.

    Args:
        delta: candidate residual, ``(H, W, 3)`` or ``(N, 3)``.
        mask: semantic mask ``M``.
        epsilon: infinity-norm bound ``ε ≥ 0``.

    Returns:
        Feasible residual ``δ``, same shape as ``delta``.

    Raises:
        NotImplementedError: always, until projection is implemented.
    """
    return MakeupResidual(epsilon).project(delta, mask)


def apply_residual(
    base: ArrayLike,
    delta: ArrayLike,
    mask: MaskLike,
    epsilon: float,
) -> ArrayLike:
    """Add a residual that has been restricted to ``M`` and to ``[-ε, ε]``.

    Args:
        base: RGB image or per-Gaussian DC color.
        delta: residual, same shape as ``base``.
        mask: semantic mask ``M``.
        epsilon: infinity-norm bound ``ε ≥ 0``.

    Returns:
        Edited base, same shape as ``base``.

    Raises:
        NotImplementedError: always, until application is implemented.
    """
    return MakeupResidual(epsilon).apply(base, delta, mask)

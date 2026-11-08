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
            mask: ``M`` with the same shape as ``delta``, or a spatial map
                ``delta.shape[:-1]`` (or that shape plus a singleton channel)
                when the last axis has length 3. A ``SemanticMask`` uses its union.

        Returns:
            ``δ`` with the same shape as ``delta``.

        Raises:
            ValueError: ``mask`` does not match ``delta`` (same shape, or a
                spatial map whose shape is ``delta.shape[:-1]`` when the last
                axis is the 3 color channels).
            ImportError: NumPy is not installed.
        """
        np = require_numpy()
        values = np.asarray(delta, dtype=float)
        weight = _aligned_weight(as_mask_array(mask), values.shape)
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
            ValueError: ``base``, ``delta``, and ``mask`` disagree in shape.
            ImportError: NumPy is not installed.
        """
        np = require_numpy()
        base_arr = np.asarray(base, dtype=float)
        delta_arr = np.asarray(delta, dtype=float)
        mask_arr = np.asarray(as_mask_array(mask), dtype=float)
        if base_arr.shape != delta_arr.shape or not _weight_matches(mask_arr.shape, delta_arr.shape):
            raise ValueError(
                "shape mismatch: "
                f"base {base_arr.shape}, delta {delta_arr.shape}, mask {mask_arr.shape}"
            )
        projected = np.asarray(self.project(delta_arr, mask_arr), dtype=float)
        edited = base_arr + projected
        if clip_rgb and self.kind is ResidualKind.RGB:
            edited = np.clip(edited, 0.0, 1.0)
        return edited


def _weight_matches(mask_shape: tuple, residual_shape: tuple) -> bool:
    """True when ``mask_shape`` is the residual or its spatial grid.

    A spatial map is accepted only for a trailing color axis of length 3,
    as ``(H, W)`` against ``(H, W, 3)`` or ``(N,)`` against ``(N, 3)``.
    A singleton channel ``(..., 1)`` is accepted on that same grid.
    Other shapes, including ones NumPy could broadcast, are mismatches.
    """
    if mask_shape == residual_shape:
        return True
    if not residual_shape or residual_shape[-1] != 3:
        return False
    spatial = residual_shape[:-1]
    return mask_shape == spatial or mask_shape == spatial + (1,)


def _aligned_weight(mask: ArrayLike, shape: tuple) -> ArrayLike:
    """Clip ``mask`` to ``[0, 1]`` and align a spatial map onto ``shape``."""
    np = require_numpy()
    weight = np.asarray(mask, dtype=float)
    if not _weight_matches(weight.shape, shape):
        raise ValueError(f"shape mismatch: delta {shape}, mask {weight.shape}")
    weight = np.clip(weight, 0.0, 1.0)
    if weight.shape == shape or weight.shape == shape[:-1] + (1,):
        return weight
    return weight.reshape(weight.shape + (1,))


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


def project_dual_budget_residual(
    delta: ArrayLike,
    mask: MaskLike,
    soft_mask: MaskLike,
    epsilon: float,
    soft_epsilon: float,
    kind: Union[ResidualKind, str] = ResidualKind.RGB,
) -> ArrayLike:
    """Project one residual with two L∞ balls on a single support.

    ``mask`` is the full support ``M``. ``soft_mask`` selects the tighter
    ball and cannot add sites outside ``M``. Overlap uses ``soft_epsilon``
    and is not written twice. Every other site inside ``M`` is
    ``project_residual(..., epsilon)``. Sites outside ``M`` stay 0.

    B2 ``--ablate softskin`` calls this from
    ``MakeupPgdHook._project_numpy``. That method is the makeup step on
    both AEGIS loops: ``adaptive_linf_pgd_attack`` (via
    ``project_configured_delta``) and ``AEGIS._run_foolbox_attack`` (via
    ``bind_foolbox_attack`` / ``reproject_foolbox``). The live B2 attack
    is the Foolbox ``linfpgd`` path. Hard makeup stays on ``epsilon``
    (0.1). Skin stays on ``soft_epsilon`` (0.07).

    Args:
        delta: candidate residual, ``(H, W, 3)`` or ``(N, 3)``.
        mask: full support ``M``.
        soft_mask: tighter subset, same spatial grid as ``mask``.
        epsilon: hard-region bound ``ε ≥ soft_epsilon ≥ 0``.
        soft_epsilon: soft-region bound.
        kind: ``"rgb"`` or ``"dc"``.

    Returns:
        Feasible residual, same shape as ``delta``.

    Raises:
        ValueError: a bound is negative, ``soft_epsilon`` exceeds
            ``epsilon``, or the masks do not match ``delta``.
    """
    if epsilon < 0 or soft_epsilon < 0:
        raise ValueError("epsilon and soft_epsilon must be >= 0")
    if soft_epsilon > epsilon:
        raise ValueError(
            f"soft_epsilon ({soft_epsilon}) must be <= epsilon ({epsilon})"
        )
    np = require_numpy()
    values = np.asarray(delta, dtype=float)
    support = _aligned_weight(as_mask_array(mask), values.shape)
    soft = _aligned_weight(as_mask_array(soft_mask), values.shape)
    soft = np.minimum(soft, support)
    hard_only = np.where(soft > 0, 0.0, support)
    hard = np.asarray(
        project_residual(values, hard_only, epsilon, kind=kind),
        dtype=float,
    )
    soft_projected = np.asarray(
        project_residual(values, soft, soft_epsilon, kind=kind),
        dtype=float,
    )
    return hard + soft_projected


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

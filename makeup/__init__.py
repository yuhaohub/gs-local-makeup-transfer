"""Semantic makeup mask, bounded residual, and naturalness hooks.

Numeric methods need NumPy (``pip install numpy``). The import is soft:
this package imports without NumPy, and those methods raise ``ImportError``
if it is missing. See ``makeup/README.md``.

Callable with NumPy: ``SemanticMask.region`` / ``union`` / ``support`` when
``regions`` is populated; ``MakeupResidual.project`` / ``apply``;
``project_residual`` / ``apply_residual``; ``NaturalnessLoss.smooth`` and
``NaturalnessLoss.__call__`` (smoothness only).

Still ``NotImplementedError``: ``SemanticMask.from_parsing``, ``semantic_mask``,
``NaturalnessLoss.palette``, and ``NaturalnessLoss.lpips``.
Nothing here optimizes ``δ`` or loads weights.
"""

from makeup.mask import MakeupRegion, SemanticMask, semantic_mask
from makeup.naturalness import NaturalnessLoss, naturalness_loss
from makeup.residual import MakeupResidual, ResidualKind, apply_residual, project_residual

__all__ = [
    "MakeupRegion",
    "MakeupResidual",
    "NaturalnessLoss",
    "ResidualKind",
    "SemanticMask",
    "apply_residual",
    "naturalness_loss",
    "project_residual",
    "semantic_mask",
]

"""Semantic makeup mask, bounded residual, and naturalness hooks.

Numeric methods need NumPy (``pip install numpy``). The import is soft:
this package imports without NumPy, and those methods raise ``ImportError``
if it is missing. ``NaturalnessLoss.lpips`` also needs ``torch`` and
``lpips`` (``pip install torch lpips``) and raises ``ImportError`` if
either is missing. See ``makeup/README.md``.

Callable with NumPy: ``SemanticMask.from_regions`` and ``from_parsing``,
``semantic_mask``, ``region`` / ``union`` / ``support`` when ``regions``
is populated; ``MakeupResidual.project`` / ``apply``; ``project_residual`` /
``apply_residual``; ``NaturalnessLoss.smooth``, ``NaturalnessLoss.palette``,
and ``NaturalnessLoss.__call__`` (smoothness only).

``NaturalnessLoss.lpips`` runs on CPU when torch and lpips are installed.
Nothing here optimizes ``δ``. LPIPS loads an AlexNet backbone on first use;
the other hooks do not.
"""

from makeup.mask import REGIONS, MakeupRegion, SemanticMask, semantic_mask
from makeup.naturalness import (
    NaturalnessLoss,
    lpips_loss,
    naturalness_loss,
    palette_loss,
    smoothness_loss,
)
from makeup.residual import (
    MakeupResidual,
    ResidualKind,
    apply_residual,
    project_dual_budget_residual,
    project_residual,
)

__all__ = [
    "REGIONS",
    "MakeupRegion",
    "MakeupResidual",
    "NaturalnessLoss",
    "ResidualKind",
    "SemanticMask",
    "apply_residual",
    "lpips_loss",
    "naturalness_loss",
    "palette_loss",
    "project_dual_budget_residual",
    "project_residual",
    "semantic_mask",
    "smoothness_loss",
]

"""Interface stubs for a semantic makeup mask and a bounded color residual.

Symbols only: mask ``M`` (lip, eye, brow, cheek), residual ``δ`` with
``supp(δ) ⊆ M`` and ``||δ||∞ ≤ ε``, and naturalness hooks ``L_nat``.
Nothing here optimizes ``δ`` or loads weights.
"""

from makeup.mask import MakeupRegion, SemanticMask, semantic_mask
from makeup.naturalness import NaturalnessLoss, naturalness_loss
from makeup.residual import MakeupResidual, ResidualKind, project_residual

__all__ = [
    "MakeupRegion",
    "MakeupResidual",
    "NaturalnessLoss",
    "ResidualKind",
    "SemanticMask",
    "naturalness_loss",
    "project_residual",
    "semantic_mask",
]

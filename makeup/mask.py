"""Semantic makeup mask ``M`` over lip, eye, brow, and cheek.

Interface only. Face parsing is not implemented.
"""

from __future__ import annotations

from enum import Enum
from typing import Mapping, Optional, Tuple

# Numeric map. Image domain is (H, W); Gaussian domain is (N,).
ArrayLike = object


class MakeupRegion(str, Enum):
    """Regions that make up the semantic support ``M``."""

    LIP = "lip"
    EYE = "eye"
    BROW = "brow"
    CHEEK = "cheek"


REGIONS: Tuple[MakeupRegion, ...] = (
    MakeupRegion.LIP,
    MakeupRegion.EYE,
    MakeupRegion.BROW,
    MakeupRegion.CHEEK,
)


class SemanticMask:
    """Spatial mask ``M`` on an image grid or on per-Gaussian indices.

    Values are intended to lie in ``[0, 1]`` (hard or soft). The union of
    the four regions is the support a residual may occupy.
    """

    def __init__(
        self,
        regions: Optional[Mapping[MakeupRegion, ArrayLike]] = None,
    ) -> None:
        """Hold per-region maps. No alignment or thresholding is applied.

        Args:
            regions: region name to a spatial map, shape ``(H, W)`` or ``(N,)``.
                ``None`` stores an empty mapping.

        TODO: check that every region shares one grid and that values lie in ``[0, 1]``.
        """
        self.regions = dict(regions) if regions is not None else {}

    @classmethod
    def from_parsing(
        cls,
        parsing: ArrayLike,
        *,
        image: Optional[ArrayLike] = None,
        label_map: Optional[Mapping[int, MakeupRegion]] = None,
    ) -> "SemanticMask":
        """Build ``M`` from a face-parsing label map.

        Args:
            parsing: integer labels, shape ``(H, W)`` or ``(N,)``.
            image: optional RGB reference, shape ``(H, W, 3)``, for alignment only.
            label_map: parser id to ``MakeupRegion``. Ids not listed fall outside ``M``.

        Returns:
            A ``SemanticMask`` whose union is ``M``.

        Raises:
            NotImplementedError: parsing is not implemented.
        """
        del parsing, image, label_map
        raise NotImplementedError("SemanticMask.from_parsing is not implemented")

    def region(self, name: MakeupRegion) -> ArrayLike:
        """Return the map for one region, same spatial shape as ``M``.

        Raises:
            NotImplementedError: region lookup is not implemented.
        """
        del name
        raise NotImplementedError("SemanticMask.region is not implemented")

    def union(self) -> ArrayLike:
        """Return ``M``, the soft union of lip, eye, brow, and cheek.

        Raises:
            NotImplementedError: the union is not implemented.
        """
        raise NotImplementedError("SemanticMask.union is not implemented")

    def support(self, threshold: float = 0.0) -> ArrayLike:
        """Return the boolean support of ``M`` (entries strictly above ``threshold``).

        Raises:
            NotImplementedError: support extraction is not implemented.
        """
        del threshold
        raise NotImplementedError("SemanticMask.support is not implemented")


def semantic_mask(
    parsing: ArrayLike,
    *,
    image: Optional[ArrayLike] = None,
    label_map: Optional[Mapping[int, MakeupRegion]] = None,
) -> SemanticMask:
    """Build ``M`` from a face-parsing label map.

    Args:
        parsing: integer labels, shape ``(H, W)`` or ``(N,)``.
        image: optional RGB reference, shape ``(H, W, 3)``.
        label_map: parser id to ``MakeupRegion``.

    Returns:
        Semantic mask ``M`` over lip, eye, brow, and cheek.

    Raises:
        NotImplementedError: always, until parsing is implemented.
    """
    return SemanticMask.from_parsing(parsing, image=image, label_map=label_map)

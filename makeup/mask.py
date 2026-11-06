"""Semantic makeup mask ``M`` over lip, eye, brow, and cheek.

``region``, ``union``, and ``support`` run with NumPy when ``regions`` is
populated. ``from_parsing`` is not implemented. NumPy is a soft import;
see ``makeup/README.md``.
"""

from __future__ import annotations

from enum import Enum
from typing import Mapping, Optional, Tuple

from makeup._numpy import require_numpy

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

    Values are intended to lie in ``[0, 1]`` (hard or soft). ``union`` is the
    elementwise maximum of the stored maps, clipped to ``[0, 1]``.
    """

    def __init__(
        self,
        regions: Optional[Mapping[MakeupRegion, ArrayLike]] = None,
    ) -> None:
        """Hold per-region maps. Keys are normalized to ``MakeupRegion``.

        Args:
            regions: region name to a spatial map, shape ``(H, W)`` or ``(N,)``.
                ``None`` stores an empty mapping. Arrays are not copied or
                range-checked. Use ``from_regions`` for hand-built maps.
        """
        normalized = {}
        if regions:
            for key, value in regions.items():
                normalized[MakeupRegion(key)] = value
        self.regions = normalized

    @classmethod
    def from_regions(
        cls,
        regions: Mapping[MakeupRegion, ArrayLike],
        *,
        clip: bool = True,
    ) -> "SemanticMask":
        """Build ``M`` from hand-built or precomputed region maps.

        This is not face parsing. Every map must share one shape. Values are
        soft-clipped to ``[0, 1]`` when ``clip`` is True. When ``clip`` is
        False, a value outside ``[0, 1]`` raises ``ValueError``.

        Args:
            regions: region name to a spatial map. At least one entry.
            clip: soft-clip into ``[0, 1]`` (default) or require that range.

        Returns:
            A ``SemanticMask`` whose stored maps are finite float arrays.

        Raises:
            ValueError: no maps, shapes differ, non-finite values, or (when
                ``clip`` is False) a value outside ``[0, 1]``.
            ImportError: NumPy is not installed.
        """
        np = require_numpy()
        if not regions:
            raise ValueError("from_regions requires at least one region map")
        stored: dict = {}
        shape = None
        for key, value in regions.items():
            name = key if isinstance(key, MakeupRegion) else MakeupRegion(key)
            arr = np.array(value, dtype=float, copy=True)
            if shape is None:
                shape = arr.shape
            elif arr.shape != shape:
                raise ValueError(
                    f"shape mismatch: region {name.value} has shape {arr.shape}, expected {shape}"
                )
            if not np.all(np.isfinite(arr)):
                raise ValueError(f"region {name.value} has non-finite values")
            if clip:
                arr = np.clip(arr, 0.0, 1.0)
            elif np.any((arr < 0.0) | (arr > 1.0)):
                raise ValueError(f"region {name.value} has values outside [0, 1]")
            stored[name] = arr
        return cls(stored)

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
        """Return a float copy of one region map.

        Args:
            name: ``MakeupRegion`` or its string value (``"lip"``, ``"eye"``,
                ``"brow"``, ``"cheek"``).

        Returns:
            Float array, same shape as the stored map. Not clipped.

        Raises:
            KeyError: that region was not provided.
            ImportError: NumPy is not installed.
        """
        np = require_numpy()
        key = name if isinstance(name, MakeupRegion) else MakeupRegion(name)
        if key not in self.regions:
            raise KeyError(key.value)
        return np.array(self.regions[key], dtype=float, copy=True)

    def union(self) -> ArrayLike:
        """Return ``M``, the elementwise max of stored regions, clipped to ``[0, 1]``.

        Missing regions are skipped. Every stored map must share one shape.

        Returns:
            Float array. Overlaps keep the larger weight. Values above 1 or
            below 0 are clipped.

        Raises:
            ValueError: ``regions`` is empty, or shapes disagree.
            ImportError: NumPy is not installed.
        """
        np = require_numpy()
        if not self.regions:
            raise ValueError("SemanticMask.regions is empty")
        merged = None
        shape = None
        for name, value in self.regions.items():
            arr = np.asarray(value, dtype=float)
            if shape is None:
                shape = arr.shape
            elif arr.shape != shape:
                raise ValueError(
                    f"region {name.value} has shape {arr.shape}, expected {shape}"
                )
            merged = arr if merged is None else np.maximum(merged, arr)
        return np.clip(merged, 0.0, 1.0)

    def support(self, threshold: float = 0.0) -> ArrayLike:
        """Return the boolean support of ``M`` (``union > threshold``).

        Args:
            threshold: cutoff. Default ``0`` marks every strictly positive weight.

        Returns:
            Boolean array, same shape as ``union``.

        Raises:
            ValueError: ``regions`` is empty.
            ImportError: NumPy is not installed.
        """
        np = require_numpy()
        return np.asarray(self.union()) > threshold


def as_mask_array(mask: ArrayLike) -> ArrayLike:
    """Return ``M`` as a float array. A ``SemanticMask`` contributes its union."""
    np = require_numpy()
    if isinstance(mask, SemanticMask):
        return np.asarray(mask.union(), dtype=float)
    return np.asarray(mask, dtype=float)


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

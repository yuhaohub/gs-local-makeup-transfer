"""Semantic makeup mask ``M`` over lip, eye, brow, and cheek.

``region``, ``union``, ``support``, ``from_regions``, and ``from_parsing``
run with NumPy. ``from_parsing`` reads an integer face-parsing map.
NumPy is a soft import; see ``makeup/README.md``.
"""

from __future__ import annotations

from enum import Enum
from types import MappingProxyType
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

# BiSeNet / face-parsing.PyTorch order (prepropess_data.py enumerates
# CelebAMask-HQ attributes from 1; 0 is background). Not the index order
# in the CelebAMask-HQ dataset README. Label 1 (skin) stands in for cheek.
DEFAULT_PARSING_LABELS: Mapping[int, MakeupRegion] = MappingProxyType(
    {
        12: MakeupRegion.LIP,  # u_lip
        13: MakeupRegion.LIP,  # l_lip
        4: MakeupRegion.EYE,  # l_eye
        5: MakeupRegion.EYE,  # r_eye
        2: MakeupRegion.BROW,  # l_brow
        3: MakeupRegion.BROW,  # r_brow
        1: MakeupRegion.CHEEK,  # skin
    }
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
        """Build hard region maps from a face-parsing label map.

        Default ids are ``DEFAULT_PARSING_LABELS`` (BiSeNet /
        face-parsing.PyTorch on CelebAMask-HQ):

        | id | parser name | region |
        | --- | --- | --- |
        | 1 | skin | cheek (no cheek class; override with ``label_map``) |
        | 2, 3 | l_brow, r_brow | brow |
        | 4, 5 | l_eye, r_eye | eye |
        | 12, 13 | u_lip, l_lip | lip |

        Other ids stay outside ``M``, including background ``0``, eyeglasses
        ``6``, nose ``10``, inner mouth ``11``, and hair ``17``. ``label_map``
        replaces the default; it is not merged with it. Ids that share a
        region are combined with a pixelwise max. Every region named in the
        map is stored, even when no pixel has that id (an all-zero map).

        Args:
            parsing: integer labels, shape ``(H, W)`` or ``(N,)``. Float
                arrays are accepted when every value is finite and integral.
            image: optional RGB reference for alignment only. Shape
                ``(H, W, 3)`` when ``parsing`` is ``(H, W)``, or ``(N, 3)``
                when ``parsing`` is ``(N,)``. Pixel values are ignored.
            label_map: parser id to ``MakeupRegion`` (or its string value).
                Ids not listed fall outside ``M``. ``None`` uses
                ``DEFAULT_PARSING_LABELS``.

        Returns:
            A ``SemanticMask`` of hard ``0`` / ``1`` maps. The union is ``M``.

        Raises:
            ValueError: ``parsing`` is not an integer label grid, ``image``
                does not align, ``label_map`` is empty, or a key or region
                name is invalid.
            ImportError: NumPy is not installed.
        """
        np = require_numpy()
        labels = _as_parsing_labels(parsing)
        _check_parsing_alignment(labels, image)
        mapping = _coerce_label_map(label_map)
        stored: dict = {}
        for label_id, region in mapping.items():
            hit = (labels == label_id).astype(float)
            if region in stored:
                stored[region] = np.maximum(stored[region], hit)
            else:
                stored[region] = hit
        return cls.from_regions(stored)

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


def _as_parsing_labels(parsing: ArrayLike) -> ArrayLike:
    """Return ``parsing`` as an ``int64`` label array of shape ``(H, W)`` or ``(N,)``."""
    np = require_numpy()
    labels = np.asarray(parsing)
    if labels.dtype.kind not in "iuf" or labels.ndim not in (1, 2):
        raise ValueError(
            "parsing must be integer labels of shape (H, W) or (N,), "
            f"got dtype {labels.dtype} shape {labels.shape}"
        )
    if labels.dtype.kind == "f":
        if labels.size and not np.all(np.isfinite(labels)):
            raise ValueError("parsing has non-finite values")
        if labels.size and not np.all(labels == np.rint(labels)):
            raise ValueError("parsing labels must be integers")
        labels = np.rint(labels)
    return np.asarray(labels, dtype=np.int64)


def _coerce_label_map(label_map: Optional[Mapping[int, MakeupRegion]]) -> dict:
    """Copy ``label_map`` into ``{int id: MakeupRegion}``. ``None`` uses the default."""
    np = require_numpy()
    source = DEFAULT_PARSING_LABELS if label_map is None else label_map
    if not source:
        raise ValueError("label_map is empty")
    normalized = {}
    for raw_id, raw_region in source.items():
        if isinstance(raw_id, (bool, np.bool_)) or not isinstance(raw_id, (int, np.integer)):
            raise ValueError(f"label id must be an int, got {raw_id!r}")
        region = raw_region if isinstance(raw_region, MakeupRegion) else MakeupRegion(raw_region)
        normalized[int(raw_id)] = region
    return normalized


def _check_parsing_alignment(labels: ArrayLike, image: Optional[ArrayLike]) -> None:
    """Raise when ``image`` is present and its shape does not match ``labels``."""
    if image is None:
        return
    np = require_numpy()
    rgb = np.asarray(image)
    if labels.ndim == 2:
        expected = labels.shape + (3,)
    else:
        expected = (int(labels.shape[0]), 3)
    if rgb.shape != expected:
        raise ValueError(
            f"image shape {rgb.shape} does not align with parsing shape {labels.shape}; "
            f"expected {expected}"
        )


def semantic_mask(
    parsing: ArrayLike,
    *,
    image: Optional[ArrayLike] = None,
    label_map: Optional[Mapping[int, MakeupRegion]] = None,
) -> SemanticMask:
    """Build ``M`` from a face-parsing label map.

    See ``SemanticMask.from_parsing``. ``image`` is an alignment check only.

    Args:
        parsing: integer labels, shape ``(H, W)`` or ``(N,)``.
        image: optional RGB reference. ``(H, W, 3)`` or, for a flat map, ``(N, 3)``.
        label_map: parser id to ``MakeupRegion``. ``None`` uses
            ``DEFAULT_PARSING_LABELS``.

    Returns:
        Semantic mask ``M`` over the regions named by the label map.

    Raises:
        ValueError: ``parsing``, ``image``, or ``label_map`` is invalid.
        ImportError: NumPy is not installed.
    """
    return SemanticMask.from_parsing(parsing, image=image, label_map=label_map)

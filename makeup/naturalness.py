"""Hooks for the naturalness term ``L_nat`` (palette, smoothness, LPIPS).

``smooth`` is a masked spatial total variation on ``H×W×3`` residuals.
``palette`` and ``lpips`` are not implemented. ``__call__`` returns only
``smooth``. NumPy is a soft import; see ``makeup/README.md``. No perceptual
network is loaded.
"""

from __future__ import annotations

from typing import Optional

from makeup._numpy import require_numpy
from makeup.mask import ArrayLike, as_mask_array


class NaturalnessLoss:
    """``L_nat`` hooks. Only the smoothness term is computed."""

    def palette(
        self,
        delta: ArrayLike,
        mask: ArrayLike,
        reference: Optional[ArrayLike] = None,
    ) -> float:
        """Palette term: colors of ``δ`` on ``M`` stay inside a makeup palette.

        Args:
            delta: residual, shape ``(H, W, 3)`` or ``(N, 3)``.
            mask: semantic mask ``M``, broadcastable over ``delta``.
            reference: optional palette ``(K, 3)`` or a reference image.

        Returns:
            A scalar loss.

        Raises:
            NotImplementedError: the palette term is not implemented.
        """
        del delta, mask, reference
        raise NotImplementedError("NaturalnessLoss.palette is not implemented")

    def smooth(self, delta: ArrayLike, mask: ArrayLike) -> float:
        """Masked spatial total variation of an ``H×W×3`` residual.

        The residual is multiplied by ``M`` (clipped to ``[0, 1]``), then
        absolute forward differences are summed over horizontal edges,
        vertical edges, and the three channels:

            Σ |Δ_h (δ ⊙ M)| + Σ |Δ_v (δ ⊙ M)|

        Variation on pixels where ``M`` is 0 does not contribute. A single
        pixel has no edges and scores 0.

        Args:
            delta: residual, shape ``(H, W, 3)``.
            mask: ``M`` as ``(H, W)``, ``(H, W, 1)``, ``(H, W, 3)``, or a
                ``SemanticMask`` (its union).

        Returns:
            A non-negative scalar.

        Raises:
            ValueError: ``delta`` is not ``H×W×3``, or ``mask`` does not match.
            ImportError: NumPy is not installed.
        """
        np = require_numpy()
        values = np.asarray(delta, dtype=float)
        if values.ndim != 3 or values.shape[-1] != 3:
            raise ValueError(f"smooth expects an HxWx3 residual, got shape {values.shape}")
        spatial = _spatial_mask(as_mask_array(mask), values.shape)
        masked = values * spatial[..., None]
        total = 0.0
        if values.shape[1] >= 2:
            total += float(np.abs(masked[:, 1:, :] - masked[:, :-1, :]).sum())
        if values.shape[0] >= 2:
            total += float(np.abs(masked[1:, :, :] - masked[:-1, :, :]).sum())
        return total

    def lpips(
        self,
        edited: ArrayLike,
        original: ArrayLike,
        mask: Optional[ArrayLike] = None,
    ) -> float:
        """LPIPS placeholder between an edited image and the original.

        Does not download or load weights.

        Args:
            edited: edited RGB image, shape ``(H, W, 3)``.
            original: original RGB image, same shape.
            mask: optional ``M``. When given, the term is meant to focus on that region.

        Returns:
            A scalar perceptual distance.

        Raises:
            NotImplementedError: LPIPS is not implemented.
        """
        del edited, original, mask
        raise NotImplementedError("NaturalnessLoss.lpips is not implemented")

    def __call__(
        self,
        edited: ArrayLike,
        original: ArrayLike,
        delta: ArrayLike,
        mask: ArrayLike,
        reference: Optional[ArrayLike] = None,
    ) -> float:
        """Return only the smoothness term.

        Palette and LPIPS stay unimplemented and are not added.
        ``edited``, ``original``, and ``reference`` are accepted to keep the
        hook signature and are ignored.

        Args:
            edited: unused RGB image.
            original: unused RGB image.
            delta: residual inside ``M``, shape ``(H, W, 3)``.
            mask: semantic mask ``M``.
            reference: unused palette.

        Returns:
            The scalar from ``smooth``.
        """
        del edited, original, reference
        return self.smooth(delta, mask)


def _spatial_mask(mask: ArrayLike, residual_shape: tuple) -> ArrayLike:
    """Reduce ``mask`` to ``(H, W)`` matching ``residual_shape[:2]``."""
    np = require_numpy()
    spatial = np.clip(np.asarray(mask, dtype=float), 0.0, 1.0)
    height, width = residual_shape[:2]
    if spatial.shape == (height, width):
        return spatial
    if spatial.shape == residual_shape:
        return np.max(spatial, axis=-1)
    if spatial.shape == (height, width, 1):
        return spatial[..., 0]
    raise ValueError(
        f"mask shape {spatial.shape} does not match residual spatial shape {(height, width)}"
    )


def palette_loss(
    delta: ArrayLike,
    mask: ArrayLike,
    reference: Optional[ArrayLike] = None,
) -> float:
    """Palette hook. See ``NaturalnessLoss.palette``.

    Raises:
        NotImplementedError: always.
    """
    return NaturalnessLoss().palette(delta, mask, reference)


def smoothness_loss(delta: ArrayLike, mask: ArrayLike) -> float:
    """Smoothness hook. See ``NaturalnessLoss.smooth``."""
    return NaturalnessLoss().smooth(delta, mask)


def lpips_loss(
    edited: ArrayLike,
    original: ArrayLike,
    mask: Optional[ArrayLike] = None,
) -> float:
    """LPIPS hook. See ``NaturalnessLoss.lpips``. Loads no weights.

    Raises:
        NotImplementedError: always.
    """
    return NaturalnessLoss().lpips(edited, original, mask)


def naturalness_loss(
    edited: ArrayLike,
    original: ArrayLike,
    delta: ArrayLike,
    mask: ArrayLike,
    reference: Optional[ArrayLike] = None,
) -> float:
    """``L_nat`` hook. Currently equal to ``smoothness_loss`` on ``delta`` and ``mask``."""
    return NaturalnessLoss()(edited, original, delta, mask, reference)

"""Hooks for the naturalness term ``L_nat`` (palette, smoothness, LPIPS).

``smooth`` is a masked spatial total variation on ``H×W×3`` residuals.
``palette`` is a masked NumPy color term. ``lpips`` is spatial AlexNet
LPIPS and needs ``torch`` and ``lpips``. ``__call__`` returns only
``smooth``. NumPy is a soft import; see ``makeup/README.md``.
"""

from __future__ import annotations

from typing import Optional

from makeup._lpips import require_lpips
from makeup._numpy import require_numpy
from makeup.mask import ArrayLike, as_mask_array

_ALEX_LPIPS = None


class NaturalnessLoss:
    """``L_nat`` hooks.

    ``smooth`` and ``palette`` need NumPy. ``lpips`` also needs ``torch``
    and ``lpips``. ``__call__`` returns only the smoothness term.
    """

    def palette(
        self,
        delta: ArrayLike,
        mask: ArrayLike,
        reference: Optional[ArrayLike] = None,
    ) -> float:
        """Masked palette consistency of ``δ`` on ``M``.

        Let ``M_p`` be the mask weight at pixel or Gaussian ``p``, clipped
        to ``[0, 1]``, and ``Z = Σ_p M_p``. The loss is ``0`` when ``Z`` is
        ``0``. A 3-channel mask is reduced with a channel-wise max, matching
        ``smooth``.

        - ``reference is None``: compactness around the masked mean color
          ``δ̄ = Σ_p M_p δ_p / Z``

              Σ_p M_p ||δ_p − δ̄||₂² / Z

        - ``reference.shape == delta.shape``: masked MSE against that color
          field (an RGB image or a per-Gaussian DC target)

              Σ_p M_p ||δ_p − r_p||₂² / Z

        - otherwise ``reference`` must be a palette ``C`` of shape ``(K, 3)``:
          squared distance to the nearest palette color

              Σ_p M_p min_k ||δ_p − C_k||₂² / Z

        A ``(N, 3)`` array with the same shape as a DC residual is the color
        field, not an N-color palette. Use a different ``K`` for a palette.

        Args:
            delta: residual, shape ``(H, W, 3)`` or ``(N, 3)``.
            mask: semantic mask ``M``, broadcastable over ``delta``. A
                ``SemanticMask`` contributes its union.
            reference: optional palette ``(K, 3)`` or a color field with the
                same shape as ``delta``.

        Returns:
            A non-negative scalar.

        Raises:
            ValueError: shapes disagree, or a value is non-finite.
            ImportError: NumPy is not installed.
        """
        np = require_numpy()
        values = np.asarray(delta, dtype=float)
        if values.ndim not in (2, 3) or values.shape[-1] != 3:
            raise ValueError(
                f"palette expects a residual of shape (H, W, 3) or (N, 3), got {values.shape}"
            )
        if not np.all(np.isfinite(values)):
            raise ValueError("delta has non-finite values")
        weight = _palette_weight(as_mask_array(mask), values.shape)
        colors = values.reshape(-1, 3)
        if reference is None:
            return _masked_compactness(colors, weight)
        ref = np.asarray(reference, dtype=float)
        if not np.all(np.isfinite(ref)):
            raise ValueError("reference has non-finite values")
        if ref.shape == values.shape:
            return _masked_mean_sq(np.sum((colors - ref.reshape(-1, 3)) ** 2, axis=-1), weight)
        if ref.ndim == 2 and ref.shape[1] == 3 and ref.shape[0] >= 1:
            return _nearest_palette(colors, ref, weight)
        raise ValueError(
            "reference must be None, a color field with the same shape as delta, "
            f"or a palette of shape (K, 3); got {ref.shape}"
        )

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
        """Spatial AlexNet LPIPS on RGB images, optionally weighted by ``M``.

        Images are ``(H, W, 3)`` in ``[0, 1]`` (values outside that range are
        clipped). The first call loads the AlexNet LPIPS network on CPU; the
        ``lpips`` package may download weights. No GPU is used.

        With no mask, the result is the mean of the spatial distance map.
        With a mask, it is ``Σ M_p D_p / Σ M_p`` over that map (``M`` clipped
        to ``[0, 1]``). An all-zero mask scores 0.

        Args:
            edited: edited RGB image, shape ``(H, W, 3)``.
            original: original RGB image, same shape.
            mask: optional ``M``. When given, the term is the mean of the
                spatial LPIPS map over that region. A ``SemanticMask``
                contributes its union.

        Returns:
            A scalar perceptual distance.

        Raises:
            ValueError: images are not matching ``H×W×3`` arrays, a value is
                non-finite, or ``mask`` does not match.
            ImportError: NumPy, torch, or lpips is not installed.
            RuntimeError: the network returns a spatial map of the wrong size.
        """
        np = require_numpy()
        edited_arr = np.asarray(edited, dtype=float)
        original_arr = np.asarray(original, dtype=float)
        if edited_arr.ndim != 3 or edited_arr.shape[-1] != 3:
            raise ValueError(f"lpips expects an HxWx3 image, got shape {edited_arr.shape}")
        if original_arr.shape != edited_arr.shape:
            raise ValueError(
                f"shape mismatch: edited {edited_arr.shape}, original {original_arr.shape}"
            )
        if not np.all(np.isfinite(edited_arr)) or not np.all(np.isfinite(original_arr)):
            raise ValueError("lpips images have non-finite values")
        spatial = None
        if mask is not None:
            raw = as_mask_array(mask)
            if not np.all(np.isfinite(np.asarray(raw, dtype=float))):
                raise ValueError("mask has non-finite values")
            spatial = _spatial_mask(raw, edited_arr.shape)
        torch, lpips_mod = require_lpips()
        if spatial is not None and float(np.sum(spatial)) == 0.0:
            return 0.0
        distance = _spatial_lpips(torch, lpips_mod, edited_arr, original_arr)
        if spatial is None:
            return float(np.mean(distance))
        weight = np.asarray(spatial, dtype=float)
        return float(np.sum(distance * weight) / np.sum(weight))

    def __call__(
        self,
        edited: ArrayLike,
        original: ArrayLike,
        delta: ArrayLike,
        mask: ArrayLike,
        reference: Optional[ArrayLike] = None,
    ) -> float:
        """Return only the smoothness term.

        ``palette`` and ``lpips`` are separate hooks. They are not added, so
        this call does not load a network and does not read ``reference``.
        ``edited`` and ``original`` are accepted to keep the hook signature.

        Args:
            edited: unused RGB image.
            original: unused RGB image.
            delta: residual inside ``M``, shape ``(H, W, 3)``.
            mask: semantic mask ``M``.
            reference: unused palette or color field.

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


def _palette_weight(mask: ArrayLike, residual_shape: tuple) -> ArrayLike:
    """Return a flat ``[0, 1]`` weight for each color of a ``(..., 3)`` residual."""
    np = require_numpy()
    spatial = np.asarray(mask, dtype=float)
    if not np.all(np.isfinite(spatial)):
        raise ValueError("mask has non-finite values")
    spatial = np.clip(spatial, 0.0, 1.0)
    grid = residual_shape[:-1]
    if spatial.shape == grid:
        weight = spatial
    elif spatial.shape == grid + (1,):
        weight = spatial[..., 0]
    elif spatial.shape == residual_shape:
        weight = np.max(spatial, axis=-1)
    else:
        raise ValueError(
            f"mask shape {spatial.shape} does not match residual shape {residual_shape}"
        )
    return np.reshape(weight, (-1,))


def _masked_mean_sq(per_point: ArrayLike, weight: ArrayLike) -> float:
    """``Σ w_p v_p / Σ w_p``, or ``0`` when the weight sum is ``0``."""
    np = require_numpy()
    total = float(np.sum(weight))
    if total == 0.0:
        return 0.0
    return float(np.sum(per_point * weight) / total)


def _masked_compactness(colors: ArrayLike, weight: ArrayLike) -> float:
    """Masked mean squared deviation from the masked mean color."""
    np = require_numpy()
    total = float(np.sum(weight))
    if total == 0.0:
        return 0.0
    mean = np.sum(colors * weight[:, None], axis=0) / total
    per_point = np.sum((colors - mean) ** 2, axis=-1)
    return float(np.sum(per_point * weight) / total)


def _nearest_palette(colors: ArrayLike, palette: ArrayLike, weight: ArrayLike) -> float:
    """Masked mean squared distance from each color to the nearest palette row."""
    np = require_numpy()
    nearest = np.min(np.sum((colors[:, None, :] - palette[None, :, :]) ** 2, axis=-1), axis=1)
    return _masked_mean_sq(nearest, weight)


def _alex_lpips(torch, lpips_mod):
    """Load AlexNet spatial LPIPS on CPU once."""
    global _ALEX_LPIPS
    if _ALEX_LPIPS is None:
        model = lpips_mod.LPIPS(net="alex", spatial=True, verbose=False)
        model = model.to("cpu")
        model.eval()
        _ALEX_LPIPS = model
    return _ALEX_LPIPS


def _spatial_lpips(torch, lpips_mod, edited: ArrayLike, original: ArrayLike) -> ArrayLike:
    """Return an ``(H, W)`` LPIPS map. Images are clipped to ``[0, 1]``."""
    np = require_numpy()
    model = _alex_lpips(torch, lpips_mod)
    edited_t = _hwc_to_nchw(torch, np.clip(edited, 0.0, 1.0))
    original_t = _hwc_to_nchw(torch, np.clip(original, 0.0, 1.0))
    with torch.no_grad():
        distance = model(edited_t, original_t, normalize=True)
    spatial = distance.detach().float().cpu().numpy()
    if spatial.ndim == 4:
        spatial = spatial[0, 0]
    elif spatial.ndim == 3:
        spatial = spatial[0]
    height, width = edited.shape[:2]
    if spatial.shape != (height, width):
        raise RuntimeError(
            f"LPIPS spatial map has shape {spatial.shape}, expected {(height, width)}"
        )
    return spatial


def _hwc_to_nchw(torch, image: ArrayLike):
    """``(H, W, 3)`` float32 array to a CPU ``(1, 3, H, W)`` tensor."""
    np = require_numpy()
    channels_first = np.transpose(np.asarray(image, dtype=np.float32), (2, 0, 1))
    contiguous = np.ascontiguousarray(channels_first)
    return torch.from_numpy(contiguous).unsqueeze(0)


def palette_loss(
    delta: ArrayLike,
    mask: ArrayLike,
    reference: Optional[ArrayLike] = None,
) -> float:
    """Palette hook. See ``NaturalnessLoss.palette``."""
    return NaturalnessLoss().palette(delta, mask, reference)


def smoothness_loss(delta: ArrayLike, mask: ArrayLike) -> float:
    """Smoothness hook. See ``NaturalnessLoss.smooth``."""
    return NaturalnessLoss().smooth(delta, mask)


def lpips_loss(
    edited: ArrayLike,
    original: ArrayLike,
    mask: Optional[ArrayLike] = None,
) -> float:
    """LPIPS hook. See ``NaturalnessLoss.lpips``.

    Raises:
        ImportError: torch or lpips is not installed.
    """
    return NaturalnessLoss().lpips(edited, original, mask)


def naturalness_loss(
    edited: ArrayLike,
    original: ArrayLike,
    delta: ArrayLike,
    mask: ArrayLike,
    reference: Optional[ArrayLike] = None,
) -> float:
    """``L_nat`` hook. Equal to ``smoothness_loss`` on ``delta`` and ``mask``.

    Does not add ``palette_loss`` or ``lpips_loss``.
    """
    return NaturalnessLoss()(edited, original, delta, mask, reference)

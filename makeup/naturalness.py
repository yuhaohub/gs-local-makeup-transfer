"""Hooks for the naturalness term ``L_nat`` (palette, smoothness, LPIPS).

Placeholders only. No perceptual network is loaded and no loss is computed.
"""

from __future__ import annotations

from typing import Optional

from makeup.mask import ArrayLike


class NaturalnessLoss:
    """Names the three ``L_nat`` terms. Calling any of them is unimplemented."""

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
        """Smoothness term: spatial variation of ``δ`` inside ``M``.

        Args:
            delta: residual, shape ``(H, W, 3)`` or ``(N, 3)``.
            mask: semantic mask ``M``.

        Returns:
            A scalar loss.

        Raises:
            NotImplementedError: the smoothness term is not implemented.
        """
        del delta, mask
        raise NotImplementedError("NaturalnessLoss.smooth is not implemented")

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
        """``L_nat`` as the combination of palette, smoothness, and LPIPS.

        Relative weights are not fixed in this stub.

        Args:
            edited: edited RGB image ``(H, W, 3)``.
            original: original RGB image, same shape.
            delta: residual inside ``M``.
            mask: semantic mask ``M``.
            reference: optional palette passed to the palette term.

        Returns:
            A scalar loss.

        Raises:
            NotImplementedError: ``L_nat`` is not implemented.
        """
        del edited, original, delta, mask, reference
        raise NotImplementedError("NaturalnessLoss is not implemented")


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
    """Smoothness hook. See ``NaturalnessLoss.smooth``.

    Raises:
        NotImplementedError: always.
    """
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
    """``L_nat`` hook. See ``NaturalnessLoss.__call__``.

    Raises:
        NotImplementedError: always.
    """
    return NaturalnessLoss()(edited, original, delta, mask, reference)

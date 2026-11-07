"""Smoke test: mask algebra, parsing, residual bounds, palette. No GPU.

LPIPS is exercised when torch and lpips are installed. Otherwise the test
expects ImportError.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np

import makeup
from makeup import (
    REGIONS,
    MakeupRegion,
    SemanticMask,
    apply_residual,
    lpips_loss,
    palette_loss,
    project_residual,
    semantic_mask,
    smoothness_loss,
)
from makeup.mask import DEFAULT_PARSING_LABELS
from makeup.naturalness import NaturalnessLoss
from makeup.residual import MakeupResidual


class MakeupSmokeTest(unittest.TestCase):
    def test_union_is_clipped_max_and_support_is_threshold(self) -> None:
        lip = np.zeros((2, 2))
        lip[0, 0] = 0.4
        cheek = np.zeros((2, 2))
        cheek[0, 0] = 0.9
        cheek[0, 1] = 1.4
        mask = SemanticMask({MakeupRegion.LIP: lip, "cheek": cheek})
        union = mask.union()
        self.assertTrue(np.allclose(union[0, 0], 0.9))
        self.assertTrue(np.allclose(union[0, 1], 1.0))
        self.assertTrue(np.allclose(union[1, 1], 0.0))
        self.assertTrue(mask.support(0.5)[0, 0])
        self.assertFalse(mask.support(0.95)[0, 0])
        self.assertTrue(np.allclose(mask.region("lip"), lip))
        self.assertTrue(np.allclose(mask.region(MakeupRegion.CHEEK)[0, 1], 1.4))

    def test_project_zeros_outside_mask_and_respects_epsilon(self) -> None:
        lip = np.zeros((4, 5))
        lip[1:3, 1:3] = 1.0
        mask = SemanticMask({MakeupRegion.LIP: lip})
        epsilon = 0.1
        positive = project_residual(np.full((4, 5, 3), 10.0), mask, epsilon)
        negative = project_residual(np.full((4, 5, 3), -10.0), mask, epsilon)
        self.assertTrue(np.all(positive[0] == 0.0))
        self.assertTrue(np.all(positive[:, 4] == 0.0))
        self.assertTrue(np.allclose(positive[1, 1], epsilon))
        self.assertTrue(np.allclose(negative[1, 1], -epsilon))
        self.assertLessEqual(float(np.max(np.abs(positive))), epsilon)
        self.assertLessEqual(float(np.max(np.abs(negative))), epsilon)

        base = np.full((4, 5, 3), 0.2)
        edited = apply_residual(base, np.full((4, 5, 3), 10.0), mask, epsilon)
        self.assertTrue(np.allclose(edited[0, 0], 0.2))
        self.assertTrue(np.allclose(edited[1, 1], 0.3))

    def test_rgb_clip_defaults_on_and_dc_stays_unclipped(self) -> None:
        delta = np.ones((2, 2, 3))
        mask = np.ones((2, 2))
        clipped = MakeupResidual(0.2).apply(np.ones((2, 2, 3)), delta, mask)
        raw = MakeupResidual(0.2).apply(np.ones((2, 2, 3)), delta, mask, clip_rgb=False)
        self.assertTrue(np.all(clipped == 1.0))
        self.assertTrue(np.allclose(raw, 1.2))
        dc = MakeupResidual(0.25, kind="dc").apply(
            np.full((1, 3), 2.0),
            np.ones((1, 3)),
            np.ones((1,)),
            clip_rgb=True,
        )
        self.assertTrue(np.allclose(dc, 2.25))

    def test_smooth_ignores_outside_mask_and_call_matches_smooth(self) -> None:
        mask = np.zeros((3, 4))
        mask[:, 1:3] = 1.0
        inside = np.zeros((3, 4, 3))
        inside[:, 2, :] = 1.0
        outside = np.zeros((3, 4, 3))
        outside[:, 0, :] = 4.0
        loss = NaturalnessLoss()
        self.assertEqual(loss.smooth(np.zeros((3, 4, 3)), mask), 0.0)
        self.assertEqual(loss.smooth(outside, mask), 0.0)
        self.assertGreater(loss.smooth(inside, mask), 0.0)
        self.assertGreater(loss.palette(inside, mask), 0.0)
        self.assertEqual(loss(None, None, inside, mask), loss.smooth(inside, mask))

    def test_public_exports_include_helpers_and_loss_hooks(self) -> None:
        expected = {
            "REGIONS",
            "apply_residual",
            "palette_loss",
            "smoothness_loss",
            "lpips_loss",
        }
        self.assertTrue(expected <= set(makeup.__all__))
        self.assertEqual(
            tuple(REGIONS),
            (MakeupRegion.LIP, MakeupRegion.EYE, MakeupRegion.BROW, MakeupRegion.CHEEK),
        )
        self.assertEqual(palette_loss(np.zeros((2, 2, 3)), np.ones((2, 2))), 0.0)
        flat = np.zeros((2, 2, 3))
        self.assertEqual(smoothness_loss(flat, np.ones((2, 2))), 0.0)

    def test_from_regions_soft_clips_and_checks_shape(self) -> None:
        brow = np.array([[-0.2, 0.3], [1.2, 0.0]])
        mask = SemanticMask.from_regions({"brow": brow})
        self.assertTrue(np.allclose(mask.region(MakeupRegion.BROW), [[0.0, 0.3], [1.0, 0.0]]))
        self.assertTrue(mask.support()[0, 1])
        self.assertFalse(mask.support()[0, 0])
        self.assertFalse(mask.support(0.5)[0, 1])
        with self.assertRaises(ValueError):
            SemanticMask.from_regions({"brow": brow}, clip=False)
        with self.assertRaises(ValueError):
            SemanticMask.from_regions(
                {"lip": np.ones((2, 2)), "eye": np.ones((2, 3))}
            )

    def test_soft_mask_support_scales_residual(self) -> None:
        lip = np.zeros((3, 3))
        lip[1, 1] = 0.4
        mask = SemanticMask.from_regions({MakeupRegion.LIP: lip})
        self.assertTrue(mask.support()[1, 1])
        self.assertFalse(mask.support(0.5)[1, 1])
        self.assertFalse(mask.support()[0, 0])
        projected = project_residual(np.ones((3, 3, 3)), mask, 0.2)
        self.assertTrue(np.allclose(projected[1, 1], 0.08))
        self.assertTrue(np.all(projected[0, 0] == 0.0))
        self.assertLessEqual(float(np.max(np.abs(projected))), 0.2)

    def test_shape_mismatch_raises(self) -> None:
        delta = np.ones((4, 5, 3))
        with self.assertRaises(ValueError) as ctx:
            project_residual(delta, np.ones((1, 1, 1)), 0.1)
        self.assertIn("shape mismatch", str(ctx.exception))
        with self.assertRaises(ValueError) as ctx:
            apply_residual(np.ones((3, 5, 3)), delta, np.ones((4, 5)), 0.1)
        self.assertIn("shape mismatch", str(ctx.exception))
        message = str(ctx.exception)
        self.assertIn(str((3, 5, 3)), message)
        self.assertIn(str((4, 5, 3)), message)
        self.assertIn(str((4, 5)), message)

    def test_dc_sum_stays_unclipped_and_clip_rgb_false_keeps_overflow(self) -> None:
        dc = apply_residual(
            np.full((2, 3), 2.0),
            np.ones((2, 3)),
            np.ones((2,)),
            0.5,
            kind="dc",
        )
        self.assertTrue(np.allclose(dc, 2.5))
        self.assertGreater(float(np.max(dc)), 1.0)
        raw = apply_residual(
            np.ones((2, 2, 3)),
            np.ones((2, 2, 3)),
            np.ones((2, 2)),
            0.3,
            clip_rgb=False,
        )
        self.assertTrue(np.allclose(raw, 1.3))

    def test_negative_epsilon_raises(self) -> None:
        with self.assertRaises(ValueError):
            MakeupResidual(-0.01)

    def test_from_parsing_uses_bisenet_ids(self) -> None:
        self.assertEqual(DEFAULT_PARSING_LABELS[12], MakeupRegion.LIP)
        self.assertEqual(DEFAULT_PARSING_LABELS[13], MakeupRegion.LIP)
        self.assertEqual(DEFAULT_PARSING_LABELS[4], MakeupRegion.EYE)
        self.assertEqual(DEFAULT_PARSING_LABELS[5], MakeupRegion.EYE)
        self.assertEqual(DEFAULT_PARSING_LABELS[2], MakeupRegion.BROW)
        self.assertEqual(DEFAULT_PARSING_LABELS[3], MakeupRegion.BROW)
        self.assertEqual(DEFAULT_PARSING_LABELS[1], MakeupRegion.CHEEK)
        self.assertNotIn(0, DEFAULT_PARSING_LABELS)
        self.assertNotIn(6, DEFAULT_PARSING_LABELS)
        self.assertNotIn(10, DEFAULT_PARSING_LABELS)
        self.assertNotIn(11, DEFAULT_PARSING_LABELS)
        self.assertNotIn(17, DEFAULT_PARSING_LABELS)

        parsing = np.zeros((4, 5), dtype=int)
        parsing[0, 1] = 12  # u_lip
        parsing[1, 1] = 13  # l_lip
        parsing[0, 2] = 4  # l_eye
        parsing[0, 3] = 5  # r_eye
        parsing[2, 3] = 2  # l_brow
        parsing[2, 4] = 3  # r_brow
        parsing[3, 0] = 1  # skin, cheek stand-in
        parsing[3, 4] = 17  # hair
        parsing[1, 4] = 11  # inner mouth
        parsing[1, 3] = 6  # eyeglasses
        image = np.zeros((4, 5, 3))
        mask = semantic_mask(parsing, image=image)
        direct = SemanticMask.from_parsing(parsing.astype(float), image=image)
        self.assertTrue(np.allclose(mask.union(), direct.union()))
        self.assertTrue(np.allclose(mask.region("lip")[0, 1], 1.0))
        self.assertTrue(np.allclose(mask.region("lip")[1, 1], 1.0))
        self.assertTrue(np.allclose(mask.region(MakeupRegion.EYE)[0, 2], 1.0))
        self.assertTrue(np.allclose(mask.region("eye")[0, 3], 1.0))
        self.assertTrue(np.allclose(mask.region("brow")[2, 3:], 1.0))
        self.assertTrue(np.allclose(mask.region("cheek")[3, 0], 1.0))
        self.assertFalse(mask.support()[3, 4])
        self.assertFalse(mask.support()[1, 4])
        self.assertFalse(mask.support()[1, 3])
        self.assertFalse(mask.support()[0, 0])
        projected = project_residual(np.ones((4, 5, 3)), mask, 0.2)
        self.assertTrue(np.allclose(projected[0, 1], 0.2))
        self.assertTrue(np.all(projected[3, 4] == 0.0))
        self.assertTrue(np.all(projected[1, 4] == 0.0))

    def test_from_parsing_custom_map_flat_index_and_errors(self) -> None:
        parsing = np.array([[1, 7]], dtype=int)
        mask = SemanticMask.from_parsing(parsing, label_map={7: "eye"})
        self.assertNotIn(MakeupRegion.CHEEK, mask.regions)
        self.assertTrue(np.allclose(mask.region("eye"), [[0.0, 1.0]]))
        self.assertTrue(np.allclose(mask.union(), [[0.0, 1.0]]))

        flat = np.array([0, 12, 13, 4, 1, 11, 6])
        flat_mask = semantic_mask(flat, image=np.zeros((7, 3)))
        self.assertTrue(np.allclose(flat_mask.region("lip"), [0, 1, 1, 0, 0, 0, 0]))
        self.assertTrue(np.allclose(flat_mask.region("eye"), [0, 0, 0, 1, 0, 0, 0]))
        self.assertTrue(np.allclose(flat_mask.region("cheek"), [0, 0, 0, 0, 1, 0, 0]))
        self.assertTrue(np.allclose(flat_mask.region("brow"), np.zeros(7)))
        self.assertFalse(flat_mask.support()[5])
        self.assertFalse(flat_mask.support()[6])

        with self.assertRaises(ValueError):
            semantic_mask(np.array([[0.0, 1.5]]))
        with self.assertRaises(ValueError):
            semantic_mask(np.array([[np.nan, 1.0]]))
        with self.assertRaises(ValueError):
            semantic_mask(np.zeros((2, 2, 1), dtype=int))
        with self.assertRaises(ValueError):
            semantic_mask(np.ones((2, 2), dtype=bool))
        with self.assertRaises(ValueError) as ctx:
            semantic_mask(np.zeros((2, 3), dtype=int), image=np.zeros((2, 4, 3)))
        self.assertIn("align", str(ctx.exception))
        with self.assertRaises(ValueError):
            SemanticMask.from_parsing(np.zeros((2, 2), dtype=int), label_map={})
        with self.assertRaises(ValueError):
            SemanticMask.from_parsing(np.zeros((2, 2), dtype=int), label_map={"lip": "lip"})
        with self.assertRaises(ValueError):
            SemanticMask.from_parsing(np.zeros((2,), dtype=int), image=np.zeros((2, 1, 3)))

    def test_palette_compactness_field_and_nearest_color(self) -> None:
        loss = NaturalnessLoss()
        delta = np.zeros((2, 2, 3))
        delta[0, 0] = (1.0, 0.0, 0.0)
        delta[0, 1] = (0.0, 1.0, 0.0)
        mask = np.zeros((2, 2))
        mask[0, :] = 1.0
        self.assertTrue(np.allclose(loss.palette(delta, mask), 0.5))
        leaked = delta.copy()
        leaked[1, 0] = (9.0, 9.0, 9.0)
        self.assertTrue(np.allclose(loss.palette(leaked, mask), 0.5))
        self.assertEqual(loss.palette(leaked, np.zeros((2, 2))), 0.0)
        lip_only = mask.copy()
        lip_only[0, 1] = 0.0
        self.assertEqual(loss.palette(delta, lip_only), 0.0)
        self.assertTrue(np.allclose(loss.palette(delta, mask[..., None]), 0.5))
        self.assertTrue(np.allclose(loss.palette(delta, np.repeat(mask[..., None], 3, axis=-1)), 0.5))
        semantic = SemanticMask.from_regions({MakeupRegion.LIP: mask})
        self.assertTrue(np.allclose(palette_loss(delta, semantic), 0.5))

        field = np.zeros_like(delta)
        field[0, 0] = (1.0, 0.0, 0.0)
        field[0, 1] = (0.0, 1.0, 0.0)
        self.assertEqual(loss.palette(delta, mask, field), 0.0)
        self.assertTrue(np.allclose(loss.palette(np.zeros_like(delta), mask, field), 1.0))

        palette = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
        self.assertTrue(np.allclose(loss.palette(delta, mask, palette), 1.0))

        dc = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        ones = np.ones((2,))
        self.assertTrue(np.allclose(palette_loss(dc, ones), 0.5))
        self.assertTrue(np.allclose(palette_loss(dc, ones, np.zeros_like(dc)), 1.0))
        self.assertTrue(np.allclose(palette_loss(dc, ones, np.array([[1.0, 0.0, 0.0]])), 1.0))

        weighted = np.zeros((2, 3))
        weighted[0] = (1.0, 0.0, 0.0)
        weights = np.array([0.25, 0.75])
        # Masked mean is (0.25, 0, 0). Deviations: 0.75 and 0.25, squared 0.5625 and 0.0625.
        # Weighted mean of those squares: (0.25 * 0.5625 + 0.75 * 0.0625) / 1 = 0.1875.
        self.assertTrue(np.allclose(palette_loss(weighted, weights), 0.1875))

        with self.assertRaises(ValueError):
            loss.palette(np.ones((2, 2)), np.ones((2, 2)))
        with self.assertRaises(ValueError):
            loss.palette(np.ones((2, 2, 3)), np.ones((3, 3)))
        with self.assertRaises(ValueError):
            loss.palette(np.ones((2, 2, 3)), np.ones((2, 2)), reference=np.ones((2, 2)))
        bad = delta.copy()
        bad[0, 0, 0] = np.nan
        with self.assertRaises(ValueError):
            loss.palette(bad, mask)

    def test_lpips_rejects_bad_shapes_and_soft_depends_on_torch(self) -> None:
        original = np.zeros((8, 8, 3))
        edited = np.ones((8, 8, 3))
        with self.assertRaises(ValueError):
            lpips_loss(np.zeros((2, 2)), original)
        with self.assertRaises(ValueError):
            lpips_loss(edited, np.zeros((8, 7, 3)))
        with self.assertRaises(ValueError):
            lpips_loss(edited, original, mask=np.ones((7, 8)))
        try:
            import torch
            import lpips
        except ImportError:
            with self.assertRaises(ImportError) as ctx:
                lpips_loss(edited, original)
            message = str(ctx.exception)
            self.assertIn("pip install", message)
            self.assertTrue("torch" in message or "lpips" in message)
            with self.assertRaises(ImportError):
                lpips_loss(edited, original, np.zeros((8, 8)))
            return
        del torch, lpips
        original = np.zeros((64, 64, 3))
        edited = np.ones((64, 64, 3))
        same = lpips_loss(original, original)
        self.assertLess(abs(same), 1e-3)
        changed = lpips_loss(edited, original)
        self.assertGreater(changed, same)
        self.assertEqual(lpips_loss(edited, original, np.zeros((64, 64))), 0.0)
        self.assertAlmostEqual(lpips_loss(edited, original, np.ones((64, 64))), changed, places=5)

    def test_lpips_weights_spatial_map_without_a_network(self) -> None:
        distance = np.array([[0.0, 2.0], [4.0, 6.0]])

        def fake_spatial(torch, lpips_mod, edited, original):
            del torch, lpips_mod, original
            self.assertEqual(edited.shape, (2, 2, 3))
            return distance

        edited = np.ones((2, 2, 3))
        original = np.zeros((2, 2, 3))
        with patch("makeup.naturalness.require_lpips", return_value=(None, None)), patch(
            "makeup.naturalness._spatial_lpips", side_effect=fake_spatial
        ):
            self.assertAlmostEqual(lpips_loss(edited, original), 3.0)
            self.assertAlmostEqual(lpips_loss(edited, original, np.array([[0.0, 1.0], [1.0, 0.0]])), 3.0)
            self.assertAlmostEqual(lpips_loss(edited, original, np.array([[1.0, 0.5], [0.0, 0.0]])), 2.0 / 3.0)
            self.assertEqual(lpips_loss(edited, original, np.zeros((2, 2))), 0.0)


if __name__ == "__main__":
    unittest.main()

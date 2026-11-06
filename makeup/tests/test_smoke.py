"""Smoke test: project zeros outside M and respects ε. No GPU."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

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
        with self.assertRaises(NotImplementedError):
            palette_loss(np.zeros((2, 2, 3)), np.ones((2, 2)))
        with self.assertRaises(NotImplementedError):
            lpips_loss(np.zeros((2, 2, 3)), np.zeros((2, 2, 3)))
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

    def test_parsing_palette_and_lpips_stay_unimplemented(self) -> None:
        with self.assertRaises(NotImplementedError):
            semantic_mask(np.zeros((2, 2), dtype=int))
        loss = NaturalnessLoss()
        with self.assertRaises(NotImplementedError):
            loss.palette(np.zeros((2, 2, 3)), np.ones((2, 2)))
        with self.assertRaises(NotImplementedError):
            loss.lpips(np.zeros((2, 2, 3)), np.zeros((2, 2, 3)))
        with self.assertRaises(ValueError):
            MakeupResidual(-0.01)


if __name__ == "__main__":
    unittest.main()

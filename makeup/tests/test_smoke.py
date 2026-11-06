"""Smoke test: project zeros outside M and respects ε. No GPU."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np

from makeup.mask import MakeupRegion, SemanticMask, semantic_mask
from makeup.naturalness import NaturalnessLoss
from makeup.residual import MakeupResidual, apply_residual, project_residual


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

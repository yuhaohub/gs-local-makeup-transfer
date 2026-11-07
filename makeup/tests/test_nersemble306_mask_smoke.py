"""CPU smoke: NeRSemble 306 residual stays inside M and ε. No GPU."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np

from makeup import SemanticMask
from makeup.aegis_mask import (
    AEGIS_HOOK_TODO,
    SUBJECT,
    aegis_pgd_hook_stub,
    command_text,
    documented_fallback_labels,
    mask_avatar_argv,
    projected_pgd_step,
    resolve_parsing,
    run_cpu_mask_smoke,
    support_violation,
)
from makeup.residual import project_residual

_SCRIPT = _ROOT / "scripts" / "nersemble306_makeup_mask_smoke.py"
_B0A = _ROOT / "results" / "b0_aegis_sanity_summary.json"


class NerSemble306MaskSmokeTest(unittest.TestCase):
    def test_gpu_command_matches_b0a_and_stays_on_306(self) -> None:
        summary = json.loads(_B0A.read_text(encoding="utf-8"))
        command = command_text()
        self.assertEqual(command, summary["metrics"]["command"])
        self.assertEqual(summary["subject"], SUBJECT)
        self.assertIn("NeRSembleReconst/avatars/306", command)
        self.assertIn("--embedder adaface", command)
        self.assertIn("--epsilons 0.1", command)
        self.assertIn("--target-features DC", command)
        self.assertNotIn("FaceScape", command)
        self.assertNotIn("527", command)
        argv = mask_avatar_argv()
        self.assertEqual(argv[0:4], ["uv", "run", "python", "scripts/mask_avatar.py"])
        with self.assertRaises(ValueError):
            mask_avatar_argv("527")
        with self.assertRaises(ValueError):
            mask_avatar_argv(74)

    def test_fallback_labels_go_through_from_parsing(self) -> None:
        labels = documented_fallback_labels((12,))
        self.assertTrue(
            np.array_equal(labels, [12, 13, 4, 5, 2, 3, 1, 0, 12, 17, 10, 0])
        )
        mask = SemanticMask.from_parsing(labels)
        self.assertGreater(float(mask.region("lip")[0]), 0.0)
        self.assertGreater(float(mask.region("eye")[2]), 0.0)
        self.assertGreater(float(mask.region("brow")[4]), 0.0)
        self.assertGreater(float(mask.region("cheek")[6]), 0.0)
        support = mask.support()
        self.assertTrue(support[0])
        self.assertFalse(support[9])
        self.assertFalse(support[10])
        self.assertFalse(support[11])
        resolved = resolve_parsing(aegis_root=Path("/no/such/aegis"))
        self.assertEqual(resolved.mask_source, "documented_fallback")
        self.assertIsNone(resolved.path)

    def test_projected_step_zeros_outside_support_and_raw_step_does_not(self) -> None:
        labels = documented_fallback_labels((12,))
        mask = SemanticMask.from_parsing(labels)
        raw = np.full((12, 3), 0.1)
        self.assertGreater(support_violation(raw, mask), 0.0)
        projected = project_residual(raw, mask, 0.1, kind="dc")
        self.assertEqual(support_violation(projected, mask), 0.0)
        self.assertLessEqual(float(np.max(np.abs(projected))), 0.1)

        delta = np.zeros((12, 3))
        stepped = projected_pgd_step(
            delta, np.ones((12, 3)), mask, 0.1, 0.05, kind="dc"
        )
        self.assertEqual(support_violation(stepped, mask), 0.0)
        self.assertTrue(np.allclose(stepped[0], -0.05))
        self.assertTrue(np.all(stepped[11] == 0.0))
        leaked = delta - 0.05 * np.ones((12, 3))
        self.assertGreater(support_violation(leaked, mask), 0.0)

    def test_soft_mask_scales_the_projected_step(self) -> None:
        lip = np.zeros((4,))
        lip[1] = 0.4
        mask = SemanticMask.from_regions({"lip": lip})
        out = projected_pgd_step(
            np.zeros((4, 3)),
            -np.ones((4, 3)),
            mask,
            0.1,
            1.0,
            kind="dc",
        )
        self.assertTrue(np.allclose(out[1], 0.04))
        self.assertTrue(np.all(out[0] == 0.0))
        self.assertLessEqual(float(np.max(np.abs(out))), 0.1)

    def test_cpu_loop_enforces_the_constraint_on_the_fallback(self) -> None:
        resolved = resolve_parsing()
        smoke = run_cpu_mask_smoke(resolved, steps=3)
        self.assertEqual(smoke["mask_source"], "documented_fallback")
        self.assertEqual(smoke["built_with"], "SemanticMask.from_parsing")
        self.assertEqual(smoke["kind"], "dc")
        self.assertTrue(smoke["constraint_holds"])
        self.assertEqual(smoke["pgd_outside_support_max_abs"], 0.0)
        self.assertAlmostEqual(smoke["pgd_linf"], 0.1)
        self.assertGreater(smoke["unprojected_outside_support_max_abs"], 0.0)
        self.assertEqual(smoke["posthoc_outside_support_max_abs"], 0.0)
        self.assertTrue(smoke["apply_outside_unchanged"])
        self.assertTrue(smoke["apply_matches_projected_delta"])
        self.assertEqual(smoke["aegis_hook"], "in_loop")
        self.assertEqual(
            smoke["aegis_hook_api"],
            "makeup.aegis_pgd_hook.install_makeup_pgd_hook",
        )
        self.assertEqual(smoke["hooked_adaptive_outside_support_max_abs"], 0.0)
        self.assertEqual(smoke["hooked_foolbox_outside_support_max_abs"], 0.0)
        self.assertGreater(smoke["epsilon_only_adaptive_outside_support_max_abs"], 0.0)
        self.assertGreater(smoke["epsilon_only_foolbox_outside_support_max_abs"], 0.0)
        self.assertIn("TODO(james)", smoke["aegis_hook_todo"])
        self.assertIn("_run_foolbox_attack", smoke["aegis_hook_todo"])
        self.assertIn("adaptive_linf_pgd_attack", smoke["aegis_hook_todo"])
        self.assertEqual(aegis_pgd_hook_stub(), AEGIS_HOOK_TODO)

    def test_explicit_parsing_array_and_sidecar(self) -> None:
        labels = np.zeros((2, 4), dtype=np.int64)
        labels[0, 0] = 12
        labels[0, 1] = 4
        labels[1, 0] = 2
        labels[1, 1] = 1
        resolved = resolve_parsing(parsing=labels)
        self.assertEqual(resolved.mask_source, "from_parsing")
        smoke = run_cpu_mask_smoke(resolved, steps=2)
        self.assertEqual(smoke["kind"], "rgb")
        self.assertTrue(smoke["constraint_holds"])
        self.assertEqual(smoke["pgd_outside_support_max_abs"], 0.0)
        with self.assertRaises(ValueError):
            resolve_parsing(parsing=labels, parsing_path=Path("also.npy"))
        with self.assertRaises(ValueError):
            resolve_parsing(subject="527")

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            sidecar = root / "datasets" / "NeRSembleReconst" / "parsing" / "306.npy"
            sidecar.parent.mkdir(parents=True)
            stored = np.array([12, 0, 4, 1, 2, 13, 5, 3], dtype=np.int64)
            np.save(sidecar, stored)
            loaded = resolve_parsing(aegis_root=root)
            self.assertEqual(loaded.mask_source, "from_parsing")
            self.assertEqual(loaded.path, sidecar)
            self.assertTrue(np.array_equal(loaded.labels, stored))
            self.assertTrue(run_cpu_mask_smoke(loaded, steps=1)["constraint_holds"])

    def test_cli_writes_cpu_summary_and_rejects_527(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            proc = subprocess.run(
                [
                    sys.executable,
                    str(_SCRIPT),
                    "--results-dir",
                    tmp,
                    "--aegis-root",
                    str(Path(tmp) / "missing-aegis"),
                ],
                cwd=_ROOT,
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(proc.returncode, 0, proc.stderr + proc.stdout)
            payload = json.loads(
                (Path(tmp) / "nersemble306_makeup_mask_smoke.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(payload["status"], "cpu_smoke_ok")
            self.assertEqual(payload["subject"], "306")
            self.assertTrue(payload["constraint_holds"])
            self.assertFalse(payload["gpu_launched"])
            self.assertEqual(payload["mask_source"], "documented_fallback")
            self.assertEqual(payload["pgd_outside_support_max_abs"], 0.0)
            self.assertGreater(payload["unprojected_outside_support_max_abs"], 0.0)
            self.assertNotIn("FaceScape", payload["command"])
            self.assertNotIn("527", payload["command"])
            self.assertTrue(payload["gpu_blockers"])
            joined = " ".join(payload["gpu_blockers"])
            self.assertNotIn("FaceScape", joined)
            csv_text = (
                Path(tmp) / "nersemble306_makeup_mask_smoke.csv"
            ).read_text(encoding="utf-8")
            self.assertIn("306-makeup-mask-smoke", csv_text)
            self.assertIn("cpu_smoke_ok", csv_text)

            blocked = subprocess.run(
                [
                    sys.executable,
                    str(_SCRIPT),
                    "--results-dir",
                    str(Path(tmp) / "launch"),
                    "--aegis-root",
                    str(Path(tmp) / "missing-aegis"),
                    "--launch",
                ],
                cwd=_ROOT,
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(blocked.returncode, 0, blocked.stderr + blocked.stdout)
            launched = json.loads(
                (Path(tmp) / "launch" / "nersemble306_makeup_mask_smoke.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertFalse(launched["gpu_launched"])
            self.assertEqual(launched["status"], "cpu_smoke_ok")

        rejected = subprocess.run(
            [sys.executable, str(_SCRIPT), "--subject", "527"],
            cwd=_ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(rejected.returncode, 2)
        self.assertIn("306", rejected.stderr)


if __name__ == "__main__":
    unittest.main()

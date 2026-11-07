"""CPU tests for the in-loop AEGIS makeup projection. No GPU and no clone."""

from __future__ import annotations

import json
import os
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
from makeup.aegis_mask import documented_fallback_labels
from makeup.aegis_pgd_hook import (
    ADAPTIVE_CLAMP_ANCHOR,
    FOOLBOX_ATTACK_ANCHOR,
    HELPERS_TAIL,
    HOOK_CONFIG_ENV,
    MODEL_TAIL,
    MakeupPgdHook,
    bind_foolbox_attack,
    clear_hook_state,
    epsilon_only_adaptive_step,
    epsilon_only_foolbox_project,
    in_loop_leakage_report,
    install_makeup_pgd_hook,
    locate_aegis_call_sites,
    outside_support_max_abs,
    patch_aegis_sources,
)

_SCRIPT = _ROOT / "scripts" / "nersemble306_makeup_mask_smoke.py"


def _fallback_mask() -> SemanticMask:
    return SemanticMask.from_parsing(documented_fallback_labels((12,)))


def _excerpt_tree(root: Path, helpers: str, model: str) -> None:
    core = root / "src" / "aegis" / "core"
    core.mkdir(parents=True)
    (core / "helpers.py").write_text(helpers, encoding="utf-8")
    (core / "model.py").write_text(model, encoding="utf-8")


def _ready_sources() -> tuple[str, str]:
    helpers = (
        "def adaptive_linf_pgd_attack():\n"
        "    delta_new = delta - step_size * grad_sign\n"
        f"{ADAPTIVE_CLAMP_ANCHOR}"
        "        delta = delta_new.clone().detach().requires_grad_(True)\n"
    )
    model = (
        "def _run_foolbox_attack(self, target_class):\n"
        f"{FOOLBOX_ATTACK_ANCHOR}"
        "        return attack\n"
    )
    return helpers, model


class LinfProjectedGradientDescentAttack:
    """Stand-in with the upstream class name and ε-only project."""

    def project(self, x, x0, epsilon):
        return epsilon_only_foolbox_project(x, x0, epsilon)


class _OtherAttack:
    def project(self, x, x0, epsilon):
        return x


class AegisPgdHookTest(unittest.TestCase):
    def tearDown(self) -> None:
        clear_hook_state()
        os.environ.pop(HOOK_CONFIG_ENV, None)

    def test_hooked_steps_stay_inside_support_and_epsilon_clamp_leaks(self) -> None:
        mask = _fallback_mask()
        report = in_loop_leakage_report(mask, 0.1, "dc", steps=3)
        self.assertEqual(report["hooked_adaptive_outside_support_max_abs"], 0.0)
        self.assertEqual(report["hooked_foolbox_outside_support_max_abs"], 0.0)
        self.assertAlmostEqual(report["epsilon_only_adaptive_outside_support_max_abs"], 0.1)
        self.assertAlmostEqual(report["epsilon_only_foolbox_outside_support_max_abs"], 0.1)
        self.assertAlmostEqual(report["hooked_adaptive_linf"], 0.1)
        self.assertAlmostEqual(report["hooked_foolbox_linf"], 0.1)

        hook = MakeupPgdHook(mask, 0.1, kind="dc")
        origin = np.zeros((1, 12, 1, 3))
        stepped = np.full((1, 12, 1, 3), -0.25)
        hooked = hook.foolbox_project(stepped, origin)
        leaked = epsilon_only_foolbox_project(stepped, origin, 0.1)
        self.assertEqual(hooked.shape, origin.shape)
        self.assertEqual(outside_support_max_abs(hooked - origin, mask, "dc"), 0.0)
        self.assertGreater(outside_support_max_abs(leaked - origin, mask, "dc"), 0.0)
        self.assertTrue(np.allclose(hooked[0, 0, 0], -0.1))
        self.assertTrue(np.all(hooked[0, 11, 0] == 0.0))
        self.assertTrue(np.allclose(leaked[0, 11, 0], -0.1))

    def test_adaptive_keeps_tighter_per_element_epsilon_inside_mask(self) -> None:
        mask = _fallback_mask()
        hook = MakeupPgdHook(mask, 0.1, kind="dc")
        per_element = np.full(12, 0.1)
        per_element[0] = 0.02
        hooked = hook.adaptive_step(
            np.zeros((12, 3)), np.ones((12, 3)), per_element, 1.0
        )
        leaked = epsilon_only_adaptive_step(
            np.zeros((12, 3)), np.ones((12, 3)), per_element, 1.0
        )
        self.assertTrue(np.allclose(hooked[0], -0.02))
        self.assertTrue(np.all(hooked[11] == 0.0))
        self.assertEqual(outside_support_max_abs(hooked, mask, "dc"), 0.0)
        self.assertTrue(np.allclose(leaked[11], -0.1))
        self.assertGreater(outside_support_max_abs(leaked, mask, "dc"), 0.0)

        capped = MakeupPgdHook(mask, 0.05, kind="dc").adaptive_step(
            np.zeros((12, 3)), np.ones((12, 3)), 0.1, 1.0
        )
        self.assertTrue(np.allclose(capped[0], -0.05))
        self.assertTrue(np.all(capped[11] == 0.0))

    def test_soft_mask_scales_the_in_loop_step(self) -> None:
        lip = np.zeros((4,))
        lip[1] = 0.4
        mask = SemanticMask.from_regions({"lip": lip})
        hook = MakeupPgdHook(mask, 0.1, kind="dc")
        out = hook.adaptive_step(
            np.zeros((4, 3)), -np.ones((4, 3)), 0.1, 1.0
        )
        self.assertTrue(np.allclose(out[1], 0.04))
        self.assertTrue(np.all(out[0] == 0.0))
        self.assertLessEqual(float(np.max(np.abs(out))), 0.1)

    def test_subject_527_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            MakeupPgdHook(_fallback_mask(), 0.1, subject="527")

    def test_missing_tree_is_reported_and_not_created(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "AEGIS"
            report = locate_aegis_call_sites(root)
            self.assertEqual(report["status"], "tree_missing")
            self.assertIn("nothing was cloned", report["note"])
            installed = install_makeup_pgd_hook(
                MakeupPgdHook(_fallback_mask(), 0.1, kind="dc"),
                root,
                mask_source="from_parsing",
            )
            self.assertEqual(installed["status"], "tree_missing")
            self.assertFalse(installed["armed"])
            self.assertFalse(root.exists())

    def test_patch_inserts_both_call_sites_and_is_idempotent(self) -> None:
        helpers, model = _ready_sources()
        patched_helpers, patched_model = patch_aegis_sources(helpers, model)
        call = patched_helpers.index("_gs_makeup_project_delta(delta_new)")
        clamp = patched_helpers.index("delta_new < -eps")
        clone = patched_helpers.index("delta_new.clone()")
        sign = patched_helpers.index("delta - step_size * grad_sign")
        self.assertLess(sign, clamp)
        self.assertLess(clamp, call)
        self.assertLess(call, clone)
        self.assertIn("project_configured_delta", patched_helpers)
        bind = patched_model.index("_gs_makeup_bind_foolbox_attack(attack)")
        attack_call = patched_model.index("attack = get_foolbox_attack(")
        self.assertLess(attack_call, bind)
        self.assertIn("bind_foolbox_attack", patched_model)
        again_helpers, again_model = patch_aegis_sources(patched_helpers, patched_model)
        self.assertEqual(again_helpers, patched_helpers)
        self.assertEqual(again_model, patched_model)
        self.assertEqual(again_helpers.count("_gs_makeup_project_delta(delta_new)"), 1)
        compile(HELPERS_TAIL, "<helpers-tail>", "exec")
        compile(MODEL_TAIL, "<model-tail>", "exec")

    def test_install_on_a_ready_tree_arms_the_executed_stubs(self) -> None:
        mask = _fallback_mask()
        hook = MakeupPgdHook(mask, 0.1, kind="dc")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "AEGIS"
            helpers, model = _ready_sources()
            _excerpt_tree(root, helpers, model)
            self.assertEqual(locate_aegis_call_sites(root)["status"], "ready")
            refused = install_makeup_pgd_hook(
                hook, root, mask_source="documented_fallback"
            )
            self.assertEqual(refused["status"], "refused_fallback")
            self.assertFalse((root / ".gs_makeup_hook.json").exists())
            self.assertNotIn(
                "_gs_makeup_project_delta(delta_new)",
                (root / "src" / "aegis" / "core" / "helpers.py").read_text(
                    encoding="utf-8"
                ),
            )

            installed = install_makeup_pgd_hook(
                hook, root, mask_source="from_parsing"
            )
            self.assertEqual(installed["status"], "patched")
            self.assertTrue(installed["armed"])
            self.assertEqual(locate_aegis_call_sites(root)["status"], "patched")
            install_makeup_pgd_hook(hook, root, mask_source="from_parsing")
            helpers_text = (root / "src" / "aegis" / "core" / "helpers.py").read_text(
                encoding="utf-8"
            )
            self.assertEqual(helpers_text.count("_gs_makeup_project_delta(delta_new)"), 1)

            clear_hook_state()
            os.environ[HOOK_CONFIG_ENV] = installed["config"]
            helper_ns: dict = {}
            exec(HELPERS_TAIL, helper_ns)
            projected = helper_ns["_gs_makeup_project_delta"](np.full((12, 3), 0.1))
            self.assertEqual(outside_support_max_abs(projected, mask, "dc"), 0.0)
            self.assertTrue(np.allclose(projected[0], 0.1))

            os.environ[HOOK_CONFIG_ENV] = str(Path(tmp) / "missing.json")
            clear_hook_state()
            untouched = helper_ns["_gs_makeup_project_delta"](np.full((12, 3), 0.1))
            self.assertTrue(np.all(untouched == 0.1))

            os.environ[HOOK_CONFIG_ENV] = installed["config"]
            clear_hook_state()
            model_ns: dict = {}
            exec(MODEL_TAIL, model_ns)
            attack = LinfProjectedGradientDescentAttack()
            bound = model_ns["_gs_makeup_bind_foolbox_attack"](attack)
            origin = np.zeros((1, 12, 1, 3))
            edited = bound.project(np.full_like(origin, 0.4), origin, 0.1)
            self.assertEqual(outside_support_max_abs(edited - origin, mask, "dc"), 0.0)
            self.assertTrue(np.allclose(edited[0, 0, 0], 0.1))
            self.assertTrue(np.all(edited[0, 11, 0] == 0.0))
            raw = epsilon_only_foolbox_project(np.full_like(origin, 0.4), origin, 0.1)
            self.assertGreater(outside_support_max_abs(raw - origin, mask, "dc"), 0.0)
            other = _OtherAttack()
            self.assertIs(bind_foolbox_attack(other), other)

    def test_moved_anchors_are_not_patched(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "AEGIS"
            _excerpt_tree(
                root,
                "def adaptive_linf_pgd_attack():\n    pass\n",
                "def _run_foolbox_attack(self):\n    pass\n",
            )
            before = (root / "src" / "aegis" / "core" / "helpers.py").read_text(
                encoding="utf-8"
            )
            report = install_makeup_pgd_hook(
                MakeupPgdHook(_fallback_mask(), 0.1, kind="dc"),
                root,
                mask_source="from_parsing",
            )
            self.assertEqual(report["status"], "anchors_mismatch")
            self.assertFalse(report["armed"])
            self.assertEqual(
                (root / "src" / "aegis" / "core" / "helpers.py").read_text(
                    encoding="utf-8"
                ),
                before,
            )
            self.assertFalse((root / ".gs_makeup_hook.json").exists())

    def test_cli_arm_hook_degrades_without_aegis(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "missing-aegis"
            proc = subprocess.run(
                [
                    sys.executable,
                    str(_SCRIPT),
                    "--results-dir",
                    tmp,
                    "--aegis-root",
                    str(root),
                    "--arm-hook",
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
            self.assertEqual(payload["aegis_hook"], "in_loop")
            self.assertEqual(payload["aegis_hook_attach"], "tree_missing")
            self.assertIn("nothing was cloned", payload["aegis_hook_attach_note"])
            self.assertEqual(payload["hooked_adaptive_outside_support_max_abs"], 0.0)
            self.assertEqual(payload["hooked_foolbox_outside_support_max_abs"], 0.0)
            self.assertGreater(payload["epsilon_only_adaptive_outside_support_max_abs"], 0.0)
            self.assertGreater(payload["epsilon_only_foolbox_outside_support_max_abs"], 0.0)
            self.assertFalse(root.exists())
            self.assertIn("mentor greenlights", payload["note"])
            self.assertNotIn("FaceScape", payload["command"])


if __name__ == "__main__":
    unittest.main()

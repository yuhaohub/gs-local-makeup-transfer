"""CPU tests for B2 wiring. No GPU, no AEGIS import, no dataset download."""

from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

_BASELINES = Path(__file__).resolve().parents[1]
_REPO = Path(__file__).resolve().parents[3]
if str(_BASELINES) not in sys.path:
    sys.path.insert(0, str(_BASELINES))
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import numpy as np

import b1_targeted_impersonation as b1
import b2_makeup_impersonation as b2
from makeup.aegis_pgd_hook import (
    ADAPTIVE_CLAMP_ANCHOR,
    FOOLBOX_ATTACK_ANCHOR,
    HOOK_CONFIG_ENV,
    clear_hook_state,
    dc_hook_from_parsing,
)


def _boom(*_args, **_kwargs):
    raise AssertionError("B2 launched a process")


def _labels() -> np.ndarray:
    """(N,) BiSeNet ids. The last three sites stay outside M."""
    return np.array([12, 13, 4, 5, 2, 3, 1, 0, 17, 10], dtype=np.int64)


def _write_parsing(directory: Path) -> Path:
    parsing = directory / "parsing" / "306.npy"
    parsing.parent.mkdir(parents=True, exist_ok=True)
    np.save(parsing, _labels())
    return parsing


def _excerpt_tree(root: Path) -> None:
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
    core = root / "src" / "aegis" / "core"
    core.mkdir(parents=True)
    (core / "helpers.py").write_text(helpers, encoding="utf-8")
    (core / "model.py").write_text(model, encoding="utf-8")


class PathsAndHyperparamsTest(unittest.TestCase):
    def tearDown(self) -> None:
        clear_hook_state()
        os.environ.pop(HOOK_CONFIG_ENV, None)
        os.environ.pop(b2.IMAGE_ENV, None)
        os.environ.pop(b2.PARSING_ENV, None)
        os.environ.pop("B2_MAKEUP_HOOK", None)
        os.environ.pop("B1_MAKEUP_HOOK", None)

    def test_frozen_paths_match_b1_and_parsing_contract(self) -> None:
        self.assertEqual(
            b2.DEFAULT_IMPERSONATION_IMAGE,
            "/root/autodl-fs/gs-assets/targets/celeba_T0.jpg",
        )
        self.assertEqual(b2.DEFAULT_IMPERSONATION_IMAGE, b1.DEFAULT_IMPERSONATION_IMAGE)
        self.assertEqual(
            b2.LFW_IMPERSONATION_IMAGE,
            "/root/autodl-fs/gs-assets/targets/lfw_T0.jpg",
        )
        self.assertEqual(b2.LFW_IMPERSONATION_IMAGE, b1.LFW_IMPERSONATION_IMAGE)
        self.assertEqual(
            b2.DEFAULT_PARSING,
            "/root/autodl-fs/gs-assets/parsing/306.npy",
        )
        self.assertNotIn("/masks/", b2.DEFAULT_PARSING)
        self.assertTrue(b2.DEFAULT_PARSING.endswith("/parsing/306.npy"))

    def test_env_does_not_inherit_b1_image_override(self) -> None:
        with patch.dict(
            os.environ,
            {
                "B1_IMPERSONATION_IMAGE": "/tmp/from-b1.jpg",
                b2.IMAGE_ENV: "/tmp/from-b2.jpg",
            },
        ):
            self.assertEqual(b2.resolve_impersonation_image(None), "/tmp/from-b2.jpg")
            self.assertEqual(
                b2.resolve_impersonation_image("/tmp/from-flag.jpg"),
                "/tmp/from-flag.jpg",
            )
        with patch.dict(os.environ, {"B1_IMPERSONATION_IMAGE": "/tmp/from-b1.jpg"}):
            os.environ.pop(b2.IMAGE_ENV, None)
            self.assertEqual(
                b2.resolve_impersonation_image(None),
                b2.DEFAULT_IMPERSONATION_IMAGE,
            )

    def test_parsing_env_and_flag(self) -> None:
        with patch.dict(os.environ, {b2.PARSING_ENV: "/tmp/parsing/306.npy"}):
            self.assertEqual(
                b2.resolve_parsing_path(None),
                Path("/tmp/parsing/306.npy"),
            )
            self.assertEqual(
                b2.resolve_parsing_path("/tmp/other/parsing/306.npy"),
                Path("/tmp/other/parsing/306.npy"),
            )
        os.environ.pop(b2.PARSING_ENV, None)
        self.assertEqual(b2.resolve_parsing_path(None), Path(b2.DEFAULT_PARSING))

    def test_summary_dirs_are_one_knife_each(self) -> None:
        celeba = b2.default_summary_dir(Path(b2.DEFAULT_IMPERSONATION_IMAGE))
        lfw = b2.default_summary_dir(Path(b2.LFW_IMPERSONATION_IMAGE))
        self.assertEqual(celeba, b2.RESULTS_DIR / "b2_celeba_T0")
        self.assertEqual(lfw, b2.RESULTS_DIR / "b2_lfw_T0")
        self.assertNotEqual(celeba, lfw)
        self.assertNotEqual(celeba, b2.RESULTS_DIR)
        self.assertEqual(b2.summary_slot(Path("celeba_T0.jpg")), "b2_celeba_T0")

    def test_flat_results_and_historical_names_are_refused(self) -> None:
        with self.assertRaises(b2.SummarySlotError):
            b2.assert_summary_slot(b2.RESULTS_DIR)
        with self.assertRaises(b2.SummarySlotError):
            b2.assert_summary_slot(b2.RESULTS_DIR / "b1_targeted_impersonation_summary")
        with self.assertRaises(b2.SummarySlotError):
            b2.assert_summary_slot(b2.RESULTS_DIR / "b0_aegis_sanity_summary.json")
        b2.assert_summary_slot(b2.RESULTS_DIR / "b2_celeba_T0")

    def test_masks_directory_and_wrong_basename_are_refused(self) -> None:
        with self.assertRaises(b2.MaskPathError):
            b2.assert_parsing_path(Path("/root/autodl-fs/gs-assets/masks/306.npy"))
        with self.assertRaises(b2.MaskPathError):
            b2.assert_parsing_path(Path("/tmp/parsing/not-306.npy"))
        b2.assert_parsing_path(Path("/tmp/parsing/306.npy"))
        b2.assert_parsing_path(Path(b2.DEFAULT_PARSING))

    def test_plan_keeps_b1_hyperparams_and_b1_stays_unmasked(self) -> None:
        plan = b1.build_plan(b2.DEFAULT_IMPERSONATION_IMAGE, launch=False)
        self.assertEqual(plan.subject, "306")
        self.assertEqual(plan.embedder, "adaface")
        self.assertEqual(plan.epsilon, 0.1)
        self.assertEqual(plan.attack_steps, 300)
        self.assertEqual(plan.target_features, "DC")
        self.assertEqual(plan.adv_attack, "linfpgd")
        self.assertFalse(plan.adaptive_epsilon)
        self.assertEqual(plan.selected_regions, ())
        self.assertFalse(plan.arm_makeup_hook)
        kwargs = b2.b2_aegis_kwargs(plan)
        self.assertEqual(kwargs["embedder_name"], "adaface")
        self.assertEqual(kwargs["epsilons"], [0.1])
        self.assertEqual(kwargs["attack_steps"], 300)
        self.assertEqual(kwargs["targeted_features"], "DC")
        self.assertEqual(kwargs["adv_attack"], "linfpgd")
        self.assertFalse(kwargs["adaptive_epsilon"])
        self.assertEqual(kwargs["selected_regions"], [])
        self.assertEqual(kwargs["output_name"], "NeRSembleMakeup306_celeba_T0")
        self.assertNotEqual(kwargs["output_name"], plan.output_name)
        self.assertFalse(b1.b1_criterion_spec()["makeup_hook"])
        spec = b2.b2_criterion_spec()
        self.assertEqual(spec["foolbox_criterion"], "TargetedMisclassification")
        self.assertEqual(spec["target_class"], 1)
        self.assertEqual(spec["hook_kind"], "dc")
        self.assertTrue(spec["makeup_hook"])
        self.assertFalse(b1.b1_criterion_spec()["makeup_hook"])
        self.assertEqual(
            b1.similarity_step_sign(spec["foolbox_criterion"], spec["target_class"]),
            1,
        )

    def test_cli_defaults_do_not_launch_or_arm(self) -> None:
        args = b2.parse_args([])
        self.assertFalse(args.launch)
        self.assertFalse(args.arm_hook)
        self.assertFalse(args.refuse_hook)
        self.assertFalse(args.adaptive_epsilon)
        self.assertFalse(args.select_regions)
        self.assertIsNone(args.summary_dir)
        self.assertEqual(args.subject, "306")
        flagged = b2.parse_args(
            [
                "--victim-image",
                "/root/autodl-fs/gs-assets/targets/lfw_T0.jpg",
                "--parsing",
                "/root/autodl-fs/gs-assets/parsing/306.npy",
            ]
        )
        self.assertEqual(
            flagged.impersonation_image,
            "/root/autodl-fs/gs-assets/targets/lfw_T0.jpg",
        )
        self.assertFalse(flagged.launch)


class OutsideMaskTest(unittest.TestCase):
    def test_projection_zeros_outside_m_and_raw_residual_does_not(self) -> None:
        hook = dc_hook_from_parsing(_labels(), 0.1, subject="306")
        self.assertEqual(hook.kind, "dc")
        self.assertEqual(tuple(np.asarray(hook.mask.union()).shape), (10,))
        clean = np.zeros((10, 1, 3), dtype=float)
        raw = np.full((10, 1, 3), 0.25, dtype=float)
        projected = np.asarray(hook.project_like(raw), dtype=float)
        self.assertEqual(b2.outside_m_abs_delta(clean, projected, hook.mask), 0.0)
        self.assertGreater(b2.outside_m_abs_delta(clean, raw, hook.mask), 0.0)
        self.assertLessEqual(float(np.max(np.abs(projected))), 0.1 + 1e-8)

    def test_gate_requires_all_four_measurements(self) -> None:
        base = {
            "return_code": 0,
            "cos_adv_source": 0.2,
            "cos_adv_target": 0.8,
            "outside_m_abs_delta": 0.0,
        }
        self.assertTrue(b2.gate_holds(base))
        for key in ("cos_adv_source", "cos_adv_target", "outside_m_abs_delta"):
            missing = dict(base)
            missing[key] = None
            self.assertFalse(b2.gate_holds(missing))
        leaked = dict(base)
        leaked["outside_m_abs_delta"] = 0.1
        self.assertFalse(b2.gate_holds(leaked))
        failed = dict(base)
        failed["return_code"] = 1
        self.assertFalse(b2.gate_holds(failed))


class CliRefusalTest(unittest.TestCase):
    def tearDown(self) -> None:
        clear_hook_state()
        os.environ.pop(HOOK_CONFIG_ENV, None)
        os.environ.pop(b2.IMAGE_ENV, None)
        os.environ.pop(b2.PARSING_ENV, None)

    def _run(self, extra: list[str]) -> tuple[int, dict]:
        with tempfile.TemporaryDirectory() as tmp:
            code = b2.main(["--summary-dir", tmp, *extra], runner=_boom)
            payload = json.loads((Path(tmp) / "summary.json").read_text(encoding="utf-8"))
            csv_text = (Path(tmp) / "summary.csv").read_text(encoding="utf-8")
        self.assertIn("outside_m_abs_delta", csv_text)
        self.assertIn("pipeline_ok", csv_text)
        return code, payload

    def test_default_main_does_not_launch_and_writes_b2_schema(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with patch.dict(os.environ, {}, clear=False):
                os.environ.pop(b2.IMAGE_ENV, None)
                os.environ.pop(b2.PARSING_ENV, None)
                code = b2.main(["--summary-dir", tmp], runner=_boom)
            payload = json.loads((Path(tmp) / "summary.json").read_text(encoding="utf-8"))
            csv_text = (Path(tmp) / "summary.csv").read_text(encoding="utf-8")
            self.assertFalse((Path(tmp) / "b1_targeted_impersonation_summary.json").exists())
            self.assertFalse((Path(tmp) / "b0_aegis_sanity_summary.json").exists())
        self.assertEqual(code, 0)
        self.assertEqual(payload["experiment_id"], "B2")
        self.assertEqual(payload["subject"], "306")
        self.assertFalse(payload["pipeline_ok"])
        self.assertEqual(payload["status"], "awaiting_target")
        self.assertEqual(payload["summary_slot"], "b2_celeba_T0")
        self.assertEqual(payload["parsing_path"], b2.DEFAULT_PARSING)
        metrics = payload["metrics"]
        self.assertIsNone(metrics["cos_adv_source"])
        self.assertIsNone(metrics["cos_adv_target"])
        self.assertIsNone(metrics["outside_m_abs_delta"])
        self.assertEqual(metrics["outside_m_abs_delta_gate"], 0)
        self.assertFalse(metrics["makeup_hook_armed"])
        self.assertFalse(metrics["command_attempted"])
        self.assertEqual(metrics["hook"]["kind"], "dc")
        self.assertEqual(metrics["hook"]["attach_status"], "not_requested")
        self.assertEqual(metrics["epsilon"], 0.1)
        self.assertEqual(metrics["attack_steps"], 300)
        self.assertEqual(metrics["target_features"], "DC")
        self.assertEqual(metrics["adv_attack"], "linfpgd")
        self.assertFalse(metrics["adaptive_epsilon"])
        self.assertEqual(
            metrics["impersonation_image"],
            "/root/autodl-fs/gs-assets/targets/celeba_T0.jpg",
        )
        self.assertEqual(metrics["criterion"]["foolbox_criterion"], "TargetedMisclassification")
        self.assertTrue(metrics["criterion"]["makeup_hook"])
        self.assertIn("--arm-hook --launch", metrics["command"])
        self.assertNotIn("masks/", metrics["parsing_path"])
        self.assertIn("outside_m_abs_delta", csv_text)
        self.assertIn("kind=dc", payload["note"])

    def test_flat_results_dir_does_not_write(self) -> None:
        historical = _REPO / "results" / "b1_targeted_impersonation_summary.json"
        before = historical.read_bytes()
        code = b2.main(["--summary-dir", "results"], runner=_boom)
        self.assertEqual(code, 2)
        self.assertEqual(historical.read_bytes(), before)
        self.assertFalse((_REPO / "results" / "summary.json").exists())
        code = b2.main(
            ["--summary-dir", "results/b1_targeted_impersonation_summary"],
            runner=_boom,
        )
        self.assertEqual(code, 2)
        self.assertEqual(historical.read_bytes(), before)
        self.assertFalse(
            (_REPO / "results" / "b1_targeted_impersonation_summary" / "summary.json").exists()
        )

    def test_launch_without_arm_hook_is_refused(self) -> None:
        code, payload = self._run(["--launch"])
        self.assertEqual(code, 2)
        self.assertEqual(payload["status"], "refused_unarmed_launch")
        self.assertFalse(payload["metrics"]["command_attempted"])
        self.assertFalse(payload["metrics"]["makeup_hook_armed"])
        self.assertIsNone(payload["metrics"]["outside_m_abs_delta"])

    def test_refuse_hook_and_conflict_are_refused(self) -> None:
        code, payload = self._run(["--refuse-hook"])
        self.assertEqual(code, 2)
        self.assertEqual(payload["status"], "refused_makeup_hook")
        self.assertFalse(payload["metrics"]["command_attempted"])
        code, payload = self._run(["--arm-hook", "--refuse-hook", "--launch"])
        self.assertEqual(code, 2)
        self.assertEqual(payload["status"], "refused_conflicting_hook")
        self.assertFalse(payload["metrics"]["makeup_hook_armed"])

    def test_masks_path_and_bad_subject_and_hyperparams_are_refused(self) -> None:
        code, payload = self._run(
            ["--parsing", "/root/autodl-fs/gs-assets/masks/306.npy", "--arm-hook"]
        )
        self.assertEqual(code, 2)
        self.assertEqual(payload["status"], "refused_mask_path")
        code, payload = self._run(["--subject", "527"])
        self.assertEqual(code, 2)
        self.assertEqual(payload["status"], "refused_subject")
        code, payload = self._run(["--adaptive-epsilon", "--launch", "--arm-hook"])
        self.assertEqual(code, 2)
        self.assertEqual(payload["status"], "refused_hyperparams")
        code, payload = self._run(["--select-regions"])
        self.assertEqual(code, 2)
        self.assertEqual(payload["status"], "refused_hyperparams")

    def test_launch_refuses_306_render_as_target(self) -> None:
        code, payload = self._run(
            [
                "--launch",
                "--arm-hook",
                "--impersonation-image",
                str(b1.default_source_image()),
            ]
        )
        self.assertEqual(code, 2)
        self.assertEqual(payload["status"], "refused_source_image")
        self.assertFalse(payload["metrics"]["command_attempted"])

    def test_arm_hook_without_parsing_does_not_fallback_or_launch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            missing = Path(tmp) / "parsing" / "306.npy"
            with patch.object(b2, "install_b2_hook", side_effect=AssertionError("installed")):
                code = b2.main(
                    [
                        "--arm-hook",
                        "--parsing",
                        str(missing),
                        "--summary-dir",
                        tmp,
                    ],
                    runner=_boom,
                )
            payload = json.loads((Path(tmp) / "summary.json").read_text(encoding="utf-8"))
        self.assertEqual(code, 2)
        self.assertEqual(payload["status"], "awaiting_parsing")
        self.assertFalse(payload["metrics"]["makeup_hook_armed"])
        self.assertIn("No documented_fallback labels were built", payload["note"])
        self.assertIsNone(payload["metrics"]["hook"]["mask_shape"])

    def test_arm_hook_on_missing_tree_records_dc_spec_and_does_not_launch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            parsing = _write_parsing(root)
            summary = root / "summary"
            code = b2.main(
                [
                    "--arm-hook",
                    "--parsing",
                    str(parsing),
                    "--aegis-root",
                    str(root / "no-aegis"),
                    "--summary-dir",
                    str(summary),
                    "--impersonation-image",
                    str(root / "celeba_T0.jpg"),
                ],
                runner=_boom,
            )
            payload = json.loads((summary / "summary.json").read_text(encoding="utf-8"))
        self.assertEqual(code, 0)
        self.assertEqual(payload["status"], "tree_missing")
        self.assertFalse(payload["pipeline_ok"])
        self.assertFalse(payload["metrics"]["makeup_hook_armed"])
        self.assertFalse(payload["metrics"]["command_attempted"])
        self.assertIsNone(payload["metrics"]["cos_adv_source"])
        self.assertIsNone(payload["metrics"]["outside_m_abs_delta"])
        hook = payload["metrics"]["hook"]
        self.assertEqual(hook["kind"], "dc")
        self.assertEqual(hook["subject"], "306")
        self.assertEqual(hook["epsilon"], 0.1)
        self.assertEqual(hook["mask_shape"], [10])
        self.assertEqual(hook["attach_status"], "tree_missing")
        self.assertFalse(hook["armed"])

    def test_arm_hook_writes_dc_config_and_does_not_launch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            parsing = _write_parsing(root)
            aegis = root / "AEGIS"
            _excerpt_tree(aegis)
            summary = root / "out"
            code = b2.main(
                [
                    "--arm-hook",
                    "--parsing",
                    str(parsing),
                    "--aegis-root",
                    str(aegis),
                    "--summary-dir",
                    str(summary),
                ],
                runner=_boom,
            )
            payload = json.loads((summary / "summary.json").read_text(encoding="utf-8"))
            config = json.loads((aegis / ".gs_makeup_hook.json").read_text(encoding="utf-8"))
        self.assertEqual(code, 0)
        self.assertEqual(payload["status"], "hook_armed")
        self.assertFalse(payload["pipeline_ok"])
        self.assertTrue(payload["metrics"]["makeup_hook_armed"])
        self.assertFalse(payload["metrics"]["command_attempted"])
        self.assertIsNone(payload["metrics"]["outside_m_abs_delta"])
        self.assertEqual(config["kind"], "dc")
        self.assertEqual(config["subject"], "306")
        self.assertEqual(config["epsilon"], 0.1)
        self.assertEqual(config["mask_shape"], [10])
        self.assertEqual(config["mask_domain"], "dc")
        self.assertEqual(payload["metrics"]["hook"]["mask_shape"], [10])

    def test_launch_with_files_but_no_aegis_does_not_start(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            parsing = _write_parsing(root)
            image = root / "celeba_T0.jpg"
            image.write_bytes(b"\xff\xd8\xff not-a-download")
            code = b2.main(
                [
                    "--arm-hook",
                    "--launch",
                    "--impersonation-image",
                    str(image),
                    "--parsing",
                    str(parsing),
                    "--aegis-root",
                    str(root / "missing-tree"),
                    "--summary-dir",
                    str(root / "out"),
                ],
                runner=_boom,
            )
            payload = json.loads((root / "out" / "summary.json").read_text(encoding="utf-8"))
        self.assertEqual(code, 2)
        self.assertIn(
            payload["status"],
            {"bootstrap_required", "deps_missing", "gpu_required", "hook_not_armed", "tree_missing"},
        )
        self.assertFalse(payload["metrics"]["command_attempted"])
        self.assertFalse(payload["pipeline_ok"])
        self.assertIsNone(payload["metrics"]["cos_adv_target"])

    def test_armed_launch_reaches_worker_only_with_dc_config(self) -> None:
        seen = {}

        def runner(command, **kwargs):
            seen["command"] = command
            seen["env"] = kwargs["env"]
            seen["cwd"] = kwargs["cwd"]
            out = Path(command[command.index("--summary-out") + 1])
            out.write_text(
                json.dumps(
                    {
                        "ok": True,
                        "cos_adv_source": 0.21,
                        "cos_adv_target": 0.83,
                        "outside_m_abs_delta": 0.0,
                        "gpu": "FakeGPU",
                    }
                ),
                encoding="utf-8",
            )

            class Proc:
                returncode = 0
                stdout = "Epsilon: 0.100 -> Aggregated Cosine Similarity: 0.8300\n"
                stderr = ""

            return Proc()

        def fake_install(hook, aegis_root):
            config = Path(aegis_root) / ".gs_makeup_hook.json"
            config.parent.mkdir(parents=True, exist_ok=True)
            config.write_text(
                json.dumps({"kind": "dc", "subject": "306", "epsilon": 0.1}),
                encoding="utf-8",
            )
            return {
                "status": "patched",
                "armed": True,
                "config": str(config),
                "note": "test install",
            }

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            parsing = _write_parsing(root)
            image = root / "celeba_T0.jpg"
            image.write_bytes(b"still")
            aegis = root / "AEGIS"
            aegis.mkdir()
            with (
                patch.object(b2, "install_b2_hook", side_effect=fake_install),
                patch.object(b2, "collect_b2_blockers", return_value=[]),
            ):
                code = b2.main(
                    [
                        "--arm-hook",
                        "--launch",
                        "--impersonation-image",
                        str(image),
                        "--parsing",
                        str(parsing),
                        "--aegis-root",
                        str(aegis),
                        "--summary-dir",
                        str(root / "out"),
                    ],
                    runner=runner,
                )
            payload = json.loads((root / "out" / "summary.json").read_text(encoding="utf-8"))
            self.assertIn("--arm-hook", seen["command"])
            self.assertNotIn("--refuse-hook", seen["command"])
            self.assertNotIn("--adaptive-epsilon", seen["command"])
            b2.assert_makeup_armed(seen["env"])
            self.assertEqual(seen["env"]["B2_MAKEUP_HOOK"], "on")
            self.assertNotEqual(seen["env"].get("B1_MAKEUP_HOOK"), "off")
            self.assertEqual(seen["cwd"], str(aegis))
        self.assertEqual(code, 0)
        self.assertTrue(payload["pipeline_ok"])
        self.assertEqual(payload["status"], "pipeline_ok")
        self.assertEqual(payload["metrics"]["cos_adv_source"], 0.21)
        self.assertEqual(payload["metrics"]["cos_adv_target"], 0.83)
        self.assertEqual(payload["metrics"]["outside_m_abs_delta"], 0.0)
        self.assertTrue(payload["metrics"]["command_attempted"])

    def test_nonzero_outside_m_keeps_pipeline_ok_false(self) -> None:
        def runner(command, **kwargs):
            out = Path(command[command.index("--summary-out") + 1])
            out.write_text(
                json.dumps(
                    {
                        "cos_adv_source": 0.2,
                        "cos_adv_target": 0.4,
                        "outside_m_abs_delta": 0.05,
                    }
                ),
                encoding="utf-8",
            )

            class Proc:
                returncode = 0
                stdout = ""
                stderr = ""

            return Proc()

        def fake_install(hook, aegis_root):
            config = Path(aegis_root) / ".gs_makeup_hook.json"
            config.parent.mkdir(parents=True, exist_ok=True)
            config.write_text(
                json.dumps({"kind": "dc", "subject": "306", "epsilon": 0.1}),
                encoding="utf-8",
            )
            return {"status": "patched", "armed": True, "config": str(config), "note": "test"}

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            parsing = _write_parsing(root)
            image = root / "celeba_T0.jpg"
            image.write_bytes(b"still")
            with (
                patch.object(b2, "install_b2_hook", side_effect=fake_install),
                patch.object(b2, "collect_b2_blockers", return_value=[]),
            ):
                code = b2.main(
                    [
                        "--arm-hook",
                        "--launch",
                        "--impersonation-image",
                        str(image),
                        "--parsing",
                        str(parsing),
                        "--aegis-root",
                        str(root / "AEGIS"),
                        "--summary-dir",
                        str(root / "out"),
                    ],
                    runner=runner,
                )
            payload = json.loads((root / "out" / "summary.json").read_text(encoding="utf-8"))
        self.assertEqual(code, 1)
        self.assertEqual(payload["status"], "support_failed")
        self.assertFalse(payload["pipeline_ok"])
        self.assertEqual(payload["metrics"]["outside_m_abs_delta"], 0.05)

    def test_worker_command_arms_dc_hook(self) -> None:
        plan = b1.build_plan(b2.DEFAULT_IMPERSONATION_IMAGE, launch=True)
        command = b2.worker_command(
            plan,
            Path(b2.DEFAULT_PARSING),
            Path("/tmp/b2-worker.json"),
            b1.AEGIS_ROOT,
        )
        text = " ".join(command)
        self.assertIn("--worker", command)
        self.assertIn("--arm-hook", command)
        self.assertIn(b2.DEFAULT_PARSING, command)
        self.assertIn(b2.DEFAULT_IMPERSONATION_IMAGE, command)
        self.assertNotIn("--refuse-hook", command)
        self.assertNotIn("adaptive-epsilon", text)
        self.assertNotIn("select-regions", text)
        self.assertNotIn("masks/", text)

    def test_disarmed_b1_env_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config = Path(tmp) / ".gs_makeup_hook.json"
            config.write_text(
                json.dumps({"kind": "dc", "subject": "306", "epsilon": 0.1}),
                encoding="utf-8",
            )
            env = b2.armed_environment(config, Path(tmp), {})
            b2.assert_makeup_armed(env)
            self.assertNotIn("B1_MAKEUP_HOOK", env)
            poisoned = dict(env)
            poisoned["B1_MAKEUP_HOOK"] = "off"
            with self.assertRaises(b2.MakeupHookRequired):
                b2.assert_makeup_armed(poisoned)
            rgb = dict(env)
            rgb[HOOK_CONFIG_ENV] = str(config)
            config.write_text(
                json.dumps({"kind": "rgb", "subject": "306", "epsilon": 0.1}),
                encoding="utf-8",
            )
            with self.assertRaises(b2.MakeupHookRequired):
                b2.assert_makeup_armed(rgb)

    def test_b1_module_still_refuses_its_own_hook(self) -> None:
        self.assertIn("def refuse_makeup_hook", b1.MODULE_PATH.read_text(encoding="utf-8"))
        self.assertNotIn("b2_makeup_impersonation", b1.MODULE_PATH.read_text(encoding="utf-8"))
        with tempfile.TemporaryDirectory() as tmp:
            code = b1.main(["--arm-hook", "--launch", "--summary-dir", tmp], runner=_boom)
            payload = json.loads(
                (Path(tmp) / "b1_targeted_impersonation_summary.json").read_text(encoding="utf-8")
            )
        self.assertEqual(code, 2)
        self.assertEqual(payload["status"], "refused_makeup_hook")
        self.assertFalse(payload["metrics"]["makeup_hook_armed"])


class DocsTest(unittest.TestCase):
    def test_readme_documents_b2_cli_without_default_launch(self) -> None:
        text = (_REPO / "README.md").read_text(encoding="utf-8")
        b2_at = text.find("## B2")
        self.assertGreater(b2_at, text.find("## B1"))
        section = text[b2_at:]
        self.assertIn("python src/baselines/b2_makeup_impersonation.py", section)
        self.assertIn("--arm-hook", section)
        self.assertIn("--refuse-hook", section)
        self.assertIn("--launch", section)
        self.assertIn("/root/autodl-fs/gs-assets/targets/celeba_T0.jpg", section)
        self.assertIn("/root/autodl-fs/gs-assets/targets/lfw_T0.jpg", section)
        self.assertIn("/root/autodl-fs/gs-assets/parsing/306.npy", section)
        self.assertIn("results/b2_celeba_T0/", section)
        self.assertIn("results/b2_lfw_T0/", section)
        self.assertIn("outside_m_abs_delta", section)
        self.assertIn("kind=dc", section)
        self.assertIn("默认不加 `--launch`", section)
        matrix = text[text.find("## 实验矩阵") : text.find("## 协作约定")]
        self.assertIn("B2", matrix)

    def test_manifest_stub_reuses_frozen_names(self) -> None:
        text = (_REPO / "assets" / "MANIFEST.md").read_text(encoding="utf-8")
        self.assertIn("/root/autodl-fs/gs-assets/parsing/306.npy", text)
        self.assertIn("results/b2_celeba_T0/", text)
        self.assertIn("results/b2_lfw_T0/", text)
        self.assertNotIn("/root/autodl-fs/gs-assets/masks/306.npy", text)
        stub = text[text.find("## B2 stub") :]
        self.assertIn("celeba_T0.jpg", stub)
        self.assertIn("lfw_T0.jpg", stub)
        self.assertIn("parsing/306.npy", stub)
        self.assertIn("does not authorize", stub.lower())

    def test_checklist_names_b2_slot_without_enabling_launch(self) -> None:
        text = (_REPO / "docs" / "gpu_launch_checklist.md").read_text(encoding="utf-8")
        self.assertIn("does not enable B2", text)
        self.assertIn("results/b2_celeba_T0/", text)
        self.assertIn("results/b2_lfw_T0/", text)
        self.assertIn("parsing/306.npy", text)


if __name__ == "__main__":
    unittest.main()

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
        self.assertEqual(
            b2.default_summary_dir(Path(b2.DEFAULT_IMPERSONATION_IMAGE), "ablate_noskin"),
            b2.RESULTS_DIR / "b2_celeba_T0_ablate_noskin",
        )
        self.assertEqual(
            b2.default_summary_dir(Path(b2.LFW_IMPERSONATION_IMAGE), "ablate_noskin"),
            b2.RESULTS_DIR / "b2_lfw_T0_ablate_noskin",
        )
        self.assertNotEqual(
            b2.default_summary_dir(Path(b2.DEFAULT_IMPERSONATION_IMAGE), "ablate_noskin"),
            celeba,
        )
        self.assertEqual(
            b2.default_summary_dir(Path(b2.DEFAULT_IMPERSONATION_IMAGE), "ablate_onlyskin"),
            b2.RESULTS_DIR / "b2_celeba_T0_ablate_onlyskin",
        )
        self.assertEqual(
            b2.default_summary_dir(Path(b2.LFW_IMPERSONATION_IMAGE), "ablate_onlyskin"),
            b2.RESULTS_DIR / "b2_lfw_T0_ablate_onlyskin",
        )
        self.assertNotEqual(
            b2.default_summary_dir(Path(b2.DEFAULT_IMPERSONATION_IMAGE), "ablate_onlyskin"),
            b2.default_summary_dir(Path(b2.DEFAULT_IMPERSONATION_IMAGE), "ablate_noskin"),
        )

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
        self.assertIsNone(args.ablate)
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

    def test_outside_m_aligns_att_tensor_with_foolbox_batch(self) -> None:
        """Reproduce the AutoDL crash: (89021, 1, 3) vs (1, 89021, 1, 3).

        Clean DC stays ``att_tensor`` ``(N, 1, 3)``. Foolbox saves
        ``unsqueeze(0)`` as ``(1, N, 1, 3)``. N=89021 is the 306 count
        from that run. Outside-M |δ| follows Gaussian index after both
        sides are squeezed to ``(N, 3)``.
        """
        n = 89021
        inside = 7
        outside = n - 1
        mask = np.zeros((n,), dtype=float)
        mask[inside] = 1.0
        clean = np.zeros((n, 1, 3), dtype=float)
        adv = np.zeros((1, n, 1, 3), dtype=float)
        clean[inside, 0, 0] = 0.5
        adv[0, inside, 0, 0] = 1.0
        clean[outside, 0, 2] = 0.125
        adv[0, outside, 0, 2] = 0.375
        self.assertEqual(clean.shape, (89021, 1, 3))
        self.assertEqual(adv.shape, (1, 89021, 1, 3))
        self.assertNotEqual(clean.shape, adv.shape)
        self.assertAlmostEqual(b2.outside_m_abs_delta(clean, adv, mask), 0.25)

        matched = adv.copy()
        matched[0, outside, 0, 2] = clean[outside, 0, 2]
        self.assertEqual(b2.outside_m_abs_delta(clean, matched, mask), 0.0)

        clean_n3 = clean[:, 0, :]
        adv_n13 = adv[0]
        self.assertEqual(clean_n3.shape, (n, 3))
        self.assertEqual(adv_n13.shape, (n, 1, 3))
        self.assertAlmostEqual(b2.outside_m_abs_delta(clean_n3, adv, mask), 0.25)
        self.assertAlmostEqual(b2.outside_m_abs_delta(clean, adv_n13, mask), 0.25)
        self.assertAlmostEqual(b2.outside_m_abs_delta(clean_n3, adv_n13, mask), 0.25)

        with self.assertRaises(ValueError) as caught:
            b2.outside_m_abs_delta(clean, np.zeros((1, n - 1, 1, 3)), mask)
        message = str(caught.exception)
        self.assertIn("clean and adversarial DC shapes differ", message)
        self.assertIn("(89021, 1, 3)", message)
        self.assertIn("(1, 89020, 1, 3)", message)

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
        self.assertEqual(payload["mask_labels"], "full")
        self.assertEqual(payload["parsing_label_ids"], [1, 2, 3, 4, 5, 12, 13])
        self.assertFalse(payload["ablation_changes_default_m"])
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
        self.assertNotIn("--ablate", metrics["command"])
        self.assertNotIn("masks/", metrics["parsing_path"])
        self.assertIn("mask_labels", csv_text)
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


class AblateNoskinTest(unittest.TestCase):
    def tearDown(self) -> None:
        clear_hook_state()
        os.environ.pop(HOOK_CONFIG_ENV, None)
        os.environ.pop(b2.IMAGE_ENV, None)
        os.environ.pop(b2.PARSING_ENV, None)

    def test_default_map_is_unchanged_and_noskin_drops_only_label_1(self) -> None:
        from makeup.mask import DEFAULT_PARSING_LABELS, MakeupRegion

        before = dict(DEFAULT_PARSING_LABELS)
        mode, mapping = b2.resolve_mask_labels(None)
        self.assertEqual(mode, "full")
        self.assertIsNone(mapping)
        mode, mapping = b2.resolve_mask_labels("full")
        self.assertEqual(mode, "full")
        self.assertIsNone(mapping)
        mode, mapping = b2.resolve_mask_labels("default")
        self.assertEqual(mode, "full")
        self.assertIsNone(mapping)
        self.assertEqual(dict(DEFAULT_PARSING_LABELS), before)
        self.assertEqual(set(DEFAULT_PARSING_LABELS), {1, 2, 3, 4, 5, 12, 13})
        self.assertEqual(DEFAULT_PARSING_LABELS[1], MakeupRegion.CHEEK)
        for excluded in (0, 6, 7, 8, 10, 11, 14, 17):
            self.assertNotIn(excluded, DEFAULT_PARSING_LABELS)

        mode, mapping = b2.resolve_mask_labels("ablate_noskin")
        self.assertEqual(mode, "ablate_noskin")
        self.assertEqual(set(mapping), {2, 3, 4, 5, 12, 13})
        self.assertNotIn(1, mapping)
        self.assertTrue(
            all(region != MakeupRegion.CHEEK for region in mapping.values())
        )
        self.assertEqual(dict(DEFAULT_PARSING_LABELS), before)
        self.assertIn(1, DEFAULT_PARSING_LABELS)
        for rejected in ("no_skin", "lips_eyes_brows", "makeup_points", "nose"):
            with self.assertRaises(b2.MaskLabelError):
                b2.resolve_mask_labels(rejected)

    def test_hook_default_keeps_skin_and_noskin_drops_it(self) -> None:
        labels = _labels()
        full = dc_hook_from_parsing(labels, 0.1, subject="306")
        same = dc_hook_from_parsing(labels, 0.1, subject="306", label_map=None)
        _mode, mapping = b2.resolve_mask_labels("ablate_noskin")
        dropped = dc_hook_from_parsing(labels, 0.1, subject="306", label_map=mapping)
        full_u = np.asarray(full.mask.union())
        drop_u = np.asarray(dropped.mask.union())
        self.assertTrue(np.array_equal(np.asarray(same.mask.union()), full_u))
        self.assertEqual(full_u[6], 1.0)
        self.assertEqual(drop_u[6], 0.0)
        self.assertTrue(np.array_equal(full_u[:6], np.ones(6)))
        self.assertTrue(np.array_equal(drop_u[:6], np.ones(6)))
        self.assertTrue(np.array_equal(full_u[7:], np.zeros(3)))
        self.assertTrue(np.array_equal(drop_u[7:], np.zeros(3)))
        self.assertEqual(dropped.kind, "dc")

        grid = np.zeros((2, 4), dtype=np.int64)
        grid[0, 0] = 1
        grid[0, 1] = 12
        grid[0, 2] = 10
        full_image = dc_hook_from_parsing(grid, 0.1, means=None)
        narrow_image = dc_hook_from_parsing(grid, 0.1, means=None, label_map=mapping)
        self.assertEqual(np.asarray(full_image.image_mask)[0, 0], 1.0)
        self.assertEqual(np.asarray(narrow_image.image_mask)[0, 0], 0.0)
        self.assertEqual(np.asarray(narrow_image.image_mask)[0, 1], 1.0)
        self.assertEqual(np.asarray(narrow_image.image_mask)[0, 2], 0.0)
        self.assertEqual(full_image.dc_mask_lift, "pending-get-xyz")

    def test_summary_slot_uses_ablate_noskin_suffix_and_spares_the_product_dir(self) -> None:
        image = Path(b2.DEFAULT_IMPERSONATION_IMAGE)
        self.assertEqual(b2.summary_slot(image), "b2_celeba_T0")
        self.assertEqual(b2.summary_slot(image, "ablate_noskin"), "b2_celeba_T0_ablate_noskin")
        self.assertEqual(
            b2.user_command(image, Path(b2.DEFAULT_PARSING)),
            "python src/baselines/b2_makeup_impersonation.py "
            "--impersonation-image /root/autodl-fs/gs-assets/targets/celeba_T0.jpg "
            "--parsing /root/autodl-fs/gs-assets/parsing/306.npy "
            "--arm-hook --launch",
        )
        noskin_command = b2.user_command(image, Path(b2.DEFAULT_PARSING), "ablate_noskin")
        self.assertIn("--ablate ablate_noskin --arm-hook --launch", noskin_command)
        plan = b1.build_plan(b2.DEFAULT_IMPERSONATION_IMAGE, launch=True)
        default_worker = b2.worker_command(
            plan,
            Path(b2.DEFAULT_PARSING),
            Path("/tmp/b2-worker.json"),
            b1.AEGIS_ROOT,
        )
        self.assertNotIn("--ablate", default_worker)
        noskin_worker = b2.worker_command(
            plan,
            Path(b2.DEFAULT_PARSING),
            Path("/tmp/b2-worker.json"),
            b1.AEGIS_ROOT,
            "ablate_noskin",
        )
        self.assertEqual(noskin_worker[noskin_worker.index("--ablate") + 1], "ablate_noskin")

        product = b2.RESULTS_DIR / "b2_celeba_T0"
        json_path = product / "summary.json"
        csv_path = product / "summary.csv"
        before_json = json_path.read_bytes()
        before_csv = csv_path.read_bytes()
        code = b2.main(
            ["--ablate", "ablate_noskin", "--summary-dir", str(product)],
            runner=_boom,
        )
        self.assertEqual(code, 2)
        self.assertEqual(json_path.read_bytes(), before_json)
        self.assertEqual(csv_path.read_bytes(), before_csv)
        self.assertFalse((b2.RESULTS_DIR / "b2_celeba_T0_ablate_noskin").exists())
        with self.assertRaises(SystemExit) as caught:
            b2.parse_args(["--ablate", "no_skin"])
        self.assertEqual(caught.exception.code, 2)

    def test_dry_run_writes_the_ablation_slot_and_leaves_full_m_alone(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with patch.object(b2, "RESULTS_DIR", root):
                code = b2.main(["--mask-labels", "ablate_noskin"], runner=_boom)
                payload = json.loads(
                    (root / "b2_celeba_T0_ablate_noskin" / "summary.json").read_text(
                        encoding="utf-8"
                    )
                )
                self.assertFalse((root / "b2_celeba_T0").exists())
            self.assertEqual(code, 0)
            self.assertFalse(payload["metrics"]["command_attempted"])
            self.assertFalse(payload["pipeline_ok"])
            self.assertEqual(payload["summary_slot"], "b2_celeba_T0_ablate_noskin")
            self.assertEqual(payload["mask_labels"], "ablate_noskin")
            self.assertEqual(payload["metrics"]["mask_labels_requested"], "ablate_noskin")
            self.assertEqual(payload["parsing_label_ids"], [2, 3, 4, 5, 12, 13])
            self.assertFalse(payload["ablation_changes_default_m"])
            self.assertIn("--ablate ablate_noskin", payload["metrics"]["command"])
            self.assertIn("--arm-hook --launch", payload["metrics"]["command"])
            self.assertIn("pending-get-xyz", payload["note"])
            self.assertIn("ensure_dc_mask", payload["note"])
            self.assertIn("mask (N,) nz=", payload["note"])
            self.assertIsNone(payload["metrics"]["dc_mask_support_line"])
            self.assertNotIn("2.31%", payload["note"])
            self.assertNotIn("23.2", payload["note"])

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with patch.object(b2, "RESULTS_DIR", root):
                code = b2.main([], runner=_boom)
                self.assertTrue((root / "b2_celeba_T0" / "summary.json").is_file())
                self.assertFalse((root / "b2_celeba_T0_ablate_noskin").exists())
            self.assertEqual(code, 0)

    def test_arm_hook_noskin_excludes_label_1_and_does_not_launch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            parsing = _write_parsing(root)
            aegis = root / "AEGIS"
            _excerpt_tree(aegis)
            code = b2.main(
                [
                    "--arm-hook",
                    "--ablate",
                    "ablate_noskin",
                    "--parsing",
                    str(parsing),
                    "--aegis-root",
                    str(aegis),
                    "--summary-dir",
                    str(root / "out"),
                ],
                runner=_boom,
            )
            payload = json.loads((root / "out" / "summary.json").read_text(encoding="utf-8"))
            weight = np.load(aegis / ".gs_makeup_mask.npy")
        self.assertEqual(code, 0)
        self.assertEqual(payload["status"], "hook_armed")
        self.assertEqual(payload["mask_labels"], "ablate_noskin")
        self.assertEqual(payload["summary_slot"], "b2_celeba_T0_ablate_noskin")
        self.assertFalse(payload["metrics"]["command_attempted"])
        self.assertFalse(payload["pipeline_ok"])
        self.assertEqual(float(weight[6]), 0.0)
        self.assertEqual(float(weight[0]), 1.0)
        self.assertEqual([float(value) for value in weight[7:]], [0.0, 0.0, 0.0])
        self.assertEqual(payload["metrics"]["hook"]["mask_labels"], "ablate_noskin")
        self.assertEqual(payload["metrics"]["hook"]["parsing_label_ids"], [2, 3, 4, 5, 12, 13])

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            parsing = _write_parsing(root)
            aegis = root / "AEGIS"
            _excerpt_tree(aegis)
            code = b2.main(
                [
                    "--arm-hook",
                    "--parsing",
                    str(parsing),
                    "--aegis-root",
                    str(aegis),
                    "--summary-dir",
                    str(root / "out"),
                ],
                runner=_boom,
            )
            payload = json.loads((root / "out" / "summary.json").read_text(encoding="utf-8"))
            weight = np.load(aegis / ".gs_makeup_mask.npy")
        self.assertEqual(code, 0)
        self.assertEqual(payload["mask_labels"], "full")
        self.assertEqual(payload["summary_slot"], "b2_celeba_T0")
        self.assertEqual(payload["parsing_label_ids"], [1, 2, 3, 4, 5, 12, 13])
        self.assertNotIn("--ablate", payload["metrics"]["command"])
        self.assertEqual(float(weight[6]), 1.0)
        self.assertEqual(payload["metrics"]["hook"]["label_map"], "DEFAULT_PARSING_LABELS")

    def test_arm_hook_on_image_parsing_stays_pending_get_xyz(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            labels = np.zeros((4, 6), dtype=np.int64)
            labels[0, 0] = 12
            labels[1, 1] = 1
            labels[2, 2] = 10
            parsing = root / "parsing" / "306.npy"
            parsing.parent.mkdir(parents=True)
            np.save(parsing, labels)
            code = b2.main(
                [
                    "--arm-hook",
                    "--parsing",
                    str(parsing),
                    "--aegis-root",
                    str(root / "no-aegis"),
                    "--summary-dir",
                    str(root / "out"),
                ],
                runner=_boom,
            )
            payload = json.loads((root / "out" / "summary.json").read_text(encoding="utf-8"))
            product = b2.RESULTS_DIR / "b2_celeba_T0" / "summary.json"
            self.assertTrue(product.is_file())
        self.assertEqual(code, 0)
        hook = payload["metrics"]["hook"]
        self.assertEqual(hook["dc_mask_lift"], "pending-get-xyz")
        self.assertEqual(hook["mask_shape"], [4, 6])
        self.assertEqual(hook["image_mask_nonzero"], 2)
        self.assertAlmostEqual(hook["image_mask_fraction"], 2 / 24)
        self.assertIsNone(payload["metrics"]["dc_mask_support_line"])
        self.assertIsNone(payload["metrics"]["dc_mask_shape"])
        self.assertFalse(payload["metrics"]["command_attempted"])
        self.assertIn("pending-get-xyz", hook["layout"])
        self.assertNotIn("projected-means", hook["dc_mask_lift"])

    def test_launch_summary_records_materialized_n_support(self) -> None:
        def runner(command, **kwargs):
            out = Path(command[command.index("--summary-out") + 1])
            out.write_text(
                json.dumps(
                    {
                        "ok": True,
                        "cos_adv_source": 0.2,
                        "cos_adv_target": 0.8,
                        "outside_m_abs_delta": 0.0,
                        "dc_mask_shape": [89021],
                        "dc_mask_nonzero": 178,
                        "dc_mask_fraction": 178 / 89021,
                        "dc_mask_support_line": "mask (89021,) nz=178 frac=0.001999",
                    }
                ),
                encoding="utf-8",
            )

            class Proc:
                returncode = 0
                stdout = "mask (89021,) nz=178 frac=0.001999\n"
                stderr = ""

            return Proc()

        def fake_install(hook, aegis_root):
            config = Path(aegis_root) / ".gs_makeup_hook.json"
            config.parent.mkdir(parents=True, exist_ok=True)
            config.write_text(
                json.dumps(
                    {
                        "kind": "dc",
                        "subject": "306",
                        "epsilon": 0.1,
                        "dc_mask_lift": "pending-get-xyz",
                    }
                ),
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
                        "--ablate",
                        "ablate_noskin",
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
            csv_text = (root / "out" / "summary.csv").read_text(encoding="utf-8")
            self.assertFalse((b2.RESULTS_DIR / "b2_celeba_T0_ablate_noskin").exists())
        self.assertEqual(code, 0)
        self.assertEqual(payload["summary_slot"], "b2_celeba_T0_ablate_noskin")
        metrics = payload["metrics"]
        self.assertEqual(metrics["dc_mask_shape"], [89021])
        self.assertEqual(metrics["dc_mask_nonzero"], 178)
        self.assertAlmostEqual(metrics["dc_mask_fraction"], 178 / 89021)
        self.assertEqual(metrics["dc_mask_support_line"], "mask (89021,) nz=178 frac=0.001999")
        self.assertEqual(metrics["hook"]["dc_mask_lift"], "projected-means")
        self.assertEqual(metrics["hook"]["mask_shape"], [89021])
        self.assertIn("mask (89021,) nz=178 frac=0.001999", csv_text)


class AblateOnlyskinTest(unittest.TestCase):
    def tearDown(self) -> None:
        clear_hook_state()
        os.environ.pop(HOOK_CONFIG_ENV, None)
        os.environ.pop(b2.IMAGE_ENV, None)
        os.environ.pop(b2.PARSING_ENV, None)

    def test_onlyskin_keeps_label_1_and_leaves_the_default_map(self) -> None:
        from makeup.mask import DEFAULT_PARSING_LABELS, MakeupRegion

        before = dict(DEFAULT_PARSING_LABELS)
        mode, mapping = b2.resolve_mask_labels("ablate_onlyskin")
        self.assertEqual(mode, "ablate_onlyskin")
        self.assertEqual(set(mapping), {1})
        self.assertEqual(mapping[1], MakeupRegion.CHEEK)
        self.assertEqual(b2.parsing_label_ids("ablate_onlyskin"), [1])
        self.assertEqual(dict(DEFAULT_PARSING_LABELS), before)
        self.assertEqual(set(DEFAULT_PARSING_LABELS), {1, 2, 3, 4, 5, 12, 13})
        _noskin_mode, noskin = b2.resolve_mask_labels("ablate_noskin")
        self.assertEqual(set(mapping).isdisjoint(set(noskin)), True)
        self.assertEqual(set(mapping) | set(noskin), set(DEFAULT_PARSING_LABELS))
        for rejected in ("only_skin", "onlyskin", "skin", "cheek"):
            with self.assertRaises(b2.MaskLabelError):
                b2.resolve_mask_labels(rejected)
        with self.assertRaises(SystemExit) as caught:
            b2.parse_args(["--ablate", "only_skin"])
        self.assertEqual(caught.exception.code, 2)
        parsed = b2.parse_args(["--mask-labels", "ablate_onlyskin"])
        self.assertEqual(parsed.ablate, "ablate_onlyskin")

    def test_hook_onlyskin_keeps_skin_and_drops_lips_eyes_brows(self) -> None:
        labels = _labels()
        full = dc_hook_from_parsing(labels, 0.1, subject="306")
        _mode, mapping = b2.resolve_mask_labels("ablate_onlyskin")
        kept = dc_hook_from_parsing(labels, 0.1, subject="306", label_map=mapping)
        full_u = np.asarray(full.mask.union())
        skin_u = np.asarray(kept.mask.union())
        self.assertTrue(np.array_equal(full_u[:6], np.ones(6)))
        self.assertTrue(np.array_equal(skin_u[:6], np.zeros(6)))
        self.assertEqual(full_u[6], 1.0)
        self.assertEqual(skin_u[6], 1.0)
        self.assertTrue(np.array_equal(skin_u[7:], np.zeros(3)))
        self.assertEqual(kept.kind, "dc")

        grid = np.zeros((2, 4), dtype=np.int64)
        grid[0, 0] = 1
        grid[0, 1] = 12
        grid[0, 2] = 10
        full_image = dc_hook_from_parsing(grid, 0.1, means=None)
        skin_image = dc_hook_from_parsing(grid, 0.1, means=None, label_map=mapping)
        self.assertEqual(np.asarray(full_image.image_mask)[0, 0], 1.0)
        self.assertEqual(np.asarray(skin_image.image_mask)[0, 0], 1.0)
        self.assertEqual(np.asarray(skin_image.image_mask)[0, 1], 0.0)
        self.assertEqual(np.asarray(skin_image.image_mask)[0, 2], 0.0)
        self.assertEqual(skin_image.dc_mask_lift, "pending-get-xyz")
        self.assertEqual(full_image.dc_mask_lift, "pending-get-xyz")

    def test_summary_slots_stay_apart_from_full_and_noskin(self) -> None:
        image = Path(b2.DEFAULT_IMPERSONATION_IMAGE)
        lfw = Path(b2.LFW_IMPERSONATION_IMAGE)
        self.assertEqual(b2.summary_slot(image, "ablate_onlyskin"), "b2_celeba_T0_ablate_onlyskin")
        self.assertEqual(b2.summary_slot(lfw, "ablate_onlyskin"), "b2_lfw_T0_ablate_onlyskin")
        self.assertNotEqual(
            b2.summary_slot(image, "ablate_onlyskin"),
            b2.summary_slot(image, "ablate_noskin"),
        )
        self.assertNotEqual(b2.summary_slot(image, "ablate_onlyskin"), b2.summary_slot(image))
        command = b2.user_command(image, Path(b2.DEFAULT_PARSING), "ablate_onlyskin")
        self.assertIn("--ablate ablate_onlyskin --arm-hook --launch", command)
        self.assertNotIn("--ablate ablate_noskin", command)
        default_command = b2.user_command(image, Path(b2.DEFAULT_PARSING))
        self.assertNotIn("--ablate", default_command)
        plan = b1.build_plan(b2.DEFAULT_IMPERSONATION_IMAGE, launch=True)
        worker = b2.worker_command(
            plan,
            Path(b2.DEFAULT_PARSING),
            Path("/tmp/b2-worker.json"),
            b1.AEGIS_ROOT,
            "ablate_onlyskin",
        )
        self.assertEqual(worker[worker.index("--ablate") + 1], "ablate_onlyskin")

        product = b2.RESULTS_DIR / "b2_celeba_T0"
        noskin = b2.RESULTS_DIR / "b2_celeba_T0_ablate_noskin"
        onlyskin = b2.RESULTS_DIR / "b2_celeba_T0_ablate_onlyskin"
        before_json = (product / "summary.json").read_bytes()
        before_csv = (product / "summary.csv").read_bytes()
        for slot in (
            product,
            b2.RESULTS_DIR / "b2_lfw_T0",
            noskin,
            b2.RESULTS_DIR / "b2_lfw_T0_ablate_noskin",
        ):
            code = b2.main(
                ["--ablate", "ablate_onlyskin", "--summary-dir", str(slot)],
                runner=_boom,
            )
            self.assertEqual(code, 2)
        code = b2.main(
            ["--ablate", "ablate_noskin", "--summary-dir", str(onlyskin)],
            runner=_boom,
        )
        self.assertEqual(code, 2)
        code = b2.main(["--summary-dir", str(onlyskin)], runner=_boom)
        self.assertEqual(code, 2)
        code = b2.main(["--summary-dir", str(noskin)], runner=_boom)
        self.assertEqual(code, 2)
        self.assertEqual((product / "summary.json").read_bytes(), before_json)
        self.assertEqual((product / "summary.csv").read_bytes(), before_csv)
        self.assertFalse(noskin.exists())
        self.assertFalse(onlyskin.exists())
        self.assertFalse((b2.RESULTS_DIR / "b2_lfw_T0_ablate_onlyskin").exists())
        self.assertFalse((b2.RESULTS_DIR / "b2_lfw_T0_ablate_noskin").exists())

    def test_dry_run_writes_onlyskin_and_spares_full_and_noskin(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with patch.object(b2, "RESULTS_DIR", root):
                code = b2.main(["--ablate", "ablate_onlyskin"], runner=_boom)
                payload = json.loads(
                    (root / "b2_celeba_T0_ablate_onlyskin" / "summary.json").read_text(
                        encoding="utf-8"
                    )
                )
                self.assertFalse((root / "b2_celeba_T0").exists())
                self.assertFalse((root / "b2_celeba_T0_ablate_noskin").exists())
            self.assertEqual(code, 0)
            self.assertFalse(payload["metrics"]["command_attempted"])
            self.assertFalse(payload["pipeline_ok"])
            self.assertEqual(payload["summary_slot"], "b2_celeba_T0_ablate_onlyskin")
            self.assertEqual(payload["mask_labels"], "ablate_onlyskin")
            self.assertEqual(payload["parsing_label_ids"], [1])
            self.assertFalse(payload["ablation_changes_default_m"])
            self.assertIn("--ablate ablate_onlyskin", payload["metrics"]["command"])
            self.assertIn("--arm-hook --launch", payload["metrics"]["command"])
            self.assertIn("complementary to ablate_noskin", payload["note"])
            self.assertIn("pending-get-xyz", payload["note"])
            self.assertIn("ensure_dc_mask", payload["note"])
            self.assertIn("mask (N,) nz=", payload["note"])
            self.assertIsNone(payload["metrics"]["dc_mask_support_line"])
            self.assertIsNone(payload["metrics"]["dc_mask_shape"])
            self.assertNotIn("2.307%", payload["note"])
            self.assertNotIn("1873", payload["note"])

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with patch.object(b2, "RESULTS_DIR", root):
                code = b2.main(
                    ["--ablate", "ablate_onlyskin", "--impersonation-image", b2.LFW_IMPERSONATION_IMAGE],
                    runner=_boom,
                )
                self.assertTrue((root / "b2_lfw_T0_ablate_onlyskin" / "summary.json").is_file())
                self.assertFalse((root / "b2_lfw_T0").exists())
                self.assertFalse((root / "b2_lfw_T0_ablate_noskin").exists())
                self.assertFalse((root / "b2_celeba_T0_ablate_onlyskin").exists())
            self.assertEqual(code, 0)

    def test_arm_hook_onlyskin_is_label_1_and_image_stays_pending(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            parsing = _write_parsing(root)
            aegis = root / "AEGIS"
            _excerpt_tree(aegis)
            code = b2.main(
                [
                    "--arm-hook",
                    "--ablate",
                    "ablate_onlyskin",
                    "--parsing",
                    str(parsing),
                    "--aegis-root",
                    str(aegis),
                    "--summary-dir",
                    str(root / "out"),
                ],
                runner=_boom,
            )
            payload = json.loads((root / "out" / "summary.json").read_text(encoding="utf-8"))
            weight = np.load(aegis / ".gs_makeup_mask.npy")
        self.assertEqual(code, 0)
        self.assertEqual(payload["status"], "hook_armed")
        self.assertEqual(payload["mask_labels"], "ablate_onlyskin")
        self.assertEqual(payload["summary_slot"], "b2_celeba_T0_ablate_onlyskin")
        self.assertFalse(payload["metrics"]["command_attempted"])
        self.assertEqual([float(value) for value in weight], [0, 0, 0, 0, 0, 0, 1, 0, 0, 0])
        self.assertEqual(payload["metrics"]["hook"]["parsing_label_ids"], [1])
        self.assertEqual(payload["metrics"]["hook"]["label_map"], "ablation:ablate_onlyskin")

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            labels = np.zeros((4, 6), dtype=np.int64)
            labels[0, 0] = 12
            labels[1, 1] = 1
            labels[2, 2] = 4
            labels[3, 3] = 10
            parsing = root / "parsing" / "306.npy"
            parsing.parent.mkdir(parents=True)
            np.save(parsing, labels)
            code = b2.main(
                [
                    "--arm-hook",
                    "--ablate",
                    "ablate_onlyskin",
                    "--parsing",
                    str(parsing),
                    "--aegis-root",
                    str(root / "no-aegis"),
                    "--summary-dir",
                    str(root / "out"),
                ],
                runner=_boom,
            )
            payload = json.loads((root / "out" / "summary.json").read_text(encoding="utf-8"))
        self.assertEqual(code, 0)
        hook = payload["metrics"]["hook"]
        self.assertEqual(hook["dc_mask_lift"], "pending-get-xyz")
        self.assertEqual(hook["mask_shape"], [4, 6])
        self.assertEqual(hook["image_mask_nonzero"], 1)
        self.assertAlmostEqual(hook["image_mask_fraction"], 1 / 24)
        self.assertIsNone(payload["metrics"]["dc_mask_support_line"])
        self.assertIsNone(payload["metrics"]["dc_mask_shape"])
        self.assertIsNone(payload["metrics"]["dc_mask_nonzero"])
        self.assertIsNone(payload["metrics"]["dc_mask_fraction"])
        self.assertIn("pending-get-xyz", hook["layout"])
        self.assertNotIn("projected-means", hook["dc_mask_lift"])
        self.assertFalse(payload["metrics"]["command_attempted"])

    def test_launch_summary_records_materialized_n_support_in_onlyskin_slot(self) -> None:
        def runner(command, **kwargs):
            out = Path(command[command.index("--summary-out") + 1])
            out.write_text(
                json.dumps(
                    {
                        "ok": True,
                        "cos_adv_source": 0.2,
                        "cos_adv_target": 0.8,
                        "outside_m_abs_delta": 0.0,
                        "dc_mask_shape": [89021],
                        "dc_mask_nonzero": 1873,
                        "dc_mask_fraction": 1873 / 89021,
                        "dc_mask_support_line": "mask (89021,) nz=1873 frac=0.021040",
                    }
                ),
                encoding="utf-8",
            )

            class Proc:
                returncode = 0
                stdout = "mask (89021,) nz=1873 frac=0.021040\n"
                stderr = ""

            return Proc()

        def fake_install(hook, aegis_root):
            config = Path(aegis_root) / ".gs_makeup_hook.json"
            config.parent.mkdir(parents=True, exist_ok=True)
            config.write_text(
                json.dumps(
                    {
                        "kind": "dc",
                        "subject": "306",
                        "epsilon": 0.1,
                        "dc_mask_lift": "pending-get-xyz",
                    }
                ),
                encoding="utf-8",
            )
            return {"status": "patched", "armed": True, "config": str(config), "note": "test"}

        product = b2.RESULTS_DIR / "b2_celeba_T0" / "summary.json"
        before = product.read_bytes()
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
                        "--ablate",
                        "ablate_onlyskin",
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
            self.assertFalse((b2.RESULTS_DIR / "b2_celeba_T0_ablate_onlyskin").exists())
            self.assertFalse((b2.RESULTS_DIR / "b2_celeba_T0_ablate_noskin").exists())
        self.assertEqual(product.read_bytes(), before)
        self.assertEqual(code, 0)
        self.assertEqual(payload["summary_slot"], "b2_celeba_T0_ablate_onlyskin")
        metrics = payload["metrics"]
        self.assertEqual(metrics["dc_mask_shape"], [89021])
        self.assertEqual(metrics["dc_mask_nonzero"], 1873)
        self.assertAlmostEqual(metrics["dc_mask_fraction"], 1873 / 89021)
        self.assertEqual(metrics["dc_mask_support_line"], "mask (89021,) nz=1873 frac=0.021040")
        self.assertEqual(metrics["hook"]["dc_mask_lift"], "projected-means")
        self.assertEqual(metrics["hook"]["mask_shape"], [89021])
        self.assertIn("--ablate ablate_onlyskin", metrics["worker_command"])


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
        self.assertIn("results/b2_celeba_T0_ablate_noskin/", section)
        self.assertIn("results/b2_celeba_T0_ablate_onlyskin/", section)
        self.assertIn("results/b2_lfw_T0_ablate_onlyskin/", section)
        self.assertIn("--ablate ablate_noskin", section)
        self.assertIn("--ablate ablate_onlyskin", section)
        self.assertIn("DEFAULT_PARSING_LABELS", section)
        self.assertIn("0.7888%", section)
        self.assertIn("9.43%", section)
        self.assertIn("2.307%", section)
        self.assertIn("0.2033%", section)
        self.assertIn("1873", section)
        self.assertIn("provisional", section)
        self.assertIn("互补", section)
        self.assertIn("pending-get-xyz", section)
        self.assertIn("ensure_dc_mask", section)
        self.assertIn("[540, 960]", section)
        self.assertIn("10.22%", section)
        self.assertIn("mask (N,) nz=", section)
        self.assertNotIn("2.31%", section)
        self.assertNotIn("23.2", section)
        self.assertNotIn("no_skin", section)
        self.assertIn("outside_m_abs_delta", section)
        self.assertIn("cos(adv, source) ↓ relative to goal", section)
        self.assertIn("cos(adv, T) ↑", section)
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

    def test_checklist_names_b2_post_run_gate_without_launch(self) -> None:
        text = (_REPO / "docs" / "gpu_launch_checklist.md").read_text(encoding="utf-8")
        self.assertNotIn("does not enable B2", text)
        self.assertIn("Wiring may merge", text)
        self.assertIn("mentor gate 4", text)
        self.assertIn("cos(adv, source)", text)
        self.assertIn("↓ relative to goal", text)
        self.assertIn("cos(adv, T)", text)
        self.assertIn("outside_m_abs_delta", text)
        self.assertIn("results/b2_celeba_T0/", text)
        self.assertIn("results/b2_lfw_T0/", text)
        self.assertIn("results/b2_celeba_T0_ablate_noskin/", text)
        self.assertIn("results/b2_celeba_T0_ablate_onlyskin/", text)
        self.assertIn("results/b2_lfw_T0_ablate_onlyskin/", text)
        self.assertIn("--ablate ablate_noskin", text)
        self.assertIn("--ablate ablate_onlyskin", text)
        self.assertIn("DEFAULT_PARSING_LABELS", text)
        self.assertIn("0.7888%", text)
        self.assertIn("9.43%", text)
        self.assertIn("2.307%", text)
        self.assertIn("0.2033%", text)
        self.assertIn("1873", text)
        self.assertIn("provisional", text)
        self.assertIn("pending-get-xyz", text)
        self.assertIn("ensure_dc_mask", text)
        self.assertIn("[540, 960]", text)
        self.assertIn("10.22%", text)
        self.assertIn("mask (N,) nz=", text)
        self.assertNotIn("2.31%", text)
        self.assertNotIn("23.2", text)
        self.assertIn("parsing/306.npy", text)


if __name__ == "__main__":
    unittest.main()

"""CPU tests for B1 targeted impersonation. No GPU, no AEGIS import, no datasets."""

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

import b1_targeted_impersonation as b1


def _boom(*_args, **_kwargs):
    raise AssertionError("B1 launched a process")


class CriterionWiringTest(unittest.TestCase):
    def test_logits_match_upstream_columns(self) -> None:
        no_match, match = b1.similarity_to_logits(0.1720)
        self.assertEqual(no_match, 0.0)
        self.assertEqual(match, 0.0)
        no_match, match = b1.similarity_to_logits(0.2720)
        self.assertAlmostEqual(no_match, -1.0)
        self.assertAlmostEqual(match, 1.0)
        self.assertEqual(b1.similarity_to_logits(0.3, None), (-3.0, 3.0))
        low = b1.similarity_to_logits(0.0)[b1.MATCH_CLASS]
        high = b1.similarity_to_logits(0.9)[b1.MATCH_CLASS]
        self.assertGreater(high, low)

    def test_b0_untargeted_match_label_decreases_cosine(self) -> None:
        spec = b1.b0_criterion_spec()
        self.assertEqual(spec["foolbox_criterion"], "Misclassification")
        self.assertEqual(spec["class_index"], b1.MATCH_CLASS)
        for threshold in (b1.VER_THRESHOLD, None):
            for probe in (0.0, 0.2, 0.5):
                sign = b1.similarity_step_sign(
                    spec["foolbox_criterion"],
                    spec["class_index"],
                    ver_threshold=threshold,
                    probe=probe,
                )
                self.assertEqual(sign, -1, (threshold, probe))

    def test_b1_targeted_match_increases_cosine(self) -> None:
        spec = b1.b1_criterion_spec()
        self.assertEqual(spec["foolbox_criterion"], "TargetedMisclassification")
        self.assertEqual(spec["target_class"], b1.MATCH_CLASS)
        self.assertEqual(spec["reference_embedding"], "impersonation_image")
        self.assertFalse(spec["makeup_hook"])
        for threshold in (b1.VER_THRESHOLD, None):
            for probe in (0.0, 0.2, 0.5):
                sign = b1.similarity_step_sign(
                    spec["foolbox_criterion"],
                    spec["target_class"],
                    ver_threshold=threshold,
                    probe=probe,
                )
                self.assertEqual(sign, 1, (threshold, probe))

    def test_wrapper_replaces_bare_match_label(self) -> None:
        class Host:
            def __init__(self) -> None:
                self.device = "cpu"

            def _run_foolbox_attack(self, criterion):
                return criterion

        def factory(device):
            return {
                "foolbox_criterion": "TargetedMisclassification",
                "target_class": b1.MATCH_CLASS,
                "device": device,
            }

        b1.apply_targeted_criterion(Host, factory)
        seen = Host()._run_foolbox_attack(b1.MATCH_CLASS)
        self.assertEqual(seen["foolbox_criterion"], "TargetedMisclassification")
        self.assertEqual(seen["target_class"], 1)
        self.assertEqual(seen["device"], "cpu")
        self.assertIsNot(seen, b1.MATCH_CLASS)

    def test_cosine_similarity(self) -> None:
        self.assertAlmostEqual(b1.cosine_similarity([1.0, 0.0], [1.0, 0.0]), 1.0)
        self.assertAlmostEqual(b1.cosine_similarity([1.0, 0.0], [-1.0, 0.0]), -1.0)
        self.assertAlmostEqual(b1.cosine_similarity([1.0, 0.0], [0.0, 2.0]), 0.0)
        self.assertAlmostEqual(b1.cosine_similarity([[1.0, 0.0]], [1.0, 0.0]), 1.0)
        with self.assertRaises(ValueError):
            b1.cosine_similarity([1.0], [1.0, 0.0])
        with self.assertRaises(ValueError):
            b1.cosine_similarity([0.0, 0.0], [1.0, 0.0])


class PlanAndCliTest(unittest.TestCase):
    def test_default_image_is_celeba_t0(self) -> None:
        self.assertEqual(
            b1.DEFAULT_IMPERSONATION_IMAGE,
            "/root/autodl-fs/gs-assets/targets/celeba_T0.jpg",
        )
        self.assertEqual(
            b1.LFW_IMPERSONATION_IMAGE,
            "/root/autodl-fs/gs-assets/targets/lfw_T0.jpg",
        )
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop(b1.IMPERSONATION_IMAGE_ENV, None)
            self.assertEqual(
                b1.resolve_impersonation_image(None),
                b1.DEFAULT_IMPERSONATION_IMAGE,
            )

    def test_env_and_flag_override_default(self) -> None:
        with patch.dict(
            os.environ,
            {b1.IMPERSONATION_IMAGE_ENV: "/tmp/from-env.jpg"},
        ):
            self.assertEqual(b1.resolve_impersonation_image(None), "/tmp/from-env.jpg")
            self.assertEqual(
                b1.resolve_impersonation_image("/tmp/from-flag.jpg"),
                "/tmp/from-flag.jpg",
            )

    def test_cli_parses_impersonation_image_and_alias(self) -> None:
        flagged = b1.parse_args(
            ["--impersonation-image", "/root/autodl-fs/gs-assets/targets/celeba_T0.jpg"]
        )
        self.assertEqual(
            flagged.impersonation_image,
            "/root/autodl-fs/gs-assets/targets/celeba_T0.jpg",
        )
        self.assertFalse(flagged.launch)
        self.assertFalse(flagged.arm_hook)
        alias = b1.parse_args(
            ["--victim-image", "/root/autodl-fs/gs-assets/targets/lfw_T0.jpg", "--launch"]
        )
        self.assertEqual(
            alias.impersonation_image,
            "/root/autodl-fs/gs-assets/targets/lfw_T0.jpg",
        )
        self.assertTrue(alias.launch)

    def test_plan_keeps_b0_hyperparams_and_full_face(self) -> None:
        plan = b1.build_plan(b1.DEFAULT_IMPERSONATION_IMAGE, launch=False)
        self.assertEqual(plan.subject, "306")
        self.assertEqual(plan.embedder, "adaface")
        self.assertEqual(plan.epsilon, 0.1)
        self.assertEqual(plan.attack_steps, 300)
        self.assertEqual(plan.target_features, "DC")
        self.assertEqual(plan.adv_attack, "linfpgd")
        self.assertEqual(plan.radius, 1.0)
        self.assertEqual(plan.seed, 42)
        self.assertEqual(plan.camera_boundary_angles, (-0.5, 0.5, -0.5, 0.5, 0.0, 0.0))
        self.assertEqual(plan.angle_aggregation, "mean")
        self.assertEqual(plan.ver_threshold, 0.1720)
        self.assertFalse(plan.adaptive_epsilon)
        self.assertEqual(plan.selected_regions, ())
        self.assertFalse(plan.arm_makeup_hook)
        self.assertTrue(str(plan.avatar_dir).endswith("avatars/306"))
        self.assertTrue(str(plan.source_image).endswith("renders/306.png"))
        kwargs = plan.aegis_kwargs()
        self.assertEqual(kwargs["target_image"], b1.DEFAULT_IMPERSONATION_IMAGE)
        self.assertEqual(kwargs["embedder_name"], "adaface")
        self.assertEqual(kwargs["epsilons"], [0.1])
        self.assertEqual(kwargs["targeted_features"], "DC")
        self.assertEqual(kwargs["adv_attack"], "linfpgd")
        self.assertEqual(kwargs["attack_steps"], 300)
        self.assertEqual(kwargs["selected_regions"], [])
        self.assertFalse(kwargs["adaptive_epsilon"])
        self.assertIsNone(kwargs["surrogate_keys"])
        self.assertEqual(kwargs["output_name"], "NeRSembleImpersonation306_celeba_T0")
        self.assertIn("celeba_T0.jpg", plan.user_command())
        self.assertIn("--launch", plan.user_command())
        self.assertNotIn("--arm-hook", plan.user_command())

    def test_reference_image_is_the_still_and_refuses_306_render(self) -> None:
        plan = b1.build_plan("/tmp/celeba_T0.jpg", launch=False)
        self.assertEqual(b1.reference_image_for_run(plan), Path("/tmp/celeba_T0.jpg"))
        source = b1.default_source_image()
        refused = b1.build_plan(str(source), launch=True)
        with self.assertRaises(ValueError):
            b1.reference_image_for_run(refused)
        self.assertTrue(
            b1.is_source_render(Path("datasets/NeRSembleReconst/renders/306.png"))
        )

    def test_default_main_does_not_launch_and_writes_schema(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with patch.dict(os.environ, {b1.IMPERSONATION_IMAGE_ENV: ""}, clear=False):
                os.environ.pop(b1.IMPERSONATION_IMAGE_ENV, None)
                code = b1.main(
                    ["--summary-dir", tmp],
                    runner=_boom,
                )
            self.assertEqual(code, 0)
            payload = json.loads(
                (Path(tmp) / "b1_targeted_impersonation_summary.json").read_text(
                    encoding="utf-8"
                )
            )
            csv_text = (
                Path(tmp) / "b1_targeted_impersonation_summary.csv"
            ).read_text(encoding="utf-8")
        self.assertEqual(payload["experiment_id"], "B1")
        self.assertEqual(payload["subject"], "306")
        self.assertFalse(payload["pipeline_ok"])
        self.assertEqual(payload["status"], "awaiting_target")
        self.assertEqual(payload["corpus"], "CelebA")
        self.assertEqual(payload["smoke_target"], "celeba_T0")
        metrics = payload["metrics"]
        self.assertIsNone(metrics["return_code"])
        self.assertIsNone(metrics["cos_adv_source"])
        self.assertIsNone(metrics["cos_adv_target"])
        self.assertFalse(metrics["makeup_hook_armed"])
        self.assertFalse(metrics["command_attempted"])
        self.assertFalse(metrics["adaptive_epsilon"])
        self.assertEqual(metrics["criterion"]["foolbox_criterion"], "TargetedMisclassification")
        self.assertEqual(metrics["criterion"]["target_class"], 1)
        self.assertEqual(
            metrics["impersonation_image"],
            "/root/autodl-fs/gs-assets/targets/celeba_T0.jpg",
        )
        self.assertEqual(
            metrics["lfw_image"],
            "/root/autodl-fs/gs-assets/targets/lfw_T0.jpg",
        )
        self.assertIn("cos_adv_source", csv_text)
        self.assertIn("cos_adv_target", csv_text)
        self.assertIn("pipeline_ok", csv_text)
        self.assertIn("makeup_hook_armed", csv_text)

    def test_launch_without_still_does_not_start_gpu(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            code = b1.main(
                [
                    "--launch",
                    "--impersonation-image",
                    str(Path(tmp) / "celeba_T0.jpg"),
                    "--summary-dir",
                    tmp,
                ],
                runner=_boom,
            )
            payload = json.loads(
                (Path(tmp) / "b1_targeted_impersonation_summary.json").read_text(
                    encoding="utf-8"
                )
            )
        self.assertEqual(code, 2)
        self.assertEqual(payload["status"], "awaiting_target")
        self.assertFalse(payload["pipeline_ok"])
        self.assertFalse(payload["metrics"]["command_attempted"])
        self.assertFalse(payload["metrics"]["makeup_hook_armed"])

    def test_launch_with_still_but_no_aegis_does_not_start(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            image = Path(tmp) / "celeba_T0.jpg"
            image.write_bytes(b"\xff\xd8\xff not-a-real-download")
            code = b1.main(
                ["--launch", "--impersonation-image", str(image), "--summary-dir", tmp],
                runner=_boom,
            )
            payload = json.loads(
                (Path(tmp) / "b1_targeted_impersonation_summary.json").read_text(
                    encoding="utf-8"
                )
            )
        self.assertEqual(code, 2)
        self.assertIn(payload["status"], {"bootstrap_required", "deps_missing", "gpu_required"})
        self.assertFalse(payload["metrics"]["command_attempted"])
        self.assertFalse(payload["metrics"]["makeup_hook_armed"])

    def test_launch_refuses_306_render_as_target(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            code = b1.main(
                [
                    "--launch",
                    "--impersonation-image",
                    str(b1.default_source_image()),
                    "--summary-dir",
                    tmp,
                ],
                runner=_boom,
            )
            payload = json.loads(
                (Path(tmp) / "b1_targeted_impersonation_summary.json").read_text(
                    encoding="utf-8"
                )
            )
        self.assertEqual(code, 2)
        self.assertEqual(payload["status"], "refused_source_image")
        self.assertFalse(payload["metrics"]["command_attempted"])

    def test_readme_documents_exact_filenames_and_command(self) -> None:
        text = (_REPO / "README.md").read_text(encoding="utf-8")
        self.assertIn("/root/autodl-fs/gs-assets/targets/celeba_T0.jpg", text)
        self.assertIn("/root/autodl-fs/gs-assets/targets/lfw_T0.jpg", text)
        self.assertNotIn("celeba_000001.jpg", text)
        self.assertIn("python src/baselines/b1_targeted_impersonation.py", text)
        self.assertIn("--impersonation-image", text)
        self.assertIn("--launch", text)
        self.assertIn("james", text.lower())
        self.assertIn("CelebA", text)
        self.assertIn("LFW", text)
        self.assertIn("不把整库", text)
        self.assertNotIn("install_makeup_pgd_hook", text[text.find("## B1") : text.find("## 实验矩阵")])


class MakeupHookRefusalTest(unittest.TestCase):
    def test_refuse_does_not_call_installer(self) -> None:
        called = []

        def installer():
            called.append("armed")

        with self.assertRaises(b1.MakeupHookForbidden):
            b1.refuse_makeup_hook(installer)
        self.assertEqual(called, [])

    def test_disarm_env_is_not_a_hook_config_file(self) -> None:
        env = b1.worker_environment({})
        b1.assert_makeup_disarmed(env)
        self.assertEqual(env["B1_MAKEUP_HOOK"], "off")
        self.assertFalse(Path(env[b1.MAKEUP_HOOK_ENV]).is_file())
        self.assertIn(str(b1.AEGIS_ROOT / "src"), env["PYTHONPATH"])
        with tempfile.TemporaryDirectory() as tmp:
            hook = Path(tmp) / ".gs_makeup_hook.json"
            hook.write_text("{}", encoding="utf-8")
            armed = dict(env)
            armed[b1.MAKEUP_HOOK_ENV] = str(hook)
            with self.assertRaises(b1.MakeupHookForbidden):
                b1.assert_makeup_disarmed(armed)

    def test_worker_command_has_no_makeup_hook(self) -> None:
        plan = b1.build_plan(b1.DEFAULT_IMPERSONATION_IMAGE, launch=True)
        command = b1.worker_command(plan, Path("/tmp/b1-worker.json"))
        text = " ".join(command)
        self.assertIn("--worker", command)
        self.assertIn(b1.DEFAULT_IMPERSONATION_IMAGE, command)
        self.assertNotIn("--arm-hook", command)
        self.assertNotIn("adaptive-epsilon", text)
        self.assertNotIn("select-regions", text)
        self.assertNotIn("kind=dc", text)

    def test_start_worker_passes_disarmed_env(self) -> None:
        plan = b1.build_plan(b1.DEFAULT_IMPERSONATION_IMAGE, launch=True)
        seen = {}

        def runner(command, **kwargs):
            seen["command"] = command
            seen["env"] = kwargs["env"]
            seen["cwd"] = kwargs["cwd"]

            class Proc:
                returncode = 0
                stdout = ""
                stderr = ""

            return Proc()

        b1.start_worker(plan, Path("/tmp/b1-worker.json"), runner)
        self.assertNotIn("--arm-hook", seen["command"])
        b1.assert_makeup_disarmed(seen["env"])
        self.assertEqual(seen["cwd"], b1.AEGIS_ROOT)

    def test_arm_hook_flag_refuses_before_launch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            code = b1.main(
                ["--arm-hook", "--launch", "--summary-dir", tmp],
                runner=_boom,
            )
            payload = json.loads(
                (Path(tmp) / "b1_targeted_impersonation_summary.json").read_text(
                    encoding="utf-8"
                )
            )
        self.assertEqual(code, 2)
        self.assertEqual(payload["status"], "refused_makeup_hook")
        self.assertFalse(payload["metrics"]["makeup_hook_armed"])
        self.assertFalse(payload["metrics"]["command_attempted"])

    def test_module_does_not_call_makeup_installer(self) -> None:
        text = b1.MODULE_PATH.read_text(encoding="utf-8")
        self.assertNotIn("install_makeup_pgd_hook(", text)
        self.assertNotIn("arm_makeup_pgd_hook(", text)
        self.assertIn("def refuse_makeup_hook", text)


if __name__ == "__main__":
    unittest.main()

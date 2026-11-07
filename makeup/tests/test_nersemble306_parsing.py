"""Smoke for the NeRSemble 306 BiSeNet label writer. No weights, no GPU.

The real checkpoint is ``79999_iter.pth``. These tests mock that load or
stop at the missing-file errors. See ``makeup/README.md`` for the real run.
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np

from makeup import SemanticMask
from makeup.nersemble_parsing import (
    ParsingSetupError,
    coerce_state_dict,
    default_output_paths,
    disable_resnet_imagenet_download,
    format_summary,
    logits_to_label_map,
    make_nersemble306_parsing,
    region_presence,
    resize_label_map,
    resolve_weights,
    select_device,
)

_SCRIPT = _ROOT / "scripts" / "make_nersemble306_parsing.py"
_MODULE = _ROOT / "makeup" / "nersemble_parsing.py"


class _Device:
    def __init__(self, name: str) -> None:
        self.type = name

    def __str__(self) -> str:
        return self.type


class NerSemble306ParsingTest(unittest.TestCase):
    def test_output_paths_are_the_306_sidecars(self) -> None:
        primary, alternate = default_output_paths(Path("/repo"))
        self.assertEqual(
            primary,
            Path("/repo/third_party/AEGIS/datasets/NeRSembleReconst/parsing/306.npy"),
        )
        self.assertEqual(
            alternate,
            Path(
                "/repo/third_party/AEGIS/datasets/NeRSembleReconst/avatars/306/parsing.npy"
            ),
        )
        self.assertNotIn("527", str(primary))
        self.assertNotIn("527", str(alternate))

    def test_resize_copies_ids_without_blending(self) -> None:
        src = np.array([[12, 13], [4, 1]], dtype=np.int64)
        out = resize_label_map(src, (4, 4))
        self.assertEqual(out.shape, (4, 4))
        self.assertEqual(out.dtype, np.dtype(np.int64))
        self.assertTrue(np.all(out[0:2, 0:2] == 12))
        self.assertTrue(np.all(out[0:2, 2:4] == 13))
        self.assertTrue(np.all(out[2:4, 0:2] == 4))
        self.assertTrue(np.all(out[2:4, 2:4] == 1))
        self.assertTrue(np.array_equal(resize_label_map(src, (2, 2)), src))

    def test_logits_argmax_matches_default_ids(self) -> None:
        logits = np.zeros((19, 2, 2), dtype=np.float32)
        logits[12, 0, 0] = 4
        logits[13, 0, 1] = 4
        logits[4, 1, 0] = 4
        logits[1, 1, 1] = 4
        labels = logits_to_label_map(logits, (2, 2))
        self.assertTrue(np.array_equal(labels, np.array([[12, 13], [4, 1]])))
        resized = logits_to_label_map(logits[None, ...], (4, 2))
        self.assertEqual(resized.shape, (4, 2))
        self.assertTrue(np.array_equal(resized[0], np.array([12, 13])))
        self.assertTrue(np.array_equal(resized[3], np.array([4, 1])))
        presence = region_presence(labels)
        self.assertTrue(presence["lip"])
        self.assertTrue(presence["eye"])
        self.assertFalse(presence["brow"])
        self.assertTrue(presence["cheek"])
        outside = np.array([[0, 17], [10, 11]], dtype=np.int64)
        absent = region_presence(outside)
        self.assertEqual(
            absent, {"lip": False, "eye": False, "brow": False, "cheek": False}
        )

    def test_state_dict_strips_module_prefix(self) -> None:
        cleaned = coerce_state_dict({"module.cp.conv.weight": [1], "cp.bn.bias": [2]})
        self.assertEqual(set(cleaned), {"cp.conv.weight", "cp.bn.bias"})
        wrapped = coerce_state_dict({"state_dict": {"module.a": 1}})
        self.assertEqual(wrapped, {"a": 1})
        with self.assertRaises(ParsingSetupError):
            coerce_state_dict({})

    def test_imagenet_download_hook_is_a_noop(self) -> None:
        class Resnet18:
            def __init__(self) -> None:
                self.init_weight()

            def init_weight(self) -> None:
                raise RuntimeError("download")

        module = type("ResnetModule", (), {"Resnet18": Resnet18})
        disable_resnet_imagenet_download(module)
        module.Resnet18()

    def test_missing_torch_names_cpu_install(self) -> None:
        with patch.dict(sys.modules, {"torch": None}):
            with self.assertRaises(ParsingSetupError) as caught:
                select_device(False)
        message = str(caught.exception)
        self.assertIn("torch", message)
        self.assertIn("CPU", message)
        self.assertNotIn("Traceback", message)

    def test_missing_weights_name_the_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            with self.assertRaises(ParsingSetupError) as caught:
                resolve_weights(None, repo)
        message = str(caught.exception)
        self.assertIn("79999_iter.pth", message)
        self.assertIn("does not download", message)

    def test_mocked_bisenet_writes_default_npy(self) -> None:
        labels = np.zeros((5, 7), dtype=np.int64)
        labels[0, 0] = 12
        labels[0, 1] = 13
        labels[1, 0] = 4
        labels[1, 1] = 5
        labels[2, 0] = 2
        labels[2, 1] = 3
        labels[3, 0] = 1
        labels[4, 6] = 17
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image = root / "306.png"
            image.write_bytes(b"not-opened")
            face = root / "third_party" / "face-parsing.PyTorch"
            (face / "res" / "cp").mkdir(parents=True)
            (face / "model.py").write_text("class BiSeNet: pass\n", encoding="utf-8")
            weights = face / "res" / "cp" / "79999_iter.pth"
            weights.write_bytes(b"stub")
            with (
                patch(
                    "makeup.nersemble_parsing.select_device",
                    return_value=_Device("cpu"),
                ) as select,
                patch(
                    "makeup.nersemble_parsing.load_bisenet",
                    return_value=object(),
                ) as load,
                patch(
                    "makeup.nersemble_parsing.produce_label_map",
                    return_value=labels,
                ) as produce,
            ):
                result = make_nersemble306_parsing(
                    image=image,
                    repo=face,
                    weights=weights,
                    force_cpu=True,
                    repo_root=root,
                )
            select.assert_called_once_with(True)
            load.assert_called_once()
            produce.assert_called_once()
            primary, alternate = default_output_paths(root)
            self.assertEqual(result.path, primary)
            self.assertFalse(alternate.exists())
            stored = np.load(primary, allow_pickle=False)
            self.assertEqual(stored.shape, (5, 7))
            self.assertEqual(stored.dtype, np.dtype(np.int64))
            self.assertTrue(np.array_equal(stored, labels))
            mask = SemanticMask.from_parsing(stored, image=np.zeros((5, 7, 3)))
            self.assertEqual(float(mask.region("lip")[0, 0]), 1.0)
            self.assertEqual(float(mask.region("eye")[1, 0]), 1.0)
            self.assertEqual(float(mask.region("brow")[2, 1]), 1.0)
            self.assertEqual(float(mask.region("cheek")[3, 0]), 1.0)
            self.assertEqual(float(mask.support()[4, 6]), 0.0)
            self.assertIn("shape: (5, 7)", result.summary)
            self.assertIn("unique_ids: 0 1 2 3 4 5 12 13 17", result.summary)
            self.assertIn("lip=yes eye=yes brow=yes cheek=yes", result.summary)
            self.assertIn("device: cpu", result.summary)
            text = format_summary(
                path=primary,
                labels=stored,
                image=image,
                device=_Device("cpu"),
                regions=region_presence(stored),
            )
            self.assertIn("DEFAULT_PARSING_LABELS:", text)

    def test_sources_do_not_launch_masking(self) -> None:
        for path in (_SCRIPT, _MODULE):
            text = path.read_text(encoding="utf-8")
            self.assertNotIn("mask_avatar", text)
            self.assertNotIn("FaceScape", text)
            self.assertNotIn("527", text)
            self.assertNotIn("subprocess", text)

    def test_cli_missing_inputs_exit_2_without_traceback(self) -> None:
        env = os.environ.copy()
        env.pop("FACE_PARSING_PYTORCH", None)
        help_proc = subprocess.run(
            [sys.executable, str(_SCRIPT), "--help"],
            cwd=_ROOT,
            capture_output=True,
            text=True,
            check=False,
            env=env,
        )
        self.assertEqual(help_proc.returncode, 0, help_proc.stderr)
        help_text = "".join(help_proc.stdout.split())
        self.assertIn("parsing/306.npy", help_text)
        self.assertIn("avatars/306/parsing.npy", help_text)
        self.assertIn("--cpu", help_text)
        self.assertNotIn("527", help_text)

        missing_image = subprocess.run(
            [sys.executable, str(_SCRIPT), "--image", "/no/such/306.png"],
            cwd=_ROOT,
            capture_output=True,
            text=True,
            check=False,
            env=env,
        )
        self.assertEqual(missing_image.returncode, 2)
        self.assertIn("Input image not found", missing_image.stderr)
        self.assertNotIn("Traceback", missing_image.stderr)

        with tempfile.TemporaryDirectory() as tmp:
            image = Path(tmp) / "face.png"
            image.write_bytes(b"x")
            missing_repo = subprocess.run(
                [sys.executable, str(_SCRIPT), "--image", str(image)],
                cwd=_ROOT,
                capture_output=True,
                text=True,
                check=False,
                env=env,
            )
            self.assertEqual(missing_repo.returncode, 2, missing_repo.stderr)
            self.assertIn("face-parsing.PyTorch", missing_repo.stderr)
            self.assertIn("git clone", missing_repo.stderr)
            self.assertNotIn("Traceback", missing_repo.stderr)

            repo = Path(tmp) / "repo"
            repo.mkdir()
            (repo / "model.py").write_text("class BiSeNet: pass\n", encoding="utf-8")
            missing_weights = subprocess.run(
                [
                    sys.executable,
                    str(_SCRIPT),
                    "--image",
                    str(image),
                    "--repo",
                    str(repo),
                    "--cpu",
                ],
                cwd=_ROOT,
                capture_output=True,
                text=True,
                check=False,
                env=env,
            )
            self.assertEqual(missing_weights.returncode, 2, missing_weights.stderr)
            self.assertIn("79999_iter.pth", missing_weights.stderr)
            self.assertIn("does not download", missing_weights.stderr)
            self.assertNotIn("Traceback", missing_weights.stderr)


if __name__ == "__main__":
    unittest.main()

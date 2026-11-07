"""Smoke for the NeRSemble 306 BiSeNet label writer. No weights, no GPU.

The real checkpoint is ``79999_iter.pth``. These tests mock that load or
stop at the missing-file errors. See ``makeup/README.md`` for the real run.
"""

from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
import tempfile
import unittest
from io import StringIO
from pathlib import Path
from unittest.mock import patch

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np

from makeup import SemanticMask
from makeup.nersemble_parsing import (
    CropBox,
    CropParse,
    FaceBox,
    ParsingRejectedError,
    ParsingSetupError,
    bgr_uint8,
    coerce_state_dict,
    default_output_paths,
    detect_largest_face,
    disable_resnet_imagenet_download,
    face_crop_box,
    format_summary,
    labels_from_crop,
    load_insightface_detector,
    logits_to_label_map,
    make_nersemble306_parsing,
    paste_crop_labels,
    produce_label_map,
    region_pixel_counts,
    region_presence,
    require_nonzero_makeup_regions,
    resize_label_map,
    resolve_weights,
    select_device,
    select_largest_face,
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
            box = CropBox(0, 5, 0, 7)
            with (
                patch(
                    "makeup.nersemble_parsing.load_insightface_detector",
                    return_value=object(),
                ) as detector,
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
                    return_value=CropParse(labels, box, 0.91),
                ) as produce,
            ):
                result = make_nersemble306_parsing(
                    image=image,
                    repo=face,
                    weights=weights,
                    force_cpu=True,
                    repo_root=root,
                )
            detector.assert_called_once_with(True)
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
            self.assertIn("region_counts: lip=2 eye=2 brow=2 cheek=1", result.summary)
            self.assertIn("crop: y0=0 y1=5 x0=0 x1=7", result.summary)
            self.assertIn("detector: insightface buffalo_l det_10g.onnx", result.summary)
            self.assertIn("next: python scripts/viz_nersemble306_parsing.py", result.summary)
            self.assertEqual(result.counts["lip"], 2)
            self.assertEqual(result.crop, box)
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
        self.assertIn("--margin", help_text)
        self.assertIn("--no-square", help_text)
        self.assertIn("det_10g.onnx", help_text)
        self.assertIn("viz_nersemble306_parsing.py", help_text)
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

            weights = repo / "res" / "cp" / "79999_iter.pth"
            weights.parent.mkdir(parents=True)
            weights.write_bytes(b"stub")
            missing_detector = subprocess.run(
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
            self.assertEqual(missing_detector.returncode, 2, missing_detector.stderr)
            self.assertNotIn("Traceback", missing_detector.stderr)
            if importlib.util.find_spec("insightface") is None:
                self.assertIn("insightface", missing_detector.stderr)
                self.assertIn("onnxruntime", missing_detector.stderr)
                self.assertIn("det_10g.onnx", missing_detector.stderr)
                self.assertIn("does not download", missing_detector.stderr)
                self.assertIn("pip install insightface onnxruntime", missing_detector.stderr)


def _full_region_crop(shape: tuple[int, int]) -> np.ndarray:
    labels = np.zeros(shape, dtype=np.int64)
    labels[0, 0] = 12
    labels[0, min(1, shape[1] - 1)] = 13
    labels[min(1, shape[0] - 1), 0] = 4
    if shape[1] > 1:
        labels[min(1, shape[0] - 1), 1] = 5
    labels[min(2, shape[0] - 1), 0] = 2
    if shape[1] > 1:
        labels[min(2, shape[0] - 1), 1] = 3
    labels[min(3, shape[0] - 1), min(2, shape[1] - 1)] = 1
    return labels


class _FakeRgb:
    def __init__(self, hw: tuple[int, int]) -> None:
        self.hw = hw
        self.size = (hw[1], hw[0])

    def crop(self, box: tuple[int, int, int, int]) -> "_FakeRgb":
        x0, y0, x1, y1 = box
        return _FakeRgb((y1 - y0, x1 - x0))


class FaceCropParsingTest(unittest.TestCase):
    def test_paste_back_keeps_full_canvas_and_background(self) -> None:
        crop = np.array([[12, 13], [4, 5], [2, 3]], dtype=np.int64)
        full = paste_crop_labels(crop, (6, 5), (1, 2))
        self.assertEqual(full.shape, (6, 5))
        self.assertEqual(full.dtype, np.dtype(np.int64))
        self.assertTrue(np.array_equal(full[1:4, 2:4], crop))
        outside = np.ones(full.shape, dtype=bool)
        outside[1:4, 2:4] = False
        self.assertTrue(np.all(full[outside] == 0))
        counts = region_pixel_counts(full)
        self.assertEqual(counts["lip"], 2)
        self.assertEqual(counts["eye"], 2)
        self.assertEqual(counts["brow"], 2)
        self.assertEqual(counts["cheek"], 0)

        clipped = paste_crop_labels(
            np.array([[12, 1], [13, 4]], dtype=np.int64),
            (6, 5),
            (5, 4),
        )
        self.assertEqual(int(clipped[5, 4]), 12)
        self.assertTrue(np.all(clipped[:5] == 0))
        self.assertTrue(np.all(clipped[:, :4] == 0))
        self.assertEqual(int(clipped.sum()), 12)

    def test_square_margin_and_border_clip(self) -> None:
        box = face_crop_box((40, 30, 80, 50), (100, 100), margin=0.5, square=True)
        self.assertEqual(box, CropBox(0, 80, 20, 100))
        self.assertEqual(box.shape, (80, 80))
        tight = face_crop_box((10, 20, 30, 40), (80, 90), margin=0.0, square=False)
        self.assertEqual(tight, CropBox(20, 40, 10, 30))
        clipped = face_crop_box((0, 0, 10, 10), (15, 15), margin=1.0, square=False)
        self.assertEqual(clipped, CropBox(0, 15, 0, 15))
        with self.assertRaises(ParsingRejectedError):
            face_crop_box((120, 10, 140, 40), (50, 50), margin=0.2, square=True)

    def test_largest_face_wins_and_empty_is_rejected(self) -> None:
        boxes = np.array(
            [[0, 0, 10, 10, 0.99], [5, 5, 35, 45, 0.40]],
            dtype=np.float32,
        )
        chosen = select_largest_face(boxes)
        self.assertEqual((chosen.x0, chosen.y0, chosen.x1, chosen.y1), (5.0, 5.0, 35.0, 45.0))
        self.assertAlmostEqual(chosen.score, 0.40, places=5)
        with self.assertRaises(ParsingRejectedError) as caught:
            select_largest_face(np.zeros((0, 5), dtype=np.float32))
        self.assertIn("no face", str(caught.exception))
        self.assertIn("det_10g.onnx", str(caught.exception))

        class _Empty:
            def detect(self, img, max_num=0):
                self.seen = img
                return np.zeros((0, 5), dtype=np.float32), None

        detector = _Empty()
        rgb = np.zeros((4, 6, 3), dtype=np.uint8)
        rgb[0, 0, 0] = 30
        with self.assertRaises(ParsingRejectedError):
            detect_largest_face(detector, rgb)
        self.assertEqual(detector.seen.shape, (4, 6, 3))
        self.assertEqual(int(detector.seen[0, 0, 2]), 30)
        self.assertEqual(int(detector.seen[0, 0, 0]), 0)

    def test_bgr_scales_unit_float(self) -> None:
        rgb = np.zeros((1, 1, 3), dtype=np.float32)
        rgb[0, 0, 0] = 1.0
        bgr = bgr_uint8(rgb)
        self.assertEqual(bgr.dtype, np.dtype(np.uint8))
        self.assertEqual(int(bgr[0, 0, 2]), 255)
        self.assertEqual(int(bgr[0, 0, 0]), 0)

    def test_validation_uses_from_parsing_counts(self) -> None:
        labels = np.zeros((4, 4), dtype=np.int64)
        labels[0, 0] = 12
        labels[0, 1] = 4
        labels[1, 0] = 2
        labels[1, 1] = 1
        counts = require_nonzero_makeup_regions(labels)
        self.assertTrue(all(counts[name] > 0 for name in ("lip", "eye", "brow", "cheek")))
        labels[0, 0] = 11
        with self.assertRaises(ParsingRejectedError) as caught:
            require_nonzero_makeup_regions(labels, path=Path("parsing/306.npy"))
        message = str(caught.exception)
        self.assertIn("SemanticMask.from_parsing", message)
        self.assertIn("Empty: lip", message)
        self.assertIn("lip=0", message)
        self.assertIn("parsing/306.npy", message)
        self.assertIn("viz_nersemble306_parsing.py", message)
        self.assertNotIn("Traceback", message)

    def test_produce_pastes_mocked_crop_onto_original_canvas(self) -> None:
        full_hw = (8, 10)

        def fake_parse(net, image, device):
            self.assertEqual(image.hw, (6, 4))
            self.assertEqual(device.type, "cpu")
            return _full_region_crop(image.hw)

        with (
            patch(
                "makeup.nersemble_parsing.read_rgb",
                return_value=(_FakeRgb(full_hw), full_hw),
            ),
            patch(
                "makeup.nersemble_parsing.detect_largest_face",
                return_value=FaceBox(2, 1, 6, 7, 0.8),
            ),
            patch(
                "makeup.nersemble_parsing.parse_rgb_image",
                side_effect=fake_parse,
            ),
        ):
            parsed = produce_label_map(
                object(),
                Path("306.png"),
                _Device("cpu"),
                object(),
                margin=0.0,
                square=False,
            )
        self.assertEqual(parsed.box, CropBox(1, 7, 2, 6))
        self.assertEqual(parsed.labels.shape, full_hw)
        self.assertEqual(int(parsed.labels[1, 2]), 12)
        self.assertEqual(int(parsed.labels[1, 3]), 13)
        self.assertEqual(int(parsed.labels[2, 2]), 4)
        self.assertEqual(int(parsed.labels[2, 3]), 5)
        self.assertEqual(int(parsed.labels[3, 2]), 2)
        self.assertEqual(int(parsed.labels[3, 3]), 3)
        self.assertEqual(int(parsed.labels[4, 4]), 1)
        self.assertEqual(int(parsed.labels[0, 0]), 0)
        self.assertEqual(int(parsed.labels[7, 9]), 0)
        counts = require_nonzero_makeup_regions(parsed.labels)
        self.assertTrue(all(value > 0 for value in counts.values()))
        pasted = labels_from_crop(
            _full_region_crop((6, 4)),
            full_hw,
            CropBox(1, 7, 2, 6),
        )
        self.assertTrue(np.array_equal(pasted, parsed.labels))

    def test_zero_lip_is_written_then_rejected(self) -> None:
        labels = np.zeros((4, 4), dtype=np.int64)
        labels[0, 0] = 4
        labels[0, 1] = 5
        labels[1, 0] = 2
        labels[1, 1] = 3
        labels[2, 0] = 1
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image = root / "306.png"
            image.write_bytes(b"not-opened")
            face = root / "repo"
            face.mkdir()
            (face / "model.py").write_text("class BiSeNet: pass\n", encoding="utf-8")
            weights = face / "79999_iter.pth"
            weights.write_bytes(b"stub")
            with (
                patch(
                    "makeup.nersemble_parsing.load_insightface_detector",
                    return_value=object(),
                ),
                patch(
                    "makeup.nersemble_parsing.select_device",
                    return_value=_Device("cpu"),
                ),
                patch(
                    "makeup.nersemble_parsing.load_bisenet",
                    return_value=object(),
                ),
                patch(
                    "makeup.nersemble_parsing.produce_label_map",
                    return_value=CropParse(labels, CropBox(0, 4, 0, 4), 0.5),
                ),
            ):
                with self.assertRaises(ParsingRejectedError) as caught:
                    make_nersemble306_parsing(
                        image=image,
                        repo=face,
                        weights=weights,
                        force_cpu=True,
                        repo_root=root,
                    )
            primary, _alternate = default_output_paths(root)
            self.assertTrue(primary.is_file())
            stored = np.load(primary, allow_pickle=False)
            self.assertTrue(np.array_equal(stored, labels))
            self.assertEqual(region_pixel_counts(stored)["lip"], 0)
        message = str(caught.exception)
        self.assertIn("Empty: lip", message)
        self.assertIn("not accepted", message)

    def test_missing_insightface_names_the_install(self) -> None:
        with patch.dict(sys.modules, {"insightface": None}):
            with self.assertRaises(ParsingSetupError) as caught:
                load_insightface_detector(True)
        message = str(caught.exception)
        self.assertIn("insightface", message)
        self.assertIn("onnxruntime", message)
        self.assertIn("det_10g.onnx", message)
        self.assertIn("does not download", message)
        self.assertIn("pip install insightface onnxruntime", message)
        self.assertNotIn("Traceback", message)

    def test_missing_onnx_names_the_file(self) -> None:
        fake = type(sys)("insightface")
        with (
            patch.dict(sys.modules, {"insightface": fake}),
            patch(
                "makeup.nersemble_parsing.insightface_detector_path",
                return_value=Path("/no/such/det_10g.onnx"),
            ),
        ):
            with self.assertRaises(ParsingSetupError) as caught:
                load_insightface_detector(True)
        message = str(caught.exception)
        self.assertIn("/no/such/det_10g.onnx", message)
        self.assertIn("onnxruntime", message)
        self.assertIn("ensure_available", message)

    def test_cli_rejected_map_exits_1(self) -> None:
        spec = importlib.util.spec_from_file_location(
            "make_nersemble306_parsing_cli", _SCRIPT
        )
        self.assertIsNotNone(spec)
        self.assertIsNotNone(spec.loader)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        stderr = StringIO()
        with (
            patch.object(
                module,
                "make_nersemble306_parsing",
                side_effect=ParsingRejectedError("Empty: lip. counts: lip=0"),
            ),
            patch("sys.stderr", stderr),
        ):
            code = module.main(["--cpu"])
        self.assertEqual(code, 1)
        self.assertIn("lip=0", stderr.getvalue())
        self.assertNotIn("Traceback", stderr.getvalue())


if __name__ == "__main__":
    unittest.main()

"""CPU check for the NeRSemble 306 parsing overlay. No BiSeNet, no GPU.

A synthetic label grid stands in for ``parsing/306.npy``. Counts come from
``SemanticMask.from_parsing`` with ``DEFAULT_PARSING_LABELS``.
"""

from __future__ import annotations

import os
import struct
import subprocess
import sys
import tempfile
import unittest
import zlib
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np

from makeup.parsing_viz import (
    DEFAULT_ALPHA,
    PAINT_ORDER,
    REGION_COLORS,
    ParsingVizError,
    build_parsing_report,
    color_hex,
    decode_png_rgb,
    write_png_rgb,
)
from makeup.mask import DEFAULT_PARSING_LABELS

_SCRIPT = _ROOT / "scripts" / "viz_nersemble306_parsing.py"
_MODULE = _ROOT / "makeup" / "parsing_viz.py"


def _blend(base: np.ndarray, rgb: tuple, alpha: float = DEFAULT_ALPHA) -> np.ndarray:
    color = np.asarray(rgb, dtype=np.float64)
    return np.rint((1.0 - alpha) * base.astype(np.float64) + alpha * color).astype(np.uint8)


def _full_grid() -> np.ndarray:
    labels = np.zeros((4, 4), dtype=np.int64)
    labels[0, 0] = 12
    labels[0, 1] = 13
    labels[0, 2] = 4
    labels[0, 3] = 5
    labels[1, 0] = 2
    labels[1, 1] = 3
    labels[1, 2] = 1
    labels[1, 3] = 17
    return labels


def _mentor_id_grid() -> np.ndarray:
    """Small grid whose ids match the AutoDL 306 map: no 12/13, no 3, no 5."""
    labels = np.zeros((6, 8), dtype=np.int64)
    labels[0, 0] = 1
    labels[0, 1] = 1
    labels[0, 2] = 2
    labels[0, 3] = 4
    labels[0, 4] = 7
    labels[0, 5] = 8
    labels[0, 6] = 14
    labels[0, 7] = 16
    labels[1, 0] = 17
    labels[1, 1] = 18
    return labels


def _chunk(tag: bytes, data: bytes) -> bytes:
    crc = zlib.crc32(tag + data) & 0xFFFFFFFF
    return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", crc)


def _encode_filtered(rgb: np.ndarray, ftype: int) -> bytes:
    """Build a PNG whose scanlines all use one filter. Independent of the reader."""
    height, width, _ = rgb.shape
    channels = 3
    stride = width * channels
    raw = bytearray()
    previous = bytearray(stride)
    for y in range(height):
        row = bytearray(np.ascontiguousarray(rgb[y]).tobytes())
        filtered = bytearray(stride)
        for x in range(stride):
            left = row[x - channels] if x >= channels else 0
            up = previous[x] if y else 0
            upper_left = previous[x - channels] if y and x >= channels else 0
            if ftype == 0:
                pred = 0
            elif ftype == 1:
                pred = left
            elif ftype == 2:
                pred = up
            elif ftype == 3:
                pred = (left + up) // 2
            else:
                estimate = left + up - upper_left
                pa = abs(estimate - left)
                pb = abs(estimate - up)
                pc = abs(estimate - upper_left)
                if pa <= pb and pa <= pc:
                    pred = left
                elif pb <= pc:
                    pred = up
                else:
                    pred = upper_left
            filtered[x] = (row[x] - pred) & 255
        raw.append(ftype)
        raw.extend(filtered)
        previous = row
    ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    sig = b"\x89PNG\r\n\x1a\n"
    return sig + _chunk(b"IHDR", ihdr) + _chunk(b"IDAT", zlib.compress(bytes(raw), 9)) + _chunk(b"IEND", b"")


class NerSemble306ParsingVizTest(unittest.TestCase):
    def test_full_grid_counts_and_overlay(self) -> None:
        labels = _full_grid()
        image = np.zeros(labels.shape + (3,), dtype=np.uint8)
        image[:] = (10, 20, 30)
        image[1, 3] = (1, 2, 3)
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "overlay.png"
            report = build_parsing_report(
                labels,
                image,
                out,
                parsing_name="grid.npy",
                image_name="grid.png",
            )
            self.assertEqual(report.exit_code, 0)
            self.assertEqual(report.status, "ok")
            self.assertEqual(report.empty_regions, ())
            self.assertEqual(report.incomplete_regions, ())
            self.assertEqual(report.shape, (4, 4))
            self.assertEqual(report.counts["lip"], 2)
            self.assertEqual(report.counts["eye"], 2)
            self.assertEqual(report.counts["brow"], 2)
            self.assertEqual(report.counts["cheek"], 1)
            self.assertEqual(report.counts["union"], 7)
            pixels = 16
            self.assertEqual(report.fractions["lip"], 2 / pixels)
            self.assertEqual(report.fractions["union"], 7 / pixels)
            self.assertEqual(
                report.id_counts,
                ((0, 8), (1, 1), (2, 1), (3, 1), (4, 1), (5, 1), (12, 1), (13, 1), (17, 1)),
            )
            self.assertIn("missing_default_ids: none", report.text)
            self.assertNotIn("FLAG:", report.text)
            self.assertIn("status: ok", report.text)
            self.assertIn("built_with: SemanticMask.from_parsing", report.text)
            self.assertIn("label_map: DEFAULT_PARSING_LABELS", report.text)
            for region in PAINT_ORDER:
                self.assertIn(color_hex(REGION_COLORS[region]), report.text)
            self.assertTrue(out.is_file())
            stored = decode_png_rgb(out.read_bytes())
            self.assertEqual(stored.shape, image.shape)
            self.assertTrue(np.array_equal(stored[0, 0], _blend(image[0, 0], REGION_COLORS["lip"])))
            self.assertTrue(np.array_equal(stored[0, 2], _blend(image[0, 2], REGION_COLORS["eye"])))
            self.assertTrue(np.array_equal(stored[1, 1], _blend(image[1, 1], REGION_COLORS["brow"])))
            self.assertTrue(np.array_equal(stored[1, 2], _blend(image[1, 2], REGION_COLORS["cheek"])))
            self.assertTrue(np.array_equal(stored[1, 3], image[1, 3]))
            self.assertTrue(np.array_equal(stored[3, 3], image[3, 3]))

    def test_missing_lip_ids_flag_and_still_write_overlay(self) -> None:
        labels = _mentor_id_grid()
        image = np.full(labels.shape + (3,), 80, dtype=np.uint8)
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "nested" / "overlay.png"
            report = build_parsing_report(labels, image, out)
            self.assertEqual(report.exit_code, 1)
            self.assertEqual(report.status, "empty_region")
            self.assertEqual(report.counts["lip"], 0)
            self.assertEqual(report.counts["eye"], 1)
            self.assertEqual(report.counts["brow"], 1)
            self.assertEqual(report.counts["cheek"], 2)
            self.assertEqual(report.counts["union"], 4)
            self.assertEqual(report.fractions["lip"], 0.0)
            self.assertEqual(report.fractions["cheek"], 2 / 48)
            self.assertEqual(report.empty_regions, ("lip",))
            self.assertEqual(report.incomplete_regions, ("eye", "brow"))
            present = [label_id for label_id, _ in report.id_counts]
            self.assertEqual(present, [0, 1, 2, 4, 7, 8, 14, 16, 17, 18])
            self.assertNotIn(12, present)
            self.assertNotIn(13, present)
            self.assertNotIn(3, present)
            self.assertNotIn(5, present)
            self.assertIn("FLAG: lip count is 0", report.text)
            self.assertIn("FLAG: required region empty: lip", report.text)
            self.assertIn("INCOMPLETE: eye missing 5 r_eye; present 4 l_eye=1", report.text)
            self.assertIn("INCOMPLETE: brow missing 3 r_brow; present 2 l_brow=1", report.text)
            self.assertIn("12 u_lip (lip)", report.text)
            self.assertIn("13 l_lip (lip)", report.text)
            self.assertIn("exit_code: 1", report.text)
            self.assertIn("  1 skin count=2 fraction=0.04166667 region=cheek", report.text)
            self.assertTrue(out.is_file())
            stored = decode_png_rgb(out.read_bytes())
            self.assertTrue(np.array_equal(stored[0, 3], _blend(image[0, 3], REGION_COLORS["eye"])))
            self.assertTrue(np.array_equal(stored[0, 2], _blend(image[0, 2], REGION_COLORS["brow"])))
            self.assertTrue(np.array_equal(stored[2, 2], image[2, 2]))

    def test_region_counts_match_default_ids(self) -> None:
        labels = _full_grid()
        image = np.zeros(labels.shape + (3,), dtype=np.uint8)
        with tempfile.TemporaryDirectory() as tmp:
            report = build_parsing_report(labels, image, Path(tmp) / "o.png")
        histogram = dict(report.id_counts)
        for region, ids in (
            ("lip", (12, 13)),
            ("eye", (4, 5)),
            ("brow", (2, 3)),
            ("cheek", (1,)),
        ):
            self.assertEqual(
                report.counts[region],
                sum(histogram.get(label_id, 0) for label_id in ids),
            )
            self.assertEqual(DEFAULT_PARSING_LABELS[ids[0]].value, region)

    def test_png_filters_roundtrip_to_the_same_rgb(self) -> None:
        rgb = np.arange(2 * 3 * 3, dtype=np.uint8).reshape(2, 3, 3)
        rgb[1, 2] = (255, 0, 128)
        for ftype in (0, 1, 2, 3, 4):
            decoded = decode_png_rgb(_encode_filtered(rgb, ftype))
            self.assertTrue(np.array_equal(decoded, rgb), f"filter {ftype}")
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "written.png"
            write_png_rgb(path, rgb)
            self.assertTrue(np.array_equal(decode_png_rgb(path.read_bytes()), rgb))

    def test_hand_built_sub_filter_and_rgba(self) -> None:
        # One row, two RGB pixels. Sub filter stores the second pixel as a delta.
        # Pixels (10, 20, 30) and (40, 50, 60) -> filtered 10,20,30,30,30,30.
        raw = bytes([1, 10, 20, 30, 30, 30, 30])
        sig = b"\x89PNG\r\n\x1a\n"
        ihdr = struct.pack(">IIBBBBB", 2, 1, 8, 2, 0, 0, 0)
        png = sig + _chunk(b"IHDR", ihdr) + _chunk(b"IDAT", zlib.compress(raw)) + _chunk(b"IEND", b"")
        decoded = decode_png_rgb(png)
        self.assertTrue(
            np.array_equal(decoded, np.array([[[10, 20, 30], [40, 50, 60]]], dtype=np.uint8))
        )
        rgba = np.zeros((1, 1, 4), dtype=np.uint8)
        rgba[0, 0] = (9, 8, 7, 255)
        raw_rgba = bytes([0, 9, 8, 7, 255])
        ihdr_rgba = struct.pack(">IIBBBBB", 1, 1, 8, 6, 0, 0, 0)
        png_rgba = (
            sig
            + _chunk(b"IHDR", ihdr_rgba)
            + _chunk(b"IDAT", zlib.compress(raw_rgba))
            + _chunk(b"IEND", b"")
        )
        self.assertTrue(
            np.array_equal(decode_png_rgb(png_rgba), np.array([[[9, 8, 7]]], dtype=np.uint8))
        )
        raw_gray = bytes([0, 12, 34])
        ihdr_gray = struct.pack(">IIBBBBB", 2, 1, 8, 0, 0, 0, 0)
        png_gray = (
            sig
            + _chunk(b"IHDR", ihdr_gray)
            + _chunk(b"IDAT", zlib.compress(raw_gray))
            + _chunk(b"IEND", b"")
        )
        self.assertTrue(
            np.array_equal(
                decode_png_rgb(png_gray),
                np.array([[[12, 12, 12], [34, 34, 34]]], dtype=np.uint8),
            )
        )
        plte = bytes([0, 0, 0, 1, 2, 3])
        raw_idx = bytes([0, 1])
        ihdr_idx = struct.pack(">IIBBBBB", 1, 1, 8, 3, 0, 0, 0)
        png_idx = (
            sig
            + _chunk(b"IHDR", ihdr_idx)
            + _chunk(b"PLTE", plte)
            + _chunk(b"IDAT", zlib.compress(raw_idx))
            + _chunk(b"IEND", b"")
        )
        self.assertTrue(
            np.array_equal(decode_png_rgb(png_idx), np.array([[[1, 2, 3]]], dtype=np.uint8))
        )

    def test_every_empty_required_region_is_flagged(self) -> None:
        labels = np.zeros((2, 2), dtype=np.int64)
        labels[0, 0] = 12
        labels[0, 1] = 13
        image = np.zeros((2, 2, 3), dtype=np.uint8)
        with tempfile.TemporaryDirectory() as tmp:
            report = build_parsing_report(labels, image, Path(tmp) / "o.png")
        self.assertEqual(report.counts["lip"], 2)
        self.assertEqual(report.counts["eye"], 0)
        self.assertEqual(report.counts["brow"], 0)
        self.assertEqual(report.counts["cheek"], 0)
        self.assertEqual(report.counts["union"], 2)
        self.assertEqual(report.exit_code, 1)
        self.assertEqual(report.empty_regions, ("eye", "brow", "cheek"))
        self.assertNotIn("FLAG: lip count is 0", report.text)
        for name in ("eye", "brow", "cheek"):
            self.assertIn(f"FLAG: required region empty: {name}", report.text)

    def test_bad_inputs_do_not_write(self) -> None:
        labels = _full_grid()
        image = np.zeros((4, 4, 3), dtype=np.uint8)
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "nope.png"
            with self.assertRaises(ParsingVizError) as caught:
                build_parsing_report(labels, np.zeros((3, 3, 3), dtype=np.uint8), out)
            self.assertIn("does not align", str(caught.exception))
            self.assertFalse(out.exists())
            with self.assertRaises(ParsingVizError):
                build_parsing_report(labels, image, out, alpha=1.5)
            self.assertFalse(out.exists())
        with self.assertRaises(ParsingVizError) as missing:
            from makeup.parsing_viz import load_label_grid

            load_label_grid(Path("/no/such/306.npy"))
        self.assertIn("not found", str(missing.exception))
        self.assertIn("BiSeNet", str(missing.exception))

    def test_sources_do_not_launch_masking_or_bisenet(self) -> None:
        for path in (_SCRIPT, _MODULE):
            text = path.read_text(encoding="utf-8")
            self.assertNotIn("mask_avatar", text)
            self.assertNotIn("import torch", text)
            self.assertNotIn("BiSeNet(", text)
            self.assertNotIn("subprocess", text)

    def test_readme_documents_the_autodl_command_and_legend(self) -> None:
        readme = (_ROOT / "README.md").read_text(encoding="utf-8")
        makeup = (_ROOT / "makeup" / "README.md").read_text(encoding="utf-8")
        for text in (readme, makeup):
            self.assertIn("python scripts/viz_nersemble306_parsing.py", text)
            self.assertIn("results/nersemble306_parsing_overlay.png", text)
            for region in REGION_COLORS:
                self.assertIn(color_hex(REGION_COLORS[region]), text)

    def test_cli_on_synthetic_grid(self) -> None:
        labels = _mentor_id_grid()
        image = np.full(labels.shape + (3,), 40, dtype=np.uint8)
        env = os.environ.copy()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            parsing = root / "306.npy"
            png = root / "306.png"
            out = root / "overlay.png"
            np.save(parsing, labels)
            write_png_rgb(png, image)
            proc = subprocess.run(
                [
                    sys.executable,
                    str(_SCRIPT),
                    "--parsing",
                    str(parsing),
                    "--image",
                    str(png),
                    "--out",
                    str(out),
                ],
                cwd=_ROOT,
                capture_output=True,
                text=True,
                check=False,
                env=env,
            )
            self.assertEqual(proc.returncode, 1, proc.stderr)
            self.assertNotIn("Traceback", proc.stderr)
            self.assertIn("FLAG: lip count is 0", proc.stdout)
            self.assertIn(f"overlay: {out}", proc.stdout)
            self.assertIn("union count=4", proc.stdout)
            self.assertTrue(out.is_file())
            stored = decode_png_rgb(out.read_bytes())
            self.assertEqual(stored.shape, image.shape)

            help_proc = subprocess.run(
                [sys.executable, str(_SCRIPT), "--help"],
                cwd=_ROOT,
                capture_output=True,
                text=True,
                check=False,
                env=env,
            )
            self.assertEqual(help_proc.returncode, 0, help_proc.stderr)
            help_text = help_proc.stdout
            self.assertIn("parsing/306.npy", help_text)
            self.assertIn("renders/306.png", help_text)
            self.assertIn("nersemble306_parsing_overlay.png", help_text)
            self.assertIn("#DC283C", help_text)

            missing = subprocess.run(
                [sys.executable, str(_SCRIPT), "--parsing", str(root / "absent.npy")],
                cwd=_ROOT,
                capture_output=True,
                text=True,
                check=False,
                env=env,
            )
            self.assertEqual(missing.returncode, 2)
            self.assertIn("not found", missing.stderr)
            self.assertNotIn("Traceback", missing.stderr)


if __name__ == "__main__":
    unittest.main()

#!/usr/bin/env python3
"""Write the NeRSemble 306 face-parsing label map.

The NeRSemble demo does not ship ``parsing/306.npy``. This detects the face
with the InsightFace buffalo_l detector AEGIS already uses
(``~/.insightface/models/buffalo_l/det_10g.onnx``, RetinaFace/SCRFD
``detect``), crops it with a margin, runs BiSeNet from a local
face-parsing.PyTorch checkout (CelebAMask-HQ, 19 classes) on the crop, and
pastes the labels back onto a zero canvas the size of the original image.
The written ``(H, W)`` int64 ``.npy`` stays aligned with ``renders/306.png``.
Ids match ``makeup.mask.DEFAULT_PARSING_LABELS`` (lip 12/13, eye 4/5, brow
2/3, skin 1 as cheek).

After the write, lip, eye, brow, and cheek must each have a pixel count
greater than 0 under ``SemanticMask.from_parsing``. Otherwise the process
exits 1 and the map is not accepted. Re-run
``python scripts/viz_nersemble306_parsing.py`` after a successful write.

Default image:
  ``third_party/AEGIS/datasets/NeRSembleReconst/renders/306.png`` when present,
  otherwise ``--image``.

Default output:
  ``third_party/AEGIS/datasets/NeRSembleReconst/parsing/306.npy``
Alternate:
  ``third_party/AEGIS/datasets/NeRSembleReconst/avatars/306/parsing.npy``
  (pass that path as ``--out``).

CPU is the default when CUDA is not already available. ``--cpu`` forces CPU
for both the detector and BiSeNet. The script does not clone
face-parsing.PyTorch, does not download ``79999_iter.pth``, does not download
``det_10g.onnx``, and does not start AEGIS masking. GPU masking still waits
for a mentor greenlight.

Missing checkout, weights, input image, or detector deps exit 2 and name
what to install. No network weights are required to import this module.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from makeup.nersemble_parsing import (  # noqa: E402
    DEFAULT_CROP_MARGIN,
    FACE_PARSING_DIRNAME,
    ParsingRejectedError,
    ParsingSetupError,
    WEIGHT_FILENAME,
    make_nersemble306_parsing,
)


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build NeRSemble 306 parsing/306.npy. InsightFace buffalo_l "
            "det_10g.onnx crops the face, then face-parsing.PyTorch BiSeNet "
            "(CelebAMask-HQ ids) parses the crop and the labels are pasted "
            "back onto the full image. Lip, eye, brow, and cheek must each "
            "be non-empty. CPU is enough. Does not download weights and does "
            "not start AEGIS masking. After a successful write, re-run "
            "scripts/viz_nersemble306_parsing.py."
        )
    )
    parser.add_argument(
        "--image",
        type=Path,
        default=None,
        help=(
            "Frontal RGB render of subject 306. Default: "
            "third_party/AEGIS/datasets/NeRSembleReconst/renders/306.png "
            "when that file is already on disk."
        ),
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help=(
            "Output int64 .npy, shape (H, W). Default: "
            "third_party/AEGIS/datasets/NeRSembleReconst/parsing/306.npy. "
            "Alternate sidecar: "
            "third_party/AEGIS/datasets/NeRSembleReconst/avatars/306/parsing.npy."
        ),
    )
    parser.add_argument(
        "--repo",
        type=Path,
        default=None,
        help=(
            "face-parsing.PyTorch checkout (must contain model.py). "
            f"Default: third_party/{FACE_PARSING_DIRNAME}, or "
            "FACE_PARSING_PYTORCH when that environment variable is set."
        ),
    )
    parser.add_argument(
        "--weights",
        type=Path,
        default=None,
        help=(
            f"CelebAMask-HQ BiSeNet checkpoint ({WEIGHT_FILENAME}, 19 classes). "
            f"Default: <repo>/res/cp/{WEIGHT_FILENAME}. Not downloaded."
        ),
    )
    parser.add_argument(
        "--cpu",
        action="store_true",
        help=(
            "Force CPU for InsightFace and BiSeNet even when CUDA is "
            "available. CUDA is optional. This does not start GPU masking."
        ),
    )
    parser.add_argument(
        "--margin",
        type=float,
        default=DEFAULT_CROP_MARGIN,
        help=(
            "Fraction of the detected box added on each side before BiSeNet. "
            f"Default: {DEFAULT_CROP_MARGIN}."
        ),
    )
    parser.add_argument(
        "--square",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Expand the cropped box to a square about its center (default). "
            "--no-square keeps the margin-expanded rectangle."
        ),
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        result = make_nersemble306_parsing(
            image=args.image,
            out=args.out,
            repo=args.repo,
            weights=args.weights,
            force_cpu=args.cpu,
            repo_root=REPO_ROOT,
            margin=args.margin,
            square=args.square,
        )
    except ParsingRejectedError as exc:
        print(exc, file=sys.stderr)
        return 1
    except (ParsingSetupError, ImportError) as exc:
        print(exc, file=sys.stderr)
        return 2
    print(result.summary)
    return 0


if __name__ == "__main__":
    sys.exit(main())

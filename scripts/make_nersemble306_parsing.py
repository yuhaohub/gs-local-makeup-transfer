#!/usr/bin/env python3
"""Write the NeRSemble 306 face-parsing label map.

The NeRSemble demo does not ship ``parsing/306.npy``. This runs BiSeNet from
a local face-parsing.PyTorch checkout (CelebAMask-HQ, 19 classes) on one RGB
image and writes an ``(H, W)`` int64 ``.npy``. Ids match
``makeup.mask.DEFAULT_PARSING_LABELS`` (lip 12/13, eye 4/5, brow 2/3, skin 1
as cheek).

Default image:
  ``third_party/AEGIS/datasets/NeRSembleReconst/renders/306.png`` when present,
  otherwise ``--image``.

Default output:
  ``third_party/AEGIS/datasets/NeRSembleReconst/parsing/306.npy``
Alternate:
  ``third_party/AEGIS/datasets/NeRSembleReconst/avatars/306/parsing.npy``
  (pass that path as ``--out``).

CPU is the default when CUDA is not already available. ``--cpu`` forces CPU.
The script does not clone face-parsing.PyTorch, does not download
``79999_iter.pth``, and does not start AEGIS masking. GPU masking still waits
for a mentor greenlight.

Missing checkout, weights, or the input image exit 2 with the reason. No
network weights are required to import this module.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from makeup.nersemble_parsing import (  # noqa: E402
    FACE_PARSING_DIRNAME,
    ParsingSetupError,
    WEIGHT_FILENAME,
    make_nersemble306_parsing,
)


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build NeRSemble 306 parsing/306.npy with face-parsing.PyTorch "
            "BiSeNet (CelebAMask-HQ ids). CPU is enough. Does not download "
            "weights and does not start AEGIS masking."
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
        help="Force CPU even when CUDA is available. CUDA is optional.",
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
        )
    except (ParsingSetupError, ImportError) as exc:
        print(exc, file=sys.stderr)
        return 2
    print(result.summary)
    return 0


if __name__ == "__main__":
    sys.exit(main())

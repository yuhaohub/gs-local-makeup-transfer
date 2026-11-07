#!/usr/bin/env python3
"""Overlay NeRSemble 306 parsing regions on the render and print counts.

CPU only. Builds ``M`` with ``SemanticMask.from_parsing`` and
``DEFAULT_PARSING_LABELS`` (lip 12/13, eye 4/5, brow 2/3, skin 1 as cheek).
Prints pixel counts and the fraction of ``H×W`` for lip, eye, brow, cheek,
and their union, plus the raw id histogram. Writes an RGB overlay PNG.

Legend (alpha 0.55: ``(1 - alpha) * image + alpha * color``):

| region | color |
| --- | --- |
| cheek | ``#F0B428`` rgb(240, 180, 40) |
| brow | ``#326EE6`` rgb(50, 110, 230) |
| eye | ``#28C850`` rgb(40, 200, 80) |
| lip | ``#DC283C`` rgb(220, 40, 60) |

Pixels outside ``M`` stay the original image. Paint order is cheek, brow,
eye, lip.

Exit ``1`` when lip, eye, brow, or cheek has no pixels (the overlay is still
written). Exit ``2`` when the npy or png is missing or invalid. A missing
left/right id is printed as ``INCOMPLETE`` and does not by itself change the
exit code. This script does not run BiSeNet and does not start GPU masking.

From the repo root on AutoDL, when the 306 files are already on disk:

    python scripts/viz_nersemble306_parsing.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from makeup.parsing_viz import (  # noqa: E402
    DEFAULT_ALPHA,
    DEFAULT_IMAGE_REL,
    DEFAULT_OVERLAY_REL,
    DEFAULT_PARSING_REL,
    ParsingVizError,
    visualize_parsing,
)


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "CPU quality check for NeRSemble 306 parsing/306.npy. "
            "Prints SemanticMask.from_parsing counts and writes an RGB "
            "overlay. Does not run BiSeNet and does not start GPU masking."
        )
    )
    parser.add_argument(
        "--parsing",
        type=Path,
        default=None,
        help=(
            "Integer (H, W) label map. Default: "
            "third_party/AEGIS/datasets/NeRSembleReconst/parsing/306.npy"
        ),
    )
    parser.add_argument(
        "--image",
        type=Path,
        default=None,
        help=(
            "RGB render aligned with the label map. Default: "
            "third_party/AEGIS/datasets/NeRSembleReconst/renders/306.png"
        ),
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help=(
            "RGB overlay PNG. Default: results/nersemble306_parsing_overlay.png. "
            "Legend: cheek #F0B428, brow #326EE6, eye #28C850, lip #DC283C."
        ),
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=DEFAULT_ALPHA,
        help="Region tint strength in [0, 1]. Default: 0.55.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    parsing = args.parsing if args.parsing is not None else REPO_ROOT / DEFAULT_PARSING_REL
    image = args.image if args.image is not None else REPO_ROOT / DEFAULT_IMAGE_REL
    out = args.out if args.out is not None else REPO_ROOT / DEFAULT_OVERLAY_REL
    try:
        report = visualize_parsing(parsing, image, out, alpha=args.alpha)
    except ParsingVizError as exc:
        print(exc, file=sys.stderr)
        return 2
    print(report.text)
    return report.exit_code


if __name__ == "__main__":
    sys.exit(main())

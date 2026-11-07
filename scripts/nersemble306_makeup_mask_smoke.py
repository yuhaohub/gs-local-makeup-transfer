#!/usr/bin/env python3
"""CPU smoke: makeup mask constrains a NeRSemble 306 AEGIS-style residual.

Builds ``M`` with ``SemanticMask.from_parsing`` (or the documented fallback
when the 306 demo has no label sidecar), projects a DC/RGB residual onto
``supp(δ) ⊆ M`` and ``||δ||∞ ≤ 0.1``, and writes:

- ``results/nersemble306_makeup_mask_smoke.json``
- ``results/nersemble306_makeup_mask_smoke.csv``

``--launch`` starts the B0-a ``mask_avatar.py`` argv only when the 306
ply, png, AdaFace checkpoint, and CUDA device are already on disk. It does
not clone AEGIS and does not download FaceScape or weights. The makefile
target ``mask_avatar`` is FaceScape 527; this script does not call it.

The Foolbox loop inside AEGIS is not patched. See ``makeup/aegis_mask.py``.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from makeup.aegis_mask import (  # noqa: E402
    SUBJECT,
    assert_subject_306,
    collect_launch_blockers,
    command_text,
    launch_mask_avatar,
    resolve_parsing,
    run_cpu_mask_smoke,
    smoke_note,
)

SUMMARY_STEM = "nersemble306_makeup_mask_smoke"
EXPERIMENT_ID = "306-makeup-mask-smoke"
EXPERIMENT_DESC = (
    "NeRSemble 306 AEGIS-style masking with makeup mask "
    "supp(delta) ⊆ M (not full B2)"
)


def build_payload(smoke: dict, launch: dict) -> dict:
    """Tracked summary. The CPU projection is the constraint check."""
    if not smoke["constraint_holds"]:
        status = "constraint_failed"
    elif launch.get("launched"):
        status = "gpu_started"
    else:
        status = "cpu_smoke_ok"
    return {
        "experiment_id": EXPERIMENT_ID,
        "desc": EXPERIMENT_DESC,
        "status": status,
        "subject": SUBJECT,
        "epsilon": smoke["epsilon"],
        "cpu_residual_kind": smoke["kind"],
        "gpu_target_features": "DC",
        "mask_source": smoke["mask_source"],
        "parsing_path": smoke["parsing_path"],
        "built_with": smoke["built_with"],
        "constraint_holds": smoke["constraint_holds"],
        "pgd_steps": smoke["steps"],
        "pgd_outside_support_max_abs": smoke["pgd_outside_support_max_abs"],
        "pgd_linf": smoke["pgd_linf"],
        "unprojected_outside_support_max_abs": smoke[
            "unprojected_outside_support_max_abs"
        ],
        "posthoc_outside_support_max_abs": smoke["posthoc_outside_support_max_abs"],
        "posthoc_linf": smoke["posthoc_linf"],
        "apply_outside_unchanged": smoke["apply_outside_unchanged"],
        "apply_matches_projected_delta": smoke["apply_matches_projected_delta"],
        "support_fraction": smoke["support_fraction"],
        "gpu_launched": bool(launch.get("launched")),
        "gpu_return_code": launch.get("return_code"),
        "gpu_blockers": list(launch.get("blockers") or []),
        "gpu_log_tail": launch.get("log_tail") or "",
        "command": command_text(),
        "aegis_hook": smoke["aegis_hook"],
        "aegis_hook_todo": smoke["aegis_hook_todo"],
        "note": smoke_note(smoke, launch),
    }


def write_summary(results_dir: Path, payload: dict) -> tuple[Path, Path]:
    """Write the JSON and CSV summaries. Leaves ``.gitkeep`` in place."""
    results_dir.mkdir(parents=True, exist_ok=True)
    gitkeep = results_dir / ".gitkeep"
    if results_dir == REPO_ROOT / "results" and not gitkeep.exists():
        gitkeep.write_text("")
    json_path = results_dir / f"{SUMMARY_STEM}.json"
    csv_path = results_dir / f"{SUMMARY_STEM}.csv"
    json_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "experiment_id",
                "status",
                "subject",
                "mask_source",
                "constraint_holds",
                "epsilon",
                "gpu_launched",
                "command",
                "note",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "experiment_id": payload["experiment_id"],
                "status": payload["status"],
                "subject": payload["subject"],
                "mask_source": payload["mask_source"],
                "constraint_holds": str(bool(payload["constraint_holds"])).lower(),
                "epsilon": payload["epsilon"],
                "gpu_launched": str(bool(payload["gpu_launched"])).lower(),
                "command": payload["command"],
                "note": payload["note"],
            }
        )
    return json_path, csv_path


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "NeRSemble 306 makeup-mask smoke. Constrains a residual with "
            "supp(delta) ⊆ M and ||delta||_inf ≤ 0.1. Does not download "
            "FaceScape or weights."
        )
    )
    parser.add_argument(
        "--subject",
        default=SUBJECT,
        help="Avatar subject. Only 306 is accepted. Default: 306.",
    )
    parser.add_argument(
        "--parsing",
        type=Path,
        default=None,
        help=(
            "Optional integer label map (.npy), shape (H, W) or (N,). "
            "Default ids are BiSeNet / CelebAMask-HQ. When omitted, a sidecar "
            "under the AEGIS tree is used if present, otherwise the documented "
            "fallback label grid."
        ),
    )
    parser.add_argument(
        "--aegis-root",
        type=Path,
        default=REPO_ROOT / "third_party" / "AEGIS",
        help="Checkout of third_party/AEGIS. Used for the parsing sidecar and --launch.",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=REPO_ROOT / "results",
        help="Directory for the JSON and CSV summaries.",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=3,
        help="Projected L∞ steps in the CPU smoke. Default: 3.",
    )
    parser.add_argument(
        "--launch",
        action="store_true",
        help=(
            "Run the B0-a 306 mask_avatar.py command when ply, png, AdaFace "
            "weights, and CUDA are already present. Does not download them. "
            "That process does not apply the makeup mask; the CPU projection does."
        ),
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        assert_subject_306(args.subject)
        resolved = resolve_parsing(
            parsing_path=args.parsing,
            aegis_root=args.aegis_root,
            subject=args.subject,
        )
        smoke = run_cpu_mask_smoke(resolved, steps=args.steps)
    except (ValueError, FileNotFoundError, OSError, ImportError) as exc:
        print(exc, file=sys.stderr)
        return 2

    if args.launch:
        launch = launch_mask_avatar(args.aegis_root)
    else:
        launch = {
            "launched": False,
            "return_code": None,
            "blockers": collect_launch_blockers(args.aegis_root),
            "log_tail": "",
        }
    payload = build_payload(smoke, launch)
    json_path, csv_path = write_summary(args.results_dir, payload)
    print(
        f"status={payload['status']} "
        f"constraint_holds={str(payload['constraint_holds']).lower()} "
        f"mask_source={payload['mask_source']} "
        f"gpu_launched={str(payload['gpu_launched']).lower()}"
    )
    try:
        print(f"wrote {json_path.relative_to(REPO_ROOT)}")
        print(f"wrote {csv_path.relative_to(REPO_ROOT)}")
    except ValueError:
        print(f"wrote {json_path}")
        print(f"wrote {csv_path}")
    if not payload["constraint_holds"]:
        return 1
    if payload["gpu_launched"] and payload["gpu_return_code"] not in (0, None):
        return 1
    if payload["gpu_launched"] and payload["gpu_return_code"] is None:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
# B0 scaffold only: original AEGIS de-identification sanity check.
# Spec: docs/创新章_交付文档_AEGIS底座.md §6.2 item 1
#       ("AEGIS 原设定复现 sanity check" — de-id metrics, prove the base is wired).
# Matrix: configs/experiment_matrix.yaml baselines.B0
#         ("AEGIS original de-identification").
# Bootstrap with scripts/00_bootstrap_aegis.sh. third_party/AEGIS stays
# gitignored; do not vendor it. No training, weights, or secrets here.

"""Write a gitkeep-friendly B0 summary. The AEGIS run itself is still a TODO.

Upstream (after bootstrap) documents the original path in its own README
(`make mask_avatar` / `scripts/mask_avatar.py`, then the eval Make targets).
This module does not import that tree and does not launch it.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
AEGIS_ROOT = REPO_ROOT / "third_party" / "AEGIS"
RESULTS_DIR = REPO_ROOT / "results"
SUMMARY_JSON = RESULTS_DIR / "b0_aegis_sanity_summary.json"
SUMMARY_CSV = RESULTS_DIR / "b0_aegis_sanity_summary.csv"

EXPERIMENT_ID = "B0"
EXPERIMENT_DESC = "AEGIS original de-identification"


def aegis_bootstrapped() -> bool:
    """True when scripts/00_bootstrap_aegis.sh has cloned the upstream tree."""
    return (AEGIS_ROOT / ".git").is_dir()


def load_aegis_pipeline() -> None:
    """TODO: after bootstrap, load the original AEGIS de-id path.

    Read third_party/AEGIS/README and wire the documented sanity entry
    (mask, then one de-id eval). Do not invent an import API in this file,
    and do not start training or download weights from here.
    """
    raise NotImplementedError(
        "B0 scaffold: original AEGIS pipeline is not loaded yet. "
        "Bootstrap with scripts/00_bootstrap_aegis.sh, then replace this TODO."
    )


def run_original_deid_sanity() -> dict:
    """TODO: run one original-AEGIS de-identification sanity path.

    Record upstream de-id metrics only (no new loss, no impersonation).
    Return a small dict suitable for the JSON/CSV summary.
    """
    raise NotImplementedError(
        "B0 scaffold: original AEGIS de-id sanity path is not wired yet."
    )


def write_summary(status: str, note: str, bootstrapped: bool) -> None:
    """Write a tiny tracked summary. Leaves results/.gitkeep in place."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    gitkeep = RESULTS_DIR / ".gitkeep"
    if not gitkeep.exists():
        gitkeep.write_text("")

    payload = {
        "experiment_id": EXPERIMENT_ID,
        "desc": EXPERIMENT_DESC,
        "status": status,
        "bootstrapped": bootstrapped,
        "metrics": {},
        "note": note,
    }
    SUMMARY_JSON.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    with SUMMARY_CSV.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["experiment_id", "status", "bootstrapped", "note"],
        )
        writer.writeheader()
        writer.writerow(
            {
                "experiment_id": EXPERIMENT_ID,
                "status": status,
                "bootstrapped": str(bootstrapped).lower(),
                "note": note,
            }
        )


def main() -> int:
    bootstrapped = aegis_bootstrapped()
    if bootstrapped:
        status = "not_run"
        note = (
            "AEGIS tree is present. Sanity path is still a TODO in "
            "load_aegis_pipeline and run_original_deid_sanity; this stub "
            "does not launch it."
        )
    else:
        status = "bootstrap_required"
        note = (
            "third_party/AEGIS is missing. Run scripts/00_bootstrap_aegis.sh, "
            "then wire the TODOs in src/baselines/b0_aegis_reproduce.py."
        )

    # TODO: when the hooks above are real, call them and fill metrics:
    #   load_aegis_pipeline()
    #   metrics = run_original_deid_sanity()
    write_summary(status, note, bootstrapped)
    print(f"wrote {SUMMARY_JSON.relative_to(REPO_ROOT)}")
    print(f"wrote {SUMMARY_CSV.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

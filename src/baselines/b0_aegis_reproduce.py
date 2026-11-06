#!/usr/bin/env python3
# B0: original AEGIS de-identification sanity check.
# Spec: docs/创新章_交付文档_AEGIS底座.md §6.2 item 1
#       ("AEGIS 原设定复现 sanity check" — de-id metrics, prove the base is wired).
# Matrix: configs/experiment_matrix.yaml baselines.B0
#         ("AEGIS original de-identification").
# Bootstrap with scripts/00_bootstrap_aegis.sh. third_party/AEGIS stays
# gitignored; do not vendor it. This module does not download weights,
# datasets, or secrets, and it does not run the large-dataset eval targets
# (eval_rank_k / eval_verification / eval_utility).

"""Wire B0 to the documented upstream masking entry, or record why it cannot run.

After bootstrap, the upstream README's sanity entry is ``make mask_avatar``
(``scripts/mask_avatar.py`` via the repo ``makefile``). This file shells out
to that make target only when the recipe's local inputs, the default
embedder checkpoint, and a CUDA device are already present. It does not
import the AEGIS package and does not invent a Python API for it.
"""

from __future__ import annotations

import csv
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
AEGIS_ROOT = REPO_ROOT / "third_party" / "AEGIS"
RESULTS_DIR = REPO_ROOT / "results"
SUMMARY_JSON = RESULTS_DIR / "b0_aegis_sanity_summary.json"
SUMMARY_CSV = RESULTS_DIR / "b0_aegis_sanity_summary.csv"

EXPERIMENT_ID = "B0"
EXPERIMENT_DESC = "AEGIS original de-identification"
MAKE_TARGET = "mask_avatar"
MAKE_COMMAND = ["make", MAKE_TARGET]
# Documented default embedder in the mask_avatar recipe (EMBEDDER unset).
# Filename matches upstream src/aegis/models/adaface.py variant ir50.
# Presence is checked only; this module never fetches the file.
ADAFACE_CHECKPOINT = Path("models/adaface_ir50_ms1mv2.ckpt")
# InsightFace buffalo_l detector. Upstream marks this asset auto-download,
# so a missing file blocks launch instead of fetching it.
INSIGHTFACE_DETECTOR = Path("~/.insightface/models/buffalo_l/det_10g.onnx")
SANITY_TIMEOUT_SECONDS = 1800
LOG_TAIL_LINES = 20
LOG_TAIL_CHARS = 1500

_OR_DEFAULT = re.compile(r"\$\(\s*or\s+\$\([^)]+\)\s*,\s*([^)]+?)\s*\)")


def _rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(REPO_ROOT.resolve()).as_posix()
    except ValueError:
        return str(path)


def aegis_bootstrapped() -> bool:
    """True when scripts/00_bootstrap_aegis.sh has cloned the upstream tree."""
    return (AEGIS_ROOT / ".git").is_dir()


def cuda_available() -> bool:
    """True when this process can see a CUDA GPU. Does not import AEGIS."""
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible is not None and visible.strip() in {"", "-1"}:
        return False
    nvidia = shutil.which("nvidia-smi")
    if nvidia:
        try:
            proc = subprocess.run(
                [nvidia, "-L"],
                capture_output=True,
                text=True,
                timeout=15,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired):
            proc = None
        if proc is not None and proc.returncode == 0 and "GPU" in proc.stdout:
            return True
    try:
        import torch
    except ImportError:
        return False
    try:
        return bool(torch.cuda.is_available())
    except Exception:
        return False


def _find_makefile() -> Path | None:
    for name in ("makefile", "Makefile", "GNUmakefile"):
        candidate = AEGIS_ROOT / name
        if candidate.is_file():
            return candidate
    return None


def _expand_make_or(recipe: str) -> str:
    """Replace ``$(or $(VAR),default)`` with ``default`` for local preflight."""
    return _OR_DEFAULT.sub(lambda match: match.group(1).strip(), recipe)


def _extract_recipe(makefile_text: str, target: str) -> str | None:
    lines = makefile_text.splitlines()
    start = None
    pattern = re.compile(rf"^{re.escape(target)}\s*:")
    for index, line in enumerate(lines):
        if pattern.match(line):
            start = index + 1
            break
    if start is None:
        return None
    parts: list[str] = []
    for line in lines[start:]:
        if line.startswith("\t"):
            parts.append(line[1:].rstrip())
        elif not line.strip():
            if parts:
                break
            continue
        else:
            break
    if not parts:
        return None
    joined = "\n".join(parts).replace("\\\n", " ")
    return " ".join(joined.split())


def _flag_value(recipe: str, flag: str) -> str | None:
    match = re.search(rf"{re.escape(flag)}\s+(\S+)", recipe)
    if match is None:
        return None
    return match.group(1)


def _resolve_recipe_path(token: str | None) -> Path | None:
    if not token:
        return None
    path = Path(token)
    if not path.is_absolute():
        path = AEGIS_ROOT / path
    return path


def load_aegis_pipeline() -> dict:
    """Locate the documented ``make mask_avatar`` entry.

    Returns a structured failure (``ok`` false) when ``third_party/AEGIS/.git``
    is absent or the makefile / ``scripts/mask_avatar.py`` entry cannot be
    found. Does not import AEGIS and does not clone or download anything.
    """
    root = _rel(AEGIS_ROOT)
    failure = {
        "ok": False,
        "status": "bootstrap_required",
        "note": (
            "third_party/AEGIS is missing. Run scripts/00_bootstrap_aegis.sh. "
            "The upstream tree stays gitignored."
        ),
        "aegis_root": root,
        "makefile": None,
        "mask_script": None,
        "make_target": MAKE_TARGET,
        "embedder": None,
        "avatar_dir": None,
        "target_image": None,
    }
    if not aegis_bootstrapped():
        return failure

    makefile = _find_makefile()
    mask_script = AEGIS_ROOT / "scripts" / "mask_avatar.py"
    if makefile is None or not mask_script.is_file():
        missing = []
        if makefile is None:
            missing.append("makefile")
        if not mask_script.is_file():
            missing.append("scripts/mask_avatar.py")
        failure["status"] = "not_run"
        failure["note"] = (
            "AEGIS tree is present but the documented sanity entry is missing: "
            + ", ".join(missing)
            + "."
        )
        failure["makefile"] = _rel(makefile) if makefile else None
        failure["mask_script"] = _rel(mask_script) if mask_script.is_file() else None
        return failure

    recipe = _extract_recipe(makefile.read_text(encoding="utf-8"), MAKE_TARGET)
    if recipe is None:
        failure["status"] = "not_run"
        failure["note"] = (
            f"{_rel(makefile)} has no `{MAKE_TARGET}` recipe. "
            "B0 did not invent another entry point."
        )
        failure["makefile"] = _rel(makefile)
        failure["mask_script"] = _rel(mask_script)
        return failure

    expanded = _expand_make_or(recipe)
    avatar_dir = _resolve_recipe_path(_flag_value(expanded, "--avatar-dir"))
    target_image = _resolve_recipe_path(_flag_value(expanded, "--target-image"))
    embedder = _flag_value(expanded, "--embedder")
    return {
        "ok": True,
        "status": "ok",
        "note": (
            "Located documented sanity entry "
            f"`make {MAKE_TARGET}` -> scripts/mask_avatar.py."
        ),
        "aegis_root": root,
        "makefile": _rel(makefile),
        "mask_script": _rel(mask_script),
        "make_target": MAKE_TARGET,
        "embedder": embedder,
        "avatar_dir": avatar_dir,
        "target_image": target_image,
    }


def _log_tail(text: str) -> str:
    lines = [line for line in text.splitlines() if line.strip()]
    tail = "\n".join(lines[-LOG_TAIL_LINES:])
    if len(tail) > LOG_TAIL_CHARS:
        tail = tail[-LOG_TAIL_CHARS:]
    return tail


def _checkpoint_path(embedder: str | None) -> Path | None:
    if embedder == "adaface":
        return AEGIS_ROOT / ADAFACE_CHECKPOINT
    return None


def _collect_blockers(pipeline: dict, *, has_cuda: bool) -> list[str]:
    blockers: list[str] = []
    if shutil.which("make") is None:
        blockers.append("deps_missing: make is not on PATH")
    if shutil.which("uv") is None:
        blockers.append(
            "deps_missing: uv is not on PATH (mask_avatar recipe starts with uv run)"
        )

    avatar_dir = pipeline.get("avatar_dir")
    if not isinstance(avatar_dir, Path):
        blockers.append("deps_missing: mask_avatar recipe has no --avatar-dir")
    else:
        ply = avatar_dir / "point_cloud.ply"
        if not avatar_dir.is_dir() or not ply.is_file():
            blockers.append(
                "deps_missing: recipe avatar input missing "
                f"({_rel(ply)})"
            )

    target_image = pipeline.get("target_image")
    if not isinstance(target_image, Path):
        blockers.append("deps_missing: mask_avatar recipe has no --target-image")
    elif not target_image.is_file():
        blockers.append(
            "deps_missing: recipe target image missing "
            f"({_rel(target_image)})"
        )

    embedder = pipeline.get("embedder")
    checkpoint = _checkpoint_path(embedder if isinstance(embedder, str) else None)
    if checkpoint is None:
        blockers.append(
            "deps_missing: no local checkpoint preflight for embedder "
            f"{embedder!r}; refusing to launch so weights are not downloaded"
        )
    elif not checkpoint.is_file():
        blockers.append(
            "deps_missing: embedder checkpoint missing "
            f"({_rel(checkpoint)}); not downloaded"
        )

    detector = INSIGHTFACE_DETECTOR.expanduser()
    if not detector.is_file():
        blockers.append(
            "deps_missing: insightface detector missing "
            f"({INSIGHTFACE_DETECTOR}); running mask_avatar would auto-download it"
        )

    gaussian_avatars = AEGIS_ROOT / "GaussianAvatars"
    if not gaussian_avatars.is_dir():
        blockers.append(
            "deps_missing: GaussianAvatars checkout missing "
            f"({_rel(gaussian_avatars)})"
        )

    if not has_cuda:
        blockers.append("gpu_required: CUDA device not available")
    return blockers


def _primary_status(blockers: list[str]) -> str:
    codes = [item.split(":", 1)[0] for item in blockers]
    if "deps_missing" in codes:
        return "deps_missing"
    if "gpu_required" in codes:
        return "gpu_required"
    return "ok"


def _demo_hint() -> str:
    demo = AEGIS_ROOT / "datasets/NeRSembleReconst/avatars/306/point_cloud.ply"
    if demo.is_file():
        return (
            " Bootstrap clone includes datasets/NeRSembleReconst avatar 306, "
            "used by other makefile targets; B0 stays on make mask_avatar."
        )
    return ""


def _base_metrics(pipeline: dict, *, has_cuda: bool) -> dict:
    avatar_dir = pipeline.get("avatar_dir")
    target_image = pipeline.get("target_image")
    return {
        "command": " ".join(MAKE_COMMAND),
        "command_attempted": False,
        "return_code": None,
        "log_tail": "",
        "cuda_available": has_cuda,
        "makefile": pipeline.get("makefile"),
        "mask_script": pipeline.get("mask_script"),
        "embedder": pipeline.get("embedder"),
        "avatar_dir": _rel(avatar_dir) if isinstance(avatar_dir, Path) else None,
        "target_image": _rel(target_image) if isinstance(target_image, Path) else None,
        "blockers": [],
    }


def run_original_deid_sanity(pipeline: dict) -> dict:
    """Run one original AEGIS de-id masking path, or record why it did not start.

    The only command launched is ``make mask_avatar`` in the bootstrapped
    tree. Eval targets are not invoked. Metrics keep the command, return
    code, and a short log tail — no new loss and no impersonation scores.
    """
    has_cuda = cuda_available()
    metrics = _base_metrics(pipeline, has_cuda=has_cuda)
    if not pipeline.get("ok"):
        return {
            "status": pipeline.get("status") or "not_run",
            "note": pipeline.get("note") or "AEGIS pipeline is not loadable.",
            "metrics": metrics,
        }

    blockers = _collect_blockers(pipeline, has_cuda=has_cuda)
    metrics["blockers"] = blockers
    if blockers:
        status = _primary_status(blockers)
        reasons = "; ".join(item.split(":", 1)[1].strip() for item in blockers)
        note = (
            "B0 AEGIS original de-identification (§6.2 item 1) did not launch "
            f"`make mask_avatar` ({status}). {reasons}. "
            "Eval targets were not run. No weights or datasets were downloaded."
            + _demo_hint()
        )
        return {"status": status, "note": note, "metrics": metrics}

    metrics["command_attempted"] = True
    try:
        proc = subprocess.run(
            MAKE_COMMAND,
            cwd=AEGIS_ROOT,
            capture_output=True,
            text=True,
            timeout=SANITY_TIMEOUT_SECONDS,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        log = ""
        if isinstance(exc.stdout, str):
            log += exc.stdout
        if isinstance(exc.stderr, str):
            log += exc.stderr
        metrics["return_code"] = None
        metrics["log_tail"] = _log_tail(log)
        return {
            "status": "not_run",
            "note": (
                f"`make mask_avatar` exceeded {SANITY_TIMEOUT_SECONDS}s "
                "and was stopped. Eval targets were not run."
            ),
            "metrics": metrics,
        }
    except OSError as exc:
        metrics["return_code"] = None
        metrics["log_tail"] = str(exc)
        return {
            "status": "deps_missing",
            "note": f"`make mask_avatar` could not be started: {exc}",
            "metrics": metrics,
        }

    log = (proc.stdout or "") + (proc.stderr or "")
    metrics["return_code"] = proc.returncode
    metrics["log_tail"] = _log_tail(log)
    if proc.returncode == 0:
        status = "ok"
        note = (
            "B0 AEGIS original de-identification (§6.2 item 1) finished "
            "`make mask_avatar`. Eval targets were not run."
        )
    else:
        status = "not_run"
        note = (
            f"`make mask_avatar` exited {proc.returncode}. "
            "Eval targets were not run. See metrics.log_tail."
        )
    return {"status": status, "note": note, "metrics": metrics}


def write_summary(
    status: str,
    note: str,
    bootstrapped: bool,
    metrics: dict | None = None,
) -> None:
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
        "metrics": metrics or {},
        "note": note,
    }
    SUMMARY_JSON.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    return_code = None if not metrics else metrics.get("return_code")
    command = "" if not metrics else str(metrics.get("command") or "")
    with SUMMARY_CSV.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "experiment_id",
                "status",
                "bootstrapped",
                "command",
                "return_code",
                "note",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "experiment_id": EXPERIMENT_ID,
                "status": status,
                "bootstrapped": str(bootstrapped).lower(),
                "command": command,
                "return_code": "" if return_code is None else return_code,
                "note": note,
            }
        )


def _skipped_outcome(loaded: dict) -> dict:
    """Summary payload when the documented entry is not ready to launch."""
    return {
        "status": loaded.get("status") or "not_run",
        "note": loaded.get("note") or "AEGIS pipeline is not loadable.",
        "metrics": {
            "command": " ".join(MAKE_COMMAND),
            "command_attempted": False,
            "return_code": None,
            "log_tail": "",
            "cuda_available": cuda_available(),
            "makefile": loaded.get("makefile"),
            "mask_script": loaded.get("mask_script"),
            "embedder": loaded.get("embedder"),
            "blockers": [loaded.get("note") or "pipeline not loadable"],
        },
    }


def main() -> int:
    bootstrapped = aegis_bootstrapped()
    try:
        loaded = load_aegis_pipeline()
        # Run the original masking entry only after bootstrap locates it.
        if bootstrapped and loaded.get("ok"):
            outcome = run_original_deid_sanity(loaded)
        else:
            outcome = _skipped_outcome(loaded)
    except Exception as exc:
        outcome = {
            "status": "not_run",
            "note": f"B0 sanity path failed before a summary could be built: {exc}",
            "metrics": {
                "command": " ".join(MAKE_COMMAND),
                "command_attempted": False,
                "return_code": None,
                "log_tail": "",
                "cuda_available": False,
                "blockers": [f"not_run: {exc}"],
            },
        }

    write_summary(
        outcome["status"],
        outcome["note"],
        bootstrapped,
        outcome.get("metrics"),
    )
    print(f"status={outcome['status']} bootstrapped={str(bootstrapped).lower()}")
    print(f"wrote {SUMMARY_JSON.relative_to(REPO_ROOT)}")
    print(f"wrote {SUMMARY_CSV.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

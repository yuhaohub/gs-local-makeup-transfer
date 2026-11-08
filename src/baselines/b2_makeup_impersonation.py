#!/usr/bin/env python3
# B2: B1 full-face targeted impersonation plus makeup mask M.
# Same attack hyperparameters as B1 / B0-a (AdaFace, eps=0.1, 300 steps,
# target-features DC, linfpgd, radius 1, seed 42). The Foolbox criterion
# stays TargetedMisclassification of match class 1, so PGD raises cosine
# with T. The residual is additionally constrained by the #15 makeup PGD
# hook (kind=dc): supp(delta) ⊆ M and ||delta||_inf ≤ eps.
#
# M is the on-disk parse /root/autodl-fs/gs-assets/parsing/306.npy.
# That directory is parsing/, not masks/. An (H, W) map stays the semantic
# source; dc_hook_from_parsing samples it to one weight per Gaussian.
#
# This module does not download CelebA, LFW, weights, or AEGIS, and it
# does not authorize an AutoDL GPU run. --launch is off unless the caller
# passes it together with --arm-hook, and even then the process starts
# only when the still, the parsing map, the 306 avatar, and CUDA are
# already on disk.

"""Wire B2 (targeted impersonation + makeup mask) without launching by default.

B1's attack module is imported and left on its full-face path. Passing no
B2 flags does not arm a hook and does not start a process.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import subprocess
import sys
import tempfile
import traceback
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

_BASELINES = Path(__file__).resolve().parent
if str(_BASELINES) not in sys.path:
    sys.path.insert(0, str(_BASELINES))

import b1_targeted_impersonation as b1

MODULE_PATH = Path(__file__).resolve()
RESULTS_DIR = REPO_ROOT / "results"
SUMMARY_JSON_NAME = "summary.json"
SUMMARY_CSV_NAME = "summary.csv"

EXPERIMENT_ID = "B2"
EXPERIMENT_DESC = (
    "B1 targeted impersonation + makeup semantic mask M "
    "(kind=dc, supp(delta) within M, ||delta||_inf <= eps)"
)

# Frozen on-disk names. B2 does not introduce another basename.
DEFAULT_IMPERSONATION_IMAGE = b1.DEFAULT_IMPERSONATION_IMAGE
LFW_IMPERSONATION_IMAGE = b1.LFW_IMPERSONATION_IMAGE
DEFAULT_PARSING = "/root/autodl-fs/gs-assets/parsing/306.npy"
IMAGE_ENV = "B2_IMPERSONATION_IMAGE"
PARSING_ENV = "B2_PARSING"
OUTPUT_PREFIX = "NeRSembleMakeup306"
PARSING_BASENAME = "306.npy"
FORBIDDEN_MASK_DIR = "masks"

# Flat files already in results/. B2 must not replace them.
HISTORICAL_SUMMARY_NAMES = frozenset(
    {
        "b0_aegis_sanity_summary.json",
        "b0_aegis_sanity_summary.csv",
        "b1_targeted_impersonation_summary.json",
        "b1_targeted_impersonation_summary.csv",
        "nersemble306_makeup_mask_smoke.json",
        "nersemble306_makeup_mask_smoke.csv",
    }
)

# Float dust from a round-trip. The product gate is outside-M |δ| = 0.
OUTSIDE_GATE = 1e-8


class SummarySlotError(ValueError):
    """The requested summary directory would clobber another experiment."""


class MaskPathError(ValueError):
    """The parsing path is not the frozen ``parsing/306.npy`` contract."""


class MakeupHookRequired(RuntimeError):
    """B2 will not run the attack without the kind=dc makeup hook."""


def b2_criterion_spec() -> dict:
    """B1 targeted-match criterion, with the makeup hook required."""
    spec = b1.b1_criterion_spec()
    spec["makeup_hook"] = True
    spec["hook_kind"] = "dc"
    spec["objective"] = "maximize cos(render, emb(T)) with supp(delta) ⊆ M"
    return spec


def resolve_impersonation_image(flag: str | None) -> str:
    """Flag, then ``B2_IMPERSONATION_IMAGE``, then ``celeba_T0.jpg``.

    ``B1_IMPERSONATION_IMAGE`` is not read. A B1 override must not retarget B2.
    """
    if flag:
        return flag
    env = os.environ.get(IMAGE_ENV)
    if env:
        return env
    return DEFAULT_IMPERSONATION_IMAGE


def resolve_parsing_path(flag: str | Path | None) -> Path:
    """Flag, then ``B2_PARSING``, then ``parsing/306.npy``."""
    if flag:
        return Path(flag).expanduser()
    env = os.environ.get(PARSING_ENV)
    if env:
        return Path(env).expanduser()
    return Path(DEFAULT_PARSING)


def uses_masks_directory(path: Path) -> bool:
    """True when a path component is ``masks`` (the rejected directory name)."""
    return any(part.casefold() == FORBIDDEN_MASK_DIR for part in Path(path).expanduser().parts)


def assert_parsing_path(path: Path) -> None:
    """Require basename ``306.npy`` and reject a ``masks/`` directory.

    The directory may differ so a test can point at a copy. The frozen
    default is ``/root/autodl-fs/gs-assets/parsing/306.npy``.

    Raises:
        MaskPathError: the path uses ``masks/`` or a different basename.
    """
    if uses_masks_directory(path):
        raise MaskPathError(
            "B2 refuses a makeup mask under a masks/ directory. "
            f"The frozen file is {DEFAULT_PARSING}."
        )
    if path.name != PARSING_BASENAME:
        raise MaskPathError(
            f"Makeup mask basename must be {PARSING_BASENAME}, got {path.name}. "
            f"The frozen file is {DEFAULT_PARSING}."
        )


def summary_stem(path: Path) -> str:
    """Still name used in ``results/b2_<T>/``."""
    if path.stem in {"celeba_T0", "lfw_T0"}:
        return path.stem
    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", path.stem).strip("._") or "T"
    return stem


def summary_slot(path: Path) -> str:
    return f"b2_{summary_stem(path)}"


def default_summary_dir(image: Path) -> Path:
    """One still, one directory. Never the flat ``results/`` folder."""
    return RESULTS_DIR / summary_slot(image)


def resolve_summary_dir(flag: str | None, image: Path) -> Path:
    if flag is None:
        return default_summary_dir(image)
    path = Path(flag).expanduser()
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path


def assert_summary_slot(directory: Path) -> None:
    """Refuse the shared ``results/`` folder and the historical flat names.

    B2 writes ``summary.json`` and ``summary.csv`` inside ``results/b2_<T>/``
    (or another directory the caller names). It does not replace
    ``results/b0_aegis_sanity_summary.*``,
    ``results/b1_targeted_impersonation_summary.*``, or
    ``results/nersemble306_makeup_mask_smoke.*``.

    Raises:
        SummarySlotError: ``directory`` is that shared folder or a historical name.
    """
    resolved = directory.resolve()
    results = RESULTS_DIR.resolve()
    if resolved == results:
        raise SummarySlotError(
            "B2 writes results/b2_<T>/summary.json and summary.csv "
            "(for example results/b2_celeba_T0/). "
            "Refusing the flat results/ directory so historical summaries stay put."
        )
    historical = {results / name for name in HISTORICAL_SUMMARY_NAMES}
    if resolved in historical:
        raise SummarySlotError(
            f"Refusing to write B2 over the historical summary {resolved.name}."
        )
    if resolved.name in HISTORICAL_SUMMARY_NAMES or resolved.stem in {
        Path(name).stem for name in HISTORICAL_SUMMARY_NAMES
    }:
        raise SummarySlotError(
            f"Refusing summary directory {resolved.name}: "
            "that name is a historical flat summary."
        )


def output_name_for(path: Path) -> str:
    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", path.stem).strip("._") or "T"
    return f"{OUTPUT_PREFIX}_{stem}"


def b2_aegis_kwargs(plan: b1.AttackPlan) -> dict:
    """B1 keyword arguments with a B2 output name.

    Hyperparameters stay on the B0-a / B1 Foolbox path: no adaptive epsilon,
    no FLAME ``selected_regions``, DC, linfpgd, AdaFace.
    """
    kwargs = dict(plan.aegis_kwargs())
    kwargs["output_name"] = output_name_for(plan.impersonation_image)
    kwargs["adaptive_epsilon"] = False
    kwargs["selected_regions"] = []
    kwargs["surrogate_keys"] = None
    kwargs["targeted_features"] = b1.TARGET_FEATURES
    kwargs["adv_attack"] = b1.ADV_ATTACK
    kwargs["attack_steps"] = b1.ATTACK_STEPS
    kwargs["embedder_name"] = b1.EMBEDDER
    kwargs["epsilons"] = [b1.EPSILON]
    return kwargs


def expected_outputs(plan: b1.AttackPlan, aegis_root: Path) -> tuple[Path, Path]:
    name = output_name_for(plan.impersonation_image)
    base = (
        Path(aegis_root)
        / "datasets"
        / f"seed{plan.seed}"
        / f"{name}_{plan.embedder}_all"
        / f"eps_{plan.epsilon:.3f}"
    )
    avatar = base / "avatars" / plan.subject / "point_cloud.ply"
    render = base / "renders" / f"{plan.subject}.png"
    return avatar, render


def user_command(image: Path, parsing: Path) -> str:
    """The GPU command. The default CLI does not pass ``--launch``."""
    return (
        "python src/baselines/b2_makeup_impersonation.py "
        f"--impersonation-image {image} "
        f"--parsing {parsing} "
        "--arm-hook --launch"
    )


def _rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(REPO_ROOT.resolve()).as_posix()
    except (OSError, ValueError):
        return str(path)


def _now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def _as_float_array(value):
    from makeup._numpy import require_numpy

    np = require_numpy()
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        try:
            return np.asarray(value.numpy(), dtype=float)
        except Exception:
            pass
    return np.asarray(value, dtype=float)


def _dc_colors(values):
    """Per-Gaussian DC colors as ``(N, 3)``.

    Clean ``att_tensor`` is ``(N, 1, 3)``. Foolbox unsqueezes that tensor,
    so the saved adversarial features are ``(1, N, 1, 3)``. Adaptive DC
    after the singleton squeeze is ``(N, 3)``. All three are the same
    Gaussian order.
    """
    from makeup.aegis_pgd_hook import _reshape_residual

    colors, _restore = _reshape_residual(values, "dc")
    return colors


def outside_m_abs_delta(clean, adv, mask) -> float:
    """Max ``|adv - clean|`` outside ``M``. The B2 gate is 0.

    ``(N, 3)``, ``(N, 1, 3)``, and ``(1, N, 1, 3)`` may be mixed. Each
    side is squeezed to ``(N, 3)`` before the subtraction, so a clean
    ``att_tensor`` and a Foolbox batch still score the same Gaussians.
    A different Gaussian count still raises.

    Raises:
        ValueError: a side is not one of those DC layouts, ``N`` differs,
            or the mask does not match.
        ImportError: NumPy is not installed.
    """
    from makeup.aegis_pgd_hook import outside_support_max_abs

    origin = _as_float_array(clean)
    updated = _as_float_array(adv)
    try:
        origin_colors = _dc_colors(origin)
        updated_colors = _dc_colors(updated)
    except ValueError as exc:
        raise ValueError(
            f"clean and adversarial DC shapes differ: {origin.shape} vs {updated.shape}"
        ) from exc
    if origin_colors.shape != updated_colors.shape:
        raise ValueError(
            f"clean and adversarial DC shapes differ: {origin.shape} vs {updated.shape}"
        )
    delta = updated_colors - origin_colors
    return float(outside_support_max_abs(delta, mask, "dc"))


def gate_holds(metrics: dict) -> bool:
    """True when pipeline, both cosines, and outside-M ``|δ| = 0`` are present.

    A missing measurement stays a failure. This does not invent a number.
    """
    if metrics.get("return_code") != 0:
        return False
    if metrics.get("cos_adv_source") is None or metrics.get("cos_adv_target") is None:
        return False
    outside = metrics.get("outside_m_abs_delta")
    if outside is None:
        return False
    return float(outside) <= OUTSIDE_GATE


def intended_hook(parsing: Path) -> dict:
    """Hook contract recorded before a parse is loaded."""
    return {
        "kind": "dc",
        "epsilon": b1.EPSILON,
        "subject": b1.SUBJECT,
        "parsing_path": str(parsing),
        "parsing_present": b1.image_ready(parsing),
        "mask_dirname_forbidden": FORBIDDEN_MASK_DIR,
        "semantic_source": None,
        "dc_mask_lift": None,
        "mask_shape": None,
        "means_source": None,
        "armed": False,
        "attach_status": "not_requested",
        "config": None,
        "layout": (
            "(H, W) parsing is the semantic source; "
            "kind=dc samples M to (N,) at Gaussian means"
        ),
    }


def hook_config_path(aegis_root: Path) -> Path:
    from makeup.aegis_pgd_hook import CONFIG_NAME

    return Path(aegis_root) / CONFIG_NAME


def build_dc_hook(parsing: Path, aegis_root: Path):
    """Build the #15 DC hook from a real ``306.npy``. No documented fallback.

    Returns:
        ``(hook, means_source)``. ``means_source`` is ``ply-xyz``,
        ``ply-missing``, ``flame-binding-deferred``, or ``per-gaussian-labels``.

    Raises:
        FileNotFoundError: the npy is not on disk.
        ValueError: the array is not a parsing map the DC hook can arm.
        ImportError: NumPy is not installed.
    """
    from makeup.aegis_mask import load_b0a_dc_means, load_parsing_npy
    from makeup.aegis_pgd_hook import dc_hook_from_parsing

    labels = load_parsing_npy(parsing)
    means = None
    means_source = "per-gaussian-labels"
    if getattr(labels, "ndim", 0) == 2:
        means, means_source = load_b0a_dc_means(aegis_root, b1.SUBJECT)
    hook = dc_hook_from_parsing(labels, b1.EPSILON, means, subject=b1.SUBJECT)
    if hook.kind != "dc":
        raise ValueError(f"B2 makeup hook kind must be dc, got {hook.kind}")
    return hook, means_source


def describe_hook(hook, means_source: str, attach: dict) -> dict:
    from makeup.mask import as_mask_array

    weight = as_mask_array(hook.mask)
    return {
        "kind": hook.kind,
        "epsilon": hook.epsilon,
        "subject": hook.subject,
        "semantic_source": hook.semantic_source,
        "dc_mask_lift": hook.dc_mask_lift,
        "mask_shape": [int(dim) for dim in weight.shape],
        "means_source": means_source,
        "armed": bool(attach.get("armed")),
        "attach_status": attach.get("status"),
        "config": attach.get("config"),
        "layout": (
            "(H, W) parsing is the semantic source; "
            "kind=dc samples M to (N,) at Gaussian means"
        ),
    }


def install_b2_hook(hook, aegis_root: Path) -> dict:
    """Arm ``hook`` in an existing AEGIS tree. A missing tree is ``tree_missing``.

    ``mask_source`` is ``from_parsing``. The documented fallback grid is not
    passed, so the installer cannot accept it.
    """
    from makeup.aegis_pgd_hook import install_makeup_pgd_hook

    return install_makeup_pgd_hook(hook, aegis_root, mask_source="from_parsing")


def armed_environment(config_path: Path, aegis_root: Path, base: dict | None = None) -> dict:
    """Environment for the AEGIS worker. The makeup config must be a real file.

    ``B1_MAKEUP_HOOK=off`` is removed. Leaving it set would ask the worker to
    disarm the mask B2 exists to apply.
    """
    from makeup.aegis_pgd_hook import HOOK_CONFIG_ENV

    env = dict(os.environ if base is None else base)
    env.pop("B1_MAKEUP_HOOK", None)
    env[HOOK_CONFIG_ENV] = str(config_path)
    env["B2_MAKEUP_HOOK"] = "on"
    src = str(Path(aegis_root) / "src")
    repo = str(REPO_ROOT)
    previous = env.get("PYTHONPATH", "")
    parts = [repo, src]
    if previous:
        parts.append(previous)
    env["PYTHONPATH"] = os.pathsep.join(parts)
    return env


def assert_makeup_armed(env: dict) -> None:
    """Raise unless this process will load a kind=dc hook config."""
    from makeup.aegis_pgd_hook import HOOK_CONFIG_ENV

    if env.get("B1_MAKEUP_HOOK") == "off":
        raise MakeupHookRequired(
            "B1_MAKEUP_HOOK=off disarms M. B2 refuses that combination."
        )
    if env.get("B2_MAKEUP_HOOK") != "on":
        raise MakeupHookRequired("B2_MAKEUP_HOOK must be on.")
    value = env.get(HOOK_CONFIG_ENV, "")
    path = Path(value).expanduser() if value else None
    if path is None or not path.is_file():
        raise MakeupHookRequired(
            f"{HOOK_CONFIG_ENV} must point at an existing makeup hook config."
        )
    try:
        meta = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise MakeupHookRequired(f"makeup hook config is not readable json: {exc}") from exc
    if meta.get("kind") != "dc":
        raise MakeupHookRequired(
            f"B2 hook kind must be dc, got {meta.get('kind')!r}."
        )
    if str(meta.get("subject")) != b1.SUBJECT:
        raise MakeupHookRequired(
            f"B2 hook subject must be {b1.SUBJECT}, got {meta.get('subject')!r}."
        )


def worker_command(
    plan: b1.AttackPlan,
    parsing: Path,
    summary_out: Path,
    aegis_root: Path,
) -> list[str]:
    return [
        "uv",
        "run",
        "python",
        str(MODULE_PATH),
        "--worker",
        "--arm-hook",
        "--impersonation-image",
        str(plan.impersonation_image),
        "--parsing",
        str(parsing),
        "--aegis-root",
        str(aegis_root),
        "--summary-out",
        str(summary_out),
    ]


def start_worker(plan, parsing, summary_out, config_path, aegis_root, runner, env=None):
    """Spawn the uv worker with the makeup hook armed."""
    disarm_check = armed_environment(config_path, aegis_root) if env is None else env
    assert_makeup_armed(disarm_check)
    command = worker_command(plan, parsing, summary_out, aegis_root)
    joined = " ".join(command)
    if "--refuse-hook" in command or "--adaptive-epsilon" in joined or "--select-regions" in joined:
        raise MakeupHookRequired(
            "B2 worker command must keep the dc hook and the B1 hyperparameters."
        )
    if "--arm-hook" not in command:
        raise MakeupHookRequired("B2 worker command must include --arm-hook.")
    return runner(
        command,
        cwd=str(aegis_root),
        env=disarm_check,
        capture_output=True,
        text=True,
        timeout=b1.LAUNCH_TIMEOUT_SECONDS,
        check=False,
    )


def collect_b2_blockers(plan: b1.AttackPlan, parsing: Path, config_path: Path | None) -> list[str]:
    """Reasons ``--launch`` must not start. Empty only when the attack may run."""
    if not b1.image_ready(plan.impersonation_image):
        return [
            "awaiting_target: impersonation still is not on disk "
            f"({plan.impersonation_image}). First smoke is {DEFAULT_IMPERSONATION_IMAGE}. "
            f"Later run is {LFW_IMPERSONATION_IMAGE}."
        ]
    if not b1.image_ready(parsing):
        return [
            "awaiting_parsing: makeup mask is not on disk "
            f"({parsing}). Frozen path is {DEFAULT_PARSING}. "
            "B2 does not build a documented_fallback mask."
        ]
    blockers = list(b1.collect_blockers(plan))
    if blockers:
        return blockers
    if config_path is None or not Path(config_path).is_file():
        return [
            "hook_not_armed: B2 --launch needs the kind=dc makeup hook "
            "installed (--arm-hook) and a config file on disk."
        ]
    try:
        meta = json.loads(Path(config_path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return [f"hook_not_armed: makeup hook config is unreadable ({exc})."]
    if meta.get("kind") != "dc":
        return [
            "hook_not_armed: installed makeup hook kind is "
            f"{meta.get('kind')!r}, B2 requires dc."
        ]
    return []


def _primary_status(blockers: list[str]) -> str:
    if not blockers:
        return "ok"
    codes = [item.split(":", 1)[0] for item in blockers]
    for code in (
        "awaiting_target",
        "awaiting_parsing",
        "invalid_image",
        "invalid_parsing",
        "refused_source_image",
        "bootstrap_required",
        "deps_missing",
        "hook_not_armed",
        "gpu_required",
    ):
        if code in codes:
            return code
    return codes[0]


def _empty_metrics(plan: b1.AttackPlan, parsing: Path, summary_dir: Path, aegis_root: Path, gpu) -> dict:
    avatar_out, render_out = expected_outputs(plan, aegis_root)
    return {
        "command": user_command(plan.impersonation_image, parsing),
        "command_attempted": False,
        "return_code": None,
        "log_tail": "",
        "cuda_available": b1.cuda_available(),
        "gpu": gpu,
        "embedder": plan.embedder,
        "epsilon": plan.epsilon,
        "attack_steps": plan.attack_steps,
        "target_features": plan.target_features,
        "adv_attack": plan.adv_attack,
        "radius": plan.radius,
        "seed": plan.seed,
        "ver_threshold": plan.ver_threshold,
        "camera_boundary_angles": list(plan.camera_boundary_angles),
        "angle_aggregation": plan.angle_aggregation,
        "k_angles": plan.k_angles,
        "adaptive_epsilon": False,
        "selected_regions": [],
        "output_name": output_name_for(plan.impersonation_image),
        "avatar_dir": _rel(plan.avatar_dir),
        "source_image": _rel(plan.source_image),
        "impersonation_image": str(plan.impersonation_image),
        "impersonation_image_present": b1.image_ready(plan.impersonation_image),
        "lfw_image": LFW_IMPERSONATION_IMAGE,
        "parsing_path": str(parsing),
        "parsing_present": b1.image_ready(parsing),
        "corpus": b1.corpus_for(plan.impersonation_image),
        "smoke_target": b1.smoke_target_for(plan.impersonation_image),
        "summary_slot": summary_slot(plan.impersonation_image),
        "criterion": b2_criterion_spec(),
        "reference_embedding": "impersonation_image",
        "cos_adv_source": None,
        "cos_adv_target": None,
        "outside_m_abs_delta": None,
        "outside_m_abs_delta_gate": 0,
        "aggregated_cosine_to_target": None,
        "makeup_hook_armed": False,
        "hook": intended_hook(parsing),
        "aegis_root": _rel(aegis_root),
        "expected_output_avatar": _rel(avatar_out),
        "expected_output_render": _rel(render_out),
        "blockers": [],
    }


def _payload(plan, parsing, summary_dir, *, status, note, metrics, pipeline_ok=False) -> dict:
    return {
        "experiment_id": EXPERIMENT_ID,
        "desc": EXPERIMENT_DESC,
        "status": status,
        "pipeline_ok": pipeline_ok,
        "subject": plan.subject,
        "corpus": b1.corpus_for(plan.impersonation_image),
        "smoke_target": b1.smoke_target_for(plan.impersonation_image),
        "summary_slot": summary_slot(plan.impersonation_image),
        "summary_dir": _rel(summary_dir),
        "parsing_path": str(parsing),
        "bootstrapped": b1.aegis_bootstrapped(),
        "recorded_at": _now(),
        "metrics": metrics,
        "note": note,
    }


def write_summary(summary_dir: Path, payload: dict) -> tuple[Path, Path]:
    """Write ``summary.json`` and ``summary.csv`` inside ``summary_dir`` only."""
    assert_summary_slot(summary_dir)
    summary_dir.mkdir(parents=True, exist_ok=True)
    json_path = summary_dir / SUMMARY_JSON_NAME
    csv_path = summary_dir / SUMMARY_CSV_NAME
    names = {json_path.name, csv_path.name}
    if names & HISTORICAL_SUMMARY_NAMES:
        raise SummarySlotError("B2 summary filenames collided with a historical summary.")
    json_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    metrics = payload.get("metrics") or {}
    hook = metrics.get("hook") or {}
    return_code = metrics.get("return_code")
    cos_source = metrics.get("cos_adv_source")
    cos_target = metrics.get("cos_adv_target")
    outside = metrics.get("outside_m_abs_delta")
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "experiment_id",
                "status",
                "subject",
                "pipeline_ok",
                "return_code",
                "cos_adv_source",
                "cos_adv_target",
                "outside_m_abs_delta",
                "gpu",
                "makeup_hook_armed",
                "hook_kind",
                "parsing_path",
                "impersonation_image",
                "summary_dir",
                "command",
                "note",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "experiment_id": payload.get("experiment_id", ""),
                "status": payload.get("status", ""),
                "subject": payload.get("subject", ""),
                "pipeline_ok": str(bool(payload.get("pipeline_ok"))).lower(),
                "return_code": "" if return_code is None else return_code,
                "cos_adv_source": "" if cos_source is None else cos_source,
                "cos_adv_target": "" if cos_target is None else cos_target,
                "outside_m_abs_delta": "" if outside is None else outside,
                "gpu": metrics.get("gpu") or "",
                "makeup_hook_armed": str(bool(metrics.get("makeup_hook_armed"))).lower(),
                "hook_kind": hook.get("kind") or "",
                "parsing_path": metrics.get("parsing_path") or payload.get("parsing_path") or "",
                "impersonation_image": metrics.get("impersonation_image") or "",
                "summary_dir": payload.get("summary_dir") or "",
                "command": metrics.get("command") or "",
                "note": payload.get("note") or "",
            }
        )
    return json_path, csv_path


def _print_summary_paths(summary_dir: Path, status: str) -> None:
    print(f"status={status}")
    print(f"wrote {summary_dir / SUMMARY_JSON_NAME}")
    print(f"wrote {summary_dir / SUMMARY_CSV_NAME}")


def _log_tail(text: str) -> str:
    lines = [line for line in text.splitlines() if line.strip()]
    tail = "\n".join(lines[-b1.LOG_TAIL_LINES:])
    if len(tail) > b1.LOG_TAIL_CHARS:
        tail = tail[-b1.LOG_TAIL_CHARS:]
    return tail


def _read_worker_payload(path: Path) -> dict:
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def _dry_status(plan: b1.AttackPlan, parsing: Path) -> str:
    if not b1.image_ready(plan.impersonation_image):
        return "awaiting_target"
    if not b1.image_ready(parsing):
        return "awaiting_parsing"
    return "not_launched"


def _dry_note(plan: b1.AttackPlan, parsing: Path) -> str:
    return (
        "B2 is B1 targeted impersonation plus makeup mask M (kind=dc). "
        "Wired and not launched. "
        f"First T is {DEFAULT_IMPERSONATION_IMAGE}. "
        f"Later T is {LFW_IMPERSONATION_IMAGE}. "
        f"Makeup mask is {parsing} (parsing/306.npy, not masks/). "
        "Gate metrics pipeline_ok, cos(adv, source), cos(adv, T), and "
        "outside-M |delta| stay unset until a real run. "
        "Pass --arm-hook --launch only after those files are on disk and "
        "docs/gpu_launch_checklist.md is green. "
        "This default command does not authorize an AutoDL run. "
        f"Still on disk: {str(b1.image_ready(plan.impersonation_image)).lower()}. "
        f"Parsing on disk: {str(b1.image_ready(parsing)).lower()}."
    )


def _refuse(summary_dir, plan, parsing, metrics, status: str, note: str) -> int:
    metrics["blockers"] = [f"{status}: {note}"]
    metrics["command_attempted"] = False
    write_summary(
        summary_dir,
        _payload(plan, parsing, summary_dir, status=status, note=note, metrics=metrics),
    )
    print(note, file=sys.stderr)
    _print_summary_paths(summary_dir, status)
    return 2


def _clear_hook_state() -> None:
    try:
        from makeup.aegis_pgd_hook import clear_hook_state
    except ImportError:
        return
    clear_hook_state()


def execute(argv: list[str] | None = None, *, runner=subprocess.run) -> int:
    """Parse CLI, write the B2 summary, and launch only when asked and armed."""
    args = parse_args(argv)
    if args.worker:
        return worker_main(args)

    image_text = resolve_impersonation_image(args.impersonation_image)
    parsing = resolve_parsing_path(args.parsing)
    summary_dir = resolve_summary_dir(args.summary_dir, Path(image_text))
    try:
        assert_summary_slot(summary_dir)
    except SummarySlotError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    try:
        from makeup.aegis_mask import assert_subject_306

        assert_subject_306(args.subject)
    except ValueError as exc:
        plan = b1.build_plan(image_text, launch=False)
        metrics = _empty_metrics(plan, parsing, summary_dir, Path(args.aegis_root), b1.gpu_name())
        return _refuse(summary_dir, plan, parsing, metrics, "refused_subject", str(exc))

    plan = b1.build_plan(image_text, launch=False)
    aegis_root = Path(args.aegis_root).expanduser()
    metrics = _empty_metrics(plan, parsing, summary_dir, aegis_root, b1.gpu_name())
    try:
        return _execute_plan(
            args,
            plan,
            parsing,
            summary_dir,
            aegis_root,
            metrics,
            runner=runner,
        )
    finally:
        _clear_hook_state()


def _execute_plan(args, plan, parsing, summary_dir, aegis_root, metrics, *, runner) -> int:
    if args.adaptive_epsilon or args.select_regions:
        note = (
            "Refused hyperparameter override. B2 keeps the B1/B0-a Foolbox path: "
            "AdaFace, eps=0.1, 300 steps, target-features DC, linfpgd, "
            "adaptive epsilon off, no FLAME --select-regions. "
            "--select-regions is not makeup mask M."
        )
        return _refuse(
            summary_dir, plan, parsing, metrics, "refused_hyperparams", note
        )

    if args.arm_hook and args.refuse_hook:
        note = (
            "Refused --arm-hook together with --refuse-hook. "
            "Pass one switch. B2 arms the kind=dc makeup hook, or it stops."
        )
        return _refuse(
            summary_dir, plan, parsing, metrics, "refused_conflicting_hook", note
        )

    if args.refuse_hook:
        note = (
            "Refused --refuse-hook. B2 is B1 plus makeup mask M. "
            "It will not run the full-face attack with the makeup hook off. "
            "Use src/baselines/b1_targeted_impersonation.py for the unmasked path."
        )
        return _refuse(
            summary_dir, plan, parsing, metrics, "refused_makeup_hook", note
        )

    try:
        assert_parsing_path(parsing)
    except MaskPathError as exc:
        return _refuse(
            summary_dir, plan, parsing, metrics, "refused_mask_path", str(exc)
        )

    try:
        b1.reference_image_for_run(plan)
    except ValueError as exc:
        return _refuse(
            summary_dir, plan, parsing, metrics, "refused_source_image", str(exc)
        )

    if args.launch and not args.arm_hook:
        note = (
            "Refused --launch without --arm-hook. "
            "That combination is full-face B1, not B2. "
            "B2 starts the GPU attack only with the kind=dc makeup hook armed."
        )
        return _refuse(
            summary_dir, plan, parsing, metrics, "refused_unarmed_launch", note
        )

    attach = None
    if args.arm_hook:
        if not b1.image_ready(parsing):
            note = (
                "B2 --arm-hook needs the makeup mask on disk "
                f"({parsing}). Frozen path is {DEFAULT_PARSING}. "
                "No documented_fallback labels were built. GPU was not started."
            )
            return _refuse(
                summary_dir, plan, parsing, metrics, "awaiting_parsing", note
            )
        try:
            hook, means_source = build_dc_hook(parsing, aegis_root)
            attach = install_b2_hook(hook, aegis_root)
        except (ValueError, OSError, ImportError) as exc:
            note = f"B2 could not build the kind=dc hook: {exc}"
            return _refuse(
                summary_dir, plan, parsing, metrics, "invalid_parsing", note
            )
        spec = describe_hook(hook, means_source, attach)
        spec["parsing_path"] = str(parsing)
        spec["parsing_present"] = True
        spec["mask_dirname_forbidden"] = FORBIDDEN_MASK_DIR
        metrics["hook"] = spec
        metrics["makeup_hook_armed"] = bool(attach.get("armed"))
        if attach.get("status") == "anchors_mismatch":
            note = attach.get("note") or "AEGIS anchors moved. The makeup hook was not armed."
            return _refuse(
                summary_dir, plan, parsing, metrics, "anchors_mismatch", note
            )
        if attach.get("status") == "refused_fallback":
            note = attach.get("note") or "Refused to arm a fallback mask."
            return _refuse(
                summary_dir, plan, parsing, metrics, "refused_fallback", note
            )

    if not args.launch:
        if args.arm_hook and attach is not None and attach.get("armed"):
            status = "hook_armed"
            note = (
                "B2 armed the kind=dc makeup hook and did not launch. "
                f"Config: {attach.get('config')}. "
                "GPU stays off until --launch and the GPU checklist is green. "
                "outside-M |delta|, cos(adv, source), and cos(adv, T) were not measured."
            )
        elif args.arm_hook and attach is not None:
            status = str(attach.get("status") or "tree_missing")
            note = (
                "B2 built the kind=dc hook and did not launch. "
                + str(attach.get("note") or "The AEGIS tree was not patched.")
                + " No GPU numbers were recorded."
            )
        else:
            status = _dry_status(plan, parsing)
            note = _dry_note(plan, parsing)
            if status != "not_launched":
                metrics["blockers"] = [f"{status}: {note}"]
        write_summary(
            summary_dir,
            _payload(plan, parsing, summary_dir, status=status, note=note, metrics=metrics),
        )
        _print_summary_paths(summary_dir, status)
        return 0

    config_path = None
    if attach and attach.get("config"):
        config_path = Path(attach["config"])
    blockers = collect_b2_blockers(plan, parsing, config_path)
    if blockers:
        status = _primary_status(blockers)
        metrics["blockers"] = blockers
        note = (
            "B2 did not launch. "
            + "; ".join(item.split(":", 1)[1].strip() for item in blockers)
        )
        write_summary(
            summary_dir,
            _payload(plan, parsing, summary_dir, status=status, note=note, metrics=metrics),
        )
        print(note, file=sys.stderr)
        _print_summary_paths(summary_dir, status)
        return 2

    fd, worker_name = tempfile.mkstemp(prefix="b2-worker-", suffix=".json")
    os.close(fd)
    worker_path = Path(worker_name)
    command = worker_command(plan, parsing, worker_path, aegis_root)
    metrics["command_attempted"] = True
    metrics["worker_command"] = " ".join(command)
    proc = None
    try:
        proc = start_worker(
            plan,
            parsing,
            worker_path,
            config_path,
            aegis_root,
            runner,
        )
    except MakeupHookRequired as exc:
        metrics["command_attempted"] = False
        return _refuse(
            summary_dir, plan, parsing, metrics, "refused_makeup_hook", str(exc)
        )
    except subprocess.TimeoutExpired as exc:
        log = ""
        if isinstance(exc.stdout, str):
            log += exc.stdout
        if isinstance(exc.stderr, str):
            log += exc.stderr
        metrics["return_code"] = None
        metrics["log_tail"] = _log_tail(log)
        note = f"B2 worker exceeded {b1.LAUNCH_TIMEOUT_SECONDS}s and was stopped."
        write_summary(
            summary_dir,
            _payload(plan, parsing, summary_dir, status="not_run", note=note, metrics=metrics),
        )
        return 1
    except OSError as exc:
        metrics["return_code"] = None
        metrics["log_tail"] = str(exc)
        note = f"B2 worker could not be started: {exc}"
        write_summary(
            summary_dir,
            _payload(plan, parsing, summary_dir, status="deps_missing", note=note, metrics=metrics),
        )
        return 2
    finally:
        if proc is None and worker_path.exists():
            worker_path.unlink()

    log = (getattr(proc, "stdout", "") or "") + (getattr(proc, "stderr", "") or "")
    metrics["return_code"] = getattr(proc, "returncode", None)
    metrics["log_tail"] = _log_tail(log)
    worker_payload = _read_worker_payload(worker_path)
    if worker_path.exists():
        worker_path.unlink()
    for key in (
        "cos_adv_source",
        "cos_adv_target",
        "outside_m_abs_delta",
        "aggregated_cosine_to_target",
        "output_avatar",
        "output_render",
        "output_avatar_bytes",
        "output_render_bytes",
    ):
        if key in worker_payload and worker_payload[key] is not None:
            metrics[key] = worker_payload[key]
    if worker_payload.get("gpu"):
        metrics["gpu"] = worker_payload["gpu"]
    parsed = _parse_aggregated_cosine(log)
    if metrics["aggregated_cosine_to_target"] is None and parsed is not None:
        metrics["aggregated_cosine_to_target"] = parsed

    return_code = metrics["return_code"]
    outside = metrics.get("outside_m_abs_delta")
    if gate_holds(metrics):
        status = "pipeline_ok"
        pipeline_ok = True
        note = (
            "B2 targeted impersonation with makeup mask M finished. "
            f"cos(adv, source)={metrics['cos_adv_source']:.4f}, "
            f"cos(adv, T)={metrics['cos_adv_target']:.4f}, "
            f"outside-M |delta|={float(outside):.6g}."
        )
        exit_code = 0
    elif (
        return_code == 0
        and metrics["cos_adv_source"] is not None
        and metrics["cos_adv_target"] is not None
        and outside is not None
    ):
        status = "support_failed"
        pipeline_ok = False
        note = (
            "B2 attack exited 0 but outside-M |delta| is "
            f"{outside}, not 0. pipeline_ok stays false."
        )
        exit_code = 1
    elif return_code == 0:
        status = "scoring_failed"
        pipeline_ok = False
        note = (
            "B2 attack process exited 0 but cos(adv, source), cos(adv, T), "
            "and outside-M |delta| were not all recorded. "
            + str(worker_payload.get("error") or "See metrics.log_tail.")
        )
        exit_code = 1
    else:
        status = "not_run"
        pipeline_ok = False
        note = (
            f"B2 worker exited {return_code}. "
            + str(worker_payload.get("error") or "See metrics.log_tail.")
        )
        exit_code = return_code if isinstance(return_code, int) and return_code != 0 else 1

    write_summary(
        summary_dir,
        _payload(
            plan,
            parsing,
            summary_dir,
            status=status,
            note=note,
            metrics=metrics,
            pipeline_ok=pipeline_ok,
        ),
    )
    _print_summary_paths(summary_dir, status)
    return exit_code


def _parse_aggregated_cosine(text: str):
    matches = b1.AGGREGATED_COSINE_RE.findall(text)
    if not matches:
        return None
    return float(matches[-1])


def run_attack(plan: b1.AttackPlan, config_path: Path, aegis_root: Path) -> dict:
    """Import AEGIS, keep B1's targeted criterion, and require the DC hook.

    Called from ``--worker`` inside ``uv run``. The makeup hook config must
    already be installed. This function does not clone AEGIS.
    """
    env = armed_environment(config_path, aegis_root)
    assert_makeup_armed(env)
    from makeup.aegis_pgd_hook import HOOK_CONFIG_ENV, clear_hook_state, current_hook

    os.environ[HOOK_CONFIG_ENV] = env[HOOK_CONFIG_ENV]
    os.environ["B2_MAKEUP_HOOK"] = "on"
    os.environ.pop("B1_MAKEUP_HOOK", None)

    src = Path(aegis_root) / "src"
    if src.is_dir() and str(src) not in sys.path:
        sys.path.insert(0, str(src))

    import torch
    from aegis import AEGIS

    clear_hook_state()
    b1.reference_image_for_run(plan)
    original = b1.apply_targeted_criterion(AEGIS, b1.make_foolbox_targeted_match)
    try:
        model = AEGIS(**b2_aegis_kwargs(plan))
        clean = model.att_tensor.detach().clone()
        captured: dict = {}
        original_save = model.save_results

        def save_results(adv_features, *args, **kwargs):
            captured["adv"] = adv_features
            return original_save(adv_features, *args, **kwargs)

        model.save_results = save_results
        import contextlib
        import io

        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            model.run()
        printed = buffer.getvalue()
        sys.stdout.write(printed)
        avatar_out, render_out = expected_outputs(plan, aegis_root)
        result = {
            "ok": True,
            "cos_adv_source": None,
            "cos_adv_target": None,
            "outside_m_abs_delta": None,
            "aggregated_cosine_to_target": _parse_aggregated_cosine(printed),
            "gpu": b1.gpu_name(),
            "output_avatar": _rel(avatar_out) if avatar_out.is_file() else None,
            "output_render": _rel(render_out) if render_out.is_file() else None,
            "output_avatar_bytes": avatar_out.stat().st_size if avatar_out.is_file() else None,
            "output_render_bytes": render_out.stat().st_size if render_out.is_file() else None,
            "error": None,
        }
        try:
            features = b1._first_epsilon(captured["adv"])
            with torch.no_grad():
                rgb = model.render_frame_in_rgb(features)
                adv_embedding = model.ver_model(rgb)
            source_embedding = model._get_reference_id_embedding(str(plan.source_image))
            target_embedding = model.ref_emb
            result["cos_adv_source"] = b1.cosine_similarity(adv_embedding, source_embedding)
            result["cos_adv_target"] = b1.cosine_similarity(adv_embedding, target_embedding)
            hook = current_hook()
            if hook is None:
                raise RuntimeError("makeup hook was not armed inside the attack")
            result["outside_m_abs_delta"] = outside_m_abs_delta(clean, features, hook.mask)
        except Exception as exc:
            result["error"] = (
                f"scoring failed after the attack saved: {exc.__class__.__name__}: {exc}"
            )
        return result
    finally:
        AEGIS._run_foolbox_attack = original


def worker_main(args: argparse.Namespace) -> int:
    if args.refuse_hook or args.adaptive_epsilon or args.select_regions or not args.arm_hook:
        print(
            "B2 worker requires --arm-hook and refuses an unarmed, adaptive, "
            "or FLAME-region run.",
            file=sys.stderr,
        )
        return 2
    if not args.summary_out:
        print("B2 worker requires --summary-out.", file=sys.stderr)
        return 2
    parsing = resolve_parsing_path(args.parsing)
    try:
        assert_parsing_path(parsing)
    except MaskPathError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    image = resolve_impersonation_image(args.impersonation_image)
    plan = b1.build_plan(image, launch=True)
    aegis_root = Path(args.aegis_root).expanduser()
    config = hook_config_path(aegis_root)
    out = Path(args.summary_out).expanduser()
    try:
        assert_makeup_armed(armed_environment(config, aegis_root))
        payload = run_attack(plan, config, aegis_root)
        payload["return_code"] = 0
    except Exception as exc:
        traceback.print_exc()
        payload = {
            "ok": False,
            "return_code": 1,
            "cos_adv_source": None,
            "cos_adv_target": None,
            "outside_m_abs_delta": None,
            "aggregated_cosine_to_target": None,
            "gpu": b1.gpu_name(),
            "error": f"{exc.__class__.__name__}: {exc}",
        }
        out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        return 1
    out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    if payload.get("error"):
        print(payload["error"], file=sys.stderr)
    return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "B2: B1 targeted impersonation on NeRSemble 306 plus makeup mask M. "
            "Default does not launch and does not arm the hook."
        )
    )
    parser.add_argument(
        "--impersonation-image",
        "--victim-image",
        default=None,
        help=(
            "Still face photo of T. "
            f"Default: {DEFAULT_IMPERSONATION_IMAGE}. "
            f"Override with {IMAGE_ENV}. "
            f"Later LFW run: {LFW_IMPERSONATION_IMAGE}."
        ),
    )
    parser.add_argument(
        "--parsing",
        type=Path,
        default=None,
        help=(
            "Subject 306 parsing map. Basename must be 306.npy. "
            f"Default: {DEFAULT_PARSING}. "
            "A path under masks/ is refused. "
            f"Override with {PARSING_ENV}."
        ),
    )
    parser.add_argument(
        "--arm-hook",
        action="store_true",
        help=(
            "Arm the existing makeup PGD hook with kind=dc. "
            "Does not start the GPU attack. "
            "Requires parsing/306.npy. Does not build a fallback mask."
        ),
    )
    parser.add_argument(
        "--refuse-hook",
        action="store_true",
        help=(
            "Refuse the makeup hook and exit. "
            "B2 will not run as full-face impersonation without M."
        ),
    )
    parser.add_argument(
        "--launch",
        action="store_true",
        help=(
            "Run the GPU attack. Requires --arm-hook. "
            "Refused until the still, parsing/306.npy, the 306 avatar, "
            "the AdaFace checkpoint, and CUDA are already on disk. "
            "Does not authorize an AutoDL run by itself."
        ),
    )
    parser.add_argument(
        "--adaptive-epsilon",
        action="store_true",
        help="Refused. B2 stays on the B1/B0-a Foolbox linfpgd path.",
    )
    parser.add_argument(
        "--select-regions",
        action="store_true",
        help="Refused. FLAME regions are not makeup mask M.",
    )
    parser.add_argument(
        "--subject",
        default=b1.SUBJECT,
        help="Avatar subject. Only 306 is accepted. Default: 306.",
    )
    parser.add_argument(
        "--aegis-root",
        type=Path,
        default=b1.AEGIS_ROOT,
        help="Checkout of third_party/AEGIS. The hook is installed here when --arm-hook is set.",
    )
    parser.add_argument(
        "--summary-dir",
        default=None,
        help=(
            "Directory for summary.json and summary.csv. "
            "Default: results/b2_<T>/ (results/b2_celeba_T0/ for the first still). "
            "The flat results/ directory is refused."
        ),
    )
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--summary-out", default=None, help=argparse.SUPPRESS)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None, *, runner=subprocess.run) -> int:
    return execute(argv, runner=runner)


if __name__ == "__main__":
    sys.exit(main())

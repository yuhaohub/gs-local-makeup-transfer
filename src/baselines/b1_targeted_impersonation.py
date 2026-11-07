#!/usr/bin/env python3
# B1: full-face targeted impersonation on NeRSemble 306.
# Same attack hyperparameters as B0-a (AdaFace, eps=0.1, 300 steps,
# target-features DC, linfpgd, radius 1, seed 42). No makeup mask M.
#
# Verified on upstream AEGIS commit e3d20445942cc3d41e912897675290e090f116f7:
#   FaceRenderVerification._similarity_to_logits stacks
#     column 0 = (threshold - cosine) * 10   # "no match"
#     column 1 = (cosine - threshold) * 10   # "match"
#   (or (-cosine, +cosine) * 10 when ver_threshold is None).
#   AEGIS.run passes torch.tensor([1]) into Foolbox. A bare tensor becomes
#   Misclassification, and Linf PGD ascends cross-entropy, which lowers the
#   match logit and therefore lowers cosine with the reference embedding.
#   CLI --target-image is only that reference (e_r). It is the source
#   avatar's verification photo, not an impersonation identity.
#   There is no upstream targeted switch.
#
# B1 keeps Foolbox L∞ PGD and changes two inputs:
#   e_r is the AdaFace embedding of a still photo of T
#   the criterion is TargetedMisclassification of match class 1, so PGD
#   descends cross-entropy and raises cosine with T.
# The still photo is CelebA or LFW. It is not a second 3DGS avatar.

"""Wire B1 targeted impersonation without arming the makeup PGD hook.

The GPU attack stays off unless ``--launch`` is passed and the still photo
is already on disk. This module does not download CelebA, LFW, weights, or
AEGIS.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
import traceback
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
AEGIS_ROOT = REPO_ROOT / "third_party" / "AEGIS"
MODULE_PATH = Path(__file__).resolve()
RESULTS_DIR = REPO_ROOT / "results"
SUMMARY_NAME = "b1_targeted_impersonation_summary"

EXPERIMENT_ID = "B1"
EXPERIMENT_DESC = (
    "AEGIS interface + targeted impersonation (full face, CelebA/LFW still, no makeup mask)"
)
SUBJECT = "306"
UPSTREAM_AEGIS_COMMIT = "e3d20445942cc3d41e912897675290e090f116f7"

# James placed these stills on the AutoDL filesystem. The repo does not
# vendor CelebA or LFW. First smoke is celeba_000001. The LFW still is the
# later run. celeba_T0.jpg / lfw_T0.jpg were placeholders and are not the files.
DEFAULT_IMPERSONATION_IMAGE = "/root/autodl-fs/gs-assets/targets/celeba_000001.jpg"
LFW_IMPERSONATION_IMAGE = (
    "/root/autodl-fs/gs-assets/targets/lfw_Alejandro_Toledo_0036.jpg"
)
CELEBA_T0_ALIAS = "/root/autodl-fs/gs-assets/targets/celeba_T0.jpg"
LFW_T0_ALIAS = "/root/autodl-fs/gs-assets/targets/lfw_T0.jpg"
IMPERSONATION_IMAGE_ENV = "B1_IMPERSONATION_IMAGE"

AVATAR_DIR_RELATIVE = Path("datasets/NeRSembleReconst/avatars/306")
SOURCE_IMAGE_RELATIVE = Path("datasets/NeRSembleReconst/renders/306.png")
SOURCE_IMAGE_SUFFIX = "NeRSembleReconst/renders/306.png"
ADAFACE_CHECKPOINT = Path("models/adaface_ir50_ms1mv2.ckpt")
INSIGHTFACE_DETECTOR = Path("~/.insightface/models/buffalo_l/det_10g.onnx")

EMBEDDER = "adaface"
EPSILON = 0.1
ATTACK_STEPS = 300
TARGET_FEATURES = "DC"
ADV_ATTACK = "linfpgd"
RADIUS = 1.0
SEED = 42
CAMERA_BOUNDARY_ANGLES = (-0.5, 0.5, -0.5, 0.5, 0.0, 0.0)
ANGLE_AGGREGATION = "mean"
K_ANGLES = 5
# mask_avatar.py default. B0-a did not pass --ver-threshold, so this is the
# threshold that built the B0 logits.
VER_THRESHOLD = 0.1720
OUTPUT_PREFIX = "NeRSembleImpersonation306"

# Logit columns in FaceRenderVerification._similarity_to_logits.
NO_MATCH_CLASS = 0
MATCH_CLASS = 1

MAKEUP_HOOK_ENV = "GS_MAKEUP_HOOK_CONFIG"
MAKEUP_DISARM_RELATIVE = Path("results/.b1_makeup_hook_off")
LAUNCH_TIMEOUT_SECONDS = 1800
LOG_TAIL_LINES = 20
LOG_TAIL_CHARS = 1500
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
AGGREGATED_COSINE_RE = re.compile(
    r"Aggregated Cosine Similarity:\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+))"
)


class MakeupHookForbidden(RuntimeError):
    """B1 must not arm the makeup PGD hook."""


def similarity_to_logits(
    similarity: float,
    ver_threshold: float | None = VER_THRESHOLD,
) -> tuple[float, float]:
    """Match ``FaceRenderVerification._similarity_to_logits`` for one scalar.

    Column 0 grows when cosine falls ("no match"). Column 1 grows when
    cosine rises ("match"). ``ver_threshold`` None is the upstream branch
    that skips the decision threshold.
    """
    if ver_threshold is None:
        return (-similarity * 10.0, similarity * 10.0)
    return (
        (ver_threshold - similarity) * 10.0,
        (similarity - ver_threshold) * 10.0,
    )


def _cross_entropy(logits: tuple[float, float], class_index: int) -> float:
    peak = max(logits)
    shifted = [math.exp(value - peak) for value in logits]
    total = sum(shifted)
    probability = shifted[class_index] / total
    return -math.log(probability)


def similarity_step_sign(
    criterion: str,
    class_index: int,
    *,
    ver_threshold: float | None = VER_THRESHOLD,
    probe: float = 0.2,
) -> int:
    """Sign of the cosine change implied by one Foolbox L∞ PGD step.

    Mirrors foolbox 3 ``BaseGradientDescent.run``: the loss is cross-entropy,
    ``Misclassification`` uses step sign +1 (ascent), and
    ``TargetedMisclassification`` uses step sign -1 (descent). The sign of
    the cosine update is ``step_sign * sign(dCE/d cosine)``.
    """
    if criterion == "Misclassification":
        step_sign = 1.0
    elif criterion == "TargetedMisclassification":
        step_sign = -1.0
    else:
        raise ValueError(f"unsupported Foolbox criterion: {criterion}")
    if class_index not in (NO_MATCH_CLASS, MATCH_CLASS):
        raise ValueError(f"logit column must be 0 or 1, got {class_index}")

    delta = 1e-4
    loss_high = _cross_entropy(
        similarity_to_logits(probe + delta, ver_threshold), class_index
    )
    loss_low = _cross_entropy(
        similarity_to_logits(probe - delta, ver_threshold), class_index
    )
    d_loss = loss_high - loss_low
    if d_loss == 0.0:
        raise RuntimeError("cross-entropy is flat in cosine; cannot read a step sign")
    direction = step_sign * (1.0 if d_loss > 0.0 else -1.0)
    return 1 if direction > 0.0 else -1


def b0_criterion_spec() -> dict:
    """Upstream ``AEGIS.run`` criterion, as Foolbox receives it.

    The raw label ``1`` is the match column. Foolbox wraps a bare tensor as
    untargeted ``Misclassification``, so the step lowers cosine with ``e_r``.
    """
    return {
        "foolbox_criterion": "Misclassification",
        "class_index": MATCH_CLASS,
        "reference_embedding": "source_verification_image",
        "objective": "decrease cos(render, e_r)",
    }


def b1_criterion_spec() -> dict:
    """B1 criterion: targeted match against the impersonation embedding."""
    return {
        "foolbox_criterion": "TargetedMisclassification",
        "target_class": MATCH_CLASS,
        "reference_embedding": "impersonation_image",
        "objective": "maximize cos(render, emb(T))",
        "makeup_hook": False,
        "upstream_commit": UPSTREAM_AEGIS_COMMIT,
    }


def cosine_similarity(left, right) -> float:
    """Cosine of two embedding vectors. Accepts nested sequences."""
    first = _flat_floats(left)
    second = _flat_floats(right)
    if not first or len(first) != len(second):
        raise ValueError("embeddings must be non-empty and the same length")
    dot = sum(a * b for a, b in zip(first, second))
    norm_left = math.sqrt(sum(a * a for a in first))
    norm_right = math.sqrt(sum(b * b for b in second))
    if norm_left == 0.0 or norm_right == 0.0:
        raise ValueError("cosine is undefined for a zero embedding")
    return dot / (norm_left * norm_right)


def _flat_floats(value) -> list[float]:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "tolist"):
        value = value.tolist()
    flat: list[float] = []

    def walk(item) -> None:
        if isinstance(item, (list, tuple)):
            for part in item:
                walk(part)
            return
        flat.append(float(item))

    walk(value)
    return flat


def refuse_makeup_hook(*_args, **_kwargs) -> None:
    """Reject any attempt to install or arm the makeup PGD hook from B1."""
    raise MakeupHookForbidden(
        "B1 is full-face targeted impersonation. "
        "The makeup PGD hook stays disarmed (no semantic mask M, no kind=dc)."
    )


def makeup_disarm_path() -> Path:
    """Path that must not exist. The makeup hook no-ops when this is not a file."""
    return REPO_ROOT / MAKEUP_DISARM_RELATIVE


def worker_environment(base: dict | None = None) -> dict:
    """Environment for the AEGIS process. Points the makeup hook at a missing file."""
    env = dict(os.environ if base is None else base)
    env[MAKEUP_HOOK_ENV] = str(makeup_disarm_path())
    env["B1_MAKEUP_HOOK"] = "off"
    src = str(AEGIS_ROOT / "src")
    previous = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = src if not previous else src + os.pathsep + previous
    return env


def assert_makeup_disarmed(env: dict) -> None:
    """Raise when this process could still load a makeup hook config."""
    value = env.get(MAKEUP_HOOK_ENV, "")
    if not value or Path(value).expanduser().is_file():
        raise MakeupHookForbidden(
            f"{MAKEUP_HOOK_ENV} must point at a path that is not a file "
            "so an already patched AEGIS tree does not apply M."
        )
    if env.get("B1_MAKEUP_HOOK") != "off":
        raise MakeupHookForbidden("B1_MAKEUP_HOOK must be off.")


def apply_targeted_criterion(aegis_cls, factory):
    """Replace the criterion ``AEGIS._run_foolbox_attack`` forwards to Foolbox.

    ``AEGIS.run`` still builds ``tensor([1])``. The wrapper discards that bare
    label and passes ``factory(device)`` instead. Returns the original method
    so the caller can restore it.
    """
    original = aegis_cls._run_foolbox_attack

    def _run_foolbox_attack(self, target_class):
        criterion = factory(getattr(self, "device", None))
        return original(self, criterion)

    _run_foolbox_attack._b1_targeted = True
    _run_foolbox_attack._b1_original = original
    aegis_cls._run_foolbox_attack = _run_foolbox_attack
    return original


def make_foolbox_targeted_match(device):
    """Build Foolbox ``TargetedMisclassification`` of the match logit."""
    import foolbox as fb
    import torch

    labels = torch.tensor([MATCH_CLASS], device=device)
    return fb.criteria.TargetedMisclassification(labels)


def corpus_for(path: Path) -> str:
    name = path.name.lower()
    if name.startswith("celeba"):
        return "CelebA"
    if name.startswith("lfw"):
        return "LFW"
    return "custom"


def smoke_target_for(path: Path) -> str:
    """Filename stem for the placed samples and the old placeholder names."""
    if path.stem in {
        "celeba_000001",
        "celeba_T0",
        "lfw_Alejandro_Toledo_0036",
        "lfw_T0",
    }:
        return path.stem
    return "custom"


def output_name_for(path: Path) -> str:
    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", path.stem).strip("._") or "T"
    return f"{OUTPUT_PREFIX}_{stem}"


def resolve_impersonation_image(flag: str | None) -> str:
    """Flag, then ``B1_IMPERSONATION_IMAGE``, then ``celeba_000001.jpg``."""
    if flag:
        return flag
    env = os.environ.get(IMPERSONATION_IMAGE_ENV)
    if env:
        return env
    return DEFAULT_IMPERSONATION_IMAGE


def default_avatar_dir() -> Path:
    return AEGIS_ROOT / AVATAR_DIR_RELATIVE


def default_source_image() -> Path:
    return AEGIS_ROOT / SOURCE_IMAGE_RELATIVE


def is_source_render(candidate: Path, source: Path | None = None) -> bool:
    """True when ``candidate`` is the 306 verification render used as B0 ``e_r``."""
    source_path = default_source_image() if source is None else source
    candidate_abs = candidate.expanduser()
    try:
        if candidate_abs.resolve() == source_path.expanduser().resolve():
            return True
    except OSError:
        pass
    text = candidate_abs.as_posix()
    return text.endswith(SOURCE_IMAGE_SUFFIX)


@dataclass(frozen=True)
class AttackPlan:
    """B0 hyperparameters plus the impersonation still and the match criterion."""

    subject: str
    avatar_dir: Path
    source_image: Path
    impersonation_image: Path
    lfw_image: str
    embedder: str
    epsilon: float
    attack_steps: int
    target_features: str
    adv_attack: str
    radius: float
    seed: int
    camera_boundary_angles: tuple[float, ...]
    angle_aggregation: str
    k_angles: int
    ver_threshold: float
    output_name: str
    adaptive_epsilon: bool
    selected_regions: tuple[str, ...]
    arm_makeup_hook: bool
    launch: bool

    def aegis_kwargs(self) -> dict:
        """Keyword arguments for ``AEGIS(...)``.

        Upstream names the reference photo ``target_image``. For B1 that
        photo is the impersonation still, so ``e_r`` is ``emb(T)``.
        """
        return {
            "embedder_name": self.embedder,
            "avatar_dir": str(self.avatar_dir),
            "target_image": str(self.impersonation_image),
            "epsilons": [self.epsilon],
            "selected_regions": list(self.selected_regions),
            "camera_boundary_angles": list(self.camera_boundary_angles),
            "angle_aggregation": self.angle_aggregation,
            "k_angles": self.k_angles,
            "targeted_features": self.target_features,
            "adv_attack": self.adv_attack,
            "attack_steps": self.attack_steps,
            "ver_threshold": self.ver_threshold,
            "seed": self.seed,
            "output_name": self.output_name,
            "adaptive_epsilon": self.adaptive_epsilon,
            "radius": self.radius,
            "surrogate_keys": None,
        }

    def user_command(self) -> str:
        image = self.impersonation_image
        return (
            "python src/baselines/b1_targeted_impersonation.py "
            f"--impersonation-image {image} --launch"
        )

    def criterion(self) -> dict:
        return b1_criterion_spec()


def build_plan(impersonation_image: str, *, launch: bool) -> AttackPlan:
    image = Path(impersonation_image).expanduser()
    return AttackPlan(
        subject=SUBJECT,
        avatar_dir=default_avatar_dir(),
        source_image=default_source_image(),
        impersonation_image=image,
        lfw_image=LFW_IMPERSONATION_IMAGE,
        embedder=EMBEDDER,
        epsilon=EPSILON,
        attack_steps=ATTACK_STEPS,
        target_features=TARGET_FEATURES,
        adv_attack=ADV_ATTACK,
        radius=RADIUS,
        seed=SEED,
        camera_boundary_angles=CAMERA_BOUNDARY_ANGLES,
        angle_aggregation=ANGLE_AGGREGATION,
        k_angles=K_ANGLES,
        ver_threshold=VER_THRESHOLD,
        output_name=output_name_for(image),
        adaptive_epsilon=False,
        selected_regions=(),
        arm_makeup_hook=False,
        launch=launch,
    )


def reference_image_for_run(plan: AttackPlan) -> Path:
    """Return the still photo that becomes ``e_r``.

    Refuses the NeRSemble 306 render. Using that photo would set ``e_r`` to
    the source identity.
    """
    image = plan.impersonation_image
    if is_source_render(image, plan.source_image):
        raise ValueError(
            "The impersonation still is the NeRSemble 306 render "
            f"({SOURCE_IMAGE_SUFFIX}). e_r would be the source identity. "
            "Pass a CelebA or LFW face photo via --impersonation-image."
        )
    return image


def expected_outputs(plan: AttackPlan) -> tuple[Path, Path]:
    base = (
        AEGIS_ROOT
        / "datasets"
        / f"seed{plan.seed}"
        / f"{plan.output_name}_{plan.embedder}_all"
        / f"eps_{plan.epsilon:.3f}"
    )
    avatar = base / "avatars" / plan.subject / "point_cloud.ply"
    render = base / "renders" / f"{plan.subject}.png"
    return avatar, render


def image_ready(path: Path) -> bool:
    try:
        return path.is_file() and path.stat().st_size > 0
    except OSError:
        return False


def aegis_bootstrapped() -> bool:
    return (AEGIS_ROOT / ".git").is_dir()


def cuda_available() -> bool:
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


def gpu_name() -> str | None:
    nvidia = shutil.which("nvidia-smi")
    if nvidia is None:
        return None
    try:
        proc = subprocess.run(
            [nvidia, "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if proc.returncode != 0:
        return None
    lines = [line.strip() for line in proc.stdout.splitlines() if line.strip()]
    return lines[0] if lines else None


def collect_blockers(plan: AttackPlan) -> list[str]:
    """Reasons ``--launch`` must not start. Empty when the attack may run."""
    image = plan.impersonation_image
    if not image_ready(image):
        return [
            "awaiting_target: impersonation still is not on disk "
            f"({image}). First smoke is {DEFAULT_IMPERSONATION_IMAGE}. "
            f"LFW later run is {LFW_IMPERSONATION_IMAGE}. "
            "This module does not download CelebA or LFW."
        ]
    if image.suffix.lower() not in IMAGE_SUFFIXES:
        return [
            "invalid_image: impersonation still must be a face photo "
            f"({', '.join(sorted(IMAGE_SUFFIXES))}), got {image.suffix or image.name}."
        ]
    try:
        reference_image_for_run(plan)
    except ValueError as exc:
        return [f"refused_source_image: {exc}"]

    blockers: list[str] = []
    if not aegis_bootstrapped():
        blockers.append(
            "bootstrap_required: third_party/AEGIS is missing. "
            "Run scripts/00_bootstrap_aegis.sh. The upstream tree stays gitignored."
        )
        return blockers
    if shutil.which("uv") is None:
        blockers.append("deps_missing: uv is not on PATH")

    ply = plan.avatar_dir / "point_cloud.ply"
    if not ply.is_file():
        blockers.append(f"deps_missing: 306 point_cloud.ply missing ({_rel(ply)})")
    if not plan.source_image.is_file():
        blockers.append(
            f"deps_missing: 306 source render missing ({_rel(plan.source_image)})"
        )
    checkpoint = AEGIS_ROOT / ADAFACE_CHECKPOINT
    if not checkpoint.is_file():
        blockers.append(
            "deps_missing: AdaFace checkpoint missing "
            f"({_rel(checkpoint)}); not downloaded"
        )
    detector = INSIGHTFACE_DETECTOR.expanduser()
    if not detector.is_file():
        blockers.append(
            "deps_missing: insightface detector missing "
            f"({INSIGHTFACE_DETECTOR}); not downloaded"
        )
    gaussian_avatars = AEGIS_ROOT / "GaussianAvatars"
    if not gaussian_avatars.is_dir():
        blockers.append(
            f"deps_missing: GaussianAvatars checkout missing ({_rel(gaussian_avatars)})"
        )
    if blockers:
        return blockers
    if not cuda_available():
        return ["gpu_required: CUDA device not available"]
    return []


def _primary_status(blockers: list[str]) -> str:
    if not blockers:
        return "ok"
    codes = [item.split(":", 1)[0] for item in blockers]
    for code in (
        "awaiting_target",
        "invalid_image",
        "refused_source_image",
        "bootstrap_required",
        "deps_missing",
        "gpu_required",
    ):
        if code in codes:
            return code
    return codes[0]


def worker_command(plan: AttackPlan, summary_out: Path) -> list[str]:
    return [
        "uv",
        "run",
        "python",
        str(MODULE_PATH),
        "--worker",
        "--impersonation-image",
        str(plan.impersonation_image),
        "--summary-out",
        str(summary_out),
    ]


def start_worker(plan: AttackPlan, summary_out: Path, runner, env: dict | None = None):
    """Spawn the uv worker. Refuses to start when the makeup hook could arm."""
    disarm = worker_environment() if env is None else env
    assert_makeup_disarmed(disarm)
    command = worker_command(plan, summary_out)
    if "--arm-hook" in command or "adaptive-epsilon" in " ".join(command):
        raise MakeupHookForbidden("B1 worker command must not arm makeup or adaptive epsilon.")
    return runner(
        command,
        cwd=AEGIS_ROOT,
        env=disarm,
        capture_output=True,
        text=True,
        timeout=LAUNCH_TIMEOUT_SECONDS,
        check=False,
    )


def _rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(REPO_ROOT.resolve()).as_posix()
    except ValueError:
        return str(path)


def _log_tail(text: str) -> str:
    lines = [line for line in text.splitlines() if line.strip()]
    tail = "\n".join(lines[-LOG_TAIL_LINES:])
    if len(tail) > LOG_TAIL_CHARS:
        tail = tail[-LOG_TAIL_CHARS:]
    return tail


def _parse_aggregated_cosine(text: str) -> float | None:
    matches = AGGREGATED_COSINE_RE.findall(text)
    if not matches:
        return None
    return float(matches[-1])


def _first_epsilon(adv_features):
    if isinstance(adv_features, (list, tuple)):
        return adv_features[0]
    return adv_features[0]


def run_attack(plan: AttackPlan) -> dict:
    """Import AEGIS, flip the Foolbox criterion, and score both cosines.

    Called from ``--worker`` inside ``uv run``. The makeup hook env is
    checked again here so a patched tree cannot apply ``M``.
    """
    env = worker_environment()
    assert_makeup_disarmed(env)
    os.environ[MAKEUP_HOOK_ENV] = env[MAKEUP_HOOK_ENV]
    os.environ["B1_MAKEUP_HOOK"] = "off"

    src = AEGIS_ROOT / "src"
    if src.is_dir() and str(src) not in sys.path:
        sys.path.insert(0, str(src))

    import torch
    from aegis import AEGIS

    reference_image_for_run(plan)
    original = apply_targeted_criterion(AEGIS, make_foolbox_targeted_match)
    try:
        kwargs = plan.aegis_kwargs()
        model = AEGIS(**kwargs)
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
        avatar_out, render_out = expected_outputs(plan)
        result = {
            "ok": True,
            "cos_adv_source": None,
            "cos_adv_target": None,
            "aggregated_cosine_to_target": _parse_aggregated_cosine(printed),
            "gpu": gpu_name(),
            "output_avatar": _rel(avatar_out) if avatar_out.is_file() else None,
            "output_render": _rel(render_out) if render_out.is_file() else None,
            "output_avatar_bytes": avatar_out.stat().st_size if avatar_out.is_file() else None,
            "output_render_bytes": render_out.stat().st_size if render_out.is_file() else None,
            "error": None,
        }
        try:
            features = _first_epsilon(captured["adv"])
            with torch.no_grad():
                rgb = model.render_frame_in_rgb(features)
                adv_embedding = model.ver_model(rgb)
            source_embedding = model._get_reference_id_embedding(str(plan.source_image))
            target_embedding = model.ref_emb
            result["cos_adv_source"] = cosine_similarity(adv_embedding, source_embedding)
            result["cos_adv_target"] = cosine_similarity(adv_embedding, target_embedding)
        except Exception as exc:
            result["error"] = (
                f"scoring failed after the attack saved: {exc.__class__.__name__}: {exc}"
            )
        return result
    finally:
        AEGIS._run_foolbox_attack = original


def _empty_metrics(plan: AttackPlan, *, gpu: str | None) -> dict:
    avatar_out, render_out = expected_outputs(plan)
    return {
        "command": plan.user_command(),
        "command_attempted": False,
        "return_code": None,
        "log_tail": "",
        "cuda_available": cuda_available(),
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
        "adaptive_epsilon": plan.adaptive_epsilon,
        "selected_regions": list(plan.selected_regions),
        "output_name": plan.output_name,
        "avatar_dir": _rel(plan.avatar_dir),
        "source_image": _rel(plan.source_image),
        "impersonation_image": str(plan.impersonation_image),
        "impersonation_image_present": image_ready(plan.impersonation_image),
        "lfw_image": plan.lfw_image,
        "corpus": corpus_for(plan.impersonation_image),
        "smoke_target": smoke_target_for(plan.impersonation_image),
        "criterion": plan.criterion(),
        "reference_embedding": "impersonation_image",
        "cos_adv_source": None,
        "cos_adv_target": None,
        "aggregated_cosine_to_target": None,
        "makeup_hook_armed": False,
        "expected_output_avatar": _rel(avatar_out),
        "expected_output_render": _rel(render_out),
        "blockers": [],
    }


def _now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def write_summary(summary_dir: Path, payload: dict) -> tuple[Path, Path]:
    summary_dir.mkdir(parents=True, exist_ok=True)
    gitkeep = summary_dir / ".gitkeep"
    if summary_dir.resolve() == RESULTS_DIR.resolve() and not gitkeep.exists():
        gitkeep.write_text("")
    json_path = summary_dir / f"{SUMMARY_NAME}.json"
    csv_path = summary_dir / f"{SUMMARY_NAME}.csv"
    json_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    metrics = payload.get("metrics") or {}
    return_code = metrics.get("return_code")
    cos_source = metrics.get("cos_adv_source")
    cos_target = metrics.get("cos_adv_target")
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
                "gpu",
                "makeup_hook_armed",
                "impersonation_image",
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
                "gpu": metrics.get("gpu") or "",
                "makeup_hook_armed": str(bool(metrics.get("makeup_hook_armed"))).lower(),
                "impersonation_image": metrics.get("impersonation_image") or "",
                "command": metrics.get("command") or "",
                "note": payload.get("note") or "",
            }
        )
    return json_path, csv_path


def _payload(
    plan: AttackPlan,
    *,
    status: str,
    note: str,
    metrics: dict,
    pipeline_ok: bool = False,
) -> dict:
    return {
        "experiment_id": EXPERIMENT_ID,
        "desc": EXPERIMENT_DESC,
        "status": status,
        "pipeline_ok": pipeline_ok,
        "subject": plan.subject,
        "corpus": corpus_for(plan.impersonation_image),
        "smoke_target": smoke_target_for(plan.impersonation_image),
        "bootstrapped": aegis_bootstrapped(),
        "recorded_at": _now(),
        "metrics": metrics,
        "note": note,
    }


def _awaiting_note(plan: AttackPlan, *, launch_requested: bool) -> str:
    present = image_ready(plan.impersonation_image)
    if present and not launch_requested:
        return (
            "B1 criterion is wired (e_r = AdaFace(T), TargetedMisclassification "
            "of match class 1) and the still photo is on disk. GPU was not "
            "started because --launch was omitted. Makeup hook is disarmed."
        )
    if launch_requested and not present:
        return (
            "B1 --launch was requested but the impersonation still is not on "
            f"disk ({plan.impersonation_image}). First smoke is "
            f"{DEFAULT_IMPERSONATION_IMAGE}. Later run is {LFW_IMPERSONATION_IMAGE}. "
            "No dataset was downloaded. Makeup hook is disarmed. GPU was not started."
        )
    return (
        "B1 full-face targeted impersonation is wired and was not launched. "
        f"First smoke uses {DEFAULT_IMPERSONATION_IMAGE} "
        "(CelebA still → AdaFace). "
        f"Later run: {LFW_IMPERSONATION_IMAGE}. "
        "The repo does not vendor CelebA or LFW. "
        f"Placeholder names {Path(CELEBA_T0_ALIAS).name} and "
        f"{Path(LFW_T0_ALIAS).name} are not the files on disk. "
        "Pass --launch only after celeba_000001.jpg is on disk. "
        "Makeup hook is disarmed."
    )


def execute(argv: list[str] | None = None, *, runner=subprocess.run) -> int:
    """Parse CLI, write the summary, and launch only when asked and ready."""
    args = parse_args(argv)
    if args.worker:
        return worker_main(args)

    image = resolve_impersonation_image(args.impersonation_image)
    plan = build_plan(image, launch=bool(args.launch) and not args.arm_hook)
    summary_dir = Path(args.summary_dir).expanduser()
    if not summary_dir.is_absolute():
        summary_dir = REPO_ROOT / summary_dir
    gpu = gpu_name()
    metrics = _empty_metrics(plan, gpu=gpu)

    if args.arm_hook:
        metrics["blockers"] = [
            "refused_makeup_hook: B1 does not arm the makeup PGD hook."
        ]
        note = (
            "Refused --arm-hook. B1 is full-face targeted impersonation "
            "with no semantic mask M and no kind=dc makeup hook."
        )
        write_summary(
            summary_dir,
            _payload(plan, status="refused_makeup_hook", note=note, metrics=metrics),
        )
        print(note, file=sys.stderr)
        return 2

    if not args.launch:
        status = "not_launched" if image_ready(plan.impersonation_image) else "awaiting_target"
        note = _awaiting_note(plan, launch_requested=False)
        if status == "awaiting_target":
            metrics["blockers"] = [
                "awaiting_target: " + note,
            ]
        write_summary(
            summary_dir,
            _payload(plan, status=status, note=note, metrics=metrics),
        )
        _print_summary_paths(summary_dir, status)
        return 0

    try:
        reference_image_for_run(plan)
    except ValueError as exc:
        metrics["blockers"] = [f"refused_source_image: {exc}"]
        note = str(exc)
        write_summary(
            summary_dir,
            _payload(plan, status="refused_source_image", note=note, metrics=metrics),
        )
        print(note, file=sys.stderr)
        return 2

    blockers = collect_blockers(plan)
    if blockers:
        status = _primary_status(blockers)
        metrics["blockers"] = blockers
        note = (
            "B1 did not launch. "
            + "; ".join(item.split(":", 1)[1].strip() for item in blockers)
            + " Makeup hook is disarmed."
        )
        write_summary(
            summary_dir,
            _payload(plan, status=status, note=note, metrics=metrics),
        )
        print(note, file=sys.stderr)
        _print_summary_paths(summary_dir, status)
        return 2

    disarm = worker_environment()
    fd, worker_name = tempfile.mkstemp(prefix="b1-worker-", suffix=".json")
    os.close(fd)
    worker_path = Path(worker_name)
    command = worker_command(plan, worker_path)
    metrics["command_attempted"] = True
    metrics["worker_command"] = " ".join(command)
    proc = None
    try:
        proc = start_worker(plan, worker_path, runner, disarm)
    except MakeupHookForbidden as exc:
        metrics["command_attempted"] = False
        metrics["blockers"] = [f"refused_makeup_hook: {exc}"]
        write_summary(
            summary_dir,
            _payload(plan, status="refused_makeup_hook", note=str(exc), metrics=metrics),
        )
        print(str(exc), file=sys.stderr)
        return 2
    except subprocess.TimeoutExpired as exc:
        log = ""
        if isinstance(exc.stdout, str):
            log += exc.stdout
        if isinstance(exc.stderr, str):
            log += exc.stderr
        metrics["return_code"] = None
        metrics["log_tail"] = _log_tail(log)
        note = f"B1 worker exceeded {LAUNCH_TIMEOUT_SECONDS}s and was stopped."
        write_summary(
            summary_dir,
            _payload(plan, status="not_run", note=note, metrics=metrics),
        )
        return 1
    except OSError as exc:
        metrics["return_code"] = None
        metrics["log_tail"] = str(exc)
        note = f"B1 worker could not be started: {exc}"
        write_summary(
            summary_dir,
            _payload(plan, status="deps_missing", note=note, metrics=metrics),
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
    scored = (
        return_code == 0
        and metrics["cos_adv_source"] is not None
        and metrics["cos_adv_target"] is not None
    )
    if scored:
        status = "pipeline_ok"
        pipeline_ok = True
        note = (
            "B1 targeted impersonation finished. "
            f"cos(adv, source)={metrics['cos_adv_source']:.4f}, "
            f"cos(adv, T)={metrics['cos_adv_target']:.4f}. "
            "Makeup hook stayed disarmed."
        )
        exit_code = 0
    elif return_code == 0:
        status = "scoring_failed"
        pipeline_ok = False
        note = (
            "B1 attack process exited 0 but cos(adv, source) and cos(adv, T) "
            "were not both recorded. "
            + str(worker_payload.get("error") or "See metrics.log_tail.")
        )
        exit_code = 1
    else:
        status = "not_run"
        pipeline_ok = False
        note = (
            f"B1 worker exited {return_code}. "
            + str(worker_payload.get("error") or "See metrics.log_tail.")
        )
        exit_code = return_code if isinstance(return_code, int) else 1

    write_summary(
        summary_dir,
        _payload(
            plan,
            status=status,
            note=note,
            metrics=metrics,
            pipeline_ok=pipeline_ok,
        ),
    )
    _print_summary_paths(summary_dir, status)
    return exit_code


def _read_worker_payload(path: Path) -> dict:
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def _print_summary_paths(summary_dir: Path, status: str) -> None:
    print(f"status={status}")
    print(f"wrote {summary_dir / (SUMMARY_NAME + '.json')}")
    print(f"wrote {summary_dir / (SUMMARY_NAME + '.csv')}")


def worker_main(args: argparse.Namespace) -> int:
    if args.arm_hook:
        print(
            "B1 worker refuses to arm the makeup PGD hook.",
            file=sys.stderr,
        )
        return 2
    if not args.summary_out:
        print("B1 worker requires --summary-out.", file=sys.stderr)
        return 2
    image = resolve_impersonation_image(args.impersonation_image)
    plan = build_plan(image, launch=True)
    out = Path(args.summary_out).expanduser()
    try:
        assert_makeup_disarmed(worker_environment())
        payload = run_attack(plan)
        payload["return_code"] = 0
    except Exception as exc:
        traceback.print_exc()
        payload = {
            "ok": False,
            "return_code": 1,
            "cos_adv_source": None,
            "cos_adv_target": None,
            "aggregated_cosine_to_target": None,
            "gpu": gpu_name(),
            "error": f"{exc.__class__.__name__}: {exc}",
        }
        out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        return 1
    out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    cos_source = payload.get("cos_adv_source")
    cos_target = payload.get("cos_adv_target")
    if cos_source is not None and cos_target is not None:
        print(f"cos_adv_source={cos_source:.6f} cos_adv_target={cos_target:.6f}")
    elif payload.get("error"):
        print(payload["error"], file=sys.stderr)
    return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "B1 full-face targeted impersonation on NeRSemble 306. "
            "T is one CelebA or LFW still photo. The makeup hook stays off."
        )
    )
    parser.add_argument(
        "--impersonation-image",
        "--victim-image",
        default=None,
        help=(
            "Still face photo of T. AdaFace embeds this file. "
            f"Default: {DEFAULT_IMPERSONATION_IMAGE}. "
            f"Override with {IMPERSONATION_IMAGE_ENV}. "
            f"Later LFW run: {LFW_IMPERSONATION_IMAGE}."
        ),
    )
    parser.add_argument(
        "--launch",
        action="store_true",
        help=(
            "Run the GPU attack. Refused until the still photo, the 306 "
            "avatar, the AdaFace checkpoint, and CUDA are already on disk."
        ),
    )
    parser.add_argument(
        "--arm-hook",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--summary-dir",
        default=str(RESULTS_DIR.relative_to(REPO_ROOT)),
        help="Directory for the JSON and CSV summary. Default: results.",
    )
    parser.add_argument(
        "--worker",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--summary-out",
        default=None,
        help=argparse.SUPPRESS,
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None, *, runner=subprocess.run) -> int:
    return execute(argv, runner=runner)


if __name__ == "__main__":
    sys.exit(main())

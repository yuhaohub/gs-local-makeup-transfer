"""In-loop makeup support for the AEGIS L∞ PGD steps on NeRSemble 306.

``project_residual`` is applied inside the attack step, not after the
attack returns. Two call sites were checked on upstream AEGIS ``main``
commit ``e3d20445942cc3d41e912897675290e090f116f7``:

- ``src/aegis/core/helpers.py`` ``adaptive_linf_pgd_attack`` updates
  ``delta_new = delta - step_size * grad_sign`` and then clamps each
  Gaussian to its own ±ε. The patch calls ``project_residual`` on
  ``delta_new`` immediately after that clamp.
- ``src/aegis/core/model.py`` ``AEGIS._run_foolbox_attack`` has no step
  loop of its own. B0-a (``--adv-attack linfpgd``, no
  ``--adaptive-epsilon``) uses Foolbox
  ``LinfProjectedGradientDescentAttack``. That class projects with
  ``x0 + clip(x - x0, ±ε)`` inside ``LinfBaseGradientDescent.project``.
  The patch binds a replacement ``project`` before ``attack(...)`` so
  each Foolbox step runs ``project_residual`` on the same residual.

The AEGIS tree stays gitignored. ``install_makeup_pgd_hook`` writes the
patch and a small config into an existing checkout. It does not clone
AEGIS and does not download weights or FaceScape. With no checkout, the
numpy steps below still run and ``locate_aegis_call_sites`` reports
``tree_missing``.

``aegis_pgd_hook_stub`` remains the call-site note from PR #10. This
module is the hook that note asked for.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Optional, Union

from makeup._numpy import require_numpy
from makeup.aegis_mask import assert_subject_306, support_violation
from makeup.mask import ArrayLike, as_mask_array
from makeup.residual import MaskLike, project_residual

# Blob the anchors below were read from. Not a submodule pin.
UPSTREAM_AEGIS_COMMIT = "e3d20445942cc3d41e912897675290e090f116f7"

HELPERS_RELATIVE = Path("src/aegis/core/helpers.py")
MODEL_RELATIVE = Path("src/aegis/core/model.py")
CONFIG_NAME = ".gs_makeup_hook.json"
MASK_NAME = ".gs_makeup_mask.npy"
HOOK_CONFIG_ENV = "GS_MAKEUP_HOOK_CONFIG"
TAIL_BEGIN = "# BEGIN gs-local-makeup-transfer makeup support hook"
TAIL_END = "# END gs-local-makeup-transfer makeup support hook"

# Exact lines from adaptive_linf_pgd_attack, inside ``with torch.no_grad()``.
ADAPTIVE_CLAMP_ANCHOR = (
    "            delta_new = torch.where(delta_new < -eps, -eps, delta_new)\n"
)
ADAPTIVE_CLAMP_PATCHED = (
    "            delta_new = torch.where(delta_new < -eps, -eps, delta_new)\n"
    "            delta_new = _gs_makeup_project_delta(delta_new)\n"
)
ADAPTIVE_SIGN_ANCHOR = "delta_new = delta - step_size * grad_sign"

# Exact call in AEGIS._run_foolbox_attack.
FOOLBOX_ATTACK_ANCHOR = (
    "        attack = get_foolbox_attack(\n"
    "            adv_attack_name=self.adv_attack,\n"
    "            steps=self.attack_steps,\n"
    "            random_start=False,\n"
    "        )\n"
)
FOOLBOX_ATTACK_PATCHED = (
    "        attack = get_foolbox_attack(\n"
    "            adv_attack_name=self.adv_attack,\n"
    "            steps=self.attack_steps,\n"
    "            random_start=False,\n"
    "        )\n"
    "        attack = _gs_makeup_bind_foolbox_attack(attack)\n"
)

HELPERS_TAIL = f'''
{TAIL_BEGIN}
def _gs_makeup_config_path():
    """Hook config written by makeup.aegis_pgd_hook. Absent means no-op."""
    import os
    from pathlib import Path

    env = os.environ.get("{HOOK_CONFIG_ENV}")
    if env:
        path = Path(env)
        return path if path.is_file() else None
    candidate = Path.cwd() / "{CONFIG_NAME}"
    return candidate if candidate.is_file() else None


def _gs_makeup_project_delta(delta):
    """project_residual after the adaptive per-element ε clamp.

    No-op when no hook config is present. A present config that cannot
    be imported raises, so a masked run does not silently drop M.
    """
    cfg = _gs_makeup_config_path()
    if cfg is None:
        return delta
    import json
    import sys
    from pathlib import Path

    meta = json.loads(Path(cfg).read_text(encoding="utf-8"))
    root = meta.get("repo_root")
    if root and root not in sys.path:
        sys.path.insert(0, root)
    from makeup.aegis_pgd_hook import project_configured_delta

    return project_configured_delta(delta)
{TAIL_END}
'''

MODEL_TAIL = f'''
{TAIL_BEGIN}
def _gs_makeup_bind_foolbox_attack(attack):
    """Bind project_residual into Foolbox L∞ PGD.project for this attack.

    Called from AEGIS._run_foolbox_attack before attack(...). Only
    LinfProjectedGradientDescentAttack is wrapped. Other attack names
    are returned unchanged. No-op when no hook config is present.
    """
    import os
    from pathlib import Path

    env = os.environ.get("{HOOK_CONFIG_ENV}")
    if env:
        cfg = Path(env)
        cfg = cfg if cfg.is_file() else None
    else:
        candidate = Path.cwd() / "{CONFIG_NAME}"
        cfg = candidate if candidate.is_file() else None
    if cfg is None:
        return attack
    import json
    import sys

    meta = json.loads(cfg.read_text(encoding="utf-8"))
    root = meta.get("repo_root")
    if root and root not in sys.path:
        sys.path.insert(0, root)
    from makeup.aegis_pgd_hook import bind_foolbox_attack

    return bind_foolbox_attack(attack)
{TAIL_END}
'''

_ARMED: Optional["MakeupPgdHook"] = None
_CONFIG_CACHE_KEY: Optional[tuple] = None
_CONFIG_CACHE_HOOK: Optional["MakeupPgdHook"] = None


def clear_hook_state() -> None:
    """Drop the in-process hook and the config cache."""
    global _ARMED, _CONFIG_CACHE_KEY, _CONFIG_CACHE_HOOK
    _ARMED = None
    _CONFIG_CACHE_KEY = None
    _CONFIG_CACHE_HOOK = None


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


class MakeupPgdHook:
    """Makeup support applied inside one AEGIS L∞ step.

    ``epsilon`` is the makeup bound passed to ``project_residual``. The
    AEGIS clamp still runs first. A tighter per-element adaptive ε is
    kept. A larger one is pulled back to this bound. Sites with ``M == 0``
    become 0. Subject 306 only.
    """

    def __init__(
        self,
        mask: MaskLike,
        epsilon: float,
        *,
        kind: str = "dc",
        subject: Union[str, int] = "306",
    ) -> None:
        if epsilon < 0:
            raise ValueError("epsilon must be >= 0")
        if kind not in {"dc", "rgb"}:
            raise ValueError(f"kind must be 'dc' or 'rgb', got {kind!r}")
        self.subject = assert_subject_306(subject)
        self.epsilon = float(epsilon)
        self.kind = kind
        self.mask = mask

    def project_like(self, delta: ArrayLike) -> ArrayLike:
        """``project_residual`` on ``delta``, preserving array type and shape.

        Accepted layouts, all ending in 3 channels:

        - ``(N, 3)`` — adaptive DC residual after the ``(N, 1, 3)`` squeeze
        - ``(N, 1, 3)`` — raw Gaussian DC features
        - ``(1, N, 1, 3)`` — Foolbox input ``att_tensor.unsqueeze(0)``
        - ``(H, W, 3)`` and ``(1, H, W, 3)`` — RGB

        Raises:
            ValueError: the shape is not one of those layouts, or ``M``
                does not match the spatial grid.
            ImportError: NumPy is not installed.
        """
        backend = _array_backend(delta)
        if backend == "torch":
            restored = self._project_numpy(delta.detach().cpu().numpy())
            import torch

            return torch.as_tensor(restored, dtype=delta.dtype, device=delta.device)
        if backend == "eagerpy":
            raw = delta.raw
            restored = self._project_numpy(_numpy_copy(raw))
            return _rewrap_eagerpy(raw, restored)
        return self._project_numpy(_numpy_copy(delta))

    def _project_numpy(self, array):
        np = require_numpy()
        values = np.asarray(array)
        flat, restore = _reshape_residual(values, self.kind)
        projected = np.asarray(
            project_residual(flat, self.mask, self.epsilon, kind=self.kind),
            dtype=float,
        )
        restored = restore(projected)
        if np.issubdtype(values.dtype, np.floating) and restored.dtype != values.dtype:
            restored = restored.astype(values.dtype, copy=False)
        return restored

    def adaptive_step(
        self,
        delta: ArrayLike,
        grad: ArrayLike,
        epsilon_per_element: ArrayLike,
        step_size: float,
    ) -> ArrayLike:
        """One ``adaptive_linf_pgd_attack`` step, then ``project_residual``.

        Mirrors the upstream body: ``sign`` of the gradient, subtract
        ``step_size * sign``, clamp to the per-element ±ε ball, then the
        makeup projection.
        """
        clamped = epsilon_only_adaptive_step(
            delta, grad, epsilon_per_element, step_size
        )
        return self.project_like(clamped)

    def foolbox_project(self, x: ArrayLike, x0: ArrayLike) -> ArrayLike:
        """One Foolbox L∞ ``project``, then ``project_residual`` on ``x - x0``.

        Upstream ``LinfBaseGradientDescent.project`` is
        ``x0 + clip(x - x0, -epsilon, epsilon)``.
        """
        np = require_numpy()
        values = np.asarray(x, dtype=float)
        origin = np.asarray(x0, dtype=float)
        clipped = origin + np.clip(values - origin, -self.epsilon, self.epsilon)
        return self.project_like(clipped - origin) + origin

    def reproject_foolbox(self, x_projected: ArrayLike, x0: ArrayLike) -> ArrayLike:
        """Makeup projection of a residual Foolbox has already ε-clipped.

        ``x_projected`` is the return value of ``LinfBaseGradientDescent.project``.
        Torch and EagerPy tensors are accepted when those libraries are
        installed. The CPU path uses NumPy.
        """
        backend = _array_backend(x_projected)
        if backend == "eagerpy":
            raw = x_projected.raw
            raw0 = x0.raw
            delta = raw - raw0
            new_delta = self.project_like(delta)
            return _rewrap_eagerpy(raw, _add_raw(raw0, new_delta))
        if backend == "torch":
            new_delta = self.project_like(x_projected - x0)
            return x0 + new_delta
        np = require_numpy()
        projected = np.asarray(x_projected, dtype=float)
        origin = np.asarray(x0, dtype=float)
        return origin + self.project_like(projected - origin)


def epsilon_only_adaptive_step(
    delta: ArrayLike,
    grad: ArrayLike,
    epsilon_per_element: ArrayLike,
    step_size: float,
) -> ArrayLike:
    """The adaptive step with the ε clamp and without makeup ``M``.

    This is the leak path. ``adaptive_linf_pgd_attack`` stops here.
    """
    np = require_numpy()
    if step_size < 0:
        raise ValueError("step_size must be >= 0")
    values = np.asarray(delta, dtype=float)
    signed = np.sign(np.asarray(grad, dtype=float))
    if signed.shape != values.shape:
        raise ValueError(
            f"shape mismatch: delta {values.shape}, grad {signed.shape}"
        )
    updated = values - float(step_size) * signed
    eps = _broadcast_epsilon(epsilon_per_element, updated.shape)
    return np.minimum(np.maximum(updated, -eps), eps)


def epsilon_only_foolbox_project(
    x: ArrayLike,
    x0: ArrayLike,
    epsilon: float,
) -> ArrayLike:
    """``LinfBaseGradientDescent.project`` with no makeup mask."""
    np = require_numpy()
    values = np.asarray(x, dtype=float)
    origin = np.asarray(x0, dtype=float)
    if values.shape != origin.shape:
        raise ValueError(f"shape mismatch: x {values.shape}, x0 {origin.shape}")
    return origin + np.clip(values - origin, -float(epsilon), float(epsilon))


def _numpy_copy(value):
    np = require_numpy()
    module = type(value).__module__
    if module.startswith("torch"):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _array_backend(value) -> str:
    module = type(value).__module__
    if module.startswith("eagerpy"):
        return "eagerpy"
    if module.startswith("torch"):
        return "torch"
    return "numpy"


def _add_raw(origin, delta):
    return origin + delta


def _rewrap_eagerpy(raw_template, value):
    import eagerpy as ep

    module = type(raw_template).__module__
    if module.startswith("torch"):
        import torch

        tensor = torch.as_tensor(
            _numpy_copy(value),
            dtype=raw_template.dtype,
            device=raw_template.device,
        )
        return ep.astensor(tensor)
    return ep.astensor(_numpy_copy(value))


def _reshape_residual(array, kind: str):
    """Return ``(N, 3)`` or ``(H, W, 3)`` and a function that restores ``array``."""
    np = require_numpy()
    values = np.asarray(array)
    if kind == "dc":
        if values.ndim == 2 and values.shape[-1] == 3:
            return values, lambda out: out
        if values.ndim == 3 and values.shape[1] == 1 and values.shape[2] == 3:
            return values[:, 0, :], lambda out, shape=values.shape: out.reshape(shape)
        if (
            values.ndim == 4
            and values.shape[0] == 1
            and values.shape[2] == 1
            and values.shape[3] == 3
        ):
            return values[0, :, 0, :], lambda out, shape=values.shape: out.reshape(shape)
    elif kind == "rgb":
        if values.ndim == 3 and values.shape[-1] == 3:
            return values, lambda out: out
        if values.ndim == 4 and values.shape[0] == 1 and values.shape[-1] == 3:
            return values[0], lambda out, shape=values.shape: out.reshape(shape)
    else:
        raise ValueError(f"kind must be 'dc' or 'rgb', got {kind!r}")
    raise ValueError(
        f"cannot align a {kind} residual of shape {tuple(values.shape)} "
        "with project_residual"
    )


def _broadcast_epsilon(epsilon_per_element, shape: tuple):
    np = require_numpy()
    eps = np.asarray(epsilon_per_element, dtype=float)
    if eps.ndim == 0 or eps.shape == shape:
        return eps
    if shape and shape[-1] == 3 and eps.shape == shape[:-1]:
        return eps.reshape(eps.shape + (1,))
    try:
        broadcast = np.broadcast_shapes(eps.shape, shape)
    except ValueError as exc:
        raise ValueError(
            f"epsilon shape {eps.shape} does not broadcast to {shape}"
        ) from exc
    if broadcast != shape:
        raise ValueError(
            f"epsilon shape {eps.shape} does not broadcast to {shape}"
        )
    return eps


def _spatial_from_mask(mask: MaskLike):
    np = require_numpy()
    return np.asarray(as_mask_array(mask), dtype=float)


def _residual_shapes(mask: MaskLike, kind: str) -> tuple:
    spatial = _spatial_from_mask(mask)
    if kind == "dc":
        if spatial.ndim != 1:
            raise ValueError(
                f"dc hook expects a per-Gaussian mask (N,), got {spatial.shape}"
            )
        count = int(spatial.shape[0])
        return (count, 3), (1, count, 1, 3)
    if spatial.ndim != 2:
        raise ValueError(
            f"rgb hook expects a mask (H, W), got {spatial.shape}"
        )
    height, width = int(spatial.shape[0]), int(spatial.shape[1])
    return (height, width, 3), (1, height, width, 3)


def outside_support_max_abs(delta: ArrayLike, mask: MaskLike, kind: str) -> float:
    """Max ``|δ|`` outside ``M`` after squeezing AEGIS layouts to a residual."""
    np = require_numpy()
    values = np.asarray(delta, dtype=float)
    flat, _restore = _reshape_residual(values, kind)
    return support_violation(flat, mask)


def in_loop_leakage_report(
    mask: MaskLike,
    epsilon: float,
    kind: str = "dc",
    *,
    steps: int = 3,
) -> dict:
    """Compare hooked steps with the ε-only clamp. No torch and no AEGIS.

    The gradient is ``+1``. Adaptive descent drives ``δ`` negative.
    Foolbox ``Misclassification`` ascent (what ``_run_foolbox_attack``
    gets from a raw class tensor) drives ``x`` positive. After ``steps``
    with step size ``2.5 ε / steps``, support sites sit on the makeup
    bound and outside sites stay 0 only on the hooked path.
    """
    np = require_numpy()
    if steps < 1:
        raise ValueError("steps must be >= 1")
    hook = MakeupPgdHook(mask, epsilon, kind=kind)
    adaptive_shape, foolbox_shape = _residual_shapes(mask, kind)
    step_size = 2.5 * float(epsilon) / float(steps)

    adaptive = np.zeros(adaptive_shape, dtype=float)
    adaptive_raw = np.zeros(adaptive_shape, dtype=float)
    adaptive_grad = np.ones(adaptive_shape, dtype=float)
    for _ in range(steps):
        adaptive = np.asarray(
            hook.adaptive_step(adaptive, adaptive_grad, epsilon, step_size),
            dtype=float,
        )
        adaptive_raw = epsilon_only_adaptive_step(
            adaptive_raw, adaptive_grad, epsilon, step_size
        )

    origin = np.zeros(foolbox_shape, dtype=float)
    foolbox = origin.copy()
    foolbox_raw = origin.copy()
    foolbox_grad = np.ones(foolbox_shape, dtype=float)
    for _ in range(steps):
        foolbox = np.asarray(
            hook.foolbox_project(foolbox + step_size * np.sign(foolbox_grad), origin),
            dtype=float,
        )
        foolbox_raw = epsilon_only_foolbox_project(
            foolbox_raw + step_size * np.sign(foolbox_grad),
            origin,
            epsilon,
        )

    hooked_adaptive = outside_support_max_abs(adaptive, mask, kind)
    raw_adaptive = outside_support_max_abs(adaptive_raw, mask, kind)
    hooked_foolbox = outside_support_max_abs(foolbox - origin, mask, kind)
    raw_foolbox = outside_support_max_abs(foolbox_raw - origin, mask, kind)
    return {
        "hooked_adaptive_outside_support_max_abs": hooked_adaptive,
        "epsilon_only_adaptive_outside_support_max_abs": raw_adaptive,
        "hooked_foolbox_outside_support_max_abs": hooked_foolbox,
        "epsilon_only_foolbox_outside_support_max_abs": raw_foolbox,
        "hooked_adaptive_linf": float(np.max(np.abs(adaptive))),
        "hooked_foolbox_linf": float(np.max(np.abs(foolbox - origin))),
    }


def _display_path(path: Path) -> str:
    """Prefer a path relative to the working directory."""
    try:
        return Path(path).resolve().relative_to(Path.cwd().resolve()).as_posix()
    except (OSError, ValueError):
        return str(path)


def locate_aegis_call_sites(aegis_root: Path) -> dict:
    """Read the two call sites. Does not import AEGIS and does not clone it.

    Status is ``tree_missing``, ``ready``, ``patched``, or
    ``anchors_mismatch``.
    """
    root = Path(aegis_root)
    helpers_path = root / HELPERS_RELATIVE
    model_path = root / MODEL_RELATIVE
    report = {
        "status": "tree_missing",
        "note": (
            "third_party/AEGIS is missing "
            f"({_display_path(helpers_path)} / {_display_path(model_path)}). "
            "Run scripts/00_bootstrap_aegis.sh. The hook was not applied "
            "and nothing was cloned or downloaded."
        ),
        "helpers": str(helpers_path),
        "model": str(model_path),
        "adaptive_anchor": False,
        "foolbox_anchor": False,
        "adaptive_patched": False,
        "foolbox_patched": False,
        "upstream_commit_verified": UPSTREAM_AEGIS_COMMIT,
    }
    if not helpers_path.is_file() or not model_path.is_file():
        return report

    helpers_text = helpers_path.read_text(encoding="utf-8")
    model_text = model_path.read_text(encoding="utf-8")
    report["adaptive_anchor"] = ADAPTIVE_CLAMP_ANCHOR in helpers_text and (
        ADAPTIVE_SIGN_ANCHOR in helpers_text
    )
    report["foolbox_anchor"] = FOOLBOX_ATTACK_ANCHOR in model_text
    report["adaptive_patched"] = "delta_new = _gs_makeup_project_delta(delta_new)" in helpers_text
    report["foolbox_patched"] = (
        "attack = _gs_makeup_bind_foolbox_attack(attack)" in model_text
    )
    if report["adaptive_patched"] and report["foolbox_patched"]:
        report["status"] = "patched"
        report["note"] = (
            "Makeup support calls are present in adaptive_linf_pgd_attack "
            "and AEGIS._run_foolbox_attack. Arm them with "
            "install_makeup_pgd_hook so the config exists before mask_avatar."
        )
        return report
    if report["adaptive_anchor"] and report["foolbox_anchor"]:
        report["status"] = "ready"
        report["note"] = (
            "Found adaptive_linf_pgd_attack and AEGIS._run_foolbox_attack "
            f"matching AEGIS {UPSTREAM_AEGIS_COMMIT}. The source patch has not "
            "been applied."
        )
        return report
    report["status"] = "anchors_mismatch"
    report["note"] = (
        "AEGIS sources are present but the verified L∞ step text moved. "
        "The patch was not applied. Re-check helpers.py adaptive_linf_pgd_attack "
        "and model.py AEGIS._run_foolbox_attack against "
        f"{UPSTREAM_AEGIS_COMMIT}."
    )
    return report


def _replace_once(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise ValueError(f"{label} anchor count is {count}, expected 1")
    return text.replace(old, new, 1)


def patch_aegis_sources(helpers_text: str, model_text: str) -> tuple[str, str]:
    """Return patched source text. Raises if an unpatched anchor is missing.

    Already-inserted calls are left in place.
    """
    if "delta_new = _gs_makeup_project_delta(delta_new)" not in helpers_text:
        helpers_text = _replace_once(
            helpers_text,
            ADAPTIVE_CLAMP_ANCHOR,
            ADAPTIVE_CLAMP_PATCHED,
            "adaptive_linf_pgd_attack",
        )
    if TAIL_BEGIN not in helpers_text:
        if not helpers_text.endswith("\n"):
            helpers_text += "\n"
        helpers_text += HELPERS_TAIL

    if "attack = _gs_makeup_bind_foolbox_attack(attack)" not in model_text:
        model_text = _replace_once(
            model_text,
            FOOLBOX_ATTACK_ANCHOR,
            FOOLBOX_ATTACK_PATCHED,
            "AEGIS._run_foolbox_attack",
        )
    if TAIL_BEGIN not in model_text:
        if not model_text.endswith("\n"):
            model_text += "\n"
        model_text += MODEL_TAIL
    return helpers_text, model_text


def _write_hook_config(hook: MakeupPgdHook, aegis_root: Path) -> Path:
    np = require_numpy()
    root = Path(aegis_root)
    root.mkdir(parents=True, exist_ok=True)
    mask_path = root / MASK_NAME
    weight = np.asarray(as_mask_array(hook.mask), dtype=np.float64)
    np.save(mask_path, weight)
    config_path = root / CONFIG_NAME
    payload = {
        "version": 1,
        "subject": hook.subject,
        "epsilon": hook.epsilon,
        "kind": hook.kind,
        "repo_root": str(_repo_root()),
        "mask_npy": str(mask_path.resolve()),
        "upstream_commit_verified": UPSTREAM_AEGIS_COMMIT,
        "call_sites": [
            "src/aegis/core/helpers.py:adaptive_linf_pgd_attack",
            "src/aegis/core/model.py:AEGIS._run_foolbox_attack",
        ],
    }
    config_path.write_text(
        json.dumps(payload, indent=2) + "\n",
        encoding="utf-8",
    )
    return config_path


def install_makeup_pgd_hook(
    hook: MakeupPgdHook,
    aegis_root: Path,
    *,
    mask_source: str = "from_parsing",
    allow_fallback: bool = False,
) -> dict:
    """Patch a bootstrapped AEGIS tree and arm this hook.

    Refuses ``mask_source=documented_fallback`` unless
    ``allow_fallback`` is true. That grid is not a parse of subject 306.
    A missing tree returns ``tree_missing`` and writes nothing.
    """
    global _ARMED
    sites = locate_aegis_call_sites(aegis_root)
    if sites["status"] == "tree_missing":
        sites["armed"] = False
        sites["config"] = None
        return sites
    if mask_source == "documented_fallback" and not allow_fallback:
        return {
            "status": "refused_fallback",
            "armed": False,
            "config": None,
            "note": (
                "Refusing to arm the 306 attack with documented_fallback "
                "labels. Pass a real parsing map (datasets/NeRSembleReconst/"
                "parsing/306.npy or --parsing). Nothing was written."
            ),
            "upstream_commit_verified": UPSTREAM_AEGIS_COMMIT,
        }
    if sites["status"] == "anchors_mismatch":
        sites["armed"] = False
        sites["config"] = None
        return sites

    root = Path(aegis_root)
    helpers_path = root / HELPERS_RELATIVE
    model_path = root / MODEL_RELATIVE
    helpers_text = helpers_path.read_text(encoding="utf-8")
    model_text = model_path.read_text(encoding="utf-8")
    try:
        new_helpers, new_model = patch_aegis_sources(helpers_text, model_text)
    except ValueError as exc:
        return {
            "status": "anchors_mismatch",
            "armed": False,
            "config": None,
            "note": str(exc),
            "upstream_commit_verified": UPSTREAM_AEGIS_COMMIT,
        }
    if new_helpers != helpers_text:
        helpers_path.write_text(new_helpers, encoding="utf-8")
    if new_model != model_text:
        model_path.write_text(new_model, encoding="utf-8")
    config_path = _write_hook_config(hook, root)
    clear_hook_state()
    _ARMED = hook
    return {
        "status": "patched",
        "armed": True,
        "config": str(config_path),
        "note": (
            "Patched adaptive_linf_pgd_attack and AEGIS._run_foolbox_attack. "
            f"Config {config_path.name} arms project_residual for subject "
            f"{hook.subject}, ε={hook.epsilon}, kind={hook.kind}. "
            "Run mask_avatar from this AEGIS root, or set "
            f"{HOOK_CONFIG_ENV} to the config path. Do not start that GPU "
            "run until the mentor greenlights it."
        ),
        "upstream_commit_verified": UPSTREAM_AEGIS_COMMIT,
    }


def resolve_hook_config_path() -> Optional[Path]:
    """Config from ``GS_MAKEUP_HOOK_CONFIG`` or ``./.gs_makeup_hook.json``."""
    env = os.environ.get(HOOK_CONFIG_ENV)
    if env:
        path = Path(env)
        return path if path.is_file() else None
    candidate = Path.cwd() / CONFIG_NAME
    return candidate if candidate.is_file() else None


def load_hook_from_config(path: Optional[Path] = None) -> Optional[MakeupPgdHook]:
    """Build a hook from an install config. ``None`` when no config exists.

    Raises:
        ValueError: the config is not subject 306, or the mask file is missing.
    """
    global _CONFIG_CACHE_KEY, _CONFIG_CACHE_HOOK
    np = require_numpy()
    config_path = Path(path) if path is not None else resolve_hook_config_path()
    if config_path is None or not config_path.is_file():
        return None
    stat = config_path.stat()
    key = (str(config_path.resolve()), stat.st_mtime_ns, stat.st_size)
    if key == _CONFIG_CACHE_KEY and _CONFIG_CACHE_HOOK is not None:
        return _CONFIG_CACHE_HOOK
    meta = json.loads(config_path.read_text(encoding="utf-8"))
    subject = assert_subject_306(meta.get("subject", ""))
    epsilon = float(meta["epsilon"])
    kind = str(meta.get("kind", "dc"))
    mask_path = Path(meta["mask_npy"])
    if not mask_path.is_file():
        raise ValueError(f"makeup hook mask is missing: {mask_path}")
    mask = np.load(mask_path, allow_pickle=False)
    hook = MakeupPgdHook(mask, epsilon, kind=kind, subject=subject)
    _CONFIG_CACHE_KEY = key
    _CONFIG_CACHE_HOOK = hook
    return hook


def current_hook() -> Optional[MakeupPgdHook]:
    """In-process hook, otherwise the config on disk."""
    if _ARMED is not None:
        return _ARMED
    return load_hook_from_config()


def project_configured_delta(delta: ArrayLike) -> ArrayLike:
    """Entry used by the patched ``adaptive_linf_pgd_attack`` step.

    Returns ``delta`` unchanged when no hook is armed and no config exists.
    """
    hook = current_hook()
    if hook is None:
        return delta
    return hook.project_like(delta)


def bind_foolbox_attack(attack):
    """Replace ``project`` on a Foolbox L∞ PGD attack. Other attacks pass through.

    The replacement calls the original ``project`` (ε clip) and then
    ``project_residual``. Foolbox looks up ``self.project`` on each step
    of ``BaseGradientDescent.run``.
    """
    hook = current_hook()
    if hook is None or attack is None:
        return attack
    if getattr(attack, "_gs_makeup_bound", False):
        return attack
    if type(attack).__name__ != "LinfProjectedGradientDescentAttack":
        return attack
    original = attack.project

    def project(x, x0, epsilon):
        base = original(x, x0, epsilon)
        return hook.reproject_foolbox(base, x0)

    attack.project = project
    attack._gs_makeup_bound = True
    return attack


def arm_makeup_pgd_hook(hook: Optional[MakeupPgdHook]) -> None:
    """Arm ``hook`` for this process. ``None`` disarms it."""
    global _ARMED
    _ARMED = hook

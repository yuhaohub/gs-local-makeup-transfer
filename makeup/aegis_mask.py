"""Makeup-mask constraint for the NeRSemble 306 AEGIS masking smoke.

B0-a already ran original AEGIS masking on subject 306 (AdaFace, ε=0.1,
Foolbox L∞ PGD on DC). This module builds ``M`` and projects the residual
onto ``supp(δ) ⊆ M`` and ``||δ||∞ ≤ ε``. It does not import AEGIS, does
not clone it, and does not download weights or FaceScape.

The 306 demo ships ``point_cloud.ply`` and ``renders/306.png`` only. A
``.npy`` label sidecar is read through ``SemanticMask.from_parsing``.
When that file is absent, ``documented_fallback_labels`` builds a label
grid with the same BiSeNet ids and still calls ``from_parsing``. That
grid is not a parse of subject 306.

``projected_pgd_step`` is one sign step plus ``project_residual``.
The in-loop attach path is ``makeup.aegis_pgd_hook``. It patches
``adaptive_linf_pgd_attack`` and ``AEGIS._run_foolbox_attack`` when
``third_party/AEGIS`` is present. ``aegis_pgd_hook_stub`` still returns
the call-site note and does not import AEGIS.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
from pathlib import Path
from typing import NamedTuple, Optional, Tuple, Union

from makeup._numpy import require_numpy
from makeup.mask import ArrayLike, SemanticMask, as_mask_array
from makeup.residual import MaskLike, apply_residual, project_residual

SUBJECT = "306"
B0A_EPSILON = 0.1
B0A_ATTACK_STEPS = 300
B0A_SEED = 42

# Frontal render camera in ``AEGIS._init_orbit_cam`` (GaussianAvatars
# ``OrbitCamera``). ``renders/306.png`` and ``parsing/306.npy`` are this
# view: 960×540, fovy 20°, opencv, identity orbit. B0-a passes ``--radius 1``.
FRONTAL_RENDER_HEIGHT = 540
FRONTAL_RENDER_WIDTH = 960
FRONTAL_FOVY_DEG = 20.0

# Upstream ``make mask_avatar`` points at FaceScape 527. The 306 command
# below is the one recorded in results/b0_aegis_sanity_summary.json.
MASK_AVATAR_ARGV: Tuple[str, ...] = (
    "uv",
    "run",
    "python",
    "scripts/mask_avatar.py",
    "--avatar-dir",
    "./datasets/NeRSembleReconst/avatars/306",
    "--target-image",
    "./datasets/NeRSembleReconst/renders/306.png",
    "--camera-boundary-angles",
    "-0.5",
    "0.5",
    "-0.5",
    "0.5",
    "0.0",
    "0.0",
    "--angle-aggregation",
    "mean",
    "--seed",
    str(B0A_SEED),
    "--epsilons",
    format(B0A_EPSILON, "g"),
    "--attack-steps",
    str(B0A_ATTACK_STEPS),
    "--target-features",
    "DC",
    "--adv-attack",
    "linfpgd",
    "--embedder",
    "adaface",
    "--radius",
    "1",
    "--output-name",
    "NeRSembleMasked306",
)

ADAFACE_CHECKPOINT = Path("models/adaface_ir50_ms1mv2.ckpt")
INSIGHTFACE_DETECTOR = Path("~/.insightface/models/buffalo_l/det_10g.onnx")
LAUNCH_TIMEOUT_SECONDS = 1800
# Was 20 lines / 1500 chars. That window kept the tqdm redraws and cut
# ``Epsilon: 0.100 -> Aggregated Cosine Similarity: ...`` when the line
# sat just above the tail (stdout is concatenated before a long stderr).
LOG_TAIL_LINES = 200
LOG_TAIL_CHARS = 16000
# AEGIS ``AEGIS.run`` prints this on stdout after ``save_results``.
_COSINE_LINE = re.compile(
    r"Epsilon:\s*([+-]?\d+(?:\.\d+)?)\s*->\s*"
    r"Aggregated Cosine Similarity:\s*([+-]?\d+(?:\.\d+)?)"
)

AEGIS_HOOK_TODO = (
    "TODO(james): call projected_pgd_step after the sign update in "
    "third_party/AEGIS/src/aegis/core/helpers.py:adaptive_linf_pgd_attack "
    "and from AEGIS._run_foolbox_attack in src/aegis/core/model.py. "
    "B0-a on subject 306 used Foolbox LinfPGD (scripts/mask_avatar.py, "
    "--adv-attack linfpgd, no --adaptive-epsilon); that loop only clamps "
    "ε. --select-regions is a FLAME subset inside load_gaussians, not "
    "SemanticMask.from_parsing. Keep the hook on NeRSemble 306. "
    "Superseded by makeup.aegis_pgd_hook.install_makeup_pgd_hook, which "
    "applies project_residual inside those steps when third_party/AEGIS "
    "is bootstrapped."
)


class ResolvedParsing(NamedTuple):
    """Label grid that ``SemanticMask.from_parsing`` will consume."""

    labels: ArrayLike
    mask_source: str
    path: Optional[Path]


def assert_subject_306(subject: Union[str, int]) -> str:
    """Return ``306``. Any other subject is rejected.

    Raises:
        ValueError: ``subject`` is not NeRSemble 306.
    """
    text = str(subject).strip()
    if text != SUBJECT:
        raise ValueError(
            f"formal baseline is NeRSemble subject {SUBJECT} only, got {subject!r}"
        )
    return text


def mask_avatar_argv(subject: Union[str, int] = SUBJECT) -> list[str]:
    """Argv for the B0-a 306 masking command. Does not start a process.

    The makefile target ``mask_avatar`` is FaceScape 527. This argv calls
    ``scripts/mask_avatar.py`` with the NeRSemble 306 paths from B0-a.

    Raises:
        ValueError: ``subject`` is not 306, or the argv leaves subject 306.
    """
    assert_subject_306(subject)
    argv = list(MASK_AVATAR_ARGV)
    command = " ".join(argv)
    if "FaceScape" in command or "avatars/527" in command or "renders/527" in command:
        raise ValueError("masking argv must stay on NeRSemble subject 306")
    if "NeRSembleReconst/avatars/306" not in command:
        raise ValueError("masking argv is missing the NeRSemble 306 avatar")
    return argv


def parsing_sidecar_paths(
    aegis_root: Path,
    subject: Union[str, int] = SUBJECT,
) -> Tuple[Path, Path]:
    """Return the two label-map locations checked for subject 306.

    Order: ``datasets/NeRSembleReconst/parsing/306.npy``, then
    ``datasets/NeRSembleReconst/avatars/306/parsing.npy``.
    """
    assert_subject_306(subject)
    root = Path(aegis_root)
    return (
        root / "datasets" / "NeRSembleReconst" / "parsing" / f"{subject}.npy",
        root / "datasets" / "NeRSembleReconst" / "avatars" / subject / "parsing.npy",
    )


def documented_fallback_labels(shape: Tuple[int, ...]) -> ArrayLike:
    """Integer labels with ``DEFAULT_PARSING_LABELS`` ids. Not a 306 parse.

    The 306 demo has no parsing map. This grid exists so the smoke can
    still call ``SemanticMask.from_parsing``. Sites cycle through lip
    (12, 13), eye (4, 5), brow (2, 3), cheek (1), and background (0).
    The last three sites are forced to hair ``17``, nose ``10``, and
    background ``0``, which stay outside ``M``.

    Args:
        shape: ``(N,)`` or ``(H, W)``. At least 8 sites.

    Returns:
        ``int64`` array of ``shape``.

    Raises:
        ValueError: rank or size is unusable.
        ImportError: NumPy is not installed.
    """
    np = require_numpy()
    if len(shape) not in (1, 2) or any(int(dim) <= 0 for dim in shape):
        raise ValueError(f"fallback labels need shape (N,) or (H, W), got {shape}")
    count = int(np.prod(shape))
    if count < 8:
        raise ValueError("fallback label grid needs at least 8 sites")
    pattern = np.array([12, 13, 4, 5, 2, 3, 1, 0], dtype=np.int64)
    flat = np.resize(pattern, count).astype(np.int64, copy=True)
    flat[-3] = 17
    flat[-2] = 10
    flat[-1] = 0
    return flat.reshape(shape)


def load_parsing_npy(path: Path) -> ArrayLike:
    """Load an integer label map from a ``.npy`` file.

    ``allow_pickle`` is off. ``SemanticMask.from_parsing`` checks that the
    values are integral labels.

    Raises:
        FileNotFoundError: ``path`` does not exist.
        ValueError: the suffix is not ``.npy``.
        ImportError: NumPy is not installed.
    """
    np = require_numpy()
    file_path = Path(path)
    if file_path.suffix.lower() != ".npy":
        raise ValueError(f"parsing sidecar must be a .npy label map, got {file_path.name}")
    if not file_path.is_file():
        raise FileNotFoundError(f"parsing label map not found: {file_path}")
    return np.load(file_path, allow_pickle=False)


def resolve_parsing(
    *,
    parsing: Optional[ArrayLike] = None,
    parsing_path: Optional[Path] = None,
    aegis_root: Optional[Path] = None,
    subject: Union[str, int] = SUBJECT,
    fallback_shape: Tuple[int, ...] = (12,),
) -> ResolvedParsing:
    """Choose the label grid for ``M``.

    An explicit array or ``--parsing`` file is ``mask_source=from_parsing``.
    Otherwise the first existing 306 sidecar under ``aegis_root`` is used.
    When the 306 demo has no sidecar, the result is
    ``mask_source=documented_fallback``.

    Raises:
        ValueError: both an array and a path were passed, or ``subject``
            is not 306.
        FileNotFoundError: ``parsing_path`` was passed and is missing.
        ImportError: NumPy is not installed.
    """
    assert_subject_306(subject)
    if parsing is not None and parsing_path is not None:
        raise ValueError("pass a parsing array or a parsing path, not both")
    if parsing_path is not None:
        labels = load_parsing_npy(parsing_path)
        return ResolvedParsing(labels, "from_parsing", Path(parsing_path))
    if parsing is not None:
        np = require_numpy()
        return ResolvedParsing(np.array(parsing, copy=True), "from_parsing", None)
    if aegis_root is not None:
        for candidate in parsing_sidecar_paths(aegis_root, subject):
            if candidate.is_file():
                return ResolvedParsing(
                    load_parsing_npy(candidate),
                    "from_parsing",
                    candidate,
                )
    return ResolvedParsing(
        documented_fallback_labels(fallback_shape),
        "documented_fallback",
        None,
    )


def _spatial_union(mask: MaskLike, residual_shape: tuple) -> ArrayLike:
    """Clip ``M`` to ``[0, 1]`` on the spatial grid of a 3-channel residual."""
    np = require_numpy()
    weight = np.clip(np.asarray(as_mask_array(mask), dtype=float), 0.0, 1.0)
    if not residual_shape or residual_shape[-1] != 3:
        raise ValueError(f"residual must end with 3 channels, got {residual_shape}")
    spatial = residual_shape[:-1]
    if weight.shape == spatial:
        return weight
    if weight.shape == residual_shape:
        return np.max(weight, axis=-1)
    if weight.shape == spatial + (1,):
        return weight[..., 0]
    raise ValueError(f"shape mismatch: delta {residual_shape}, mask {tuple(weight.shape)}")


def support_violation(delta: ArrayLike, mask: MaskLike) -> float:
    """Max ``|δ|`` on sites where ``M`` is 0. Zero means the support holds.

    Raises:
        ValueError: ``mask`` does not match ``delta``.
        ImportError: NumPy is not installed.
    """
    np = require_numpy()
    values = np.asarray(delta, dtype=float)
    outside = _spatial_union(mask, values.shape) <= 0.0
    if not np.any(outside):
        return 0.0
    return float(np.max(np.abs(values[outside])))


def constraint_report(
    delta: ArrayLike,
    mask: MaskLike,
    epsilon: float,
    kind: str = "dc",
) -> dict:
    """Report whether ``delta`` already lies in the feasible makeup set.

    ``constraint_holds`` is true when ``project_residual`` leaves ``delta``
    unchanged, ``|δ|`` is 0 outside ``M``, and ``||δ||∞ ≤ ε``.
    """
    np = require_numpy()
    values = np.asarray(delta, dtype=float)
    projected = np.asarray(project_residual(values, mask, epsilon, kind=kind), dtype=float)
    outside = support_violation(values, mask)
    linf = float(np.max(np.abs(values))) if values.size else 0.0
    matches = bool(np.allclose(projected, values, atol=1e-8, rtol=0.0))
    holds = matches and outside <= 1e-8 and linf <= float(epsilon) + 1e-8
    return {
        "constraint_holds": holds,
        "outside_support_max_abs": outside,
        "linf": linf,
        "matches_projection": matches,
    }


def project_attack_residual(
    base: ArrayLike,
    delta: ArrayLike,
    mask: MaskLike,
    epsilon: float,
    kind: str = "dc",
) -> tuple:
    """Project ``δ`` with ``project_residual`` and add it with ``apply_residual``.

    Returns:
        ``(projected, edited, report)``. ``report`` describes ``projected``.
    """
    projected = project_residual(delta, mask, epsilon, kind=kind)
    edited = apply_residual(base, delta, mask, epsilon, kind=kind)
    report = constraint_report(projected, mask, epsilon, kind=kind)
    return projected, edited, report


def projected_pgd_step(
    delta: ArrayLike,
    grad_sign: ArrayLike,
    mask: MaskLike,
    epsilon: float,
    step_size: float,
    kind: str = "dc",
) -> ArrayLike:
    """One L∞ sign step, then project onto ``supp(δ) ⊆ M`` and ``||δ||∞ ≤ ε``.

    AEGIS updates ``delta <- clip(delta - step_size * sign(grad), ±ε)`` and
    stops. This applies the same sign step, then ``project_residual``.
    ``grad_sign`` may be a raw gradient; only its sign is used.

    ``makeup.aegis_pgd_hook.MakeupPgdHook.adaptive_step`` is this update
    plus the per-element ε clamp from ``adaptive_linf_pgd_attack``.
    ``MakeupPgdHook.foolbox_project`` is the Foolbox L∞ project plus the
    same ``project_residual``. ``install_makeup_pgd_hook`` patches those
    two call sites when ``third_party/AEGIS`` is present. B0-a on 306
    used Foolbox (``--adv-attack linfpgd``, no ``--adaptive-epsilon``).
    ``--select-regions`` is a FLAME subset, not this mask. Keep the hook
    on subject 306.

    Raises:
        ValueError: ``step_size`` is negative, or shapes disagree.
        ImportError: NumPy is not installed.
    """
    np = require_numpy()
    if step_size < 0:
        raise ValueError("step_size must be >= 0")
    values = np.asarray(delta, dtype=float)
    signed = np.sign(np.asarray(grad_sign, dtype=float))
    if signed.shape != values.shape:
        raise ValueError(
            f"shape mismatch: delta {values.shape}, grad_sign {signed.shape}"
        )
    candidate = values - float(step_size) * signed
    return project_residual(candidate, mask, epsilon, kind=kind)


def aegis_pgd_hook_stub() -> str:
    """Return the call-site note. Does not import or patch AEGIS.

    Superseded by ``makeup.aegis_pgd_hook.install_makeup_pgd_hook``.
    """
    return AEGIS_HOOK_TODO


def _residual_layout(labels: ArrayLike) -> tuple:
    """Return ``(kind, delta_shape)`` for a ``(N,)`` or ``(H, W)`` label grid.

    This is the CPU smoke layout only. B0-a always attacks DC, so an
    ``(H, W)`` parsing map is lifted to a per-Gaussian mask before the
    hook is armed. See ``project_gaussian_means``.
    """
    np = require_numpy()
    grid = np.asarray(labels)
    if grid.ndim == 1:
        return "dc", (int(grid.shape[0]), 3)
    if grid.ndim == 2:
        return "rgb", (int(grid.shape[0]), int(grid.shape[1]), 3)
    raise ValueError(
        f"parsing labels must have shape (N,) or (H, W), got {grid.shape}"
    )


def b0a_camera_radius() -> float:
    """Return ``--radius`` from the frozen B0-a 306 argv."""
    argv = MASK_AVATAR_ARGV
    index = argv.index("--radius")
    return float(argv[index + 1])


def frontal_c2w(radius: float) -> ArrayLike:
    """Camera-to-world for the frontal AEGIS orbit camera.

    Matches ``OrbitCamera.pose`` with identity rotation, ``look_at`` at the
    origin, and ``convention="opencv"``: camera at ``(0, 0, radius)``,
    OpenCV axes (x right, y down, z forward).
    """
    np = require_numpy()
    if radius <= 0:
        raise ValueError("camera radius must be > 0")
    pose = np.eye(4, dtype=np.float64)
    pose[2, 3] = float(radius)
    pose[:, [1, 2]] *= -1.0
    return pose


def _orbit_intrinsics(height: int, width: int, fovy_deg: float) -> tuple:
    """``OrbitCamera.intrinsics`` as ``(fx, fy, cx, cy)``."""
    np = require_numpy()
    if height < 1 or width < 1:
        raise ValueError(f"image size must be positive, got {(height, width)}")
    if not 0.0 < float(fovy_deg) < 180.0:
        raise ValueError(f"fovy must be in (0, 180) degrees, got {fovy_deg}")
    focal = float(height) / (2.0 * np.tan(np.deg2rad(float(fovy_deg)) / 2.0))
    return focal, focal, int(width) // 2, int(height) // 2


def _means_xyz(means: ArrayLike) -> ArrayLike:
    """Return Gaussian means as ``(N, 3)``."""
    np = require_numpy()
    points = np.asarray(means, dtype=np.float64)
    if points.ndim == 3 and points.shape[1] == 1 and points.shape[2] == 3:
        points = points[:, 0, :]
    if points.ndim != 2 or points.shape[1] != 3 or points.shape[0] < 1:
        raise ValueError(f"Gaussian means must have shape (N, 3), got {points.shape}")
    if not np.all(np.isfinite(points)):
        raise ValueError("Gaussian means must be finite")
    return points


def project_gaussian_means(
    means: ArrayLike,
    height: int,
    width: int,
    *,
    radius: float = 1.0,
    fovy_deg: float = FRONTAL_FOVY_DEG,
) -> tuple:
    """Project world means into the frontal render.

    The camera is the identity ``OrbitCamera`` used by
    ``AEGIS._init_orbit_cam``. Pixel coordinates follow Gaussian splatting
    ``ndc2Pix`` (``((ndc + 1) * S - 1) / 2``), whose integer pixels are
    centered on those continuous coordinates. A tie at ``*.5`` resolves
    toward the higher index.

    Returns:
        ``(rows, cols, inside)``. ``inside`` is false when the mean is
        behind the camera or the nearest pixel lies outside the image.
        Outside rows and cols are ``-1``.

    Raises:
        ValueError: ``means`` is not ``(N, 3)``, or the camera is invalid.
        ImportError: NumPy is not installed.
    """
    np = require_numpy()
    points = _means_xyz(means)
    count = int(points.shape[0])
    height = int(height)
    width = int(width)
    fx, fy, cx, cy = _orbit_intrinsics(height, width, fovy_deg)
    w2c = np.linalg.inv(frontal_c2w(radius))
    hom = np.concatenate([points, np.ones((count, 1), dtype=np.float64)], axis=1)
    cam = hom @ w2c.T
    z = cam[:, 2]
    valid = z > 1e-8
    ndc_x = np.zeros(count, dtype=np.float64)
    ndc_y = np.zeros(count, dtype=np.float64)
    x_over_z = cam[:, 0] / np.where(valid, z, 1.0)
    y_over_z = cam[:, 1] / np.where(valid, z, 1.0)
    ndc_x[valid] = (2.0 * fx / float(width)) * x_over_z[valid] + (
        float(width) - 2.0 * float(cx)
    ) / float(width)
    ndc_y[valid] = (2.0 * fy / float(height)) * y_over_z[valid] + (
        float(height) - 2.0 * float(cy)
    ) / float(height)
    pix_x = np.zeros(count, dtype=np.float64)
    pix_y = np.zeros(count, dtype=np.float64)
    pix_x[valid] = ((ndc_x[valid] + 1.0) * float(width) - 1.0) * 0.5
    pix_y[valid] = ((ndc_y[valid] + 1.0) * float(height) - 1.0) * 0.5
    cols = np.floor(pix_x + 0.5).astype(np.int64)
    rows = np.floor(pix_y + 0.5).astype(np.int64)
    inside = (
        valid
        & (rows >= 0)
        & (rows < height)
        & (cols >= 0)
        & (cols < width)
    )
    rows = np.where(inside, rows, np.int64(-1))
    cols = np.where(inside, cols, np.int64(-1))
    return rows, cols, inside


def means_at_pixels(
    rows: ArrayLike,
    cols: ArrayLike,
    height: int,
    width: int,
    *,
    radius: float = 1.0,
    fovy_deg: float = FRONTAL_FOVY_DEG,
    z_cam: float = 1.0,
) -> ArrayLike:
    """World means whose ``ndc2Pix`` coordinates are the given pixel centers.

    ``rows`` and ``cols`` are integer pixel indices (the continuous
    coordinate equals that index). Used to build DC masks in tests and to
    check ``project_gaussian_means``.
    """
    np = require_numpy()
    row = np.asarray(rows, dtype=np.float64).reshape(-1)
    col = np.asarray(cols, dtype=np.float64).reshape(-1)
    if row.shape != col.shape:
        raise ValueError("rows and cols must have the same length")
    if row.size < 1:
        raise ValueError("rows and cols must be non-empty")
    if float(z_cam) <= 0.0:
        raise ValueError("z_cam must be > 0")
    height = int(height)
    width = int(width)
    fx, fy, cx, cy = _orbit_intrinsics(height, width, fovy_deg)
    ndc_x = (2.0 * col + 1.0) / float(width) - 1.0
    ndc_y = (2.0 * row + 1.0) / float(height) - 1.0
    x_over_z = (ndc_x - (float(width) - 2.0 * float(cx)) / float(width)) * (
        float(width) / (2.0 * fx)
    )
    y_over_z = (ndc_y - (float(height) - 2.0 * float(cy)) / float(height)) * (
        float(height) / (2.0 * fy)
    )
    cam = np.stack(
        [
            x_over_z * float(z_cam),
            y_over_z * float(z_cam),
            np.full(row.shape, float(z_cam), dtype=np.float64),
            np.ones(row.shape, dtype=np.float64),
        ],
        axis=1,
    )
    world = cam @ frontal_c2w(radius).T
    return world[:, :3]


def sample_image_mask(
    mask_hw: ArrayLike,
    means: ArrayLike,
    *,
    radius: float = 1.0,
    fovy_deg: float = FRONTAL_FOVY_DEG,
) -> ArrayLike:
    """Sample an ``(H, W)`` makeup mask at projected Gaussian means.

    Returns an ``(N,)`` float mask. Means that miss the image are 0.
    Values are clipped to ``[0, 1]``.
    """
    np = require_numpy()
    grid = np.clip(np.asarray(mask_hw, dtype=np.float64), 0.0, 1.0)
    if grid.ndim != 2:
        raise ValueError(f"image mask must have shape (H, W), got {grid.shape}")
    points = _means_xyz(means)
    rows, cols, inside = project_gaussian_means(
        points,
        int(grid.shape[0]),
        int(grid.shape[1]),
        radius=radius,
        fovy_deg=fovy_deg,
    )
    sampled = np.zeros(int(points.shape[0]), dtype=np.float64)
    if np.any(inside):
        sampled[inside] = grid[rows[inside], cols[inside]]
    return sampled


_PLY_DTYPES = {
    "char": "i1",
    "uchar": "u1",
    "int8": "i1",
    "uint8": "u1",
    "short": "i2",
    "ushort": "u2",
    "int16": "i2",
    "uint16": "u2",
    "int": "i4",
    "uint": "u4",
    "int32": "i4",
    "uint32": "u4",
    "float": "f4",
    "float32": "f4",
    "double": "f8",
    "float64": "f8",
}


def read_ply_xyz_binding(path: Path) -> tuple:
    """Read vertex ``x, y, z`` and optional ``binding_0`` from a Gaussian ply.

    Returns:
        ``(xyz, binding)``. ``xyz`` is ``(N, 3)`` float64 in ply order, the
        same order as ``_features_dc``. ``binding`` is ``(N,)`` int64 when
        the file has ``binding_0``, otherwise ``None``.

    Raises:
        ValueError: the ply has no vertex ``x, y, z``.
        FileNotFoundError: ``path`` is missing.
        ImportError: NumPy is not installed.
    """
    np = require_numpy()
    file_path = Path(path)
    raw = file_path.read_bytes()
    marker = b"end_header\n"
    index = raw.find(marker)
    if index < 0:
        marker = b"end_header\r\n"
        index = raw.find(marker)
    if index < 0:
        raise ValueError(f"PLY header is missing end_header: {file_path}")
    header = raw[:index].decode("ascii", errors="replace")
    body = raw[index + len(marker) :]
    fmt = "ascii"
    vertex_count = None
    properties: list[tuple[str, str]] = []
    in_vertex = False
    for line in header.splitlines():
        parts = line.strip().split()
        if not parts or parts[0] == "comment":
            continue
        if parts[0] == "format" and len(parts) >= 2:
            fmt = parts[1]
        elif parts[0] == "element" and len(parts) >= 3:
            in_vertex = parts[1] == "vertex"
            if in_vertex:
                vertex_count = int(parts[2])
        elif parts[0] == "property" and in_vertex:
            if parts[1] == "list":
                raise ValueError(f"vertex list properties are not supported: {file_path}")
            if len(parts) < 3 or parts[1] not in _PLY_DTYPES:
                raise ValueError(f"unsupported PLY property {line!r}")
            properties.append((parts[2], _PLY_DTYPES[parts[1]]))
    names = [name for name, _dtype in properties]
    if vertex_count is None or not {"x", "y", "z"} <= set(names):
        raise ValueError(f"PLY vertex element is missing x, y, z: {file_path}")
    if fmt == "ascii":
        rows = []
        for line in body.decode("ascii", errors="replace").splitlines():
            if not line.strip():
                continue
            rows.append(line.split())
            if len(rows) == vertex_count:
                break
        if len(rows) != vertex_count:
            raise ValueError(
                f"PLY vertex count is {vertex_count}, read {len(rows)} from {file_path}"
            )
        table = {name: [] for name in names}
        for row in rows:
            if len(row) < len(names):
                raise ValueError(f"short PLY vertex row in {file_path}")
            for name, value in zip(names, row):
                table[name].append(float(value))
        xyz = np.stack(
            [np.asarray(table["x"]), np.asarray(table["y"]), np.asarray(table["z"])],
            axis=1,
        ).astype(np.float64, copy=False)
        binding = None
        if "binding_0" in table:
            binding = np.rint(np.asarray(table["binding_0"], dtype=np.float64)).astype(
                np.int64
            )
        return xyz, binding
    if fmt not in {"binary_little_endian", "binary_big_endian"}:
        raise ValueError(f"unsupported PLY format {fmt!r}")
    endian = "<" if fmt == "binary_little_endian" else ">"
    dtype = np.dtype([(name, endian + code) for name, code in properties])
    need = int(vertex_count) * int(dtype.itemsize)
    if len(body) < need:
        raise ValueError(
            f"PLY body is {len(body)} bytes, vertex block needs {need}: {file_path}"
        )
    arr = np.frombuffer(body[:need], dtype=dtype, count=int(vertex_count))
    xyz = np.stack(
        [arr["x"].astype(np.float64), arr["y"].astype(np.float64), arr["z"].astype(np.float64)],
        axis=1,
    )
    binding = None
    if "binding_0" in names:
        binding = np.rint(arr["binding_0"].astype(np.float64)).astype(np.int64)
    return xyz, binding


def avatar_ply_path(aegis_root: Path, subject: Union[str, int] = SUBJECT) -> Path:
    """``point_cloud.ply`` for the B0-a subject under ``aegis_root``."""
    assert_subject_306(subject)
    return (
        Path(aegis_root)
        / "datasets"
        / "NeRSembleReconst"
        / "avatars"
        / str(subject)
        / "point_cloud.ply"
    )


def load_b0a_dc_means(aegis_root: Path, subject: Union[str, int] = SUBJECT) -> tuple:
    """World means for a DC mask, or ``(None, reason)`` when arm must defer.

    Unbound ply ``xyz`` is the rendered mean. A ``binding_0`` column means
    those coordinates are face-local offsets. The rendered mean is
    ``gaussians.get_xyz`` after the FLAME pose AEGIS already applies in
    ``load_gaussians``. That tensor exists only inside ``mask_avatar``, so
    a bound ply returns ``(None, "flame-binding-deferred")`` and the hook
    samples ``get_xyz`` on the first Foolbox bind.

    Returns:
        ``(means, source)`` with source ``ply-xyz``, ``ply-missing``, or
        ``flame-binding-deferred``.
    """
    assert_subject_306(subject)
    ply = avatar_ply_path(aegis_root, subject)
    if not ply.is_file():
        return None, "ply-missing"
    _xyz, binding = read_ply_xyz_binding(ply)
    if binding is not None:
        return None, "flame-binding-deferred"
    return _xyz, "ply-xyz"


def run_cpu_mask_smoke(
    resolved: ResolvedParsing,
    *,
    epsilon: float = B0A_EPSILON,
    steps: int = 3,
) -> dict:
    """Run a CPU L∞ sign loop and a post-hoc project. No embedder and no GPU.

    The gradient is ``+1`` everywhere, matching a Foolbox-style sign step
    that drives ``δ`` negative. After ``steps`` projected updates the
    support sites sit at ``-ε`` and every site outside ``M`` stays 0.
    An ε-only clip of the same steps (what AEGIS does without makeup ``M``)
    is reported as ``unprojected_outside_support_max_abs``.

    ``M`` always comes from ``SemanticMask.from_parsing``.

    Raises:
        ValueError: ``steps < 1``, or the projection breaks the constraint.
        ImportError: NumPy is not installed.
    """
    np = require_numpy()
    if steps < 1:
        raise ValueError("steps must be >= 1")
    mask = SemanticMask.from_parsing(resolved.labels)
    kind, delta_shape = _residual_layout(resolved.labels)
    step_size = 2.5 * float(epsilon) / float(steps)
    grad = np.ones(delta_shape, dtype=float)
    delta = np.zeros(delta_shape, dtype=float)
    for _ in range(steps):
        delta = np.asarray(
            projected_pgd_step(delta, grad, mask, epsilon, step_size, kind=kind),
            dtype=float,
        )
        step_report = constraint_report(delta, mask, epsilon, kind=kind)
        if not step_report["constraint_holds"]:
            raise ValueError("projected PGD step left the makeup feasible set")

    unprojected = np.zeros(delta_shape, dtype=float)
    signed = np.sign(grad)
    for _ in range(steps):
        unprojected = np.clip(unprojected - step_size * signed, -epsilon, epsilon)

    candidate = np.full(delta_shape, 10.0, dtype=float)
    base = np.full(delta_shape, 0.2, dtype=float)
    projected, edited, posthoc = project_attack_residual(
        base, candidate, mask, epsilon, kind=kind
    )
    projected = np.asarray(projected, dtype=float)
    edited = np.asarray(edited, dtype=float)
    spatial = np.asarray(mask.union(), dtype=float)
    outside = spatial <= 0.0
    if np.any(outside):
        outside_drift = float(np.max(np.abs(edited[outside] - base[outside])))
    else:
        outside_drift = 0.0
    apply_unchanged = outside_drift <= 1e-8
    apply_matches = bool(np.allclose(edited - base, projected, atol=1e-8, rtol=0.0))

    pgd_report = constraint_report(delta, mask, epsilon, kind=kind)
    holds = bool(
        pgd_report["constraint_holds"]
        and posthoc["constraint_holds"]
        and apply_unchanged
        and apply_matches
    )
    # Local import: aegis_pgd_hook imports this module.
    from makeup.aegis_pgd_hook import in_loop_leakage_report

    leakage = in_loop_leakage_report(mask, epsilon, kind, steps=steps)
    path = None if resolved.path is None else str(resolved.path)
    return {
        "subject": SUBJECT,
        "epsilon": float(epsilon),
        "steps": int(steps),
        "kind": kind,
        "mask_source": resolved.mask_source,
        "parsing_path": path,
        "built_with": "SemanticMask.from_parsing",
        "constraint_holds": holds,
        "pgd_outside_support_max_abs": pgd_report["outside_support_max_abs"],
        "pgd_linf": pgd_report["linf"],
        "unprojected_outside_support_max_abs": support_violation(unprojected, mask),
        "posthoc_outside_support_max_abs": posthoc["outside_support_max_abs"],
        "posthoc_linf": posthoc["linf"],
        "apply_outside_unchanged": apply_unchanged,
        "apply_matches_projected_delta": apply_matches,
        "support_fraction": float(np.mean(spatial > 0.0)),
        "aegis_hook": "in_loop",
        "aegis_hook_api": "makeup.aegis_pgd_hook.install_makeup_pgd_hook",
        "aegis_hook_todo": aegis_pgd_hook_stub(),
        "hooked_adaptive_outside_support_max_abs": leakage[
            "hooked_adaptive_outside_support_max_abs"
        ],
        "epsilon_only_adaptive_outside_support_max_abs": leakage[
            "epsilon_only_adaptive_outside_support_max_abs"
        ],
        "hooked_foolbox_outside_support_max_abs": leakage[
            "hooked_foolbox_outside_support_max_abs"
        ],
        "epsilon_only_foolbox_outside_support_max_abs": leakage[
            "epsilon_only_foolbox_outside_support_max_abs"
        ],
    }


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


def _shown_path(path: Path) -> str:
    """Prefer a path relative to the working directory for summary text."""
    try:
        return Path(path).resolve().relative_to(Path.cwd().resolve()).as_posix()
    except (OSError, ValueError):
        return str(path)


def collect_launch_blockers(aegis_root: Path) -> list[str]:
    """Reasons ``--launch`` must not start ``mask_avatar.py``.

    Missing weights and the InsightFace detector are blockers so the smoke
    does not download them. Nothing is fetched here.
    """
    root = Path(aegis_root)
    blockers: list[str] = []
    if not (root / ".git").is_dir():
        blockers.append(
            "bootstrap_required: third_party/AEGIS is missing "
            "(scripts/00_bootstrap_aegis.sh). The smoke does not clone it."
        )
    if shutil.which("uv") is None:
        blockers.append("deps_missing: uv is not on PATH")
    ply = root / "datasets" / "NeRSembleReconst" / "avatars" / "306" / "point_cloud.ply"
    png = root / "datasets" / "NeRSembleReconst" / "renders" / "306.png"
    checkpoint = root / ADAFACE_CHECKPOINT
    if not ply.is_file():
        blockers.append(f"deps_missing: 306 point_cloud.ply missing ({_shown_path(ply)})")
    if not png.is_file():
        blockers.append(f"deps_missing: 306 render missing ({_shown_path(png)})")
    if not checkpoint.is_file():
        blockers.append(
            "deps_missing: AdaFace checkpoint missing "
            f"({_shown_path(checkpoint)}); not downloaded"
        )
    if not INSIGHTFACE_DETECTOR.expanduser().is_file():
        blockers.append(
            "deps_missing: insightface detector missing "
            f"({INSIGHTFACE_DETECTOR}); mask_avatar would auto-download it"
        )
    gaussian_avatars = root / "GaussianAvatars"
    if not gaussian_avatars.is_dir():
        blockers.append(
            "deps_missing: GaussianAvatars checkout missing "
            f"({_shown_path(gaussian_avatars)})"
        )
    if not cuda_available():
        blockers.append("gpu_required: CUDA device not available")
    return blockers


def launch_mask_avatar(aegis_root: Path) -> dict:
    """Start the B0-a 306 command when every local input is already present.

    Returns a dict with ``launched``, ``return_code``, ``blockers``,
    ``log_tail``, ``aggregated_cosine_similarity``, and ``epsilon_cosine``.
    Cosine fields are parsed from the full stdout and stderr before the
    tail is cut. A blocker list means the process was not started.
    """
    blockers = collect_launch_blockers(aegis_root)
    if blockers:
        return _launch_payload(
            launched=False,
            return_code=None,
            blockers=blockers,
            log="",
        )
    argv = mask_avatar_argv()
    try:
        proc = subprocess.run(
            argv,
            cwd=Path(aegis_root),
            capture_output=True,
            text=True,
            timeout=LAUNCH_TIMEOUT_SECONDS,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        log = ""
        if isinstance(exc.stdout, str):
            log += exc.stdout
        if isinstance(exc.stderr, str):
            log += exc.stderr
        return _launch_payload(
            launched=True,
            return_code=None,
            blockers=[],
            log=log,
        )
    except OSError as exc:
        return _launch_payload(
            launched=False,
            return_code=None,
            blockers=[f"deps_missing: mask_avatar could not be started: {exc}"],
            log="",
        )
    log = (proc.stdout or "") + (proc.stderr or "")
    return _launch_payload(
        launched=True,
        return_code=proc.returncode,
        blockers=[],
        log=log,
    )


def _launch_payload(
    *,
    launched: bool,
    return_code: Optional[int],
    blockers: list[str],
    log: str,
) -> dict:
    """Tail plus Foolbox cosine parsed from the uncut log."""
    metrics = extract_launch_metrics(log)
    return {
        "launched": launched,
        "return_code": return_code,
        "blockers": blockers,
        "log_tail": _log_tail(log),
        "aggregated_cosine_similarity": metrics["aggregated_cosine_similarity"],
        "epsilon_cosine": metrics["epsilon_cosine"],
    }


def extract_launch_metrics(text: str) -> dict:
    """Read Foolbox ``Epsilon -> Aggregated Cosine Similarity`` lines.

    Scans the full captured stdout and stderr, before ``_log_tail`` cuts
    them. ``aggregated_cosine_similarity`` is the last printed value
    (B0-a prints one line, for ε=0.1). ``epsilon_cosine`` keeps every line.
    """
    rows = []
    for eps, sim in _COSINE_LINE.findall(text or ""):
        rows.append(
            {
                "epsilon": float(eps),
                "aggregated_cosine_similarity": float(sim),
            }
        )
    cosine = None if not rows else rows[-1]["aggregated_cosine_similarity"]
    return {
        "aggregated_cosine_similarity": cosine,
        "epsilon_cosine": rows,
    }


def _log_tail(text: str) -> str:
    """Last lines of a mask_avatar log, with cosine summaries pinned.

    A 20-line / 1500-character tail dropped
    ``Aggregated Cosine Similarity`` when tqdm redraws filled the window
    or the line sat above it. Summary lines are appended so a char trim
    of the front cannot remove them.
    """
    lines = [line for line in (text or "").splitlines() if line.strip()]
    summary = [line for line in lines if _COSINE_LINE.search(line)]
    body = [
        line
        for line in lines[-LOG_TAIL_LINES:]
        if not _COSINE_LINE.search(line)
    ]
    extra = "\n".join(summary)
    head = "\n".join(body)
    if not extra:
        if len(head) > LOG_TAIL_CHARS:
            return head[-LOG_TAIL_CHARS:]
        return head
    if len(extra) >= LOG_TAIL_CHARS:
        return extra[-LOG_TAIL_CHARS:]
    separator = "\n" if head else ""
    budget = LOG_TAIL_CHARS - len(extra) - len(separator)
    if len(head) > budget:
        head = head[-budget:] if budget > 0 else ""
        separator = "\n" if head else ""
    return f"{head}{separator}{extra}"


def _armed_dc_sentence(attach: Optional[dict]) -> str:
    """How an armed B0-a hook turns image ``M`` into a DC mask."""
    if not attach or not attach.get("hook_kind"):
        return ""
    return (
        f"Armed B0-a hook kind={attach.get('hook_kind')} "
        f"lift={attach.get('dc_mask_lift')} "
        f"mask_shape={attach.get('hook_mask_shape')} "
        f"means={attach.get('dc_means_source')}. "
        "Image M from the (H, W) parsing is sampled at ndc2Pix projections "
        "of the Gaussian means through the frontal OrbitCamera "
        "(960x540, fovy 20, radius 1, opencv), giving one weight per DC row. "
        "A FLAME-bound ply samples gaussians.get_xyz inside mask_avatar "
        "before the first Foolbox step. "
    )


def smoke_note(smoke: dict, launch: dict, attach: Optional[dict] = None) -> str:
    """One-paragraph summary for the tracked JSON/CSV."""
    if smoke["mask_source"] == "documented_fallback":
        origin = (
            "Parsing labels are absent on the 306 demo assets "
            "(point_cloud.ply and renders/306.png only). "
            "mask_source=documented_fallback is a BiSeNet-id label grid "
            "passed through SemanticMask.from_parsing; it is not a parse of "
            "subject 306. Place datasets/NeRSembleReconst/parsing/306.npy "
            "or pass --parsing to use a real map."
        )
    else:
        where = smoke.get("parsing_path") or "an in-memory label array"
        origin = f"M was built with SemanticMask.from_parsing from {where}."
    if launch.get("launched"):
        gpu = (
            f"mask_avatar.py was started (return_code={launch.get('return_code')}). "
            "It multiplies by M only when the makeup hook config is already "
            "installed in that AEGIS root."
        )
        cosine = launch.get("aggregated_cosine_similarity")
        if cosine is not None:
            gpu += f" Aggregated cosine similarity={cosine}."
    else:
        gpu = "GPU mask_avatar was not started."
        blockers = launch.get("blockers") or []
        if blockers:
            gpu += " Blockers: " + "; ".join(blockers) + "."
    return (
        "NeRSemble 306 makeup-mask smoke enforced supp(delta) ⊆ M and "
        f"||delta||_inf ≤ {smoke['epsilon']} on a CPU projected L∞ step "
        f"({smoke['kind']}). {origin} {gpu} "
        "In-loop projection is makeup.aegis_pgd_hook.MakeupPgdHook: "
        "project_residual runs inside adaptive_linf_pgd_attack after the "
        "sign step and ε clamp, and inside Foolbox L∞ project from "
        "_run_foolbox_attack. "
        + _armed_dc_sentence(attach)
        + "Hooked outside-support max |delta| is "
        f"{smoke.get('hooked_adaptive_outside_support_max_abs')} (adaptive) "
        f"and {smoke.get('hooked_foolbox_outside_support_max_abs')} (Foolbox); "
        "ε-clamp-only leaks "
        f"{smoke.get('epsilon_only_adaptive_outside_support_max_abs')} and "
        f"{smoke.get('epsilon_only_foolbox_outside_support_max_abs')}. "
        + (
            (attach.get("note") + " ")
            if attach and attach.get("note")
            else "AEGIS attach was not requested. "
        )
        + "Do not open an AutoDL GPU masking run until the mentor greenlights it."
    )


def command_text() -> str:
    """Shell-joined B0-a 306 command."""
    return " ".join(mask_avatar_argv())

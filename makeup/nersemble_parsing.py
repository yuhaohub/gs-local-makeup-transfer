"""One-image BiSeNet parse for NeRSemble subject 306.

The NeRSemble demo ships ``renders/306.png`` and no label map. This module
detects the face, crops it, runs the CelebAMask-HQ BiSeNet from
face-parsing.PyTorch on that crop, and pastes the labels back onto a
background canvas the size of the original image. The written ``(H, W)``
integer ``.npy`` uses ``DEFAULT_PARSING_LABELS`` and stays aligned with
``renders/306.png``.

The detector is the one AEGIS already uses: InsightFace ``buffalo_l``
``det_10g.onnx`` via ``insightface.model_zoo.get_model`` (RetinaFace/SCRFD
``detect``). A separate ``retinaface`` package is not required. The onnx
file is not downloaded.

Torch, torchvision, Pillow, and InsightFace are imported only when a parse
runs. The face-parsing checkout and ``79999_iter.pth`` are not downloaded.
CUDA is used only when it is already available. Nothing here starts AEGIS
masking.
"""

from __future__ import annotations

import importlib.util
import math
import os
import sys
from pathlib import Path
from typing import NamedTuple, Optional, Sequence, Tuple

from makeup._numpy import require_numpy
from makeup.aegis_mask import parsing_sidecar_paths
from makeup.mask import DEFAULT_PARSING_LABELS, REGIONS, SemanticMask

N_CLASSES = 19
INPUT_SIZE = 512
WEIGHT_FILENAME = "79999_iter.pth"
FACE_PARSING_URL = "https://github.com/zllrunning/face-parsing.PyTorch"
WEIGHTS_URL = "https://drive.google.com/open?id=154JgKpzCPW82qINcVieuPH3fZ2e0P812"
FACE_PARSING_DIRNAME = "face-parsing.PyTorch"
WEIGHTS_REL = Path("res") / "cp" / WEIGHT_FILENAME
RENDER_REL = Path("third_party/AEGIS/datasets/NeRSembleReconst/renders/306.png")
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
# Same file AEGIS refuses to auto-download before masking.
INSIGHTFACE_PACKAGE = "buffalo_l"
INSIGHTFACE_FILENAME = "det_10g.onnx"
DET_INPUT_SIZE = (640, 640)
DEFAULT_CROP_MARGIN = 0.25
REQUIRED_REGION_NAMES: Tuple[str, ...] = ("lip", "eye", "brow", "cheek")
VIZ_COMMAND = "python scripts/viz_nersemble306_parsing.py"
DETECTOR_INSTALL = (
    "Install the detector AEGIS already uses: InsightFace buffalo_l "
    "det_10g.onnx, loaded with insightface.model_zoo.get_model "
    "(RetinaFace/SCRFD detect API). A separate retinaface package is not required.\n"
    "  pip install insightface onnxruntime\n"
    "If that import fails on cv2, also install opencv-python-headless.\n"
    "Weights must already be at ~/.insightface/models/buffalo_l/det_10g.onnx. "
    "This script does not download them and does not call "
    "insightface.utils.ensure_available. "
    "On AutoDL that file is already on disk from the AEGIS stack. "
    "Pass --cpu to force the CPU provider. This does not start GPU masking."
)

REPO_ROOT = Path(__file__).resolve().parents[1]


class ParsingSetupError(RuntimeError):
    """The 306 parse cannot run because a local input or dependency is missing."""


class ParsingRejectedError(RuntimeError):
    """A label map was produced but lip, eye, brow, or cheek has no pixels."""


class ParsingResult(NamedTuple):
    """Label map written for subject 306."""

    path: Path
    labels: object
    summary: str
    device: object
    regions: dict
    counts: dict
    crop: object


class CropBox(NamedTuple):
    """Inclusive-exclusive crop in original-image pixels: ``[y0:y1, x0:x1]``."""

    y0: int
    y1: int
    x0: int
    x1: int

    @property
    def shape(self) -> Tuple[int, int]:
        return (self.y1 - self.y0, self.x1 - self.x0)


class FaceBox(NamedTuple):
    """One InsightFace box in original-image pixels, ``x`` right and ``y`` down."""

    x0: float
    y0: float
    x1: float
    y1: float
    score: float


class CropParse(NamedTuple):
    """Full-canvas labels plus the crop they were pasted from."""

    labels: object
    box: CropBox
    score: float


def insightface_detector_path() -> Path:
    """Return ``~/.insightface/models/buffalo_l/det_10g.onnx``.

    This is the detector file AEGIS already checks before masking. The path
    is not created and the file is not downloaded.
    """
    return (
        Path.home()
        / ".insightface"
        / "models"
        / INSIGHTFACE_PACKAGE
        / INSIGHTFACE_FILENAME
    )


def default_output_paths(repo_root: Path) -> Tuple[Path, Path]:
    """Primary ``parsing/306.npy`` and alternate ``avatars/306/parsing.npy``.

    Both sit under ``third_party/AEGIS``. The demo does not ship either file.
    """
    return parsing_sidecar_paths(Path(repo_root) / "third_party" / "AEGIS")


def resolve_input_image(image: Optional[Path], repo_root: Path) -> Path:
    """Return the RGB render to parse.

    ``image`` wins when it is passed. Otherwise the AEGIS demo render
    ``renders/306.png`` is used when that file is already on disk.

    Raises:
        ParsingSetupError: the chosen file is not on disk. This function
            does not download the demo.
    """
    if image is not None:
        path = Path(image)
        if not path.is_file():
            raise ParsingSetupError(
                f"Input image not found: {path}\n"
                "Pass --image pointing at a frontal RGB render of NeRSemble "
                "subject 306. This script does not download the demo."
            )
        return path
    path = Path(repo_root) / RENDER_REL
    if path.is_file():
        return path
    raise ParsingSetupError(
        "No input image. The NeRSemble 306 demo render is not at:\n"
        f"  {path}\n"
        "The demo does not ship parsing/306.npy. Pass --image PATH to a "
        "frontal RGB render of subject 306 (AEGIS renders/306.png once the "
        "demo is on disk). This script does not download the demo."
    )


def resolve_face_parsing_repo(repo: Optional[Path], repo_root: Path) -> Path:
    """Return a local face-parsing.PyTorch checkout that contains ``model.py``.

    Search order: ``repo`` argument, ``FACE_PARSING_PYTORCH`` environment
    variable, then ``third_party/face-parsing.PyTorch`` under ``repo_root``.

    Raises:
        ParsingSetupError: ``model.py`` is not in the chosen directory.
            This function does not clone the repository.
    """
    if repo is not None:
        path = Path(repo)
    else:
        env = os.environ.get("FACE_PARSING_PYTORCH", "").strip()
        path = Path(env) if env else Path(repo_root) / "third_party" / FACE_PARSING_DIRNAME
    if not (path / "model.py").is_file():
        raise ParsingSetupError(
            f"face-parsing.PyTorch checkout not found (no model.py at {path}).\n"
            "Clone it, then rerun (this script does not clone):\n"
            f"  git clone {FACE_PARSING_URL}.git "
            f"third_party/{FACE_PARSING_DIRNAME}\n"
            "Or pass --repo PATH, or set FACE_PARSING_PYTORCH."
        )
    return path


def resolve_weights(weights: Optional[Path], face_repo: Path) -> Path:
    """Return the CelebAMask-HQ BiSeNet checkpoint.

    The default location is ``res/cp/79999_iter.pth`` inside the
    face-parsing checkout, which is where that project's ``test.py`` reads it.

    Raises:
        ParsingSetupError: the file is not on disk. This function does not
            download weights.
    """
    path = Path(weights) if weights is not None else Path(face_repo) / WEIGHTS_REL
    if path.is_file():
        return path
    raise ParsingSetupError(
        f"BiSeNet weights not found: {path}\n"
        "face-parsing.PyTorch does not ship the checkpoint. Download the "
        "CelebAMask-HQ BiSeNet file 79999_iter.pth (19 classes) linked from "
        "that project's README:\n"
        f"  {WEIGHTS_URL}\n"
        "Place it at third_party/face-parsing.PyTorch/res/cp/79999_iter.pth "
        "or pass --weights PATH.\n"
        "This script does not download weights. CPU is enough; CUDA is used "
        "only when it is already available (pass --cpu to force CPU)."
    )


def require_torch():
    """Return ``torch``, or raise a setup error with a CPU install hint."""
    try:
        import torch
    except ImportError as exc:
        raise ParsingSetupError(
            "PyTorch is required to run face-parsing.PyTorch BiSeNet. "
            "A CPU wheel is enough (pip install torch torchvision). "
            "CUDA is used only when it is already available. "
            "This script does not install packages."
        ) from exc
    return torch


def require_pillow():
    """Return ``PIL.Image``, or raise a setup error with an install hint."""
    try:
        from PIL import Image
    except ImportError as exc:
        raise ParsingSetupError(
            "Pillow is required to read the input image (pip install pillow). "
            "This script does not install packages."
        ) from exc
    return Image


def select_device(force_cpu: bool):
    """Use CUDA only when it is already available and ``force_cpu`` is false.

    Raises:
        ParsingSetupError: PyTorch is not installed.
    """
    torch = require_torch()
    if force_cpu or not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device("cuda")


def resize_label_map(labels, size_hw: Tuple[int, int]):
    """Nearest-neighbor resize of an integer label grid to ``(height, width)``.

    Class ids are copied, not blended. The source pixel for a destination
    pixel ``(y, x)`` is ``(y * src_h // dst_h, x * src_w // dst_w)``.

    Raises:
        ParsingSetupError: ``labels`` is not a 2-D grid, or ``size_hw`` is
            not a positive ``(H, W)``.
        ImportError: NumPy is not installed.
    """
    np = require_numpy()
    array = np.asarray(labels)
    if array.ndim != 2 or array.size == 0:
        raise ParsingSetupError(
            f"label map must be a non-empty (H, W) grid, got shape {array.shape}"
        )
    height, width = (int(size_hw[0]), int(size_hw[1]))
    if height <= 0 or width <= 0:
        raise ParsingSetupError(f"output size must be positive, got {(height, width)}")
    if array.shape == (height, width):
        return np.array(array, dtype=np.int64, copy=True)
    src_h, src_w = array.shape
    rows = np.minimum((np.arange(height) * src_h) // height, src_h - 1)
    cols = np.minimum((np.arange(width) * src_w) // width, src_w - 1)
    return np.array(array[rows[:, None], cols[None, :]], dtype=np.int64, copy=True)


def finalize_labels(labels):
    """Return a contiguous ``int64`` ``(H, W)`` map with ids in ``0..18``.

    Integral floats are accepted and cast. Any other dtype or an id outside
    the 19 CelebAMask-HQ classes is rejected.

    Raises:
        ParsingSetupError: the array is not an integer label grid.
        ImportError: NumPy is not installed.
    """
    np = require_numpy()
    array = np.asarray(labels)
    if array.ndim != 2 or array.size == 0:
        raise ParsingSetupError(
            f"label map must be a non-empty (H, W) grid, got shape {array.shape}"
        )
    if array.dtype.kind == "f":
        if not np.all(np.isfinite(array)) or not np.all(array == np.rint(array)):
            raise ParsingSetupError("label map must be integer class ids")
        array = np.rint(array)
    elif array.dtype.kind not in "iu":
        raise ParsingSetupError(
            f"label map must be integer class ids, got dtype {array.dtype}"
        )
    out = np.array(array, dtype=np.int64, copy=True)
    low = int(out.min())
    high = int(out.max())
    if low < 0 or high >= N_CLASSES:
        raise ParsingSetupError(
            f"label ids must be in 0..{N_CLASSES - 1} "
            "(face-parsing.PyTorch CelebAMask-HQ), "
            f"got min {low} max {high}"
        )
    return out


def logits_to_label_map(logits, out_hw: Tuple[int, int], *, expect_hw: Optional[Tuple[int, int]] = None):
    """Argmax BiSeNet logits and nearest-resize them onto ``out_hw``.

    ``logits`` is ``(C, H, W)`` or ``(1, C, H, W)``, a NumPy array or a torch
    tensor. ``C`` must be 19. When ``expect_hw`` is set, the pre-resize grid
    must match it (the network input is 512×512 in ``test.py``).

    Returns:
        ``int64`` array of shape ``out_hw``.

    Raises:
        ParsingSetupError: the tensor rank, batch, class count, or spatial
            size is not the face-parsing layout.
        ImportError: NumPy is not installed.
    """
    np = require_numpy()
    if hasattr(logits, "detach"):
        array = logits.detach().cpu().numpy()
    else:
        array = np.asarray(logits)
    if array.ndim == 4:
        if array.shape[0] != 1:
            raise ParsingSetupError(
                f"BiSeNet logits batch must be 1, got shape {tuple(array.shape)}"
            )
        array = array[0]
    if array.ndim != 3:
        raise ParsingSetupError(
            "BiSeNet logits must be (C, H, W) or (1, C, H, W), "
            f"got shape {tuple(array.shape)}"
        )
    if int(array.shape[0]) != N_CLASSES:
        raise ParsingSetupError(
            f"BiSeNet returned {int(array.shape[0])} classes, expected {N_CLASSES} "
            "(face-parsing.PyTorch CelebAMask-HQ). Use 79999_iter.pth."
        )
    parsing = np.argmax(array, axis=0)
    if expect_hw is not None and tuple(parsing.shape) != tuple(expect_hw):
        raise ParsingSetupError(
            f"BiSeNet label grid is {tuple(parsing.shape)}, expected {tuple(expect_hw)} "
            "after the 512 input used by face-parsing.PyTorch test.py."
        )
    return finalize_labels(resize_label_map(parsing, out_hw))


def region_presence(labels) -> dict:
    """Whether each ``DEFAULT_PARSING_LABELS`` region has at least one pixel.

    Keys follow ``REGIONS``: lip, eye, brow, cheek. Skin id 1 counts as cheek.
    """
    np = require_numpy()
    present = {int(value) for value in np.unique(np.asarray(labels))}
    flags = {}
    for region in REGIONS:
        ids = [label_id for label_id, name in DEFAULT_PARSING_LABELS.items() if name == region]
        flags[region.value] = any(label_id in present for label_id in ids)
    return flags


def region_pixel_counts(labels) -> dict:
    """Count pixels in lip, eye, brow, and cheek via ``SemanticMask.from_parsing``.

    The mask uses ``DEFAULT_PARSING_LABELS`` (lip 12/13, eye 4/5, brow 2/3,
    skin 1 as cheek). A count is ``np.count_nonzero(region > 0)``, the same
    notion ``scripts/viz_nersemble306_parsing.py`` prints.

    Raises:
        ValueError: ``labels`` is not an integer parsing grid.
        ImportError: NumPy is not installed.
    """
    np = require_numpy()
    mask = SemanticMask.from_parsing(labels)
    counts = {}
    for name in REQUIRED_REGION_NAMES:
        counts[name] = int(np.count_nonzero(np.asarray(mask.region(name)) > 0))
    return counts


def require_nonzero_makeup_regions(labels, *, path: Optional[Path] = None) -> dict:
    """Reject a map whose lip, eye, brow, or cheek count is 0.

    Counts come from :func:`region_pixel_counts`. The ``.npy`` is left on
    disk when ``path`` is set; the message says it is not accepted.

    Returns:
        The four region counts, each greater than 0.

    Raises:
        ParsingRejectedError: one or more required regions are empty.
        ValueError: ``labels`` is not an integer parsing grid.
        ImportError: NumPy is not installed.
    """
    counts = region_pixel_counts(labels)
    empty = [name for name in REQUIRED_REGION_NAMES if counts[name] <= 0]
    if not empty:
        return counts
    detail = " ".join(f"{name}={counts[name]}" for name in REQUIRED_REGION_NAMES)
    where = f"wrote: {path}\n" if path is not None else ""
    raise ParsingRejectedError(
        "Rejected the label map: SemanticMask.from_parsing with "
        "DEFAULT_PARSING_LABELS needs lip, eye, brow, and cheek each to "
        f"have pixel count > 0. Empty: {', '.join(empty)}. counts: {detail}. "
        "Ids are lip 12/13, eye 4/5, brow 2/3, cheek/skin 1.\n"
        f"{where}"
        "The .npy was written and is not accepted. Do not use it for masking.\n"
        f"Inspect it with {VIZ_COMMAND} "
        "(that command exits 1 when a required region is empty).\n"
        "A crop that misses the mouth leaves lip at 0; retry with a larger --margin."
    )


def format_summary(
    *,
    path: Path,
    labels,
    image: Path,
    device,
    regions: dict,
    counts: Optional[dict] = None,
    crop: Optional[CropBox] = None,
    margin: Optional[float] = None,
    square: Optional[bool] = None,
    score: Optional[float] = None,
) -> str:
    """Short report: shape, unique ids, region counts, and the face crop."""
    np = require_numpy()
    array = np.asarray(labels)
    unique = " ".join(str(int(value)) for value in np.unique(array))
    flags = " ".join(
        f"{name}={'yes' if regions.get(name) else 'no'}"
        for name in REQUIRED_REGION_NAMES
    )
    if counts is None:
        counts = region_pixel_counts(array)
    count_text = " ".join(f"{name}={int(counts[name])}" for name in REQUIRED_REGION_NAMES)
    id_counts = {
        int(value): int(count)
        for value, count in zip(*np.unique(array, return_counts=True))
    }
    watched = " ".join(
        f"{label_id}={id_counts.get(label_id, 0)}" for label_id in (1, 2, 3, 4, 5, 12, 13)
    )
    device_text = getattr(device, "type", device)
    lines = [
        f"wrote: {path}",
        f"shape: {tuple(int(value) for value in array.shape)}",
        f"dtype: {array.dtype}",
        f"unique_ids: {unique}",
        f"DEFAULT_PARSING_LABELS: {flags}",
        f"region_counts: {count_text}",
        f"default_id_counts: {watched}",
        f"image: {image}",
        f"device: {device_text}",
        "detector: insightface buffalo_l det_10g.onnx",
    ]
    if crop is not None:
        lines.append(
            f"crop: y0={int(crop.y0)} y1={int(crop.y1)} x0={int(crop.x0)} x1={int(crop.x1)}"
        )
    if margin is not None:
        lines.append(f"crop_margin: {float(margin):.2f}")
    if square is not None:
        lines.append(f"crop_square: {'yes' if square else 'no'}")
    if score is not None:
        lines.append(f"face_score: {float(score):.4f}")
    lines.append(f"next: {VIZ_COMMAND}")
    return "\n".join(lines)


def save_label_map(path: Path, labels) -> Path:
    """Write ``labels`` as an ``int64`` ``.npy`` of shape ``(H, W)``.

    Parent directories are created. ``allow_pickle`` is not used.

    Raises:
        ParsingSetupError: the suffix is not ``.npy``, or the array is not
            a 19-class label grid.
        ImportError: NumPy is not installed.
    """
    np = require_numpy()
    file_path = Path(path)
    if file_path.suffix.lower() != ".npy":
        raise ParsingSetupError(f"output must be a .npy label map, got {file_path}")
    array = finalize_labels(labels)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(file_path, array)
    return file_path


def coerce_state_dict(state) -> dict:
    """Return a flat state dict, stripping a leading ``module.`` prefix.

    A checkpoint wrapped as ``{"state_dict": ...}`` is unwrapped. The
    face-parsing ``79999_iter.pth`` file is already a flat state dict.

    Raises:
        ParsingSetupError: ``state`` is not a non-empty string-keyed dict.
    """
    if isinstance(state, dict) and isinstance(state.get("state_dict"), dict):
        state = state["state_dict"]
    if not isinstance(state, dict) or not state:
        raise ParsingSetupError(
            "BiSeNet checkpoint is not a state_dict. Expected 79999_iter.pth "
            "from face-parsing.PyTorch (a dict of parameter tensors)."
        )
    cleaned = {}
    for key, value in state.items():
        if not isinstance(key, str):
            raise ParsingSetupError(f"BiSeNet checkpoint key is not a string: {key!r}")
        name = key[len("module.") :] if key.startswith("module.") else key
        cleaned[name] = value
    return cleaned


def disable_resnet_imagenet_download(resnet_module) -> None:
    """Stop ``Resnet18.init_weight`` from downloading ImageNet weights.

    face-parsing.PyTorch calls ``modelzoo.load_url`` for ResNet-18 when
    ``BiSeNet`` is constructed. ``79999_iter.pth`` already contains those
    layers and is loaded next, so the ImageNet fetch is skipped. The
    CelebAMask-HQ checkpoint is not replaced by random weights: ``load_state_dict``
    still runs after construction.

    Raises:
        ParsingSetupError: the checkout has no ``Resnet18`` class.
    """
    cls = getattr(resnet_module, "Resnet18", None)
    if cls is None:
        raise ParsingSetupError(
            "face-parsing.PyTorch resnet.py has no Resnet18. "
            f"Clone {FACE_PARSING_URL} unchanged."
        )

    def init_weight(self):
        return None

    cls.init_weight = init_weight


def import_bisenet(face_repo: Path):
    """Import ``BiSeNet`` from a face-parsing.PyTorch checkout.

    ``resnet.py`` must sit next to ``model.py``. ImageNet download inside
    ``Resnet18`` is disabled before the class is returned.

    Raises:
        ParsingSetupError: the checkout, torch, or torchvision cannot be imported.
    """
    require_torch()
    repo = Path(face_repo)
    model_path = repo / "model.py"
    if not model_path.is_file() or not (repo / "resnet.py").is_file():
        raise ParsingSetupError(
            f"face-parsing.PyTorch at {repo} needs model.py and resnet.py. "
            f"Clone {FACE_PARSING_URL} unchanged."
        )
    for name in ("resnet",):
        existing = sys.modules.get(name)
        existing_file = getattr(existing, "__file__", None) if existing is not None else None
        if existing_file is not None and Path(existing_file).resolve().parent != repo.resolve():
            del sys.modules[name]
    repo_s = str(repo)
    added = repo_s not in sys.path
    if added:
        sys.path.insert(0, repo_s)
    spec = importlib.util.spec_from_file_location("face_parsing_pytorch_bisenet", model_path)
    if spec is None or spec.loader is None:
        raise ParsingSetupError(f"Could not load BiSeNet from {model_path}")
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except ImportError as exc:
        raise ParsingSetupError(
            "Could not import BiSeNet from face-parsing.PyTorch. "
            "That checkout needs torch and torchvision (CPU wheels are enough) "
            f"and resnet.py next to model.py.\n{exc}"
        ) from exc
    except Exception as exc:
        raise ParsingSetupError(
            f"Could not import BiSeNet from {model_path}.\n{exc}"
        ) from exc
    finally:
        if added and repo_s in sys.path:
            sys.path.remove(repo_s)
    resnet_module = sys.modules.get("resnet")
    if resnet_module is None:
        raise ParsingSetupError(
            f"{model_path} did not import resnet.Resnet18. "
            f"Clone {FACE_PARSING_URL} unchanged."
        )
    disable_resnet_imagenet_download(resnet_module)
    if not hasattr(module, "BiSeNet"):
        raise ParsingSetupError(f"{model_path} has no BiSeNet class")
    return module.BiSeNet


def _torch_load(torch, path: Path, device):
    try:
        return torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=device)
    except Exception as exc:
        raise ParsingSetupError(
            f"Could not read BiSeNet weights {path} as a tensor state_dict ({exc}). "
            "Use 79999_iter.pth from face-parsing.PyTorch. "
            "This script does not download weights."
        ) from exc


def load_bisenet(face_repo: Path, weights: Path, device):
    """Build CelebAMask-HQ BiSeNet on ``device`` and load ``weights``.

    ``Resnet18.init_weight`` is not allowed to download ImageNet weights.
    The returned module is in ``eval`` mode.

    Raises:
        ParsingSetupError: import or ``load_state_dict`` failed.
    """
    torch = require_torch()
    bisenet_cls = import_bisenet(face_repo)
    try:
        net = bisenet_cls(n_classes=N_CLASSES)
    except Exception as exc:
        raise ParsingSetupError(
            "Could not construct face-parsing.PyTorch BiSeNet "
            f"(n_classes={N_CLASSES}). ImageNet ResNet-18 download is disabled; "
            f"79999_iter.pth is loaded instead.\n{exc}"
        ) from exc
    state = coerce_state_dict(_torch_load(torch, Path(weights), device))
    try:
        net.load_state_dict(state, strict=True)
    except (RuntimeError, ValueError) as exc:
        raise ParsingSetupError(
            f"Could not load {weights} into BiSeNet(n_classes={N_CLASSES}). "
            "Use the CelebAMask-HQ checkpoint 79999_iter.pth from "
            f"face-parsing.PyTorch.\n{exc}"
        ) from exc
    net.to(device)
    net.eval()
    return net


def read_rgb(path: Path):
    """Open ``path`` as an RGB Pillow image and its ``(H, W)`` size.

    Raises:
        ParsingSetupError: the file is not a readable image, or Pillow is missing.
    """
    image_cls = require_pillow()
    try:
        with image_cls.open(path) as img:
            rgb = img.convert("RGB")
            rgb.load()
            rgb = rgb.copy()
    except ParsingSetupError:
        raise
    except Exception as exc:
        raise ParsingSetupError(f"Could not read input image {path} as RGB ({exc}).") from exc
    width, height = rgb.size
    if width <= 0 or height <= 0:
        raise ParsingSetupError(f"input image has empty size: {path}")
    return rgb, (int(height), int(width))


def network_batch(image, device):
    """Match face-parsing.PyTorch ``test.py``: 512 bilinear, ImageNet norm.

    Returns a float tensor of shape ``(1, 3, 512, 512)`` on ``device``.
    """
    torch = require_torch()
    try:
        import torchvision.transforms as transforms
    except ImportError as exc:
        raise ParsingSetupError(
            "torchvision is required to match face-parsing.PyTorch preprocessing "
            "(pip install torchvision). A CPU wheel is enough."
        ) from exc
    image_cls = require_pillow()
    resized = image.resize((INPUT_SIZE, INPUT_SIZE), image_cls.BILINEAR)
    to_tensor = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
    return to_tensor(resized).unsqueeze(0).to(device)


def check_crop_margin(margin: float) -> float:
    """Return ``margin`` when it is a finite fraction in ``[0, 2]``.

    The value is the fraction of the detected box added on each side before
    the crop is optionally squared.

    Raises:
        ParsingSetupError: ``margin`` is not a finite number in that range.
    """
    if isinstance(margin, bool) or not isinstance(margin, (int, float)):
        raise ParsingSetupError(f"crop margin must be a number in [0, 2], got {margin!r}")
    value = float(margin)
    if not math.isfinite(value) or value < 0.0 or value > 2.0:
        raise ParsingSetupError(f"crop margin must be a number in [0, 2], got {margin}")
    return value


def insightface_detector_ctx_id(force_cpu: bool) -> int:
    """Return InsightFace ``ctx_id``: ``-1`` for CPU, ``0`` when CUDA is already up.

    ``prepare(ctx_id=-1)`` selects ``CPUExecutionProvider``. ``ctx_id=0``
    leaves the providers InsightFace picked, which is CUDA when that
    build of onnxruntime can see a GPU. ``--cpu`` forces ``-1``.
    """
    if force_cpu:
        return -1
    torch = require_torch()
    if torch.cuda.is_available():
        return 0
    return -1


def load_insightface_detector(force_cpu: bool):
    """Load AEGIS's InsightFace ``det_10g.onnx`` without downloading it.

    The model is ``insightface.model_zoo.get_model`` on
    ``~/.insightface/models/buffalo_l/det_10g.onnx``. That class is the
    RetinaFace/SCRFD detector. ``insightface.utils.ensure_available`` is
    not called.

    Raises:
        ParsingSetupError: InsightFace, onnxruntime, OpenCV, or the onnx
            file is missing, or ``prepare`` failed. The message names
            ``pip install insightface onnxruntime``.
    """
    try:
        import insightface
    except ImportError as exc:
        raise ParsingSetupError(
            "InsightFace is not installed, so the face cannot be cropped before BiSeNet.\n"
            + DETECTOR_INSTALL
        ) from exc
    path = insightface_detector_path()
    if not path.is_file():
        raise ParsingSetupError(
            f"InsightFace detector not found: {path}\n" + DETECTOR_INSTALL
        )
    try:
        model = insightface.model_zoo.get_model(str(path))
    except Exception as exc:
        raise ParsingSetupError(
            f"Could not load InsightFace detector {path} ({exc}).\n" + DETECTOR_INSTALL
        ) from exc
    if model is None:
        raise ParsingSetupError(
            f"insightface.model_zoo.get_model returned None for {path}.\n" + DETECTOR_INSTALL
        )
    ctx_id = insightface_detector_ctx_id(force_cpu)
    try:
        model.prepare(ctx_id=ctx_id, det_size=DET_INPUT_SIZE, input_size=DET_INPUT_SIZE)
    except Exception as exc:
        raise ParsingSetupError(
            f"Could not prepare InsightFace detector (ctx_id={ctx_id}) ({exc}).\n"
            + DETECTOR_INSTALL
        ) from exc
    return model


def bgr_uint8(image):
    """Return a contiguous BGR ``uint8`` array of an RGB image.

    InsightFace ``RetinaFace.detect`` feeds ``cv2.dnn.blobFromImage`` with
    ``swapRB=True``, which assumes OpenCV BGR. Pillow and the NeRSemble
    render are RGB. Float arrays whose maximum is at most 1 are scaled by
    255. An alpha channel is dropped.

    Raises:
        ParsingSetupError: ``image`` is not a 3-channel array.
        ImportError: NumPy is not installed.
    """
    np = require_numpy()
    array = np.asarray(image)
    if array.ndim == 3 and array.shape[2] >= 3:
        rgb = array[..., :3]
    else:
        raise ParsingSetupError(
            "face detector input must be an RGB image of shape (H, W, 3), "
            f"got shape {array.shape}"
        )
    if rgb.dtype != np.uint8:
        values = np.asarray(rgb, dtype=np.float64)
        if values.size and not np.all(np.isfinite(values)):
            raise ParsingSetupError("face detector input has non-finite pixels")
        if values.size and float(np.min(values)) >= 0.0 and float(np.max(values)) <= 1.0:
            values = values * 255.0
        rgb = np.clip(np.rint(values), 0, 255).astype(np.uint8)
    return np.ascontiguousarray(rgb[..., ::-1])


def select_largest_face(bboxes) -> FaceBox:
    """Pick the face with the largest area, then the higher score.

    ``bboxes`` is ``(N, 4)`` or ``(N, 5)`` as returned by InsightFace
    ``detect`` (``x0, y0, x1, y1`` and an optional score). An empty array
    means no face.

    Raises:
        ParsingRejectedError: there is no usable face box.
        ParsingSetupError: the array is not a box table.
        ImportError: NumPy is not installed.
    """
    np = require_numpy()
    if bboxes is None:
        raise ParsingRejectedError(_no_face_message())
    arr = np.asarray(bboxes, dtype=np.float64)
    if arr.size == 0:
        raise ParsingRejectedError(_no_face_message())
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    if arr.ndim != 2 or arr.shape[1] < 4:
        raise ParsingSetupError(
            f"InsightFace bboxes have shape {tuple(arr.shape)}, expected (N, 4) or (N, 5)"
        )
    best: Optional[FaceBox] = None
    best_key: Optional[Tuple[float, float]] = None
    for row in arr:
        x0, y0, x1, y1 = (float(row[0]), float(row[1]), float(row[2]), float(row[3]))
        if not all(math.isfinite(value) for value in (x0, y0, x1, y1)):
            continue
        area = (x1 - x0) * (y1 - y0)
        if area <= 0.0:
            continue
        score = float(row[4]) if row.shape[0] >= 5 and math.isfinite(float(row[4])) else 0.0
        key = (area, score)
        if best_key is None or key > best_key:
            best_key = key
            best = FaceBox(x0, y0, x1, y1, score)
    if best is None:
        raise ParsingRejectedError(_no_face_message())
    return best


def detect_largest_face(detector, image) -> FaceBox:
    """Run ``detector.detect`` on BGR pixels and return the largest face.

    Raises:
        ParsingRejectedError: the detector found no face. BiSeNet is not run.
        ParsingSetupError: ``detect`` failed or returned an unexpected layout.
        ImportError: NumPy is not installed.
    """
    bgr = bgr_uint8(image)
    try:
        detected = detector.detect(bgr, max_num=0)
    except (ParsingSetupError, ParsingRejectedError):
        raise
    except Exception as exc:
        raise ParsingSetupError(
            f"InsightFace detect failed ({exc}). "
            "Pass --cpu if the CUDA provider is missing. "
            "pip install insightface onnxruntime. "
            "This script does not download det_10g.onnx and does not start GPU masking."
        ) from exc
    if isinstance(detected, tuple):
        if not detected:
            raise ParsingSetupError("InsightFace detect returned an empty tuple.")
        bboxes = detected[0]
    else:
        bboxes = detected
    return select_largest_face(bboxes)


def face_crop_box(
    bbox: Sequence[float],
    image_hw: Tuple[int, int],
    *,
    margin: float = DEFAULT_CROP_MARGIN,
    square: bool = True,
) -> CropBox:
    """Expand ``bbox`` by ``margin``, optionally square it, and clip to the image.

    ``bbox`` is ``(x0, y0, x1, y1)`` in pixels. ``margin`` is the fraction of
    the box width added to the left and right, and of the box height added
    to the top and bottom. When ``square`` is true the shorter side grows
    about the center until it matches the longer side. The result is floored
    at the top-left and ceiled at the bottom-right, then clipped to
    ``image_hw``. Clipping can make a border crop non-square.

    Raises:
        ParsingSetupError: the image size or margin is unusable.
        ParsingRejectedError: the box is empty, non-finite, or outside the image.
        ImportError: NumPy is not installed.
    """
    np = require_numpy()
    margin = check_crop_margin(margin)
    height, width = int(image_hw[0]), int(image_hw[1])
    if height <= 0 or width <= 0:
        raise ParsingSetupError(f"image size must be positive, got {(height, width)}")
    if len(bbox) < 4:
        raise ParsingRejectedError(f"face box needs 4 values, got {tuple(bbox)}")
    x0, y0, x1, y1 = (float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3]))
    if not all(math.isfinite(value) for value in (x0, y0, x1, y1)):
        raise ParsingRejectedError("face box has non-finite coordinates")
    if x1 <= x0 or y1 <= y0:
        raise ParsingRejectedError(
            f"face box is empty: x0={x0} y0={y0} x1={x1} y1={y1}"
        )
    if x1 <= 0.0 or y1 <= 0.0 or x0 >= width or y0 >= height:
        raise ParsingRejectedError(
            "face box does not overlap the image "
            f"(box x0={x0} y0={y0} x1={x1} y1={y1}, image {(height, width)})"
        )
    box_w = x1 - x0
    box_h = y1 - y0
    x0 -= margin * box_w
    x1 += margin * box_w
    y0 -= margin * box_h
    y1 += margin * box_h
    if square:
        side = max(x1 - x0, y1 - y0)
        cx = (x0 + x1) / 2.0
        cy = (y0 + y1) / 2.0
        x0 = cx - side / 2.0
        x1 = cx + side / 2.0
        y0 = cy - side / 2.0
        y1 = cy + side / 2.0
    ix0 = int(np.floor(x0))
    iy0 = int(np.floor(y0))
    ix1 = int(np.ceil(x1))
    iy1 = int(np.ceil(y1))
    ix0 = max(0, min(ix0, width - 1))
    iy0 = max(0, min(iy0, height - 1))
    ix1 = max(ix0 + 1, min(ix1, width))
    iy1 = max(iy0 + 1, min(iy1, height))
    return CropBox(iy0, iy1, ix0, ix1)


def paste_crop_labels(crop_labels, canvas_hw: Tuple[int, int], origin_yx: Tuple[int, int]):
    """Paste crop labels onto a zero canvas of shape ``canvas_hw``.

    ``origin_yx`` is ``(y0, x0)`` in the original image. Pixels outside the
    canvas are dropped. Every other pixel stays background id ``0``, so the
    result is an ``(H, W)`` map aligned with the render BiSeNet did not see.

    Returns:
        ``int64`` array of shape ``canvas_hw``.

    Raises:
        ParsingSetupError: the crop misses the canvas, or the labels are not
            a 19-class grid.
        ImportError: NumPy is not installed.
    """
    np = require_numpy()
    crop = finalize_labels(crop_labels)
    height, width = int(canvas_hw[0]), int(canvas_hw[1])
    if height <= 0 or width <= 0:
        raise ParsingSetupError(f"canvas size must be positive, got {(height, width)}")
    y0, x0 = int(origin_yx[0]), int(origin_yx[1])
    src_y0 = 0
    src_x0 = 0
    if y0 < 0:
        src_y0 = -y0
        y0 = 0
    if x0 < 0:
        src_x0 = -x0
        x0 = 0
    if y0 >= height or x0 >= width or src_y0 >= crop.shape[0] or src_x0 >= crop.shape[1]:
        raise ParsingSetupError(
            "face-crop labels do not overlap the original image "
            f"(origin {origin_yx}, crop {tuple(crop.shape)}, canvas {(height, width)})"
        )
    src_y1 = int(crop.shape[0])
    src_x1 = int(crop.shape[1])
    dst_y1 = y0 + (src_y1 - src_y0)
    dst_x1 = x0 + (src_x1 - src_x0)
    if dst_y1 > height:
        src_y1 -= dst_y1 - height
        dst_y1 = height
    if dst_x1 > width:
        src_x1 -= dst_x1 - width
        dst_x1 = width
    if src_y0 >= src_y1 or src_x0 >= src_x1:
        raise ParsingSetupError("face-crop labels do not overlap the original image")
    canvas = np.zeros((height, width), dtype=np.int64)
    canvas[y0:dst_y1, x0:dst_x1] = crop[src_y0:src_y1, src_x0:src_x1]
    return canvas


def labels_from_crop(crop_labels, canvas_hw: Tuple[int, int], box: CropBox):
    """Paste ``crop_labels`` when their shape matches ``box``.

    Raises:
        ParsingSetupError: the label grid is not exactly the crop size.
        ImportError: NumPy is not installed.
    """
    np = require_numpy()
    array = np.asarray(crop_labels)
    expected = box.shape
    if tuple(int(dim) for dim in array.shape) != expected:
        raise ParsingSetupError(
            f"crop labels are {tuple(int(dim) for dim in array.shape)}, expected {expected} "
            "so they can be pasted back onto the original image"
        )
    return paste_crop_labels(array, canvas_hw, (box.y0, box.x0))


def parse_rgb_image(net, image, device):
    """Run BiSeNet on one RGB image and return labels at that image's size.

    The network sees a 512×512 bilinear resize, as in face-parsing.PyTorch
    ``test.py`` (``out = net(img)[0]``, then class argmax). Labels are then
    nearest-resized back to the image ``H×W``. For a face crop, that ``H×W``
    is the crop, not the original render.

    Raises:
        ParsingSetupError: the image or the BiSeNet output is unusable.
    """
    torch = require_torch()
    width, height = image.size
    out_hw = (int(height), int(width))
    if out_hw[0] <= 0 or out_hw[1] <= 0:
        raise ParsingSetupError(f"image to parse is empty: {out_hw}")
    batch = network_batch(image, device)
    with torch.no_grad():
        raw = net(batch)
    if not isinstance(raw, (tuple, list)) or len(raw) < 1:
        raise ParsingSetupError(
            "BiSeNet.forward should return (full, out16, out32) as in "
            "face-parsing.PyTorch model.py."
        )
    return logits_to_label_map(raw[0], out_hw, expect_hw=(INPUT_SIZE, INPUT_SIZE))


def produce_label_map(
    net,
    image_path: Path,
    device,
    detector,
    *,
    margin: float = DEFAULT_CROP_MARGIN,
    square: bool = True,
) -> CropParse:
    """Detect a face, parse the crop, and paste labels onto the full image.

    The returned grid has the original ``(H, W)``. Pixels outside the crop
    are background ``0``. ``detector`` is an InsightFace model with
    ``detect(bgr, max_num=0)``.

    Raises:
        ParsingSetupError: the image, the crop, or BiSeNet is unusable.
        ParsingRejectedError: no face was detected.
    """
    margin = check_crop_margin(margin)
    rgb, full_hw = read_rgb(image_path)
    face = detect_largest_face(detector, rgb)
    box = face_crop_box(
        (face.x0, face.y0, face.x1, face.y1),
        full_hw,
        margin=margin,
        square=square,
    )
    crop = rgb.crop((box.x0, box.y0, box.x1, box.y1))
    crop_labels = parse_rgb_image(net, crop, device)
    labels = labels_from_crop(crop_labels, full_hw, box)
    return CropParse(labels, box, float(face.score))


def make_nersemble306_parsing(
    *,
    image: Optional[Path] = None,
    out: Optional[Path] = None,
    repo: Optional[Path] = None,
    weights: Optional[Path] = None,
    force_cpu: bool = False,
    repo_root: Path = REPO_ROOT,
    margin: float = DEFAULT_CROP_MARGIN,
    square: bool = True,
) -> ParsingResult:
    """Detect, crop, parse, and write the NeRSemble 306 label ``.npy``.

    Default output is ``third_party/AEGIS/datasets/NeRSembleReconst/parsing/306.npy``.
    The alternate sidecar is ``avatars/306/parsing.npy`` under that same demo
    tree; pass it as ``out`` to write there instead. This does not start
    AEGIS masking and does not download the demo, the checkout, the BiSeNet
    checkpoint, or ``det_10g.onnx``.

    After the file is written, lip, eye, brow, and cheek must each have a
    pixel count greater than 0 under ``SemanticMask.from_parsing``. Otherwise
    :class:`ParsingRejectedError` is raised and the process should exit 1.

    Raises:
        ParsingSetupError: image, checkout, weights, InsightFace, or another
            runtime dependency is missing, or BiSeNet did not return 19-class
            labels.
        ParsingRejectedError: no face was found, or a required region is empty
            after the map is written.
        ImportError: NumPy is not installed.
    """
    root = Path(repo_root)
    image_path = resolve_input_image(image, root)
    face_repo = resolve_face_parsing_repo(repo, root)
    weight_path = resolve_weights(weights, face_repo)
    margin = check_crop_margin(margin)
    primary, _alternate = default_output_paths(root)
    out_path = Path(out) if out is not None else primary
    detector = load_insightface_detector(force_cpu)
    device = select_device(force_cpu)
    net = load_bisenet(face_repo, weight_path, device)
    parsed = produce_label_map(
        net,
        image_path,
        device,
        detector,
        margin=margin,
        square=square,
    )
    labels = parsed.labels
    save_label_map(out_path, labels)
    np = require_numpy()
    stored = np.load(out_path, allow_pickle=False)
    try:
        counts = require_nonzero_makeup_regions(stored, path=out_path)
    except ValueError as exc:
        raise ParsingSetupError(
            f"Written label map {out_path} is not a parsing grid SemanticMask.from_parsing can read ({exc})."
        ) from exc
    regions = region_presence(stored)
    summary = format_summary(
        path=out_path,
        labels=stored,
        image=image_path,
        device=device,
        regions=regions,
        counts=counts,
        crop=parsed.box,
        margin=margin,
        square=square,
        score=parsed.score,
    )
    return ParsingResult(out_path, stored, summary, device, regions, counts, parsed.box)


def _no_face_message() -> str:
    return (
        "InsightFace buffalo_l (det_10g.onnx) found no face in the input image. "
        "BiSeNet was not run. Pass a frontal render of NeRSemble subject 306 "
        "(renders/306.png). This script does not download weights and does not "
        "start GPU masking."
    )

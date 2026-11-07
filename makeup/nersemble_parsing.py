"""One-image BiSeNet parse for NeRSemble subject 306.

The NeRSemble demo ships ``renders/306.png`` and no label map. This module
runs the CelebAMask-HQ BiSeNet from face-parsing.PyTorch and writes an
``(H, W)`` integer ``.npy`` whose ids are ``DEFAULT_PARSING_LABELS``.

Torch, torchvision, and Pillow are imported only when a parse runs. The
face-parsing checkout and ``79999_iter.pth`` are not downloaded. CUDA is
used only when it is already available. Nothing here starts AEGIS masking.
"""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path
from typing import NamedTuple, Optional, Tuple

from makeup._numpy import require_numpy
from makeup.aegis_mask import parsing_sidecar_paths
from makeup.mask import DEFAULT_PARSING_LABELS, REGIONS

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

REPO_ROOT = Path(__file__).resolve().parents[1]


class ParsingSetupError(RuntimeError):
    """The 306 parse cannot run because a local input or dependency is missing."""


class ParsingResult(NamedTuple):
    """Label map written for subject 306."""

    path: Path
    labels: object
    summary: str
    device: object
    regions: dict


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


def format_summary(*, path: Path, labels, image: Path, device, regions: dict) -> str:
    """Short report: shape, unique ids, and which default regions appear."""
    np = require_numpy()
    array = np.asarray(labels)
    unique = " ".join(str(int(value)) for value in np.unique(array))
    flags = " ".join(
        f"{name}={'yes' if regions.get(name) else 'no'}"
        for name in ("lip", "eye", "brow", "cheek")
    )
    device_text = getattr(device, "type", device)
    return "\n".join(
        (
            f"wrote: {path}",
            f"shape: {tuple(int(value) for value in array.shape)}",
            f"dtype: {array.dtype}",
            f"unique_ids: {unique}",
            f"DEFAULT_PARSING_LABELS: {flags}",
            f"image: {image}",
            f"device: {device_text}",
        )
    )


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


def produce_label_map(net, image_path: Path, device):
    """Run BiSeNet on one RGB image and return an ``int64`` ``(H, W)`` map.

    The network sees a 512×512 bilinear resize, as in face-parsing.PyTorch
    ``test.py`` (``out = net(img)[0]``, then class argmax). Labels are then
    nearest-resized back to the source ``H×W`` so they align with the render
    passed to ``SemanticMask.from_parsing``.

    Raises:
        ParsingSetupError: the image or the BiSeNet output is unusable.
    """
    torch = require_torch()
    rgb, out_hw = read_rgb(image_path)
    batch = network_batch(rgb, device)
    with torch.no_grad():
        raw = net(batch)
    if not isinstance(raw, (tuple, list)) or len(raw) < 1:
        raise ParsingSetupError(
            "BiSeNet.forward should return (full, out16, out32) as in "
            "face-parsing.PyTorch model.py."
        )
    return logits_to_label_map(raw[0], out_hw, expect_hw=(INPUT_SIZE, INPUT_SIZE))


def make_nersemble306_parsing(
    *,
    image: Optional[Path] = None,
    out: Optional[Path] = None,
    repo: Optional[Path] = None,
    weights: Optional[Path] = None,
    force_cpu: bool = False,
    repo_root: Path = REPO_ROOT,
) -> ParsingResult:
    """Parse one NeRSemble 306 image and write the label ``.npy``.

    Default output is ``third_party/AEGIS/datasets/NeRSembleReconst/parsing/306.npy``.
    The alternate sidecar is ``avatars/306/parsing.npy`` under that same demo
    tree; pass it as ``out`` to write there instead. This does not start
    AEGIS masking and does not download the demo, the checkout, or weights.

    Raises:
        ParsingSetupError: image, checkout, weights, or a runtime dependency
            is missing, or BiSeNet did not return 19-class labels.
        ImportError: NumPy is not installed.
    """
    root = Path(repo_root)
    image_path = resolve_input_image(image, root)
    face_repo = resolve_face_parsing_repo(repo, root)
    weight_path = resolve_weights(weights, face_repo)
    primary, _alternate = default_output_paths(root)
    out_path = Path(out) if out is not None else primary
    device = select_device(force_cpu)
    net = load_bisenet(face_repo, weight_path, device)
    labels = produce_label_map(net, image_path, device)
    save_label_map(out_path, labels)
    regions = region_presence(labels)
    summary = format_summary(
        path=out_path,
        labels=labels,
        image=image_path,
        device=device,
        regions=regions,
    )
    return ParsingResult(out_path, labels, summary, device, regions)

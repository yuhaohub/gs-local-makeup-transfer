"""CPU overlay of a NeRSemble 306 face-parsing map on its render.

Reads an integer ``(H, W)`` label grid and an RGB image, builds ``M`` with
``SemanticMask.from_parsing`` and ``DEFAULT_PARSING_LABELS``, and writes one
RGB PNG. Lip, eye, brow, and cheek are tinted with the legend below. Pixels
outside ``M`` stay the original image. No network, no BiSeNet, and no GPU.

Legend (blend ``(1 - alpha) * image + alpha * color``, default alpha 0.55).
Later regions replace earlier ones where a pixel is claimed twice. Default
ids do not overlap.

| region | parser ids | RGB | hex |
| --- | --- | --- | --- |
| cheek | 1 skin | 240, 180, 40 | ``#F0B428`` |
| brow | 2 l_brow, 3 r_brow | 50, 110, 230 | ``#326EE6`` |
| eye | 4 l_eye, 5 r_eye | 40, 200, 80 | ``#28C850`` |
| lip | 12 u_lip, 13 l_lip | 220, 40, 60 | ``#DC283C`` |

Exit codes from :func:`build_parsing_report`: ``0`` when every required
region has at least one pixel, ``1`` when lip, eye, brow, or cheek is empty.
A missing left/right id (for example brow ``3`` or eye ``5``) is printed as
``INCOMPLETE`` and does not by itself change the exit code. Setup failures
raise :class:`ParsingVizError`; the command-line wrapper turns those into
exit code ``2``.
"""

from __future__ import annotations

import struct
import zlib
from pathlib import Path
from typing import Mapping, NamedTuple, Sequence, Tuple

from makeup._numpy import require_numpy
from makeup.mask import DEFAULT_PARSING_LABELS, REGIONS, MakeupRegion, SemanticMask

DEFAULT_ALPHA = 0.55
EXIT_OK = 0
EXIT_EMPTY = 1

DEFAULT_PARSING_REL = Path(
    "third_party/AEGIS/datasets/NeRSembleReconst/parsing/306.npy"
)
DEFAULT_IMAGE_REL = Path(
    "third_party/AEGIS/datasets/NeRSembleReconst/renders/306.png"
)
DEFAULT_OVERLAY_REL = Path("results/nersemble306_parsing_overlay.png")

# face-parsing.PyTorch CelebAMask-HQ names. Index is the class id (0 background).
PARSER_NAMES: Mapping[int, str] = {
    0: "background",
    1: "skin",
    2: "l_brow",
    3: "r_brow",
    4: "l_eye",
    5: "r_eye",
    6: "eye_g",
    7: "l_ear",
    8: "r_ear",
    9: "ear_r",
    10: "nose",
    11: "mouth",
    12: "u_lip",
    13: "l_lip",
    14: "neck",
    15: "neck_l",
    16: "cloth",
    17: "hair",
    18: "hat",
}

# Paint cheek first so a smaller part drawn later stays visible on overlap.
PAINT_ORDER: Tuple[MakeupRegion, ...] = (
    MakeupRegion.CHEEK,
    MakeupRegion.BROW,
    MakeupRegion.EYE,
    MakeupRegion.LIP,
)

REGION_COLORS: Mapping[MakeupRegion, Tuple[int, int, int]] = {
    MakeupRegion.CHEEK: (240, 180, 40),
    MakeupRegion.BROW: (50, 110, 230),
    MakeupRegion.EYE: (40, 200, 80),
    MakeupRegion.LIP: (220, 40, 60),
}

_PNG_SIG = b"\x89PNG\r\n\x1a\n"
_CRITICAL_CHUNKS = {b"IHDR", b"PLTE", b"IDAT", b"IEND"}


class ParsingVizError(RuntimeError):
    """The overlay cannot be built because an input is missing or invalid."""


class ParsingReport(NamedTuple):
    """Printed quality check plus the overlay that was written."""

    text: str
    exit_code: int
    status: str
    overlay_path: Path
    shape: Tuple[int, int]
    counts: dict
    fractions: dict
    id_counts: Tuple[Tuple[int, int], ...]
    empty_regions: Tuple[str, ...]
    incomplete_regions: Tuple[str, ...]


def color_hex(rgb: Sequence[int]) -> str:
    """Return ``#RRGGBB`` for a 3-channel color."""
    if len(rgb) != 3:
        raise ParsingVizError(f"legend color must be RGB, got {rgb!r}")
    return "#{:02X}{:02X}{:02X}".format(*(int(channel) for channel in rgb))


def load_label_grid(path: Path):
    """Load an integer ``(H, W)`` label map. ``allow_pickle`` is off.

    Raises:
        ParsingVizError: the file is missing, not a ``.npy``, or not a
            non-empty 2-D grid.
        ImportError: NumPy is not installed.
    """
    np = require_numpy()
    file_path = Path(path)
    if file_path.suffix.lower() != ".npy":
        raise ParsingVizError(
            f"parsing map must be a .npy file, got {file_path.name}"
        )
    if not file_path.is_file():
        raise ParsingVizError(
            f"Parsing label map not found: {file_path}\n"
            "On AutoDL the real map is "
            "third_party/AEGIS/datasets/NeRSembleReconst/parsing/306.npy.\n"
            "This script does not run BiSeNet and does not download the demo."
        )
    try:
        labels = np.load(file_path, allow_pickle=False)
    except (OSError, ValueError) as exc:
        raise ParsingVizError(
            f"Could not read parsing label map {file_path}: {exc}"
        ) from exc
    array = np.asarray(labels)
    if array.ndim != 2 or array.size == 0:
        raise ParsingVizError(
            "parsing must be a non-empty (H, W) integer label grid, "
            f"got shape {array.shape} dtype {array.dtype}"
        )
    return array


def load_rgb_image(path: Path):
    """Load an 8-bit PNG as ``uint8`` ``(H, W, 3)``.

    Color types gray, gray+alpha, indexed, RGB, and RGBA are accepted.
    Alpha is dropped. Interlaced PNGs and bit depths other than 8 are not.

    Raises:
        ParsingVizError: the file is missing or not a supported PNG.
        ImportError: NumPy is not installed.
    """
    file_path = Path(path)
    if not file_path.is_file():
        raise ParsingVizError(
            f"Render not found: {file_path}\n"
            "On AutoDL the 306 render is "
            "third_party/AEGIS/datasets/NeRSembleReconst/renders/306.png.\n"
            "This script does not download the demo."
        )
    try:
        data = file_path.read_bytes()
    except OSError as exc:
        raise ParsingVizError(f"Could not read render {file_path}: {exc}") from exc
    try:
        return decode_png_rgb(data)
    except ParsingVizError as exc:
        raise ParsingVizError(f"{file_path}: {exc}") from exc


def write_png_rgb(path: Path, rgb) -> Path:
    """Write ``uint8`` ``(H, W, 3)`` as a non-interlaced 8-bit RGB PNG.

    Parent directories are created. Scanlines use PNG filter 0.

    Raises:
        ParsingVizError: ``rgb`` is not an ``(H, W, 3)`` array.
        ImportError: NumPy is not installed.
    """
    np = require_numpy()
    image = np.asarray(rgb)
    if image.ndim != 3 or image.shape[2] != 3 or image.shape[0] == 0 or image.shape[1] == 0:
        raise ParsingVizError(
            f"overlay must be a non-empty (H, W, 3) RGB image, got shape {image.shape}"
        )
    if image.dtype != np.uint8:
        image = np.clip(np.rint(np.asarray(image, dtype=np.float64)), 0, 255).astype(np.uint8)
    else:
        image = np.ascontiguousarray(image)
    height, width = int(image.shape[0]), int(image.shape[1])
    raw = bytearray()
    for y in range(height):
        raw.append(0)
        raw.extend(image[y].tobytes())
    payload = _pack_png(width, height, color_type=2, raw=bytes(raw))
    file_path = Path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    file_path.write_bytes(payload)
    return file_path


def decode_png_rgb(data: bytes):
    """Decode PNG bytes to ``uint8`` ``(H, W, 3)``.

    Raises:
        ParsingVizError: ``data`` is not a supported 8-bit non-interlaced PNG.
        ImportError: NumPy is not installed.
    """
    np = require_numpy()
    if not isinstance(data, (bytes, bytearray)) or not bytes(data).startswith(_PNG_SIG):
        raise ParsingVizError("not a PNG")
    chunks = _read_png_chunks(bytes(data))
    if not chunks or chunks[0][0] != b"IHDR":
        raise ParsingVizError("PNG missing IHDR")
    try:
        width, height, bit_depth, color_type, compression, png_filter, interlace = struct.unpack(
            ">IIBBBBB", chunks[0][1]
        )
    except struct.error as exc:
        raise ParsingVizError(f"PNG IHDR is invalid: {exc}") from exc
    if compression != 0 or png_filter != 0:
        raise ParsingVizError("unsupported PNG compression or filter method")
    if interlace != 0:
        raise ParsingVizError(
            "interlaced PNG is not supported; re-save as a non-interlaced 8-bit RGB PNG"
        )
    if bit_depth != 8:
        raise ParsingVizError(
            f"only 8-bit PNG is supported, got bit depth {bit_depth}"
        )
    channels = _png_channels(color_type)
    palette = None
    idat = []
    saw_end = False
    for tag, chunk in chunks[1:]:
        if tag == b"IDAT":
            idat.append(chunk)
        elif tag == b"PLTE":
            palette = chunk
        elif tag == b"IEND":
            saw_end = True
            break
        elif tag not in _CRITICAL_CHUNKS and not _is_ancillary(tag):
            raise ParsingVizError(f"unsupported critical PNG chunk {tag!r}")
    if not saw_end:
        raise ParsingVizError("PNG missing IEND")
    if not idat:
        raise ParsingVizError("PNG missing IDAT")
    if color_type == 3 and not palette:
        raise ParsingVizError("indexed PNG is missing PLTE")
    try:
        inflated = zlib.decompress(b"".join(idat))
    except zlib.error as exc:
        raise ParsingVizError(f"PNG pixel data could not be inflated: {exc}") from exc
    planes = _unfilter(inflated, int(height), int(width), channels)
    if color_type == 2:
        rgb = planes
    elif color_type == 6:
        rgb = planes[..., :3]
    elif color_type == 0:
        rgb = np.repeat(planes[..., :1], 3, axis=2)
    elif color_type == 4:
        rgb = np.repeat(planes[..., :1], 3, axis=2)
    else:
        pal = np.frombuffer(palette, dtype=np.uint8)
        if pal.size % 3 != 0 or pal.size == 0:
            raise ParsingVizError("PLTE length is not a list of RGB triples")
        pal = pal.reshape(-1, 3)
        index = planes[..., 0]
        if int(index.max()) >= pal.shape[0]:
            raise ParsingVizError("indexed PNG uses a color outside PLTE")
        rgb = np.ascontiguousarray(pal[index])
    return np.array(rgb, dtype=np.uint8, copy=True)


def region_pixel_counts(mask: SemanticMask) -> dict:
    """Count strictly positive pixels in each required region and in ``M``.

    Counts come from ``SemanticMask.region`` and ``SemanticMask.union``.
    Keys are ``lip``, ``eye``, ``brow``, ``cheek``, and ``union``.

    Raises:
        KeyError: a required region was not stored.
        ImportError: NumPy is not installed.
    """
    np = require_numpy()
    counts = {}
    for region in REGIONS:
        counts[region.value] = int(np.count_nonzero(np.asarray(mask.region(region)) > 0))
    counts["union"] = int(np.count_nonzero(np.asarray(mask.union()) > 0))
    return counts


def raw_id_counts(labels) -> Tuple[Tuple[int, int], ...]:
    """Return ``(id, count)`` for every id present, sorted by id.

    Integral floats are rounded. The caller has already checked that the
    grid is a parsing map.

    Raises:
        ParsingVizError: a value is not an integer label.
        ImportError: NumPy is not installed.
    """
    np = require_numpy()
    array = _as_int_labels(labels)
    ids, counts = np.unique(array, return_counts=True)
    return tuple((int(label_id), int(count)) for label_id, count in zip(ids, counts))


def compose_overlay(image, mask: SemanticMask, alpha: float = DEFAULT_ALPHA):
    """Tint ``M`` on an RGB image. Outside ``M`` the pixels are unchanged.

    ``image`` is ``uint8`` ``(H, W, 3)`` or a float array in ``[0, 1]`` or
    ``[0, 255]``. Each region is blended against the original image, in
    :data:`PAINT_ORDER`.

    Returns:
        ``uint8`` array of shape ``(H, W, 3)``.

    Raises:
        ParsingVizError: ``alpha`` is outside ``[0, 1]``, or the image is
            not RGB aligned with the mask.
        ImportError: NumPy is not installed.
    """
    np = require_numpy()
    if not np.isfinite(alpha) or float(alpha) < 0.0 or float(alpha) > 1.0:
        raise ParsingVizError(f"alpha must be in [0, 1], got {alpha}")
    base = _as_uint8_rgb(image)
    union_shape = tuple(int(dim) for dim in np.asarray(mask.union()).shape)
    if union_shape != tuple(base.shape[:2]):
        raise ParsingVizError(
            f"image shape {base.shape} does not align with mask shape {union_shape}"
        )
    canvas = base.astype(np.float64)
    blended = canvas.copy()
    weight = float(alpha)
    for region in PAINT_ORDER:
        hit = np.asarray(mask.region(region)) > 0
        if not np.any(hit):
            continue
        color = np.asarray(REGION_COLORS[region], dtype=np.float64)
        blended[hit] = (1.0 - weight) * canvas[hit] + weight * color
    return np.clip(np.rint(blended), 0, 255).astype(np.uint8)


def build_parsing_report(
    labels,
    image,
    out: Path,
    *,
    alpha: float = DEFAULT_ALPHA,
    parsing_name: str = "(array)",
    image_name: str = "(array)",
) -> ParsingReport:
    """Build ``M``, write the overlay, and format the quality report.

    ``labels`` is an integer ``(H, W)`` grid. ``image`` is only an alignment
    check inside ``SemanticMask.from_parsing``; the same pixels are the
    overlay background. The PNG is written even when a required region is
    empty, so the picture can be sent with the counts.

    Returns:
        Report whose ``exit_code`` is ``1`` when lip, eye, brow, or cheek
        has no pixels, and ``0`` otherwise.

    Raises:
        ParsingVizError: the grid, the image, or ``alpha`` is invalid.
        ImportError: NumPy is not installed.
    """
    np = require_numpy()
    grid = np.asarray(labels)
    if grid.ndim != 2 or grid.size == 0:
        raise ParsingVizError(
            "parsing must be a non-empty (H, W) integer label grid, "
            f"got shape {grid.shape} dtype {grid.dtype}"
        )
    rgb = _as_uint8_rgb(image)
    if tuple(rgb.shape[:2]) != tuple(grid.shape):
        raise ParsingVizError(
            f"image shape {rgb.shape} does not align with parsing shape {grid.shape}; "
            f"expected {(grid.shape[0], grid.shape[1], 3)}"
        )
    try:
        mask = SemanticMask.from_parsing(grid, image=rgb)
    except ValueError as exc:
        raise ParsingVizError(str(exc)) from exc
    integer = _as_int_labels(grid)
    counts = region_pixel_counts(mask)
    pixels = int(integer.shape[0]) * int(integer.shape[1])
    fractions = {name: counts[name] / pixels for name in counts}
    id_counts = raw_id_counts(integer)
    per_region = _per_region_ids(id_counts)
    empty = tuple(
        region.value
        for region in REGIONS
        if counts[region.value] == 0
    )
    incomplete = tuple(
        region.value
        for region in REGIONS
        if counts[region.value] > 0 and any(count == 0 for _, count in per_region[region.value])
    )
    if empty:
        status = "empty_region"
        exit_code = EXIT_EMPTY
    elif incomplete:
        status = "incomplete"
        exit_code = EXIT_OK
    else:
        status = "ok"
        exit_code = EXIT_OK
    overlay = compose_overlay(rgb, mask, alpha)
    overlay_path = write_png_rgb(Path(out), overlay)
    text = format_report(
        parsing_name=parsing_name,
        image_name=image_name,
        overlay_path=overlay_path,
        shape=(int(integer.shape[0]), int(integer.shape[1])),
        dtype=str(integer.dtype),
        pixels=pixels,
        alpha=float(alpha),
        id_counts=id_counts,
        per_region=per_region,
        counts=counts,
        fractions=fractions,
        empty_regions=empty,
        incomplete_regions=incomplete,
        status=status,
        exit_code=exit_code,
    )
    return ParsingReport(
        text=text,
        exit_code=exit_code,
        status=status,
        overlay_path=overlay_path,
        shape=(int(integer.shape[0]), int(integer.shape[1])),
        counts=counts,
        fractions=fractions,
        id_counts=id_counts,
        empty_regions=empty,
        incomplete_regions=incomplete,
    )


def visualize_parsing(
    parsing: Path,
    image: Path,
    out: Path,
    *,
    alpha: float = DEFAULT_ALPHA,
) -> ParsingReport:
    """Load ``parsing`` and ``image`` from disk and build the report.

    Raises:
        ParsingVizError: either file is missing or invalid.
        ImportError: NumPy is not installed.
    """
    labels = load_label_grid(parsing)
    rgb = load_rgb_image(image)
    return build_parsing_report(
        labels,
        rgb,
        out,
        alpha=alpha,
        parsing_name=str(parsing),
        image_name=str(image),
    )


def format_report(
    *,
    parsing_name: str,
    image_name: str,
    overlay_path: Path,
    shape: Tuple[int, int],
    dtype: str,
    pixels: int,
    alpha: float,
    id_counts: Sequence[Tuple[int, int]],
    per_region: Mapping[str, Sequence[Tuple[int, int]]],
    counts: Mapping[str, int],
    fractions: Mapping[str, float],
    empty_regions: Sequence[str],
    incomplete_regions: Sequence[str],
    status: str,
    exit_code: int,
) -> str:
    """Format counts, the legend, and empty-region flags as plain text."""
    lines = [
        "NeRSemble 306 parsing quality check",
        "built_with: SemanticMask.from_parsing",
        "label_map: DEFAULT_PARSING_LABELS",
        f"parsing: {parsing_name}",
        f"image: {image_name}",
        f"overlay: {overlay_path}",
        f"shape: ({shape[0]}, {shape[1]})",
        f"dtype: {dtype}",
        f"pixels: {pixels}",
        f"alpha: {alpha:.2f}",
        "raw_ids:",
    ]
    id_to_region = {
        int(label_id): region.value for label_id, region in DEFAULT_PARSING_LABELS.items()
    }
    for label_id, count in id_counts:
        name = PARSER_NAMES.get(int(label_id), "unknown")
        region = id_to_region.get(int(label_id), "-")
        fraction = (count / pixels) if pixels else 0.0
        lines.append(
            f"  {label_id} {name} count={count} fraction={fraction:.8f} region={region}"
        )
    missing = []
    for label_id in sorted(DEFAULT_PARSING_LABELS):
        count = dict(id_counts).get(int(label_id), 0)
        if count == 0:
            region = DEFAULT_PARSING_LABELS[label_id].value
            missing.append(f"{label_id} {PARSER_NAMES.get(label_id, 'unknown')} ({region})")
    lines.append(
        "missing_default_ids: " + (", ".join(missing) if missing else "none")
    )
    lines.append("regions:")
    for name in ("lip", "eye", "brow", "cheek"):
        id_text = ",".join(f"{label_id}:{count}" for label_id, count in per_region[name])
        if name in empty_regions:
            tag = "EMPTY"
        elif name in incomplete_regions:
            tag = "INCOMPLETE"
        else:
            tag = "ok"
        lines.append(
            f"  {name} count={counts[name]} fraction={fractions[name]:.8f} "
            f"ids={id_text} {tag}"
        )
    lines.append(
        f"  union count={counts['union']} fraction={fractions['union']:.8f}"
    )
    lines.append("legend:")
    for region in PAINT_ORDER:
        rgb = REGION_COLORS[region]
        lines.append(f"  {region.value} rgb({rgb[0]}, {rgb[1]}, {rgb[2]}) {color_hex(rgb)}")
    lines.append(
        "blend: (1-alpha)*image + alpha*color on that region; "
        "pixels outside M stay the original image"
    )
    lines.append("paint_order: cheek brow eye lip")
    for name in empty_regions:
        if name == "lip":
            lines.append("FLAG: lip count is 0")
        lines.append(f"FLAG: required region empty: {name}")
    for name in incomplete_regions:
        absent = [
            f"{label_id} {PARSER_NAMES.get(label_id, 'unknown')}"
            for label_id, count in per_region[name]
            if count == 0
        ]
        present = [
            f"{label_id} {PARSER_NAMES.get(label_id, 'unknown')}={count}"
            for label_id, count in per_region[name]
            if count > 0
        ]
        lines.append(
            f"INCOMPLETE: {name} missing {', '.join(absent)}; present {', '.join(present)}"
        )
    lines.append(f"exit_code: {exit_code}")
    lines.append(f"status: {status}")
    return "\n".join(lines)


def _as_uint8_rgb(image):
    """Return a copied ``uint8`` ``(H, W, 3)`` image."""
    np = require_numpy()
    array = np.asarray(image)
    if array.ndim == 3 and array.shape[2] == 4:
        array = array[..., :3]
    if array.ndim != 3 or array.shape[2] != 3 or array.shape[0] == 0 or array.shape[1] == 0:
        raise ParsingVizError(
            f"image must be non-empty RGB of shape (H, W, 3), got shape {array.shape}"
        )
    if array.dtype == np.uint8:
        return np.array(array, copy=True)
    values = np.asarray(array, dtype=np.float64)
    if not np.all(np.isfinite(values)):
        raise ParsingVizError("image has non-finite values")
    if values.size and float(np.min(values)) >= 0.0 and float(np.max(values)) <= 1.0:
        values = values * 255.0
    return np.clip(np.rint(values), 0, 255).astype(np.uint8)


def _as_int_labels(labels):
    """Return ``int64`` labels, rejecting non-integral values."""
    np = require_numpy()
    array = np.asarray(labels)
    if array.dtype.kind == "f":
        if array.size and (
            not np.all(np.isfinite(array)) or not np.all(array == np.rint(array))
        ):
            raise ParsingVizError("parsing labels must be integers")
        array = np.rint(array)
    elif array.dtype.kind not in "iu":
        raise ParsingVizError(
            f"parsing labels must be integers, got dtype {array.dtype}"
        )
    return np.asarray(array, dtype=np.int64)


def _per_region_ids(id_counts: Sequence[Tuple[int, int]]) -> dict:
    """Map each required region to ``(default id, count)`` pairs, id order."""
    found = {int(label_id): int(count) for label_id, count in id_counts}
    per_region = {region.value: [] for region in REGIONS}
    for label_id, region in DEFAULT_PARSING_LABELS.items():
        per_region[region.value].append((int(label_id), found.get(int(label_id), 0)))
    for name in per_region:
        per_region[name] = tuple(sorted(per_region[name]))
    return per_region


def _png_channels(color_type: int) -> int:
    if color_type == 0:
        return 1
    if color_type == 2:
        return 3
    if color_type == 3:
        return 1
    if color_type == 4:
        return 2
    if color_type == 6:
        return 4
    raise ParsingVizError(f"unsupported PNG color type {color_type}")


def _is_ancillary(tag: bytes) -> bool:
    return bool(tag) and (tag[0] & 32) != 0


def _chunk(tag: bytes, data: bytes) -> bytes:
    crc = zlib.crc32(tag + data) & 0xFFFFFFFF
    return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", crc)


def _pack_png(width: int, height: int, color_type: int, raw: bytes) -> bytes:
    ihdr = struct.pack(">IIBBBBB", int(width), int(height), 8, int(color_type), 0, 0, 0)
    return _PNG_SIG + _chunk(b"IHDR", ihdr) + _chunk(b"IDAT", zlib.compress(raw, 9)) + _chunk(b"IEND", b"")


def _read_png_chunks(data: bytes) -> list:
    offset = len(_PNG_SIG)
    chunks = []
    while offset + 12 <= len(data):
        length = struct.unpack_from(">I", data, offset)[0]
        end = offset + 12 + length
        if length < 0 or end > len(data):
            raise ParsingVizError("truncated PNG chunk")
        tag = data[offset + 4 : offset + 8]
        chunk = data[offset + 8 : offset + 8 + length]
        crc_got = struct.unpack_from(">I", data, offset + 8 + length)[0]
        crc_exp = zlib.crc32(tag + chunk) & 0xFFFFFFFF
        if crc_got != crc_exp:
            raise ParsingVizError(f"bad PNG CRC for {tag!r}")
        chunks.append((tag, chunk))
        offset = end
        if tag == b"IEND":
            break
    return chunks


def _paeth(left: int, up: int, upper_left: int) -> int:
    estimate = left + up - upper_left
    pa = abs(estimate - left)
    pb = abs(estimate - up)
    pc = abs(estimate - upper_left)
    if pa <= pb and pa <= pc:
        return left
    if pb <= pc:
        return up
    return upper_left


def _unfilter(raw: bytes, height: int, width: int, channels: int):
    """Undo PNG scanline filters. ``raw`` includes one filter byte per row."""
    np = require_numpy()
    if height <= 0 or width <= 0 or channels <= 0:
        raise ParsingVizError(
            f"PNG size must be positive, got {(height, width)} channels {channels}"
        )
    stride = width * channels
    expected = height * (stride + 1)
    if len(raw) != expected:
        raise ParsingVizError(
            f"PNG inflated size {len(raw)} does not match {height}x{width} "
            f"x{channels} (expected {expected} bytes)"
        )
    out = bytearray(height * stride)
    source = memoryview(raw)
    dest = memoryview(out)
    si = 0
    oi = 0
    for y in range(height):
        ftype = source[si]
        si += 1
        if ftype == 0:
            dest[oi : oi + stride] = source[si : si + stride]
            si += stride
            oi += stride
            continue
        if ftype not in (1, 2, 3, 4):
            raise ParsingVizError(f"unsupported PNG filter type {ftype}")
        for x in range(stride):
            val = source[si]
            si += 1
            left = dest[oi - channels] if x >= channels else 0
            up = dest[oi - stride] if y else 0
            upper_left = dest[oi - stride - channels] if y and x >= channels else 0
            if ftype == 1:
                pred = left
            elif ftype == 2:
                pred = up
            elif ftype == 3:
                pred = (left + up) // 2
            else:
                pred = _paeth(left, up, upper_left)
            dest[oi] = (val + pred) & 255
            oi += 1
    return np.frombuffer(bytes(out), dtype=np.uint8).reshape(height, width, channels).copy()

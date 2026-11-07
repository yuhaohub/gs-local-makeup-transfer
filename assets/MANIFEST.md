# Asset manifest

Maps frozen names and experiment ids to paths on the AutoDL asset root.
GPU launch waits on `docs/gpu_launch_checklist.md`. This file does not enable B2
and does not vendor dataset bytes.

## Root

Single asset root: `/root/autodl-fs/gs-assets/`.

`autodl-tmp` is cache only. It is not an asset root, a weight store, or a
summary location.

| Subdir | Holds |
| --- | --- |
| `weights/` | Checkpoints the runs already name (AdaFace stub below). |
| `avatars/` | NeRSemble subject 306 gaussian avatar. |
| `masks/` | Subject 306 parsing mask. |
| `targets/` | Frozen stills `celeba_T0.jpg` and `lfw_T0.jpg`. |
| `runs/<run_id>/` | Machine-side summary and log for that run only. |

Repo summaries are not on this root. One run writes
`results/<exp_id>/summary.json`, `results/<exp_id>/summary.csv`, and the log
beside them. The machine copy of that same run is
`/root/autodl-fs/gs-assets/runs/<run_id>/`. Never
overwrite a shared summary. Historical flat files already in `results/` stay
where they are; see the checklist for the occupied slots.

Workspace paths are symlinks into this root (see README「新机」). Git does not
track the image or weight bytes. `assets/targets/*` stays local except
`.gitkeep`.

`sha256`, `bytes`, and `frozen_at` are `TBD` until james measures the file on
the machine. `frozen_at` is the calendar date those bytes were frozen
(`YYYY-MM-DD`). Do not invent a second filename for a row that already has a
frozen name.

## Rows

| path | frozen name / exp id | kind | source/notes | owner | frozen_at | sha256 | bytes |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `/root/autodl-fs/gs-assets/targets/celeba_T0.jpg` | `celeba_T0` / B1 (B1-a, first smoke) | target | CelebA still. Frozen on-disk name. Default `--impersonation-image`. Repo does not vendor CelebA. Workspace symlink example: `assets/targets/celeba_T0.jpg` (gitignored). | james | TBD | TBD | TBD |
| `/root/autodl-fs/gs-assets/targets/lfw_T0.jpg` | `lfw_T0` / B1 (later still; own `results/<exp_id>/`) | target | LFW still. Frozen on-disk name. Not the first smoke. Do not reuse the celeba summary slot. Repo does not vendor LFW. | james | TBD | TBD | TBD |
| `/root/autodl-fs/gs-assets/avatars/306/point_cloud.ply` | `306` / B0-a, B1 | avatar | NeRSemble subject 306 gaussian avatar. README: `gs-assets/avatars` holds only 306. Filename is the documented `point_cloud.ply`. Command path: `third_party/AEGIS/datasets/NeRSembleReconst/avatars/306/point_cloud.ply`. Verification render on that same subject (not a `targets/` still): `third_party/AEGIS/datasets/NeRSembleReconst/renders/306.png`. FaceScape 527 is cancelled. | james | TBD | TBD | TBD |
| `/root/autodl-fs/gs-assets/masks/306.npy` | `306.npy` / `306-makeup-mask-smoke` (not B2) | mask | Parsing map for subject 306. Filename matches README `parsing/306.npy`. Workspace paths the scripts already accept: `third_party/AEGIS/datasets/NeRSembleReconst/parsing/306.npy` and `third_party/AEGIS/datasets/NeRSembleReconst/avatars/306/parsing.npy`. `documented_fallback_labels` is not this file. This row does not open B2. | james | TBD | TBD | TBD |
| `/root/autodl-fs/gs-assets/weights/adaface_ir50_ms1mv2.ckpt` | `adaface_ir50_ms1mv2.ckpt` / B0-a, B1 | weights | Stub. AdaFace IR-50 MS1MV2 checkpoint named in README. Workspace pointer: `third_party/AEGIS/models/adaface_ir50_ms1mv2.ckpt`. Do not re-download. james fills sha256, bytes, and frozen_at. | james | TBD | TBD | TBD |
| `third_party/face-parsing.PyTorch/res/cp/79999_iter.pth` | `79999_iter.pth` / 306 parsing | weights | Stub from README. CelebAMask-HQ BiSeNet (19 classes). Scripts do not download it. Not a `targets/` name. A copy already under `weights/` gets its own row; do not rename this file to add one. | james | TBD | TBD | TBD |
| `~/.insightface/models/buffalo_l/det_10g.onnx` | `det_10g.onnx` / B0-a, B1 | weights | Stub from README. InsightFace `buffalo_l` detector AEGIS already uses. Usually already on AutoDL. Scripts do not download it and do not call `insightface.utils.ensure_available`. | james | TBD | TBD | TBD |

## Blank row

Copy this row when a new frozen asset is registered. Basename of `path` is the
frozen name. The exp id must have a free `results/<exp_id>/` slot before
`--launch` (checklist gate 3).

| path | frozen name / exp id | kind | source/notes | owner | frozen_at | sha256 | bytes |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `/root/autodl-fs/gs-assets/<subdir>/<frozen_name>` | `<frozen_name>` / `<exp_id>` | `<kind>` | `<subdir>` is `weights`, `avatars`, `masks`, or `targets`. | james | TBD | TBD | TBD |

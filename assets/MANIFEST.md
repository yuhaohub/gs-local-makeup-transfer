# Asset manifest

Maps frozen names and experiment ids to paths on the AutoDL asset root.
GPU launch waits on `docs/gpu_launch_checklist.md`. This file does not enable B2
and does not vendor dataset bytes.

Filled from machine inventory on 2026-11-08 (SSH `connect.nmb1.seetacloud.com:32791`, GPU off).
On-disk copy also at `/root/autodl-fs/gs-assets/MANIFEST.md`.

## Root

Single asset root: `/root/autodl-fs/gs-assets/`.

`autodl-tmp` is cache only. It is not an asset root, a weight store, or a
summary location.

| Subdir | Holds |
| --- | --- |
| `weights/` | Checkpoints the runs already name (AdaFace + BiSeNet). |
| `avatars/` | NeRSemble subject 306 gaussian avatar. |
| `parsing/` | Subject 306 parsing mask (`306.npy`). (Template once said `masks/`; on-disk truth is `parsing/`.) |
| `insightface/` | `buffalo_l` detector onnx copy. |
| `targets/` | Frozen stills `celeba_T0.jpg` and `lfw_T0.jpg`. |
| `renders/` | Verification render `306.png`. |
| `runs/<run_id>/` | Machine-side summary and log for that run only (not yet populated). |

Repo summaries are not on this root. One run writes
`results/<exp_id>/summary.json`, `results/<exp_id>/summary.csv`, and the log
beside them. The machine copy of that same run is
`/root/autodl-fs/gs-assets/runs/<run_id>/`. Never
overwrite a shared summary. Historical flat files already in `results/` stay
where they are; see the checklist for the occupied slots.

Workspace paths are symlinks into this root (see README「新机」). Git does not
track the image or weight bytes. `assets/targets/*` stays local except
`.gitkeep`.

## Read-only public sources (not in gs-assets)

| Path | Notes |
| --- | --- |
| `/root/autodl-pub/CelebA/` | Local CelebA (`Img/img_align_celeba.zip`, 7z variants). Read-only source library. Do **not** copy the whole set into `gs-assets`. |
| `/root/autodl-pub/CASIAWebFace/CASIA-WebFace.zip` | Local CASIA-WebFace zip. Not in frozen matrix — do not touch until a frozen name is registered. |

`celeba_T0.jpg` was frozen from HF-mirror parquet (`huggan/CelebA-faces-with-attributes`, `000001.jpg`), not extracted from `autodl-pub` this round. Future T picks may provenance to `autodl-pub/CelebA/...`.

## Rows

| path | frozen name / exp id | kind | source/notes | owner | frozen_at | sha256 | bytes |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `/root/autodl-fs/gs-assets/targets/celeba_T0.jpg` | `celeba_T0` / B1 (B1-a, first smoke) | target | CelebA still `000001.jpg` via hf-mirror parquet `huggan/CelebA-faces-with-attributes` row 0. Alias on disk: `celeba_000001.jpg` (same sha). Default `--impersonation-image`. Provenance: `targets/SOURCES.txt`. Repo does not vendor CelebA. Local pub available at `/root/autodl-pub/CelebA/` for future picks. | james | 2026-11-07 | `26df6b870cb1ad4cbab9164bd8e7e5742fd14c73cdd1278f1813f8a54f70ddde` | 21247 |
| `/root/autodl-fs/gs-assets/targets/lfw_T0.jpg` | `lfw_T0` / B1 (later still; own `results/<exp_id>/`) | target | LFW `Alejandro_Toledo_0036.jpg` via hf-mirror parquet `sammyboi1801/lfw-face-transformer-dataset` row 0. Alias: `lfw_Alejandro_Toledo_0036.jpg` (same sha). Not the first smoke. Do not reuse the celeba summary slot. | james | 2026-11-07 | `452d5000acf19243e3eeac7516068d91d5d220f53723c5a3a35d29e2d7f73955` | 17411 |
| `/root/autodl-fs/gs-assets/avatars/306/point_cloud.ply` | `306` / B0-a, B1 | avatar | NeRSemble subject 306 gaussian avatar. Companion `flame_param.npz` (69653560 bytes, sha `68a9d732a035595ea9fe4b7e32c36c135fa273d499fa3355bb76a1ef73599ae1`) sits beside it. Verification render: `renders/306.png`. FaceScape 527 cancelled. | james | 2026-11-07 | `08586dcddc38bb50e5899c131f4f7985d3fa15da2bc93cd04771e9d4265f69f7` | 22434847 |
| `/root/autodl-fs/gs-assets/parsing/306.npy` | `306.npy` / `306-makeup-mask-smoke` (B2 reads this same file; this row does not open a GPU run) | mask | Parsing map for subject 306. On-disk subdir is `parsing/` (not `masks/`). Workspace symlink: `third_party/AEGIS/datasets/NeRSembleReconst/parsing/306.npy` → this file. B2 wiring uses this path. The row does not authorize `--launch`. | james | 2026-11-07 | `5dff94658927e402754f7dd7884446760188ce10c39247b0b303a7b409d1d29d` | 4147328 |
| `/root/autodl-fs/gs-assets/weights/adaface_ir50_ms1mv2.ckpt` | `adaface_ir50_ms1mv2.ckpt` / B0-a, B1 | weights | AdaFace IR-50 MS1MV2. Workspace symlink: `third_party/AEGIS/models/adaface_ir50_ms1mv2.ckpt` → this file. Do not re-download. | james | 2026-11-07 | `234da6ce931f821e0a8e920063dc2091d8122ea46285c159d2f25d3f762fecb3` | 700286703 |
| `/root/autodl-fs/gs-assets/weights/79999_iter.pth` | `79999_iter.pth` / 306 parsing | weights | CelebAMask-HQ BiSeNet (19 classes). Lives under `gs-assets/weights/` (not only under `third_party/face-parsing.PyTorch/res/cp/`). | james | 2026-11-07 | `468e13ca13a9b43cc0881a9f99083a430e9c0a38abd935431d1c28ee94b26567` | 53289463 |
| `/root/autodl-fs/gs-assets/insightface/det_10g.onnx` | `det_10g.onnx` / B0-a, B1 | weights | InsightFace `buffalo_l` detector. Also present at `~/.insightface/models/buffalo_l/det_10g.onnx` (same size). gs-assets copy is the migrated truth for this host. | james | 2026-11-07 | `5838f7fe053675b1c7a08b633df49e7af5495cee0493c7dcf6697200b85b5b91` | 16923827 |

## Blank row

Copy this row when a new frozen asset is registered. Basename of `path` is the
frozen name. The exp id must have a free `results/<exp_id>/` slot before
`--launch` (checklist gate 3).

| path | frozen name / exp id | kind | source/notes | owner | frozen_at | sha256 | bytes |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `/root/autodl-fs/gs-assets/<subdir>/<frozen_name>` | `<frozen_name>` / `<exp_id>` | `<kind>` | `<subdir>` is `weights`, `avatars`, `parsing`, or `targets`. | james | TBD | TBD | TBD |

## B2 stub (no new asset)

B2 does not register a new file and does not authorize `--launch`. It reuses frozen names already listed above. No new sha256 or byte count.

| role | path | summary slot |
| --- | --- | --- |
| source avatar | `/root/autodl-fs/gs-assets/avatars/306/point_cloud.ply` | same 306 avatar row |
| first T | `/root/autodl-fs/gs-assets/targets/celeba_T0.jpg` | `results/b2_celeba_T0/` |
| later T | `/root/autodl-fs/gs-assets/targets/lfw_T0.jpg` | `results/b2_lfw_T0/` |
| makeup mask | `/root/autodl-fs/gs-assets/parsing/306.npy` | same `306.npy` row. Not `masks/306.npy`. |
| embedder | `/root/autodl-fs/gs-assets/weights/adaface_ir50_ms1mv2.ckpt` | same AdaFace row |

Each still gets `summary.json` and `summary.csv` inside its own directory. Gate columns in that summary stay empty until a real run: `pipeline_ok`, `cos_adv_source`, `cos_adv_target`, `outside_m_abs_delta` (max `|δ|` outside M; the gate is 0). Flat historical summaries under `results/` stay where they are.

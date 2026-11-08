# GPU launch checklist

bob enforces this before any AutoDL GPU run. james adds `--launch` only after
bob greenlights. B2 wiring may merge. A GPU run, including B2, still requires
all four gates green: gates 1–3 first, then mentor authorization (gate 4).
This checklist does not change attack code, does not pass `--launch` by
default, and does not download datasets.

Research wording stays in `docs/创新章_交付文档_AEGIS底座.md`. Asset paths stay
in `assets/MANIFEST.md`.

## Where bytes live

- Asset root: `/root/autodl-fs/gs-assets/`.
- Subdirs: `weights/`, `avatars/`, `parsing/`, `targets/`.
- Machine-side per-run summary and log: `/root/autodl-fs/gs-assets/runs/<run_id>/`.
- Repo summary for that run: `results/<exp_id>/summary.json`,
  `results/<exp_id>/summary.csv`, and the log in the same directory.
- `autodl-tmp` is cache only.

One run, one summary. A new run gets a new `<exp_id>` and a new `<run_id>`.
It does not overwrite a shared summary.

Frozen names already on disk:

- `/root/autodl-fs/gs-assets/targets/celeba_T0.jpg` (B1-a, default)
- `/root/autodl-fs/gs-assets/targets/lfw_T0.jpg` (later B1 still)
- `/root/autodl-fs/gs-assets/parsing/306.npy` (subject 306 parsing mask)
- NeRSemble subject 306 avatar under `avatars/` (row in the manifest)

## Roles

| Role | Duty |
| --- | --- |
| james | Fills `assets/MANIFEST.md` (`sha256`, `bytes`, `frozen_at`) and self-checks gates 1–3 before asking bob. Passes `--launch` only after bob greenlights. |
| jack | Keeps PR defaults on the frozen names (`celeba_T0.jpg`, `lfw_T0.jpg`, subject 306). A PR does not introduce another target basename. |
| bob | Refuses the GPU when gate 1 (master), gate 2 (MANIFEST), or gate 3 (summary slot) fails. Greenlights only when all four gates are green. |
| mentor | Authorizes only after gates 1–3 are green. That authorization is gate 4. An earlier verbal go does not open the GPU. |

## Four gates

1. **Change is on master (PR merged).** The tree that will run is `master`, not an open feature branch.
2. **MANIFEST entry exists and paths match frozen names.** Every path the run reads has a row in `assets/MANIFEST.md`. The path uses the frozen basename (`celeba_T0.jpg`, `lfw_T0.jpg`, subject `306` avatar, `parsing/306.npy`, AdaFace `adaface_ir50_ms1mv2.ckpt`). A missing row or a substitute filename fails this gate.
3. **`results/<exp_id>/` summary path is free (no clobber).** That directory does not already contain `summary.json`, `summary.csv`, or the run log. Occupied historical summaries stay in place and are not the write target:
   - B0-a: `results/b0_aegis_sanity_summary.json` and `.csv` (`pipeline_ok`, subject 306)
   - B1 celeba wiring: `results/b1_targeted_impersonation_summary.json` and `.csv` (not launched)
   - 306 makeup-mask CPU smoke: `results/nersemble306_makeup_mask_smoke.json` and `.csv` (not B2)
   - B2 write targets are not those flat files. One still, one directory: `results/b2_celeba_T0/summary.json` and `summary.csv`, and later `results/b2_lfw_T0/`. A wiring record there has `pipeline_ok` false and null measurements. That record is not a finished attack and does not by itself clobber gate 3. The authorized run of that same still updates the same directory. The post-run gate is in the B2 section below. Wiring may merge; `--launch` still waits for gates 1–3 and then mentor gate 4.
4. **Mentor authorizes only after gates 1–3 are green.**

When all four are green, bob greenlights. Then james may `--launch`.

If any gate fails, bob refuses the GPU. No `--launch`.

## james self-check

- [ ] Gate 1: the change is merged to `master`.
- [ ] Gate 2: each input path is a MANIFEST row and the basename is the frozen name.
- [ ] Gate 2: `sha256`, `bytes`, and `frozen_at` are filled, or still `TBD` with a reason bob can read.
- [ ] Gate 3: `results/<exp_id>/` has no `summary.json`, `summary.csv`, or log. The historical flat summaries above are untouched.
- [ ] Machine log and summary for this run go to `/root/autodl-fs/gs-assets/runs/<run_id>/`. `autodl-tmp` is cache only.
- [ ] Ask bob. Wait for mentor (gate 4) and for bob's greenlight before `--launch`.

## B2

`--attack-steps` defaults to 300. Dry `python src/baselines/b2_makeup_impersonation.py --attack-steps 900` and arm `python src/baselines/b2_makeup_impersonation.py --attack-steps 900 --arm-hook` write only `results/b2_celeba_T0_steps900/` (optional 600 writes only `results/b2_celeba_T0_steps600/`); both refuse product B2 and the other ablation slots and do not pass `--launch`. Makeup-carrier design, also not wired and with no `--launch`, is `docs/makeup_xfer_design.md`.

Wiring may merge. This section does not start a GPU run and does not add `--launch` to the default command. GPU, including B2, still requires all four gates green: gates 1–3, then mentor gate 4. The CLI is in the README「B2」section.

| still | summary directory |
| --- | --- |
| `celeba_T0.jpg` (first) | `results/b2_celeba_T0/summary.json` and `summary.csv` |
| `lfw_T0.jpg` (later) | `results/b2_lfw_T0/summary.json` and `summary.csv` |

In-M ablation does not change the makeup definition. `DEFAULT_PARSING_LABELS` stays the product default: label 1 skin/cheek, 2/3 brow, 4/5 eye, 12/13 lip. Nose, hair, ears, and neck stay out of that default. Do not expand that set. `--ablate full` (omit the flag; `--mask-labels default` is the same name) writes the product directory above. Two locked ablations are complementary. Each directory is its own `<exp_id>` for gate 3.

- `--ablate ablate_noskin` (`--mask-labels ablate_noskin` is the same switch) drops label 1 and keeps {2, 3, 4, 5, 12, 13}. Those runs write `results/b2_celeba_T0_ablate_noskin/` and later `results/b2_lfw_T0_ablate_noskin/`.
- `--ablate ablate_onlyskin` (`--mask-labels ablate_onlyskin` is the same switch) keeps label `{1}` only. Those runs write `results/b2_celeba_T0_ablate_onlyskin/` and later `results/b2_lfw_T0_ablate_onlyskin/`.

`ablate_onlyskin` must not overwrite `results/b2_celeba_T0/`, `results/b2_lfw_T0/`, or either `*_ablate_noskin/` slot. `ablate_noskin` must not overwrite the product slots or either `*_ablate_onlyskin/` slot.

`--ablate softskin` keeps that same label set and the same support width. Hard makeup labels {2, 3, 4, 5, 12, 13} stay on `project_residual` at ε=0.1. Skin label 1 uses ε_skin=0.07, the AEGIS region-adaptive 0.7× skin factor (https://arxiv.org/html/2511.17747). Adv-Makeup is the makeup-local carrier (https://arxiv.org/abs/2105.03162). No new 2D labels. The dual budget is `project_dual_budget_residual` inside `MakeupPgdHook._project_numpy`, which is the residual step on both AEGIS loops: `adaptive_linf_pgd_attack` via `project_configured_delta`, and `AEGIS._run_foolbox_attack` via `bind_foolbox_attack` / `reproject_foolbox`. The live B2 attack is the Foolbox `linfpgd` path. Softskin writes only `results/b2_celeba_T0_softskin/` and `results/b2_lfw_T0_softskin/`. It refuses the product directories above and the other ablation slots (`ablate_noskin`, onlyskin). Each of those two directories is its own `<exp_id>` for gate 3.

`--arm-hook` on the bound 306 ply does not sample Gaussians. The armed summary stays `dc_mask_lift=pending-get-xyz` with an image-domain `mask_shape`. B2-a/b recorded `mask_shape=[540, 960]` and about 10.22% of those pixels inside M. The `(N,)` lift is later: `ensure_dc_mask` calls `sample_image_mask` when `get_xyz` is available, at launch, and only then sets `dc_mask_lift=projected-means`. That step logs one line, `mask (N,) nz=… frac=…`, and the same `dc_mask_shape`, `dc_mask_nonzero`, and `dc_mask_fraction` land in `summary.json`. Do not describe the armed state as already `projected-means`. An older smoke print of an `(N,)` fraction is not the B2-a/b armed record; those runs did not print one.

Image-domain coverage already measured on `(H, W)`, not on Gaussians: default M is about 10.22% of pixels, `ablate_noskin` about 0.7888%, so `ablate_onlyskin` is about 9.43% (about 10.22 − 0.79). `softskin` keeps that default support width and changes only the skin amplitude. Those fractions are not an `(N,)` lift. `--arm-hook` on the bound ply stays `pending-get-xyz`.

PLY `projected-means` counts below are provisional until that launch line exists. Do not copy them into `dc_mask_shape`, `dc_mask_nonzero`, `dc_mask_fraction`, or `dc_mask_support_line` ahead of the log. Full is about 2.307% (2054/89021). Noskin is about 0.2033% (181). Onlyskin is about 2.10% (about 1873 = 2054 − 181). Softskin does not invent a separate `(N,)` count.

The default command still has no `--launch`. Dry coverage is `python src/baselines/b2_makeup_impersonation.py` for `full`, the same command with `--ablate ablate_noskin` for lips/eyes/brows, `--ablate ablate_onlyskin` for the complementary skin-only knife, and `--ablate softskin` for the dual budget. Add `--arm-hook --launch` only after gates 1–3 and mentor gate 4.

The wiring record in that directory has `pipeline_ok` false and null measurements. It is not a finished attack. The authorized run of that same still updates the same directory. A second finished run needs a new directory.

**Post-run gate** (bob freeze). Judge this on the summary after the run. The wiring record leaves these empty and is not a pass. Do not fill them with invented GPU numbers.

| check | pass |
| --- | --- |
| `pipeline_ok` | true |
| cos(adv, source) | ↓ relative to goal (`cos_adv_source`) |
| cos(adv, T) | ↑ (`cos_adv_target`) |
| outside-M `\|δ\|` | 0 (`outside_m_abs_delta`) |

**softskin scoring** (relative compare to B2-a, onlyskin, and noskin). Judge the softskin summary on these records. The wiring record leaves them empty and is not a pass. Do not fill them with invented GPU numbers. This compare is against B2-a and the in-M ablations onlyskin and noskin.

| check | pass |
| --- | --- |
| `pipeline_ok` | true |
| outside-M `\|δ\|` | 0 (`outside_m_abs_delta`) |
| `(N,)` support line | present (`dc_mask_support_line`, from `ensure_dc_mask`) |
| hard vs soft `\|δ\|` | `hard_abs_delta_mean` / `p50` / `p95` and `soft_abs_delta_mean` / `p50` / `p95` recorded |

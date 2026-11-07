# GPU launch checklist

bob enforces this before any AutoDL GPU run. james adds `--launch` only after
bob greenlights. This checklist does not enable B2, does not change attack
code, and does not download datasets.

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

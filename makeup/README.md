# makeup

Semantic mask `M` (lip, eye, brow, cheek), residual `δ` with `supp(δ) ⊆ M` and `||δ||∞ ≤ ε`, and `L_nat` hooks.

**Dependency:** [NumPy](https://numpy.org/) (`pip install numpy`). There is no separate requirements file for this package. NumPy is imported softly: `import makeup` works without it. Mask algebra, residual projection, palette, and `smooth` raise `ImportError` until NumPy is installed.

**LPIPS:** `torch` and [`lpips`](https://pypi.org/project/lpips/) (`pip install torch lpips`), also a soft import. `NaturalnessLoss.lpips` raises `ImportError` until both are installed. It runs on CPU. The first call loads AlexNet LPIPS weights; the package may download them. No GPU is required.

| Callable | Behavior |
| --- | --- |
| `SemanticMask.from_regions`, `region`, `union`, `support` | Hand-built maps. Shared shape; `from_regions` soft-clips to `[0, 1]`. `union` is the elementwise max. `support` is `union > threshold`. |
| `SemanticMask.from_parsing`, `semantic_mask` | Hard `0`/`1` regions from integer labels of shape `(H, W)` or `(N,)`. Default ids are below. `label_map` replaces that default. Optional `image` is an alignment check only. |
| `MakeupResidual.project` / `apply`, `project_residual`, `apply_residual` | A mask must match the residual grid; other shapes raise `ValueError`. |
| `NaturalnessLoss.smooth` | Masked spatial total variation on `H×W×3`. |
| `NaturalnessLoss.palette` | Masked palette term on `(H, W, 3)` or `(N, 3)`. Formula below. |
| `NaturalnessLoss.lpips` | Masked spatial LPIPS when `torch` and `lpips` are installed. |

`NaturalnessLoss.__call__` returns only `smooth`. It does not add palette or LPIPS, and it does not load a network. `apply(..., clip_rgb=True)` clips an RGB sum to `[0, 1]`; DC residuals are not clipped to that range.

### Parsing labels

Default map: `makeup.mask.DEFAULT_PARSING_LABELS`. These are the class ids emitted by BiSeNet trained as in [face-parsing.PyTorch](https://github.com/zllrunning/face-parsing.PyTorch) (`prepropess_data.py` enumerates CelebAMask-HQ attributes from 1; 0 is background). This is not the index order printed in the CelebAMask-HQ dataset README.

| id | parser name | region in `M` |
| --- | --- | --- |
| 1 | skin | cheek (CelebAMask-HQ has no cheek class; pass `label_map` to override) |
| 2, 3 | l_brow, r_brow | brow |
| 4, 5 | l_eye, r_eye | eye |
| 12, 13 | u_lip, l_lip | lip |

Other ids, including background `0`, eyeglasses `6`, nose `10`, inner mouth `11`, and hair `17`, stay outside `M`. Regions named in the label map are stored even when no pixel has that id. Ids that share a region are combined with a pixelwise max.

### Palette term

On residual colors `δ_p` and mask weights `M_p` clipped to `[0, 1]`, with `Z = Σ_p M_p` (the loss is 0 when `Z` is 0):

- No `reference`: compactness, `Σ_p M_p ||δ_p − δ̄||₂² / Z`, where `δ̄` is the masked mean color.
- `reference.shape == delta.shape`: masked MSE against that color field, `Σ_p M_p ||δ_p − r_p||₂² / Z`.
- `reference` of shape `(K, 3)` (and not the same shape as `delta`): nearest palette color, `Σ_p M_p min_k ||δ_p − C_k||₂² / Z`.

A `(N, 3)` array passed with a DC residual of the same shape is the color-field case, not an N-color palette.

### LPIPS

`lpips` compares RGB images in `[0, 1]`, shape `(H, W, 3)`, with AlexNet spatial LPIPS on CPU. Inputs are clipped to `[0, 1]`. With no mask, the result is the mean of the spatial distance map. With a mask, it is the weighted mean over `M`. An all-zero mask scores 0.

Smoke check (no GPU): `python makeup/tests/test_smoke.py`. If `torch` or `lpips` is missing, the LPIPS test expects `ImportError` and the other tests still run.

### NeRSemble 306 masking smoke

`makeup/aegis_mask.py` builds `M` with `SemanticMask.from_parsing` and projects a residual through `project_residual` / `apply_residual` (`supp(δ) ⊆ M`, `||δ||∞ ≤ ε`). The NeRSemble 306 demo has `point_cloud.ply` and `renders/306.png` and no parsing labels. A `.npy` sidecar is used when it exists:

- `third_party/AEGIS/datasets/NeRSembleReconst/parsing/306.npy`
- `third_party/AEGIS/datasets/NeRSembleReconst/avatars/306/parsing.npy`

Otherwise `documented_fallback_labels` fills a label grid with the ids above and still calls `from_parsing`. That grid is not a parse of subject 306.

`projected_pgd_step` is one L∞ sign step followed by that projection. AEGIS Foolbox L∞ PGD (the B0-a 306 path) and `adaptive_linf_pgd_attack` only clamp ε on their own. `--select-regions` is a FLAME subset, not this mask.

`makeup/aegis_pgd_hook.py` is the in-loop hook (it supersedes `aegis_pgd_hook_stub`). Checked against AEGIS `main` `e3d20445942cc3d41e912897675290e090f116f7`:

- `adaptive_linf_pgd_attack` in `src/aegis/core/helpers.py` does `delta - step_size * sign(grad)` and clamps each Gaussian to ±ε. The patch calls `project_residual` on `delta_new` immediately after that clamp.
- `AEGIS._run_foolbox_attack` in `src/aegis/core/model.py` delegates the B0-a `linfpgd` attack to Foolbox `LinfProjectedGradientDescentAttack`. The in-loop projection there is `LinfBaseGradientDescent.project`: `x0 + clip(x - x0, ±ε)`. The patch binds a `project` that runs `project_residual` on that residual before the next step. Other attack names are left alone.

`install_makeup_pgd_hook` writes `.gs_makeup_hook.json` and `.gs_makeup_mask.npy` into the AEGIS root and inserts the calls above. It does not clone the tree. If `src/aegis/core/helpers.py` or `model.py` is missing, the status is `tree_missing`. If those files exist but the verified step text moved, the status is `anchors_mismatch` and nothing is written. `documented_fallback` is refused unless `allow_fallback=True` (that grid is not a parse of subject 306).

Run `mask_avatar.py` from the AEGIS root so the config is picked up, or set `GS_MAKEUP_HOOK_CONFIG` to the json path. With no config, the inserted calls are no-ops. Do not start the AutoDL GPU run until the mentor greenlights it.

```bash
python scripts/nersemble306_makeup_mask_smoke.py
python scripts/nersemble306_makeup_mask_smoke.py --arm-hook --parsing path/to/306.npy
python makeup/tests/test_nersemble306_mask_smoke.py
python makeup/tests/test_aegis_pgd_hook.py
```

`--arm-hook` calls `install_makeup_pgd_hook`. A missing tree still exits 0 after the CPU projection and records `aegis_hook_attach=tree_missing`. Without a real parsing map it records `refused_fallback` and does not patch.

Run steps for the existing 306 ply/png and AdaFace weights on AutoDL are in the repo `README.md`. The makefile `mask_avatar` target is FaceScape 527; the 306 smoke does not call it.

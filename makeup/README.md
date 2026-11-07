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

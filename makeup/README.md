# makeup

Semantic mask `M` (lip, eye, brow, cheek), residual `δ` with `supp(δ) ⊆ M` and `||δ||∞ ≤ ε`, and `L_nat` hooks.

**Dependency:** [NumPy](https://numpy.org/) (`pip install numpy`). There is no separate requirements file for this package. NumPy is imported softly: `import makeup` works without it. `region` / `union` / `support`, `project` / `apply`, and `smooth` raise `ImportError` until NumPy is installed.

| Callable with NumPy | Still `NotImplementedError` |
| --- | --- |
| `SemanticMask.from_regions` (hand-built maps: shared shape, values soft-clipped to `[0, 1]`), `region`, `union` (elementwise max), `support` (`union > threshold`) | `SemanticMask.from_parsing`, `semantic_mask` |
| `MakeupResidual.project` / `apply`, `project_residual`, `apply_residual`. A mask must match the residual grid; other shapes raise `ValueError` | |
| `NaturalnessLoss.smooth` (masked spatial total variation on `H×W×3`) | `palette`, `lpips` |

`NaturalnessLoss.__call__` returns only `smooth`. It does not add palette or LPIPS. `apply(..., clip_rgb=True)` clips an RGB sum to `[0, 1]`; DC residuals are not clipped to that range.

Smoke check (no GPU): `python makeup/tests/test_smoke.py`

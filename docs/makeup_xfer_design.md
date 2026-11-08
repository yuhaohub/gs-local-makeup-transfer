# Makeup carrier (`makeup_xfer`) — design only

中文提要：当前 B2 是皮肤代理，不是妆容攻击。`onlyskin` 贴近 B2-a，`ablate_noskin` 塌掉；`softskin` 双 ε 已归档为负结果（0.675/0.461，比 B2-a 的 0.523/0.558 更钝）。recipe knife（steps ×3 / `steps900`）由另一份 PR 负责，本文件不实现 `--attack-steps`。下一刀是生成式妆容载体（DiffAM 为主，Adv-Makeup 为辅），不是再改皮肤 ε。下面只定钩子、槽位、dry 命令和 bob 的五条可证伪门。本 PR 不接线、不加 `--launch`、不写历史结果目录。

This file specifies the next B2 knife. It does not add a CLI flag, a result directory, a model, or a default `--launch`. Wiring waits until the mentor recipe knife (`steps900`) is green. That knife is a separate PR (cloud agent `bc-703cc5ed-7566-58ed-8f72-9a83ac82fd3c`). `softskin` stays archived. `DEFAULT_PARSING_LABELS` and the dual ε (0.1 / 0.07) stay as they are on master `c7a6ae7`.

## Papers

| role | paper | id | what this knife takes from it |
| --- | --- | --- | --- |
| primary forward | DiffAM (CVPR 2024) | [arXiv:2405.09882](https://arxiv.org/abs/2405.09882) | Makeup transfer plus adversarial makeup. The carrier forward, when wired, is this model. Summary records `carrier=makeup_xfer` and `carrier_model=DiffAM`. |
| secondary reference | Adv-Makeup | [arXiv:2105.03162](https://arxiv.org/abs/2105.03162) | Periocular generation and blend. It justifies keeping the attack on makeup texture (lip, eye, brow, periocular) rather than a skin blob. It is not the forward this knife runs. |

`docs/创新章_交付文档_AEGIS底座.md` already lists both papers as protocol references and says DiffAM is not reproduced as a 2D diffusion system inside the current objective. This design is the later knife that would add that forward. It still does not vendor or call either model.

## Hypothesis

The live B2 attack is a skin proxy.

- Product B2 (`full`) builds `M` from `DEFAULT_PARSING_LABELS` (skin/cheek 1, brow 2/3, eye 4/5, lip 12/13) and projects the Foolbox residual onto that support at ε=0.1.
- `ablate_onlyskin` (label 1 only) tracks B2-a. `ablate_noskin` (drop label 1, keep `{2, 3, 4, 5, 12, 13}`) collapses. The cosine that moves is the cheek/skin mass, not lipstick or eye makeup.
- `softskin` keeps the same support and only tightens skin to ε=0.07 via `project_dual_budget_residual`. Mentor recorded that pair as **0.675/0.461**, duller than B2-a **0.523/0.558**. That result stays negative. Do not open another skin-ε knife.

The knife after recipe exclusion is a generative makeup carrier. The attack energy has to come from a makeup-transfer forward. `project_residual` on skin is a feasibility clip (`supp(δ) ⊆ M`, `||δ||∞ ≤ ε`), not the budget that creates the makeup.

## What the code does today

Same attack hyperparameters as B1 / B0-a: AdaFace, ε=0.1, 300 steps, target-features DC, Foolbox `linfpgd`, radius 1, seed 42. `b2_aegis_kwargs` in `src/baselines/b2_makeup_impersonation.py` writes `attack_steps` from `b1.ATTACK_STEPS` (300). This design does not change that line.

### Where `M` is built

1. `resolve_mask_labels` (`src/baselines/b2_makeup_impersonation.py`) returns `("full", None)` for the product run. `None` means `SemanticMask.from_parsing` reads `DEFAULT_PARSING_LABELS` in `makeup/mask.py`. `ablate_noskin` returns a map of that default with label 1 removed (`{2, 3, 4, 5, 12, 13}`). `ablate_onlyskin` returns `{1}`. `softskin` returns `("softskin", None)` and does not narrow the map.
2. `build_dc_hook` loads `parsing/306.npy` through `load_parsing_npy` (`makeup/aegis_mask.py`) and calls `dc_hook_from_parsing` (`makeup/aegis_pgd_hook.py`) at `HARD_EPSILON` (0.1). A dual budget is the only caller that also passes `soft_label_ids=(1,)` and `soft_epsilon=0.07`.
3. `dc_hook_from_parsing` calls `SemanticMask.from_parsing(grid, label_map=label_map)` and stores `union()` as the image mask. On the bound 306 ply, means are absent at arm time, so the hook stays `semantic_source=image-parsing`, `dc_mask_lift=pending-get-xyz`, shape `(H, W)`.
4. `ensure_dc_mask` (`makeup/aegis_pgd_hook.py`) runs later, from `bind_foolbox_attack`, once `gaussians.get_xyz` exists. It samples with `sample_image_mask` (`makeup/aegis_mask.py`), logs `mask (N,) nz=… frac=…`, and sets `dc_mask_lift=projected-means`. A dual-budget hook samples `soft_image_mask` at the same means.

`install_b2_hook` calls `install_makeup_pgd_hook(..., mask_source="from_parsing")`. That patches two AEGIS files when the tree is present:

- `src/aegis/core/helpers.py` — `adaptive_linf_pgd_attack` (side path; B2 does not launch it)
- `src/aegis/core/model.py` — `AEGIS._run_foolbox_attack` (the live path)

Upstream commit the patch anchors to: `e3d20445942cc3d41e912897675290e090f116f7`.

### Where the residual is projected

`MakeupPgdHook._project_numpy` (`makeup/aegis_pgd_hook.py`) is the single makeup step both loops reach.

- Uniform B2 calls `project_residual` (`makeup/residual.py`) at `self.epsilon` (0.1). That is `MakeupResidual.project`: support inside `M`, values inside `[-ε, ε]`.
- `softskin` calls `project_dual_budget_residual` (`makeup/residual.py`): hard labels `{2, 3, 4, 5, 12, 13}` stay at 0.1, skin label 1 uses 0.07. Overlap keeps the tighter ball.

Call chain on the live attack:

```
run_attack
  → b1.apply_targeted_criterion(AEGIS, b1.make_foolbox_targeted_match)
  → AEGIS(**b2_aegis_kwargs(plan)); model.run()
  → AEGIS._run_foolbox_attack          # patched
  → get_foolbox_attack(linfpgd, steps=self.attack_steps)
  → bind_foolbox_attack
       → ensure_dc_mask
       → attack.project = ε-clip, then MakeupPgdHook.reproject_foolbox
            → project_like → _project_numpy → project_residual
```

`b1.apply_targeted_criterion` (`src/baselines/b1_targeted_impersonation.py`) replaces the label `AEGIS._run_foolbox_attack` would forward. `make_foolbox_targeted_match` builds Foolbox `TargetedMisclassification` of match class 1, so PGD raises cosine with `T`. The makeup hook does not change that criterion.

`reproject_foolbox` takes the residual Foolbox has already clipped to ±ε (`LinfBaseGradientDescent.project` is `x0 + clip(x - x0, ±ε)`) and runs `_project_numpy` on `x - x0`. `project_configured_delta` is the same `_project_numpy` entry for `adaptive_linf_pgd_attack`. B2's launched attack is the Foolbox one.

`outside_m_abs_delta` scores `max |adv - clean|` outside `M` (gate is 0). `abs_delta_breakdown` splits hard vs soft `|δ|` (mean, p50, p95, max, count) using the support mask and a soft mask. Softskin uses the skin ball as that soft mask. Those two functions are the scorers a carrier summary should keep using.

## Where a makeup-xfer forward injects

The carrier is a new forward. Nothing in this repo calls DiffAM. When a later PR wires it, the forward's output is an appearance residual or a rendered makeup map, and that map is what becomes the DC residual. Skin `project_residual` / `project_dual_budget_residual` stays the clip that enforces `M` and ε. It is not the main budget.

Two injection sites, in the order a wiring PR should try them. Both record `carrier=makeup_xfer` / `carrier_model=DiffAM` on the summary from `run_attack` (the fields land in `_empty_metrics` / `_payload`).

1. **Before `model.run()` (preferred).** `run_attack` already owns `AEGIS(**b2_aegis_kwargs(plan))` and `model.run()`. A DiffAM forward runs there, in image space, and produces the makeup map. Lift it with the same camera `M` already uses: `sample_image_mask` at `gaussians.get_xyz` (the call `ensure_dc_mask` makes). Add the lifted residual with `apply_residual` (`makeup/residual.py`) so the write still obeys `supp(δ) ⊆ M` and `||δ||∞ ≤ 0.1`. Foolbox then starts from that DC state. Later steps still go through `reproject_foolbox` → `_project_numpy` → `project_residual`, which only keeps the iterate inside `M`.
2. **Inside the step, if the carrier has to be reapplied.** The call sits in `MakeupPgdHook.reproject_foolbox`, after Foolbox's ε clip and before `_project_numpy`. The summary still records the carrier. This site is the fallback when a single initialization is too weak. It is still a makeup-transfer forward, not a second skin ε.

`project_dual_budget_residual` is off this path. Wiring must not pass `budget="dual"`, must not change `SOFT_SKIN_EPSILON`, and must not edit `DEFAULT_PARSING_LABELS`.

`apply_residual` is the add. `project_residual` is the clip after the carrier (or the per-step clip). The sentence bob needs in the summary is that the forward which produced the residual is `carrier=makeup_xfer` (DiffAM), and that skin-budget `project_residual` was not the main update.

## Proposed CLI (do not implement in this PR)

Either of these names, matching the existing `--ablate` slot style:

- `--ablate makeup_xfer`
- `--carrier makeup_xfer`

`--mask-labels` is already an alias of `--ablate`. A later PR can accept `makeup_xfer` there too. The mode keeps the product label set (`full` / `DEFAULT_PARSING_LABELS`) and ε=0.1. It does not reuse `softskin`, `ablate_noskin`, or `ablate_onlyskin` as the primary slot.

Today `parse_args` only accepts `MASK_LABEL_CHOICES` (`full`, `default`, `ablate_noskin`, `ablate_onlyskin`, `softskin`). `resolve_mask_labels("makeup_xfer")` raises `MaskLabelError`. That is expected until the wiring PR.

## Slots

Writable only after the wiring PR, and only these directories:

| still | directory |
| --- | --- |
| `celeba_T0.jpg` (first) | `results/b2_celeba_T0_makeup_xfer/` |
| `lfw_T0.jpg` (later) | `results/b2_lfw_T0_makeup_xfer/` |

Each directory is its own `<exp_id>` for gate 3 in `docs/gpu_launch_checklist.md`.

Read-only, including for a noskin-under-carrier companion:

- `results/b2_celeba_T0/` and `results/b2_lfw_T0/` (product B2-a)
- `results/b2_*_ablate_noskin/`
- `results/b2_*_ablate_onlyskin/`
- `results/b2_*_softskin/`
- `results/b2_*_steps900/` once the other PR creates it

The noskin-under-carrier measurement is a field on the makeup_xfer summary (`noskin_under_carrier`), not a write into `results/b2_*_ablate_noskin/`. A later directory, if one is required, still has to match `b2_*_makeup_xfer`.

## Expected versus B2-a

B2-a floor pair recorded by the mentor: **0.523/0.558** (`cos_adv_source` / `cos_adv_target`). Softskin **0.675/0.461** is the archived miss: source cosine went up, target cosine went down.

The existing post-run gate (`docs/gpu_launch_checklist.md`) scores `cos_adv_source` down and `cos_adv_target` up, plus `pipeline_ok` and outside-M `|δ| = 0`.

For this carrier, on CelebA:

- The dual cosine is not worse than 0.523/0.558. Read bob's floor as: `cos_adv_source` stays at or below 0.523, and `cos_adv_target` stays at or above 0.558.
- A noskin-style ablation under the carrier does not collapse the way current B2 noskin collapses, or it collapses far less. That is the evidence the channel is makeup texture.
- If the carrier misses that pair, or the noskin companion still collapses like the product knife, the narrative is rewritten. The response is not another ε on skin.

## Noskin under the new carrier

`ablate_noskin` today is `resolve_mask_labels`: drop parser label 1, keep `{2, 3, 4, 5, 12, 13}`, pass that map as `label_map` into `build_dc_hook` → `dc_hook_from_parsing` → `SemanticMask.from_parsing`. `project_residual` then zeros every Gaussian outside that support. Skin never enters `M`.

Under `makeup_xfer` the carrier runs first. The same label cut still applies to the carrier residual, after the generative forward and before the residual is treated as the attack budget. Two equivalent procedures; the wiring PR picks one and records it as `noskin_carrier`:

1. **`mask_residual` (preferred).** Build the noskin map with `resolve_mask_labels("ablate_noskin")`. Build a label-1 weight the same way `dc_hook_from_parsing` builds a subset (`_soft_weight` in `makeup/aegis_pgd_hook.py`, called with `{1}`). Zero those skin sites on the transferred makeup residual (image map or lifted DC) before `project_residual` / `_project_numpy` / `apply_residual`. Labels `{2, 3, 4, 5, 12, 13}` stay. The carrier still ran; the ablation only removes the skin part of its output.
2. **`rerun_support`.** Run the same DiffAM forward again with the noskin `label_map` passed through `build_dc_hook`. `ensure_dc_mask` never lifts label 1, and `project_residual` drops whatever the carrier put on skin.

Bob's gate: if that companion still collapses the way the product noskin knife collapses, the hypothesis is red. The channel is still the skin blob. Do not answer that result by raising skin ε or by reopening `softskin`.

## Bob falsifiability

All five have to hold before this knife is called a makeup attack. A miss is a red hypothesis, not a prompt to edit `DEFAULT_PARSING_LABELS` or the softskin ε.

1. The forward path is makeup-xfer / generative. The summary records `carrier=makeup_xfer` (DiffAM). Skin-budget `project_residual` (and `project_dual_budget_residual`) is not the main budget.
2. The noskin-style ablation does not collapse like current B2.
3. CelebA dual cos is at least the B2-a floor **0.523/0.558**. Scoring direction from the GPU checklist: `cos_adv_source` ≤ 0.523 and `cos_adv_target` ≥ 0.558. Softskin 0.675/0.461 fails that reading.
4. Slots are only `results/b2_*_makeup_xfer/`. Product B2-a, softskin, noskin, onlyskin, and the steps900 slot are read-only.
5. Hard-makeup `|δ|` dominates the skin blob. Score with `abs_delta_breakdown`: hard sites are labels `{2, 3, 4, 5, 12, 13}`, skin blob is label 1. Pass label 1 as the breakdown's soft mask for scoring only. That call does not switch the attack onto `project_dual_budget_residual`. Dominance means `hard_abs_delta_mean` and `hard_abs_delta_p95` above the skin aggregate, with both sides recorded.

## Dry command

Design only. Not wired until the mentor recipe knife (`steps900`) is green. No `--launch`.

```bash
python src/baselines/b2_makeup_impersonation.py --ablate makeup_xfer
```

Equivalent name, also not wired:

```bash
python src/baselines/b2_makeup_impersonation.py --carrier makeup_xfer
```

On current master `parse_args` rejects both. `--ablate` only accepts `MASK_LABEL_CHOICES` (`full`, `default`, `ablate_noskin`, `ablate_onlyskin`, `softskin`), and `--carrier` is not a flag. `resolve_mask_labels("makeup_xfer")` would raise `MaskLabelError` if a caller skipped argparse. That refusal is the current contract. A wiring PR is what makes the dry command write `results/b2_celeba_T0_makeup_xfer/summary.json` with `pipeline_ok` false, null cosines, and `carrier=makeup_xfer`. It still must not pass `--launch` by default, and it still must not write `results/b2_celeba_T0/`.

GPU, if it is ever authorized, still follows `docs/gpu_launch_checklist.md`: gates 1–3, then mentor gate 4, then james adds `--arm-hook --launch`. This design does not authorize that.

## Hooks for the wiring PR

Exact symbols as of master `c7a6ae7`.

| step | file | symbol | carrier use |
| --- | --- | --- | --- |
| product labels | `makeup/mask.py` | `DEFAULT_PARSING_LABELS`, `SemanticMask.from_parsing`, `SemanticMask.union` | Build `M`. Do not edit the constant. |
| mode → label map | `src/baselines/b2_makeup_impersonation.py` | `resolve_mask_labels`, `parsing_label_ids` | `full` keeps the constant. Noskin companion uses the `ablate_noskin` map. |
| hook build | `src/baselines/b2_makeup_impersonation.py` | `build_dc_hook` | Loads `306.npy`, calls `dc_hook_from_parsing`. Carrier stays `budget="uniform"`. |
| image mask | `makeup/aegis_pgd_hook.py` | `dc_hook_from_parsing` | `(H, W)` mask, `dc_mask_lift=pending-get-xyz` when means are absent. |
| label-1 weight | `makeup/aegis_pgd_hook.py` | `_soft_weight` | Noskin-under-carrier procedure `mask_residual` zeros this weight on the carrier residual. |
| Gaussian lift | `makeup/aegis_pgd_hook.py` | `ensure_dc_mask` | Samples at `get_xyz` before the first Foolbox step. |
| pixel → Gaussian | `makeup/aegis_mask.py` | `sample_image_mask` | Same sampler for `M` and for a rendered makeup map. |
| means | `makeup/aegis_pgd_hook.py` | `_attacked_means` | Reads `gaussians.get_xyz`. |
| install | `makeup/aegis_pgd_hook.py` | `install_makeup_pgd_hook` | Patches the two AEGIS loops. |
| B2 install | `src/baselines/b2_makeup_impersonation.py` | `install_b2_hook` | `mask_source="from_parsing"`. |
| attack entry | `src/baselines/b2_makeup_impersonation.py` | `run_attack` | Preferred site for the DiffAM forward, before `model.run()`. Summary `carrier` is written from this path. |
| hyperparameters | `src/baselines/b2_makeup_impersonation.py` | `b2_aegis_kwargs` | Forces `linfpgd`, ε=0.1, `attack_steps=b1.ATTACK_STEPS`. Leave the steps line to the recipe-knife PR. |
| criterion | `src/baselines/b1_targeted_impersonation.py` | `apply_targeted_criterion`, `make_foolbox_targeted_match` | `TargetedMisclassification` of class 1. |
| Foolbox bind | `makeup/aegis_pgd_hook.py` | `bind_foolbox_attack` | Lifts `M`, wraps `LinfProjectedGradientDescentAttack.project`. |
| per-step clip | `makeup/aegis_pgd_hook.py` | `MakeupPgdHook.reproject_foolbox`, `MakeupPgdHook.project_like`, `MakeupPgdHook._project_numpy` | Fallback injection site. `_project_numpy` must keep calling `project_residual` as a clip. |
| clip implementation | `makeup/residual.py` | `project_residual`, `MakeupResidual.project` | Feasibility. Not the main budget. |
| add implementation | `makeup/residual.py` | `apply_residual` | Adds the lifted carrier residual under `M` and ε. |
| archived dual budget | `makeup/residual.py` | `project_dual_budget_residual` | Softskin only. Carrier does not use it as the attack. |
| adaptive side path | `makeup/aegis_pgd_hook.py` | `project_configured_delta` | `adaptive_linf_pgd_attack`. Not the live B2 attack. |
| outside-M gate | `src/baselines/b2_makeup_impersonation.py` | `outside_m_abs_delta` | Stays 0. |
| hard vs skin `|δ|` | `src/baselines/b2_makeup_impersonation.py` | `abs_delta_breakdown` | Falsifiability point 5. Skin mask is scoring-only. |
| summary | `src/baselines/b2_makeup_impersonation.py` | `_empty_metrics`, `_payload`, `write_summary` | Record `carrier`, `carrier_model`, `noskin_carrier`, `noskin_under_carrier`. Seed and ε stay the B2-a values. |
| slot guard | `src/baselines/b2_makeup_impersonation.py` | `summary_slot`, `default_summary_dir`, `assert_ablation_slot` | Pattern to copy from `softskin`: the new mode writes only its two directories and refuses the others. |
| AEGIS Foolbox | `third_party/AEGIS/src/aegis/core/model.py` | `AEGIS._run_foolbox_attack` | `get_foolbox_attack(..., steps=self.attack_steps)` then the bind call the hook inserts. |
| AEGIS adaptive | `third_party/AEGIS/src/aegis/core/helpers.py` | `adaptive_linf_pgd_attack` | Side path. Do not make it the carrier. |

## Out of scope

- No `--attack-steps`, no `--ablate steps900`, no `results/b2_*_steps900/` writer. That PR is the other cloud agent.
- No DiffAM or Adv-Makeup weights, download, or forward code.
- No change to `DEFAULT_PARSING_LABELS`, `SOFT_SKIN_EPSILON`, or the softskin slot.
- No `--launch` on the default command.
- No writes under `results/b2_celeba_T0/`, `results/b2_lfw_T0/`, `*_ablate_noskin/`, `*_ablate_onlyskin/`, or `*_softskin/`.

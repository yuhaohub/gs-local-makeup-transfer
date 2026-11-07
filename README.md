# gs-local-makeup-transfer


面向黑盒人脸 API 的三维高斯局部妆容域通用迁移（独立创新章）统一代码仓。

## 冻结口径
- **实现底座**：AEGIS（https://github.com/Woleek/AEGIS ，arXiv:2511.17747）
- **主方法**：AEGIS 接口 → 定向冒充 + 妆容语义 mask + DPA 多样代理 + 预处理不变
- **消融**：中期最优；可选 \(\mathcal{L}_{univ}\)
- **对照**：DiffAM/Adv-Makeup（协议）；3DAO（有全文后任务对照）
- **唯一思路文档**：`docs/创新章_交付文档_AEGIS底座.md`（与群内导师主文件同步，勿另起文档分叉）

## 目录（导师冻结布局）
```
third_party/AEGIS/   上游底座
makeup/              语义 mask 与自然性
transfer/            DPA 代理 + 预处理 A
eval/                API / mean_delta / 自然性协议
baselines/midterm/   中期脚本（仅消融）
docs/                只放一份交付文档
configs/ scripts/ results/
```

`makeup/` 用 NumPy 实现语义 mask（`from_parsing` 默认 BiSeNet / CelebAMask-HQ 标签）、残差投影（\(\mathrm{supp}(\delta)\subseteq M\)、\(\|\delta\|_\infty\le\varepsilon\)）、平滑项与调色板项；LPIPS 在安装 `torch` 与 `lpips` 后于 CPU 上计算（见 `makeup/README.md`）。

## 接入 AEGIS（底座）
```bash
git submodule add https://github.com/Woleek/AEGIS.git third_party/AEGIS
# 或：git clone --depth 1 https://github.com/Woleek/AEGIS.git third_party/AEGIS
```
按 AEGIS README 装依赖后，先跑通 **B0 去身份复现**，再改 B1→B2→M（见交付文档 §8 / 工程第一刀）。

## B0 gate

`scripts/check_b0_gate.sh` 检查冻结目录；`third_party/AEGIS` 不在时提示 `scripts/00_bootstrap_aegis.sh`（`--bootstrap` 会执行它）。然后调用 `python src/baselines/b0_aegis_reproduce.py`，由该模块写入摘要（门禁不另起一套字段）：

- `results/b0_aegis_sanity_summary.json`：`experiment_id`, `desc`, `status`, `bootstrapped`, `metrics`, `note`
- `results/b0_aegis_sanity_summary.csv`：`experiment_id,status,bootstrapped,command,return_code,note`

`status` 取 `bootstrap_required`、`deps_missing`、`gpu_required`、`not_run`、`ok`。缺依赖或无 CUDA（`deps_missing` / `gpu_required`）时门禁退出码为 0。

```bash
bash scripts/check_b0_gate.sh
bash scripts/check_b0_gate.sh --bootstrap
python src/baselines/b0_aegis_reproduce.py
```

## NeRSemble 306 妆容 mask 冒烟

正式基线是 **NeRSemble subject 306**。B0-a 已在该头像上跑通原文 AEGIS masking（AdaFace，`eps=0.1`，`target-features DC`，`linfpgd`，聚合余弦约 -0.2488），摘要在 `results/b0_aegis_sanity_summary.*`。这一刀只把妆容语义 mask 接到同一条 masking 路径上，检查残差满足 \(\mathrm{supp}(\delta)\subseteq M\) 且 \(\|\delta\|_\infty\le\varepsilon\)。它不是完整 B2（没有定向冒充，也没有论文评测）。

`third_party/AEGIS` 的 makefile 目标 `mask_avatar` 写的是 FaceScape 527。527 已取消。306 冒烟直接调用 `scripts/mask_avatar.py`，参数与 B0-a 摘要里的命令一致。

306 demo 只有 `datasets/NeRSembleReconst/avatars/306/point_cloud.ply` 和 `datasets/NeRSembleReconst/renders/306.png`，没有 parsing 标签。有 `.npy` 整数标签时走 `SemanticMask.from_parsing`（BiSeNet / CelebAMask-HQ id，见 `makeup/README.md`）。没有标签时，CPU 冒烟用 `documented_fallback_labels` 造一张同样 id 的标签图再交给 `from_parsing`。这张图不是 306 的解析结果。

AEGIS 的 Foolbox L∞ PGD（B0-a 那条，`AEGIS._run_foolbox_attack`）和 `--adaptive-epsilon` 用的 `adaptive_linf_pgd_attack` 都只做 ε 裁剪，不乘妆容 mask。`--select-regions` 是 FLAME 区域子集，不是 `from_parsing`。约束在 `makeup.residual.project` / `apply` 和 `makeup/aegis_mask.py` 的 `projected_pgd_step`。步内钩子留给 james，见该文件的 TODO。

CPU（无 GPU、无权重、不下载数据）：

```bash
python scripts/nersemble306_makeup_mask_smoke.py
python makeup/tests/test_nersemble306_mask_smoke.py
```

摘要写到 `results/nersemble306_makeup_mask_smoke.json` 和 `.csv`。`status=cpu_smoke_ok` 表示 CPU 投影满足支撑约束；`gpu_launched=false` 表示没有启动 `mask_avatar.py`。

AutoDL 上用机器上已有的 306 ply/png 和 AdaFace 权重 `third_party/AEGIS/models/adaface_ir50_ms1mv2.ckpt`。不要重新下载，也不要跑 `make mask_avatar`。在 `third_party/AEGIS` 目录执行：

```bash
uv run python scripts/mask_avatar.py \
  --avatar-dir ./datasets/NeRSembleReconst/avatars/306 \
  --target-image ./datasets/NeRSembleReconst/renders/306.png \
  --camera-boundary-angles -0.5 0.5 -0.5 0.5 0.0 0.0 \
  --angle-aggregation mean \
  --seed 42 \
  --epsilons 0.1 \
  --attack-steps 300 \
  --target-features DC \
  --adv-attack linfpgd \
  --embedder adaface \
  --radius 1 \
  --output-name NeRSembleMasked306
```

真实标签放在 `third_party/AEGIS/datasets/NeRSembleReconst/parsing/306.npy`（或 `avatars/306/parsing.npy`），也可以：

```bash
python scripts/nersemble306_makeup_mask_smoke.py \
  --parsing third_party/AEGIS/datasets/NeRSembleReconst/parsing/306.npy
```

`--launch` 只在 uv、CUDA、306 ply/png、AdaFace checkpoint、InsightFace `det_10g.onnx`、`GaussianAvatars` 都已经在盘上时才启动上面的命令。缺任何一项就跳过，避免脚本自己去下载。启动后的 AEGIS 进程仍然是原文 Foolbox 路径；妆容支撑约束以本仓库的投影为准。

## 实验矩阵（提交代码时对齐）
| ID | 含义 |
|----|------|
| B0 | AEGIS 原文去身份复现 |
| B1 | 同接口改定向冒充（全脸） |
| B2 | + 妆容 mask + 幅度约束 |
| M  | + DPA + 预处理不变（主方法） |
| A1/A2/A3 | 消融 DPA / 预处理 / univ |
| C1 | 中期最优管线 |

## 协作约定
1. 默认分支 `main`；功能分支 `feat/*`、`exp/*`。
2. 一人一 PR；大结果与权重放 `results/` 本地或网盘，git 只跟踪小表 CSV/JSON。
3. API 密钥用环境变量，禁止提交。
4. 文档只改 `docs/创新章_交付文档_AEGIS底座.md` 这一份。

## 本地路径
本机共享盘：`/workspace/gs-local-makeup-transfer`

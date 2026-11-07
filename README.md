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

正式复现主体冻结为 **NeRSemble 306**。已合并的 B0-a（`subject=306`，`status=pipeline_ok`，`return_code=0`）就是正式 B0。**B0-b（FaceScape 527）已取消**，不在关键路径上，门禁不再等待该主体。

`scripts/check_b0_gate.sh` 检查冻结目录。若 `results/b0_aegis_sanity_summary.json` 已是上述正式记录，门禁直接通过，不重跑、不改写摘要。否则在 `third_party/AEGIS` 不在时提示 `scripts/00_bootstrap_aegis.sh`（`--bootstrap` 会执行它），然后调用 `python src/baselines/b0_aegis_reproduce.py`，由该模块写入摘要（门禁不另起一套字段）：

- `results/b0_aegis_sanity_summary.json`：`experiment_id`, `desc`, `status`, `subject`, `bootstrapped`, `metrics`, `note`
- `results/b0_aegis_sanity_summary.csv`：`experiment_id,status,subject,pipeline_ok,bootstrapped,command,return_code,note`

正式通过条件：`status=pipeline_ok` 且 `return_code=0` 且 `subject=306`。`status` 另可取 `bootstrap_required`、`deps_missing`、`gpu_required`、`not_run`、`ok`。缺依赖或无 CUDA（`deps_missing` / `gpu_required`）且摘要不是 FaceScape 527 / B0-b 时，门禁退出码仍为 0；这不是正式 B0。

```bash
bash scripts/check_b0_gate.sh
bash scripts/check_b0_gate.sh --bootstrap
python src/baselines/b0_aegis_reproduce.py
```

## NeRSemble 306 妆容 mask 冒烟

正式基线是 **NeRSemble subject 306**。B0-a 已在该头像上跑通原文 AEGIS masking（AdaFace，`eps=0.1`，`target-features DC`，`linfpgd`，聚合余弦约 -0.2488），摘要在 `results/b0_aegis_sanity_summary.*`。这一刀只把妆容语义 mask 接到同一条 masking 路径上，检查残差满足 \(\mathrm{supp}(\delta)\subseteq M\) 且 \(\|\delta\|_\infty\le\varepsilon\)。它不是完整 B2（没有定向冒充，也没有论文评测）。

`third_party/AEGIS` 的 makefile 目标 `mask_avatar` 写的是 FaceScape 527。527 已取消。306 冒烟直接调用 `scripts/mask_avatar.py`，参数与 B0-a 摘要里的命令一致。

306 demo 只有 `datasets/NeRSembleReconst/avatars/306/point_cloud.ply` 和 `datasets/NeRSembleReconst/renders/306.png`，没有 parsing 标签，也不带 `parsing/306.npy`。有 `.npy` 整数标签时走 `SemanticMask.from_parsing`（BiSeNet / CelebAMask-HQ id，见 `makeup/README.md`）。没有标签时，CPU 冒烟用 `documented_fallback_labels` 造一张同样 id 的标签图再交给 `from_parsing`。这张图不是 306 的解析结果。真实 `306.npy` 用下面的 BiSeNet 脚本生成。

AEGIS 的 Foolbox L∞ PGD（B0-a 那条，`AEGIS._run_foolbox_attack`）和 `--adaptive-epsilon` 用的 `adaptive_linf_pgd_attack` 原文都只做 ε 裁剪。`--select-regions` 是 FLAME 区域子集，不是 `from_parsing`。妆容约束仍是 `project_residual`。步内钩子在 `makeup/aegis_pgd_hook.py`（接上 PR #10 留给 james 的两个调用点，不再只是 stub）：

- `helpers.py` 的 `adaptive_linf_pgd_attack`：符号步和逐高斯 ε 裁剪之后调用 `project_residual`。
- `model.py` 的 `AEGIS._run_foolbox_attack`：B0-a 的 `linfpgd` 走 Foolbox `LinfProjectedGradientDescentAttack.project`（`x0 + clip(x-x0, ±ε)`）。补丁在 `attack(...)` 之前换上 `project`，并把 `self.gaussians` 传进去。图像上的 `M` 先按高斯均值采样成 `(N,)`，每一步再对 DC 残差做同一次投影。

`install_makeup_pgd_hook` 只在已经有 `third_party/AEGIS` 时改那两个文件，并写入 `.gs_makeup_hook.json`。树不在时状态是 `tree_missing`，不克隆、不下载。`documented_fallback` 默认拒绝装进真实攻击。不要在导师点头前开 AutoDL GPU。细节见 `makeup/README.md`。

CPU（无 GPU、无权重、不下载数据）：

```bash
python scripts/nersemble306_makeup_mask_smoke.py
python makeup/tests/test_nersemble306_mask_smoke.py
```

摘要写到 `results/nersemble306_makeup_mask_smoke.json` 和 `.csv`。`status=cpu_smoke_ok` 表示 CPU 投影满足支撑约束；`gpu_launched=false` 表示没有启动 `mask_avatar.py`。

### 真实 306.npy

NeRSemble demo 不带 `parsing/306.npy`。批准的做法是用 [face-parsing.PyTorch](https://github.com/zllrunning/face-parsing.PyTorch) 的 BiSeNet（CelebAMask-HQ，19 类）解析这一张 306 渲染，写出妆容管线要的整数标签图。类别 id 与 `makeup.mask.DEFAULT_PARSING_LABELS` 一致：唇 12/13，眼 4/5，眉 2/3，皮肤 1 当作脸颊。`documented_fallback_labels` 只给 CPU 冒烟用，不是这张图。

```bash
git clone https://github.com/zllrunning/face-parsing.PyTorch.git third_party/face-parsing.PyTorch
```

权重不在该仓库里。从 face-parsing.PyTorch README 的 Google Drive 下载 `79999_iter.pth`（https://drive.google.com/open?id=154JgKpzCPW82qINcVieuPH3fZ2e0P812），放到 `third_party/face-parsing.PyTorch/res/cp/79999_iter.pth`。脚本不克隆、不下载。`Resnet18.init_weight` 原本会再拉一份 ImageNet ResNet-18；这里关掉那次下载，因为 `79999_iter.pth` 里已经有这些层。

```bash
python scripts/make_nersemble306_parsing.py
python scripts/make_nersemble306_parsing.py --image /path/to/306.png --cpu
```

默认读 `third_party/AEGIS/datasets/NeRSembleReconst/renders/306.png`（没有就 `--image`）。BiSeNet 之前先做人脸检测：用 AEGIS 已经在用的 InsightFace `buffalo_l` / `det_10g.onnx`（`insightface.model_zoo.get_model` 的 RetinaFace/SCRFD `detect`）。不另装 `retinaface` 包。检测框按 `--margin`（默认 0.25）外扩，默认再收成正方形（`--no-square` 可关），然后裁到图内。BiSeNet 只跑这块裁剪；标签最近邻缩回裁剪尺寸，再贴回与原图同大的全 0 背景。因此 `parsing/306.npy` 仍是 `(H, W)`、`int64`，和 `renders/306.png` 对齐。另一处 sidecar 是 `third_party/AEGIS/datasets/NeRSembleReconst/avatars/306/parsing.npy`（用 `--out` 指过去）。

写出之后用 `SemanticMask.from_parsing` 和 `DEFAULT_PARSING_LABELS` 检查唇、眼、眉、脸颊的像素数都大于 0。有任一为 0（上一张被拒就是 lip=0，缺 id 3/5/12/13）则退出码 1，npy 不算接受。检测依赖缺失时退出码 2，并写明要装什么：`pip install insightface onnxruntime`（若报 `cv2` 再装 `opencv-python-headless`），权重必须已经在 `~/.insightface/models/buffalo_l/det_10g.onnx`。脚本不下载该 onnx，也不调用 `insightface.utils.ensure_available`。AutoDL 上这个文件通常已经在。检测到 CUDA 才用 GPU，否则走 CPU；`--cpu` 强制 CPU。缺仓库、缺 BiSeNet 权重或缺图同样退出码 2。

成功写出（退出码 0）后重新跑质量检查：

```bash
python scripts/viz_nersemble306_parsing.py
```

这不是 masking，也不装 AEGIS 攻击依赖。不要在导师点头前开 AutoDL 上的 GPU masking。不跑 BiSeNet 权重的检查：

```bash
python makeup/tests/test_nersemble306_parsing.py
```

### 306 parsing 质量检查（CPU）

`scripts/make_nersemble306_parsing.py` 成功写出新的 `parsing/306.npy` 之后，重新跑这一步，再把终端输出和叠加图交给导师。`scripts/viz_nersemble306_parsing.py` 用 `SemanticMask.from_parsing` 统计 lip / eye / brow / cheek 和并集的像素数（以及占 `H×W` 的比例），打印图里实际出现的 id 和个数，并把这些区域叠到 `renders/306.png` 上。唇、眼、眉、脸颊用不同颜色；`M` 以外保持原图像素。唇或任一必需区域像素数为 0 时退出码是 1，叠加图仍然会写出来。缺左/右半边 id（例如眉 3、眼 5）会打出 `INCOMPLETE`，退出码仍由「区域是否为空」决定。这一步不跑 BiSeNet，也不启动 GPU masking。

在仓库根目录执行（AutoDL 上 306 的 npy 和 png 已经在默认路径时，这一条就够）：

```bash
python scripts/viz_nersemble306_parsing.py
```

显式路径：

```bash
python scripts/viz_nersemble306_parsing.py \
  --parsing third_party/AEGIS/datasets/NeRSembleReconst/parsing/306.npy \
  --image third_party/AEGIS/datasets/NeRSembleReconst/renders/306.png \
  --out results/nersemble306_parsing_overlay.png
```

把整段终端输出和 `overlay:` 那一行交给导师。图例（alpha 0.55，`(1-alpha)*image + alpha*color`，绘制顺序 cheek、brow、eye、lip）：cheek `#F0B428`，brow `#326EE6`，eye `#28C850`，lip `#DC283C`。叠加图在 `results/nersemble306_parsing_overlay.png`（git 不跟踪这张 png）。不读真实 npy 的检查：

```bash
python makeup/tests/test_nersemble306_parsing_viz.py
```

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

真实标签用上面的 `scripts/make_nersemble306_parsing.py` 生成（demo 不自带这个文件），放在 `third_party/AEGIS/datasets/NeRSembleReconst/parsing/306.npy`（或 `avatars/306/parsing.npy`），也可以：

```bash
python scripts/nersemble306_makeup_mask_smoke.py \
  --parsing third_party/AEGIS/datasets/NeRSembleReconst/parsing/306.npy
```

B0-a 的 Foolbox 残差是 DC，形状 `(N, 1, 3)`（306 上 N=89021）。真实 `parsing/306.npy` 仍是语义来源，形状 `(H, W)` = `(540, 960)`，与正面渲染 `renders/306.png` 对齐。`--arm-hook` 用 `SemanticMask.from_parsing` 得到妆容 `M`（`DEFAULT_PARSING_LABELS` 不含鼻尖 id 10），再把它采样成与 DC 同行的 `(N,)` 掩码，钩子 `kind=dc`。

采样相机与 `AEGIS._init_orbit_cam` 相同：`OrbitCamera(960, 540, r=1, fovy=20, convention="opencv")`，单位旋转、看向原点。像素用 GaussianAvatars / 3DGS 的 `ndc2Pix`（像素 `i` 的中心在连续坐标 `i`）。带 FLAME `binding_0` 的 ply 里 `xyz` 是面片局部偏移；`(N,)` 在 `mask_avatar` 进程里、第一次 Foolbox `project` 之前，用已经加载的 `gaussians.get_xyz`（和光栅化器同一个均值）采样。这样 DC 残差和妆容掩码的形状对齐。

改完后在 AutoDL 上重跑这条命令装钩子（超参仍是原来的 B0-a：AdaFace，eps=0.1，300 steps，target-features DC，linfpgd），再到 `third_party/AEGIS` 下执行上面的 `mask_avatar.py`。同一条命令如果带上 `--launch`，会先装钩子再启动。`--launch` 只在 uv、CUDA、306 ply/png、AdaFace checkpoint、InsightFace `det_10g.onnx`、`GaussianAvatars` 都已经在盘上时才启动。缺任何一项就跳过，避免脚本自己去下载。没装钩子时进程仍是原文 Foolbox，CPU 冒烟里的投影才是支撑约束。不要在导师点头前开这次 GPU。

```bash
python scripts/nersemble306_makeup_mask_smoke.py --arm-hook \
  --parsing third_party/AEGIS/datasets/NeRSembleReconst/parsing/306.npy
```

## 实验矩阵（提交代码时对齐）
| ID | 含义 |
|----|------|
| B0 | AEGIS 原文去身份复现（正式主体 NeRSemble 306；B0-b / FaceScape 527 已取消） |
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

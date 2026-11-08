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
docs/                交付文档 + gpu_launch_checklist.md
assets/              MANIFEST.md（冻结名 ↔ AutoDL 路径）
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

正式基线是 **NeRSemble subject 306**。B0-a 已在该头像上跑通原文 AEGIS masking（AdaFace，`eps=0.1`，`target-features DC`，`linfpgd`，聚合余弦约 -0.2488），摘要在 `results/b0_aegis_sanity_summary.*`。这一刀只把妆容语义 mask 接到同一条 masking 路径上，检查残差满足 \(\mathrm{supp}(\delta)\subseteq M\) 且 \(\|\delta\|_\infty\le\varepsilon\)。它不是完整 B2（没有定向冒充，也没有论文评测）。定向冒充加上这张 mask 的命令在「B2」一节。

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

摘要写到 `results/nersemble306_makeup_mask_smoke.json` 和 `.csv`。`status=cpu_smoke_ok` 表示 CPU 投影满足支撑约束；`gpu_launched=false` 表示没有启动 `mask_avatar.py`。GPU 日志里若有 `Epsilon: ... -> Aggregated Cosine Similarity: ...`，该数从完整 stdout/stderr 写入 `gpu_aggregated_cosine_similarity`（JSON 另有逐 ε 的 `gpu_epsilon_cosine`）。`gpu_log_tail` 只是日志摘录。

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

B1 用同一套超参做全脸定向冒充，不装妆容钩子。第一刀照片是 `/root/autodl-fs/gs-assets/targets/celeba_T0.jpg`。文件不在盘上时不要加 `--launch`。完整说明在下面的「B1」一节。

```bash
python src/baselines/b1_targeted_impersonation.py \
  --impersonation-image /root/autodl-fs/gs-assets/targets/celeba_T0.jpg \
  --launch
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

## 新机

资产留在 `/root/autodl-fs/gs-assets/`，软链进工作区。冻结名、实验 ID 和路径以 `assets/MANIFEST.md` 为准；GPU 放行以 `docs/gpu_launch_checklist.md` 为准。本节不重列这两份文件。一次运行一个摘要槽，规则在 GPU 清单里。

### 必搬（真相源）

整份 `/root/autodl-fs/gs-assets/`：`weights/`、`avatars/`、`parsing/`（`parsing/306.npy`）、`targets/`，以及资产根里的 `MANIFEST.md`。`runs/<run_id>/` 在同一根下，整棵搬走时一起走。优先把同一块 autodl-fs 重新挂上。不能挂时，`rsync` 整棵目录，再按仓库 `assets/MANIFEST.md` 的 sha256 和 bytes 核对。

### 可重建（别当资产）

这些在新机上重做，不当作资产搬走：

- `autodl-tmp` 里的缓存。
- uv 下载缓存 `uv-cache`。它只是可选加速，不是真相源。wheelhouse 同样只作备用。
- 虚拟环境。不要把正在使用的 `.venv` 软链到另一台机器。同一代 CUDA 优先用**同一路径**的 venv tar（shebang 和 `pyvenv.cfg` 写的是绝对路径；#16 没有另给 tar 的文件路径）。否则按 lockfile 执行 `uv sync` 重建。AEGIS 仍在 `third_party/AEGIS` 里用 `uv pip install -e .`；FLAME 导入失败时再装 `chumpy`。

### 新机三步

1. 挂载同一块 autodl-fs，或 `rsync` `/root/autodl-fs/gs-assets/`，再按 `assets/MANIFEST.md` 核对 sha256 / bytes。挂上同一块盘就是快迁。只拷系统盘、不带这份资产根，不算迁完。
2. `git clone` 或 `git pull` 到 `master`，再准备环境：按 lockfile 执行 `uv sync`，或按上面的同一路径解开 venv tar。
3. `docs/gpu_launch_checklist.md` 全绿之前，不要加 `--launch`。

代码不要在 NFS 上多个人同时写同一份工作区。文件存储上只放 bare / 只读 mirror，或者 `rsync` / `clone` 到本机工作区再改。

#15 合并后，DC 冒烟要重新装钩子再启动（超参仍是原来的 B0-a）：

```bash
python scripts/nersemble306_makeup_mask_smoke.py --arm-hook \
  --parsing third_party/AEGIS/datasets/NeRSembleReconst/parsing/306.npy
```

同一条命令加上 `--launch` 会先装钩子再启动 `mask_avatar.py`。

## B1 全脸定向冒充（无妆容 mask）

B1 与上面的 B0-a 命令使用同一套攻击超参：AdaFace，`eps=0.1`，300 steps，`target-features DC`，`linfpgd`，半径 1，seed 42，相机边界 `-0.5 0.5 -0.5 0.5 0.0 0.0`，视角聚合 mean。源头像仍是 NeRSemble **306**（`datasets/NeRSembleReconst/avatars/306/point_cloud.ply`，核验参考 `renders/306.png`）。

上游 AEGIS（`e3d20445942cc3d41e912897675290e090f116f7`）没有定向冒充开关。`FaceRenderVerification._similarity_to_logits` 里第 0 列是「不匹配」（余弦下降时变大），第 1 列是「匹配」（余弦上升时变大）。`AEGIS.run` 把裸 `torch.tensor([1])` 交给 Foolbox，裸 tensor 会变成未定向 `Misclassification`，L∞ PGD 对交叉熵做梯度上升，于是渲染和参考嵌入 `e_r` 的余弦被往下推。CLI `--target-image` 只提供这个 `e_r`，是源头像的核验照片。只换照片、不改准则，优化方向仍是远离那张照片。

B1 在本仓做两件事，不改写整份 AEGIS：把 `e_r` 设成 T 的 AdaFace 嵌入；把 Foolbox 准则换成 `TargetedMisclassification`，目标类是匹配列（下标 1）。PGD 对这一类的交叉熵做梯度下降，最大化 `cos(adv, emb(T))`。扰动是全脸。这条路径不调用妆容 PGD 钩子，也不使用 `kind=dc` 的语义 mask `M`。worker 把 `GS_MAKEUP_HOOK_CONFIG` 指到一个不存在的路径，已经打过妆容补丁的树在 B1 进程里也不会套上 `M`。

T 是一张静止人脸照片，语料是 **CelebA 或 LFW**，不是第二套 3DGS。`/root/autodl-fs/gs-assets/avatars` 和 `NeRSembleReconst/avatars` 里只有主体 306。仓库不收录 CelebA / LFW，不把整库拷进来，也不下载数据集。盘上冻结的文件名是：

- 第一刀（B1-a，默认）：`/root/autodl-fs/gs-assets/targets/celeba_T0.jpg`
- 后一刀：`/root/autodl-fs/gs-assets/targets/lfw_T0.jpg`

命令和默认路径只用这两个名字。james 另外留了来源文件作对照，启动时不要改用别的文件名。需要链进工作区时，只链这一张样本（例如 `assets/targets/celeba_T0.jpg`），git 不跟踪该图片。环境变量 `B1_IMPERSONATION_IMAGE` 可以改默认路径。

照片不在盘上时不要加 `--launch`。下面这条只写 `results/b1_targeted_impersonation_summary.json` 和 `.csv`（`pipeline_ok` 为 false，`cos_adv_source` / `cos_adv_target` 为 null），不占 GPU：

```bash
python src/baselines/b1_targeted_impersonation.py
```

在仓库根目录执行。脚本进入 `third_party/AEGIS` 并用 `uv run`，与 B0 命令同一环境。缺 ply、权重、InsightFace 检测器或 CUDA 时不会自己下载，也不会启动：

```bash
python src/baselines/b1_targeted_impersonation.py \
  --impersonation-image /root/autodl-fs/gs-assets/targets/celeba_T0.jpg \
  --launch
```

LFW 是后一刀，等 celeba 这一刀有结果再开：

```bash
python src/baselines/b1_targeted_impersonation.py \
  --impersonation-image /root/autodl-fs/gs-assets/targets/lfw_T0.jpg \
  --launch
```

无 GPU 的接线检查：

```bash
python src/baselines/tests/test_b1_targeted_impersonation.py
```

摘要字段包括 `pipeline_ok`、`return_code`、`cos_adv_source`、`cos_adv_target`，以及能看到 GPU 时的显卡名。`makeup_hook_armed` 保持 false。

## B2 定向冒充 + 妆容 mask

B2 是 B1 的全脸定向冒充，再加上妆容语义 mask \(M\)。源头像仍是 NeRSemble **306**。攻击超参与 B1 / B0-a 相同：AdaFace，`eps=0.1`，300 steps，`target-features DC`，`linfpgd`，半径 1，seed 42。Foolbox 准则仍是匹配列（下标 1）的 `TargetedMisclassification`，最大化 `cos(adv, emb(T))`。不带这条命令时，B1 自己的行为不变：B1 仍拒绝 `--arm-hook`，妆容钩子保持关闭。

\(M\) 读盘上的 `/root/autodl-fs/gs-assets/parsing/306.npy`。目录是 `parsing/`，不是 `masks/`。钩子复用 `makeup/aegis_pgd_hook.py` 里已有的 DC 投影（`dc_hook_from_parsing`，`kind=dc`）。`(H, W)` 标签图仍是语义来源，按高斯均值采样成 `(N,)` 后再投影，使 \(\mathrm{supp}(\delta)\subseteq M\) 且 \(\|\delta\|_\infty\le 0.1\)。不使用 `documented_fallback` 标签。

T 与 B1 相同，仓库不收录整库：

- 第一刀：`/root/autodl-fs/gs-assets/targets/celeba_T0.jpg`
- 后一刀：`/root/autodl-fs/gs-assets/targets/lfw_T0.jpg`

一刀一个目录，文件名是 `summary.json` 和 `summary.csv`。不要写进 `results/` 根目录，也不要覆盖已有的扁平摘要（`b0_aegis_sanity_summary.*`、`b1_targeted_impersonation_summary.*`、`nersemble306_makeup_mask_smoke.*`）：

- `results/b2_celeba_T0/`
- `results/b2_lfw_T0/`

接线摘要里这几项保持空，不填 GPU 数字。跑完之后的门禁（bob 冻结，见 `docs/gpu_launch_checklist.md` 的 B2 一节）是：`pipeline_ok`，并且 cos(adv, source) ↓ relative to goal（`cos_adv_source`），cos(adv, T) ↑（`cos_adv_target`），以及 outside-M \(|\delta|=0\)（`outside_m_abs_delta`）。接线可以合并。GPU 仍要四道门都绿：先 1–3，再是导师的第 4 道。默认不加 `--launch`。

默认不加 `--launch`，也不装钩子，不授权 AutoDL。下面这条只写摘要：

```bash
python src/baselines/b2_makeup_impersonation.py
```

开关：

- `--arm-hook`：装上 `kind=dc` 钩子。单独使用不启动 GPU。解析文件不在盘上就退出，不造一张替代 mask。
- `--refuse-hook`：明确拒绝妆容钩子并退出。B2 不能当成没有 \(M\) 的 B1 来跑。
- `--arm-hook` 和 `--refuse-hook` 一起会被拒绝。
- `--launch` 必须和 `--arm-hook` 一起用。缺照片、缺 `parsing/306.npy`、缺 306 头像或 CUDA 时不启动。
- `--parsing` 如果落在 `masks/` 下，或文件名不是 `306.npy`，会被拒绝。
- `--adaptive-epsilon` 和 `--select-regions` 会被拒绝。前者不是这条 Foolbox 路径，后者是 FLAME 区域，不是 `from_parsing`。
- `--ablate`（同 `--mask-labels`）：这次运行的 in-M 消融。默认 `full`（省略，或写 `default`）仍用 `DEFAULT_PARSING_LABELS`。导师锁定的两刀互补：`ablate_noskin` 去掉标签 1，`ablate_onlyskin` 只留标签 1。`softskin` 不改标签集：硬妆 `{2, 3, 4, 5, 12, 13}` 仍是 `project_residual` 的 ε=0.1，皮肤标签 1 用更紧的 ε=0.07。这不改变妆容 M 的定义，也不把默认标签集扩大。默认命令仍然不加 `--launch`。

### in-M 消融（不改变妆容定义）

`makeup/mask.py` 的 `DEFAULT_PARSING_LABELS` 仍是产品默认 M：1（skin/cheek）、2/3（brow）、4/5（eye）、12/13（lip）。鼻子、头发、耳朵、脖子不进入这个默认。消融只在这一次 B2 里把 `label_map` 交给 `from_parsing`，不改那个常量，也不把结果写进 `results/b2_celeba_T0/`。

`--arm-hook` 在绑定了 FLAME 的 306 ply 上还不是高斯 mask。武装状态是 `dc_mask_lift=pending-get-xyz`，`mask_shape` 是图像 `(H, W)`。B2-a/b 记到的是 `[540, 960]`，图像域 M 大约占 10.22% 像素。`(N,)` 要等 launch 时 `get_xyz` 可用，`ensure_dc_mask` 才调用 `sample_image_mask`。那一步才把 `dc_mask_lift` 写成 `projected-means`，并打一行 `mask (N,) nz=… frac=…`。这行和同样的 `dc_mask_shape` / `dc_mask_nonzero` / `dc_mask_fraction` 会进 summary JSON。不要把武装状态说成已经是 `projected-means`，也不要用更早的 smoke 打印代替这次的 `(N,)` 计数。

| 模式 | 标签 | 摘要目录 |
| --- | --- | --- |
| `full`（默认，可省略；`default` 同义） | 上面的默认全集，ε=0.1 | `results/b2_celeba_T0/` |
| `ablate_noskin` | 去掉 1，只留 {2, 3, 4, 5, 12, 13} | `results/b2_celeba_T0_ablate_noskin/` |
| `ablate_onlyskin` | 只留 {1}，去掉唇、眼、眉 | `results/b2_celeba_T0_ablate_onlyskin/` |
| `softskin` | 默认全集不变。硬妆 {2, 3, 4, 5, 12, 13} ε=0.1；皮肤 1 ε=0.07 | `results/b2_celeba_T0_softskin/` |

`ablate_onlyskin` 是 `ablate_noskin` 的互补刀：两刀的标签并起来仍是默认 M，交集为空。LFW 后一刀对应 `results/b2_lfw_T0_ablate_noskin/`、`results/b2_lfw_T0_ablate_onlyskin/` 和 `results/b2_lfw_T0_softskin/`。`ablate_onlyskin` 不覆盖 `results/b2_celeba_T0/`、`results/b2_lfw_T0/`，也不覆盖 `*_ablate_noskin/`。`softskin` 只写这两个 `*_softskin/` 目录，拒绝产品目录，也拒绝 noskin / onlyskin 等其他消融槽。默认命令仍然不加 `--launch`。不把鼻子、头发、耳朵、脖子补进 M，也不改 `DEFAULT_PARSING_LABELS`。

图像域覆盖来自已经测过的 `(H, W)` 像素，不是高斯计数。默认 M 大约 10.22%；`ablate_noskin` 大约 0.7888%；因此 `ablate_onlyskin` 大约 9.43%（约 10.22 − 0.79）。`softskin` 的支撑宽度与默认 M 相同，变的只是皮肤幅度。`--arm-hook` 在绑定 ply 上仍是 `dc_mask_lift=pending-get-xyz`，这些百分比不是已经 lift 的 `(N,)`。

PLY `projected-means` 是暂定（provisional）数字，要等 launch 打出 `mask (N,) nz=… frac=…` 才写进 summary 的 `dc_mask_shape` / `dc_mask_nonzero` / `dc_mask_fraction` / `dc_mask_support_line`。在那一行出现之前不要把下面的计数当成这次运行的测量：full 大约 2.307%（2054/89021）；noskin 大约 0.2033%（181）；onlyskin 大约 2.10%（约 1873 = 2054 − 181）。`softskin` 不改这组支撑，所以不要另造一份 `(N,)` 计数。

### softskin（同一支撑，皮肤球更紧）

支撑集仍是 `DEFAULT_PARSING_LABELS` `{1, 2, 3, 4, 5, 12, 13}`。不新增 2D 标签。图像域 M 与 B2-a/b 相同（`[540, 960]`，约 10.22% 像素）。`(N,)` 仍只在 `ensure_dc_mask` 之后出现，摘要里的一行是 `mask (N,) nz=… frac=…`。武装状态保持 `dc_mask_lift=pending-get-xyz`。变的只是皮肤高斯上的幅度。

预算来自两篇论文，不改攻击超参本身（AEGIS 仍先按 ε=0.1 做 L∞）：

- AEGIS 区域自适应 ε：皮肤用 0.7×（[arXiv:2511.17747](https://arxiv.org/html/2511.17747)）。这里冻结为 ε_skin=0.07。
- Adv-Makeup 把攻击收在妆容 / 眶周（[arXiv:2105.03162](https://arxiv.org/abs/2105.03162)）。唇、眼、眉仍走原来的硬 `project_residual`，ε=0.1。

实现落在 `MakeupPgdHook._project_numpy`：它调用 `project_dual_budget_residual`。两条 AEGIS 循环都进这个方法。`adaptive_linf_pgd_attack` 经 `project_configured_delta`；`AEGIS._run_foolbox_attack` 经 `bind_foolbox_attack` / `reproject_foolbox`。B2 实际开跑的是 Foolbox `linfpgd` 那条。

softskin 的门禁是相对 B2-a、onlyskin、noskin 比较。计分要有 `pipeline_ok`、outside-M `|δ|=0`（`outside_m_abs_delta`）、`dc_mask_support_line`，以及硬区 / 软区 `|δ|`（`hard_abs_delta_{mean,p50,p95}` 与 `soft_abs_delta_{mean,p50,p95}`）。接线摘要里这些测量保持空。

无 GPU 的 softskin 接线（只写摘要，不装钩子，不开 GPU）：

```bash
python src/baselines/b2_makeup_impersonation.py --ablate softskin
```

清单放行之后才加 `--arm-hook --launch`。摘要进 `results/b2_celeba_T0_softskin/`，不覆盖 `results/b2_celeba_T0/`，也不写入 `results/b2_celeba_T0_ablate_noskin/` 或 `results/b2_celeba_T0_ablate_onlyskin/`：

```bash
python src/baselines/b2_makeup_impersonation.py \
  --impersonation-image /root/autodl-fs/gs-assets/targets/celeba_T0.jpg \
  --parsing /root/autodl-fs/gs-assets/parsing/306.npy \
  --ablate softskin \
  --arm-hook \
  --launch
```

无 GPU 的消融接线（只写摘要，不装钩子，不开 GPU）：

```bash
python src/baselines/b2_makeup_impersonation.py --ablate ablate_noskin
python src/baselines/b2_makeup_impersonation.py --ablate ablate_onlyskin
```

清单放行之后才加 `--arm-hook --launch`。`ablate_noskin` 的摘要进 `results/b2_celeba_T0_ablate_noskin/`，不覆盖 `results/b2_celeba_T0/`：

```bash
python src/baselines/b2_makeup_impersonation.py \
  --impersonation-image /root/autodl-fs/gs-assets/targets/celeba_T0.jpg \
  --parsing /root/autodl-fs/gs-assets/parsing/306.npy \
  --ablate ablate_noskin \
  --arm-hook \
  --launch
```

互补刀 `ablate_onlyskin` 同样先不加 `--launch`。放行之后摘要进 `results/b2_celeba_T0_ablate_onlyskin/`，不覆盖上面的 full 目录，也不覆盖 `results/b2_celeba_T0_ablate_noskin/`：

```bash
python src/baselines/b2_makeup_impersonation.py \
  --impersonation-image /root/autodl-fs/gs-assets/targets/celeba_T0.jpg \
  --parsing /root/autodl-fs/gs-assets/parsing/306.npy \
  --ablate ablate_onlyskin \
  --arm-hook \
  --launch
```

真正开 GPU 之前仍要过 `docs/gpu_launch_checklist.md`。在仓库根目录执行。脚本进入 `third_party/AEGIS` 并用 `uv run`，与 B1 同一环境：

```bash
python src/baselines/b2_makeup_impersonation.py \
  --impersonation-image /root/autodl-fs/gs-assets/targets/celeba_T0.jpg \
  --parsing /root/autodl-fs/gs-assets/parsing/306.npy \
  --arm-hook \
  --launch
```

LFW 是后一刀，摘要目录换成 `results/b2_lfw_T0/`：

```bash
python src/baselines/b2_makeup_impersonation.py \
  --impersonation-image /root/autodl-fs/gs-assets/targets/lfw_T0.jpg \
  --parsing /root/autodl-fs/gs-assets/parsing/306.npy \
  --arm-hook \
  --launch
```

无 GPU 的接线检查：

```bash
python src/baselines/tests/test_b2_makeup_impersonation.py
```

## 实验矩阵（提交代码时对齐）
| ID | 含义 |
|----|------|
| B0 | AEGIS 原文去身份复现（正式主体 NeRSemble 306；B0-b / FaceScape 527 已取消） |
| B1 | 同接口改定向冒充（全脸，CelebA/LFW 静止照片，无妆容 mask；第一刀 celeba_T0.jpg） |
| B2 | B1 + 妆容 mask \(M\)（`kind=dc`，`parsing/306.npy`）+ 幅度约束；摘要 `results/b2_<T>/`；in-M 消融 `ablate_noskin` 写 `results/b2_<T>_ablate_noskin/`，互补刀 `ablate_onlyskin`（只留标签 1）写 `results/b2_<T>_ablate_onlyskin/`；`softskin` 同一支撑、皮肤 ε=0.07，只写 `results/b2_<T>_softskin/`，门禁相对 B2-a / onlyskin / noskin；不改默认 M，也不互相覆盖；默认不 `--launch` |
| M  | + DPA + 预处理不变（主方法） |
| A1/A2/A3 | 消融 DPA / 预处理 / univ |
| C1 | 中期最优管线 |

## 协作约定
1. 默认分支 `master`；功能分支 `feat/*`、`exp/*`。
2. 一人一 PR。一次运行只写 `results/<exp_id>/summary.{json,csv}` 和同目录日志，不覆盖已有摘要；机器侧副本在 `/root/autodl-fs/gs-assets/runs/<run_id>/`。大结果与权重放资产根或网盘，git 只跟踪小表 CSV/JSON。
3. API 密钥用环境变量，禁止提交。
4. 研究表述只改 `docs/创新章_交付文档_AEGIS底座.md`。资产清单是 `assets/MANIFEST.md`，GPU 门禁是 `docs/gpu_launch_checklist.md`。

## 本地路径
本机共享盘：`/workspace/gs-local-makeup-transfer`

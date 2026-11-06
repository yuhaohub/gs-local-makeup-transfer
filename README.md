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

`makeup/` 用 NumPy 实现语义 mask 的 union/support、残差投影（\(\mathrm{supp}(\delta)\subseteq M\)、\(\|\delta\|_\infty\le\varepsilon\)）与平滑项；`from_parsing`、调色板与 LPIPS 仍未实现（见 `makeup/README.md`）。

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

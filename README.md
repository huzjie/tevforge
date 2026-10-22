# tevforge

**Long-horizon agent dense-reward RL training framework** — a zero-dependency
Python reference implementation of the UCLA x Adobe Research **Tev** method
(arXiv 2610.18868): answer-world-first environment synthesis (Tev-Gym) +
Progress Potential Reward Shaping (PPRS) + long-horizon GRPO.

[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Zero deps](https://img.shields.io/badge/dependencies-zero-brightgreen.svg)]()

---

## 这个热点背后的技术原理是什么

长程智能体（连续工作数小时到数天）正成为大模型竞争焦点，但开源模型在长程
任务上明显落后。UCLA x Adobe 的 Tev 团队定位到**两道坎**：

1. **没有训练数据**——现有智能体语料（SWE-Smith、TMAX 等）每个任务只是一次
   PR 修复或几十秒终端会话，前沿智能体跑一次的中位耗时不到 2 分钟；真正的
   长程任务只存在于人工评测基准里（每道题专家要写数周，且只用于评测）。
2. **奖励稀疏**——奖励只在任务提交时出现。GRPO 在同一任务的 G 条 rollout 里
   归一化奖励，一旦整组全失败，优势值全为零，这组样本就白跑了。解题概率为 p
   时，被浪费的组占比 `(1-p)^G`：p=0.05、G=8 时 66%，p=0.01 时高达 92%，
   而且恰恰集中在训练初期——策略最弱、最需要信号的时候。

Tev 的解法：先造好「答案世界」再出长程考题（Tev-Gym，1170 环境/7 领域），并
用**进度势能奖励塑形 PPRS** 把稀疏最终奖励改造成稠密渐进式势能奖励。tevforge
把这两条思路做成了一个可复用的零依赖参考实现。

## 技术拆解（四块可独立替换的工程件）

| 模块 | 解决什么问题 | 关键技术 |
|------|--------------|----------|
| `world` | 长程任务要有「可验证答案」 | 确定性答案世界：ground-truth 动作 + progress + 精确 `is_complete` |
| `synth` | 没有长程训练数据 | Tev-Gym 答案世界优先合成 + 稠密奖励预标注 |
| `reward` | 奖励稀疏导致组内优势为零 | PPRS 势能塑形 `r' = r + γΦ(s') − Φ(s)`（策略不变） |
| `trainer` | GRPO 整组失败零梯度 | 稠密 return 上的组优势估计 + 浪费组检测 |

### PPRS 的核心数学

```
r' = r_env + gamma * Phi(s') - Phi(s)
```

Phi 是进度势能（已完成正确步骤占比）。因为它是势能差（可望远镜求和），塑形后
的 MDP 与原始 MDP **最优策略相同**（策略不变性）。当 `gamma = 1` 且
`Phi(empty) = 0` 时，塑形 return 精确望远镜求和为：

```
shaped_return = sparse_return + Phi(s_T)
```

即最终进度总会被计入——即使任务没完成，有进度就有奖励。

---

## 快速开始（零依赖，Python 3.8+）

```bash
python -m tevforge doctor     # 冒烟自检
python -m tevforge bench      # 跑基准
python -m tevforge train --steps 40 --tasks 8
python -m tevforge serve      # stdlib HTTP 服务（:8090）
```

```python
from tevforge.config import Config
from tevforge.engine.train_engine import TrainEngine

engine = TrainEngine(Config())
result = engine.fit(n_tasks=8, max_steps=40)
print(result["final_skill"])   # 0.5 -> ~1.0，完成率单调爬升
```

---

## 基准（真实测量，见 `python -m tevforge bench`）

skill 扫描（completion = s^L，progress = s，单调）：

| skill | completion_rate | progress_rate |
|-------|-----------------|---------------|
| 0.5   | 0.002 | 0.505 |
| 0.75  | 0.021 | 0.756 |
| 1.0   | 1.000 | 1.000 |

稀疏 vs 稠密（PPRS）在训练起点 skill=0.5 的对比——这是本框架的核心卖点：

| 指标 | sparse | dense (PPRS) |
|------|--------|--------------|
| wasted_group_rate | 1.00 | 0.00 |
| advantage_density | 0.00 | 0.96 |
| reward_sparsity（零奖励步占比） | 1.00 | 0.51 |

> 训练起点（skill=0.5，随机游走）下，稀疏 GRPO 几乎 100% 的组被浪费、几乎
> 没有可用梯度；PPRS 稠密奖励把浪费组压到接近 0、可用梯度密度接近 1。

---

## 可复用的工程技巧（干货块）

1. **答案世界优先，别先写 rollout**。先让任务「有可验证的答案」，再谈训练；
   完成判定必须是确定性函数，绝不用 LLM 当裁判（奖励可被 hack）。
2. **势能塑形保最优策略**。只要塑形项是 `γΦ(s')−Φ(s)` 的势能差，就不会改
   变最优策略；把 `γ=1`、`Φ(初态)=0` 记死，return 就能望远镜求和成
   `sparse + progress`，可被单测严格断言。
3. **单调训练信号，别用「对+1 错−1」**。`skill += lr·(1−skill)` 几何爬升，
   低起点不会被随机游走压到 0；mock 的 `boost=(skill−0.5)·2` 让 skill=0.5
   恰为随机基线、skill=1.0 恰为完美策略。
4. **单一 oracle**。mock 策略、答案世界、所有基准共享同一个真值来源，所以
   skill=1.0 时完成率精确=1.0，训练效果可被断言而非目测。
5. **声明规模与实例化解耦**。27B/320B 只作配置声明；参考实现只实例化极小
   结构，避免纯 Python 大张量卡死。
6. **确定性随机**：`md5(key)` 前 8 字节做种子驱动 `random.Random`，任何输出
   可复现、可回归。

---

## 目录结构

```
tevforge/
  world/       答案世界（真值 + progress + 完成判定）
  synth/       Tev-Gym 环境合成 + 稠密奖励标注
  reward/      PPRS / sparse / dense 奖励
  agent/       策略 + 长程 rollout
  trainer/     GRPO + 组优势 + 浪费组检测
  model/       5 后端（mock/cpu/openai/vllm/transformers）
  engine/      一键训练/评估引擎
  bench/       5 项基准
  serving/     CLI + stdlib HTTP
  configs/     YAML 配置（含 yamlish 零依赖回退）
tests/         7 个测试模块
examples/      7 个可运行示例
docs/          9 篇文档（架构/PPRS/Tev-Gym/GRPO…）
prompts/       8 套提示词/工作流模板
recipes/       6 套训练配方
deploy/        Docker / K8s / Helm + GitHub Actions CI
```

## License

MIT

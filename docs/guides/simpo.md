# SimPO（无参考奖励偏好优化）

阶段十一 **11.4「无参考对齐」** 的实现：**SimPO**（Meng et al., 2024,
*SimPO: Simple Preference Optimization with a Reference-Free Reward*，
arXiv:2405.14734）。与 DPO 不同，SimPO **不冻结参考模型**，直接以策略自身的
**长度归一化平均 log-prob** 作为隐式奖励，因此更省显存/算力、也没有参考模型
漂移/续训时参考快照不一致的问题（对比 DPO 需要持久化冻结参考，见
`docs/guides/dpo_vs_ppo.md` 与 RIL 第 60 轮结论）。

## 公式

\[
r(x,y) = \beta \cdot \frac{1}{|y|} \sum_t \log \pi_\theta(y_t \mid x, y_{<t})
\]

\[
\mathcal{L} = -\log\sigma\big( r(x,y_w) - r(x,y_l) - \gamma \big)

- \lambda \cdot \frac{1}{|y_w|}\sum_t \log \pi_\theta(y_{w,t} \mid x, y_{w,<t})
\]
- **奖励项**：pushed 与 rejected 响应平均 log-prob 的差距，经 `gamma` 抬升目标间隔后过卷 Logistic。
- **SFT 正则项**：`-lambda * mean_logp(chosen)`，鼓励策略保持对 chosen 的生成质量。
  注意 `beta` 只作用于奖励项，不乘在 SFT 项上（配置 docstring 已按此校正）。

## 用法

`--task simpo`（`SimPOTask` + `DPODataModule`，标准训练循环）。对齐旋钮走 YAML
配置（`TrainingConfig`），与 DPO/GRPO 一致：

```yaml
model:
  hidden_size: 128
  num_layers: 2
training:
  batch_size: 2
  epochs: 1
  simpo_beta: 2.0     # 隐式奖励缩放（奖励项梯度强度），文献标准值 2.0
  simpo_gamma: 0.5    # 想要的最低 chosen-rejected 奖励间隔
  simpo_lambda: 1.0   # chosen 响应 SFT 正则权重 (-lambda * mean_logp)，0 关闭
data:
  dataset_path: data/prefs.jsonl
  max_seq_len: 1024
```

```bash
uv run llm-train --task simpo --config-path your_simpo.yaml
```

配置旋钮（均可省略走默认值）：

| 字段           | 默认  | 说明                                                              |
| -------------- | ----- | ----------------------------------------------------------------- |
| `simpo_beta`   | `2.0` | 隐式奖励缩放（奖励项梯度强度），文献标准值 2.0                    |
| `simpo_gamma`  | `0.0` | 想要的最低 chosen−rejected 奖励间隔，过小样本上可上调             |
| `simpo_lambda` | `1.0` | chosen 响应 SFT 正则权重（`-lambda * mean_logp`），`0` 关闭正则项 |

## 数据格式

复用 `DPODataModule` 的 JSONL（每行一条）：

```json
{"prompt": "Q", "chosen": "好回答", "rejected": "差回答"}
```

与 DPO 完全一致：prompt+completion 拼接、completion 部分才计入
log-prob（prompt 及 padding 标签为 `-100`）。prompt 单独超过 `max_seq_len`
的行会被丢弃并告警（空偏好信号）。

## 与 DPO 的差异

|          | DPO                                        | SimPO                                |
| -------- | ------------------------------------------ | ------------------------------------ |
| 参考模型 | 冻结初始策略，checkpoint 需持久化          | 无（reference-free）                 |
| 奖励形式 | `β·log π(y_w) − β·log π_ref(y_w)` 相对参考 | `β·mean_logp` 长度归一化、与参考无关 |
| 长度偏置 | 短响应可能占优                             | 长度归一化显式抑制                   |
| 续训     | 需 `on_checkpoint_loaded` 恢复原始参考     | 无此复杂度                           |

## 验证

CPU 上即可端到端验证（`tests/e2e/test_simpo.py`）：合成 `prompt/chosen/rejected`
对，跑 `--task simpo` 一个 epoch，检查 `global_step > 0` 且
`reward_acc`（chosen 隐式奖励 > rejected 的比例）随训练上升。任务指标：
`loss`、`reward_chosen`、`reward_rejected`、`reward_acc`、`reward_margin`。

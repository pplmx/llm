# Constitutional AI（规则自批判 → 改写切片）

Constitutional AI（Bai et al. 2022）用一种 **constitution（原则集）** 外加 **自批判 →
自改写** 循环，让模型自己的输出更好地遵守这些原则。ROADMAP 阶段十一 11.4 的落地：本切片
把它做成一个 **CPU 可验证、基于规则** 的核心（真实批判模型的占位，类比
`TargetTokenJudge` 之于真实偏好 judge）。

代码在 `llm/training/rlhf/constitutional.py`。

## 概念

- **Principle（原则）**：一条宪法规矩，看一个 response 是否满足。
    - `ForbiddenToken({7})`——不得包含 token 7（安全/毒化 token 回避）。
    - `EndsWithToken(1)`——必须以 token 1 结尾（格式/结构约束）。
- **Constitution**：一组原则 + 打分。`score(response)` = 满足的原则占比（∈[0,1]）。
- **critique(response)**：报告违反了哪些原则（自批判文本）。
- **revise(response, safe_token=0)**：确定性改写——把违规 token 替换为安全 token、
  强制结尾 token 等，使改写后的响应满足 constitution。

## 用法

```python
from llm.training.rlhf.constitutional import (
    Constitution, ForbiddenToken, EndsWithToken, constitutional_loop,
)

constitution = Constitution([ForbiddenToken({7}), EndsWithToken(1)])
result = constitutional_loop(responses, constitution, safe_token=0)
# result["scores_before"] / ["scores_after"] / ["critiques"] / ["revisions"]
```

## CPU 观测结论

在合成 token 序列上，随机生成的 response 初始常只有较低的比例满足 constitution（很多
含 forbidden token、不以目标结尾）。经过 `critique -> revise` 后，改写版 **全部满足**
constitution（score=1.0），对每个原本违规的 response 分数严格提升——这正是
"批判 → 改写 → 合规提升" 的 CPU 可复现信号。真正用 LLM 做批判/改写、以及把合规提升
作为奖励训练策略（constitutional RL）是后续可以挂载的层，本切片提供其可度量的规则内核。

## 测试

`tests/training/test_constitutional.py` 覆盖：forbidden/end-with 原则判定；constitution
打分与违规列表；`critique` 文本；`revise` 的确定性改写与合规性；e2e（改写版全合规且分数
严格上升）。

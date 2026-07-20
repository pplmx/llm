# 推理教程

训练完成后如何使用 `llm-serve` 把模型部署为 OpenAI 兼容的 HTTP 服务。和 `01-pretraining.md` / `02-finetuning.md` 一样，所有命令都对齐 `llm-serve` CLI + YAML 配置 —— 而不是单独的 demo 脚本。

---

## 概述

本教程涵盖：

- **基础部署**：`llm-serve` 启动 + OpenAI 兼容 API（`/v1/chat/completions` + `/generate`）
- **主流路径**：`llm-serve` + `configs/serve_local_demo.yaml` + `configs/serve_pretrained.yaml`
- **训练 → 服务闭环**：PEFT 适配器（LoRA / IA³ / BitFit / Adapter / Pfeiffer / AdaLoRA）通过 `LLM_SERVING_PEFT_*` env vars 直接挂载
- **认证 + 安全**：API key（HMAC-SHA256 timing-safe）+ 公开主机守卫（拒绝无 key 绑定 0.0.0.0）
- **可观测性**：Prometheus 域内指标（tokens / batch fill ratio / KV cache hit / 延迟）+ 结构化 JSON 日志
- **生产化**：性能开关（paged attention / prefix cache / torch.compile / batched backend）+ 高并发 semaphore

> **前提**：上游已经有训练好的 checkpoint（参考 [01-pretraining.md](./01-pretraining.md) 或 [02-finetuning.md](./02-finetuning.md)）。`llm-serve` 不需要 checkpoint —— 它能从 `ServingConfig` 的 arch 字段构建一个 dummy 模型用于冒烟测试。

---

## 1. 30 秒上手：`llm-serve` smoke test

最简部署：本地回环接口、不需要 checkpoint、不需要 API key。先确认 HTTP 通路本身工作。

### 1.1 启动服务（无 checkpoint）

```bash
# 启动 llm-serve：不传 model_path 时，loader 从 arch 字段构建 dummy 模型
uv run llm-serve
```

启动日志包含一条结构化 `server_config` 行（机器可读的运维元数据 —— model class / 参数量 / dtype / device / attention impl / api_key_set 等）：

```json
{"event": "server_config", "model_class": "DecoderModel", "param_count_total": 113125,
 "param_count_trainable": 113125, "dtype": "torch.float32", "device": "cpu",
 "max_seq_len": 128, "attn_impl": "mha", "mlp_impl": "mlp",
 "generation_backend": "eager", "enable_prefix_cache": false,
 "use_paged_attention": false, "api_key_set": false}
```

服务监听 `http://127.0.0.1:8000`。

### 1.2 用 curl 探活

```bash
curl http://127.0.0.1:8000/health
# → {"status":"ok"}
```

### 1.3 调 OpenAI 兼容 chat completions

```bash
curl http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
        "model": "llm",
        "messages": [{"role": "user", "content": "hello"}],
        "max_tokens": 10
      }'
```

返回 OpenAI 兼容 envelope：

```json
{"id": "chatcmpl-6c10da385890", "object": "chat.completion", "created": 1753040015,
 "model": "llm", "choices": [{"index": 0, "message": {"role": "assistant", "content": "xxx"},
 "finish_reason": "stop"}], "usage": {"prompt_tokens": 23, "completion_tokens": 4, "total_tokens": 27}}
```

> **小贴士**：因为是 dummy 模型（未训练），输出是随机 token 解码出来的字符串。要看到有意义的输出，请先训练一个 checkpoint，然后跳到 §3。

### 1.4 查 Prometheus 指标

```bash
curl http://127.0.0.1:8000/metrics
```

暴露的域内指标（T2 #22）：

| 指标 | 类型 | 含义 |
|------|------|------|
| `llm_tokens_generated_total{endpoint}` | Counter | 每个 endpoint 累计生成的 token 数 |
| `llm_tokens_per_request{endpoint}` | Histogram | 单次请求的 token 数（buckets: 16/64/256/1024/4096） |
| `llm_request_duration_seconds{endpoint,status}` | Histogram | 单次请求延迟（buckets: 0.05/0.25/1/5/30 秒） |
| `llm_batch_fill_ratio` | Gauge | `ContinuousBatchingEngine.step()` 的批填充率（0-1） |
| `llm_kv_cache_hit_ratio` | Gauge | KV cache 命中率（0-1） |
| `llm_inflight_requests` | Gauge | 当前 in-flight 请求数 |

加上 `prometheus-fastapi-instrumentator` 自带的 HTTP RED 指标（请求率 / 错误率 / 延迟 per-route）。

### 1.5 关掉服务

`Ctrl+C` 即可。日志会打印 `Shutting down...`，`engine.unload_model()` 释放 GPU/CPU 内存。

---

## 2. 主流路径：`llm-serve` + YAML

### 2.1 配置风格：env vars vs YAML

两种风格等价：

| 风格 | 例子 | 适用场景 |
|------|------|----------|
| **env vars** | `LLM_SERVING_API_KEY=... uv run llm-serve` | container / docker-compose / Kubernetes ConfigMap |
| **YAML** | `from llm.serving.config import ServingConfig; cfg = ServingConfig.from_yaml("configs/serve_pretrained.yaml")` | 配置文件版本化、与训练配置对称 |

`configs/serve_local_demo.yaml` 是一个完整的 env-vars-as-YAML 例子。YAML 字段名对应 env vars 去掉 `LLM_SERVING_` 前缀（如 YAML 的 `api_key` ↔ env var 的 `LLM_SERVING_API_KEY`）。

### 2.2 Smoke config：`configs/serve_local_demo.yaml`

最小可行配置 —— 无 checkpoint、CPU 跑、回环 host、无认证。适合 CI 冒烟测试或本地 API 探索。

```bash
# 启动（用 YAML 的所有默认 + 可能的 env var 覆盖）
LLM_SERVING_HOST=127.0.0.1 uv run llm-serve
```

### 2.3 生产 config：`configs/serve_pretrained.yaml`

完整生产预设 —— 训练 checkpoint + HF tokenizer + PEFT 适配器 + 公开主机 + paged attention + prefix cache + torch.compile。

```bash
# 1. 先训练（参考 02-finetuning.md，用 LoRA 写 sidecar）
uv run llm-train sft --config configs/sft_alpaca.yaml \
  --peft-method lora \
  --peft-kwargs '{"rank": 8, "alpha": 16.0}' \
  --peft-save-path checkpoints_sft_alpaca/peft_adapter_lora.bin

# 2. 用 production preset 部署
LLM_SERVING_API_KEY=$(openssl rand -hex 32) \
LLM_SERVING_HOST=0.0.0.0 \
uv run llm-serve
```

`configs/serve_pretrained.yaml` 在 `model_path` / `tokenizer_path` / `peft_*` 字段已经填好了默认值，env vars 只覆盖需要按机器调整的字段（`api_key` / `host`）。

### 2.4 配置结构详解

`ServingConfig` 字段分组：

| 组 | 字段 | 说明 |
|----|------|------|
| **模型 checkpoint** | `model_path`, `tokenizer_path`, `tokenizer_type` | `model_path=None` → dummy 模型（smoke test） |
| **架构（dummy 用）** | `hidden_size`, `num_layers`, `num_heads`, `max_seq_len`, `num_kv_heads`, `num_experts`, `top_k`, `attn_impl`, `mlp_impl` | 加载 checkpoint 时被覆盖 |
| **安全** | `host`, `api_key`, `log_level` | 公开主机守卫：`api_key=None` + `host` 非回环 → 启动失败 |
| **生成** | `generation_backend` (`eager` / `batched`), `compile_model` | `batched` = `ContinuousBatchingEngine`（高并发推荐） |
| **并发** | `max_concurrent_requests`, `request_timeout` | semaphore 上限 + 单请求超时 |
| **KV cache** | `use_paged_attention`, `max_blocks`, `block_size`, `enable_prefix_cache`, `max_prefixes` | paged attention 节省显存 / prefix cache 摊销 system prompt |
| **Chat template** | `chat_message_template`, `chat_generation_prefix` | OpenAI `/v1/chat/completions` 的消息渲染模板 |
| **PEFT（T2 PEFT #49）** | `peft_method`, `peft_kwargs`, `peft_adapter_path`, `peft_merge` | 详见 §3 |

---

## 3. 训练 → 服务闭环：PEFT 适配器挂载

> 这是 Main Path #3 的核心 user-facing 故事：训练完一个 LoRA adapter，几行配置就能直接挂到生产服务上。

### 3.1 训练时：写 adapter sidecar

```bash
# SFT + LoRA：训练完自动写 peft_adapter_lora.bin
uv run llm-train sft --config configs/sft_alpaca.yaml \
  --peft-method lora \
  --peft-kwargs '{"rank": 8, "alpha": 16.0}' \
  --peft-save-path checkpoints_sft_alpaca/peft_adapter_lora.bin
```

`PEFTAdapterCheckpointCallback`（T2 PEFT #48）在 `on_train_end` 时调 `save_peft` 写 sidecar，envelope 包含 `format_version` + `method_name` + `state_dict` + `peft_kwargs`。文件 ~MB 级（远小于 base checkpoint）。

### 3.2 服务时：env vars 挂载 adapter

```bash
LLM_SERVING_MODEL_PATH=checkpoints_sft_alpaca/epoch_5.pt \
LLM_SERVING_TOKENIZER_PATH=tokenizer \
LLM_SERVING_TOKENIZER_TYPE=hf \
LLM_SERVING_PEFT_METHOD=lora \
LLM_SERVING_PEFT_ADAPTER_PATH=checkpoints_sft_alpaca/peft_adapter_lora.bin \
LLM_SERVING_PEFT_KWARGS='{"rank": 8, "alpha": 16.0}' \
LLM_SERVING_API_KEY=$(openssl rand -hex 32) \
uv run llm-serve
```

加载顺序（`load_model_and_tokenizer` → `_apply_peft_if_configured`）：

1. **base checkpoint**：从 `model_path` 加载训练权重
2. **apply method**：从 `peft_method` 通过 `PEFT_REGISTRY` dispatch 到 `apply_peft`（创建 wrapper）
3. **load sidecar**：从 `peft_adapter_path` 调 `load_peft` 把训练好的 adapter 参数 copy 进 wrapper

任何一步失败 → 启动失败，绝不 silently serve un-adapted base。

### 3.3 切换 PEFT 方法

8 种内置方法都可以走这套工作流：

```bash
# LoRA
LLM_SERVING_PEFT_METHOD=lora LLM_SERVING_PEFT_ADAPTER_PATH=.../peft_adapter_lora.bin

# IA³
LLM_SERVING_PEFT_METHOD=ia3 LLM_SERVING_PEFT_ADAPTER_PATH=.../peft_adapter_ia3.bin

# BitFit（bias-only；不需要 wrapper，但 method 必须设，否则 base weights 被 frozen 后无法推理）
LLM_SERVING_PEFT_METHOD=bitfit

# Adapter (Houlsby) / Pfeiffer Adapter
LLM_SERVING_PEFT_METHOD=adapter LLM_SERVING_PEFT_ADAPTER_PATH=.../peft_adapter_adapter.bin LLM_SERVING_PEFT_KWARGS='{"bottleneck_dim": 64}'
LLM_SERVING_PEFT_METHOD=pfeiffer_adapter LLM_SERVING_PEFT_ADAPTER_PATH=.../peft_adapter_pfeiffer_adapter.bin LLM_SERVING_PEFT_KWARGS='{"bottleneck_dim": 64}'
```

### 3.4 合并 adapter 进 base weights（`peft_merge=true`）

需要极限吞吐时，把 adapter fold 进 base weights：

```bash
LLM_SERVING_PEFT_MERGE=true
```

- **优点**：消除每 token 的 routing 开销，推理更快
- **缺点**：失去 `disable_peft` / `unmerge_peft` 运行时切换能力
- **不适用**：`bitfit` / `qlora` / `prefix_tuning`（这三个方法没有 merge helper —— 启动时 raise `ValueError`，不会 silently no-op）

---

## 4. 认证 + 安全

### 4.1 API key（HMAC-SHA256 timing-safe）

```bash
# 生成随机 key（32 字节十六进制 = 64 字符）
LLM_SERVING_API_KEY=$(openssl rand -hex 32) uv run llm-serve
```

请求时二选一 header：

```bash
curl -H "X-API-Key: $KEY" http://server:8000/v1/chat/completions -d '...'

# 或
curl -H "Authorization: Bearer $KEY" http://server:8000/v1/chat/completions -d '...'
```

错误处理：key 错误 → `403` + `{"code": "unauthorized", "message": "Could not validate credentials"}`。比较走 `hmac.compare_digest`，不会因响应时间泄漏 key 字节（T2 #2 hardening）。

### 4.2 公开主机守卫

`llm-serve` 启动时会检查 host：

```bash
# OK：回环地址 + 无 key
LLM_SERVING_HOST=127.0.0.1 uv run llm-serve

# OK：非回环 + 有 key
LLM_SERVING_HOST=0.0.0.0 LLM_SERVING_API_KEY=... uv run llm-serve

# 启动失败：非回环 + 无 key（防止匿名暴露到公网）
LLM_SERVING_HOST=0.0.0.0 uv run llm-serve
# → RuntimeError: Refusing to start: ServingConfig.host='0.0.0.0' binds to a
#   non-loopback address but api_key is not set. Anonymous access on a public
#   interface is unsafe. ...
```

T2 #7 hardening —— 失败在启动阶段，不是在第一次请求时 silently。

### 4.3 OpenAI Python SDK 直连

`llm-serve` 是 OpenAI 兼容的，所以官方 SDK 直接能用：

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://127.0.0.1:8000/v1",
    api_key="dummy",  # 不需要 key（loopback）或设成 LLM_SERVING_API_KEY 的值
)

response = client.chat.completions.create(
    model="llm",
    messages=[{"role": "user", "content": "hello"}],
    max_tokens=20,
)
print(response.choices[0].message.content)
```

LangChain / LlamaIndex / vLLM 客户端同样适用（任何走 OpenAI HTTP 协议的框架都行）。

---

## 5. 监控

### 5.1 Prometheus 抓取

`/metrics` 是标准 Prometheus 暴露端点。在 `prometheus.yml` 加：

```yaml
scrape_configs:
  - job_name: 'llm-serve'
    scrape_interval: 15s
    static_configs:
      - targets: ['llm-serve:8000']
```

### 5.2 常用 PromQL 查询

```promql
# p95 chat completion 延迟（过去 5 分钟）
histogram_quantile(0.95,
  rate(llm_request_duration_seconds_bucket{endpoint="chat_completions"}[5m])
)

# tokens/s 吞吐
rate(llm_tokens_generated_total{endpoint=~"chat_completions|generate"}[1m])

# 平均 batch fill ratio（理想值 0.6-0.9）
avg_over_time(llm_batch_fill_ratio[5m])

# KV cache 命中率（prefix cache 是否生效）
avg_over_time(llm_kv_cache_hit_ratio[5m])

# in-flight 请求数（接近 max_concurrent_requests 表示打满）
llm_inflight_requests

# 错误率
rate(llm_request_duration_seconds_count{status!="200"}[5m])
  / rate(llm_request_duration_seconds_count[5m])
```

### 5.3 结构化 JSON 日志

所有 log 行是 JSON：

```json
{"asctime": "2026-07-20 20:13:35,752", "levelname": "INFO", "name": "root",
 "message": "server_config", "event": "server_config",
 "model_class": "DecoderModel", "param_count_total": 113125, ...}
```

每条请求都带 `request_id`（`X-Request-ID` middleware 生成）：

```json
{"event": "request", "request_id": "c56d4d60c6244dddae29843b091da3bf",
 "method": "POST", "path": "/v1/chat/completions", "status": 200, "duration_ms": 41.49}
```

客户端可传 `X-Request-ID` header 自定义 ID，便于端到端 trace 串联。

---

## 6. 故障排除

### 6.1 CUDA OOM（最常见）

```
torch.cuda.OutOfMemoryError: CUDA out of memory.
```

优先级从低到高的修复：

1. **缩 `max_concurrent_requests`**：semaphore 上限降低，引擎并发压力减小
2. **开 paged attention**：`LLM_SERVING_USE_PAGED_ATTENTION=true` —— block allocator 把 KV cache 切成小块（见 ADR-004）
3. **关 `compile_model`**：torch.compile 在首次推理时分配 graph memory
4. **用更小的 base 模型 / 量化 checkpoint**
5. **CPU 跑**：`LLM_SERVING_DEVICE=cpu` —— 仅限冒烟测试，生产不可用

### 6.2 PEFT 加载失败

```
FileNotFoundError: peft_adapter_path='...' does not exist
ValueError: method_name mismatch: sidecar saved as 'lora', requested 'ia3'
ValueError: Unknown PEFT method 'loraa'. Registered methods: ['lora', 'qlora', ...]
```

- 路径错：检查 `--peft-save-path` 和 `LLM_SERVING_PEFT_ADAPTER_PATH` 是否一致
- 方法不匹配：训练时用的 `peft_method` 和服务时用的 `peft_method` 必须相同
- 拼写错：`PEFT_REGISTRY.names()` 列出所有可用方法

### 6.3 启动失败：公开主机无 key

```
RuntimeError: Refusing to start: ServingConfig.host='0.0.0.0' binds to a
non-loopback address but api_key is not set.
```

显式设 `LLM_SERVING_API_KEY=<32 字节随机>`，或回 `LLM_SERVING_HOST=127.0.0.1`。

### 6.4 chat 输出乱码

dummy 模型（`model_path=None`）输出的是随机 token 解码字符串。要有意义输出：

1. 训练一个 checkpoint（参考 01 / 02）
2. 配 `model_path` + `tokenizer_path` + （如果用了 PEFT）`peft_*` 字段
3. 重启服务

### 6.5 请求超时

```json
{"code": "timeout", "message": "Request timeout"}
```

`request_timeout` 默认 60 秒。`max_new_tokens=4096` + 长 prompt + 慢 GPU 可能超过：

```bash
LLM_SERVING_REQUEST_TIMEOUT=300 uv run llm-serve
```

### 6.6 `batch_engine` 看不到 `llm_batch_fill_ratio` 变化

`llm_batch_fill_ratio` 是 `ContinuousBatchingEngine.step()` 的回报，需要：

- `LLM_SERVING_GENERATION_BACKEND=batched`（不是 `eager`）
- 有真正的并发请求（一个接一个串行调用不会触发 batch fill）

---

## 7. 从教程到生产

### 7.1 Checkpoint 路径

| 来源 | 命令 | 怎么 serve |
|------|------|------------|
| `stream_lm` 预训练 | `uv run llm-train stream_lm --config configs/streaming_c4.yaml` | 设 `LLM_SERVING_MODEL_PATH=<checkpoint_dir>/epoch_N.pt` |
| `sft` 微调 | `uv run llm-train sft --config configs/sft_alpaca.yaml` | 同上（base weights serve） |
| `sft` + LoRA | 加 `--peft-method lora --peft-kwargs '{...}' --peft-save-path <path>` | 同时设 `LLM_SERVING_PEFT_*` 字段 |
| `dpo` | `uv run llm-train dpo --config configs/dpo_ultrafeedback.yaml` | 同 sft |

### 7.2 性能开关（按场景）

| 场景 | 推荐配置 |
|------|----------|
| **低延迟 / 单请求** | `generation_backend=eager`，`compile_model=true`，`max_concurrent_requests=1` |
| **高吞吐 / 多并发** | `generation_backend=batched`，`use_paged_attention=true`，`max_concurrent_requests=16` |
| **长 system prompt 的多轮 chat** | `enable_prefix_cache=true`，`max_prefixes=32`（摊销 system prompt） |
| **极限吞吐 / 不需要 swap adapter** | `peft_merge=true` |
| **多卡 / 多模型** | 每个 GPU 起一个 `llm-serve` 实例，前面挂 nginx / Envoy |

### 7.3 Docker

最小 Dockerfile 思路（参考项目自带 `compose.yml`）：

```dockerfile
FROM python:3.14-slim
COPY . /app
WORKDIR /app
RUN pip install -e .[serve]
ENV LLM_SERVING_HOST=0.0.0.0
ENTRYPOINT ["llm-serve"]
```

build + run：

```bash
docker build -t llm-serve .
docker run -p 8000:8000 \
  -e LLM_SERVING_API_KEY=$(openssl rand -hex 32) \
  -e LLM_SERVING_MODEL_PATH=/checkpoints/epoch_5.pt \
  -v $PWD/checkpoints:/checkpoints:ro \
  llm-serve
```

`compose.yml` 已经把这个串好，可以直接 `docker compose up llm-serve`。

### 7.4 客户端集成

| 客户端 | 怎么连 |
|--------|--------|
| `curl` | 见 §1.3 |
| OpenAI Python SDK | `OpenAI(base_url="http://server:8000/v1", api_key=...)` |
| LangChain | `ChatOpenAI(base_url="http://server:8000/v1", api_key=...)` |
| LlamaIndex | 同 OpenAI |
| vLLM client | 同 OpenAI |
| 自定义 HTTP | `/v1/chat/completions` 完全兼容 OpenAI schema |

---

## 下一步

- 想深入 KV cache 内部（paged / prefix）？看 [docs/reference/architecture.md §KV Cache](../reference/architecture.md)
- 想部署到生产集群？看 [docs/guides/inference.md](../guides/inference.md)
- 想加新的 PEFT 方法并自动挂载到 serving？看 [src/llm/core/peft/](../../src/llm/core/peft/) + `LLM_SERVING_PEFT_METHOD=<your_method>` —— 只要在 `PEFT_REGISTRY` 注册了，`llm-serve` 自动 dispatch
- 想看完整端到端 e2e tests？跑 `pytest -m e2e tests/e2e/test_serve_main_path.py`

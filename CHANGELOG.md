# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Planned

- Flash Attention 2 integration
- Paged Attention full model forward path (sidecar exists; see ADR-004)

### Added

- **Export registry parity** (Tier 3 #32): `EXPORT_REGISTRY` mirrors `BACKEND_REGISTRY`. Built-in `onnx` target plus the `llm.export_backends` setuptools entry-point group for third-party targets (`torch.compile`, `vLLM`, `TensorRT-LLM`, `torch.export`, `OpenVINO`, ...). `export_model("onnx", model, output_path, **kwargs)` is a drop-in for `export_to_onnx(model, output_path, **kwargs)`; the legacy ONNX API is preserved for backward compatibility.
- **TorchScript export target** (Tier 3 #33): first non-built-in backend to exercise the `EXPORT_REGISTRY` plug-in path. Registered through `pyproject.toml`'s `llm.export_backends` entry point as `torchscript = "llm.export.torchscript:build_torchscript_exporter"`. Trace method (`torch.jit.trace`) supported; script method (`torch.jit.script`) wired but `xfail`-tracked for `DecoderModel` (the model's `PositionalEncoding` uses dynamic attribute access that the TorchScript compiler rejects).
- **Docs sync for export registry** (Tier 3 #34): `docs/reference/architecture.md` now lists `export/`'s full layout (`registry.py`, `onnx.py`, `torchscript.py`, `_wrapper.py`) and adds the `llm.export_backends` / `EXPORT_REGISTRY` row to the plugin-kernel table. New `docs/adr/005-export-registry.md` records the architectural decision and cross-references the audit Finding BH plus Tier 3 #32/#33. ADR README now also lists `004-paged-attention-serving.md` (previously missing from the index). `ROADMAP.md` §阶段十四 checkboxes synced to current state — `safetensors 保存`, `TorchScript 导出`, `Export registry`, `HuggingFace Hub publish` all marked done; `checkpoint 格式统一` left unchecked with a one-line note explaining the gap.
- **OpenAI-compat `frequency_penalty`** (Tier 3 #35): the OpenAI-compatible chat / generate endpoints now actually apply the `frequency_penalty` parameter instead of silently dropping it. New `apply_frequency_penalty` sampling helper subtracts `frequency_penalty * count(token)` from each seen token's logit (matches OpenAI's documented semantics). `GenerationConfig` carries the field; every backend (Eager / Batched / Speculative) forwards it; chat + generate routers plumb `request.frequency_penalty` to the service; `ChatCompletionRequest`, `GenerationRequest`, and `BatchGenerationRequest` schemas all accept it (range `[-2.0, 2.0]` per OpenAI). The `(not implemented)` hint in the schema drops.
- **OpenAI-compat `presence_penalty` with flat-per-token semantics** (Tier 3 #37): the OpenAI-compatible chat endpoint now applies `presence_penalty` correctly — penalising a token **once** if it has appeared at all in the generated text, regardless of how many times — instead of the legacy `repetition_penalty = 1.0 + presence_penalty` alias (which produced magnitude-scaled, count-independent behaviour opposite to OpenAI's spec). New `apply_presence_penalty` sampling helper subtracts a flat `presence_penalty` from every seen token's logit (count-independent — that's the key distinction from `frequency_penalty`, which scales by count). `GenerationConfig` carries the field; every backend (Eager / Batched / Speculative) forwards it; chat router plumbs `request.presence_penalty` to the service as its own kwarg; `ChatCompletionRequest.presence_penalty` schema widens to OpenAI's `[-2.0, 2.0]` range and drops the `(mapped to repetition_penalty)` hint. The `repetition_penalty` alias hack at `routers/chat.py:81` is gone.
- **OpenAI-compat `logit_bias`** (Tier 3 #38): the OpenAI-compatible chat / generate endpoints now accept the `logit_bias` parameter (a JSON object mapping token ids to additive logit biases in `[-100, 100]`). New `apply_logit_bias` sampling helper adds the bias to each affected logit via `index_add_` — applied **after** the penalty helpers (repetition / frequency / presence) so the bias dominates any natural penalty, matching OpenAI's reference ordering. String keys (JSON's natural type for object keys) are coerced to `int` at the helper boundary; invalid keys (non-numeric or out-of-vocab) are silently dropped. `GenerationConfig` carries the field; every backend (Eager / Batched / Speculative) forwards it; chat + generate + batch_generate routers plumb `request.logit_bias` to the service; `ChatCompletionRequest`, `GenerationRequest`, and `BatchGenerationRequest` schemas all accept it.
- **Data dedup `TextSource` wrapper** (Tier 3 #39, P0 pretraining productization): new `DedupTextSource` class in `src/llm/data/sources.py` wraps any inner `TextSource` and drops duplicate records by content hash. Default normalization is **case-sensitive** strip + collapse-internal-whitespace (conflating "Apple" vs "apple" would silently drop legitimate records, so case is preserved). `iter_texts(skip=N)` delegates `skip` to the inner source so the `line_index` resume semantics used by `StreamingTextDataset` stay consistent with non-dedup sources. `source_fingerprint()` exposes the inner source's fingerprint plus the dedup strategy so `validate_source_fingerprint` catches config drift on checkpoint resume. Two new `SOURCE_REGISTRY` entries — `dedup_local` and `dedup_hf` — compose the wrapper with the existing `LocalLineTextSource` / `HFStreamTextSource` builders, so `data_source="dedup_local"` / `data_source="dedup_hf"` work end-to-end with the streaming DataModule. `DataConfig` gains three optional fields (`seen_hashes_path`, `write_seen_hashes`, `hash_algo`) and its `data_source` regex widens to `^(local|hf|dedup_local|dedup_hf)$`. The `seen_hashes_path` file is loaded on construction (so dedup state survives across runs) and appended to in append-only mode when `write_seen_hashes=True` (so the seen-set grows monotonically with minimal I/O). Fuzzy / MinHash dedup is a deliberate follow-up — the interface intentionally allows `normalize` and `hash_algo` to be swapped so that can layer on top of this foundation.
- **AdaLoRA foundation slice** (Tier 3 #40, P2 efficient fine-tuning): new `src/llm/core/adalora.py` module with `AdaLoRALinear` — a drop-in alternative to `LoRALinear` that parameterizes the increment matrix in **SVD form** (`ΔW = P · diag(λ · mask) · Q`) with QR-based orthonormalization on every forward. Init defaults match the paper: `init_rank=12`, `target_rank=init_rank // 2`, `alpha=32.0`. `λ` is initialized to zero so `ΔW = 0` exactly at construction — the layer behaves identically to the base layer, matching LoRA's zero-initialized-B invariant. `orth_reg_loss()` exposes `||PᵀP − I||²_F + ||QQ̃ᵀ − I||²_F` so trainers can add it to the task loss with one call. A buffer `mask: (init_rank,)` registered with `register_buffer` lets the **next** slice (T3 #41) zero out low-importance components without a public-API break — this foundation slice does **not** prune, but the API is ready. The helper surface (`apply_adalora` / `merge_adalora` / `unmerge_adalora` / `get_adalora_parameters` / `count_adalora_parameters` / `disable_adalora` / `enable_adalora`) mirrors LoRA by name so swapping LoRA → AdaLoRA in user code is a one-import change. 34 unit tests cover: shape contracts, base-layer freeze, trainable parameter count, initial-output-equals-base, λ-init-to-zero, mask-as-buffer-not-parameter, mask-pruning-zeros-contribution, QR-orthonormality-holds (loss ≤ 1e-9), merge/unmerge roundtrip, `apply_adalora` targeting by substring, `get_adalora_parameters` enumerates P/Q/λ, gradient flow through forward and through `orth_reg_loss`.
- **AdaLoRA pruning slice** (Tier 3 #41, P2 efficient fine-tuning): completes the AdaLoRA story by wiring importance scoring + periodic pruning on top of the T3 #40 foundation. `AdaLoRALinear.compute_importance_scores(gradient_ema=None)` returns the paper's per-component importance — default `|λ_i|`, combined `|λ_i| · |∂L/∂λ_i|` when the trainer passes in a gradient EMA. `AdaLoRALinear.prune_to_rank(target_rank, scores=None)` mutates the `mask` buffer so the top-`target_rank` components survive (idempotent, raises on un-pruning). `AdaLoRALinear.update_budget(current_step, tinit, tfinal)` returns a linearly interpolated rank from `init_rank` at `tinit` to `target_rank` at `tfinal`. New module-level `prune_adalora(model, target_rank=None, schedule=(tinit, tfinal), current_step=None, gradient_emas=None)` walks every `AdaLoRALinear` in the model and applies the same prune-to-rank call — the trainer just calls it on its periodic pruning cadence. Either `target_rank` (uniform across layers) or `schedule` + `current_step` (linear budget schedule) is accepted; an optional `gradient_emas` dict keyed by `id(layer)` lets the trainer thread gradient-EMA scoring into the prune step. **Trainer-side** EMA smoothing and SFT/DPO task integration are deliberately deferred — this slice ships the layer-side half of Algorithm 1. 26 new tests cover: default + gradient-EMA scoring, prune-to-rank drops the right components and is idempotent, prune-to-rank rejects un-pruning, prune-to-rank with explicit scores, update_budget at boundaries + linear interpolation + argument validation, `prune_adalora` walks all layers + target_rank + schedule modes + error paths + gradient-EMA threading.
- **AdaLoRA trainer-integration slice** (Tier 3 #42, P2 efficient fine-tuning): closes the trainer-side half of AdaLoRA Algorithm 1 — the gradient-EMA tracker, the pruning callback, and the SFT/DPO wiring so existing tasks pick up adaptive-budget pruning with a single config flag. New `AdaLoRAGradientEMA` (in `src/llm/core/adalora.py`) walks every `AdaLoRALinear` in the model on construction, inits a per-layer zero EMA buffer, and on every `update()` folds `|lora_lambda.grad|` into the running average with the standard recursion `ema ← α · ema + (1−α) · grad_abs` (layers without a gradient — frozen or unused this step — are left untouched). `state_dict()` / `load_state_dict()` are keyed by the layer's **qualified name** (not `id`, which isn't stable across pickle) so EMA state survives checkpoint resume; unknown / stale keys are silently skipped. New `AdaLoRAPruningCallback` (in `src/llm/training/core/callbacks.py`) is a strict no-op when `use_adalora=False` (safe to register unconditionally); with the flag on, it builds the tracker in `on_train_start`, calls `update()` on a new **`on_optimizer_step`** hook (which fires between `optimizer.step()` and `optimizer.zero_grad()` — the only window in which `.grad` is still populated; without this hook the EMA would always see zero gradients), and every `adalora_prune_every` optimizer steps invokes `prune_adalora(model, schedule=(adalora_tinit, adalora_tfinal), current_step=engine.global_step, gradient_emas=tracker.as_dict())` and logs the mean `adalora/effective_rank` at rank 0. Checkpoint roundtrip reuses the existing `CheckpointContributor` protocol — the `Callback` base now extends `CheckpointContributor`, the engine threads `*self.callbacks` into both `collect_extra_state` and `load_extra_state`, so EMA state survives save/load; `get_checkpoint_state()` returns the tracker's `state_dict()` (or `None` when disabled); `load_checkpoint_state()` restores it. `TrainingConfig` gains ten opt-in fields (`use_adalora` master switch + `adalora_init_rank` / `adalora_target_rank` / `adalora_alpha` / `adalora_orth_reg_weight` / `adalora_ema_alpha` / `adalora_tinit` / `adalora_tfinal` / `adalora_prune_every` / `adalora_target_modules`) with validators (`adalora_ema_alpha ∈ (0, 1)`, `adalora_prune_every ≥ 1`, `adalora_tinit ≥ 0`, cross-field `adalora_target_rank ≤ adalora_init_rank` and `adalora_tfinal > adalora_tinit`); defaults preserve current behavior, so existing configs are unaffected. `LanguageModelingTask.build_model` calls `apply_adalora(...)` with the config fields when `use_adalora=True`; a new `TrainingTask.build_callbacks()` hook (default `[]`) lets the task register the pruning callback on the engine without modifying the engine's constructor signature, and `LanguageModelingTask.build_callbacks()` returns the configured callback. SFT and DPO inherit both — DPOTask builds an AdaLoRA-adapted reference model alongside the policy. 33 new tests cover: EMA math (zero init, smoothing recursion, no-op when no gradient), EMA state roundtrip keyed by qualified name + tolerant of stale keys + tolerant of `None`, callback no-op when disabled + tracker construction + cadence + global-step-driven prune + EMA-on-every-step via `on_optimizer_step` + a regression test that `on_train_step_end` after `zero_grad` does NOT update the EMA (guards against reintroducing the bug), `adalora/effective_rank` logging at rank 0 only, checkpoint `get_checkpoint_state` returns EMA snapshot / `None` when disabled, `load_checkpoint_state` roundtrip + `None` safety, `TrainingConfig` defaults + per-field validation + cross-field validation, `LanguageModelingTask.build_model` applies AdaLoRA + honors `target_modules`, SFT and DPO inherit the wiring, end-to-end SFT-stepped integration test asserts `effective_rank` collapses from `init_rank` to `target_rank` past `adalora_tfinal`, end-to-end engine-construction test asserts the callback is auto-registered without user wiring, end-to-end engine-step test verifies the EMA captures real gradients through the `on_optimizer_step` ordering, end-to-end checkpoint test verifies the EMA survives save/load through the engine's `collect_extra_state` / `load_extra_state` path.
- **Prefix Tuning foundation slice** (T2 PEFT, P2 efficient fine-tuning): first end-to-end Prefix Tuning implementation — the base attention accepts an optional prefix K/V to prepend, the wrapper holds trainable prefix parameters plus two reparameterization MLPs, and module-level helpers mirror LoRA / AdaLoRA so swapping PEFT methods in user code is a one-import change. New `src/llm/core/attn/base.py` defines the `PrefixCapableAttention` Protocol (runtime-checkable, opt-in extension point for adapters). `MultiHeadAttention.forward` gains an optional `prefix_kv: tuple[Tensor, Tensor] | None = None` argument that prepends ``(prefix_k, prefix_v)`` of shape ``[B, num_kv_heads, prefix_len, head_dim]`` to the projected K/V **before** the GQA repeat (so the prefix is treated like a regular token and replicated to all query heads) and **after** any KV-cache write (so the cache only stores dynamic tokens — the prefix is recomputed or folded to a static buffer on every forward). Shape validation raises on mismatched K/V shapes or wrong `num_kv_heads` / `head_dim`. New `src/llm/core/prefix_tuning.py` provides `PrefixTuningAttention(base_attn, prefix_len, reparam_hidden=None)` which freezes the base MHA, holds `prefix_small: nn.Parameter[(prefix_len, reparam_hidden)]` plus `_reparam_k` / `_reparam_v` MLPs that project the small prefix into K/V space (Li & Liang 2021 reparameterization), and dispatches to `base_attn(x, prefix_kv=..., **kwargs)` — every existing MHA kwarg (attn_mask, is_causal, kv_cache, use_cache, batch_indices, start_pos, paged_kv_cache, layer_idx) is forwarded untouched, so cache / paged-cache / GQA / sliding-window paths continue to work. Init uses Kaiming for `prefix_small` and both reparam weight matrices (a zero-init for the reparam would make `d_pk / d_prefix_small = 0` and stall the prefix path at step 1 — a chicken-and-egg problem), biases zero. Module-level `apply_prefix_tuning(model, prefix_len, reparam_hidden=None, target_modules=None)` walks the model and wraps every matching `MultiHeadAttention` (substring-targeted if `target_modules` is given); `get_prefix_parameters(model)` yields exactly the 5 trainable parameters per wrapper (`prefix_small` + 2 reparam MLPs × {weight, bias}) and nothing from the base MHA; `fold_reparameterization(model_or_attn)` collapses the reparam MLPs into static `prefix_k` / `prefix_v` buffers (idempotent, no-op on empty models) so inference needs no trainable params and skips one matmul per layer per step. Non-MHA backends (Flash Attention, MLA, SDPA) intentionally raise `TypeError` at construction time so the failure mode is loud, not silent — wiring them up is a follow-up slice gated on each backend supporting prefix K/V prepending. 29 new tests cover: MHA accepts `prefix_kv=None` with byte-identical baseline; zero prefix rescaled output magnitude by the analytically-correct `Z/(Z+P)` factor (the previous-session test assumed `T/(T+P)` which is wrong — `Z = sum(exp(scores))` is the actual softmax denominator); non-zero prefix changes output; mismatched shapes / wrong `num_kv_heads` raise; GQA repeats the prefix to match `num_query_heads`; `PrefixTuningAttention` shape contracts for `prefix_small` and reparam outputs; forward runs and frozen-prefix version differs; non-MHA base raises `TypeError`; gradients flow to all 5 prefix params at step 1; base MHA stays frozen; fold drops reparam MLPs and registers static buffers; fold preserves forward output (within FP tolerance); fold is idempotent; module-level `apply_prefix_tuning` wraps every MHA, filters by `target_modules` substring, no-ops on empty models; `get_prefix_parameters` yields 5 trainable params per wrapper and excludes base MHA weights; module-level `fold_reparameterization` walks and folds every wrapper, no-ops on empty models.
- **Prefix Tuning trainer-integration slice** (T2 PEFT, P2 efficient fine-tuning): closes the trainer-side half of Prefix Tuning — opt-in via `TrainingConfig`, one-shot wrap at `LanguageModelingTask.build_model`, no scheduler / tracker needed (unlike AdaLoRA). New `TrainingConfig` fields: `use_prefix_tuning` master switch (default `False`), `prefix_tuning_len: int = 10` (Li & Liang 2021 paper default), `prefix_reparam_hidden: int | None = None` (None → defaults to `kv_dim` at the wrapper layer), `prefix_target_modules: list[str] | None = None` (None → every `MultiHeadAttention`). Validators reject `prefix_tuning_len ≤ 0` and `prefix_reparam_hidden ≤ 0` when set (None is the documented "use layer default" sentinel). Defaults preserve current behavior — existing configs are unaffected. `LanguageModelingTask.build_model` calls `apply_prefix_tuning(...)` after the AdaLoRA branch when `use_prefix_tuning=True`, so the two PEFT methods compose cleanly (they wrap different module classes — AdaLoRA wraps `nn.Linear`, Prefix Tuning wraps `MultiHeadAttention` — so there's no structural conflict). `SFTTask` inherits the wrapping unchanged; `DPOTask` calls `super().build_model()` twice so prefix tuning is applied to **both** the policy and the reference model without any extra wiring. There is intentionally **no callback**: Prefix Tuning has no periodic action to schedule — `apply_prefix_tuning` is a one-shot wrap at construction time, and the user calls `fold_reparameterization` at inference time (matching the LoRA apply / merge pattern). 20 new tests cover: `TrainingConfig` defaults + per-field validation (prefix_len > 0, prefix_reparam_hidden > 0 when set, None is allowed) + co-existence with AdaLoRA fields + opt-in via explicit kwargs; `LanguageModelingTask.build_model` does NOT wrap when `use_prefix_tuning=False`; opt-in wraps every MHA when True; `prefix_target_modules` substring forwards correctly; a non-matching `prefix_target_modules` filter leaves the MHA untouched; base MHA stays frozen (qkv_proj + out_proj `requires_grad=False`) while all 5 prefix params are trainable; an optimizer wired via `get_prefix_parameters` updates only the prefix path (base MHA weights byte-identical before/after one Adam step); `SFTTask` inherits the wiring (opt-in wraps, default off); `DPOTask` wraps **both** the policy and the reference model (off by default leaves both unwrapped); a model with no `MultiHeadAttention` modules at all leaves `apply_prefix_tuning` a no-op without raising.
- **IA³ foundation slice** (T2 PEFT, P2 efficient fine-tuning): first end-to-end IA³ implementation — the multiplicative counterpart to LoRA. New `src/llm/core/ia3.py` provides `IA3Linear(base_layer, init_scale=1.0)` which freezes the base `nn.Linear`, holds a single learned `ia3_l: nn.Parameter[(out_features,)]` that multiplicatively scales the base output (`y = (Wx + b) * l`), and exposes `merge_weights` / `unmerge_weights` for inference-time folding (after merge the wrapper is identity on top of the already-scaled base, with `ia3_l` snapshotted for unmerge). Init defaults to `ia3_l = ones` so the wrapper is the identity transform at step 1 — no chicken-and-egg training stall. The helper surface (`apply_ia3` / `merge_ia3` / `unmerge_ia3` / `get_ia3_parameters` / `count_ia3_parameters` / `disable_ia3` / `enable_ia3`) mirrors LoRA by name for a one-import swap. Per-layer cost is `out_features` trainable params — typically two orders of magnitude smaller than LoRA's `rank * (in + out)` at the same `out_features` (e.g. 128 vs. 6144 at `in=64, out=128, rank=8`). No scheduler, no callback, no reparameterization MLPs — IA³ has no periodic action, so it's the lightest PEFT method in the stack. 33 new tests cover: default init (`ia3_l == ones`) + custom init_scale; base weight frozen + `ia3_l` trainable; per-channel scaling produces the expected `base_out * l` output (scalar and per-channel scales); forward broadcasts over batch and seq; backward populates only `ia3_l.grad` (base weight `grad is None`); `merge_weights` makes the wrapper identity on top of the already-folded base + zeroing `ia3_l` zeros the output (verifies the merge was complete, not partial); `unmerge_weights` restores the original `ia3_l` snapshot AND the pre-merge base weight; `merge_weights` handles `bias=False`; `merge_weights` actually mutates the base weight under a non-trivial scale; `apply_ia3` wraps every Linear by default + filters by `target_modules` substring + supports multiple patterns + is in-place + accepts an empty target list as "wrap everything"; module-level `merge_ia3` / `unmerge_ia3` walk and fold all wrappers; `get_ia3_parameters` yields exactly one `ia3_l` per wrapper (and zero when none exist) + an optimizer wired via the helper updates only `ia3_l` (base weight byte-identical before/after one Adam step); `count_ia3_parameters` returns `sum(out_features)` after `apply_ia3` (much smaller than `sum(in * out)`) and reports all-trainable before `apply_ia3`; `disable_ia3` zeros the effective scale (sets to ones) + `enable_ia3` restores the snapshot + a `disable_ia3` model produces output byte-identical to a fresh base Linear; the headline property test pins that IA³ trainable params are strictly smaller than LoRA's `rank * (in + out)` at matched `out_features`.
- **IA³ trainer-integration slice** (T2 PEFT, P2 efficient fine-tuning): closes the trainer-side half of IA³ — opt-in via `TrainingConfig`, one-shot wrap at `LanguageModelingTask.build_model`, no scheduler / tracker needed (matches the Prefix Tuning pattern). New `TrainingConfig` fields: `use_ia3` master switch (default `False`), `ia3_init_scale: float = 1.0` (positive-only — the wrapper is the identity transform at step 1 by design), `ia3_target_modules: list[str] | None = None` (None → every `nn.Linear`). Defaults preserve current behavior — existing configs are unaffected. `LanguageModelingTask.build_model` calls `apply_ia3(...)` after the AdaLoRA / Prefix Tuning branches when `use_ia3=True`. Note: IA³ and AdaLoRA both wrap `nn.Linear`, so in practice users should not enable both at the same time (the second wrap would wrap an already-wrapped `IA3Linear`'s base `Linear` — silent no-op); the config does NOT forbid the combination, since picking the right method is a user decision. `SFTTask` inherits the wrapping unchanged; `DPOTask` calls `super().build_model()` twice so IA³ is applied to **both** the policy and the reference model without any extra wiring. There is intentionally **no callback**: IA³ has no periodic action to schedule — `apply_ia3` is a one-shot wrap at construction time, and the user calls `merge_ia3` at inference time (matching the LoRA apply / merge pattern). 18 new tests cover: `TrainingConfig` defaults (off, init_scale=1.0, target_modules=None) + positive-init_scale validator (rejects 0.0 and negative) + opt-in via explicit kwargs + three-way coexistence with AdaLoRA + Prefix Tuning config flags; `LanguageModelingTask.build_model` does NOT wrap when `use_ia3=False`; opt-in wraps every `nn.Linear` when True; `ia3_target_modules` substring forwards correctly; a non-matching filter leaves the Linear untouched; `ia3_init_scale` flows into the wrapper's `ia3_l` initial value; base Linear stays frozen (`weight.requires_grad=False`) while `ia3_l` is trainable; an optimizer wired via `get_ia3_parameters` updates only `ia3_l` (base Linear weight byte-identical before/after one Adam step); `SFTTask` inherits the wiring (opt-in wraps, default off); `DPOTask` wraps **both** the policy and the reference model (off by default leaves both unwrapped); a model with no `nn.Linear` modules at all leaves `apply_ia3` a no-op without raising.
- **BitFit foundation slice** (T2 PEFT, P2 efficient fine-tuning): first end-to-end BitFit implementation — the simplest PEFT method, train only bias parameters. New `src/llm/core/bitfit.py` provides `apply_bitfit(model, target_modules=None)` which freezes every parameter via `requires_grad=False`, then enables gradients on every bias (matched on the **`.bias` suffix**, not substring — substring would falsely match module names like `fc_with_bias` whose `weight` parameter is NOT a bias). The pre-BitFit `requires_grad` state is snapshotted on the model under `_bitfit_original_requires_grad` so `unapply_bitfit(model)` can restore it exactly — useful for roundtrip experiments and checkpoint validation. `get_bitfit_parameters(model)` yields only bias params that are trainable (suffix-checked, so the helper stays correct if a user manually enables a non-bias parameter after applying BitFit). `count_bitfit_parameters` and `is_bitfit_applied` round out the API surface — `is_bitfit_applied` checks for the snapshot attribute. No scheduler, no callback, no wrappers, no new parameters — BitFit is the lightest possible PEFT method (just toggles `requires_grad`). 22 new tests cover: every weight is frozen + every bias is trainable (the headline post-condition); bias-free models yield zero trainable params; `target_modules` substring filter restricts which biases are trainable (multi-pattern supported); in-place mutation; idempotent across repeated `apply_bitfit` calls; `is_bitfit_applied` correctly tracks state; `unapply_bitfit` restores a model that had a mix of frozen / trainable params before BitFit (verified end-to-end with a `_ModelWithFrozenWeights` fixture); `unapply_bitfit` clears the snapshot + is a no-op when never applied + apply→unapply→apply converges to the same state; `get_bitfit_parameters` yields only biases (suffix-checked) + yields nothing on a bias-free model + an optimizer wired via the helper updates only biases (base weight byte-identical before/after one Adam step) + is robust against a user manually enabling a non-bias parameter after apply; `count_bitfit_parameters` reports `sum(bias sizes)` after apply + 0 trainable when no biases; the headline property tests pin BitFit as strictly smaller than LoRA at typical scales + post-condition: trainable == sum of bias sizes + BitFit adds zero new parameters (total param count unchanged).
- **BitFit trainer-integration slice** (T2 PEFT, P2 efficient fine-tuning): closes the trainer-side half of BitFit — opt-in via `TrainingConfig`, one-shot wrap at `LanguageModelingTask.build_model`, no scheduler / tracker / inference-merge needed (BitFit's biases are simply left in place at inference — no extra cost, no merge step). New `TrainingConfig` fields: `use_bitfit` master switch (default `False`), `bitfit_target_modules: list[str] | None = None` (None → every `.bias` is enabled). Defaults preserve current behavior — existing configs are unaffected. `LanguageModelingTask.build_model` calls `apply_bitfit(...)` after the AdaLoRA / Prefix Tuning / IA³ branches when `use_bitfit=True`. Note: BitFit is compatible with all the other PEFT methods at the config level — the user picks one — but in practice the right call is to enable exactly one PEFT method per training run. `SFTTask` inherits the wrapping unchanged; `DPOTask` calls `super().build_model()` twice so BitFit is applied to **both** the policy and the reference model (DPO's standard `ref_model.eval()` + freeze is layered on top of BitFit, so the reference is fully frozen regardless). There is intentionally **no callback** and **no merge helper**: BitFit has no periodic action and no inference-time fold — the biases are simply part of the model at serve time. 14 new tests cover: `TrainingConfig` defaults (off, target_modules=None) + opt-in via explicit kwargs + four-way coexistence with AdaLoRA / IA³ / Prefix Tuning config flags; `LanguageModelingTask.build_model` does NOT apply BitFit when `use_bitfit=False` (every weight stays trainable, no snapshot saved); opt-in freezes every weight + enables every `.bias`; `bitfit_target_modules` substring filter restricts which biases are enabled; an optimizer wired via `get_bitfit_parameters` updates only biases (every weight byte-identical before/after one Adam step); `get_bitfit_parameters` after `build_model` yields exactly the bias parameters (with the fixture: fc1.bias + fc2.bias + norm.bias); `SFTTask` inherits the wiring (opt-in freezes weights + enables biases, default off); `DPOTask` applies BitFit to **both** the policy and the reference model (off by default leaves neither side touched); a model with no biases at all leaves `apply_bitfit` a no-op that produces a fully-frozen model.
- **Hypothesis invariant tests for the PEFT slice** (T2 #19, P2 test coverage): adds `tests/core/test_hypothesis_peft.py` with 14 property-based tests covering IA³, BitFit, and LoRA — verifying the **invariants** that should hold for ANY shape / scale / bias-count, not just the specific scenarios the example-based tests pin. IA³: `IA3Linear.forward(x) == base_layer(x) * ia3_l` for any input and scale (broadcast invariant); at `init_scale=1.0` the wrapper is the identity on top of the base; `merge_weights()` preserves the forward output for any input (the scale is folded into the base, then `ia3_l = ones` makes the wrapper identity); `merge → unmerge` restores the base weight exactly (modulo float drift, with `assume` constraining the strategy to skip the degenerate zero-scale case where division by the restored zero would yield NaN); `disable_ia3` makes the wrapper identity for any input; `disable → enable` round-trips the saved snapshot. BitFit: after `apply_bitfit` every `.bias`-suffixed param has `requires_grad=True` and every other param has `requires_grad=False` (the headline post-condition, now verified across the strategy's random bias / no-bias configurations); the trainable count equals exactly the sum of bias sizes; `apply → unapply` restores the original `requires_grad` state exactly; `get_bitfit_parameters` yields only the trainable biases (never weights, never frozen biases). LoRA: `LoRALinear.forward(x) == base_layer(x) + (lora_A @ lora_B) * scaling` for any input / rank / alpha; `merge → unmerge` returns the base weight to its pre-merge state. Cross-method invariant: on the same model shape, both BitFit and IA³ produce strictly smaller trainable footprints than LoRA at rank=4 — pinning the "small PEFT suite" headline property across the strategy's parameter space. The fixture is a `_Holder` with 3 Linears + 1 LayerNorm where each Linear's bias presence is randomly drawn — exercises the bitfit / ia3 / lora filters independently.


### Changed

- **2026 Q2 architecture convergence** (Phases 1–4, Waves 1–3, P2 cleanup):
    - Unified `runtime.Registry` for all component registries; removed legacy `ComponentRegistry`
    - KV cache: single `KVCache` / `kv_caches` API; removed `past_key_value` tuple path
    - Norm: `norm_impl` config + `NORM_REGISTRY` wiring (`layer_norm`, `rms_norm`)
    - Eval: removed `evaluator.py` / `infer_task.py`; unified `EvaluationRunner`
    - Serving: removed `priority_scheduler.py`, `serving/prefix_cache.py`; `ContinuousBatchingEngine` gains `SlotPrefixCache`, `from_serving_config()`
    - Data: `TokenizedMapDataModule.setup_tokenized_file_dataset()`, `SamplerMapDataModule`, `TokenizerFactory` helpers
    - Tasks: `regression_mlp` via `llm.models` entry point; `RegressionTask` uses `ModelFactory`
    - Bootstrap: model registration via setuptools entry points only (`decoder`, `regression_mlp`)
    - MLA: registered as `@register_attention("mla")`; supports linear `KVCache` and paged `PagedKVCache` (Tier 3 #31). The current implementation is the placeholder variant (learnable latent queries + uniform-mean output); DeepSeek-V2-style latent-compressed K, V is a separate follow-up.

### Refactored

- **Code Organization**:
    - Extracted `make_factory_kwargs()` and `init_lora_weights()` utilities
    - Migrated all `__main__` demo code to test files
    - Added custom exception module with hierarchical exception types

- **Error Handling**:
    - Replaced broad `except Exception` with specific exception types
    - Improved API error logging and message handling

- **Type Annotations**:
    - Fixed `pad_token_id` duplicate definition
    - Fixed `normalized_shape` tuple type mismatches
    - Added missing type annotations in MoE module
    - Fixed None handling in config utilities

- **Code Quality**:
    - Removed ~600 lines of demo code from source modules
    - Preserved educational NumPy implementations for learning
    - Added comprehensive test coverage for demo functionality

## [0.0.5] - 2026-01-08

### Added

- **SFT (Supervised Fine-tuning)**:
    - `SFTDataset` for instruction tuning with input masking
    - `SFTDataModule` for data loading
    - `SFTTask` registered as `--task sft` in CLI
    - Tests for all SFT components

- **DPO (Direct Preference Optimization)**:
    - `DPODataset` handling chosen/rejected pairs
    - `DPODataModule` for preference data loading
    - `DPOTask` with reference model management and DPO loss
    - Registered as `--task dpo` in CLI
    - Tests for all DPO components

- **Continuous Batching Engine** (Serving):
    - `src/llm/serving/engine.py` with `ContinuousBatchingEngine` class
    - Iteration-level scheduling via `Scheduler` and `SlotAllocator`
    - Pre-allocated KV cache pool for efficient memory management
    - Supports mixed prefill/decode batching with automatic padding
    - Clean API: requires `model` and `tokenizer` instances upfront
    - `src/llm/serving/scheduler.py` with FCFS scheduling logic

- **LoRA (Low-Rank Adaptation)**:
    - `src/llm/core/lora.py` with `LoRALinear` class for parameter-efficient fine-tuning
    - `apply_lora()`, `merge_lora()`, `get_lora_parameters()` helper functions
    - Device/dtype handling for CUDA compatibility
    - 17 tests covering training and weight merging

- **QLoRA (Quantized LoRA)**:
    - `src/llm/core/qlora.py` with `QLoRALinear` class
    - NF4 4-bit quantization for base weights (~4x memory reduction)
    - LoRA adapters remain in fp16/bf16 for training stability
    - `apply_qlora()` and `get_qlora_parameters()` helpers

- **RoPE (Rotary Position Embedding)**:
    - `src/llm/core/rope.py` with `RotaryPositionEmbedding` class
    - Linear, dynamic, and NTK-aware scaling methods for extended context
    - `apply_rotary_pos_emb()`, `get_rope_scaling_factor()` utilities
    - 15 tests

- **ALiBi (Attention with Linear Biases)**:
    - `src/llm/core/alibi.py` with `ALiBiPositionBias` class
    - `get_alibi_slopes()`, `build_alibi_bias()` functions
    - Cached bias computation for efficiency
    - 13 tests

- **Sliding Window Attention**:
    - `window_size` parameter in `scaled_dot_product_attention`
    - Propagated through `MultiHeadAttention`, `TransformerBlock`, `DecoderModel`
    - Reduces memory for long sequences by limiting attention scope
    - 10 tests

- **KV Cache Optimization**:
    - `src/llm/core/kv_cache.py` with `KVCache` class for pre-allocated cache buffers
    - In-place updates during autoregressive generation (avoids O(n²) memory operations)
    - Integrated into `MHA`, `TransformerBlock`, `DecoderModel`
    - Factory method `KVCache.from_model_config()` for easy instantiation
    - Unified `kv_caches` API; legacy tuple format removed in Wave 3

- **E2E Testing Infrastructure**:
    - `tests/e2e/` directory with comprehensive pipeline tests
    - `test_training.py`, `test_sft.py`, `test_dpo.py`
    - `test_gradient_accumulation.py`, `test_resume_training.py`
    - Advanced inference and callback tests

- **Documentation**:
    - `notebooks/quick_start.ipynb` interactive tutorial
    - Covers model building, training, inference, and advanced features

### Changed

- **SDPA Refactoring**:
    - Consolidated `scaled_dot_product_attention` wrapper into `src/llm/core/attn/sdpa.py`
    - Refactored `MultiHeadAttention` and `MultiLatentAttention` to use common `sdpa` wrapper
    - Archived custom implementation to `_learning/03_lab/experiments/custom_sdpa.py`

- **Test Suite Refactoring**:
    - Organized test files into subdirectories (`tests/training/`, `tests/inference/`, etc.)
    - Converted to functional testing style (real components over mocks)
    - Added shared fixtures in `tests/conftest.py`
    - Test count: 385 → 432

- **TrainingEngine**:
    - Support for dictionary batches in training/validation loops
    - Gradient accumulation implementation

- **DPO Reference Model**:
    - Use model reconstruction instead of `deepcopy` for ref_model creation

- **Documentation**:
    - Added `docs/README.md` as documentation entry point
    - Added MkDocs Material configuration (`mkdocs.yml`) for documentation site
    - Added GitHub Actions workflow for automatic GitHub Pages deployment
    - Added `guide-finetuning.md` (LoRA/QLoRA) and `guide-inference.md` (KVCache/GQA/Continuous Batching)
    - Enhanced `architecture.md` with detailed component diagrams and data flow analysis
    - Updated ROADMAP Phase 10.2 (Continuous Batching complete)

## [0.0.4] - 2026-01-07

### Added

- **Gradient Checkpointing**:
    - Memory-efficient training via `gradient_checkpointing` parameter in `DecoderModel`
    - `enable_gradient_checkpointing()` / `disable_gradient_checkpointing()` methods
    - Automatic incompatibility check with `use_cache=True`

- **E2E Pipeline Automation**:
    - `scripts/e2e_pipeline.py` for automated Train → Evaluate → Inference workflow
    - `src/llm/utils/e2e.py` with reusable E2E core functions (`E2EConfig`, `E2EResult`, `run_e2e_pipeline`)
    - Rich progress UI and configurable CLI options

- **OpenAI-Compatible Chat API** (`/v1/chat/completions`):
    - Compatible with official OpenAI Python SDK
    - Streaming and non-streaming chat completions
    - Bearer token authentication support
    - Multi-turn conversation handling
    - 8 new test cases for compatibility layer

- **Batch Inference**:
    - `batch_generate` function in `inference.py` with left-padding and batched forward pass
    - `BatchGenerationRequest` / `BatchGenerationResponse` schemas
    - `/batch_generate` API endpoint
    - 3 tests for batch inference (basic, single, empty)

- **Request Queue and Concurrency Control**:
    - `max_concurrent_requests` and `request_timeout` in `ServingConfig`
    - `asyncio.Semaphore` for concurrency limiting
    - `asyncio.timeout` for request timeout handling (504 response)

- **CLI Entry Points**:
    - `llm-train` command for training models
    - `llm-serve` command for starting inference server

- **Testing Infrastructure**:
    - Pytest markers using decorators: `quick`, `slow`, `heavy`, `e2e`
    - MoE integration tests (6 tests for expert routing, gradient flow)
    - E2E pipeline tests (full workflow, streaming consistency)
    - Gradient checkpointing tests (8 tests)
    - Total test count: 296 → 337

- **Examples Directory**:
    - `inference_demo.py` for basic text generation
    - `openai_client_demo.py` for OpenAI SDK usage

- **Documentation**:
    - `scripts/README.md` documenting all available scripts
    - HFTokenizer example in `usage.md`
    - Updated root `README.md` with links to Examples and Scripts

### Changed

- **Makefile Reorganization**:
    - `make test` now runs all tests by default
    - `make test-fast` for daily development (excludes heavy/e2e)
    - `make test-quick` for rapid iteration (~6s)
    - `make test-cov` for CI with coverage and allure reports
    - Removed redundant `test-all` and `test-integration`

- **CLI Standardization**:
    - CLI parameters changed from snake_case to kebab-case (`--file-path`, `--batch-size`)
    - Replace `typer` with `typer-slim[standard]` for reduced dependencies

- **Code Quality Improvements**:
    - Translate Chinese docstrings to English in serving module
    - Remove ~75 lines of redundant comments
    - Simplify section comments while preserving algorithm clarity

- **Documentation Refactoring**:
    - Eliminated redundancy between README, usage.md, and development.md
    - Clear document responsibility separation
    - Updated all docs to use new CLI commands
    - Enhanced package metadata (keywords, classifiers)

- **Module Exports**:
    - Enhanced `llm/__init__.py` with public API exports (`DecoderModel`, `generate`, etc.)
    - Enhanced `llm.serving` module exports (`LLMEngine`, `ServingConfig`, OpenAI schemas)

### Fixed

- Removed obsolete TODO comment in `engine.py`
- Removed duplicate `num_kv_heads` field in `ModelConfig`
- Fixed MD051/link-fragments in `tutorial-cpu-llm.md` and `faq.md`
- Fixed `train.py` task registration for `lm` task

## [0.0.3] - 2025-12-23

### Added

- **Inference Serving**:
    - Production-ready REST API with FastAPI
    - Streaming support via Server-Sent Events (SSE)
    - Advanced sampling strategies (nucleus sampling/top-p, repetition penalty)
    - Prometheus metrics endpoint for monitoring
    - API key authentication (`X-API-Key` header)
    - Structured logging with `python-json-logger`
    - Real PyTorch model weights loading from checkpoint files
    - Pickled tokenizer object loading support

- **Component Registry**:
    - Automatic component registration system (`ComponentRegistry`)
    - Core components (MHA, MLP, MoE) auto-registered via side-effect imports
    - Prevents "component not found" errors in simplified scripts

- **Data Abstraction**:
    - Formalized `BaseTokenizer` protocol
    - `BaseDataModule` abstraction for flexible data handling
    - Environment variable configuration support (e.g., `LLM_TRAINING__EPOCHS`)

- **Testing & CLI**:
    - `--num-samples` flag in `train.py` for rapid regression testing
    - Scheduler edge case tests (`test_scheduler_edge_cases.py`)
    - Validation logging tests (`test_engine_logging.py`)
    - Component registry tests (`test_init.py`)
    - Model loading verification tests
    - Auto-device detection in training scripts (prioritizes CUDA)

- **Documentation**:
    - Comprehensive usage guide (`docs/usage.md`)
    - Architecture documentation (`docs/architecture.md`)
    - Engineering documentation (ADRs, PR templates, FAQ)
    - VS Code configuration and extensions

### Changed

- **Architecture Modernization**:
    - Migrated to Pydantic v2 (`BaseSettings`, `BaseModel`) for configuration
    - Fully typed and validated configuration system
    - CLI migration from `argparse` to `typer` for better UX

- **Naming Standardization**:
    - Unified `ffn_hidden_size` → `intermediate_size` across codebase
    - Standardized input parameter `x` → `hidden_states` in forward methods
    - Applied to `MLP`, `LayerNorm`, `RMSNorm`, `DecoderModel`, `TransformerBlock`
    - Updated all 309 tests to reflect API changes

- **Code Quality**:
    - Standardized punctuation in documentation (full-width → half-width)
    - Improved type hints and documentation comments
    - Refactored `TransformerBlock.forward` for clarity

### Fixed

- **Core Bugs**:
    - `CosineAnnealingLR` `T_max` calculation when `epochs == warmup_epochs` (ZeroDivisionError)
    - `TrainingEngine` validation logging crash when `gradient_norms` is empty (IndexError)
    - PAD token generation issue in inference (logits masking)
    - `SyntheticDataModule` `prefetch_factor` handling with `num_workers=0`
    - `TransformerBlock` shared norm instance bug (independent `norm1`/`norm2`)
    - Scheduler/optimizer step order warnings in tests
    - PositionalEncoding support for `start_pos` in incremental generation
    - MLP SwiGLU operation order for numerical consistency
    - Prompt truncation respecting `max_seq_len` with new tokens
    - Auto AMP dtype resolution for CPU-only environments

- **Registry & Imports**:
    - Package auto-registration via `import llm`
    - Component not found errors in simplified execution

## [0.0.2] - 2025-12-21

### Added

- **Modern Architecture Features**:
    - Grouped Query Attention (GQA) for balanced performance and memory efficiency
    - SwiGLU activation function in MLP layers
    - Unified QKV projection optimization for improved memory layout and throughput
    - RMSNorm support as alternative normalization layer

- **Tokenization & Training**:
    - BPETokenizer for production-ready subword tokenization
    - LanguageModelingTask for language model training
    - Automatic BF16/FP16 mixed precision detection and support
    - Robust NaN loss handling

- **Inference Capabilities**:
    - KV Cache support in MHA, TransformerBlock, and DecoderModel
    - Top-k and Top-p sampling strategies
    - Greedy search decoding (temperature=0)
    - Dynamic sequence length support
    - Simple autoregressive generation loop

- **Testing & Quality**:
    - 262 comprehensive unit test cases covering all core functionality
    - Functional tests for causal masking, KV cache consistency, architecture properties
    - Convergence tests for training validation
    - Mock-free test design using real components

- **Documentation**:
    - Comprehensive ROADMAP.md (405 lines) with 15 development stages
    - Priority levels (P1-P4), timelines, and success metrics
    - Detailed training framework documentation (8 comprehensive guides)
    - CPU-friendly LLM tutorial and development guide
    - FAQ document covering core topics
    - ADR (Architecture Decision Records) system with 4 initial records
    - PR template for standardized contributions

### Changed

- **Architecture Optimization**:
    - Refactored DecoderModel with configurable components
    - Optimized padding mask and KV cache handling
    - Improved GradScaler usage for bfloat16

- **Training Enhancements**:
    - Enhanced TrainingEngine with improved callback system
    - Performance monitoring and logging improvements
    - Auto AMP dtype resolution for CPU-only environments

- **Code Quality**:
    - Enhanced Ruff linting rules (SIM, RUF, PTH for pathlib)
    - PEP 561 compliance with py.typed marker
    - Standardized punctuation across documentation
    - Project structure improvements for modularity

- **Documentation**:
    - Updated Quick Start example from regression to lm task
    - Enhanced feature descriptions with technical highlights
    - Better cross-references and examples throughout

### Fixed

- **Core Issues**:
    - All 262 test regressions resolved
    - PositionalEncoding support for `start_pos` in incremental generation
    - MLP SwiGLU operation order for numerical consistency
    - Prompt truncation respecting `max_seq_len` with new tokens
    - Device mismatch in MLP when norm instance provided
    - Auto AMP dtype test failures on CUDA environments

- **Quality & Stability**:
    - Type checking issues across the codebase
    - Memory management in distributed training
    - Edge cases in attention masking and positional encoding
    - Device comparisons robustness (comparing device.type)
    - Failed runs on CPU-only environments

## [0.0.1] - 2024

### Added

- Initial project setup with modern Python tooling (uv, hatchling)
- Basic Decoder-only Transformer architecture
- Multi-Head Attention (MHA) implementation
- Standard MLP with GELU activation
- SimpleCharacterTokenizer for basic experimentation
- Positional encoding (sinusoidal and learned)
- TrainingEngine with Distributed Data Parallel (DDP) support
- Automatic Mixed Precision (AMP) training
- Basic Mixture of Experts (MoE) implementation
- Core data loading and processing infrastructure
- BaseDataModule abstraction for flexible data handling
- pytest-based testing infrastructure
- CI/CD pipeline with GitHub Actions
- Code quality tools (ruff for linting/formatting, mypy for type checking)
- Pre-commit hooks for code quality enforcement
- Docker support for containerized development
- Comprehensive README and contributing guidelines

"""lm_eval ``LM`` adapter for our :class:`DecoderModel`.

lm-evaluation-harness expects model wrappers to implement the
``lm_eval.api.model.LM`` protocol (``loglikelihood``,
``loglikelihood_rolling``, ``generate_until``). This module
provides a minimal adapter that conforms to that interface so any
trained :class:`DecoderModel` + tokenizer can be evaluated with
``lm_eval.evaluator.evaluate(lm=LlamaLmEvalLM(model, tokenizer))``.

Soft dependency on ``lm_eval`` — this module imports lazily inside
``__init__`` so importing :mod:`llm.evaluation.harness.lm_eval_lm`
never raises on hosts without ``lm_eval`` installed. The
``ImportError`` fires at ``__init__`` time with the install hint.

Why a dedicated wrapper rather than reusing :class:`HFLM`?
HFLM's surface requires a ``torch.device`` and several HF-only
arguments (``prefix_token``, ``backend``) that don't apply here.
A 100-line minimal adapter keeps the dependency tree honest and
the contract obvious.
"""

from __future__ import annotations

import importlib.util
from typing import Any

import torch

# Soft-dependency probe so callers can branch on availability
# without paying for an import.
_lm_eval_spec = importlib.util.find_spec("lm_eval")
LM_EVAL_AVAILABLE: bool = _lm_eval_spec is not None


def _require_lm_eval() -> None:
    if not LM_EVAL_AVAILABLE:
        raise ImportError(
            "LlamaLmEvalLM requires the optional 'lm_eval' package. "
            "Install with `pip install 'llm[eval]'`."
        )


class LlamaLmEvalLM:
    """Minimal :class:`lm_eval.api.model.LM` adapter for ``DecoderModel``.

    Implements just enough of the lm_eval protocol to run the
    standard multiple-choice (loglikelihood) and generation
    (generate_until) tasks. Each request is processed individually
    with a single forward pass; ``batch_size`` controls how many
    requests are handled between ``no_grad`` context rebuilds and
    Python-side bookkeeping, not the model's batch dimension.

    Args:
        model: A trained :class:`llm.models.DecoderModel`.
        tokenizer: Tokenizer with ``encode``, ``decode``, ``bos_token_id``
            (optional), ``eos_token_id``. Must produce token ids
            accepted by the model's vocabulary.
        batch_size: Maximum number of requests per forward pass.
        max_length: Hard cap on sequence length (model's
            ``max_seq_len`` is the default).
        device: Target device; defaults to the model's parameter device.
    """

    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        *,
        batch_size: int = 8,
        max_length: int | None = None,
        device: str | torch.device | None = None,
    ) -> None:
        _require_lm_eval()
        # We import lazily so this module is import-safe without lm_eval.
        from lm_eval.api.model import LM  # noqa: F401  (validates install)

        self.model = model
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.max_length = max_length or getattr(model, "max_seq_len", 2048)
        if device is None:
            device = next(model.parameters()).device
        self.device = torch.device(device) if not isinstance(device, torch.device) else device
        self.model.eval()

    # --- lm_eval protocol surface ------------------------------------------

    def loglikelihood(self, requests):
        """Compute (log_likelihood, is_greedy_match) for each request.

        Each request is an ``lm_eval.api.request.Request`` whose
        ``args`` is ``(context_str, continuation_str)``. Returns a
        list of ``(sum_logprob, is_greedy)`` tuples.
        """
        results = []
        with torch.no_grad():
            for batch_start in range(0, len(requests), self.batch_size):
                batch = requests[batch_start : batch_start + self.batch_size]
                batch_results = self._loglikelihood_batch(batch)
                results.extend(batch_results)
        return results

    def loglikelihood_rolling(self, requests):
        """Compute total log-probability of each string (perplexity-style).

        Returns ``list[float]`` — one scalar sum-log-prob per request,
        matching the lm_eval ``LM.loglikelihood_rolling`` protocol.
        """
        results = []
        with torch.no_grad():
            for batch_start in range(0, len(requests), self.batch_size):
                batch = requests[batch_start : batch_start + self.batch_size]
                batch_results = self._loglikelihood_rolling_batch(batch)
                results.extend(batch_results)
        return results

    def generate_until(self, requests):
        """Greedy generation until ``until`` token sequences appear.

        Each request is an ``lm_eval.api.request.Request`` whose
        ``args`` is ``(context_str, {"until": [...], "max_gen_toks": N})``.
        """
        results = []
        with torch.no_grad():
            for batch_start in range(0, len(requests), self.batch_size):
                batch = requests[batch_start : batch_start + self.batch_size]
                batch_results = self._generate_until_batch(batch)
                results.extend(batch_results)
        return results

    # --- batched implementations -------------------------------------------

    def _loglikelihood_batch(self, batch):
        """Tokenize + forward + per-token log-prob extraction for a batch.

        For each (context, continuation):
        1. Concatenate ``context + continuation`` token ids.
        2. Pad to the longest sequence in the batch.
        3. Forward; collect log-probs at each continuation position.
        4. Sum and compare to greedy argmax for ``is_greedy_match``.
        """
        results = []
        # Tokenize once, defer padding to after we know the longest length.
        encoded: list[tuple[list[int], list[int]]] = []
        for req in batch:
            context, continuation = req.args
            ctx_ids = self._encode(context)
            cont_ids = self._encode(continuation)
            if not cont_ids:
                # lm_eval generally guarantees a non-empty continuation,
                # but guard against pathological inputs.
                cont_ids = [0]
            encoded.append((ctx_ids, cont_ids))

        max_len = min(
            self.max_length,
            max(len(c) + len(k) for c, k in encoded),
        )

        for ctx_ids, cont_ids in encoded:
            full = (ctx_ids + cont_ids)[-max_len:]
            ctx_len = max(0, len(full) - len(cont_ids))
            cont_len = len(cont_ids)

            input_tensor = torch.tensor([full], dtype=torch.long, device=self.device)
            model_out = self.model(input_tensor, use_cache=False)
            logits = model_out[0] if isinstance(model_out, tuple) else model_out
            # Log-probs at each continuation position: logits at
            # ``ctx_len - 1 + i`` predicts ``full[ctx_len + i]``.
            relevant = logits[0, max(0, ctx_len - 1) : ctx_len - 1 + cont_len, :]
            log_probs = torch.log_softmax(relevant, dim=-1)
            cont_tensor = torch.tensor(full[ctx_len:], device=self.device, dtype=torch.long)
            target_log_probs = log_probs[
                torch.arange(cont_len, device=self.device), cont_tensor
            ]
            sum_logprob = float(target_log_probs.sum().item())

            # Greedy match: argmax at each continuation position
            # equals the continuation token.
            greedy_tokens = relevant.argmax(dim=-1)
            is_greedy = bool(torch.equal(greedy_tokens, cont_tensor))

            results.append((sum_logprob, is_greedy))
        return results

    def _loglikelihood_rolling_batch(self, batch):
        """Sum log-probs across every token of each request's string.

        Returns one ``float`` per request (NOT a tuple) so the result
        is compatible with the lm_eval ``loglikelihood_rolling``
        protocol: it appends each element to ``req.resps`` and
        downstream code (e.g. WikiText perplexity) does
        ``(loglikelihood,) = results`` — unpacking a 1-tuple here
        would corrupt the metric tuples downstream.
        """
        results = []
        for req in batch:
            (text,) = req.args
            ids = self._encode(text)[: self.max_length]
            if len(ids) < 2:
                results.append(0.0)
                continue
            input_tensor = torch.tensor([ids], dtype=torch.long, device=self.device)
            model_out = self.model(input_tensor, use_cache=False)
            logits = model_out[0] if isinstance(model_out, tuple) else model_out
            # Log-probs at positions 0..len-1 predict ids 1..len.
            relevant = logits[0, :-1, :]
            log_probs = torch.log_softmax(relevant, dim=-1)
            targets = torch.tensor(ids[1:], device=self.device, dtype=torch.long)
            token_log_probs = log_probs[torch.arange(len(targets), device=self.device), targets]
            results.append(float(token_log_probs.sum().item()))
        return results

    def _generate_until_batch(self, batch):
        """Greedy ``generate_until`` for a batch of requests."""
        results = []
        for req in batch:
            context, gen_kwargs = req.args
            until = gen_kwargs.get("until", [])
            max_gen_toks = int(gen_kwargs.get("max_gen_toks", 64))

            ctx_ids = self._encode(context)
            generated: list[int] = []
            for _ in range(max_gen_toks):
                full = (ctx_ids + generated)[-self.max_length :]
                input_tensor = torch.tensor([full], dtype=torch.long, device=self.device)
                model_out = self.model(input_tensor, use_cache=False)
                logits = model_out[0] if isinstance(model_out, tuple) else model_out
                next_token = int(logits[0, -1, :].argmax(dim=-1).item())
                generated.append(next_token)
                # Stop if the suffix matches any ``until`` token sequence.
                if self._matches_any_suffix(generated, until):
                    break

            results.append(self.tokenizer.decode(generated))
        return results

    # --- helpers ------------------------------------------------------------

    def _encode(self, text: str) -> list[int]:
        """Encode text using the bound tokenizer, with BOS handling."""
        ids = self.tokenizer.encode(text)
        if not isinstance(ids, list):
            ids = ids.tolist() if hasattr(ids, "tolist") else list(ids)
        return list(ids)

    @staticmethod
    def _matches_any_suffix(generated: list[int], until: list) -> bool:
        """Return True if any of the ``until`` token sequences is a suffix of ``generated``.

        ``until`` entries are expected to be token-id sequences
        (``list[int]``). String entries are skipped silently because
        we can't match by text without a tokenizer round-trip here
        — callers that want string-based stopping should tokenize
        them into id sequences first.
        """
        if not until:
            return False
        for stop in until:
            if isinstance(stop, str):
                # lm_eval sometimes passes strings; we don't decode here,
                # so we can't match by text. Skip string stops rather
                # than failing the whole generate call.
                continue
            stop_ids = list(stop) if not isinstance(stop, list) else stop
            if not stop_ids:
                continue
            if len(generated) >= len(stop_ids) and generated[-len(stop_ids) :] == stop_ids:
                return True
        return False

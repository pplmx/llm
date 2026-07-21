"""AdaLoRA (Adaptive LoRA) module.

SVD-form parameter-efficient fine-tuning. The increment matrix is
parameterized as ``ΔW = P · diag(λ · mask) · Q`` where P, Q are
orthonormalized on every forward (QR decomposition) and λ is a
learnable diagonal. A buffer ``mask`` lets the future pruning slice
zero out low-importance components without a public-API break.

Reference: Zhang et al., 2023 — *Adaptive Budget Allocation for
Parameter-Efficient Fine-Tuning*, arXiv:2303.10512.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import cast

import torch
import torch.nn as nn


def _orthonormalize(matrix: torch.Tensor) -> torch.Tensor:
    """Return the Q factor of the QR decomposition of ``matrix``.

    For an ``(m, n)`` input, ``torch.linalg.qr(mode='reduced')`` returns
    Q of shape ``(m, min(m, n))`` with orthonormal columns. The returned
    matrix has the same dtype and device as the input.
    """
    Q, _ = torch.linalg.qr(matrix, mode="reduced")  # noqa: N806
    return Q


class AdaLoRALinear(nn.Module):
    """AdaLoRA-adapted linear layer with SVD-form parameterization.

    The increment matrix is parameterized as ``ΔW = P · diag(λ · mask)
    · Q`` where:

    - ``P: (out_features, init_rank)`` — left singular vectors, trainable.
    - ``Q: (init_rank, in_features)`` — right singular vectors, trainable.
    - ``λ: (init_rank,)`` — singular values, trainable.
    - ``mask: (init_rank,)`` — binary mask registered as a buffer (not
      a Parameter) so the future pruning slice can write to it
      without breaking autograd.

    Forward pass:

    1. Orthonormalize P via QR → ``P̃``.
    2. Orthonormalize Q via QR on ``Qᵀ`` → ``Q̃`` (rows of Q̃ are
       orthonormal — i.e. ``Q̃ Q̃ᵀ = I``).
    3. Compute ``ΔW = P̃ · diag(λ · mask) · Q̃`` of shape
       ``(out_features, in_features)``.
    4. Return ``base(x) + scaling · x · ΔWᵀ``.

    At initialization ``λ = 0`` so ``ΔW = 0`` exactly — the layer
    behaves identically to the base layer, matching LoRA's
    zero-initialized-B invariant.

    Args:
        base_layer: The original ``nn.Linear`` to adapt (will be frozen).
        init_rank: Initial rank budget (upper bound on the number of
            singular components). Must satisfy ``init_rank ≤
            min(in_features, out_features)`` so QR can preserve the
            full column / row count.
        target_rank: Final target rank after pruning. Stored on the
            layer so the future pruning slice can consult it. This
            foundation slice does **not** prune — the mask starts
            all-ones.
        alpha: Scaling factor. Forward scales ``ΔW`` by
            ``alpha / init_rank`` (the same convention LoRA uses for
            ``alpha / rank``).
        dropout: Dropout probability for the LoRA path. ``0.0`` (default)
            keeps the layer fully deterministic.
        orth_reg_weight: Default weight for the orthogonality
            regularization. Stored on the layer so trainers can read
            it without hard-coding; the loss itself is opt-in (call
            :meth:`orth_reg_loss` and add to the total loss).
    """

    def __init__(
        self,
        base_layer: nn.Linear,
        init_rank: int = 12,
        target_rank: int | None = None,
        alpha: float = 32.0,
        dropout: float = 0.0,
        orth_reg_weight: float = 0.5,
    ):
        super().__init__()
        in_features = base_layer.in_features
        out_features = base_layer.out_features

        if init_rank <= 0:
            raise ValueError(f"init_rank must be positive, got {init_rank}")
        if init_rank > min(in_features, out_features):
            raise ValueError(
                f"init_rank ({init_rank}) must be ≤ "
                f"min(in_features={in_features}, out_features={out_features}) "
                "so QR decomposition can preserve the full column/row count."
            )
        if target_rank is not None:
            if target_rank <= 0:
                raise ValueError(f"target_rank must be positive, got {target_rank}")
            if target_rank > init_rank:
                raise ValueError(f"target_rank ({target_rank}) must be ≤ init_rank ({init_rank})")

        self.base_layer = base_layer
        self.init_rank = init_rank
        self.target_rank = target_rank if target_rank is not None else init_rank // 2
        self.alpha = alpha
        self.scaling = alpha / init_rank
        self.orth_reg_weight = orth_reg_weight

        device = base_layer.weight.device
        dtype = base_layer.weight.dtype

        # SVD-form parameters. P and Q are overwritten by their
        # orthonormalized versions on every forward, so the persistent
        # values stored in the state-dict are best understood as
        # "raw" coefficients — they're not used directly in forward.
        self.lora_P = nn.Parameter(torch.empty(out_features, init_rank, device=device, dtype=dtype))
        self.lora_Q = nn.Parameter(torch.empty(init_rank, in_features, device=device, dtype=dtype))
        self.lora_lambda = nn.Parameter(torch.empty(init_rank, device=device, dtype=dtype))

        # Pruning mask. Registered as a buffer (not Parameter) because
        # it is not trained by gradient descent — the future pruning
        # slice updates it based on importance scores. ``persistent=True``
        # so it travels through state-dict and checkpoint load/save.
        self.register_buffer(
            "mask",
            torch.ones(init_rank, device=device, dtype=dtype),
            persistent=True,
        )

        self.lora_dropout = nn.Dropout(p=dropout) if dropout > 0 else nn.Identity()

        # Init per the paper: P and Q random Gaussian, Λ zero.
        # Resulting ΔW = P · 0 · Q = 0, matching LoRA's zero-B invariant.
        nn.init.normal_(self.lora_P, mean=0.0, std=0.02)
        nn.init.normal_(self.lora_Q, mean=0.0, std=0.02)
        nn.init.zeros_(self.lora_lambda)

        # Freeze base layer.
        self.base_layer.weight.requires_grad = False
        if self.base_layer.bias is not None:
            self.base_layer.bias.requires_grad = False

    @property
    def effective_rank(self) -> int:
        """Number of currently active (masked-in) components.

        Reads through ``self.mask`` so the future pruning slice can
        drive this property by zeroing mask entries — no other
        bookkeeping needed.
        """
        mask = cast(torch.Tensor, self.mask)
        return int(mask.sum().item())

    @property
    def trainable_parameters(self) -> int:
        """Number of trainable AdaLoRA parameters (P + Q + λ)."""
        return self.lora_P.numel() + self.lora_Q.numel() + self.lora_lambda.numel()

    def _orthonormalized_P(self) -> torch.Tensor:  # noqa: N802
        """Orthonormal columns of P via QR decomposition.

        ``P`` has shape ``(out_features, init_rank)``. With
        ``init_rank ≤ out_features`` (validated in ``__init__``) the
        reduced-QR returns Q of shape ``(out_features, init_rank)``
        with orthonormal columns.
        """
        return _orthonormalize(self.lora_P)

    def _orthonormalized_Q(self) -> torch.Tensor:  # noqa: N802
        """Orthonormal rows of Q via QR on the transpose.

        We want ``Q · Qᵀ = I_{init_rank}``. Computing QR on ``Qᵀ``
        orthonormalizes its columns, which are the rows of Q in
        transposed form. We then transpose back to recover a
        ``(init_rank, in_features)`` tensor whose rows are orthonormal.
        """
        # ``Qᵀ`` has shape ``(in_features, init_rank)``; with
        # ``init_rank ≤ in_features`` (validated) reduced-QR returns
        # Q of shape ``(in_features, init_rank)`` whose columns (i.e.
        # the rows of the transposed-back tensor) are orthonormal.
        Q_ortho_T = _orthonormalize(self.lora_Q.T)  # noqa: N806
        return Q_ortho_T.T

    def _effective_increment(self) -> torch.Tensor:
        """Compute ``ΔW = P̃ · diag(λ · mask) · Q̃``.

        Returns:
            Tensor of shape ``(out_features, in_features)``.
        """
        P_ortho = self._orthonormalized_P()  # noqa: N806
        Q_ortho = self._orthonormalized_Q()  # noqa: N806
        # Broadcast (out, rank) * (rank,) → (out, rank), then matmul
        # with Q_ortho (rank, in) → (out, in).
        mask = cast(torch.Tensor, self.mask)
        scaled = self.lora_lambda * mask
        return (P_ortho * scaled) @ Q_ortho

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass: ``base(x) + scaling · x · ΔWᵀ``."""
        base_output = self.base_layer(x)
        if self.scaling == 0.0:
            return base_output
        delta_w = self._effective_increment()
        lora_output = self.lora_dropout(x) @ delta_w.T
        return base_output + lora_output * self.scaling

    def orth_reg_loss(self) -> torch.Tensor:
        """Orthogonality regularization loss.

        Per the AdaLoRA paper, the loss penalises deviation from
        orthonormality::

            ||P^T P - I||_F^2 + ||Q Q^T - I||_F^2

        Returns a scalar (sum of both terms). Trainers add this to the
        task loss (scaled by :attr:`orth_reg_weight`) to keep P and Q
        well-conditioned during training.

        The loss is exactly zero when P and Q are themselves
        orthonormal, which the QR step on every forward already
        enforces up to floating-point noise.
        """
        P_ortho = self._orthonormalized_P()  # noqa: N806
        Q_ortho = self._orthonormalized_Q()  # noqa: N806
        identity = torch.eye(self.init_rank, device=P_ortho.device, dtype=P_ortho.dtype)
        loss_p = torch.linalg.norm(P_ortho.T @ P_ortho - identity, ord="fro") ** 2
        loss_q = torch.linalg.norm(Q_ortho @ Q_ortho.T - identity, ord="fro") ** 2
        return loss_p + loss_q

    def merge_weights(self) -> None:
        """Merge ``ΔW`` into the base layer for efficient inference.

        After calling this, the layer's forward path is redundant —
        ``base_layer`` already carries the full delta. The companion
        :meth:`unmerge_weights` reverses it for continued training.
        """
        with torch.no_grad():
            delta_w = self._effective_increment()
            self.base_layer.weight.add_(delta_w * self.scaling)

    def unmerge_weights(self) -> None:
        """Unmerge ``ΔW`` from the base layer."""
        with torch.no_grad():
            delta_w = self._effective_increment()
            self.base_layer.weight.sub_(delta_w * self.scaling)

    def compute_importance_scores(self, gradient_ema: torch.Tensor | None = None) -> torch.Tensor:
        """Per-component importance scores, one per singular value.

        Per the AdaLoRA paper (Algorithm 1, line 7), the combined
        importance score is ``|λ_i| · |∂L/∂λ_i|``. Trainers compute
        the EMA of ``|∂L/∂λ_i|`` themselves (the optimizer owns
        gradient statistics), then pass it here.

        Args:
            gradient_ema: Optional EMA tensor of shape
                ``(init_rank,)`` holding ``|∂L/∂λ_i|`` averages.
                ``None`` falls back to magnitude-only scoring
                ``(|λ_i|)``, which is enough to rank components when
                the trainer does not track gradients.

        Returns:
            Tensor of shape ``(init_rank,)`` with one score per
            component. Components with higher scores carry more of
            the model's expressive capacity and should be kept under
            pruning.
        """
        magnitude = self.lora_lambda.abs()
        if gradient_ema is None:
            return magnitude
        if gradient_ema.shape != magnitude.shape:
            raise ValueError(
                f"gradient_ema shape {tuple(gradient_ema.shape)} does not "
                f"match lora_lambda shape {tuple(magnitude.shape)}"
            )
        return magnitude * gradient_ema.abs()

    def prune_to_rank(
        self,
        target_rank: int,
        scores: torch.Tensor | None = None,
    ) -> None:
        """Zero out mask entries for the lowest-importance components.

        Mutates :attr:`mask` in-place so that exactly ``target_rank``
        entries remain ``1.0`` and the rest are ``0.0``. The kept
        entries are the ``target_rank`` components with **highest**
        importance score (see :meth:`compute_importance_scores`).

        Args:
            target_rank: Number of components to keep. Must satisfy
                ``0 ≤ target_rank ≤ self.effective_rank`` — un-pruning
                is not supported (the dropped λ entries have been
                overwritten by the optimizer and cannot be recovered
                in-place).
            scores: Optional pre-computed importance scores of shape
                ``(init_rank,)``. When ``None``, falls back to
                :meth:`compute_importance_scores` with default
                magnitude-only scoring. Pass scores explicitly when
                wiring gradient-EMA scoring through a trainer.

        Raises:
            ValueError: if ``target_rank`` is out of range or larger
                than the current ``effective_rank``.
        """
        if target_rank < 0:
            raise ValueError(f"target_rank must be ≥ 0, got {target_rank}")
        if target_rank > self.effective_rank:
            raise ValueError(
                f"target_rank ({target_rank}) exceeds effective_rank "
                f"({self.effective_rank}); un-pruning is not supported"
            )
        if target_rank == self.effective_rank:
            # Already at the requested rank (or below); nothing to do.
            return

        if scores is None:
            scores = self.compute_importance_scores()
        # `topk` with largest=True returns the indices of the
        # highest-scoring components. Build a fresh mask from those
        # indices — this is robust to repeated calls (idempotent:
        # pruning to k twice yields the same mask, modulo ties).
        _, keep_indices = torch.topk(scores, k=target_rank, largest=True)
        mask = cast(torch.Tensor, self.mask)
        new_mask = torch.zeros_like(mask)
        new_mask[keep_indices] = 1.0
        mask.copy_(new_mask)

    def update_budget(
        self,
        current_step: int,
        tinit: int,
        tfinal: int,
    ) -> int:
        """Return the rank budget for the current training step.

        Linear schedule from ``init_rank`` at ``tinit`` to
        ``target_rank`` at ``tfinal``. Useful for periodic pruning
        during fine-tuning: train at full rank through warmup, then
        gradually reallocate the budget down to ``target_rank``.

        Args:
            current_step: The current training step (≥ 0).
            tinit: Step at and before which the budget is held at
                ``init_rank``. The first pruning-eligible step is
                ``tinit + 1``.
            tfinal: Step at and after which the budget is held at
                ``target_rank``. Must be strictly greater than
                ``tinit``.

        Returns:
            Integer rank budget to use for this step. Round to
            ``int`` to keep the mask-integer contract.

        Raises:
            ValueError: if ``current_step < 0`` or ``tinit >= tfinal``.
        """
        if current_step < 0:
            raise ValueError(f"current_step must be ≥ 0, got {current_step}")
        if tinit >= tfinal:
            raise ValueError(f"tinit ({tinit}) must be strictly less than tfinal ({tfinal})")
        if current_step <= tinit:
            return self.init_rank
        if current_step >= tfinal:
            return self.target_rank
        progress = (current_step - tinit) / (tfinal - tinit)
        return round(self.init_rank - progress * (self.init_rank - self.target_rank))

    def extra_repr(self) -> str:
        return (
            f"init_rank={self.init_rank}, target_rank={self.target_rank}, "
            f"alpha={self.alpha}, scaling={self.scaling:.4f}, "
            f"effective_rank={self.effective_rank}"
        )


# --- Helper surface (mirrors llm.core.lora) --------------------------------


def apply_adalora(
    model: nn.Module,
    init_rank: int = 12,
    target_rank: int | None = None,
    alpha: float = 32.0,
    dropout: float = 0.0,
    target_modules: list[str] | None = None,
    orth_reg_weight: float = 0.5,
) -> nn.Module:
    """Apply AdaLoRA to specified linear layers in a model.

    Mirrors :func:`llm.core.lora.apply_lora` so swapping LoRA → AdaLoRA
    in user code is a one-import change. ``target_rank`` is stored on
    each layer but does **not** trigger pruning in this foundation
    slice — that lands in the follow-up.

    Args:
        model: The model to adapt. Modified in-place.
        init_rank: Initial rank budget.
        target_rank: Final target rank after pruning (deferred).
        alpha: Scaling factor (``alpha / init_rank``).
        dropout: Dropout probability for the AdaLoRA path.
        target_modules: List of module-name substring patterns. If
            ``None``, every ``nn.Linear`` is replaced.
        orth_reg_weight: Default weight for orthogonality regularization.

    Returns:
        The same model, modified in-place.
    """
    if target_modules is None:
        target_modules = []

    def should_apply(name: str) -> bool:
        if not target_modules:
            return True
        return any(pattern in name for pattern in target_modules)

    replacements: list[tuple[str, nn.Linear]] = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and should_apply(name):
            replacements.append((name, module))

    for name, module in replacements:
        adalora_layer = AdaLoRALinear(
            module,
            init_rank=init_rank,
            target_rank=target_rank,
            alpha=alpha,
            dropout=dropout,
            orth_reg_weight=orth_reg_weight,
        )
        parts = name.split(".")
        parent = model
        for part in parts[:-1]:
            parent = getattr(parent, part)
        setattr(parent, parts[-1], adalora_layer)

    return model


def merge_adalora(model: nn.Module) -> nn.Module:
    """Merge all AdaLoRA deltas into the corresponding base layers."""
    for module in model.modules():
        if isinstance(module, AdaLoRALinear):
            module.merge_weights()
    return model


def unmerge_adalora(model: nn.Module) -> nn.Module:
    """Unmerge all AdaLoRA deltas from the corresponding base layers."""
    for module in model.modules():
        if isinstance(module, AdaLoRALinear):
            module.unmerge_weights()
    return model


def get_adalora_parameters(model: nn.Module) -> Iterator[nn.Parameter]:
    """Yield every AdaLoRA ``P``, ``Q``, and ``λ`` parameter in the model.

    Trainers pass this to the optimizer so only the AdaLoRA path is
    updated — the base weights stay frozen.
    """
    for module in model.modules():
        if isinstance(module, AdaLoRALinear):
            yield module.lora_P
            yield module.lora_Q
            yield module.lora_lambda


def count_adalora_parameters(model: nn.Module) -> tuple[int, int]:
    """Return ``(trainable_params, total_params)`` for a model with AdaLoRA.

    Trainable count follows whatever ``requires_grad`` is set on every
    parameter — the base layer weights are frozen at construction, so
    ``trainable_params`` will equal the sum of all AdaLoRA
    ``trainable_parameters`` plus any other unfrozen parameters the
    caller has chosen to add (e.g. norms).
    """
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total


def disable_adalora(model: nn.Module) -> None:
    """Disable AdaLoRA by setting scaling to 0 (model falls back to base)."""
    for module in model.modules():
        if isinstance(module, AdaLoRALinear):
            module._original_scaling = module.scaling  # type: ignore[attr-defined]
            module.scaling = 0.0


def enable_adalora(model: nn.Module) -> None:
    """Re-enable AdaLoRA after :func:`disable_adalora`."""
    for module in model.modules():
        if isinstance(module, AdaLoRALinear):
            original = getattr(module, "_original_scaling", None)
            if original is not None:
                module.scaling = original


def prune_adalora(
    model: nn.Module,
    target_rank: int | None = None,
    schedule: tuple[int, int] | None = None,
    current_step: int | None = None,
    gradient_emas: dict[int, torch.Tensor] | None = None,
) -> None:
    """Walk every ``AdaLoRALinear`` in ``model`` and prune to a target rank.

    Two calling modes:

    1. **Explicit target rank**::

        prune_adalora(model, target_rank=8)

       Every AdaLoRALinear layer is pruned to ``target_rank``
       (subject to the per-layer ``effective_rank`` upper bound).

    2. **Budget schedule**::

        prune_adalora(model, schedule=(tinit, tfinal), current_step=step)

       Each layer is pruned to the rank that
       ``layer.update_budget(step, tinit, tfinal)`` returns — i.e.
       the budget is re-evaluated per call, so a training loop can
       invoke this helper periodically and the rank shrinks over time.

    Args:
        model: Model containing one or more ``AdaLoRALinear`` layers.
        target_rank: Explicit rank to prune every layer to. Mutually
            exclusive with ``schedule``.
        schedule: ``(tinit, tfinal)`` tuple for a linear rank-budget
            schedule. Must be paired with ``current_step``.
        current_step: Current training step, used only when
            ``schedule`` is given.
        gradient_emas: Optional dict mapping ``id(layer)`` to a
            gradient-EMA tensor of shape ``(init_rank,)`` for that
            layer's λ. Passed through to
            :meth:`AdaLoRALinear.compute_importance_scores` so the
            trainer can supply ``|∂L/∂λ_i|`` averages.

    Raises:
        ValueError: if neither ``target_rank`` nor ``schedule`` is
            provided, or if ``schedule`` is given without
            ``current_step``.
    """
    if (target_rank is None) == (schedule is None):
        raise ValueError(
            "prune_adalora requires exactly one of target_rank or "
            "schedule=(tinit, tfinal); got "
            f"target_rank={target_rank!r}, schedule={schedule!r}"
        )

    layers = [m for m in model.modules() if isinstance(m, AdaLoRALinear)]
    if not layers:
        return

    if target_rank is not None:
        for layer in layers:
            rank = min(target_rank, layer.effective_rank)
            scores = (
                layer.compute_importance_scores(gradient_emas.get(id(layer))) if gradient_emas is not None else None
            )
            layer.prune_to_rank(rank, scores=scores)
        return

    # Schedule branch. The mutually-exclusive check at the top of
    # this function guarantees ``schedule`` is not None here.
    assert schedule is not None  # noqa: S101
    if current_step is None:
        raise ValueError("prune_adalora with schedule=(tinit, tfinal) requires current_step to be provided")
    tinit, tfinal = schedule
    for layer in layers:
        rank = layer.update_budget(current_step, tinit, tfinal)
        scores = layer.compute_importance_scores(gradient_emas.get(id(layer))) if gradient_emas is not None else None
        layer.prune_to_rank(rank, scores=scores)


# --- Gradient-EMA tracker (trainer-side half of Algorithm 1) --------------


class AdaLoRAGradientEMA:
    """Per-layer EMA of ``|∂L/∂λ|`` for AdaLoRA's importance scoring.

    Implements the EMA half of AdaLoRA Algorithm 1
    (Zhang et al. 2023, page 4)::

        I_avg_i <- alpha * I_avg_i + (1 - alpha) * |dL/dlambda_i|

    The tracker is constructed against a model that already has
    ``AdaLoRALinear`` layers in place. After each backward pass, the
    trainer calls :meth:`update` to fold the current gradient into the
    EMA. The result is consumed by :func:`prune_adalora` (via
    :meth:`as_dict`) to weight components by their combined
    ``|λ| · |∂L/∂λ|`` score during pruning.

    State is checkpointable through :meth:`state_dict` /
    :meth:`load_state_dict`, both keyed by the layer's **qualified
    name** (``"0"``, ``"layer1.attn"``, ...). ``id(layer)`` is *not*
    stable across pickle roundtrips, so the checkpoint key must be a
    structural path.

    Args:
        model: The model whose ``AdaLoRALinear`` layers will be
            tracked. Walked via :meth:`nn.Module.named_modules`, so the
            DDP/FSDP unwrap path is just ``model.modules()``.
        alpha: EMA smoothing factor (the weight on the *previous* EMA).
            ``alpha=0.95`` matches the paper's recommendation. Must
            satisfy ``0 < alpha < 1``.

    Raises:
        ValueError: if ``alpha`` is outside ``(0, 1)``.
    """

    def __init__(self, model: nn.Module, alpha: float = 0.95):
        if not (0.0 < alpha < 1.0):
            raise ValueError(f"alpha must be in (0, 1), got {alpha}")
        self.alpha = alpha
        self._current_step_weight = 1.0 - alpha

        # Walk named_modules so we capture the qualified name once at
        # construction. The same walk also picks up the AdaLoRALinear
        # in case the user is passing a sub-module (e.g. just the
        # decoder body); every AdaLoRALinear reachable from ``model``
        # gets its own EMA tensor.
        self._layers: dict[str, AdaLoRALinear] = {}
        self._emas: dict[str, torch.Tensor] = {}
        for name, module in model.named_modules():
            if isinstance(module, AdaLoRALinear):
                self._layers[name] = module
                # Match the layer's dtype and device so a subsequent
                # ``ema + grad_abs`` does not silently promote / move.
                params = list(module.parameters())
                ref = params[0] if params else module.mask
                self._emas[name] = torch.zeros(module.init_rank, dtype=ref.dtype, device=ref.device)

    def update(self) -> None:
        """Fold ``|∂L/∂λ|`` from each layer's last backward into the EMA.

        Layers whose ``lora_lambda.grad`` is ``None`` (frozen, or no
        path through them this step) are left untouched — only the
        components training actually drove get smoothed in.
        """
        for name, layer in self._layers.items():
            grad = layer.lora_lambda.grad
            if grad is None:
                continue
            ema = self._emas[name]
            ema.mul_(self.alpha).add_(grad.abs(), alpha=self._current_step_weight)

    def as_dict(self) -> dict[int, torch.Tensor]:
        """Return ``{id(layer): ema_tensor}`` for :func:`prune_adalora`.

        The trainer passes this directly as ``gradient_emas=`` to
        :func:`prune_adalora`. ``id()`` is fine here because the
        tracker and the prune call share the same Python process; it
        is **only** the checkpoint roundtrip that needs a stable key.
        """
        return {id(layer): self._emas[name] for name, layer in self._layers.items()}

    def state_dict(self) -> dict[str, torch.Tensor]:
        """Return a serializable snapshot keyed by qualified name.

        Tensors are detached so a subsequent ``load_state_dict`` on a
        fresh tracker doesn't keep a reference to the autograd graph.
        """
        return {name: ema.detach().clone() for name, ema in self._emas.items()}

    def load_state_dict(self, state: dict[str, torch.Tensor] | None) -> None:
        """Restore EMA tensors from a :meth:`state_dict` snapshot.

        Unknown keys (e.g. from a stale checkpoint where the model has
        since been pruned of a layer) are silently ignored. ``None``
        is a no-op so a fresh checkpoint doesn't crash the trainer.
        """
        if not state:
            return
        for name, tensor in state.items():
            target = self._emas.get(name)
            if target is None:
                # Stale key — layer was renamed or removed. Skip.
                continue
            # Move + cast defensively so a checkpoint saved on a
            # different device/dtype still loads without surprises.
            target.copy_(tensor.to(device=target.device, dtype=target.dtype))

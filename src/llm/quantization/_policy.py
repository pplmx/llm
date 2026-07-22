"""Per-layer quantization policy — algorithm-agnostic.

LayerQuantPolicy binds a set of target layer names to a bundle of override
fields (bits / group_size / sym / act_order). All override fields are
optional; None means "inherit from the algorithm's base config".

This module is intentionally algorithm-agnostic: the four override fields
are the public subset shared by all PTQ-style quantization algorithms
(GPTQ today, AWQ / SmoothQuant / QAT in future slices). The
`resolve_layer_policies` helper is generic over the base config dataclass,
so future algorithms reuse it without modification.

See ADR-008 and docs/superpowers/specs/2026-07-22-mixed-precision-quantization-design.md.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

# Type variable used by `resolve_layer_policies[T]`. Algorithm-agnostic
# helper accepts any dataclass with the four override fields
# (bits, group_size, sym, act_order).
type _BaseConfig = Any  # type: ignore[misc]


@dataclass(frozen=True)
class LayerQuantPolicy:
    """Atomic per-layer quantization override policy. Algorithm-agnostic.

    A LayerQuantPolicy binds a set of target layer names to a bundle of
    override fields. Fields set to None mean "inherit from the algorithm's
    base config". Multiple LayerQuantPolicy in a config are additive; each
    target module must appear in at most one policy (overlap raises
    ValueError at resolve time).

    Field semantics are universal across PTQ-style algorithms:
        bits:       4 or 8 — quantization bit-width
        group_size: -1 (per-channel) or positive int — quantization group size
        sym:        True (symmetric) or False (asymmetric)
        act_order:  True (sort columns by diag(H) descending) or False

    Attributes:
        target_modules: Tuple of fully-qualified layer names (dotted notation,
            matching the `target_modules` arg style of quantize_model_gptq).
        bits: Override bit-width (None = inherit).
        group_size: Override group size (None = inherit).
        sym: Override symmetry (None = inherit).
        act_order: Override act-order (None = inherit).
    """

    target_modules: tuple[str, ...]
    bits: int | None = None
    group_size: int | None = None
    sym: bool | None = None
    act_order: bool | None = None

    def __post_init__(self):
        # target_modules: non-empty, no duplicates within one policy
        if not self.target_modules:
            raise ValueError("LayerQuantPolicy.target_modules cannot be empty; specify at least one layer name.")
        if len(set(self.target_modules)) != len(self.target_modules):
            duplicates = sorted({n for n in self.target_modules if list(self.target_modules).count(n) > 1})
            raise ValueError(f"LayerQuantPolicy.target_modules has duplicates within a single policy: {duplicates}.")
        # bits: None or {4, 8}
        if self.bits is not None and self.bits not in (4, 8):
            raise ValueError(f"LayerQuantPolicy.bits must be 4, 8, or None (inherit); got {self.bits}.")
        # group_size: None, -1, or positive int
        if self.group_size is not None:
            if not isinstance(self.group_size, int) or isinstance(self.group_size, bool):
                raise ValueError(
                    f"LayerQuantPolicy.group_size must be int or None; got {type(self.group_size).__name__}."
                )
            if self.group_size != -1 and self.group_size <= 0:
                raise ValueError(
                    f"LayerQuantPolicy.group_size must be -1 (per-channel) or positive; got {self.group_size}."
                )
        # sym / act_order: None or bool (dataclass rejects other types at
        # construction, but we keep the explicit check for symmetric error
        # messages with group_size/bits paths)
        if self.sym is not None and not isinstance(self.sym, bool):
            raise ValueError(f"LayerQuantPolicy.sym must be bool or None; got {type(self.sym).__name__}.")
        if self.act_order is not None and not isinstance(self.act_order, bool):
            raise ValueError(f"LayerQuantPolicy.act_order must be bool or None; got {type(self.act_order).__name__}.")


def resolve_layer_policies[T](
    policies: tuple[LayerQuantPolicy, ...],
    available_names: set[str],
    base_config: T,
) -> dict[str, T]:
    """Build layer-name -> effective config map from policies.

    Generic over the base config type (T). Works for GPTQConfig today and
    any future algorithm config (AWQConfig, SmoothQuantConfig, ...) as long
    as the base config dataclass has the four override fields
    (bits, group_size, sym, act_order).

    Args:
        policies: Tuple of LayerQuantPolicy to resolve. Empty tuple is a no-op.
        available_names: Set of layer names that are actually eligible to be
            quantized (typically the post-`target_modules` filter set).
        base_config: The base algorithm config to inherit from.

    Returns:
        Dict mapping each policy-targeted layer name to its effective config
        (= base_config with non-None policy fields applied via
        `dataclasses.replace`). Empty dict if no policies.

    Raises:
        ValueError: If any policy targets a name not in `available_names`,
            or if the same layer name appears in multiple policies.
    """
    if not policies:
        return {}

    # Phase 1: validate every policy's targets exist in available_names,
    # AND collect into a single map (later writes win on intra-iteration
    # duplicates; cross-policy overlap is detected in Phase 2).
    name_to_policy: dict[str, LayerQuantPolicy] = {}
    for i, policy in enumerate(policies):
        unmatched = set(policy.target_modules) - available_names
        if unmatched:
            sample_available = sorted(available_names)[:10]
            more = "..." if len(available_names) > 10 else ""
            raise ValueError(
                f"LayerQuantPolicy[{i}].target_modules {sorted(unmatched)} "
                f"not found in available layers. Available: "
                f"{sample_available}{more}"
            )
        for name in policy.target_modules:
            name_to_policy[name] = policy

    # Phase 2: detect cross-policy overlaps (fail-fast).
    target_counts: dict[str, int] = {}
    for policy in policies:
        for name in policy.target_modules:
            target_counts[name] = target_counts.get(name, 0) + 1
    duplicates = sorted(n for n, c in target_counts.items() if c > 1)
    if duplicates:
        raise ValueError(
            f"LayerQuantPolicy.target_modules overlap detected across "
            f"policies: {duplicates}. Each layer name must appear in at "
            f"most one policy."
        )

    # Phase 3: build effective configs (base + non-None overrides).
    effective_map: dict[str, T] = {}
    for name, policy in name_to_policy.items():
        overrides: dict[str, object] = {}
        if policy.bits is not None:
            overrides["bits"] = policy.bits
        if policy.group_size is not None:
            overrides["group_size"] = policy.group_size
        if policy.sym is not None:
            overrides["sym"] = policy.sym
        if policy.act_order is not None:
            overrides["act_order"] = policy.act_order
        # Strip recursion-vector field if base config has one (e.g.
        # GPTQConfig.layer_policies). Effective configs must NOT carry the
        # policies that produced them, or infinite recursion ensues.
        # We explicitly set it to () (not just remove from overrides) so
        # `dataclasses.replace` resets it instead of preserving the base's
        # value.
        if hasattr(base_config, "layer_policies"):
            overrides["layer_policies"] = ()
        # `base_config` is T (any dataclass by convention); the `replace()` signature
        # bounds its first arg to `DataclassInstance` (typeshed-only Protocol not
        # available at runtime in Python 3.14). Suppress: the helper is documented
        # as requiring a dataclass with the four override fields.
        effective_map[name] = replace(base_config, **overrides)  # ty: ignore[invalid-argument-type]

    return effective_map

"""Generic plugin registry for runtime extensibility."""

from __future__ import annotations

from collections.abc import Callable
from typing import TypeVar

T = TypeVar("T")


class Registry[T]:
    """Name-to-object registry with explicit registration and lookup."""

    def __init__(self, name: str) -> None:
        self._name = name
        self._entries: dict[str, T] = {}
        self._warmup: Callable[[], None] | None = None
        self._warming: bool = False

    def set_warmup(self, warmup: Callable[[], None]) -> None:
        """Attach an idempotent bootstrap that runs before the first read.

        Lazy registries declare their built-ins in a separate ``ensure_*``
        function so that importing the registry does not pull in every
        implementation. That works for code that remembers to call it, but
        ``names()`` and ``get()`` are public API: a user following the docs
        (``PEFT_REGISTRY.names()``) or probing an unknown name saw an EMPTY
        registry and a misleading ``Available: `` message, concluding nothing
        was available.

        The bootstrap is expected to be idempotent (the ``ensure_*`` functions
        already are, with double-checked locking), so calling it on every read
        is safe and keeps the "import stays light" property: it runs on first
        READ, never at import time.
        """
        self._warmup = warmup

    def _ensure_warm(self) -> None:
        """Run the bootstrap once, without recursing back into itself.

        The ``ensure_*`` functions are NOT re-entrant: they hold a lock and
        call ``register``/``__contains__`` on this same registry. Now that
        ``__contains__`` triggers the bootstrap, an unguarded call deadlocks on
        the non-reentrant lock (observed: ``PEFT_REGISTRY.names()`` hung until
        killed). Clearing ``_warmup`` for the duration turns the inner read into
        a plain lookup, which is exactly the pre-existing semantics.
        """
        warmup = self._warmup
        if warmup is None or self._warming:
            return
        self._warming = True
        try:
            warmup()
        finally:
            self._warming = False

    def has_raw(self, name: str) -> bool:
        """Membership test that never triggers the warm-up bootstrap.

        Internal infrastructure (``load_entry_point_registry``) reads a registry
        precisely WHILE its ``ensure_*`` bootstrap is populating it. Routing that
        read through ``__contains__`` re-entered ``ensure_*`` and deadlocked on
        the non-reentrant registration lock — observed as ``import
        llm.serving`` hanging forever, because ``ServingConfig`` validates
        ``generation_backend`` at module import.
        """
        return name in self._entries

    def register(self, name: str, entry: T) -> T:
        if name in self._entries:
            raise ValueError(f"'{name}' is already registered in {self._name} registry.")
        self._entries[name] = entry
        return entry

    def replace(self, name: str, entry: T) -> T:
        """Register ``entry`` under ``name``, overwriting any existing entry.

        Unlike :meth:`register`, this does not raise on a duplicate — it is
        the explicit-override path used by
        :func:`llm.runtime.plugins.load_entry_point_registry(overwrite=True)`
        so third-party plugins can override a built-in implementation. For an unknown name it behaves exactly like ``register``.
        """
        self._entries[name] = entry
        return entry

    def get(self, name: str) -> T:
        self._ensure_warm()
        if name not in self._entries:
            available = ", ".join(sorted(self._entries))
            raise ValueError(f"'{name}' not found in {self._name} registry. Available: {available}")
        return self._entries[name]

    def names(self) -> list[str]:
        self._ensure_warm()
        return sorted(self._entries)

    def __contains__(self, name: str) -> bool:
        self._ensure_warm()
        return name in self._entries


def decorator_register(registry: Registry[type]) -> Callable[[str], Callable[[type], type]]:
    """Class decorator factory compatible with legacy register_model usage."""

    def register(name: str) -> Callable[[type], type]:
        def wrapper(cls: type) -> type:
            registry.register(name, cls)
            return cls

        return wrapper

    return register

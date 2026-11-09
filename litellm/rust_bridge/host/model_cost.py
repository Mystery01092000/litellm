"""The Python host of the model cost table: ``litellm.model_cost``, its row type, every write litellm makes to it and
the version that moves with each write. Rust asks it where a registration lands and how rows merge (shims until
lookup moves to Rust), and writes registrations through it.

Anything derived from the table is cached under ``model_cost_version()``, so a write is never followed by a read of
what was derived before it. The version also carries the table's identity and size, so a caller that swaps
``litellm.model_cost`` or adds or removes a key without coming here still moves it. Changing a value in place
without coming here does not.

Contract for every function Rust calls here:
- Rust calls it synchronously with the GIL held and no ``NativeModelCostMap`` lock held, since ``resolve`` reaches
  ``find_key`` through ``get_model_info``.
- Values cross as Python objects. Rows and registrations can hold values JSON cannot (``datetime`` in router
  ``model_info``), so nothing here is converted to JSON.
- Exceptions propagate to the ``register_model`` caller unchanged.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from functools import lru_cache
from typing import Final, Generic, ParamSpec, TypeVar

P = ParamSpec("P")
R = TypeVar("R")


class AliasedCostEntry(dict[str, object]):
    """A ``litellm.model_cost`` row, the one object every alias name of a model is served under."""

    __slots__ = ("names", "registered_fields")

    def __init__(
        self, entry: Mapping[str, object], names: tuple[str, ...], registered_fields: frozenset[str] = frozenset()
    ) -> None:
        super().__init__(entry)
        self.names: tuple[str, ...] = names
        self.registered_fields: frozenset[str] = registered_fields


@dataclass(frozen=True, slots=True)
class ModelCostVersion:
    table: int
    size: int
    writes: int


class _Writes:
    __slots__ = ("count",)

    def __init__(self) -> None:
        self.count: int = 0

    def record(self) -> None:
        self.count += 1


_WRITES: Final = _Writes()


def model_cost_version() -> ModelCostVersion:
    import litellm

    table: Final = litellm.model_cost
    return ModelCostVersion(table=id(table), size=len(table), writes=_WRITES.count)


def changed_in_place() -> None:
    """Moves the version after a caller changed ``litellm.model_cost`` in place without going through here."""
    _WRITES.record()


class VersionCached(Generic[P, R]):
    """``function`` cached for the ``litellm.model_cost`` version each call read, with ``lru_cache``'s
    ``cache_clear`` and ``cache_info``."""

    __slots__ = ("_at", "cache_clear", "cache_info")

    def __init__(self, function: Callable[P, R], maxsize: int) -> None:
        def at(
            _version: ModelCostVersion,
            *args: P.args,
            **kwargs: P.kwargs,  # kwargs-ok: forwards a ParamSpec unchanged
        ) -> R:
            return function(*args, **kwargs)

        cached: Final = lru_cache(maxsize=maxsize)(at)
        self._at: Final = cached
        self.cache_clear: Final = cached.cache_clear
        self.cache_info: Final = cached.cache_info

    def __call__(self, *args: P.args, **kwargs: P.kwargs) -> R:  # kwargs-ok: forwards a ParamSpec unchanged
        return self._at(model_cost_version(), *args, **kwargs)


def cached_per_version(maxsize: int) -> Callable[[Callable[P, R]], VersionCached[P, R]]:
    return lambda function: VersionCached(function, maxsize)


def publish(table: dict[str, object]) -> None:
    """Makes ``table`` the live ``litellm.model_cost``."""
    import litellm

    litellm.model_cost = table
    _WRITES.record()


@dataclass(frozen=True, slots=True)
class Landing:
    key: str
    base: Mapping[str, object]
    matched_builtin: bool


def entry(fields: dict[str, object], names: tuple[str, ...], registered_fields: frozenset[str]) -> AliasedCostEntry:
    """The row object every alias name of one model shares in ``litellm.model_cost``."""
    return AliasedCostEntry(fields, names, registered_fields)


def resolve(
    rows: Mapping[str, Mapping[str, object]],
    key: str,
    registration: Mapping[str, object],
    custom_llm_provider: str | None,
) -> Landing | None:
    """Where a ``register_model`` entry lands in ``rows`` and the row it starts from. ``None`` when ``rows`` is not
    the published table, because Python lookups read only ``litellm.model_cost``, so the caller retries on the new
    table.

    Shim: this is lookup logic, not host state. It stays here until ``get_model_info`` moves to Rust, when its body
    becomes a native call. ``model_cost_registration_row`` asks the same question through ``registration_base``,
    so there is one resolver for both callers."""
    import litellm
    from litellm.utils import registration_base

    if rows is not litellm.model_cost:
        return None
    base = registration_base(key, registration, custom_llm_provider)
    return Landing(base.key, base.entry, base.matched_builtin)


def merged(
    current: Mapping[str, object] | None, base: Mapping[str, object], registration: Mapping[str, object]
) -> AliasedCostEntry:
    """The row a registration writes over ``current``, the row now under its landing key.

    Shim: the merge rule belongs in core, but rows hold Python objects JSON cannot, so it stays here until core has a
    merge generic over the value type."""
    from litellm.utils import model_cost_registration_write

    return model_cost_registration_write(current, base, registration)


def kept(previous: dict[str, object] | None, registration: Mapping[str, object]) -> dict[str, object]:
    """The registration ``register_model(persist_across_reloads=True)`` keeps for a key: ``registration`` merged
    into the one kept before it, as main merged them."""
    from litellm.utils import update_dictionary

    return update_dictionary(dict(previous or {}), registration)


def write(rows: dict[str, object], written: Mapping[str, object]) -> None:
    """Writes one registration's rows into ``rows``, the published table. Rust calls it once per entry, because the
    next entry's lookup reads through caches keyed on the version this moves."""
    rows.update(written)
    _WRITES.record()

from typing import Final

import pytest

import litellm
from litellm.litellm_core_utils.get_model_cost_map import AliasedCostEntry
from litellm.rust_bridge.host.model_cost import changed, entry, merged, resolve
from litellm.utils import get_model_cost_mutation_generation

pytestmark = pytest.mark.usefixtures("local_model_cost_map")


def test_entry_is_the_row_type_alias_names_share() -> None:
    row: Final = entry({"mode": "chat"}, ("owner", "alias"), frozenset({"mode"}))

    assert isinstance(row, AliasedCostEntry)
    assert row == {"mode": "chat"}
    assert (row.names, row.registered_fields) == (("owner", "alias"), frozenset({"mode"}))


def test_resolve_lands_on_the_row_a_lookup_in_the_published_table_resolves() -> None:
    landing: Final = resolve(litellm.model_cost, "openai/gpt-4o", {"input_cost_per_token": 1.0}, None)

    assert landing is not None
    assert (landing.key, landing.matched_builtin) == ("gpt-4o", True)


def test_resolve_does_not_answer_for_a_table_that_is_not_published() -> None:
    assert resolve(dict(litellm.model_cost), "openai/gpt-4o", {"input_cost_per_token": 1.0}, None) is None


def test_merged_keeps_the_alias_names_and_adds_only_the_fields_a_registration_set() -> None:
    current: Final = entry({"mode": "chat", "x": 1}, ("owner", "alias"), frozenset({"mode"}))

    row: Final = merged(current, {"x": 2}, {"litellm_provider": "openai", "output_cost_per_token": None})

    assert row == {"mode": "chat", "x": 2, "litellm_provider": "openai"}
    assert (row.names, row.registered_fields) == (("owner", "alias"), frozenset({"mode", "litellm_provider"}))


def test_changed_moves_what_python_derives_from_the_table() -> None:
    before: Final = get_model_cost_mutation_generation()

    changed()

    assert get_model_cost_mutation_generation() == before + 1

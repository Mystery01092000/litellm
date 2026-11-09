from typing import Final

import pytest

import litellm
from litellm.cost_calculator import cost_per_token
from litellm.rust_bridge.host.model_cost import (
    AliasedCostEntry,
    entry,
    merged,
    model_cost_version,
    publish,
    resolve,
    write,
)
from litellm.utils import cached_cost_map_key

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


def test_write_lands_the_rows_and_moves_the_version_a_value_change_alone_would_not() -> None:
    row: Final = litellm.model_cost["gpt-4o"]
    before: Final = model_cost_version()

    write(litellm.model_cost, {"gpt-4o": {**row, "input_cost_per_token": 0.5}})

    assert litellm.model_cost["gpt-4o"]["input_cost_per_token"] == 0.5
    assert model_cost_version() != before


_MODEL: Final = "lit-table-model"


def _row(price: float) -> dict[str, object]:
    return {"litellm_provider": "openai", "mode": "chat", "input_cost_per_token": price, "output_cost_per_token": price}


def _billed() -> float:
    prompt, completion = cost_per_token(
        model=_MODEL, custom_llm_provider="openai", prompt_tokens=1, completion_tokens=0
    )
    return prompt + completion


def test_a_published_table_is_read_in_place_of_what_was_cached_for_the_one_before() -> None:
    publish({**litellm.model_cost, _MODEL: _row(1.0)})
    assert litellm.get_model_info(_MODEL)["input_cost_per_token"] == 1.0
    assert _billed() == 1.0

    publish({**litellm.model_cost, _MODEL: _row(2.0)})

    assert litellm.get_model_info(_MODEL)["input_cost_per_token"] == 2.0
    assert _billed() == 2.0


def test_rows_written_into_the_live_table_replace_a_cached_value_of_the_same_size() -> None:
    publish({**litellm.model_cost, _MODEL: _row(1.0)})
    assert _billed() == 1.0

    write(litellm.model_cost, {_MODEL: _row(3.0)})

    assert litellm.get_model_info(_MODEL)["input_cost_per_token"] == 3.0
    assert _billed() == 3.0


def test_a_table_swapped_in_without_publishing_is_still_read(monkeypatch: pytest.MonkeyPatch) -> None:
    assert cached_cost_map_key(_MODEL, "openai") is None

    monkeypatch.setattr(litellm, "model_cost", {**litellm.model_cost, _MODEL: _row(4.0)})

    assert cached_cost_map_key(_MODEL, "openai") == _MODEL
    assert _billed() == 4.0


def test_a_key_added_in_place_without_publishing_is_found(monkeypatch: pytest.MonkeyPatch) -> None:
    assert cached_cost_map_key(_MODEL, "openai") is None

    monkeypatch.setitem(litellm.model_cost, _MODEL, _row(5.0))

    assert cached_cost_map_key(_MODEL, "openai") == _MODEL
    assert _billed() == 5.0

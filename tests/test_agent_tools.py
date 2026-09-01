"""Smoke tests for agent-callable tools (registry names only, no training)."""

from agent.schemas import CALLABLE_TOOL_NAMES
from agent.tools import call_tool, get_registry, register_default_tools


def test_registry_contains_documented_tools():
    register_default_tools()
    registered = set(get_registry())
    required = {
        "acquire_data",
        "preprocess_covid",
        "prepare_forecast_dataset",
        "validate_inputs",
        "train_model",
        "forecast_single_target",
        "explain_target_iz_with_geoshapley",
    }
    assert required.issubset(registered)
    assert set(CALLABLE_TOOL_NAMES) == registered


def test_unknown_tool_raises():
    try:
        call_tool("not_a_real_tool")
    except KeyError as error:
        assert "not_a_real_tool" in str(error)
    else:
        raise AssertionError("expected KeyError")

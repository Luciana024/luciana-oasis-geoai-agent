"""Stop when required user choices are missing. Do not invent them."""

from __future__ import annotations

from agent.instructions import (
    CANDIDATE_SITES_SOURCE_PROMPT,
    SOURCE_PROMPT,
    TRAVEL_TIME_SOURCE_PROMPT,
    YEAR_PROMPT,
)
from agent.state import AgentState


def request_approval(state: AgentState) -> dict[str, str]:
    prompts: dict[str, str] = {}
    if "years" in state.missing_parameters:
        prompts["years"] = YEAR_PROMPT
    if "data_source" in state.missing_parameters:
        if state.task == "travel_time_prepare":
            prompts["data_source"] = TRAVEL_TIME_SOURCE_PROMPT
        elif state.task == "candidate_sites_prepare":
            prompts["data_source"] = CANDIDATE_SITES_SOURCE_PROMPT
        else:
            prompts["data_source"] = SOURCE_PROMPT
    return prompts


def must_stop_for_user(state: AgentState) -> bool:
    return bool(state.missing_parameters)

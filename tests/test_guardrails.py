from agent import interpret_request
from agent.guardrails import must_stop_for_user, request_approval


def test_guardrails_stop_when_year_and_source_missing():
    state = interpret_request("请合并爱丁堡COVID数据")
    assert must_stop_for_user(state)
    prompts = request_approval(state)
    assert "years" in prompts
    assert "data_source" in prompts

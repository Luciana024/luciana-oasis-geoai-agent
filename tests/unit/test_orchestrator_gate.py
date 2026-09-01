from agent import run_plan


def test_orchestrator_asks_for_year_and_source():
    result = run_plan("请做数据合并和缺失值补全")
    assert result["status"] == "awaiting_user"
    assert "years" in result["missing_parameters"]
    assert "data_source" in result["missing_parameters"]
    assert "years" in result["prompts"]
    assert "data_source" in result["prompts"]


def test_orchestrator_asks_for_source_when_only_year_given():
    result = run_plan({"years": [2023], "request": "merge 2023"})
    assert result["status"] == "awaiting_user"
    assert result["missing_parameters"] == ["data_source"]

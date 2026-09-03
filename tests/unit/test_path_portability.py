"""Repository paths stored in frozen artefacts must remain portable."""

import json

from common.utils import project_relative_path, project_root, resolve_project_path, write_json


def test_relative_project_path_resolves_from_repository_root():
    relative = "data/results/regions/S12000049/planning/latest.json"
    assert resolve_project_path(relative) == project_root() / relative
    assert resolve_project_path(relative).is_file()


def test_missing_legacy_absolute_path_is_relocated_into_repository():
    legacy = (
        "/old/author/machine/oasis_geoai_agent/"
        "data/results/regions/S12000049/forecast_for_allocation.csv"
    )
    expected = project_root() / "data/results/regions/S12000049/forecast_for_allocation.csv"
    assert resolve_project_path(legacy) == expected
    assert expected.is_file()


def test_local_absolute_path_serialises_as_repository_relative():
    target = project_root() / "data/results/planning/latest.json"
    assert project_relative_path(target) == "data/results/planning/latest.json"


def test_write_json_removes_local_repository_prefix(tmp_path):
    target = project_root() / "data/results/planning/latest.json"
    output = tmp_path / "metadata.json"
    write_json({"nested": {"path": str(target)}}, output)
    assert json.loads(output.read_text(encoding="utf-8")) == {
        "nested": {"path": "data/results/planning/latest.json"}
    }

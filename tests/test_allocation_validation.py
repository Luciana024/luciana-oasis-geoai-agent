from allocation.validate import validate_allocation_result
from common.errors import ModelError
import pytest


def test_not_wired_is_not_valid_for_display():
    checked = validate_allocation_result(
        {"status": "not_wired", "diagnostics": {"message": "unwired"}, "invented": False}
    )
    assert checked["valid"] is False
    assert checked["n_sites_selected"] == 0


def test_unknown_site_id_is_rejected():
    with pytest.raises(ModelError) as error:
        validate_allocation_result(
            {
                "status": "ok",
                "invented": False,
                "selected_sites": [{"site_id": f"GP_{i}"} for i in range(6)],
            },
            candidate_site_ids=["PH_1"],
        )
    assert error.value.code == "invented_sites"

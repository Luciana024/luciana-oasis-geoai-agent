"""Validate allocator output against candidate sites and IZ codes."""

from __future__ import annotations

from typing import Any, Iterable

from allocation.contracts import N_SITES, empty_allocation_result
from common.errors import ModelError


def validate_allocation_result(
    result: dict[str, Any],
    *,
    candidate_site_ids: Iterable[str] | None = None,
    iz_codes: Iterable[str] | None = None,
) -> dict[str, Any]:
    """Fail closed if the solver invented ids or skipped the six-site contract."""
    if result.get("status") == "not_wired":
        return {
            "status": "not_wired",
            "valid": False,
            "n_sites_selected": 0,
            "message": result.get("diagnostics", {}).get("message"),
        }
    if result.get("invented"):
        raise ModelError(
            "Allocator marked invented=True. Refusing to display invented sites.",
            code="invented_sites",
        )
    selected = [str(item.get("site_id") if isinstance(item, dict) else item) for item in result.get("selected_sites") or []]
    allowed_sites = set(map(str, candidate_site_ids or []))
    allowed_iz = set(map(str, iz_codes or []))
    if allowed_sites:
        unknown = [site_id for site_id in selected if site_id not in allowed_sites]
        if unknown:
            raise ModelError(
                f"Allocator returned site_id values not in the candidate table: {unknown[:10]}.",
                code="invented_sites",
            )
    if len(selected) != N_SITES and result.get("status") == "ok":
        raise ModelError(
            f"Prototype requires exactly {N_SITES} sites; allocator returned {len(selected)}.",
            code="invalid_allocation",
        )
    if allowed_iz:
        assigned = []
        for row in result.get("assignments") or []:
            if isinstance(row, dict) and row.get("iz_code"):
                assigned.append(str(row["iz_code"]))
        unknown_iz = [code for code in assigned if code not in allowed_iz]
        if unknown_iz:
            raise ModelError(
                f"Allocator assigned unknown iz_code values: {unknown_iz[:10]}.",
                code="invented_iz",
            )
    return {
        "status": "ok",
        "valid": True,
        "n_sites_selected": len(selected),
        "site_ids": selected,
    }

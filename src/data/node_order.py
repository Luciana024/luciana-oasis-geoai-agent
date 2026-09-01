"""Canonical IZ order and hashes.

See docs/model.md section 4. COVID node_index is the authoritative order.
Tensors, graphs, embeddings, forecasts, GeoShapley, and map tables must share
the same IZ sequence. On mismatch, stop. Do not silently reorder.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import pandas as pd

from model.constants import CANONICAL_HASH_JOIN, LEGACY_HASH_JOIN
from common.errors import ModelError
from common.utils import EXPECTED_IZ_COUNT, NODE_KEY


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_node_order_hash(codes: Sequence[str]) -> str:
    """SHA256 of newline-joined IZ codes. Same algorithm as the mobility report hash."""
    payload = CANONICAL_HASH_JOIN.join(str(code) for code in codes).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def legacy_node_order_hash(codes: Sequence[str]) -> str:
    """SHA256 of pipe-joined IZ codes used by forecast.py. Metadata only after migration."""
    payload = LEGACY_HASH_JOIN.join(str(code) for code in codes).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True)
class NodeOrder:
    codes: tuple[str, ...]
    node_index: tuple[int, ...]
    canonical_hash: str
    legacy_hash: str

    @property
    def n_nodes(self) -> int:
        return len(self.codes)

    def as_dict(self) -> dict[str, Any]:
        return {
            "n_nodes": self.n_nodes,
            "iz_codes": list(self.codes),
            "node_index": list(self.node_index),
            "canonical_node_order_hash": self.canonical_hash,
            "legacy_node_order_hash": self.legacy_hash,
            "node_order_hash_algorithm": "SHA256",
            "node_order_hash_join": "newline",
        }


def validate_node_table(nodes: pd.DataFrame, *, source: str) -> pd.DataFrame:
    if NODE_KEY not in nodes.columns or "node_index" not in nodes.columns:
        raise ModelError(
            f"{source} is missing {NODE_KEY} or node_index.",
            code="node_order_mismatch",
        )
    table = nodes[[NODE_KEY, "node_index"]].copy()
    table[NODE_KEY] = table[NODE_KEY].astype(str)
    table["node_index"] = pd.to_numeric(table["node_index"], errors="coerce")
    if table["node_index"].isna().any():
        raise ModelError(f"{source} has non-integer node_index.", code="node_order_mismatch")
    table["node_index"] = table["node_index"].astype(int)
    table = table.sort_values("node_index").reset_index(drop=True)
    codes = table[NODE_KEY].tolist()
    if any(not code or code.lower() == "nan" for code in codes):
        raise ModelError(f"{source} has empty IZ codes.", code="node_order_mismatch")
    if len(set(codes)) != len(codes):
        raise ModelError(f"{source} has duplicate IZ codes.", code="node_order_mismatch")
    if table["node_index"].tolist() != list(range(len(table))):
        raise ModelError(
            f"{source} node_index must be unique contiguous integers 0..N-1.",
            code="node_order_mismatch",
        )
    return table


def node_order_from_codes(codes: Sequence[str]) -> NodeOrder:
    ordered = tuple(str(code) for code in codes)
    if len(set(ordered)) != len(ordered):
        raise ModelError("IZ codes are not unique.", code="node_order_mismatch")
    if any(not code for code in ordered):
        raise ModelError("IZ codes contain empty values.", code="node_order_mismatch")
    return NodeOrder(
        codes=ordered,
        node_index=tuple(range(len(ordered))),
        canonical_hash=canonical_node_order_hash(ordered),
        legacy_hash=legacy_node_order_hash(ordered),
    )


def load_node_order(path: Path) -> NodeOrder:
    table = validate_node_table(pd.read_csv(path), source=str(path))
    return node_order_from_codes(table[NODE_KEY].astype(str).tolist())


def assert_same_node_order(
    left: NodeOrder,
    right: NodeOrder,
    *,
    left_name: str,
    right_name: str,
) -> None:
    if left.codes != right.codes:
        raise ModelError(
            f"IZ sequence mismatch: {left_name} vs {right_name}. Silent reordering is forbidden.",
            code="node_order_mismatch",
            details={
                "left_n": left.n_nodes,
                "right_n": right.n_nodes,
                "left_hash": left.canonical_hash,
                "right_hash": right.canonical_hash,
            },
        )
    if left.canonical_hash != right.canonical_hash:
        raise ModelError(
            f"Canonical node-order hash mismatch: {left_name} vs {right_name}.",
            code="node_order_mismatch",
        )


def assert_edinburgh_count(n_nodes: int, expected: int = EXPECTED_IZ_COUNT) -> None:
    """Edinburgh data check. Not a layer constant."""
    if n_nodes != expected:
        raise ModelError(
            f"Edinburgh node count is {n_nodes}, expected {expected}.",
            code="node_order_mismatch",
            details={"n_nodes": n_nodes, "expected": expected},
        )

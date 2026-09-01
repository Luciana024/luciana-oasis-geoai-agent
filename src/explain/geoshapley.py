"""Target-IZ-local GeoShapley with location-feature interactions.

See docs/model.md section 13 and Li (2024).

Method reused: 128 coalitions, joint location player, Shapley kernel weights
for 0 < s < n. Not reused: official 1e8 endpoint weights, background-mean
baseline, or tabular predict_f.

phi0 = f(empty). The full decomposition reconstructs f(observed) by equality
constraint. Other IZs stay at observed values. Scalers stay frozen.
Without coordinates, location and location_x_* are omitted; that path must
not be called GeoShapley.
"""

from __future__ import annotations

import itertools
from math import comb
from typing import Any, Callable

import numpy as np

from model.constants import (
    EXPLANATION_SCOPE,
    FEATURE_PLAYER_NAMES,
    INTERACTION_LABEL,
    INTERACTION_PLAYER_NAMES,
    LOCATION_LABEL,
    LOCATION_PLAYER,
)
from common.errors import LEVEL_REVIEW, ModelWarning


def shapley_kernel_weight(n_players: int, coalition_size: int) -> float:
    """Kernel SHAP weight for 0 < s < n. Endpoints are equality constraints, not huge weights."""
    if coalition_size <= 0 or coalition_size >= n_players:
        raise ValueError("Shapley kernel weights are undefined at the empty and full coalitions.")
    return (n_players - 1) / (comb(n_players, coalition_size) * coalition_size * (n_players - coalition_size))


def study_area_median_reference(features: np.ndarray, coords: np.ndarray | None) -> tuple[np.ndarray, np.ndarray | None]:
    feature_ref = np.median(features, axis=0)
    coord_ref = None if coords is None else np.median(coords, axis=0)
    return feature_ref, coord_ref


def apply_coalition(
    features: np.ndarray,
    coords: np.ndarray | None,
    *,
    target_index: int,
    feature_mask: np.ndarray,
    location_in: bool,
    feature_reference: np.ndarray,
    coord_reference: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Copy the full matrices, then replace only the target IZ. Never edit in place across coalitions."""
    features_s = np.array(features, copy=True)
    for feature_index, present in enumerate(feature_mask):
        if present:
            continue
        features_s[target_index, feature_index] = feature_reference[feature_index]
    coords_s = None
    if coords is not None:
        coords_s = np.array(coords, copy=True)
        if not location_in:
            if coord_reference is None:
                raise ValueError("Location reference is required when coordinates exist.")
            coords_s[target_index] = coord_reference
    return features_s, coords_s


def _design_row(feature_mask: np.ndarray, location_in: bool) -> np.ndarray:
    """14 columns: intercept, location, 6 mains, 6 location x feature interactions."""
    z_loc = 1.0 if location_in else 0.0
    z = feature_mask.astype(np.float64)
    interactions = z_loc * z
    return np.concatenate(([1.0, z_loc], z, interactions))


def constrained_geoshapley_wls(
    coalitions: list[dict[str, Any]],
    *,
    n_features: int = 6,
    additivity_tolerance: float = 1e-6,
) -> dict[str, Any]:
    """Weighted least squares with exact empty/full constraints. No 1e8 endpoint weights."""
    empty = next(item for item in coalitions if item["s"] == 0)
    full = next(item for item in coalitions if item["s"] == n_features + 1)
    phi0 = float(empty["value"])
    f_full = float(full["value"])
    interior = [item for item in coalitions if 0 < item["s"] < n_features + 1]
    n_free = 1 + n_features + n_features  # location + mains + interactions
    design = np.stack([item["design"][1:] for item in interior], axis=0)
    target = np.asarray([item["value"] - phi0 for item in interior], dtype=np.float64)
    weights = np.asarray([item["weight"] for item in interior], dtype=np.float64)
    sqrt_w = np.sqrt(weights)
    ztwz = (design * sqrt_w[:, None]).T @ (design * sqrt_w[:, None])
    ztwy = (design * sqrt_w[:, None]).T @ (target * sqrt_w)
    # Equality: 1^T phi_free = f_full - phi0
    kkt = np.zeros((n_free + 1, n_free + 1), dtype=np.float64)
    kkt[:n_free, :n_free] = ztwz
    kkt[:n_free, n_free] = 1.0
    kkt[n_free, :n_free] = 1.0
    rhs = np.zeros(n_free + 1, dtype=np.float64)
    rhs[:n_free] = ztwy
    rhs[n_free] = f_full - phi0
    solved = np.linalg.solve(kkt, rhs)
    phi_free = solved[:n_free]
    phi_location = float(phi_free[0])
    phi_main = phi_free[1 : 1 + n_features]
    phi_interaction = phi_free[1 + n_features :]
    reconstructed = phi0 + phi_location + float(phi_main.sum()) + float(phi_interaction.sum())
    additivity_error = abs(reconstructed - f_full)
    warning = None
    if additivity_error > additivity_tolerance:
        warning = ModelWarning(
            code="geoshapley_additivity_exceeded",
            level=LEVEL_REVIEW,
            message="GeoShapley additivity residual exceeds the numeric tolerance.",
            details={"additivity_error": additivity_error, "tolerance": additivity_tolerance},
        )
    return {
        "phi0": phi0,
        "phi_location": phi_location,
        "phi_main": phi_main,
        "phi_interaction": phi_interaction,
        "reconstructed_prediction": reconstructed,
        "observed_prediction": f_full,
        "additivity_error": additivity_error,
        "warning": warning,
    }


def coalition_specs(n_features: int = 6) -> list[tuple[bool, np.ndarray]]:
    """All 2^(n_features+1) coalitions: (location_in, feature_mask)."""
    specs = []
    for bits in itertools.product([0, 1], repeat=n_features + 1):
        specs.append((bool(bits[0]), np.asarray(bits[1:], dtype=int)))
    return specs


def build_coalition_batch(
    features: np.ndarray,
    coords: np.ndarray,
    target_index: int,
) -> tuple[list[tuple[bool, np.ndarray]], np.ndarray, np.ndarray]:
    """Independent copies of the full matrices for every coalition of one target IZ."""
    feature_ref, coord_ref = study_area_median_reference(features, coords)
    specs = coalition_specs(features.shape[1])
    feature_batch = np.empty((len(specs),) + features.shape, dtype=np.float64)
    coord_batch = np.empty((len(specs),) + coords.shape, dtype=np.float64)
    for row, (location_in, feature_mask) in enumerate(specs):
        feat_s, coord_s = apply_coalition(
            features,
            coords,
            target_index=target_index,
            feature_mask=feature_mask,
            location_in=location_in,
            feature_reference=feature_ref,
            coord_reference=coord_ref,
        )
        feature_batch[row] = feat_s
        coord_batch[row] = coord_s
    return specs, feature_batch, coord_batch


def explanation_from_coalition_values(
    specs: list[tuple[bool, np.ndarray]],
    values: np.ndarray,
    *,
    n_features: int = 6,
    additivity_tolerance: float = 1e-6,
) -> dict[str, Any]:
    """Fit GeoShapley from 128 coalition predictions already evaluated."""
    coalitions: list[dict[str, Any]] = []
    for (location_in, feature_mask), value in zip(specs, values, strict=True):
        coalition_size = int(location_in) + int(feature_mask.sum())
        coalitions.append(
            {
                "s": coalition_size,
                "location_in": location_in,
                "feature_mask": feature_mask,
                "design": _design_row(feature_mask, location_in),
                "value": float(value),
                "weight": None
                if coalition_size in (0, n_features + 1)
                else shapley_kernel_weight(n_features + 1, coalition_size),
            }
        )
    fit = constrained_geoshapley_wls(
        coalitions,
        n_features=n_features,
        additivity_tolerance=additivity_tolerance,
    )
    rows = []
    for name, phi in zip(FEATURE_PLAYER_NAMES, fit["phi_main"], strict=True):
        rows.append({"player_name": name, "component": "main", "phi": float(phi)})
    rows.append(
        {
            "player_name": LOCATION_PLAYER,
            "component": "location",
            "phi": float(fit["phi_location"]),
            "label": LOCATION_LABEL,
        }
    )
    for name, phi in zip(INTERACTION_PLAYER_NAMES, fit["phi_interaction"], strict=True):
        rows.append(
            {
                "player_name": name,
                "component": "interaction",
                "phi": float(phi),
                "label": INTERACTION_LABEL,
            }
        )
    return {
        "explanation_scope": EXPLANATION_SCOPE,
        "phi_0": fit["phi0"],
        "reconstructed_prediction": fit["reconstructed_prediction"],
        "additivity_error": fit["additivity_error"],
        "components": rows,
        "n_coalitions": len(coalitions),
        "warning": None if fit["warning"] is None else fit["warning"].to_dict(),
    }


def explain_target_iz(
    features: np.ndarray,
    coords: np.ndarray | None,
    target_index: int,
    predict_mu: Callable[[np.ndarray, np.ndarray | None], float],
    *,
    additivity_tolerance: float = 1e-6,
) -> dict[str, Any]:
    """Full GeoShapley for one target IZ. Requires coordinates for location terms."""
    if coords is None:
        raise ValueError(
            "Coordinates are missing. Do not report location or location_x_* "
            "and do not call this 7-player Shapley GeoShapley."
        )
    specs, feature_batch, coord_batch = build_coalition_batch(features, coords, target_index)
    values = np.asarray(
        [predict_mu(feature_batch[i], coord_batch[i]) for i in range(len(specs))],
        dtype=np.float64,
    )
    return explanation_from_coalition_values(
        specs,
        values,
        n_features=features.shape[1],
        additivity_tolerance=additivity_tolerance,
    )

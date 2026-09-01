"""Load configs/model.yaml. Hyperparameters live in the config, not in layer code."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from common.errors import ModelError
from common.utils import load_yaml, project_root

CONFIG_RELATIVE = "configs/model.yaml"


def load_model_config(path: str | Path | None = None) -> dict[str, Any]:
    if path is None:
        return load_yaml(CONFIG_RELATIVE)
    with Path(path).open(encoding="utf-8") as handle:
        import yaml

        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Expected mapping in {path}")
    return data


def resolve_path(relative_or_absolute: str | Path) -> Path:
    path = Path(relative_or_absolute)
    if path.is_absolute():
        return path
    return project_root() / path


def config_path(cfg: dict[str, Any], key: str) -> Path:
    return resolve_path(cfg["paths"][key])


def temporal_target(cfg: dict[str, Any]) -> dict[str, Any]:
    """Single-target window: lookback_steps inputs -> one value at t+target_offset_days.

    output_steps must be 1. This is not an H-step sequence forecast.
    Frozen L7_H7_S1 files still record lookback_days / forecast_horizon_days;
    those names are only used to check the existing arrays.
    """
    block = cfg.get("temporal_target") or {}
    lookback = int(block.get("lookback_steps", 7))
    offset = int(block.get("target_offset_days", 7))
    output_steps = int(block.get("output_steps", 1))
    stride = int(block.get("window_stride_days", 1))
    if output_steps != 1:
        raise ModelError(
            f"output_steps must be 1 (one Y at t+{offset}), not {output_steps}. "
            "Do not build [B, H, N, 1] outputs.",
            code="config_mismatch",
        )
    if lookback < 1 or offset < 1 or stride < 1:
        raise ModelError(
            "lookback_steps, target_offset_days and stride must be positive.",
            code="invalid_config",
        )
    return {
        "lookback_steps": lookback,
        "target_offset_days": offset,
        "output_steps": output_steps,
        "window_stride_days": stride,
        "target_definition": block.get("target_definition", "residual_from_latest_report"),
    }


def n_geoshapley_indicator_players(cfg: dict[str, Any]) -> int:
    """Context feature count plus one joint location player when enabled."""
    n_features = len(cfg["context"]["feature_columns"])
    geo = cfg.get("geoshapley") or {}
    if not bool(geo.get("joint_location_player", True)):
        return n_features
    return n_features + 1


def geoshapley_settings(cfg: dict[str, Any]) -> dict[str, Any]:
    """Approved GeoShapley options. Indicator player count is derived, not hard-coded."""
    geo = cfg.get("geoshapley") or {}
    dates = str(geo.get("dates", "last"))
    if dates not in {"last", "all"}:
        raise ModelError(
            f"Unknown geoshapley.dates={dates}. Use 'all' or 'last'.",
            code="invalid_config",
        )
    if geo.get("reference", "study_area_median") != "study_area_median":
        raise ModelError("geoshapley.reference must be study_area_median.", code="invalid_config")
    if geo.get("explanation_scope", "target_iz_local") != "target_iz_local":
        raise ModelError("geoshapley.explanation_scope must be target_iz_local.", code="invalid_config")
    if geo.get("endpoint_constraints", "exact") != "exact":
        raise ModelError(
            "geoshapley.endpoint_constraints must be exact (empty/full equality), not 1e8 weights.",
            code="invalid_config",
        )
    if not bool(geo.get("joint_location_player", True)):
        raise ModelError("Approved GeoShapley requires a joint location player.", code="invalid_config")
    if not bool(geo.get("include_location_feature_interactions", True)):
        raise ModelError(
            "Approved GeoShapley requires location-feature interactions.",
            code="invalid_config",
        )
    n_features = len(cfg["context"]["feature_columns"])
    return {
        "reference": "study_area_median",
        "explanation_scope": "target_iz_local",
        "joint_location_player": True,
        "include_location_feature_interactions": True,
        "endpoint_constraints": "exact",
        "additivity_tolerance": float(geo.get("additivity_tolerance", 1.0e-6)),
        "dates": dates,
        "n_indicator_players": n_features + 1,
        "n_feature_players": n_features,
    }


def uncertainty_flag_settings(cfg: dict[str, Any]) -> dict[str, Any]:
    block = cfg.get("uncertainty_flag") or {}
    source = str(block.get("source_split", "validation_calibration"))
    if source != "validation_calibration":
        raise ModelError(
            "uncertainty_flag.source_split must be validation_calibration. "
            "Do not compute the σ quantile on test.",
            code="invalid_config",
        )
    return {
        "enabled": bool(block.get("enabled", True)),
        "quantile": float(block.get("quantile", 0.90)),
        "source_split": source,
        "require_calibration_available": bool(block.get("require_calibration_available", True)),
    }


def assert_operational_inference_off(cfg: dict[str, Any]) -> None:
    """Retrospective test export is not operational next-report-day inference."""
    block = cfg.get("operational_inference") or {}
    if bool(block.get("enabled", False)):
        raise ModelError(
            "Operational future inference must be exported separately. "
            "Keep operational_inference.enabled false for retrospective test outputs. "
            "'last' is the final retrospective test issue date, not a live forecast.",
            code="operational_inference_not_implemented",
        )

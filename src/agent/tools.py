"""Agent tools for the approved forecast/explanation model.

See docs/model.md section 16. Each tool returns status, outputs, warnings,
and provenance. These tools load existing S1 arrays and graph files; they
do not rebuild windows or adjacency.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from data.dataset import (
    STATIC_FEATURE_COLUMNS,
    TemporalDataset,
    load_temporal_dataset as _load_dataset,
    write_chronological_resplit,
)
from graph.diffusion import directed_supports, fuse_supports, fused_column_sum_warning
from common.errors import (
    LEVEL_CRITICAL,
    LEVEL_REVIEW,
    ModelError,
    ModelWarning,
    status_from_warnings,
)
from model.evaluate import (
    assert_artefact_matches_checkpoint,
    build_calibration_artefact,
    evaluate_split,
    predict_split,
)
from presentation.tables import (
    attach_observed_columns,
    build_embedding_table,
    build_forecast_table,
    build_geoshapley_table,
)
from explain.geoshapley import (
    build_coalition_batch,
    explain_target_iz,
    explanation_from_coalition_values,
)
from graph.supports import load_graph_bundle, load_projected_centroids, normalise_graph_set
from model.config import (
    assert_operational_inference_off,
    config_path,
    geoshapley_settings,
    load_model_config,
    temporal_target,
    uncertainty_flag_settings,
)
from model.constants import THREE_GRAPH_SET
from model.context import FrozenScaler, diagnose_embedding, fit_cross_section_scaler
from model.network import ForecastModel
from data.node_order import sha256_file
from model.residual import ResidualScalers, apply_residual_scalers, prepare_residual_dataset, reconstruct_rate_from_delta
from model.operational import (
    DEFAULT_CALIBRATION as OPERATIONAL_DEFAULT_CALIBRATION,
    DEFAULT_CHECKPOINT as OPERATIONAL_DEFAULT_CHECKPOINT,
    DEFAULT_OUTPUT_DIR as OPERATIONAL_DEFAULT_OUTPUT_DIR,
    run_operational_forecast as _run_operational_forecast,
)
from model.rolling import run_rolling_evaluation as _run_rolling_evaluation
from model.train import build_model_from_config, load_raw_checkpoint, resolve_torch_device, train_forecast_model
from common.utils import get_logger

LOGGER = get_logger("agent.tools")

_REGISTRY: dict[str, Any] = {}


def get_registry() -> dict[str, Any]:
    return _REGISTRY


def call_tool(name: str, **kwargs: Any) -> Any:
    """Call a registered agent tool by name."""
    register_default_tools()
    if name not in _REGISTRY:
        raise KeyError(f"Unknown tool {name!r}. Registered: {sorted(_REGISTRY)}")
    return _REGISTRY[name](**kwargs)


def register_default_tools() -> None:
    """Bind registered tools. Required user choices are never invented."""
    if _REGISTRY:
        return
    from data.covid import acquire_data, inventory_raw_datasets, preprocess_covid
    from data.dataset import prepare_forecast_dataset
    from data.candidate_sites import load_candidate_sites, prepare_candidate_sites
    from data.healthcare import acquire_healthcare_table, load_healthcare_layers
    from data.travel_time import export_iz_origins, load_travel_time, prepare_travel_time

    from agent.planning_tools import (
        check_model_compatibility,
        compare_allocation_scenarios,
        forecast_inference,
        generate_web_layers,
        get_site_iz_info,
        prepare_or_validate_region,
        run_location_allocation,
        select_checkpoint,
        trigger_new_region_training,
        trigger_rolling_update,
        validate_allocation_result,
    )

    _REGISTRY["acquire_data"] = acquire_data
    _REGISTRY["preprocess_covid"] = preprocess_covid
    _REGISTRY["inventory_raw_datasets"] = inventory_raw_datasets
    _REGISTRY["prepare_forecast_dataset"] = prepare_forecast_dataset
    _REGISTRY["prepare_travel_time"] = prepare_travel_time
    _REGISTRY["load_travel_time"] = load_travel_time
    _REGISTRY["prepare_candidate_sites"] = prepare_candidate_sites
    _REGISTRY["load_candidate_sites"] = load_candidate_sites
    _REGISTRY["load_healthcare_layers"] = load_healthcare_layers
    _REGISTRY["acquire_healthcare_table"] = acquire_healthcare_table
    _REGISTRY["export_iz_origins"] = export_iz_origins
    _REGISTRY["check_model_compatibility"] = check_model_compatibility
    _REGISTRY["prepare_or_validate_region"] = prepare_or_validate_region
    _REGISTRY["select_checkpoint"] = select_checkpoint
    _REGISTRY["forecast_inference"] = forecast_inference
    _REGISTRY["trigger_rolling_update"] = trigger_rolling_update
    _REGISTRY["trigger_new_region_training"] = trigger_new_region_training
    _REGISTRY["run_location_allocation"] = run_location_allocation
    _REGISTRY["validate_allocation_result"] = validate_allocation_result
    _REGISTRY["get_site_iz_info"] = get_site_iz_info
    _REGISTRY["compare_allocation_scenarios"] = compare_allocation_scenarios
    _REGISTRY["generate_web_layers"] = generate_web_layers
    _REGISTRY["validate_inputs"] = validate_inputs
    _REGISTRY["load_temporal_dataset"] = load_temporal_dataset
    _REGISTRY["build_graph_supports"] = build_graph_supports
    _REGISTRY["train_model"] = train_model
    _REGISTRY["run_rolling_evaluation"] = run_rolling_evaluation
    _REGISTRY["load_checkpoint"] = load_checkpoint
    _REGISTRY["forecast_single_target"] = forecast_single_target
    _REGISTRY["evaluate_test_period"] = evaluate_test_period
    _REGISTRY["explain_target_iz_with_geoshapley"] = explain_target_iz_with_geoshapley
    _REGISTRY["export_map_ready_results"] = export_map_ready_results
    _REGISTRY["export_operational_forecast"] = export_operational_forecast



def _dataset_from_cfg(cfg: dict[str, Any]) -> tuple[TemporalDataset, dict[str, Any]]:
    tt = temporal_target(cfg)
    dataset = _load_dataset(
        config_path(cfg, "dataset"),
        lookback_days=tt["lookback_steps"],
        forecast_horizon_days=tt["target_offset_days"],
        expected_iz_count=int(cfg.get("expected_edinburgh_iz_count", 111)),
        selection_fraction=float(cfg["validation_internal_split"]["selection_fraction"]),
    )
    return dataset, tt


def _envelope(
    warnings: list[ModelWarning],
    outputs: dict[str, Any],
    provenance: dict[str, Any],
    *,
    extra_status: str | None = None,
) -> dict[str, Any]:
    status = extra_status or status_from_warnings(warnings)
    return {
        "status": status,
        "outputs": outputs,
        "warnings": [item.to_dict() for item in warnings],
        "provenance": provenance,
    }


def _fail(error: ModelError, provenance: dict[str, Any] | None = None) -> dict[str, Any]:
    warning = ModelWarning(
        code=error.code,
        level=LEVEL_CRITICAL,
        message=str(error),
        details=error.details,
    )
    return _envelope([warning], {}, provenance or {}, extra_status="failed")


def _date_str(values: np.ndarray, index: int) -> str:
    return str(np.datetime_as_string(values[index], unit="D"))


def validate_inputs(config_path_override: str | Path | None = None) -> dict[str, Any]:
    """Check S1 tensors, node order, three graphs, SIMD, and centroids."""
    try:
        cfg = load_model_config(config_path_override)
        dataset, _tt = _dataset_from_cfg(cfg)
        graphs = load_graph_bundle(
            {
                "geo": config_path(cfg, "geo"),
                "transport": config_path(cfg, "transport"),
                "mobility": config_path(cfg, "mobility"),
            },
            canonical=dataset.node_order,
            graph_set=cfg.get("graph_set", THREE_GRAPH_SET),
            reports={
                "geo": config_path(cfg, "geo_report"),
                "transport": config_path(cfg, "transport_report"),
                "mobility": config_path(cfg, "mobility_report"),
            },
        )
        coords = load_projected_centroids(config_path(cfg, "road_nodes"), dataset.node_order)
        warnings: list[ModelWarning] = []
        for graph in graphs.values():
            warnings.extend(graph.warnings)
        _, simd_warnings = fit_cross_section_scaler(
            dataset.x_static_raw,
            STATIC_FEATURE_COLUMNS,
            epsilon=float(cfg["context"]["zero_variance_epsilon"]),
            ddof=int(cfg["context"]["scaler_ddof"]),
        )
        warnings.extend(simd_warnings)
        return _envelope(
            warnings,
            {
                "n_nodes": dataset.n_nodes,
                "canonical_node_order_hash": dataset.node_order.canonical_hash,
                "legacy_node_order_hash": dataset.node_order.legacy_hash,
                "graph_set": list(graphs),
                "n_static_features": int(dataset.x_static_raw.shape[1]),
                "coords_shape": list(coords.shape),
                "location_crs": cfg.get("location_crs"),
            },
            {
                "dataset": str(dataset.directory),
                "config_id": dataset.config_id,
                "validation_internal_split": dataset.internal_split_provenance,
            },
        )
    except ModelError as error:
        return _fail(error)


def load_temporal_dataset(config_path_override: str | Path | None = None) -> dict[str, Any]:
    """Load L7_H7_S1 without rebuilding windows."""
    try:
        cfg = load_model_config(config_path_override)
        dataset, tt = _dataset_from_cfg(cfg)
        shapes = {name: list(split.y_target_scaled.shape) for name, split in dataset.splits.items()}
        y_shape = shapes.get("test") or next(iter(shapes.values()))
        if len(y_shape) != 3 or y_shape[-1] != 1 or y_shape[-2] != dataset.n_nodes:
            raise ModelError(
                f"Expected y_target [B, N, 1], got {y_shape}. output_steps must be 1.",
                code="config_mismatch",
            )
        return _envelope(
            [],
            {
                "directory": str(dataset.directory),
                "n_nodes": dataset.n_nodes,
                "lookback_steps": tt["lookback_steps"],
                "target_offset_days": tt["target_offset_days"],
                "output_steps": tt["output_steps"],
                "window_stride_days": tt["window_stride_days"],
                "target_definition": tt["target_definition"],
                "frozen_dataset_lookback_days": dataset.lookback_days,
                "frozen_dataset_forecast_horizon_days": dataset.forecast_horizon_days,
                "target_shape": shapes,
                "canonical_node_order_hash": dataset.node_order.canonical_hash,
                "legacy_node_order_hash": dataset.node_order.legacy_hash,
            },
            {
                "config_id": dataset.config_id,
                "validation_internal_split": dataset.internal_split_provenance,
                "do_not_rebuild_windows": True,
            },
        )
    except ModelError as error:
        return _fail(error)


def write_resplit_dataset(
    config_path_override: str | Path | None = None,
    *,
    train_frac: float = 0.60,
    val_frac: float = 0.15,
    output_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Recut existing S1 windows 60/15/25. Does not rebuild forecast.py samples."""
    try:
        cfg = load_model_config(config_path_override)
        dataset, tt = _dataset_from_cfg(cfg)
        dest = Path(output_dir) if output_dir is not None else config_path(cfg, "dataset").parent / (
            config_path(cfg, "dataset").name
            + f"_split{int(round(train_frac * 100))}_{int(round(val_frac * 100))}_{int(round((1.0 - train_frac - val_frac) * 100))}"
        )
        summary = write_chronological_resplit(
            dataset,
            dest,
            train_frac=train_frac,
            val_frac=val_frac,
            lookback_days=tt["lookback_steps"],
        )
        return _envelope(
            [],
            summary,
            {
                "source_dataset": str(dataset.directory),
                "output_dir": str(Path(dest).resolve()),
                "windows_not_rebuilt": True,
                "did_not_overwrite_frozen_s1": True,
            },
        )
    except ModelError as error:
        return _fail(error)


def _supports_from_graphs(graphs: dict[str, Any], graph_set: tuple[str, ...]):
    fwd = []
    bwd = []
    warnings: list[ModelWarning] = []
    for name in graph_set:
        packed = directed_supports(graphs[name].adjacency, name=name)
        fwd.append(packed.s_fwd)
        bwd.append(packed.s_bwd)
    return fwd, bwd, warnings


def build_graph_supports(
    config_path_override: str | Path | None = None,
    graph_set: list[str] | None = None,
) -> dict[str, Any]:
    """Build T/S supports from existing raw adjacency. Do not overwrite graph files."""
    try:
        cfg = load_model_config(config_path_override)
        dataset, _tt = _dataset_from_cfg(cfg)
        requested = normalise_graph_set(graph_set or cfg.get("graph_set", THREE_GRAPH_SET))
        graphs = load_graph_bundle(
            {
                "geo": config_path(cfg, "geo"),
                "transport": config_path(cfg, "transport"),
                "mobility": config_path(cfg, "mobility"),
            },
            canonical=dataset.node_order,
            graph_set=requested,
            reports={
                "geo": config_path(cfg, "geo_report"),
                "transport": config_path(cfg, "transport_report"),
                "mobility": config_path(cfg, "mobility_report"),
            },
        )
        warnings: list[ModelWarning] = []
        for graph in graphs.values():
            warnings.extend(graph.warnings)
        fwd, bwd, _ = _supports_from_graphs(graphs, requested)
        uniform = np.full(len(requested), 1.0 / len(requested))
        fused_fwd = fuse_supports(fwd, uniform)
        fused_bwd = fuse_supports(bwd, uniform)
        for name, fused in (("fwd", fused_fwd), ("bwd", fused_bwd)):
            warning = fused_column_sum_warning(fused, name=f"S_{name}")
            if warning is not None:
                warnings.append(warning)
        return _envelope(
            warnings,
            {
                "graph_set": list(requested),
                "n_nodes": dataset.n_nodes,
                "support_shapes": [list(item.shape) for item in fwd],
                "alpha_placeholder_uniform": uniform.tolist(),
                "note": "Training learns one softmax alpha; this tool does not renormalise fused supports.",
            },
            {
                "canonical_node_order_hash": dataset.node_order.canonical_hash,
                "graph_hashes": {name: graphs[name].file_sha256 for name in requested},
            },
        )
    except ModelError as error:
        return _fail(error)


def _prepare_context(dataset: TemporalDataset, cfg: dict[str, Any], coords: np.ndarray | None):
    simd_scaler, simd_warnings = fit_cross_section_scaler(
        dataset.x_static_raw,
        STATIC_FEATURE_COLUMNS,
        epsilon=float(cfg["context"]["zero_variance_epsilon"]),
        ddof=int(cfg["context"]["scaler_ddof"]),
    )
    simd_scaled = simd_scaler.transform(dataset.x_static_raw)
    coord_scaler = None
    coords_scaled = None
    if coords is not None:
        coord_scaler, coord_warnings = fit_cross_section_scaler(
            coords,
            ("easting", "northing"),
            epsilon=float(cfg["context"]["zero_variance_epsilon"]),
            ddof=int(cfg["context"]["scaler_ddof"]),
        )
        simd_warnings.extend(coord_warnings)
        coords_scaled = coord_scaler.transform(coords)
    return simd_scaler, simd_scaled, coord_scaler, coords_scaled, simd_warnings


def train_model(config_path_override: str | Path | None = None) -> dict[str, Any]:
    """Train on S1 train; select checkpoint with validation_selection NLL only."""
    try:
        cfg = load_model_config(config_path_override)
        assert_operational_inference_off(cfg)
        dataset, _tt = _dataset_from_cfg(cfg)
        graph_set = normalise_graph_set(cfg.get("graph_set", THREE_GRAPH_SET))
        graphs = load_graph_bundle(
            {
                "geo": config_path(cfg, "geo"),
                "transport": config_path(cfg, "transport"),
                "mobility": config_path(cfg, "mobility"),
            },
            canonical=dataset.node_order,
            graph_set=graph_set,
            reports={
                "geo": config_path(cfg, "geo_report"),
                "transport": config_path(cfg, "transport_report"),
                "mobility": config_path(cfg, "mobility_report"),
            },
        )
        coords = load_projected_centroids(config_path(cfg, "road_nodes"), dataset.node_order)
        simd_scaler, simd_scaled, coord_scaler, coords_scaled, warnings = _prepare_context(dataset, cfg, coords)
        for graph in graphs.values():
            warnings.extend(graph.warnings)
        fwd, bwd, _ = _supports_from_graphs(graphs, graph_set)
        prepare_residual_dataset(dataset, cfg)
        device = resolve_torch_device(cfg=cfg)
        LOGGER.info("Training on %s", device)
        output_dir = config_path(cfg, "checkpoints") / "_".join(graph_set)
        result = train_forecast_model(
            dataset,
            fwd,
            bwd,
            simd_scaled,
            coords_scaled,
            graph_set=graph_set,
            graph_hashes={name: graphs[name].file_sha256 for name in graph_set},
            context_scaler=simd_scaler,
            coord_scaler=coord_scaler,
            output_dir=output_dir,
            config=cfg,
            device_name=str(device),
        )
        artefact = None
        checkpoint_path = Path(result["checkpoint_path"])
        payload = load_raw_checkpoint(checkpoint_path)
        model = build_model_from_config(cfg, n_graphs=len(graph_set), has_location=True)
        model.load_state_dict(payload["model_state_dict"])
        model.to(device)
        model.eval()
        flag_cfg = uncertainty_flag_settings(cfg)
        artefact = build_calibration_artefact(
            model,
            dataset,
            simd_scaled,
            coords_scaled,
            np.stack(fwd, axis=0),
            np.stack(bwd, axis=0),
            checkpoint_path=checkpoint_path,
            gamma=float(cfg["calibration"]["gamma"]),
            n_min=int(cfg["calibration"]["n_min"]),
            sigma_quantile=flag_cfg["quantile"],
            sigma_source_split=flag_cfg["source_split"],
            uncertainty_flag_enabled=flag_cfg["enabled"],
            require_calibration_available=flag_cfg["require_calibration_available"],
            output_path=output_dir / "calibration.json",
            device=device,
        )
        for item in artefact.get("warnings", []):
            warnings.append(ModelWarning(code=item["code"], level=item["level"], message=item["message"], details=item.get("details", {})))
        result["calibration_artefact"] = artefact
        exports = _write_complete_outputs(
            runtime={
                "payload": payload,
                "dataset": dataset,
                "graphs": graphs,
                "graph_set": graph_set,
                "coords": coords,
                "simd_scaler": simd_scaler,
                "coord_scaler": coord_scaler,
                "simd_scaled": simd_scaled,
                "coords_scaled": coords_scaled,
                "supports_fwd": np.stack(fwd, axis=0),
                "supports_bwd": np.stack(bwd, axis=0),
                "model": model,
                "warnings": warnings,
                "device": device,
            },
            artefact=artefact,
            checkpoint_path=checkpoint_path,
            calibration_path=output_dir / "calibration.json",
            output_dir=output_dir / "exports",
            cfg=cfg,
        )
        result["export_paths"] = exports
        if not result.get("persistence_gate_passed", True):
            warnings.append(
                ModelWarning(
                    code="persistence_gate_failed",
                    level=LEVEL_REVIEW,
                    message="No epoch had validation_selection MAE at or below persistence. "
                    "Saved the lowest-MAE epoch instead.",
                    details=result.get("selection_metrics") or {},
                )
            )
        return _envelope(
            warnings,
            result,
            {
                "graph_set": list(graph_set),
                "did_not_use_test_for_selection": True,
                "device": str(device),
            },
        )
    except ModelError as error:
        return _fail(error)


def run_rolling_evaluation(config_path_override: str | Path | None = None) -> dict[str, Any]:
    """Leakage-safe rolling-origin evaluation. Never overwrites the fixed S1 checkpoint."""
    try:
        cfg = load_model_config(config_path_override)
        assert_operational_inference_off(cfg)
        dataset, _tt = _dataset_from_cfg(cfg)
        graph_set = normalise_graph_set(cfg.get("graph_set", THREE_GRAPH_SET))
        graphs = load_graph_bundle(
            {
                "geo": config_path(cfg, "geo"),
                "transport": config_path(cfg, "transport"),
                "mobility": config_path(cfg, "mobility"),
            },
            canonical=dataset.node_order,
            graph_set=graph_set,
            reports={
                "geo": config_path(cfg, "geo_report"),
                "transport": config_path(cfg, "transport_report"),
                "mobility": config_path(cfg, "mobility_report"),
            },
        )
        coords = load_projected_centroids(config_path(cfg, "road_nodes"), dataset.node_order)
        fwd, bwd, _ = _supports_from_graphs(graphs, graph_set)
        result = _run_rolling_evaluation(
            cfg,
            dataset=dataset,
            graph_set=graph_set,
            graphs=graphs,
            coords=coords,
            fwd=fwd,
            bwd=bwd,
        )
        return _envelope(
            [],
            result,
            {
                "graph_set": list(graph_set),
                "canonical_node_order_hash": dataset.node_order.canonical_hash,
                "did_not_overwrite_fixed_s1": True,
                "fixed_s1_split_unchanged": True,
            },
        )
    except ModelError as error:
        return _fail(error)


def load_checkpoint(
    checkpoint_path: str | Path,
    calibration_path: str | Path,
    *,
    graph_set: list[str] | None = None,
    config_path_override: str | Path | None = None,
) -> dict[str, Any]:
    """Load a checkpoint and its matching calibration artefact. Graph set must match exactly."""
    try:
        cfg = load_model_config(config_path_override)
        checkpoint_path = Path(checkpoint_path)
        payload = load_raw_checkpoint(checkpoint_path)
        stored_set = tuple(payload["graph_set"])
        requested = normalise_graph_set(graph_set or stored_set)
        if requested != stored_set:
            raise ModelError(
                "Cannot drop or add graphs from a trained checkpoint. "
                "Load a separately trained two-graph checkpoint if mobility is unavailable.",
                code="graph_set_mismatch",
                details={"checkpoint_graph_set": list(stored_set), "requested": list(requested)},
            )
        if int(payload["alpha_dim"]) != len(stored_set):
            raise ModelError("Checkpoint alpha dimension does not match graph_set.", code="alpha_dim_mismatch")
        tt = temporal_target(cfg)
        run_cfg = payload.get("run_config") or {}
        horizon = int(run_cfg.get("target_offset_days", run_cfg.get("forecast_horizon_days", tt["target_offset_days"])))
        if horizon != tt["target_offset_days"]:
            raise ModelError(
                f"Checkpoint target_offset_days={horizon} does not match config {tt['target_offset_days']}.",
                code="config_mismatch",
            )
        output_steps = int(run_cfg.get("output_steps", 1))
        if output_steps != 1:
            raise ModelError("Checkpoint output_steps must be 1.", code="config_mismatch")
        artefact = json.loads(Path(calibration_path).read_text(encoding="utf-8"))
        assert_artefact_matches_checkpoint(artefact, checkpoint_path)
        return _envelope(
            [],
            {
                "checkpoint_path": str(checkpoint_path),
                "checkpoint_sha256": sha256_file(checkpoint_path),
                "graph_set": list(stored_set),
                "alpha_dim": payload["alpha_dim"],
                "calibration_status": artefact.get("calibration_status"),
            },
            {
                "canonical_node_order_hash": payload.get("node_order", {}).get("canonical_node_order_hash"),
                "selected_epoch": payload.get("selected_epoch"),
            },
        )
    except ModelError as error:
        return _fail(error)


def _restore_runtime(cfg: dict[str, Any], checkpoint_path: Path):
    payload = load_raw_checkpoint(checkpoint_path)
    dataset, _tt = _dataset_from_cfg(cfg)
    if payload.get("residual_scalers"):
        apply_residual_scalers(dataset, ResidualScalers.from_dict(payload["residual_scalers"]))
    else:
        raise ModelError(
            "This checkpoint has no residual scalers. The residual architecture cannot load "
            "an absolute-rate checkpoint.",
            code="checkpoint_incompatible",
        )
    graph_set = tuple(payload["graph_set"])
    graphs = load_graph_bundle(
        {
            "geo": config_path(cfg, "geo"),
            "transport": config_path(cfg, "transport"),
            "mobility": config_path(cfg, "mobility"),
        },
        canonical=dataset.node_order,
        graph_set=graph_set,
        reports={
            "geo": config_path(cfg, "geo_report"),
            "transport": config_path(cfg, "transport_report"),
            "mobility": config_path(cfg, "mobility_report"),
        },
    )
    coords = load_projected_centroids(config_path(cfg, "road_nodes"), dataset.node_order)
    simd_scaler = FrozenScaler.from_dict(payload["context_scaler"])
    coord_scaler = None if payload.get("coord_scaler") is None else FrozenScaler.from_dict(payload["coord_scaler"])
    simd_scaled = simd_scaler.transform(dataset.x_static_raw)
    coords_scaled = None if coord_scaler is None else coord_scaler.transform(coords)
    fwd, bwd, _ = _supports_from_graphs(graphs, graph_set)
    # Frozen inference must remain runnable on reviewer machines without CUDA.
    # Training keeps its explicit CUDA policy; only operational restoration
    # uses the separately configured inference device.
    inference_device = (cfg.get("operational_inference") or {}).get("device", "cpu")
    device = resolve_torch_device(name=inference_device, cfg=cfg)
    model = build_model_from_config(cfg, n_graphs=len(graph_set), has_location=coord_scaler is not None)
    model.load_state_dict(payload["model_state_dict"])
    model.to(device)
    model.eval()
    return {
        "payload": payload,
        "dataset": dataset,
        "graphs": graphs,
        "graph_set": graph_set,
        "coords": coords,
        "simd_scaler": simd_scaler,
        "coord_scaler": coord_scaler,
        "simd_scaled": simd_scaled,
        "coords_scaled": coords_scaled,
        "supports_fwd": np.stack(fwd, axis=0),
        "supports_bwd": np.stack(bwd, axis=0),
        "model": model,
        "warnings": [item for graph in graphs.values() for item in graph.warnings],
        "device": device,
    }


def _predict_residual(runtime: dict[str, Any], arrays) -> dict[str, np.ndarray]:
    dataset: TemporalDataset = runtime["dataset"]
    if dataset.residual_scalers is None:
        raise ModelError("Residual scalers are missing on the restored dataset.", code="invalid_scaler")
    return predict_split(
        runtime["model"],
        arrays,
        runtime["simd_scaled"],
        runtime["coords_scaled"],
        runtime["supports_fwd"],
        runtime["supports_bwd"],
        dataset.residual_scalers,
        device=runtime.get("device"),
    )


def _geoshapley_rate_from_delta(
    dataset: TemporalDataset,
    arrays,
    sample_index: int,
    node_index: int,
    mu_delta_z: np.ndarray,
) -> np.ndarray:
    """Explain Y_t + μ_Δ with Y_t held fixed, so φ_j match explaining μ_Δ."""
    reconstructed = reconstruct_rate_from_delta(
        mu_delta_z,
        np.ones_like(mu_delta_z),
        np.ones_like(mu_delta_z),
        delta_scaler=dataset.residual_scalers.delta,
        y_anchor=arrays.y_anchor_raw[sample_index : sample_index + 1],
    )
    return reconstructed["mu"][:, node_index, 0]


def _geoshapley_sample_indices(n_samples: int, mode: str) -> np.ndarray:
    if mode == "last":
        return np.asarray([n_samples - 1], dtype=int)
    if mode == "all":
        return np.arange(n_samples, dtype=int)
    raise ModelError(
        f"Unknown geoshapley.dates={mode}. Use 'all' or 'last'.",
        code="invalid_config",
    )


def _explain_all_iz_for_samples(
    runtime: dict[str, Any],
    arrays,
    sample_indices: np.ndarray,
    *,
    additivity_tolerance: float,
) -> pd.DataFrame:
    """Write GeoShapley for every IZ on the selected issue dates. Coalitions are batched."""
    device = runtime.get("device") or torch.device("cpu")
    dataset: TemporalDataset = runtime["dataset"]
    model: ForecastModel = runtime["model"]
    model.eval()
    s_fwd = torch.tensor(runtime["supports_fwd"], dtype=torch.float32, device=device)
    s_bwd = torch.tensor(runtime["supports_bwd"], dtype=torch.float32, device=device)
    if arrays.x_dynamic_model is None or arrays.y_anchor_raw is None:
        raise ModelError("Residual features are missing for GeoShapley.", code="invalid_tensor_shape")
    tables: list[pd.DataFrame] = []
    for sample_index in sample_indices:
        x_one = torch.tensor(
            arrays.x_dynamic_model[sample_index : sample_index + 1],
            dtype=torch.float32,
            device=device,
        )
        issue_date = _date_str(arrays.forecast_origin_date, int(sample_index))
        target_date = _date_str(arrays.target_date, int(sample_index))
        input_start = str(
            pd.Timestamp(arrays.forecast_origin_date[int(sample_index)])
            - pd.Timedelta(days=dataset.lookback_days - 1)
        )[:10]
        LOGGER.info("GeoShapley issue_date=%s for %s IZs", issue_date, dataset.n_nodes)
        for node_index, iz_code in enumerate(dataset.node_order.codes):
            specs, feature_batch, coord_batch = build_coalition_batch(
                dataset.x_static_raw,
                runtime["coords"],
                node_index,
            )
            simd_batch = runtime["simd_scaler"].transform(feature_batch)
            coord_scaled_batch = runtime["coord_scaler"].transform(coord_batch)
            x_covid = x_one.expand(feature_batch.shape[0], -1, -1, -1).contiguous()
            with torch.no_grad():
                outputs = model(
                    x_covid,
                    torch.tensor(simd_batch, dtype=torch.float32, device=device),
                    torch.tensor(coord_scaled_batch, dtype=torch.float32, device=device),
                    s_fwd,
                    s_bwd,
                )
            mu_z = outputs["mu"].cpu().numpy()
            values = _geoshapley_rate_from_delta(dataset, arrays, int(sample_index), node_index, mu_z)
            explanation = explanation_from_coalition_values(
                specs,
                values,
                n_features=dataset.x_static_raw.shape[1],
                additivity_tolerance=additivity_tolerance,
            )
            tables.append(
                build_geoshapley_table(
                    iz_code=iz_code,
                    node_index=node_index,
                    explanation=explanation,
                    node_order_hash=dataset.node_order.canonical_hash,
                    issue_date=issue_date,
                    target_report_date=target_date,
                    input_start_date=input_start,
                )
            )
    return pd.concat(tables, ignore_index=True)


def _write_complete_outputs(
    *,
    runtime: dict[str, Any],
    artefact: dict[str, Any],
    checkpoint_path: Path,
    calibration_path: Path,
    output_dir: Path,
    cfg: dict[str, Any],
) -> dict[str, str]:
    """Write webpage CSVs: all-IZ forecasts, optional observations, and all-IZ GeoShapley."""
    export_cfg = cfg.get("export", {})
    split = str(export_cfg.get("split", "test"))
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset: TemporalDataset = runtime["dataset"]
    arrays = dataset.splits[split]
    device = runtime.get("device") or torch.device("cpu")
    preds = _predict_residual(runtime, arrays)
    issue_dates = [_date_str(arrays.forecast_origin_date, i) for i in range(arrays.forecast_origin_date.shape[0])]
    target_dates = [_date_str(arrays.target_date, i) for i in range(arrays.target_date.shape[0])]
    input_starts = [
        str(pd.Timestamp(arrays.forecast_origin_date[i]) - pd.Timedelta(days=dataset.lookback_days - 1))[:10]
        for i in range(arrays.forecast_origin_date.shape[0])
    ]
    forecast_table = build_forecast_table(
        node_order=dataset.node_order,
        issue_dates=issue_dates,
        input_start_dates=input_starts,
        target_dates=target_dates,
        target_offset_days=int(temporal_target(cfg)["target_offset_days"]),
        mu_z=preds["mu_delta_z"],
        variance_z=preds["variance_delta_z"],
        sigma_z=preds["sigma_delta_z"],
        mu=preds["mu"],
        variance=preds["variance"],
        sigma=preds["sigma"],
        artefact=artefact,
        checkpoint_id=sha256_file(checkpoint_path),
        calibration_artefact_id=str(calibration_path),
        y_anchor=preds["y_anchor"],
        mu_delta=preds["mu_delta"],
    )
    if export_cfg.get("write_test_diagnostics", True) and split in {"test", "validation", "train"}:
        forecast_table = attach_observed_columns(
            forecast_table,
            node_order=dataset.node_order,
            issue_dates=issue_dates,
            y_raw=arrays.y_target_raw,
            artefact=artefact,
        )
    paths: dict[str, str] = {}
    if export_cfg.get("write_forecast", True):
        forecast_path = output_dir / "forecast_map.csv"
        forecast_table.to_csv(forecast_path, index=False)
        paths["forecast_map"] = str(forecast_path)
        LOGGER.info("Wrote %s (%s rows)", forecast_path, len(forecast_table))

    if export_cfg.get("write_geoshapley", True):
        geo_cfg = geoshapley_settings(cfg)
        mode = geo_cfg["dates"]
        sample_indices = _geoshapley_sample_indices(arrays.x_dynamic_scaled.shape[0], mode)
        geo_table = _explain_all_iz_for_samples(
            runtime,
            arrays,
            sample_indices,
            additivity_tolerance=geo_cfg["additivity_tolerance"],
        )
        geo_path = output_dir / "geoshapley.csv"
        geo_table.to_csv(geo_path, index=False)
        paths["geoshapley"] = str(geo_path)
        LOGGER.info("Wrote %s (%s rows)", geo_path, len(geo_table))

    if split == "test":
        metrics = evaluate_split(preds, arrays.y_target_raw, artefact)
        metrics_path = output_dir / "test_metrics.json"
        metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        paths["test_metrics"] = str(metrics_path)

    with torch.no_grad():
        embedding, _, _ = runtime["model"].embed(
            torch.tensor(runtime["simd_scaled"], dtype=torch.float32, device=device),
            None
            if runtime["coords_scaled"] is None
            else torch.tensor(runtime["coords_scaled"], dtype=torch.float32, device=device),
            torch.tensor(runtime["supports_fwd"], dtype=torch.float32, device=device),
            torch.tensor(runtime["supports_bwd"], dtype=torch.float32, device=device),
        )
    embed_table = build_embedding_table(
        embedding.cpu().numpy(),
        dataset.node_order,
        diagnose_embedding(embedding.cpu().numpy()),
    )
    embed_path = output_dir / "embedding.csv"
    embed_table.to_csv(embed_path, index=False)
    paths["embedding"] = str(embed_path)
    fusion_path = output_dir / "graph_fusion.json"
    fusion_path.write_text(
        json.dumps(
            {
                "graph_set": list(runtime["graph_set"]),
                "alpha": runtime["model"].alpha().detach().cpu().numpy().tolist(),
                "canonical_node_order_hash": dataset.node_order.canonical_hash,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    paths["graph_fusion"] = str(fusion_path)
    return paths


def forecast_single_target(
    checkpoint_path: str | Path,
    calibration_path: str | Path,
    *,
    split: str = "test",
    sample_index: int | None = None,
    config_path_override: str | Path | None = None,
) -> dict[str, Any]:
    """One Y_{t+7} distribution per IZ for each requested issue date."""
    try:
        cfg = load_model_config(config_path_override)
        artefact = json.loads(Path(calibration_path).read_text(encoding="utf-8"))
        assert_artefact_matches_checkpoint(artefact, Path(checkpoint_path))
        runtime = _restore_runtime(cfg, Path(checkpoint_path))
        dataset: TemporalDataset = runtime["dataset"]
        arrays = dataset.splits[split]
        if sample_index is not None:
            from data.dataset import subset_split

            arrays = subset_split(arrays, np.asarray([sample_index]))
        preds = _predict_residual(runtime, arrays)
        issue_dates = [_date_str(arrays.forecast_origin_date, i) for i in range(arrays.forecast_origin_date.shape[0])]
        target_dates = [_date_str(arrays.target_date, i) for i in range(arrays.target_date.shape[0])]
        input_starts = [
            str(pd.Timestamp(arrays.forecast_origin_date[i]) - pd.Timedelta(days=dataset.lookback_days - 1))[:10]
            for i in range(arrays.forecast_origin_date.shape[0])
        ]
        table = build_forecast_table(
            node_order=dataset.node_order,
            issue_dates=issue_dates,
            input_start_dates=input_starts,
            target_dates=target_dates,
            target_offset_days=int(temporal_target(cfg)["target_offset_days"]),
            mu_z=preds["mu_delta_z"],
            variance_z=preds["variance_delta_z"],
            sigma_z=preds["sigma_delta_z"],
            mu=preds["mu"],
            variance=preds["variance"],
            sigma=preds["sigma"],
            artefact=artefact,
            checkpoint_id=sha256_file(Path(checkpoint_path)),
            calibration_artefact_id=str(calibration_path),
            y_anchor=preds["y_anchor"],
            mu_delta=preds["mu_delta"],
        )
        return _envelope(
            runtime["warnings"],
            {"forecast_table": table, "n_rows": int(len(table))},
            {
                "split": split,
                "graph_set": list(runtime["graph_set"]),
                "canonical_node_order_hash": dataset.node_order.canonical_hash,
                "device": str(runtime.get("device")),
            },
        )
    except ModelError as error:
        return _fail(error)


def evaluate_test_period(
    checkpoint_path: str | Path,
    calibration_path: str | Path,
    *,
    config_path_override: str | Path | None = None,
) -> dict[str, Any]:
    """Test-set metrics only. Test is not used for checkpoint, calibration, or thresholds."""
    try:
        cfg = load_model_config(config_path_override)
        artefact = json.loads(Path(calibration_path).read_text(encoding="utf-8"))
        assert_artefact_matches_checkpoint(artefact, Path(checkpoint_path))
        runtime = _restore_runtime(cfg, Path(checkpoint_path))
        test = runtime["dataset"].splits["test"]
        preds = _predict_residual(runtime, test)
        metrics = evaluate_split(preds, test.y_target_raw, artefact)
        return _envelope(
            runtime["warnings"],
            {"metrics": metrics},
            {
                "split": "test",
                "not_used_for_selection_or_calibration": True,
                "canonical_node_order_hash": runtime["dataset"].node_order.canonical_hash,
                "device": str(runtime.get("device")),
            },
        )
    except ModelError as error:
        return _fail(error)


def explain_target_iz_with_geoshapley(
    checkpoint_path: str | Path,
    calibration_path: str | Path,
    iz_code: str,
    *,
    split: str = "test",
    sample_index: int | None = None,
    config_path_override: str | Path | None = None,
) -> dict[str, Any]:
    """GeoShapley for original SIMD + location on one IZ.

    Default sample is the last retrospective issue in the split, not an
    operational next-report-day forecast.
    """
    try:
        cfg = load_model_config(config_path_override)
        artefact = json.loads(Path(calibration_path).read_text(encoding="utf-8"))
        assert_artefact_matches_checkpoint(artefact, Path(checkpoint_path))
        runtime = _restore_runtime(cfg, Path(checkpoint_path))
        dataset: TemporalDataset = runtime["dataset"]
        if iz_code not in dataset.node_order.codes:
            raise ModelError(
                f"Target IZ {iz_code} is missing from the canonical node order.",
                code="missing_target_iz",
            )
        target_index = dataset.node_order.codes.index(iz_code)
        arrays = dataset.splits[split]
        if sample_index is None:
            sample_index = int(arrays.x_dynamic_model.shape[0] - 1)
        device = runtime.get("device") or torch.device("cpu")
        x_covid = torch.tensor(
            arrays.x_dynamic_model[sample_index : sample_index + 1],
            dtype=torch.float32,
            device=device,
        )
        model: ForecastModel = runtime["model"]

        def predict_mu(features_raw: np.ndarray, coords_raw: np.ndarray | None) -> float:
            simd_scaled = runtime["simd_scaler"].transform(features_raw)
            coords_scaled = None if coords_raw is None else runtime["coord_scaler"].transform(coords_raw)
            n_coalitions = simd_scaled.shape[0] if simd_scaled.ndim == 3 else 1
            x_batch = x_covid.expand(n_coalitions, -1, -1, -1).contiguous()
            with torch.no_grad():
                outputs = model(
                    x_batch,
                    torch.tensor(simd_scaled, dtype=torch.float32, device=device),
                    None if coords_scaled is None else torch.tensor(coords_scaled, dtype=torch.float32, device=device),
                    torch.tensor(runtime["supports_fwd"], dtype=torch.float32, device=device),
                    torch.tensor(runtime["supports_bwd"], dtype=torch.float32, device=device),
                )
            mu_z = outputs["mu"].cpu().numpy()
            rates = _geoshapley_rate_from_delta(dataset, arrays, sample_index, target_index, mu_z)
            return float(rates[0])

        geo_cfg = geoshapley_settings(cfg)
        explanation = explain_target_iz(
            dataset.x_static_raw,
            runtime["coords"],
            target_index,
            predict_mu,
            additivity_tolerance=geo_cfg["additivity_tolerance"],
        )
        warnings = list(runtime["warnings"])
        if explanation.get("warning"):
            item = explanation["warning"]
            warnings.append(ModelWarning(code=item["code"], level=item["level"], message=item["message"], details=item.get("details", {})))
        table = build_geoshapley_table(
            iz_code=iz_code,
            node_index=target_index,
            explanation=explanation,
            node_order_hash=dataset.node_order.canonical_hash,
        )
        return _envelope(
            warnings,
            {"explanation": explanation, "table": table},
            {
                "explanation_scope": explanation["explanation_scope"],
                "iz_code": iz_code,
                "sample_index": sample_index,
                "split": split,
                "geoshapley_dates_meaning": "final retrospective test issue date when sample_index is last",
                "n_indicator_players": geo_cfg["n_indicator_players"],
                "device": str(device),
            },
        )
    except ModelError as error:
        return _fail(error)


def export_operational_forecast(
    checkpoint_path: str | Path | None = None,
    calibration_path: str | Path | None = None,
    output_dir: str | Path | None = None,
    *,
    issue_date: str | None = None,
    config_path_override: str | Path | None = None,
    run_geoshapley: bool = True,
) -> dict[str, Any]:
    """U10 operational forecast for the unlabelled t+7 target after the last panel date.

    Writes a validation report first. Stops before GeoShapley if U10 is incompatible.
    Does not retrain, does not use the fixed 65/10/25 checkpoint, and does not
    compute test metrics for an unobserved target.
    """
    try:
        result = _run_operational_forecast(
            restore_runtime=_restore_runtime,
            predict_residual=_predict_residual,
            explain_all_iz=_explain_all_iz_for_samples,
            checkpoint_path=Path(checkpoint_path or OPERATIONAL_DEFAULT_CHECKPOINT),
            calibration_path=Path(calibration_path or OPERATIONAL_DEFAULT_CALIBRATION),
            output_dir=Path(output_dir or OPERATIONAL_DEFAULT_OUTPUT_DIR),
            config_path_override=config_path_override,
            issue_date=issue_date,
            run_geoshapley=run_geoshapley,
        )
        return _envelope(
            result["warnings"],
            {
                "paths": result["paths"],
                "n_iz": result["n_iz"],
                "summary": result["summary"],
                "validation_passed": result["validation"]["passed"],
            },
            {
                **result["provenance"],
                "device": result["device"],
                "retrospective_test_unchanged": True,
                "validation_report": result["paths"]["validation_report"],
            },
        )
    except ModelError as error:
        return _fail(error)


def export_map_ready_results(
    checkpoint_path: str | Path,
    calibration_path: str | Path,
    output_dir: str | Path,
    *,
    split: str | None = None,
    config_path_override: str | Path | None = None,
) -> dict[str, Any]:
    """Write complete webpage CSVs: forecasts for every IZ and GeoShapley for every IZ."""
    try:
        cfg = load_model_config(config_path_override)
        assert_operational_inference_off(cfg)
        if split is not None:
            cfg.setdefault("export", {})
            cfg["export"]["split"] = split
        artefact = json.loads(Path(calibration_path).read_text(encoding="utf-8"))
        assert_artefact_matches_checkpoint(artefact, Path(checkpoint_path))
        runtime = _restore_runtime(cfg, Path(checkpoint_path))
        paths = _write_complete_outputs(
            runtime=runtime,
            artefact=artefact,
            checkpoint_path=Path(checkpoint_path),
            calibration_path=Path(calibration_path),
            output_dir=Path(output_dir),
            cfg=cfg,
        )
        geo_cfg = geoshapley_settings(cfg)
        return _envelope(
            runtime["warnings"],
            {"paths": paths},
            {
                "split": cfg.get("export", {}).get("split", "test"),
                "canonical_node_order_hash": runtime["dataset"].node_order.canonical_hash,
                "geoshapley_dates": geo_cfg["dates"],
                "geoshapley_dates_meaning": "final retrospective test issue date when dates=last",
                "operational_inference_enabled": False,
                "n_indicator_players": geo_cfg["n_indicator_players"],
                "device": str(runtime.get("device")),
            },
        )
    except ModelError as error:
        return _fail(error)

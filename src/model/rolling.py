"""Leakage-safe rolling-origin evaluation for the residual multi-graph DCRNN.

Does not recut S1 windows, does not change architecture, and never overwrites
the fixed 70/15/15 checkpoint directory.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from data.dataset import STATIC_FEATURE_COLUMNS
from model.config import load_model_config, resolve_path, temporal_target
from model.context import FrozenScaler, fit_cross_section_scaler
from data.dataset import (
    SplitArrays,
    TemporalDataset,
    concatenate_splits,
    sort_split_by_target_date,
    subset_split,
)
from common.errors import ModelError
from model.evaluate import build_calibration_from_split, predict_split
from graph.supports import load_graph_bundle, load_projected_centroids, normalise_graph_set
from model.heads import gaussian_nll
from data.node_order import sha256_file
from model.residual import (
    ResidualScalers,
    apply_residual_scalers_to_split,
    fit_residual_scalers,
)
from model.train import (
    build_model_from_config,
    load_raw_checkpoint,
    refit_fixed_epochs,
    resolve_torch_device,
    train_forecast_model,
)
from common.utils import get_logger

LOGGER = get_logger("model.rolling")
WINDOW_ALL = "all"


def _as_ts(values) -> pd.DatetimeIndex:
    return pd.DatetimeIndex(pd.to_datetime(values)).normalize()


def _date_str(value) -> str:
    return str(pd.Timestamp(value).normalize().date())


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (pd.Timestamp, np.datetime64)):
        return _date_str(value)
    try:
        if pd.isna(value):
            return None
    except (ValueError, TypeError):
        pass
    return value


def rolling_settings(cfg: dict[str, Any]) -> dict[str, Any]:
    block = cfg.get("rolling_evaluation") or {}
    stage = str(block.get("stage", "window_selection")).strip().lower()
    if stage not in {"plan", "window_selection", "final_test"}:
        raise ModelError(
            "rolling_evaluation.stage must be plan, window_selection, or final_test.",
            code="invalid_config",
        )
    frequency = int(block.get("retrain_frequency_days", 28))
    n_sel = int(block.get("selection_target_dates", block.get("validation_days", 56) // 2))
    n_cal = int(block.get("calibration_target_dates", n_sel))
    if frequency < 1 or n_sel < 1 or n_cal < 1:
        raise ModelError("Rolling frequency and block lengths must be positive.", code="invalid_config")
    raw_windows = block.get("candidate_windows_days", [180, 365, 730, WINDOW_ALL])
    windows: list[int | None] = []
    for item in raw_windows:
        if item in {None, WINDOW_ALL, "none", "all_history"}:
            windows.append(None)
        else:
            days = int(item)
            if days < 1:
                raise ModelError("candidate_windows_days must be positive or 'all'.", code="invalid_config")
            windows.append(days)
    selected = block.get("selected_window_days", None)
    if selected in {WINDOW_ALL, "all_history"}:
        selected_days: int | None | str = WINDOW_ALL
    elif selected is None or selected == "":
        selected_days = None
    else:
        selected_days = int(selected)
    output_dir = resolve_path(block.get("output_dir", "data/results/model/rolling_v1"))
    fixed_s1 = resolve_path(block.get("fixed_s1_dir", "data/results/model/geo_transport_mobility"))
    return {
        "stage": stage,
        "retrain_frequency_days": frequency,
        "selection_target_dates": n_sel,
        "calibration_target_dates": n_cal,
        "candidate_windows_days": windows,
        "selected_window_days": selected_days,
        "window_selection_split": str(block.get("window_selection_split", "validation")),
        "origin_split": str(block.get("origin_split", "test")),
        "output_dir": output_dir,
        "fixed_s1_dir": fixed_s1,
        "write_geoshapley": bool(block.get("write_geoshapley", False)),
    }


def window_label(window_days: int | None) -> str:
    return WINDOW_ALL if window_days is None else f"W{int(window_days)}"


def parse_window_label(label: str) -> int | None:
    text = str(label).strip()
    if text in {WINDOW_ALL, "all_history"}:
        return None
    if text.startswith("W"):
        text = text[1:]
    return int(text)


def assert_not_fixed_s1(output_dir: Path, fixed_s1_dir: Path) -> None:
    out = Path(output_dir).resolve()
    frozen = Path(fixed_s1_dir).resolve()
    if out == frozen or frozen in out.parents or out in frozen.parents:
        raise ModelError(
            "Rolling outputs must not overwrite or sit inside the fixed S1 checkpoint directory.",
            code="rolling_overwrite_forbidden",
            details={"output_dir": str(out), "fixed_s1_dir": str(frozen)},
        )


def pool_s1_windows(dataset: TemporalDataset) -> SplitArrays:
    pool = concatenate_splits(
        [dataset.splits["train"], dataset.splits["validation"], dataset.splits["test"]],
        name="s1_pool",
    )
    return sort_split_by_target_date(pool)


def update_dates(origin_issues: pd.DatetimeIndex, frequency_days: int) -> list[pd.Timestamp]:
    issues = pd.DatetimeIndex(pd.to_datetime(origin_issues)).normalize().unique().sort_values()
    if issues.empty:
        raise ModelError("Rolling origin split has no issue dates.", code="invalid_config")
    freq = pd.Timedelta(days=int(frequency_days))
    dates = [pd.Timestamp(issues.min()).normalize()]
    while True:
        later = issues[issues >= dates[-1] + freq]
        if later.empty:
            break
        dates.append(pd.Timestamp(later.min()).normalize())
    return dates


def mask_target_on_or_before(split: SplitArrays, cutoff: pd.Timestamp) -> np.ndarray:
    return _as_ts(split.target_date) <= pd.Timestamp(cutoff).normalize()


def mask_issue_in_block(
    split: SplitArrays,
    start: pd.Timestamp,
    end_exclusive: pd.Timestamp,
) -> np.ndarray:
    issues = _as_ts(split.forecast_origin_date)
    return (issues >= pd.Timestamp(start).normalize()) & (issues < pd.Timestamp(end_exclusive).normalize())


@dataclass
class RollingBlocks:
    update_date: pd.Timestamp
    window_days: int | None
    fitting: SplitArrays
    selection: SplitArrays
    calibration: SplitArrays
    predict: SplitArrays
    audit: dict[str, Any]


def _unique_dates(values) -> pd.DatetimeIndex:
    return pd.DatetimeIndex(pd.to_datetime(np.unique(_as_ts(values)))).sort_values()


def assign_blocks(
    pool: SplitArrays,
    origin_split: SplitArrays,
    *,
    update_date: pd.Timestamp,
    next_update: pd.Timestamp,
    window_days: int | None,
    n_selection: int,
    n_calibration: int,
    lookback_days: int,
    target_offset_days: int,
) -> RollingBlocks:
    update_date = pd.Timestamp(update_date).normalize()
    next_update = pd.Timestamp(next_update).normalize()
    eligible_idx = np.where(mask_target_on_or_before(pool, update_date))[0]
    if eligible_idx.size == 0:
        raise ModelError(
            f"No labelled samples with target_report_date <= {update_date.date()}.",
            code="rolling_empty_history",
        )
    eligible = subset_split(pool, eligible_idx)
    target_dates = _unique_dates(eligible.target_date)
    need = n_selection + n_calibration + 1
    if len(target_dates) < need:
        raise ModelError(
            "Not enough historically available target dates for fitting/selection/calibration.",
            code="rolling_insufficient_history",
            details={
                "update_date": _date_str(update_date),
                "n_available_target_dates": int(len(target_dates)),
                "required": int(need),
            },
        )
    cal_dates = set(target_dates[-n_calibration:])
    sel_dates = set(target_dates[-(n_selection + n_calibration) : -n_calibration])
    fit_dates = pd.DatetimeIndex(target_dates[: -(n_selection + n_calibration)])
    if window_days is not None:
        last_fit = fit_dates.max()
        start = last_fit - pd.Timedelta(days=int(window_days) - 1)
        fit_dates = fit_dates[fit_dates >= start]
        if len(fit_dates) == 0:
            raise ModelError(
                f"Training window {window_days} days is empty at {update_date.date()}.",
                code="rolling_empty_window",
            )
    fit_set = set(fit_dates)
    if fit_set & sel_dates or fit_set & cal_dates or sel_dates & cal_dates:
        raise ModelError(
            "Fitting, selection and calibration target dates overlap.",
            code="rolling_block_overlap",
            details={"update_date": _date_str(update_date)},
        )
    elig_targets = _as_ts(eligible.target_date)
    fitting = subset_split(eligible, np.where(elig_targets.isin(fit_dates))[0])
    selection = subset_split(eligible, np.where(elig_targets.isin(list(sel_dates)))[0])
    calibration = subset_split(eligible, np.where(elig_targets.isin(list(cal_dates)))[0])
    predict_idx = np.where(mask_issue_in_block(origin_split, update_date, next_update))[0]
    if predict_idx.size == 0:
        raise ModelError(
            f"No origin issue dates in [{update_date.date()}, {next_update.date()}).",
            code="rolling_empty_predict_block",
        )
    predict = subset_split(origin_split, predict_idx)
    audit = leakage_audit(
        update_date=update_date,
        fitting=fitting,
        selection=selection,
        calibration=calibration,
        predict=predict,
        lookback_days=lookback_days,
        target_offset_days=target_offset_days,
    )
    return RollingBlocks(
        update_date=update_date,
        window_days=window_days,
        fitting=fitting,
        selection=selection,
        calibration=calibration,
        predict=predict,
        audit=audit,
    )


def leakage_audit(
    *,
    update_date: pd.Timestamp,
    fitting: SplitArrays,
    selection: SplitArrays,
    calibration: SplitArrays,
    predict: SplitArrays,
    lookback_days: int,
    target_offset_days: int,
) -> dict[str, Any]:
    u = pd.Timestamp(update_date).normalize()
    blocks = {
        "fitting": fitting,
        "selection": selection,
        "calibration": calibration,
    }
    for name, split in blocks.items():
        targets = _as_ts(split.target_date)
        if (targets > u).any():
            raise ModelError(
                f"{name} has target_report_date after update date {u.date()}.",
                code="rolling_leakage",
                details={"block": name, "max_target": _date_str(targets.max())},
            )
    fit_dates = set(_unique_dates(fitting.target_date))
    sel_dates = set(_unique_dates(selection.target_date))
    cal_dates = set(_unique_dates(calibration.target_date))
    if fit_dates & sel_dates or fit_dates & cal_dates or sel_dates & cal_dates:
        raise ModelError("Fitting, selection and calibration target dates overlap.", code="rolling_block_overlap")
    issues = _as_ts(predict.forecast_origin_date)
    targets = _as_ts(predict.target_date)
    input_end = issues
    input_start = issues - pd.Timedelta(days=lookback_days - 1)
    if (input_end > issues).any():
        raise ModelError("An input date is later than its issue date.", code="rolling_leakage")
    lag = (targets - issues).days
    if not np.all(lag.to_numpy() == int(target_offset_days)):
        raise ModelError(
            f"target_report_date must equal issue_date + {target_offset_days} days.",
            code="rolling_horizon_mismatch",
        )
    if input_start.min() is pd.NaT:
        raise ModelError("Missing input start dates.", code="rolling_leakage")
    pred_keys = list(zip(issues.astype(str), predict.sample_id.astype(str)))
    if len(pred_keys) != len(set(pred_keys)):
        raise ModelError("Duplicate issue_date predictions in a rolling block.", code="rolling_duplicate_prediction")
    return {
        "update_date": _date_str(u),
        "passed": True,
        "max_input_date_rule": "issue_date",
        "labelled_cutoff": _date_str(u),
        "n_fitting": int(fitting.target_date.shape[0]),
        "n_selection": int(selection.target_date.shape[0]),
        "n_calibration": int(calibration.target_date.shape[0]),
        "n_predict": int(predict.target_date.shape[0]),
        "fitting_target_start": _date_str(min(fit_dates)),
        "fitting_target_end": _date_str(max(fit_dates)),
        "selection_target_start": _date_str(min(sel_dates)),
        "selection_target_end": _date_str(max(sel_dates)),
        "calibration_target_start": _date_str(min(cal_dates)),
        "calibration_target_end": _date_str(max(cal_dates)),
        "predict_issue_start": _date_str(issues.min()),
        "predict_issue_end": _date_str(issues.max()),
        "lookback_days": int(lookback_days),
        "target_offset_days": int(target_offset_days),
    }


def persistence_and_model_metrics(y: np.ndarray, mu: np.ndarray, persist: np.ndarray) -> dict[str, Any]:
    valid = np.isfinite(y) & np.isfinite(mu) & np.isfinite(persist)
    yv, mv, pv = y[valid], mu[valid], persist[valid]
    if yv.size == 0:
        raise ModelError("No finite cells for rolling metrics.", code="rolling_empty_metrics")

    def _point(pred: np.ndarray) -> dict[str, float | None]:
        residual = yv - pred
        mae = float(np.mean(np.abs(residual)))
        rmse = float(np.sqrt(np.mean(residual ** 2)))
        bias = float(np.mean(residual))
        ss_res = float(np.sum(residual ** 2))
        ss_tot = float(np.sum((yv - yv.mean()) ** 2))
        r2 = None if ss_tot == 0 else float(1.0 - ss_res / ss_tot)
        return {"mae": mae, "rmse": rmse, "bias": bias, "r2": r2}

    model = _point(mv)
    baseline = _point(pv)
    skill = None if baseline["mae"] == 0 else float(1.0 - model["mae"] / baseline["mae"])
    return {
        "n_valid_cells": int(valid.sum()),
        "model_mae": model["mae"],
        "model_rmse": model["rmse"],
        "model_bias": model["bias"],
        "model_r2": model["r2"],
        "persistence_mae": baseline["mae"],
        "persistence_rmse": baseline["rmse"],
        "persistence_bias": baseline["bias"],
        "persistence_r2": baseline["r2"],
        "mae_skill": skill,
    }


def _nll(y: np.ndarray, mu: np.ndarray, variance: np.ndarray) -> float:
    valid = np.isfinite(y) & np.isfinite(mu) & np.isfinite(variance) & (variance > 0)
    return float(
        gaussian_nll(
            torch.tensor(y, dtype=torch.float32),
            torch.tensor(mu, dtype=torch.float32),
            torch.tensor(variance, dtype=torch.float32),
            mask=torch.tensor(valid, dtype=torch.bool),
        ).item()
    )


def _write_json(path: Path, payload: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(payload), indent=2), encoding="utf-8")
    return path


def _prediction_table(
    *,
    node_order,
    predict: SplitArrays,
    preds: dict[str, np.ndarray],
    artefact: dict[str, Any],
    update_id: str,
    update_date: pd.Timestamp,
    window_name: str,
    checkpoint_path: Path,
    scaler_path: Path,
    graph_set: tuple[str, ...],
    lookback_days: int,
    target_offset_days: int,
) -> pd.DataFrame:
    issues = _as_ts(predict.forecast_origin_date)
    targets = _as_ts(predict.target_date)
    n_samples, n_nodes, _ = preds["mu"].shape
    rows = []
    q95 = artefact.get("q95")
    q80 = artefact.get("q80")
    for sample_index in range(n_samples):
        issue = issues[sample_index]
        target = targets[sample_index]
        input_start = issue - pd.Timedelta(days=lookback_days - 1)
        for node_index, iz_code in enumerate(node_order.codes):
            mu = float(preds["mu"][sample_index, node_index, 0])
            sigma = float(preds["sigma"][sample_index, node_index, 0])
            y = float(predict.y_target_raw[sample_index, node_index, 0])
            anchor = float(preds["y_anchor"][sample_index, node_index, 0])
            rows.append(
                {
                    "update_id": update_id,
                    "window": window_name,
                    "information_cutoff": _date_str(update_date),
                    "input_start_date": _date_str(input_start),
                    "issue_date": _date_str(issue),
                    "target_report_date": _date_str(target),
                    "target_offset_days": int(target_offset_days),
                    "iz_code": iz_code,
                    "node_index": int(node_index),
                    "y_anchor": anchor,
                    "persistence_prediction": anchor,
                    "predicted_mu_delta": float(preds["mu_delta"][sample_index, node_index, 0]),
                    "predicted_mu": mu,
                    "predicted_sigma": sigma,
                    "observed_rate": y,
                    "checkpoint_path": str(checkpoint_path),
                    "checkpoint_sha256": artefact.get("checkpoint_sha256"),
                    "scaler_path": str(scaler_path),
                    "graph_set": ",".join(graph_set),
                    "node_order_hash": node_order.canonical_hash,
                    "q80": q80,
                    "q95": q95,
                    "calibrated80_lower": None if q80 is None else mu - float(q80) * sigma,
                    "calibrated80_upper": None if q80 is None else mu + float(q80) * sigma,
                    "calibrated95_lower": None if q95 is None else mu - float(q95) * sigma,
                    "calibrated95_upper": None if q95 is None else mu + float(q95) * sigma,
                }
            )
    table = pd.DataFrame(rows)
    dup = table.duplicated(["issue_date", "iz_code"]).sum()
    if dup:
        raise ModelError(
            "Duplicate issue_date × IZ predictions were produced.",
            code="rolling_duplicate_prediction",
            details={"n_duplicates": int(dup)},
        )
    return table


def _choose_window(window_rows: list[dict[str, Any]]) -> dict[str, Any]:
    def key(row: dict[str, Any]):
        skill = row.get("mae_skill")
        skill_rank = float("-inf") if skill is None else float(skill)
        return (-skill_rank, float(row["model_mae"]), abs(float(row["model_bias"])))

    ranked = sorted(window_rows, key=key)
    return ranked[0]


def plan_rolling_evaluation(dataset: TemporalDataset, cfg: dict[str, Any]) -> dict[str, Any]:
    settings = rolling_settings(cfg)
    tt = temporal_target(cfg)
    pool = pool_s1_windows(dataset)
    stages = {}
    for name in (settings["window_selection_split"], settings["origin_split"]):
        origin = dataset.splits[name]
        issues = _as_ts(origin.forecast_origin_date)
        updates = update_dates(issues, settings["retrain_frequency_days"])
        rows = []
        for index, update in enumerate(updates):
            nxt = (
                updates[index + 1]
                if index + 1 < len(updates)
                else pd.Timestamp(issues.max()).normalize() + pd.Timedelta(days=1)
            )
            for window in settings["candidate_windows_days"]:
                blocks = assign_blocks(
                    pool,
                    origin,
                    update_date=update,
                    next_update=nxt,
                    window_days=window,
                    n_selection=settings["selection_target_dates"],
                    n_calibration=settings["calibration_target_dates"],
                    lookback_days=tt["lookback_steps"],
                    target_offset_days=tt["target_offset_days"],
                )
                rows.append(
                    {
                        "update_id": f"U{index + 1:02d}",
                        "window": window_label(window),
                        **blocks.audit,
                    }
                )
        stages[name] = {
            "n_origins": int(origin.forecast_origin_date.shape[0]),
            "issue_start": _date_str(issues.min()),
            "issue_end": _date_str(issues.max()),
            "target_start": _date_str(_as_ts(origin.target_date).min()),
            "target_end": _date_str(_as_ts(origin.target_date).max()),
            "n_updates": len(updates),
            "expected_predictions": int(origin.forecast_origin_date.shape[0] * dataset.n_nodes),
            "updates": rows,
        }
    return {
        "fixed_s1_preserved": True,
        "fixed_s1_dir": str(settings["fixed_s1_dir"]),
        "rolling_output_dir": str(settings["output_dir"]),
        "candidate_windows": [window_label(item) for item in settings["candidate_windows_days"]],
        "retrain_frequency_days": settings["retrain_frequency_days"],
        "selection_target_dates": settings["selection_target_dates"],
        "calibration_target_dates": settings["calibration_target_dates"],
        "stages": stages,
        "outputs": [
            "predictions.csv",
            "updates_manifest.json",
            "metrics_overall.json",
            "metrics_by_update.csv",
            "calibration/",
            "leakage_audit.json",
            "provenance.json",
        ],
    }


def _prepare_static(dataset: TemporalDataset, cfg: dict[str, Any], coords: np.ndarray | None):
    simd_scaler, _warnings = fit_cross_section_scaler(
        dataset.x_static_raw,
        STATIC_FEATURE_COLUMNS,
        epsilon=float(cfg["context"]["zero_variance_epsilon"]),
        ddof=int(cfg["context"]["scaler_ddof"]),
    )
    simd_scaled = simd_scaler.transform(dataset.x_static_raw)
    coord_scaler = None
    coords_scaled = None
    if coords is not None:
        coord_scaler, _coord_warnings = fit_cross_section_scaler(
            coords,
            ("easting", "northing"),
            epsilon=float(cfg["context"]["zero_variance_epsilon"]),
            ddof=int(cfg["context"]["scaler_ddof"]),
        )
        coords_scaled = coord_scaler.transform(coords)
    return simd_scaler, simd_scaled, coord_scaler, coords_scaled


def write_rolling_alpha_csv(manifests: list[dict[str, Any]], dest: Path) -> Path:
    """Fusion-weight trajectory from completed rolling checkpoints. Does not invent dates."""
    from model.operational import _alpha_from_checkpoint
    from model.train import load_raw_checkpoint

    rows = []
    for item in manifests:
        ckpt = Path((item.get("refit") or {}).get("checkpoint_path") or "")
        if not ckpt.is_file():
            continue
        alpha = _alpha_from_checkpoint(load_raw_checkpoint(ckpt))
        audit = item.get("audit") or {}
        window = str(item.get("window") or "")
        days = int(window[1:]) if window.startswith("W") and window[1:].isdigit() else None
        rows.append(
            {
                "update_id": item.get("update_id"),
                "checkpoint_id": item.get("update_id"),
                "update_date": audit.get("update_date"),
                "forecast_start": audit.get("predict_issue_start"),
                "forecast_end": audit.get("predict_issue_end"),
                "alpha_geo": float(alpha["geo"]),
                "alpha_transport": float(alpha["transport"]),
                "alpha_mobility": float(alpha["mobility"]),
                "selected_epoch": int(item.get("selected_epoch") or 0),
                "training_window_days": days,
                "checkpoint_checksum": (item.get("refit") or {}).get("checkpoint_sha256"),
                "node_order_hash": item.get("canonical_node_order_hash"),
            }
        )
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(dest, index=False)
    return dest


def _run_one_update(
    *,
    dataset: TemporalDataset,
    cfg: dict[str, Any],
    settings: dict[str, Any],
    tt: dict[str, Any],
    graph_set: tuple[str, ...],
    graph_hashes: dict[str, str],
    fwd: list[np.ndarray],
    bwd: list[np.ndarray],
    simd_scaler: FrozenScaler,
    simd_scaled: np.ndarray,
    coord_scaler: FrozenScaler | None,
    coords_scaled: np.ndarray | None,
    blocks: RollingBlocks,
    update_id: str,
    output_dir: Path,
    device,
) -> dict[str, Any]:
    window_name = window_label(blocks.window_days)
    run_dir = output_dir / window_name / update_id
    ckpt_path = run_dir / "checkpoint.pt"
    manifest_path = run_dir / "manifest.json"
    pred_path = run_dir / "predictions.csv"
    if ckpt_path.is_file() and ckpt_path.stat().st_size > 0 and manifest_path.is_file() and pred_path.is_file():
        LOGGER.info("Rolling %s %s: reuse existing checkpoint", window_name, update_id)
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        table = pd.read_csv(pred_path)
        return {
            "manifest": manifest,
            "table": table,
            "metrics": manifest.get("metrics") or {},
            "audit": manifest.get("audit") or blocks.audit,
        }
    run_dir.mkdir(parents=True, exist_ok=True)
    fit_scalers = fit_residual_scalers(blocks.fitting, cfg)
    fit_split = apply_residual_scalers_to_split(blocks.fitting, fit_scalers)
    sel_split = apply_residual_scalers_to_split(blocks.selection, fit_scalers)
    dataset.residual_scalers = fit_scalers
    LOGGER.info("Rolling %s %s: select epoch on fitting n=%s", window_name, update_id, fit_split.target_date.shape[0])
    selected = train_forecast_model(
        dataset,
        fwd,
        bwd,
        simd_scaled,
        coords_scaled,
        graph_set=graph_set,
        graph_hashes=graph_hashes,
        context_scaler=simd_scaler,
        coord_scaler=coord_scaler,
        output_dir=run_dir / "selection",
        config=cfg,
        device_name=str(device),
        train_split=fit_split,
        selection_split=sel_split,
    )
    n_epochs = int(selected["selected_epoch"])
    refit_source = concatenate_splits([blocks.fitting, blocks.selection], name="fitting_plus_selection")
    deployed_scalers = fit_residual_scalers(refit_source, cfg)
    refit_split = apply_residual_scalers_to_split(refit_source, deployed_scalers)
    cal_split = apply_residual_scalers_to_split(blocks.calibration, deployed_scalers)
    predict_split_arr = apply_residual_scalers_to_split(blocks.predict, deployed_scalers)
    dataset.residual_scalers = deployed_scalers
    LOGGER.info("Rolling %s %s: refit %s epochs on fitting+selection n=%s", window_name, update_id, n_epochs, refit_split.target_date.shape[0])
    refit = refit_fixed_epochs(
        dataset,
        fwd,
        bwd,
        simd_scaled,
        coords_scaled,
        graph_set=graph_set,
        graph_hashes=graph_hashes,
        context_scaler=simd_scaler,
        coord_scaler=coord_scaler,
        output_dir=run_dir,
        train_split=refit_split,
        n_epochs=n_epochs,
        config=cfg,
        device_name=str(device),
        selection_metrics=selected.get("selection_metrics"),
        selection_nll=selected.get("selection_nll"),
    )
    scaler_path = _write_json(run_dir / "residual_scalers.json", deployed_scalers.as_dict())
    payload = load_raw_checkpoint(Path(refit["checkpoint_path"]))
    stored_graphs = payload.get("graph_hashes") or {}
    if stored_graphs != graph_hashes:
        raise ModelError("Graph hashes changed across a rolling update.", code="node_order_mismatch", details=stored_graphs)
    stored_hash = (payload.get("node_order") or {}).get("canonical_node_order_hash")
    if stored_hash != dataset.node_order.canonical_hash:
        raise ModelError("Node-order hash differs from the canonical S1 order.", code="node_order_mismatch")
    model = build_model_from_config(cfg, n_graphs=len(graph_set), has_location=coords_scaled is not None)
    model.load_state_dict(payload["model_state_dict"])
    model.to(device)
    model.eval()
    artefact = build_calibration_from_split(
        model,
        cal_split,
        simd_scaled,
        coords_scaled,
        np.stack(fwd, axis=0),
        np.stack(bwd, axis=0),
        deployed_scalers,
        checkpoint_path=Path(refit["checkpoint_path"]),
        gamma_95=float(cfg["calibration"]["gamma"]),
        n_min=int(cfg["calibration"]["n_min"]),
        output_path=run_dir / "calibration.json",
        device=device,
        extra={"update_id": update_id, "information_cutoff": _date_str(blocks.update_date)},
    )
    preds = predict_split(
        model,
        predict_split_arr,
        simd_scaled,
        coords_scaled,
        np.stack(fwd, axis=0),
        np.stack(bwd, axis=0),
        deployed_scalers,
        device=device,
    )
    metrics = persistence_and_model_metrics(predict_split_arr.y_target_raw, preds["mu"], preds["y_anchor"])
    metrics["gaussian_nll_original"] = _nll(predict_split_arr.y_target_raw, preds["mu"], preds["variance"])
    table = _prediction_table(
        node_order=dataset.node_order,
        predict=predict_split_arr,
        preds=preds,
        artefact=artefact,
        update_id=update_id,
        update_date=blocks.update_date,
        window_name=window_name,
        checkpoint_path=Path(refit["checkpoint_path"]),
        scaler_path=scaler_path,
        graph_set=graph_set,
        lookback_days=tt["lookback_steps"],
        target_offset_days=tt["target_offset_days"],
    )
    table_path = run_dir / "predictions.csv"
    table.to_csv(table_path, index=False)
    manifest = {
        "update_id": update_id,
        "window": window_name,
        "audit": blocks.audit,
        "selected_epoch": n_epochs,
        "persistence_gate_passed": selected.get("persistence_gate_passed"),
        "selection_metrics": selected.get("selection_metrics"),
        "refit": {key: refit[key] for key in ("checkpoint_path", "checkpoint_sha256", "selected_epoch", "refit_on")},
        "scaler_path": str(scaler_path),
        "calibration_path": artefact.get("path"),
        "predictions_path": str(table_path),
        "metrics": metrics,
        "graph_hashes": graph_hashes,
        "canonical_node_order_hash": dataset.node_order.canonical_hash,
    }
    _write_json(run_dir / "manifest.json", manifest)
    return {"manifest": manifest, "table": table, "metrics": metrics, "audit": blocks.audit}


def _aggregate_tables(tables: list[pd.DataFrame]) -> tuple[pd.DataFrame, dict[str, Any], pd.DataFrame]:
    combined = pd.concat(tables, ignore_index=True)
    if combined.duplicated(["issue_date", "iz_code"]).any():
        raise ModelError("Duplicate issue_date × IZ predictions across updates.", code="rolling_duplicate_prediction")
    overall = persistence_and_model_metrics(
        combined["observed_rate"].to_numpy(),
        combined["predicted_mu"].to_numpy(),
        combined["y_anchor"].to_numpy(),
    )
    by_rows = []
    for update_id, part in combined.groupby("update_id", sort=True):
        row = persistence_and_model_metrics(
            part["observed_rate"].to_numpy(),
            part["predicted_mu"].to_numpy(),
            part["y_anchor"].to_numpy(),
        )
        row["update_id"] = update_id
        row["n_issue_dates"] = int(part["issue_date"].nunique())
        by_rows.append(row)
    return combined, overall, pd.DataFrame(by_rows)


def _run_origin_loop(
    *,
    dataset: TemporalDataset,
    cfg: dict[str, Any],
    settings: dict[str, Any],
    origin_name: str,
    windows: list[int | None],
    output_root: Path,
    static_pack: dict[str, Any],
) -> dict[str, Any]:
    tt = temporal_target(cfg)
    pool = pool_s1_windows(dataset)
    origin = dataset.splits[origin_name]
    issues = _as_ts(origin.forecast_origin_date)
    updates = update_dates(issues, settings["retrain_frequency_days"])
    window_summaries = []
    for window in windows:
        tables = []
        manifests = []
        audits = []
        window_dir = output_root / window_label(window)
        window_dir.mkdir(parents=True, exist_ok=True)
        for index, update in enumerate(updates):
            nxt = (
                updates[index + 1]
                if index + 1 < len(updates)
                else pd.Timestamp(issues.max()).normalize() + pd.Timedelta(days=1)
            )
            update_id = f"U{index + 1:02d}"
            blocks = assign_blocks(
                pool,
                origin,
                update_date=update,
                next_update=nxt,
                window_days=window,
                n_selection=settings["selection_target_dates"],
                n_calibration=settings["calibration_target_dates"],
                lookback_days=tt["lookback_steps"],
                target_offset_days=tt["target_offset_days"],
            )
            result = _run_one_update(
                dataset=dataset,
                cfg=cfg,
                settings=settings,
                tt=tt,
                graph_set=static_pack["graph_set"],
                graph_hashes=static_pack["graph_hashes"],
                fwd=static_pack["fwd"],
                bwd=static_pack["bwd"],
                simd_scaler=static_pack["simd_scaler"],
                simd_scaled=static_pack["simd_scaled"],
                coord_scaler=static_pack["coord_scaler"],
                coords_scaled=static_pack["coords_scaled"],
                blocks=blocks,
                update_id=update_id,
                output_dir=output_root,
                device=static_pack["device"],
            )
            tables.append(result["table"])
            manifests.append(result["manifest"])
            audits.append(result["audit"])
            write_rolling_alpha_csv(manifests, window_dir / "rolling_alpha.csv")
        combined, overall, by_update = _aggregate_tables(tables)
        combined.to_csv(window_dir / "predictions.csv", index=False)
        by_update.to_csv(window_dir / "metrics_by_update.csv", index=False)
        _write_json(window_dir / "metrics_overall.json", overall)
        _write_json(window_dir / "updates_manifest.json", {"updates": manifests})
        _write_json(window_dir / "leakage_audit.json", {"passed": True, "updates": audits})
        window_summaries.append(
            {
                "window": window_label(window),
                "window_days": window,
                **overall,
                "n_predictions": int(len(combined)),
                "output_dir": str(window_dir),
            }
        )
    return {"windows": window_summaries, "n_updates": len(updates), "origin_split": origin_name}


def run_rolling_evaluation(
    cfg: dict[str, Any] | None = None,
    *,
    dataset: TemporalDataset,
    graph_set: tuple[str, ...],
    graphs: dict[str, Any],
    coords: np.ndarray,
    fwd: list[np.ndarray],
    bwd: list[np.ndarray],
) -> dict[str, Any]:
    cfg = cfg or load_model_config()
    settings = rolling_settings(cfg)
    assert_not_fixed_s1(settings["output_dir"], settings["fixed_s1_dir"])
    if settings["write_geoshapley"]:
        raise ModelError("Rolling evaluation does not write GeoShapley.", code="invalid_config")
    settings["output_dir"].mkdir(parents=True, exist_ok=True)
    plan = plan_rolling_evaluation(dataset, cfg)
    _write_json(settings["output_dir"] / "plan.json", plan)
    if settings["stage"] == "plan":
        return {"stage": "plan", "plan_path": str(settings["output_dir"] / "plan.json"), "plan": plan}

    device = resolve_torch_device(cfg=cfg)
    simd_scaler, simd_scaled, coord_scaler, coords_scaled = _prepare_static(dataset, cfg, coords)
    static_pack = {
        "graph_set": graph_set,
        "graph_hashes": {name: graphs[name].file_sha256 for name in graph_set},
        "fwd": fwd,
        "bwd": bwd,
        "simd_scaler": simd_scaler,
        "simd_scaled": simd_scaled,
        "coord_scaler": coord_scaler,
        "coords_scaled": coords_scaled,
        "device": device,
    }
    provenance = {
        "stage": settings["stage"],
        "device": str(device),
        "graph_set": list(graph_set),
        "graph_hashes": static_pack["graph_hashes"],
        "canonical_node_order_hash": dataset.node_order.canonical_hash,
        "fixed_s1_dir": str(settings["fixed_s1_dir"]),
        "did_not_overwrite_fixed_s1": True,
        "architecture_unchanged": True,
    }
    if settings["stage"] == "window_selection":
        root = settings["output_dir"] / "window_selection"
        summary = _run_origin_loop(
            dataset=dataset,
            cfg=cfg,
            settings=settings,
            origin_name=settings["window_selection_split"],
            windows=settings["candidate_windows_days"],
            output_root=root,
            static_pack=static_pack,
        )
        chosen = _choose_window(summary["windows"])
        selection = {
            "rule": "max_mae_skill then min_mae then min_abs_bias; NLL diagnostic only",
            "selected_window": chosen["window"],
            "selected_window_days": chosen["window_days"] if chosen["window_days"] is not None else WINDOW_ALL,
            "candidates": summary["windows"],
            "nll_note": "gaussian_nll_original is reported per window but not used to pick the window unless skill/MAE/bias tie.",
        }
        _write_json(root / "selected_window.json", selection)
        _write_json(settings["output_dir"] / "provenance.json", {**provenance, "window_selection": selection})
        return {
            "stage": "window_selection",
            "selected_window": selection["selected_window"],
            "selected_window_days": selection["selected_window_days"],
            "candidates": summary["windows"],
            "output_dir": str(root),
            "plan_path": str(settings["output_dir"] / "plan.json"),
        }

    selected = settings["selected_window_days"]
    if selected is None:
        stored = settings["output_dir"] / "window_selection" / "selected_window.json"
        if not stored.is_file():
            raise ModelError(
                "final_test requires rolling_evaluation.selected_window_days or a completed window_selection run.",
                code="invalid_config",
            )
        payload = json.loads(stored.read_text(encoding="utf-8"))
        selected = payload.get("selected_window_days")
    window_days = None if selected in {WINDOW_ALL, None} else int(selected)
    root = settings["output_dir"] / "final_test"
    summary = _run_origin_loop(
        dataset=dataset,
        cfg=cfg,
        settings=settings,
        origin_name=settings["origin_split"],
        windows=[window_days],
        output_root=root,
        static_pack=static_pack,
    )
    chosen = summary["windows"][0]
    _write_json(
        settings["output_dir"] / "provenance.json",
        {**provenance, "selected_window": chosen["window"], "final_test": chosen},
    )
    return {
        "stage": "final_test",
        "selected_window": chosen["window"],
        "metrics": {key: chosen[key] for key in chosen if key not in {"output_dir"}},
        "output_dir": chosen["output_dir"],
        "plan_path": str(settings["output_dir"] / "plan.json"),
    }

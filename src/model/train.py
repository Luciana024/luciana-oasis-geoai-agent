"""Training loop and parameter checkpoint.

See docs/model.md section 15. The checkpoint stores parameters, graph_set,
scalers, and validation_selection NLL. It does not store q95 or the P90
sigma threshold. Those belong in a separate calibration artefact.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from model.config import load_model_config
from model.context import FrozenScaler
from data.dataset import SplitArrays, TemporalDataset, subset_split
from common.errors import ModelError
from model.heads import combined_forecast_loss
from model.network import ForecastModel
from data.node_order import sha256_file
from model.residual import ResidualScalers, persistence_mae, reconstruct_rate_from_delta
from common.utils import get_logger

LOGGER = get_logger("model.train")


def _gpu_with_most_free_memory() -> int:
    best_index, best_free = 0, -1
    for index in range(torch.cuda.device_count()):
        free, _total = torch.cuda.mem_get_info(index)
        if free > best_free:
            best_free = free
            best_index = index
    return best_index


def resolve_torch_device(name: str | None = None, cfg: dict[str, Any] | None = None) -> torch.device:
    """Use CUDA when requested. auto/cuda pick the GPU with the most free memory."""
    requested = name
    if requested is None and cfg is not None:
        requested = (cfg.get("training") or {}).get("device", "cuda")
    if requested is None:
        requested = "cuda"
    requested = str(requested).strip().lower()
    if requested == "cpu":
        return torch.device("cpu")
    if not torch.cuda.is_available():
        raise ModelError(
            "CUDA was requested but this PyTorch build cannot see a GPU. "
            "Install a CUDA wheel that supports RTX 5090 (sm_120), not torch+cpu.",
            code="cuda_unavailable",
            details={
                "torch_version": torch.__version__,
                "cuda_compiled": str(torch.version.cuda),
            },
        )
    if requested in {"cuda", "gpu", "auto"}:
        return torch.device(f"cuda:{_gpu_with_most_free_memory()}")
    if requested.startswith("cuda"):
        device = torch.device(requested)
        if device.index is not None and device.index >= torch.cuda.device_count():
            raise ModelError(
                f"Requested {device} but only {torch.cuda.device_count()} CUDA device(s) are visible.",
                code="cuda_unavailable",
            )
        return device
    raise ModelError(f"Unknown training.device={requested}", code="invalid_config")


def _stack_supports(matrices: list[np.ndarray]) -> torch.Tensor:
    return torch.tensor(np.stack(matrices, axis=0), dtype=torch.float32)


def build_model_from_config(cfg: dict[str, Any], n_graphs: int, has_location: bool = True) -> ForecastModel:
    variant = str((cfg.get("model_variant") or "full")).strip().lower()
    return ForecastModel(
        n_graphs=n_graphs,
        n_features=len(cfg["context"]["feature_columns"]),
        embedding_dim=int(cfg["embedding_dim"]),
        hidden_dim=int(cfg["hidden_dim"]),
        context_layers=int(cfg["context_layers"]),
        dcrnn_layers=int(cfg["dcrnn_layers"]),
        diffusion_steps=int(cfg["diffusion_steps"]),
        dropout=float(cfg["dropout"]),
        variance_epsilon=float(cfg["variance_epsilon"]),
        has_location=has_location,
        use_context=variant != "dynamic_only",
        dynamic_input_dim=len((cfg.get("dynamic_channels") or ["rate", "first_difference"])),
    )


def _loader(
    x: np.ndarray,
    y: np.ndarray,
    batch_size: int,
    shuffle: bool,
    *,
    pin_memory: bool = False,
) -> DataLoader:
    dataset = TensorDataset(
        torch.tensor(x, dtype=torch.float32),
        torch.tensor(y, dtype=torch.float32),
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, pin_memory=pin_memory)


def _run_epoch(
    model: ForecastModel,
    loader: DataLoader,
    simd: torch.Tensor,
    coords: torch.Tensor | None,
    supports_fwd: torch.Tensor,
    supports_bwd: torch.Tensor,
    optimizer: torch.optim.Optimizer | None,
    grad_clip: float,
    device: torch.device,
    *,
    mean_loss_weight: float,
    huber_delta: float,
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    totals = {"loss": 0.0, "nll": 0.0, "huber": 0.0}
    n_batches = 0
    for x_covid, y_delta_z in loader:
        x_covid = x_covid.to(device)
        y_delta_z = y_delta_z.to(device)
        mask = torch.isfinite(y_delta_z)
        if training:
            optimizer.zero_grad(set_to_none=True)
        outputs = model(x_covid, simd, coords, supports_fwd, supports_bwd)
        loss, nll, huber = combined_forecast_loss(
            y_delta_z,
            outputs["mu"],
            outputs["variance"],
            mask=mask,
            mean_loss_weight=mean_loss_weight,
            huber_delta=huber_delta,
        )
        if training:
            loss.backward()
            if grad_clip is not None and grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
        totals["loss"] += float(loss.detach().cpu())
        totals["nll"] += float(nll.detach().cpu())
        totals["huber"] += float(huber.detach().cpu())
        n_batches += 1
    denom = max(n_batches, 1)
    return {key: value / denom for key, value in totals.items()}


def save_checkpoint(
    path: Path,
    *,
    model: ForecastModel,
    graph_set: tuple[str, ...],
    graph_hashes: dict[str, str],
    node_order_payload: dict[str, Any],
    covid_scaler: dict[str, Any],
    context_scaler: dict[str, Any],
    coord_scaler: dict[str, Any] | None,
    residual_scalers: dict[str, Any] | None,
    config: dict[str, Any],
    selected_epoch: int,
    selection_nll: float,
    seed: int,
    selection_metrics: dict[str, Any] | None = None,
    optimizer_state: dict[str, Any] | None = None,
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_state_dict": model.state_dict(),
        "graph_set": list(graph_set),
        "graph_hashes": graph_hashes,
        "alpha_dim": int(model.n_graphs),
        "node_order": node_order_payload,
        "covid_scaler": covid_scaler,
        "context_scaler": context_scaler,
        "coord_scaler": coord_scaler,
        "residual_scalers": residual_scalers,
        "model_config": model.config_dict(),
        "run_config": {
            "config_id": config.get("config_id"),
            "lookback_steps": (config.get("temporal_target") or {}).get("lookback_steps"),
            "target_offset_days": (config.get("temporal_target") or {}).get("target_offset_days"),
            "output_steps": (config.get("temporal_target") or {}).get("output_steps", 1),
            "window_stride_days": (config.get("temporal_target") or {}).get("window_stride_days"),
            "target_definition": (config.get("temporal_target") or {}).get("target_definition"),
            "model_variant": config.get("model_variant", "full"),
            "predicts": "delta_from_latest_report",
        },
        "selected_epoch": int(selected_epoch),
        "selection_nll": float(selection_nll),
        "selection_metrics": selection_metrics or {},
        "seed": int(seed),
        "optimizer_state": optimizer_state,
        # Calibration quantities must not be stored here.
    }
    torch.save(payload, path)
    sidecar = {
        "path": str(path),
        "sha256": sha256_file(path),
        "graph_set": list(graph_set),
        "alpha_dim": int(model.n_graphs),
        "selected_epoch": int(selected_epoch),
        "selection_nll": float(selection_nll),
        "selection_metrics": selection_metrics or {},
        "canonical_node_order_hash": node_order_payload.get("canonical_node_order_hash"),
    }
    path.with_suffix(".json").write_text(json.dumps(sidecar, indent=2), encoding="utf-8")
    return path


def load_raw_checkpoint(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ModelError(f"Checkpoint does not exist: {path}", code="missing_checkpoint")
    return torch.load(path, map_location="cpu", weights_only=False)


def _selection_rate_metrics(
    model: ForecastModel,
    split,
    simd: torch.Tensor,
    coords: torch.Tensor | None,
    supports_fwd: torch.Tensor,
    supports_bwd: torch.Tensor,
    residual_scalers: ResidualScalers,
    device: torch.device,
    batch_size: int,
) -> dict[str, float]:
    model.eval()
    mus = []
    with torch.no_grad():
        x_all = split.x_dynamic_model
        for start in range(0, x_all.shape[0], batch_size):
            x = torch.tensor(x_all[start : start + batch_size], dtype=torch.float32, device=device)
            outputs = model(x, simd, coords, supports_fwd, supports_bwd)
            mus.append(outputs["mu"].cpu().numpy())
    mu_z = np.concatenate(mus, axis=0)
    reconstructed = reconstruct_rate_from_delta(
        mu_z,
        np.ones_like(mu_z),
        np.ones_like(mu_z),
        delta_scaler=residual_scalers.delta,
        y_anchor=split.y_anchor_raw,
    )
    y = split.y_target_raw
    mu = reconstructed["mu"]
    valid = np.isfinite(y) & np.isfinite(mu)
    mae = float(np.mean(np.abs(y[valid] - mu[valid])))
    persist = persistence_mae(split)
    ss_res = float(np.sum((y[valid] - mu[valid]) ** 2))
    ss_tot = float(np.sum((y[valid] - y[valid].mean()) ** 2))
    r2 = None if ss_tot == 0 else float(1.0 - ss_res / ss_tot)
    skill = None if persist == 0 else float(1.0 - mae / persist)
    return {
        "mae": mae,
        "persistence_mae": persist,
        "mae_skill": skill,
        "r2": r2,
        "beats_persistence": bool(mae <= persist),
    }


def train_forecast_model(
    dataset: TemporalDataset,
    supports_fwd: list[np.ndarray],
    supports_bwd: list[np.ndarray],
    simd_scaled: np.ndarray,
    coords_scaled: np.ndarray | None,
    *,
    graph_set: tuple[str, ...],
    graph_hashes: dict[str, str],
    context_scaler: FrozenScaler,
    coord_scaler: FrozenScaler | None,
    output_dir: Path,
    config: dict[str, Any] | None = None,
    device_name: str | None = None,
    train_split: SplitArrays | None = None,
    selection_split: SplitArrays | None = None,
) -> dict[str, Any]:
    """Fit on train deltas. Checkpoint must not be worse than persistence MAE on selection."""
    cfg = config or load_model_config()
    if dataset.residual_scalers is None:
        raise ModelError("Residual scalers are missing. Call prepare_residual_dataset first.", code="invalid_scaler")
    train_cfg = cfg["training"]
    loss_cfg = cfg.get("loss") or {}
    selection_cfg = cfg.get("selection") or {}
    device = resolve_torch_device(device_name, cfg)
    seed = int(train_cfg.get("seed", 42))
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)

    model = build_model_from_config(cfg, n_graphs=len(graph_set), has_location=coords_scaled is not None)
    model.to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(train_cfg["learning_rate"]),
        weight_decay=float(train_cfg["weight_decay"]),
    )

    simd = torch.tensor(simd_scaled, dtype=torch.float32, device=device)
    coords = None if coords_scaled is None else torch.tensor(coords_scaled, dtype=torch.float32, device=device)
    s_fwd = _stack_supports(supports_fwd).to(device)
    s_bwd = _stack_supports(supports_bwd).to(device)

    train_split = train_split or dataset.splits["train"]
    if selection_split is None:
        selection_split = subset_split(dataset.splits["validation"], dataset.validation_selection_index)
    if train_split.x_dynamic_model is None or selection_split.x_dynamic_model is None:
        raise ModelError("Residual features are missing on the training splits.", code="invalid_tensor_shape")
    pin_memory = device.type == "cuda"
    batch_size = int(train_cfg["batch_size"])
    mean_loss_weight = float(loss_cfg.get("mean_loss_weight", 0.5))
    huber_delta = float(loss_cfg.get("huber_delta", 1.0))
    require_persist = bool(selection_cfg.get("require_mae_not_worse_than_persistence", True))
    train_loader = _loader(
        train_split.x_dynamic_model,
        train_split.y_delta_scaled,
        batch_size,
        shuffle=True,
        pin_memory=pin_memory,
    )
    selection_loader = _loader(
        selection_split.x_dynamic_model,
        selection_split.y_delta_scaled,
        batch_size,
        shuffle=False,
        pin_memory=pin_memory,
    )

    best_nll = float("inf")
    best_epoch = -1
    best_metrics: dict[str, Any] = {}
    fallback_epoch = -1
    fallback_nll = float("inf")
    fallback_metrics: dict[str, Any] = {}
    patience = int(train_cfg["patience"])
    wait = 0
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / "checkpoint.pt"
    history: list[dict[str, Any]] = []

    def _write(epoch: int, nll: float, metrics: dict[str, Any]) -> None:
        save_checkpoint(
            checkpoint_path,
            model=model,
            graph_set=graph_set,
            graph_hashes=graph_hashes,
            node_order_payload=dataset.node_order.as_dict(),
            covid_scaler=dataset.covid_scaler.as_dict(),
            context_scaler=context_scaler.as_dict(),
            coord_scaler=None if coord_scaler is None else coord_scaler.as_dict(),
            residual_scalers=dataset.residual_scalers.as_dict(),
            config=cfg,
            selected_epoch=epoch,
            selection_nll=nll,
            seed=seed,
            selection_metrics=metrics,
            optimizer_state=optimizer.state_dict(),
        )

    for epoch in range(1, int(train_cfg["max_epochs"]) + 1):
        train_stats = _run_epoch(
            model,
            train_loader,
            simd,
            coords,
            s_fwd,
            s_bwd,
            optimizer,
            float(train_cfg["grad_clip"]),
            device,
            mean_loss_weight=mean_loss_weight,
            huber_delta=huber_delta,
        )
        selection_stats = _run_epoch(
            model,
            selection_loader,
            simd,
            coords,
            s_fwd,
            s_bwd,
            None,
            0.0,
            device,
            mean_loss_weight=mean_loss_weight,
            huber_delta=huber_delta,
        )
        rate_metrics = _selection_rate_metrics(
            model,
            selection_split,
            simd,
            coords,
            s_fwd,
            s_bwd,
            dataset.residual_scalers,
            device,
            batch_size,
        )
        selection_nll = float(selection_stats["nll"])
        eligible = (not require_persist) or bool(rate_metrics["beats_persistence"])
        row = {
            "epoch": epoch,
            "train_loss": float(train_stats["loss"]),
            "train_nll": float(train_stats["nll"]),
            "selection_nll": selection_nll,
            "selection_mae": float(rate_metrics["mae"]),
            "persistence_mae": float(rate_metrics["persistence_mae"]),
            "mae_skill": None if rate_metrics["mae_skill"] is None else float(rate_metrics["mae_skill"]),
            "selection_r2": None if rate_metrics["r2"] is None else float(rate_metrics["r2"]),
            "beats_persistence": bool(rate_metrics["beats_persistence"]),
            "eligible": eligible,
            "improved_gated_nll": bool(eligible and selection_nll < best_nll),
        }
        history.append(row)
        LOGGER.info(
            "epoch %s train_nll=%.4f selection_nll=%.4f mae=%.2f persist=%.2f skill=%s r2=%s gate=%s%s",
            epoch,
            row["train_nll"],
            row["selection_nll"],
            row["selection_mae"],
            row["persistence_mae"],
            "nan" if row["mae_skill"] is None else f"{row['mae_skill']:.3f}",
            "nan" if row["selection_r2"] is None else f"{row['selection_r2']:.3f}",
            "pass" if row["beats_persistence"] else "fail",
            " selected" if row["improved_gated_nll"] else "",
        )
        improved_fallback = rate_metrics["mae"] < fallback_metrics.get("mae", float("inf"))
        if improved_fallback:
            fallback_epoch = epoch
            fallback_nll = selection_nll
            fallback_metrics = dict(rate_metrics)
        if eligible and selection_nll < best_nll:
            best_nll = selection_nll
            best_epoch = epoch
            best_metrics = dict(rate_metrics)
            wait = 0
            _write(best_epoch, best_nll, best_metrics)
        else:
            wait += 1
            if wait >= patience:
                break

    gated = best_epoch >= 0
    if not gated:
        if fallback_epoch < 0:
            raise ModelError("Training produced no checkpoint.", code="training_failed")
        best_epoch = fallback_epoch
        best_nll = fallback_nll
        best_metrics = dict(fallback_metrics)
        best_metrics["persistence_gate"] = "failed"
        _write(best_epoch, best_nll, best_metrics)

    history_path = output_dir / "training_history.json"
    history_path.write_text(
        json.dumps(
            {
                "selected_epoch": int(best_epoch),
                "n_epochs_run": len(history),
                "patience": patience,
                "persistence_gate_passed": gated,
                "selection_rule": "lowest selection NLL among epochs with MAE not worse than persistence",
                "history": history,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    LOGGER.info(
        "Wrote %s (%s epochs, selected epoch %s)",
        history_path,
        len(history),
        best_epoch,
    )

    return {
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "selected_epoch": best_epoch,
        "selection_nll": best_nll,
        "selection_metrics": best_metrics,
        "persistence_gate_passed": gated,
        "history": history,
        "graph_set": list(graph_set),
        "validation_internal_split": dataset.internal_split_provenance,
        "device": str(device),
        "model_variant": cfg.get("model_variant", "full"),
    }


def refit_fixed_epochs(
    dataset: TemporalDataset,
    supports_fwd: list[np.ndarray],
    supports_bwd: list[np.ndarray],
    simd_scaled: np.ndarray,
    coords_scaled: np.ndarray | None,
    *,
    graph_set: tuple[str, ...],
    graph_hashes: dict[str, str],
    context_scaler: FrozenScaler,
    coord_scaler: FrozenScaler | None,
    output_dir: Path,
    train_split: SplitArrays,
    n_epochs: int,
    config: dict[str, Any] | None = None,
    device_name: str | None = None,
    selection_metrics: dict[str, Any] | None = None,
    selection_nll: float | None = None,
) -> dict[str, Any]:
    """Reinitialise and train exactly n_epochs on fitting+selection. No calibration rows."""
    cfg = config or load_model_config()
    if dataset.residual_scalers is None:
        raise ModelError("Residual scalers are missing.", code="invalid_scaler")
    if train_split.x_dynamic_model is None:
        raise ModelError("Residual features are missing on the refit split.", code="invalid_tensor_shape")
    if n_epochs < 1:
        raise ModelError("Refit n_epochs must be >= 1.", code="training_failed")
    train_cfg = cfg["training"]
    loss_cfg = cfg.get("loss") or {}
    device = resolve_torch_device(device_name, cfg)
    seed = int(train_cfg.get("seed", 42))
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = build_model_from_config(cfg, n_graphs=len(graph_set), has_location=coords_scaled is not None)
    model.to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(train_cfg["learning_rate"]),
        weight_decay=float(train_cfg["weight_decay"]),
    )
    simd = torch.tensor(simd_scaled, dtype=torch.float32, device=device)
    coords = None if coords_scaled is None else torch.tensor(coords_scaled, dtype=torch.float32, device=device)
    s_fwd = _stack_supports(supports_fwd).to(device)
    s_bwd = _stack_supports(supports_bwd).to(device)
    loader = _loader(
        train_split.x_dynamic_model,
        train_split.y_delta_scaled,
        int(train_cfg["batch_size"]),
        shuffle=True,
        pin_memory=device.type == "cuda",
    )
    history: list[dict[str, Any]] = []
    for epoch in range(1, int(n_epochs) + 1):
        stats = _run_epoch(
            model,
            loader,
            simd,
            coords,
            s_fwd,
            s_bwd,
            optimizer,
            float(train_cfg["grad_clip"]),
            device,
            mean_loss_weight=float(loss_cfg.get("mean_loss_weight", 0.5)),
            huber_delta=float(loss_cfg.get("huber_delta", 1.0)),
        )
        history.append({"epoch": epoch, **stats})
    output_dir = Path(output_dir)
    checkpoint_path = output_dir / "checkpoint.pt"
    save_checkpoint(
        checkpoint_path,
        model=model,
        graph_set=graph_set,
        graph_hashes=graph_hashes,
        node_order_payload=dataset.node_order.as_dict(),
        covid_scaler=dataset.covid_scaler.as_dict(),
        context_scaler=context_scaler.as_dict(),
        coord_scaler=None if coord_scaler is None else coord_scaler.as_dict(),
        residual_scalers=dataset.residual_scalers.as_dict(),
        config=cfg,
        selected_epoch=int(n_epochs),
        selection_nll=float(selection_nll if selection_nll is not None else history[-1]["nll"]),
        seed=seed,
        selection_metrics=selection_metrics or {},
        optimizer_state=optimizer.state_dict(),
    )
    return {
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "selected_epoch": int(n_epochs),
        "history": history,
        "device": str(device),
        "refit_on": "fitting_plus_selection",
    }


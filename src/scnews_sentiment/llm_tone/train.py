from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.scnews_sentiment.llm_tone.data import (
    apply_time_series_split,
    build_time_series_cv_splits,
)
from src.scnews_sentiment.llm_tone.eval import metric_dict
from src.scnews_sentiment.llm_tone.models import build_tone_regressor, encode_texts
from src.scnews_sentiment.llm_tone.utils import (
    clean_generated_text,
    ensure_dir,
    parse_hidden_sizes,
    require_columns,
    resolve_device,
    save_csv,
    set_seed,
)


def prepare_forecasts(forecasts: pd.DataFrame) -> pd.DataFrame:
    require_columns(
        forecasts,
        ["sample_id", "split", "target_article_score", "predicted_event_text"],
        "forecast_events.csv",
    )
    result = forecasts.copy()
    result["predicted_event_text"] = (
        result["predicted_event_text"].fillna("").astype(str).map(clean_generated_text)
    )
    if (result["predicted_event_text"].str.len() == 0).any():
        raise ValueError("forecast_events.csv contains empty predicted_event_text values.")
    return result


def save_embedding_cache(
    forecasts: pd.DataFrame,
    embeddings: np.ndarray,
    output_dir: Path,
) -> None:
    ensure_dir(output_dir)
    np.savez(
        output_dir / "forecast_event_embeddings.npz",
        sample_id=forecasts["sample_id"].astype(str).to_numpy(),
        embedding=embeddings,
    )


def fit_mlp_from_embeddings(
    forecasts: pd.DataFrame,
    embeddings: np.ndarray,
    args: argparse.Namespace,
) -> Tuple[pd.DataFrame, Dict[str, Any], Dict[str, Any]]:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset

    y = forecasts["target_article_score"].astype(float).to_numpy()
    split = forecasts["split"].astype(str).to_numpy()
    train_mask = split == "train"
    val_mask = split == "val"
    if train_mask.sum() == 0:
        raise ValueError("No training samples after split assignment.")

    y_mean = float(y[train_mask].mean())
    y_std = float(y[train_mask].std())
    if y_std == 0:
        y_std = 1.0
    y_scaled = (y - y_mean) / y_std

    device = resolve_device(args.train_device)
    x_tensor = torch.tensor(embeddings, dtype=torch.float32)
    y_tensor = torch.tensor(y_scaled.reshape(-1, 1), dtype=torch.float32)

    train_ds = TensorDataset(x_tensor[train_mask], y_tensor[train_mask])
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)

    input_dim = embeddings.shape[1]
    model = build_tone_regressor(input_dim, args.hidden_sizes, args.dropout).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    loss_fn = nn.MSELoss()

    x_tensor = x_tensor.to(device)
    y_tensor = y_tensor.to(device)
    best_state: Optional[Dict[str, Any]] = None
    best_val = float("inf")
    best_epoch = 0
    bad_epochs = 0
    epochs_run = 0

    def eval_loss(mask: np.ndarray) -> float:
        if mask.sum() == 0:
            mask = train_mask
        model.eval()
        with torch.no_grad():
            pred = model(x_tensor[mask])
            return float(loss_fn(pred, y_tensor[mask]).item())

    for epoch in range(1, args.epochs + 1):
        epochs_run = epoch
        model.train()
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            optimizer.step()

        val_loss = eval_loss(val_mask)
        if val_loss < best_val - args.min_delta:
            best_val = val_loss
            best_epoch = epoch
            best_state = copy.deepcopy(
                {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            )
            bad_epochs = 0
        else:
            bad_epochs += 1
        if bad_epochs >= args.patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    with torch.no_grad():
        pred_scaled = model(x_tensor).detach().cpu().numpy().reshape(-1)
    pred = pred_scaled * y_std + y_mean

    result = forecasts.copy()
    result["predicted_article_score"] = pred
    result["target_article_score"] = y
    result["abs_error"] = np.abs(result["predicted_article_score"] - result["target_article_score"])
    result["squared_error"] = (
        result["predicted_article_score"] - result["target_article_score"]
    ) ** 2

    metrics: Dict[str, Any] = {
        "target_mean_train": y_mean,
        "target_std_train": y_std,
        "best_val_loss_scaled": best_val,
        "best_epoch": best_epoch,
        "epochs_run": epochs_run,
        "encoder_model": args.encoder_model,
        "encoder_device": args.encoder_device,
        "train_device": args.train_device,
        "resolved_train_device": device,
        "pooling": args.pooling,
        "win": args.win,
        "split_strategy": args.split_strategy,
        "metrics": {},
    }
    for split_name in ["train", "val", "test", "all"]:
        mask = (
            np.isin(split, ["train", "val", "test"])
            if split_name == "all"
            else split == split_name
        )
        metrics["metrics"][split_name] = metric_dict(y[mask], pred[mask])

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "input_dim": input_dim,
        "hidden_sizes": parse_hidden_sizes(args.hidden_sizes),
        "dropout": args.dropout,
        "target_mean_train": y_mean,
        "target_std_train": y_std,
        "encoder_model": args.encoder_model,
        "pooling": args.pooling,
    }
    return result, metrics, checkpoint


def write_mlp_outputs(
    output_dir: Path,
    result: pd.DataFrame,
    metrics: Dict[str, Any],
    checkpoint: Dict[str, Any],
) -> None:
    import torch

    ensure_dir(output_dir)
    save_csv(result, output_dir / "tone_predictions.csv")
    with (output_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2, ensure_ascii=False)
    torch.save(checkpoint, output_dir / "mlp_tone_regressor.pt")


def summarize_cv_metrics(
    summary: pd.DataFrame,
) -> Dict[str, Dict[str, Dict[str, Optional[float]]]]:
    aggregate: Dict[str, Dict[str, Dict[str, Optional[float]]]] = {}
    metric_names = ["count", "mse", "rmse", "mae", "r2"]
    for split_name in ["train", "val", "test", "all"]:
        aggregate[split_name] = {}
        for metric_name in metric_names:
            column = f"{split_name}_{metric_name}"
            if column not in summary.columns:
                continue
            values = pd.to_numeric(summary[column], errors="coerce").dropna().to_numpy(
                dtype=np.float64
            )
            if len(values) == 0:
                aggregate[split_name][metric_name] = {
                    "count": 0,
                    "mean": None,
                    "std": None,
                    "min": None,
                    "max": None,
                }
                continue
            aggregate[split_name][metric_name] = {
                "count": int(len(values)),
                "mean": float(np.mean(values)),
                "std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
                "min": float(np.min(values)),
                "max": float(np.max(values)),
            }
    return aggregate


def flatten_cv_aggregate(
    aggregate: Dict[str, Dict[str, Dict[str, Optional[float]]]]
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for split_name, split_metrics in aggregate.items():
        for metric_name, stats in split_metrics.items():
            row = {"split": split_name, "metric": metric_name}
            row.update(stats)
            rows.append(row)
    return pd.DataFrame(rows)


def aggregate_compat_metrics(
    aggregate: Dict[str, Dict[str, Dict[str, Optional[float]]]],
    final_fold_metrics: Dict[str, Any],
) -> Dict[str, Dict[str, Optional[float]]]:
    metrics: Dict[str, Dict[str, Optional[float]]] = {}
    for split_name in ["train", "val", "test", "all"]:
        split_metrics: Dict[str, Optional[float]] = {}
        for metric_name in ["mse", "rmse", "mae", "r2"]:
            stats = aggregate.get(split_name, {}).get(metric_name, {})
            split_metrics[metric_name] = stats.get("mean")
        final_count = final_fold_metrics["metrics"].get(split_name, {}).get("count")
        split_metrics["count"] = final_count
        metrics[split_name] = split_metrics
    return metrics


def train_mlp_time_series_cv(
    forecasts: pd.DataFrame,
    embeddings: np.ndarray,
    args: argparse.Namespace,
) -> None:
    fold_splits, fold_metadata = build_time_series_cv_splits(forecasts, args)
    save_csv(fold_metadata, args.output_dir / "mlp_time_series_split_metadata.csv")

    summary_rows: List[Dict[str, Any]] = []
    final_result: Optional[pd.DataFrame] = None
    final_metrics: Optional[Dict[str, Any]] = None
    final_checkpoint: Optional[Dict[str, Any]] = None

    for fold_index, fold_split in enumerate(fold_splits):
        fold_dir = args.output_dir / f"fold_{fold_index:02d}"
        fold_forecasts = apply_time_series_split(forecasts, fold_split, allow_unused=True)
        save_csv(fold_forecasts, fold_dir / "forecast_events.csv")
        fold_result, fold_metrics, fold_checkpoint = fit_mlp_from_embeddings(
            fold_forecasts,
            embeddings,
            args,
        )
        fold_metrics["split_strategy"] = "time-series-fold"
        fold_metrics["fold"] = fold_index
        write_mlp_outputs(fold_dir, fold_result, fold_metrics, fold_checkpoint)

        row = fold_metadata.iloc[fold_index].to_dict()
        row.update(
            {
                "best_epoch": fold_metrics.get("best_epoch"),
                "best_val_loss_scaled": fold_metrics.get("best_val_loss_scaled"),
                "epochs_run": fold_metrics.get("epochs_run"),
                "output_dir": str(fold_dir),
            }
        )
        for split_name in ["train", "val", "test", "all"]:
            for metric_name, value in fold_metrics["metrics"][split_name].items():
                row[f"{split_name}_{metric_name}"] = value
        summary_rows.append(row)

        final_result = fold_result
        final_metrics = fold_metrics
        final_checkpoint = fold_checkpoint

    if final_result is None or final_metrics is None or final_checkpoint is None:
        raise RuntimeError("No time-series CV folds were trained.")

    summary = pd.DataFrame(summary_rows)
    aggregate = summarize_cv_metrics(summary)
    aggregate_frame = flatten_cv_aggregate(aggregate)
    save_csv(summary, args.output_dir / "mlp_time_series_cv_summary.csv")
    save_csv(aggregate_frame, args.output_dir / "mlp_time_series_cv_aggregate_metrics.csv")

    top_level_metrics = dict(final_metrics)
    top_level_metrics.update(
        {
            "split_strategy": "time-series",
            "metrics_source": "time_series_cv_mean",
            "tone_predictions_source": "final_fold",
            "time_series_cv": {
                "tscv_n_splits": int(args.tscv_n_splits),
                "tscv_test_size": None if args.tscv_test_size is None else int(args.tscv_test_size),
                "tscv_gap": int(args.tscv_gap),
                "final_test_ratio": float(args.final_test_ratio),
                "final_fold": summary_rows[-1],
            },
            "aggregate_metrics": aggregate,
            "folds": summary_rows,
            "final_fold_metrics": final_metrics,
            "metrics": aggregate_compat_metrics(aggregate, final_metrics),
        }
    )
    write_mlp_outputs(args.output_dir, final_result, top_level_metrics, final_checkpoint)


def train_mlp(forecasts: pd.DataFrame, args: argparse.Namespace) -> None:
    forecasts = prepare_forecasts(forecasts)
    set_seed(args.seed)
    embeddings = encode_texts(forecasts["predicted_event_text"].tolist(), args)
    save_embedding_cache(forecasts, embeddings, args.output_dir)

    if args.split_strategy == "time-series":
        train_mlp_time_series_cv(forecasts, embeddings, args)
        return

    result, metrics, checkpoint = fit_mlp_from_embeddings(forecasts, embeddings, args)
    write_mlp_outputs(args.output_dir, result, metrics, checkpoint)

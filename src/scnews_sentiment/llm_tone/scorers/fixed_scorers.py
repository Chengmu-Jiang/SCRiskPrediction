from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

from src.scnews_sentiment.llm_tone.data import (
    apply_time_series_split,
    build_time_series_cv_splits,
)
from src.scnews_sentiment.llm_tone.models import build_tone_regressor, encode_texts
from src.scnews_sentiment.llm_tone.train import (
    aggregate_compat_metrics,
    flatten_cv_aggregate,
    summarize_cv_metrics,
)
from src.scnews_sentiment.llm_tone.utils import (
    clean_generated_text,
    ensure_dir,
    metric_dict,
    parse_hidden_sizes,
    read_csv_or_empty,
    require_columns,
    resolve_device,
    save_csv,
    set_seed,
)

SPLIT_NAMES = ["train", "val", "test", "all"]


def parse_float_list(text: str) -> List[float]:
    values = [float(part.strip()) for part in str(text).split(",") if part.strip()]
    if not values:
        raise ValueError("Expected at least one numeric value.")
    return values


def load_or_encode_embeddings(
    ids: List[str],
    texts: List[str],
    cache_path: Path,
    args: argparse.Namespace,
) -> np.ndarray:
    cache_path = Path(cache_path)
    if cache_path.exists():
        cached = np.load(cache_path, allow_pickle=True)
        cached_ids = cached["id"].astype(str).tolist()
        if cached_ids == ids:
            return cached["embedding"]

    ensure_dir(cache_path.parent)
    embeddings = encode_texts(texts, args)
    np.savez(
        cache_path,
        id=np.asarray(ids, dtype=str),
        embedding=embeddings,
    )
    return embeddings


def save_sample_aligned_real_news_embeddings(
    real_ids: List[str],
    real_embeddings: np.ndarray,
    sample_mapping: pd.DataFrame,
    output_path: Path,
) -> Path:
    """Export target-day real-news embeddings with structured sample IDs."""
    require_columns(sample_mapping, ["daily_id", "sample_id"], "daily_compressions_real.csv")

    embeddings = np.asarray(real_embeddings)
    if embeddings.ndim != 2 or embeddings.shape[0] != len(real_ids):
        raise ValueError(
            "Real-news embedding row count must match the number of real-news IDs."
        )

    normalized_ids = [str(value) for value in real_ids]
    if len(set(normalized_ids)) != len(normalized_ids):
        raise ValueError("Real-news embedding IDs must be unique.")

    mapping = sample_mapping[["daily_id", "sample_id"]].copy()
    mapping["daily_id"] = mapping["daily_id"].fillna("").astype(str).str.strip()
    mapping["sample_id"] = mapping["sample_id"].fillna("").astype(str).str.strip()
    mapping = mapping[(mapping["daily_id"] != "") & (mapping["sample_id"] != "")].reset_index(drop=True)
    if mapping.empty:
        raise ValueError("daily_compressions_real.csv has no rows with both daily_id and sample_id.")
    if mapping["daily_id"].duplicated().any():
        duplicates = mapping.loc[mapping["daily_id"].duplicated(keep=False), "daily_id"]
        raise ValueError(f"Duplicate daily_id values in real-news sample mapping: {duplicates.head(5).tolist()}")
    if mapping["sample_id"].duplicated().any():
        duplicates = mapping.loc[mapping["sample_id"].duplicated(keep=False), "sample_id"]
        raise ValueError(f"Duplicate sample_id values in real-news sample mapping: {duplicates.head(5).tolist()}")

    embedding_lookup = {daily_id: index for index, daily_id in enumerate(normalized_ids)}
    missing_ids = sorted(set(mapping["daily_id"]) - set(embedding_lookup))
    if missing_ids:
        raise ValueError(
            "Real-news sample mapping contains daily_id values without embeddings. "
            f"Examples: {missing_ids[:5]}"
        )

    selected_indices = [embedding_lookup[daily_id] for daily_id in mapping["daily_id"]]
    sample_embeddings = embeddings[selected_indices]
    ensure_dir(Path(output_path).parent)
    np.savez(
        output_path,
        sample_id=mapping["sample_id"].to_numpy(dtype=str),
        embedding=sample_embeddings,
        daily_id=mapping["daily_id"].to_numpy(dtype=str),
    )
    return Path(output_path)


def build_real_news_scorer_frame(
    compressions: pd.DataFrame,
    samples: pd.DataFrame,
) -> pd.DataFrame:
    require_columns(
        compressions,
        ["daily_id", "date_iso", "article_score", "compressed_text"],
        "daily_compressions.csv",
    )
    require_columns(
        samples,
        ["sample_id", "target_daily_id", "target_date", "split"],
        "samples",
    )

    target_lookup = samples[["sample_id", "target_daily_id", "target_date", "split"]].copy()
    if target_lookup["target_daily_id"].duplicated().any():
        duplicates = target_lookup.loc[
            target_lookup["target_daily_id"].duplicated(),
            "target_daily_id",
        ].head(10)
        raise ValueError(f"Duplicate target_daily_id values in samples: {duplicates.tolist()}")

    frame = compressions.merge(
        target_lookup,
        left_on="daily_id",
        right_on="target_daily_id",
        how="left",
    )
    frame["sample_id"] = frame["sample_id"].fillna("")
    frame["target_date"] = frame["target_date"].fillna(frame["date_iso"])
    frame["split"] = frame["split"].fillna("train")
    frame["split_source"] = np.where(frame["target_daily_id"].notna(), "sample", "early_real_news")
    frame["text"] = frame["compressed_text"].fillna("").astype(str).map(clean_generated_text)
    frame["target_article_score"] = frame["article_score"].astype(float)
    frame = frame.drop(columns=["target_daily_id"])

    if (frame["text"].str.len() == 0).any():
        empty_ids = frame.loc[frame["text"].str.len() == 0, "daily_id"].head(10).tolist()
        raise ValueError(f"daily_compressions.csv contains empty compressed_text values: {empty_ids}")
    return frame.sort_values(["topic", "date_iso"]).reset_index(drop=True)


def split_metric_summary(frame: pd.DataFrame, pred: np.ndarray) -> Dict[str, Any]:
    y_true = frame["target_article_score"].astype(float).to_numpy()
    split = frame["split"].astype(str).to_numpy()
    summary: Dict[str, Any] = {}
    active_mask = np.isin(split, ["train", "val", "test"])
    for split_name in SPLIT_NAMES:
        mask = active_mask if split_name == "all" else split == split_name
        summary[split_name] = metric_dict(y_true[mask], pred[mask])
    return summary


def prediction_frame(frame: pd.DataFrame, pred: np.ndarray, text_column: str) -> pd.DataFrame:
    result = frame.copy()
    result["predicted_article_score"] = pred
    result["abs_error"] = np.abs(
        result["predicted_article_score"] - result["target_article_score"].astype(float)
    )
    result["squared_error"] = (
        result["predicted_article_score"] - result["target_article_score"].astype(float)
    ) ** 2
    keep_columns = [
        col
        for col in [
            "daily_id",
            "sample_id",
            "split",
            "split_source",
            "topic",
            "target_date",
            "date_iso",
            "target_article_score",
            "predicted_article_score",
            "abs_error",
            "squared_error",
            text_column,
        ]
        if col in result.columns
    ]
    return result[keep_columns]


def add_metrics_to_row(row: Dict[str, Any], metrics: Dict[str, Any]) -> Dict[str, Any]:
    for split_name in SPLIT_NAMES:
        for metric_name, value in metrics[split_name].items():
            row[f"{split_name}_{metric_name}"] = value
    return row


def build_metric_row(
    metrics: Dict[str, Any],
    *,
    metadata: Dict[str, Any],
) -> Dict[str, Any]:
    row = dict(metadata)
    return add_metrics_to_row(row, metrics)


def fold_real_news_frame(
    real_frame: pd.DataFrame,
    fold_split: Dict[str, set[str]],
) -> pd.DataFrame:
    result = apply_time_series_split(real_frame, fold_split, allow_unused=True)
    active_target_dates = set().union(
        fold_split["train"],
        fold_split["val"],
        fold_split["test"],
    )
    first_active_date = min(active_target_dates)
    early_real_news = (result["split_source"] == "early_real_news") & (
        result["target_date"].astype(str) < first_active_date
    )
    result.loc[early_real_news, "split"] = "train"
    return result


def make_scorer(kind: str, args: argparse.Namespace) -> Any:
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    if kind == "ridge":
        from sklearn.linear_model import RidgeCV

        return make_pipeline(
            StandardScaler(),
            RidgeCV(alphas=parse_float_list(args.ridge_alphas)),
        )
    if kind == "huber":
        from sklearn.linear_model import HuberRegressor

        return make_pipeline(
            StandardScaler(),
            HuberRegressor(
                alpha=args.huber_alpha,
                epsilon=args.huber_epsilon,
                max_iter=args.huber_max_iter,
            ),
        )
    raise ValueError(f"Unsupported fixed scorer kind: {kind}")


def final_estimator(model: Any) -> Any:
    if hasattr(model, "steps") and model.steps:
        return model.steps[-1][1]
    return model


def save_model(model: Any, output_path: Path) -> None:
    import joblib

    ensure_dir(output_path.parent)
    joblib.dump(model, output_path)


class FixedMlpScorer:
    def __init__(
        self,
        model: Any,
        input_mean: np.ndarray,
        input_std: np.ndarray,
        target_mean: float,
        target_std: float,
        device: str,
    ) -> None:
        self.model = model
        self.input_mean = input_mean.astype(np.float32)
        self.input_std = input_std.astype(np.float32)
        self.target_mean = float(target_mean)
        self.target_std = float(target_std)
        self.device = device

    def predict(self, embeddings: np.ndarray) -> np.ndarray:
        import torch

        x = (np.asarray(embeddings, dtype=np.float32) - self.input_mean) / self.input_std
        x_tensor = torch.tensor(x, dtype=torch.float32, device=self.device)
        self.model.eval()
        with torch.no_grad():
            pred_scaled = self.model(x_tensor).detach().cpu().numpy().reshape(-1)
        return pred_scaled * self.target_std + self.target_mean


def mlp_scorer_config(kind: str, args: argparse.Namespace) -> Dict[str, Any]:
    if kind == "mlp":
        return {
            "architecture_source": "fixed_lightweight_mlp",
            "hidden_sizes": args.fixed_mlp_hidden_sizes,
            "dropout": args.fixed_mlp_dropout,
            "batch_size": args.fixed_mlp_batch_size,
            "epochs": args.fixed_mlp_epochs,
            "lr": args.fixed_mlp_lr,
            "weight_decay": args.fixed_mlp_weight_decay,
            "patience": args.fixed_mlp_patience,
            "min_delta": args.fixed_mlp_min_delta,
        }
    if kind == "mlp_original_arch":
        return {
            "architecture_source": "original_non_fixed_mlp",
            "hidden_sizes": args.hidden_sizes,
            "dropout": args.dropout,
            "batch_size": args.batch_size,
            "epochs": args.epochs,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "patience": args.patience,
            "min_delta": args.min_delta,
        }
    raise ValueError(f"Unsupported fixed MLP scorer kind: {kind}")


def train_real_news_mlp_scorer(
    kind: str,
    real_frame: pd.DataFrame,
    real_embeddings: np.ndarray,
    scorer_dir: Path,
    args: argparse.Namespace,
) -> Tuple[FixedMlpScorer, Dict[str, Any]]:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset

    split = real_frame["split"].astype(str).to_numpy()
    train_mask = split == "train"
    val_mask = split == "val"
    if train_mask.sum() == 0:
        raise ValueError("No train rows available for fixed MLP scorer training.")

    config = mlp_scorer_config(kind, args)
    set_seed(args.seed)
    y = real_frame["target_article_score"].astype(float).to_numpy()
    y_mean = float(y[train_mask].mean())
    y_std = float(y[train_mask].std())
    if y_std == 0:
        y_std = 1.0
    y_scaled = (y - y_mean) / y_std

    input_mean = real_embeddings[train_mask].mean(axis=0).astype(np.float32)
    input_std = real_embeddings[train_mask].std(axis=0).astype(np.float32)
    input_std[input_std < 1e-8] = 1.0
    x_scaled = (real_embeddings.astype(np.float32) - input_mean) / input_std

    device = resolve_device(args.train_device)
    x_tensor = torch.tensor(x_scaled, dtype=torch.float32)
    y_tensor = torch.tensor(y_scaled.reshape(-1, 1), dtype=torch.float32)

    train_ds = TensorDataset(x_tensor[train_mask], y_tensor[train_mask])
    train_loader = DataLoader(
        train_ds,
        batch_size=config["batch_size"],
        shuffle=True,
    )

    input_dim = x_scaled.shape[1]
    hidden_sizes = parse_hidden_sizes(config["hidden_sizes"])
    model = build_tone_regressor(
        input_dim=input_dim,
        hidden_sizes=config["hidden_sizes"],
        dropout=config["dropout"],
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["lr"],
        weight_decay=config["weight_decay"],
    )
    loss_fn = nn.MSELoss()

    x_tensor = x_tensor.to(device)
    y_tensor = y_tensor.to(device)
    best_state: Dict[str, Any] | None = None
    best_val = float("inf")
    best_epoch = 0
    bad_epochs = 0
    epochs_run = 0
    early_stop_mask = val_mask if val_mask.sum() > 0 else train_mask

    def eval_loss(mask: np.ndarray) -> float:
        model.eval()
        with torch.no_grad():
            pred = model(x_tensor[mask])
            return float(loss_fn(pred, y_tensor[mask]).item())

    for epoch in range(1, config["epochs"] + 1):
        epochs_run = epoch
        model.train()
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            optimizer.step()

        val_loss = eval_loss(early_stop_mask)
        if val_loss < best_val - config["min_delta"]:
            best_val = val_loss
            best_epoch = epoch
            best_state = copy.deepcopy(
                {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            )
            bad_epochs = 0
        else:
            bad_epochs += 1
        if bad_epochs >= config["patience"]:
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    scorer = FixedMlpScorer(
        model=model,
        input_mean=input_mean,
        input_std=input_std,
        target_mean=y_mean,
        target_std=y_std,
        device=device,
    )
    pred = scorer.predict(real_embeddings)

    real_predictions = prediction_frame(
        real_frame.rename(columns={"text": "compressed_text_for_scorer"}),
        pred,
        text_column="compressed_text_for_scorer",
    )
    save_csv(real_predictions, scorer_dir / "real_news_predictions.csv")

    metrics = {
        "scorer_kind": kind,
        "training_source": "daily_compressions.csv",
        "unmatched_real_news_assigned_split": "train",
        "encoder_model": args.encoder_model,
        "encoder_device": args.encoder_device,
        "train_device": args.train_device,
        "resolved_train_device": device,
        "pooling": args.pooling,
        "input_scaling": "standardize_embeddings_with_real_train_split",
        "architecture_source": config["architecture_source"],
        "target_mean_train": y_mean,
        "target_std_train": y_std,
        "best_val_loss_scaled": best_val,
        "best_epoch": best_epoch,
        "epochs_run": epochs_run,
        "early_stop_split": "val" if val_mask.sum() > 0 else "train",
        "mlp_hidden_sizes": list(hidden_sizes),
        "mlp_dropout": config["dropout"],
        "mlp_batch_size": config["batch_size"],
        "mlp_epochs": config["epochs"],
        "mlp_lr": config["lr"],
        "mlp_weight_decay": config["weight_decay"],
        "mlp_patience": config["patience"],
        "mlp_min_delta": config["min_delta"],
        "parameter_count": int(sum(param.numel() for param in model.parameters())),
        "metrics": split_metric_summary(real_frame, pred),
    }
    if kind == "mlp":
        metrics.update(
            {
                "fixed_mlp_hidden_sizes": list(hidden_sizes),
                "fixed_mlp_dropout": config["dropout"],
                "fixed_mlp_batch_size": config["batch_size"],
                "fixed_mlp_epochs": config["epochs"],
                "fixed_mlp_lr": config["lr"],
                "fixed_mlp_weight_decay": config["weight_decay"],
                "fixed_mlp_patience": config["patience"],
                "fixed_mlp_min_delta": config["min_delta"],
            }
        )
    elif kind == "mlp_original_arch":
        metrics.update(
            {
                "original_mlp_hidden_sizes": list(hidden_sizes),
                "original_mlp_dropout": config["dropout"],
                "original_mlp_batch_size": config["batch_size"],
                "original_mlp_epochs": config["epochs"],
                "original_mlp_lr": config["lr"],
                "original_mlp_weight_decay": config["weight_decay"],
                "original_mlp_patience": config["patience"],
                "original_mlp_min_delta": config["min_delta"],
            }
        )
    with (scorer_dir / "real_news_metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2, ensure_ascii=False)

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "input_dim": input_dim,
            "hidden_sizes": hidden_sizes,
            "dropout": config["dropout"],
            "scorer_kind": kind,
            "architecture_source": config["architecture_source"],
            "input_mean_train": input_mean,
            "input_std_train": input_std,
            "target_mean_train": y_mean,
            "target_std_train": y_std,
            "encoder_model": args.encoder_model,
            "pooling": args.pooling,
        },
        scorer_dir / "model.pt",
    )
    return scorer, metrics


def train_real_news_scorer(
    kind: str,
    real_frame: pd.DataFrame,
    real_embeddings: np.ndarray,
    scorer_dir: Path,
    args: argparse.Namespace,
) -> Tuple[Any, Dict[str, Any]]:
    if kind in {"mlp", "mlp_original_arch"}:
        return train_real_news_mlp_scorer(
            kind=kind,
            real_frame=real_frame,
            real_embeddings=real_embeddings,
            scorer_dir=scorer_dir,
            args=args,
        )

    train_mask = real_frame["split"].astype(str).to_numpy() == "train"
    if train_mask.sum() == 0:
        raise ValueError("No train rows available for fixed scorer training.")

    model = make_scorer(kind, args)
    y = real_frame["target_article_score"].astype(float).to_numpy()
    model.fit(real_embeddings[train_mask], y[train_mask])
    pred = model.predict(real_embeddings)

    real_predictions = prediction_frame(
        real_frame.rename(columns={"text": "compressed_text_for_scorer"}),
        pred,
        text_column="compressed_text_for_scorer",
    )
    save_csv(real_predictions, scorer_dir / "real_news_predictions.csv")

    metrics = {
        "scorer_kind": kind,
        "training_source": "daily_compressions.csv",
        "unmatched_real_news_assigned_split": "train",
        "encoder_model": args.encoder_model,
        "pooling": args.pooling,
        "metrics": split_metric_summary(real_frame, pred),
    }
    if kind == "ridge":
        estimator = final_estimator(model)
        metrics["ridge_alphas"] = parse_float_list(args.ridge_alphas)
        metrics["ridge_selected_alpha"] = float(getattr(estimator, "alpha_", np.nan))
    elif kind == "huber":
        estimator = final_estimator(model)
        metrics["huber_alpha"] = args.huber_alpha
        metrics["huber_epsilon"] = args.huber_epsilon
        metrics["huber_max_iter"] = args.huber_max_iter
        metrics["huber_n_iter"] = int(getattr(estimator, "n_iter_", -1))

    with (scorer_dir / "real_news_metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2, ensure_ascii=False)
    save_model(model, scorer_dir / "model.joblib")
    return model, metrics


def load_round_forecasts(round_dir: Path) -> pd.DataFrame:
    forecast_path = round_dir / "forecast_events.csv"
    forecasts = read_csv_or_empty(forecast_path)
    if forecasts.empty:
        raise FileNotFoundError(f"{forecast_path} is missing or empty.")
    require_columns(
        forecasts,
        [
            "sample_id",
            "split",
            "topic",
            "target_date",
            "target_article_score",
            "predicted_event_text",
        ],
        str(forecast_path),
    )
    forecasts = forecasts.copy()
    forecasts["predicted_event_text"] = (
        forecasts["predicted_event_text"].fillna("").astype(str).map(clean_generated_text)
    )
    if (forecasts["predicted_event_text"].str.len() == 0).any():
        empty_ids = forecasts.loc[
            forecasts["predicted_event_text"].str.len() == 0,
            "sample_id",
        ].head(10)
        raise ValueError(f"{forecast_path} contains empty predicted_event_text: {empty_ids.tolist()}")
    return forecasts.sort_values(["topic", "target_date"]).reset_index(drop=True)


def evaluate_round_with_scorer(
    model: Any,
    round_name: str,
    round_dir: Path,
    scorer_dir: Path,
    embedding_cache_dir: Path,
    args: argparse.Namespace,
    fold_split: Dict[str, set[str]] | None = None,
) -> Dict[str, Any]:
    forecasts = load_round_forecasts(round_dir)
    if fold_split is not None:
        forecasts = apply_time_series_split(forecasts, fold_split, allow_unused=True)
    ids = forecasts["sample_id"].astype(str).tolist()
    embeddings = load_or_encode_embeddings(
        ids=ids,
        texts=forecasts["predicted_event_text"].tolist(),
        cache_path=embedding_cache_dir / f"{round_name}_forecast_embeddings.npz",
        args=args,
    )
    pred = model.predict(embeddings)

    eval_frame = forecasts.copy()
    eval_frame["target_article_score"] = eval_frame["target_article_score"].astype(float)
    predictions = prediction_frame(eval_frame, pred, text_column="predicted_event_text")
    save_csv(predictions, scorer_dir / f"{round_name}_tone_predictions.csv")

    metrics = split_metric_summary(eval_frame, pred)
    with (scorer_dir / f"{round_name}_metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "round": round_name,
                "round_dir": str(round_dir),
                "metrics": metrics,
            },
            handle,
            indent=2,
            ensure_ascii=False,
        )

    return build_metric_row(
        metrics,
        metadata={"round": round_name, "round_dir": str(round_dir)},
    )


def discover_round_dirs(opt_dir: Path, include_final: bool = False) -> List[Path]:
    if not opt_dir.exists():
        raise FileNotFoundError(f"{opt_dir} does not exist. Run --stage optimize-prompt first.")
    round_dirs = sorted(path for path in opt_dir.glob("round_*") if path.is_dir())
    if not round_dirs:
        raise FileNotFoundError(f"No round_* directories found under {opt_dir}.")
    if include_final:
        final_dir = opt_dir / "final"
        if not final_dir.exists():
            raise FileNotFoundError(f"{final_dir} does not exist. Run --stage optimize-prompt first.")
        round_dirs.append(final_dir)
    return round_dirs


def wide_aggregate_rows(summary: pd.DataFrame, group_column: str) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    aggregate_by_group: Dict[str, Any] = {}
    for group_value, group in summary.groupby(group_column, sort=False):
        aggregate = summarize_cv_metrics(group)
        group_key = str(group_value)
        aggregate_by_group[group_key] = aggregate
        row: Dict[str, Any] = {group_column: group_key}
        if "round_dir" in group.columns:
            row["round_dir"] = str(group["round_dir"].iloc[0])
        for split_name, split_metrics in aggregate.items():
            for metric_name, stats in split_metrics.items():
                row[f"{split_name}_{metric_name}"] = stats.get("mean")
                row[f"{split_name}_{metric_name}_std"] = stats.get("std")
                row[f"{split_name}_{metric_name}_min"] = stats.get("min")
                row[f"{split_name}_{metric_name}_max"] = stats.get("max")
                row[f"{split_name}_{metric_name}_fold_count"] = stats.get("count")
        rows.append(row)
    return rows, aggregate_by_group


def flatten_grouped_cv_aggregate(
    aggregate_by_group: Dict[str, Any],
    group_column: str,
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for group_value, aggregate in aggregate_by_group.items():
        for split_name, split_metrics in aggregate.items():
            for metric_name, stats in split_metrics.items():
                row = {group_column: group_value, "split": split_name, "metric": metric_name}
                row.update(stats)
                rows.append(row)
    return pd.DataFrame(rows)


def write_real_news_cv_outputs(
    scorer_dir: Path,
    summary_rows: List[Dict[str, Any]],
    final_real_metrics: Dict[str, Any],
    args: argparse.Namespace,
) -> Dict[str, Any]:
    summary = pd.DataFrame(summary_rows)
    aggregate = summarize_cv_metrics(summary)
    aggregate_frame = flatten_cv_aggregate(aggregate)
    save_csv(summary, scorer_dir / "real_news_time_series_cv_summary.csv")
    save_csv(aggregate_frame, scorer_dir / "real_news_time_series_cv_aggregate_metrics.csv")

    final_fold_metrics = final_real_metrics
    metrics = dict(final_fold_metrics)
    metrics.update(
        {
            "split_strategy": "time-series",
            "metrics_source": "time_series_cv_mean",
            "prediction_source": "final_fold",
            "time_series_cv": {
                "tscv_n_splits": int(args.tscv_n_splits),
                "tscv_test_size": None if args.tscv_test_size is None else int(args.tscv_test_size),
                "tscv_gap": int(args.tscv_gap),
                "final_test_ratio": float(args.final_test_ratio),
                "final_fold": summary_rows[-1],
            },
            "aggregate_metrics": aggregate,
            "folds": json.loads(summary.to_json(orient="records")),
            "final_fold_metrics": final_fold_metrics,
            "metrics": aggregate_compat_metrics(aggregate, final_fold_metrics),
        }
    )
    with (scorer_dir / "real_news_metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2, ensure_ascii=False)
    return metrics


def write_round_cv_outputs(
    scorer_dir: Path,
    round_rows: List[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    summary = pd.DataFrame(round_rows)
    save_csv(summary, scorer_dir / "round_time_series_cv_summary.csv")

    wide_rows, aggregate_by_round = wide_aggregate_rows(summary, "round")
    save_csv(pd.DataFrame(wide_rows), scorer_dir / "round_metrics.csv")
    save_csv(
        flatten_grouped_cv_aggregate(aggregate_by_round, "round"),
        scorer_dir / "round_time_series_cv_aggregate_metrics.csv",
    )
    return wide_rows, aggregate_by_round


def train_fixed_scorer_time_series_cv(
    kind: str,
    real_frame: pd.DataFrame,
    real_embeddings: np.ndarray,
    round_dirs: List[Path],
    scorer_dir: Path,
    embedding_cache_dir: Path,
    samples: pd.DataFrame,
    args: argparse.Namespace,
) -> None:
    fold_splits, fold_metadata = build_time_series_cv_splits(samples, args)
    save_csv(fold_metadata, scorer_dir / "fixed_time_series_split_metadata.csv")

    real_summary_rows: List[Dict[str, Any]] = []
    round_rows: List[Dict[str, Any]] = []
    final_real_metrics: Dict[str, Any] | None = None

    for fold_index, fold_split in enumerate(fold_splits):
        fold_dir = scorer_dir / f"fold_{fold_index:02d}"
        ensure_dir(fold_dir)
        fold_real_frame = fold_real_news_frame(real_frame, fold_split)
        save_csv(fold_real_frame, fold_dir / "real_news_frame.csv")

        model, real_metrics = train_real_news_scorer(
            kind=kind,
            real_frame=fold_real_frame,
            real_embeddings=real_embeddings,
            scorer_dir=fold_dir,
            args=args,
        )
        real_metrics["split_strategy"] = "time-series-fold"
        real_metrics["fold"] = fold_index
        with (fold_dir / "real_news_metrics.json").open("w", encoding="utf-8") as handle:
            json.dump(real_metrics, handle, indent=2, ensure_ascii=False)

        row = fold_metadata.iloc[fold_index].to_dict()
        row.update(
            {
                "fold": fold_index,
                "output_dir": str(fold_dir),
                "scorer_kind": kind,
                "best_epoch": real_metrics.get("best_epoch"),
                "best_val_loss_scaled": real_metrics.get("best_val_loss_scaled"),
                "epochs_run": real_metrics.get("epochs_run"),
                "early_stop_split": real_metrics.get("early_stop_split"),
            }
        )
        add_metrics_to_row(row, real_metrics["metrics"])
        real_summary_rows.append(row)

        for round_dir in round_dirs:
            round_row = evaluate_round_with_scorer(
                model=model,
                round_name=round_dir.name,
                round_dir=round_dir,
                scorer_dir=fold_dir,
                embedding_cache_dir=embedding_cache_dir,
                args=args,
                fold_split=fold_split,
            )
            round_row.update(
                {
                    "fold": fold_index,
                    "fold_output_dir": str(fold_dir),
                    "scorer_kind": kind,
                }
            )
            round_rows.append(round_row)

        final_real_metrics = real_metrics

    if final_real_metrics is None:
        raise RuntimeError("No fixed scorer time-series CV folds were trained.")

    real_metrics = write_real_news_cv_outputs(
        scorer_dir=scorer_dir,
        summary_rows=real_summary_rows,
        final_real_metrics=final_real_metrics,
        args=args,
    )
    round_metric_rows, round_aggregate = write_round_cv_outputs(scorer_dir, round_rows)
    summary = {
        "scorer_kind": kind,
        "split_strategy": "time-series",
        "real_news_metrics": real_metrics,
        "round_metrics": round_metric_rows,
        "round_aggregate_metrics": round_aggregate,
    }
    with (scorer_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)


def evaluate_fixed_scorers(samples: pd.DataFrame, args: argparse.Namespace) -> None:
    compressions_path = args.output_dir / "daily_compressions.csv"
    compressions = read_csv_or_empty(compressions_path)
    if compressions.empty:
        raise FileNotFoundError(f"{compressions_path} is missing or empty. Run --stage compress first.")

    fixed_dir = args.output_dir / "fixed_scorers"
    embedding_cache_dir = fixed_dir / "embeddings"
    ensure_dir(embedding_cache_dir)

    real_frame = build_real_news_scorer_frame(compressions, samples)
    real_ids = real_frame["daily_id"].astype(str).tolist()
    real_embeddings = load_or_encode_embeddings(
        ids=real_ids,
        texts=real_frame["text"].tolist(),
        cache_path=embedding_cache_dir / "real_news_embeddings.npz",
        args=args,
    )
    sample_mapping_path = args.output_dir / "daily_compressions_real.csv"
    sample_mapping = read_csv_or_empty(sample_mapping_path)
    if sample_mapping.empty:
        raise FileNotFoundError(
            f"{sample_mapping_path} is missing or empty. Run --stage compress first."
        )
    save_sample_aligned_real_news_embeddings(
        real_ids=real_ids,
        real_embeddings=real_embeddings,
        sample_mapping=sample_mapping,
        output_path=args.output_dir / "real_news_sample_embeddings.npz",
    )

    round_dirs = discover_round_dirs(
        args.output_dir / "prompt_optimization",
        include_final=args.fixed_scorer_include_final,
    )
    scorer_kinds = getattr(
        args,
        "fixed_scorer_kinds",
        ["ridge", "huber", "mlp", "mlp_original_arch"],
    )
    if isinstance(scorer_kinds, str):
        scorer_kinds = [value.strip() for value in scorer_kinds.split(",") if value.strip()]
    for kind in scorer_kinds:
        scorer_dir = fixed_dir / kind
        ensure_dir(scorer_dir)
        if args.split_strategy == "time-series":
            train_fixed_scorer_time_series_cv(
                kind=kind,
                real_frame=real_frame,
                real_embeddings=real_embeddings,
                round_dirs=round_dirs,
                scorer_dir=scorer_dir,
                embedding_cache_dir=embedding_cache_dir,
                samples=samples,
                args=args,
            )
            continue

        model, real_metrics = train_real_news_scorer(
            kind=kind,
            real_frame=real_frame,
            real_embeddings=real_embeddings,
            scorer_dir=scorer_dir,
            args=args,
        )

        round_rows: List[Dict[str, Any]] = []
        for round_dir in round_dirs:
            round_rows.append(
                evaluate_round_with_scorer(
                    model=model,
                    round_name=round_dir.name,
                    round_dir=round_dir,
                    scorer_dir=scorer_dir,
                    embedding_cache_dir=embedding_cache_dir,
                    args=args,
                )
            )

        round_metrics = pd.DataFrame(round_rows)
        save_csv(round_metrics, scorer_dir / "round_metrics.csv")
        summary = {
            "scorer_kind": kind,
            "real_news_metrics": real_metrics,
            "round_metrics": round_rows,
        }
        with (scorer_dir / "summary.json").open("w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2, ensure_ascii=False)

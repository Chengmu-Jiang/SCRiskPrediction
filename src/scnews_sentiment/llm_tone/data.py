from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

from src.scnews_sentiment.llm_tone.utils import (
    ensure_dir,
    normalize_hs4,
    parse_topic_hs_map,
    require_columns,
    save_csv,
    stable_hash,
)


def load_hs_metadata(nodes_csv: Path) -> Dict[str, Dict[str, str]]:
    nodes = pd.read_csv(nodes_csv, encoding="utf-8-sig")
    require_columns(nodes, ["hs_code_4d"], str(nodes_csv))
    nodes["hs_code_4d_norm"] = nodes["hs_code_4d"].apply(normalize_hs4)
    metadata: Dict[str, Dict[str, str]] = {}
    for _, row in nodes.dropna(subset=["hs_code_4d_norm"]).iterrows():
        code = row["hs_code_4d_norm"]
        if not code or code in metadata:
            continue
        metadata[code] = {
            "short_label": str(row.get("short_label", "") or ""),
            "full_description": str(row.get("full_description", "") or ""),
        }
    return metadata


def load_daily_news(args: argparse.Namespace) -> pd.DataFrame:
    topic_hs_map = parse_topic_hs_map(args.topic_hs_map)
    hs_metadata = load_hs_metadata(args.nodes_csv)

    news = pd.read_csv(args.news_csv, encoding="utf-8-sig")
    require_columns(
        news,
        [
            args.topic_col,
            args.date_col,
            args.target_col,
            args.title_col,
            args.text_col,
        ],
        str(args.news_csv),
    )

    news = news.copy()
    news["date_raw"] = news[args.date_col].astype(str).str.replace(r"\.0$", "", regex=True)
    news["date"] = pd.to_datetime(news["date_raw"], format="%Y%m%d", errors="raise")
    news["date_iso"] = news["date"].dt.strftime("%Y-%m-%d")
    news["date_yyyymmdd"] = news["date"].dt.strftime("%Y%m%d")
    news["topic"] = news[args.topic_col].astype(str)
    news["article_score"] = pd.to_numeric(news[args.target_col], errors="raise")
    news["title"] = news[args.title_col].fillna("").astype(str)
    news["article_text"] = news[args.text_col].fillna("").astype(str)

    duplicated = news.duplicated(["topic", "date_yyyymmdd"])
    if duplicated.any():
        dup_rows = news.loc[duplicated, ["topic", "date_yyyymmdd"]].head(10)
        raise ValueError(
            "daily news must contain one row per Topic-Date. "
            f"Duplicate examples: {dup_rows.to_dict(orient='records')}"
        )

    missing_topics = sorted(set(news["topic"]) - set(topic_hs_map))
    if missing_topics:
        raise ValueError(
            f"Missing HS4 mapping for topics {missing_topics}. "
            "Pass --topic-hs-map Topic=HS4."
        )

    news["hs_code_4d"] = news["topic"].map(topic_hs_map).apply(normalize_hs4)
    news["hs_short_label"] = news["hs_code_4d"].map(
        lambda code: hs_metadata.get(code, {}).get("short_label", "")
    )
    news["hs_full_description"] = news["hs_code_4d"].map(
        lambda code: hs_metadata.get(code, {}).get("full_description", "")
    )

    unknown_codes = sorted(
        code for code in news["hs_code_4d"].unique() if code not in hs_metadata
    )
    if unknown_codes:
        raise ValueError(
            f"HS4 codes not found in nodes CSV after zero-fill normalization: {unknown_codes}"
        )

    news["daily_id"] = news.apply(
        lambda row: f"{row['topic']}|{row['date_yyyymmdd']}", axis=1
    )
    news["raw_text_hash"] = news.apply(
        lambda row: stable_hash(row["title"], row["article_text"], length=32),
        axis=1,
    )

    keep_cols = [
        "daily_id",
        "topic",
        "hs_code_4d",
        "hs_short_label",
        "hs_full_description",
        "date",
        "date_iso",
        "date_yyyymmdd",
        "article_score",
        "title",
        "article_text",
        "raw_text_hash",
    ]
    return news[keep_cols].sort_values(["topic", "date"]).reset_index(drop=True)


def split_sizes(n: int, train_ratio: float, val_ratio: float) -> Tuple[int, int, int]:
    if n <= 0:
        return 0, 0, 0
    if n == 1:
        return 1, 0, 0
    if n == 2:
        return 1, 0, 1

    n_train = max(1, int(math.floor(n * train_ratio)))
    n_val = max(1, int(math.floor(n * val_ratio)))
    n_test = n - n_train - n_val

    while n_test < 1 and n_train > 1:
        n_train -= 1
        n_test += 1
    while n_test < 1 and n_val > 1:
        n_val -= 1
        n_test += 1
    return n_train, n_val, n_test


def sorted_target_dates(samples: pd.DataFrame) -> List[str]:
    dates = samples[["target_date"]].drop_duplicates().copy()
    dates["target_date_dt"] = pd.to_datetime(dates["target_date"], errors="raise")
    return dates.sort_values("target_date_dt")["target_date"].astype(str).tolist()


def build_time_series_cv_splits(
    samples: pd.DataFrame,
    args: argparse.Namespace,
) -> Tuple[List[Dict[str, set[str]]], pd.DataFrame]:
    active_dates = sorted_target_dates(samples)
    n_total = len(active_dates)
    if n_total < 4:
        raise ValueError("Need at least four target dates for time-series CV.")

    final_test_ratio = float(getattr(args, "final_test_ratio", 0.15))
    if not (0.0 < final_test_ratio < 1.0):
        raise ValueError("--final-test-ratio must be in (0, 1).")

    n_splits = int(getattr(args, "tscv_n_splits", 5))
    if n_splits <= 0:
        raise ValueError("--tscv-n-splits must be positive.")

    test_size = getattr(args, "tscv_test_size", None)
    if test_size is not None:
        test_size = int(test_size)
        if test_size <= 0:
            raise ValueError("--tscv-test-size must be positive when provided.")

    gap = int(getattr(args, "tscv_gap", 0))
    if gap < 0:
        raise ValueError("--tscv-gap must be non-negative.")

    final_test_count = max(1, int(math.ceil(n_total * final_test_ratio)))
    cv_dates = active_dates[:-final_test_count]
    final_test_dates = active_dates[-final_test_count:]
    if len(cv_dates) < 3:
        raise ValueError(
            "Not enough pre-test target dates after reserving final holdout test."
        )

    val_size = len(cv_dates) // (n_splits + 1) if test_size is None else test_size
    if val_size <= 0:
        raise ValueError("TimeSeriesSplit validation size must be positive.")

    first_val_start = len(cv_dates) - n_splits * val_size
    first_train_end = first_val_start - gap
    if first_train_end <= 0:
        raise ValueError(
            "TimeSeriesSplit configuration leaves no training samples in the first fold. "
            "Reduce --tscv-n-splits, --tscv-test-size, or --tscv-gap."
        )

    date_position = {date_text: index for index, date_text in enumerate(active_dates)}
    split_sets: List[Dict[str, set[str]]] = []
    metadata_rows: List[Dict[str, Any]] = []
    for fold_index in range(n_splits):
        val_start = first_val_start + fold_index * val_size
        val_end = val_start + val_size
        train_end = val_start - gap
        train_dates = cv_dates[:train_end]
        val_dates = cv_dates[val_start:val_end]
        if not train_dates or not val_dates:
            raise ValueError(f"Empty train or validation split in fold {fold_index}.")

        fold_sets = {
            "train": set(train_dates),
            "val": set(val_dates),
            "test": set(final_test_dates),
        }
        split_sets.append(fold_sets)
        metadata_rows.append(
            {
                "fold": fold_index,
                "train_count": int(samples["target_date"].astype(str).isin(fold_sets["train"]).sum()),
                "val_count": int(samples["target_date"].astype(str).isin(fold_sets["val"]).sum()),
                "test_count": int(samples["target_date"].astype(str).isin(fold_sets["test"]).sum()),
                "train_date_count": int(len(train_dates)),
                "val_date_count": int(len(val_dates)),
                "test_date_count": int(len(final_test_dates)),
                "gap": int(gap),
                "val_size": int(val_size),
                "train_start_time": int(date_position[train_dates[0]]),
                "train_end_time": int(date_position[train_dates[-1]]),
                "val_start_time": int(date_position[val_dates[0]]),
                "val_end_time": int(date_position[val_dates[-1]]),
                "test_start_time": int(date_position[final_test_dates[0]]),
                "test_end_time": int(date_position[final_test_dates[-1]]),
                "train_start_date": train_dates[0],
                "train_end_date": train_dates[-1],
                "val_start_date": val_dates[0],
                "val_end_date": val_dates[-1],
                "test_start_date": final_test_dates[0],
                "test_end_date": final_test_dates[-1],
            }
        )

    return split_sets, pd.DataFrame(metadata_rows)


def apply_time_series_split(
    samples: pd.DataFrame,
    fold_split: Dict[str, set[str]],
    *,
    allow_unused: bool = False,
) -> pd.DataFrame:
    result = samples.copy()
    result["split"] = ""
    target_dates = result["target_date"].astype(str)
    for split_name in ["train", "val", "test"]:
        result.loc[target_dates.isin(fold_split[split_name]), "split"] = split_name
    if (result["split"] == "").any():
        if allow_unused:
            result.loc[result["split"] == "", "split"] = "unused"
            return result
        missing_dates = sorted(result.loc[result["split"] == "", "target_date"].astype(str).unique())
        raise ValueError(f"Time-series split left samples unassigned: {missing_dates[:10]}")
    return result


def assign_splits(samples: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    samples = samples.sort_values(["topic", "target_date"]).copy()
    samples["split"] = ""

    if args.split_strategy == "time-series":
        fold_splits, fold_metadata = build_time_series_cv_splits(samples, args)
        result = apply_time_series_split(samples, fold_splits[-1]).reset_index(drop=True)
        result.attrs["time_series_split_metadata"] = fold_metadata
        return result

    if args.split_strategy == "per-topic":
        groups = samples.groupby("topic", sort=False).groups.values()
    elif args.split_strategy == "global":
        groups = [samples.sort_values("target_date").index]
    else:
        raise ValueError(f"Unsupported split strategy: {args.split_strategy}")

    for index_values in groups:
        idx = list(index_values)
        idx = list(samples.loc[idx].sort_values("target_date").index)
        n_train, n_val, _ = split_sizes(len(idx), args.train_ratio, args.val_ratio)
        train_idx = idx[:n_train]
        val_idx = idx[n_train : n_train + n_val]
        test_idx = idx[n_train + n_val :]
        samples.loc[train_idx, "split"] = "train"
        samples.loc[val_idx, "split"] = "val"
        samples.loc[test_idx, "split"] = "test"

    return samples.reset_index(drop=True)


def build_samples(news: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for topic, group in news.sort_values(["topic", "date"]).groupby("topic", sort=False):
        group = group.reset_index(drop=True)
        for target_idx in range(1, len(group)):
            target = group.iloc[target_idx]
            hist_start = max(0, target_idx - args.win)
            history = group.iloc[hist_start:target_idx]
            history_ids = history["daily_id"].tolist()
            history_dates = history["date_iso"].tolist()
            sample_id = stable_hash(
                topic,
                target["date_yyyymmdd"],
                "|".join(history_ids),
                f"win={args.win}",
                length=20,
            )
            rows.append(
                {
                    "sample_id": sample_id,
                    "topic": topic,
                    "hs_code_4d": target["hs_code_4d"],
                    "hs_short_label": target["hs_short_label"],
                    "target_date": target["date_iso"],
                    "target_date_yyyymmdd": target["date_yyyymmdd"],
                    "target_article_score": float(target["article_score"]),
                    "target_daily_id": target["daily_id"],
                    "topic_order": target_idx + 1,
                    "history_len": len(history),
                    "history_start_date": history_dates[0],
                    "last_history_date": history_dates[-1],
                    "history_daily_ids": ";".join(history_ids),
                    "history_dates": ";".join(history_dates),
                }
            )
    samples = pd.DataFrame(rows)
    if samples.empty:
        raise ValueError("No samples were built. Each topic needs at least two effective news dates.")
    return assign_splits(samples, args)


def write_sample_metadata(samples: pd.DataFrame, args: argparse.Namespace) -> None:
    ensure_dir(args.output_dir)
    save_csv(samples, args.output_dir / "samples_metadata.csv")
    time_series_metadata = samples.attrs.get("time_series_split_metadata")
    if time_series_metadata is not None:
        save_csv(time_series_metadata, args.output_dir / "time_series_split_metadata.csv")
    summary = {
        "num_samples": int(len(samples)),
        "win": args.win,
        "split_strategy": args.split_strategy,
        "split_counts": samples["split"].value_counts().sort_index().to_dict(),
        "topic_counts": samples["topic"].value_counts().sort_index().to_dict(),
        "history_len_counts": samples["history_len"].value_counts().sort_index().to_dict(),
    }
    if time_series_metadata is not None:
        summary["time_series_cv"] = {
            "tscv_n_splits": int(getattr(args, "tscv_n_splits", len(time_series_metadata))),
            "tscv_test_size": getattr(args, "tscv_test_size", None),
            "tscv_gap": int(getattr(args, "tscv_gap", 0)),
            "final_test_ratio": float(getattr(args, "final_test_ratio", 0.15)),
            "final_fold": time_series_metadata.iloc[-1].to_dict(),
        }
    with (args.output_dir / "samples_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


def write_daily_compressions_real(
    compressions: pd.DataFrame,
    samples: pd.DataFrame,
    args: argparse.Namespace,
) -> pd.DataFrame:
    target_lookup = samples[["sample_id", "target_daily_id", "target_date"]].copy()
    duplicated_targets = target_lookup["target_daily_id"].duplicated()
    if duplicated_targets.any():
        duplicate_ids = target_lookup.loc[duplicated_targets, "target_daily_id"].tolist()
        raise ValueError(f"Duplicate target_daily_id values in samples: {duplicate_ids[:10]}")

    result = compressions.merge(
        target_lookup,
        left_on="daily_id",
        right_on="target_daily_id",
        how="left",
    ).drop(columns=["target_daily_id"])

    original_columns = list(compressions.columns)
    ordered_columns = ["daily_id", "sample_id", "target_date"] + [
        col for col in original_columns if col != "daily_id"
    ]
    result = result[ordered_columns]
    save_csv(result, args.output_dir / "daily_compressions_real.csv")
    return result

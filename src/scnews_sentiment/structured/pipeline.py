from __future__ import annotations

import argparse
import copy
import json
import random
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from src.scnews_sentiment.structured.common import *
from src.scnews_sentiment.structured.data import *
from src.scnews_sentiment.structured.models import *
from src.scnews_sentiment.structured.training_utils import *


def _build_prospective_semantic_sequence(
    alignment_batch: AlignmentBatch,
    *,
    time_count: int,
    node_count: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Scatter target-aligned prospective embeddings to a dense time-node tensor."""
    embeddings = np.asarray(alignment_batch.semantic_embeddings, dtype=np.float32)
    if embeddings.ndim != 2 or embeddings.shape[0] != len(alignment_batch.time_index):
        raise ValueError("Prospective embeddings and pair indices have incompatible shapes.")

    sequence = np.zeros(
        (time_count, node_count, embeddings.shape[1]),
        dtype=np.float32,
    )
    counts = np.zeros((time_count, node_count, 1), dtype=np.float32)
    for embedding, time_index, node_index in zip(
        embeddings,
        alignment_batch.time_index,
        alignment_batch.node_index,
    ):
        time_index = int(time_index)
        node_index = int(node_index)
        if not (0 <= time_index < time_count and 0 <= node_index < node_count):
            raise ValueError("Prospective pair index falls outside the time-node grid.")
        sequence[time_index, node_index] += embedding
        counts[time_index, node_index, 0] += 1.0

    sequence = np.divide(
        sequence,
        np.maximum(counts, 1.0),
        out=np.zeros_like(sequence),
        where=counts > 0,
    )
    return sequence, (counts > 0).astype(np.float32)


def run_graph_propagation(
    batch: RepresentationBatch,
    args: argparse.Namespace,
) -> GraphPropagationBatch:
    """运行非训练模式的图传播流程。

    Parameters
    ----------
    batch : RepresentationBatch
        新闻初始化阶段的输出。
    args : argparse.Namespace
        命令行参数对象，包含图路径、Topic-HS4 映射和传播模块超参数。

    Returns
    -------
    GraphPropagationBatch
        包含图、节点风险序列、初始状态和传播状态的结果容器。

    Raises
    ------
    ValueError
        当图加载或 Topic-HS4 节点对齐失败时抛出。
    RuntimeError
        当 PyTorch 前向传播失败时抛出。

    Notes
    -----
    该函数用于演示和导出中间传播状态，不执行监督训练。
    """
    # 计算并保存 graph，供后续逻辑复用。
    graph = load_industry_graph(args)
    # 计算并保存 topic_hs_map，供后续逻辑复用。
    topic_hs_map = parse_topic_hs_map(args.topic_hs_map)
    # 计算并保存 time_metadata, node_risk_sequence, node_risk_mask，供后续逻辑复用。
    time_metadata, node_risk_sequence, node_risk_mask = build_node_risk_sequence(
        batch=batch,
        graph=graph,
        topic_hs_map=topic_hs_map,
    )

    # 计算并保存 device，供后续逻辑复用。
    device = resolve_device(args.train_device)
    # 计算并保存 module，供后续逻辑复用。
    module = IndustryGraphPropagationModule(
        node_igpc=graph.node_metadata["igpc_value"].to_numpy(dtype=np.float32),
        normalized_adjacency=graph.normalized_adjacency,
        risk_dim=node_risk_sequence.shape[2],
        state_dim=args.graph_state_dim,
        risk_hidden_sizes=parse_hidden_sizes(args.graph_risk_hidden_sizes),
        recovery_hidden_sizes=parse_hidden_sizes(args.graph_recovery_hidden_sizes),
        dropout=args.graph_dropout,
        dt=args.propagation_dt,
        alpha_init=args.alpha_init,
        beta_init=args.beta_init,
        gamma_init=args.gamma_init,
        eta_init=args.eta_init,
        eps=args.recovery_eps,
        diffusion_mode=args.diffusion_mode,
        initial_state_mode=args.initial_state_mode,
        enable_graph_diffusion=args.enable_graph_diffusion,
        enable_two_hop_diffusion=args.enable_two_hop_diffusion,
        enable_recovery=args.enable_recovery,
    ).to(device)
    # 调用 module.load_checkpoint，执行当前步骤的副作用。
    module.load_checkpoint(args.graph_checkpoint, device)
    # 调用 module.eval，执行当前步骤的副作用。
    module.eval()

    # 计算并保存 initial_node_states, propagated_node_states，供后续逻辑复用。
    initial_node_states, propagated_node_states = module.run_numpy(
        node_risk_sequence=node_risk_sequence,
        node_risk_mask=node_risk_mask,
        device=device,
    )

    # 返回当前函数的处理结果。
    return GraphPropagationBatch(
        graph=graph,
        time_metadata=time_metadata,
        node_risk_sequence=node_risk_sequence,
        node_risk_mask=node_risk_mask,
        initial_node_states=initial_node_states,
        propagated_node_states=propagated_node_states,
        final_node_states=propagated_node_states[-1],
    )


def save_representations(batch: RepresentationBatch, args: argparse.Namespace) -> None:
    """保存新闻初始化阶段的中间表示。

    Parameters
    ----------
    batch : RepresentationBatch
        待保存的初始化结果。
    args : argparse.Namespace
        参数对象，需包含 ``output_dir`` 和模型配置。

    Returns
    -------
    None
        文件写入完成后不返回值。

    Raises
    ------
    OSError
        当输出目录无法创建或文件无法写入时抛出。

    Notes
    -----
    输出包括新闻元数据、日级元数据、NumPy 压缩矩阵和 JSON 配置。
    """
    # 调用 ensure_dir，执行当前步骤的副作用。
    ensure_dir(args.output_dir)
    # 调用 batch.news_metadata.to_csv，执行当前步骤的副作用。
    batch.news_metadata.to_csv(
        args.output_dir / "news_metadata.csv", index=False, encoding="utf-8-sig"
    )
    # 调用 batch.daily_metadata.to_csv，执行当前步骤的副作用。
    batch.daily_metadata.to_csv(
        args.output_dir / "daily_metadata.csv", index=False, encoding="utf-8-sig"
    )
    # 调用 np.savez_compressed，执行当前步骤的副作用。
    np.savez_compressed(
        args.output_dir / "structured_representations.npz",
        gdelt_features=batch.gdelt_features,
        text_embeddings=batch.text_embeddings,
        news_representations=batch.news_representations,
        daily_representations=batch.daily_representations,
    )

    # 计算并保存 config，供后续逻辑复用。
    config = {
        "news_csv": str(args.news_csv),
        "encoder_model": str(args.encoder_model),
        "gdelt_feature_columns": GDELT_FEATURE_COLUMNS,
        "gdelt_standardizer": batch.gdelt_standardizer.to_dict(),
        "daily_group_cols": parse_group_cols(args.daily_group_cols),
        "projection_dim": args.projection_dim,
        "sent_hidden_sizes": list(parse_hidden_sizes(args.sent_hidden_sizes)),
        "representation_dim": args.representation_dim,
        "pooling": args.pooling,
        "normalize_text_embeddings": args.normalize_text_embeddings,
    }
    # 使用上下文管理器处理资源，确保结束时自动释放。
    with (args.output_dir / "structured_config.json").open("w", encoding="utf-8") as handle:
        # 执行 dump，完成当前步骤的副作用。
        json.dump(config, handle, indent=2, ensure_ascii=False)


def save_graph_propagation(graph_batch: GraphPropagationBatch, args: argparse.Namespace) -> None:
    """保存图传播阶段的中间状态。

    Parameters
    ----------
    graph_batch : GraphPropagationBatch
        图传播结果容器。
    args : argparse.Namespace
        参数对象，需包含 ``output_dir``。

    Returns
    -------
    None
        文件写入完成后不返回值。

    Raises
    ------
    OSError
        当输出目录无法创建或文件无法写入时抛出。
    """
    # 调用 ensure_dir，执行当前步骤的副作用。
    ensure_dir(args.output_dir)
    # 调用 graph_batch.graph.node_metadata.to_csv，执行当前步骤的副作用。
    graph_batch.graph.node_metadata.to_csv(
        args.output_dir / "graph_node_metadata.csv",
        index=False,
        encoding="utf-8-sig",
    )
    # 调用 graph_batch.time_metadata.to_csv，执行当前步骤的副作用。
    graph_batch.time_metadata.to_csv(
        args.output_dir / "graph_time_metadata.csv",
        index=False,
        encoding="utf-8-sig",
    )
    # 调用 np.savez_compressed，执行当前步骤的副作用。
    np.savez_compressed(
        args.output_dir / "graph_propagation_states.npz",
        normalized_adjacency=graph_batch.graph.normalized_adjacency,
        node_risk_sequence=graph_batch.node_risk_sequence,
        node_risk_mask=graph_batch.node_risk_mask,
        initial_node_states=graph_batch.initial_node_states,
        propagated_node_states=graph_batch.propagated_node_states,
        final_node_states=graph_batch.final_node_states,
    )


def _parse_repeat_seeds(value: Any) -> List[int]:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        raw_items = value
    else:
        text = str(value).strip()
        if not text:
            return []
        raw_items = text.split(",")

    seeds: List[int] = []
    for item in raw_items:
        item_text = str(item).strip()
        if not item_text:
            continue
        try:
            seeds.append(int(item_text))
        except ValueError as exc:
            raise ValueError("repeat_seeds must contain comma-separated integers.") from exc
    if not seeds:
        raise ValueError("repeat_seeds must contain at least one integer when set.")
    return seeds


def _summarize_repeated_cv_metrics(repeat_metrics: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    if repeat_metrics.empty:
        return pd.DataFrame(
            columns=["split", "metric", "count", "mean", "std", "min", "max", "mean_std"]
        )

    for (split_name, metric_name), group in repeat_metrics.groupby(["split", "metric"]):
        values = pd.to_numeric(group["value"], errors="coerce").dropna().to_numpy(
            dtype=np.float64
        )
        if len(values) == 0:
            rows.append(
                {
                    "split": split_name,
                    "metric": metric_name,
                    "count": 0,
                    "mean": None,
                    "std": None,
                    "min": None,
                    "max": None,
                    "mean_std": None,
                }
            )
            continue

        mean_value = float(np.mean(values))
        std_value = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
        rows.append(
            {
                "split": split_name,
                "metric": metric_name,
                "count": int(len(values)),
                "mean": mean_value,
                "std": std_value,
                "min": float(np.min(values)),
                "max": float(np.max(values)),
                "mean_std": f"{mean_value:.6f} +/- {std_value:.6f}",
            }
        )
    return pd.DataFrame(rows)


def _summarize_repeated_cv_fold_metrics(fold_summary: pd.DataFrame) -> pd.DataFrame:
    metric_names = [
        "mse",
        "rmse",
        "mae",
        "r2",
        "pearson",
        "spearman",
        "median_ae",
        "p90_ae",
        "max_ae",
        "bias",
    ]
    rows: List[Dict[str, Any]] = []
    if fold_summary.empty:
        return pd.DataFrame(
            columns=["split", "metric", "count", "mean", "std", "min", "max", "mean_std"]
        )

    metric_name_set = set(metric_names)
    for column in fold_summary.columns:
        if "_" not in str(column):
            continue
        split_name, metric_name = str(column).split("_", 1)
        if metric_name not in metric_name_set:
            continue

        values = pd.to_numeric(fold_summary[column], errors="coerce").dropna().to_numpy(
            dtype=np.float64
        )
        if len(values) == 0:
            rows.append(
                {
                    "split": split_name,
                    "metric": metric_name,
                    "count": 0,
                    "mean": None,
                    "std": None,
                    "min": None,
                    "max": None,
                    "mean_std": None,
                }
            )
            continue

        mean_value = float(np.mean(values))
        std_value = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
        rows.append(
            {
                "split": split_name,
                "metric": metric_name,
                "count": int(len(values)),
                "mean": mean_value,
                "std": std_value,
                "min": float(np.min(values)),
                "max": float(np.max(values)),
                "mean_std": f"{mean_value:.6f} +/- {std_value:.6f}",
            }
        )
    return pd.DataFrame(rows)


def train_supervised_structured_model_repeated_time_series(args: argparse.Namespace) -> None:
    repeat_seeds = _parse_repeat_seeds(getattr(args, "repeat_seeds", ""))
    if not repeat_seeds:
        train_supervised_structured_model_time_series(args)
        return

    ensure_dir(args.output_dir)
    repeat_metric_rows: List[Dict[str, Any]] = []
    fold_summary_frames: List[pd.DataFrame] = []
    repeat_payloads: List[Dict[str, Any]] = []

    for repeat_index, seed in enumerate(repeat_seeds):
        repeat_dir = args.output_dir / f"repeat_{repeat_index:02d}"
        repeat_args = copy.copy(args)
        repeat_args.output_dir = repeat_dir
        repeat_args.seed = int(seed)
        if args.tensorboard_log_dir is not None:
            repeat_args.tensorboard_log_dir = (
                args.tensorboard_log_dir / f"repeat_{repeat_index:02d}"
            )

        print(
            f"[repeated-tscv] repeat={repeat_index} seed={seed} "
            f"folds={args.tscv_n_splits}"
        )
        train_supervised_structured_model_time_series(repeat_args)

        aggregate_path = repeat_dir / "time_series_cv_aggregate_metrics.csv"
        aggregate_frame = pd.read_csv(aggregate_path, encoding="utf-8-sig")
        for _, row in aggregate_frame.iterrows():
            repeat_metric_rows.append(
                {
                    "repeat": int(repeat_index),
                    "seed": int(seed),
                    "split": row.get("split"),
                    "metric": row.get("metric"),
                    "value": row.get("mean"),
                    "fold_count": row.get("count"),
                    "fold_std": row.get("std"),
                    "fold_min": row.get("min"),
                    "fold_max": row.get("max"),
                    "repeat_output_dir": str(repeat_dir),
                }
            )

        fold_summary_path = repeat_dir / "time_series_cv_summary.csv"
        fold_summary = pd.read_csv(fold_summary_path, encoding="utf-8-sig")
        fold_summary.insert(0, "seed", int(seed))
        fold_summary.insert(0, "repeat", int(repeat_index))
        fold_summary_frames.append(fold_summary)

        summary_json_path = repeat_dir / "time_series_cv_summary.json"
        with summary_json_path.open("r", encoding="utf-8") as handle:
            repeat_payload = json.load(handle)
        repeat_payloads.append(
            {
                "repeat": int(repeat_index),
                "seed": int(seed),
                "output_dir": str(repeat_dir),
                "aggregate_metrics": repeat_payload.get("aggregate_metrics", {}),
            }
        )

    repeat_metrics = pd.DataFrame(repeat_metric_rows)
    repeat_metrics.to_csv(
        args.output_dir / "repeated_time_series_cv_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    repeated_aggregate = _summarize_repeated_cv_metrics(repeat_metrics)
    repeated_aggregate.to_csv(
        args.output_dir / "repeated_time_series_cv_aggregate_metrics.csv",
        index=False,
        encoding="utf-8-sig",
    )

    all_fold_summary = pd.DataFrame()
    if fold_summary_frames:
        all_fold_summary = pd.concat(fold_summary_frames, ignore_index=True)
        all_fold_summary.to_csv(
            args.output_dir / "repeated_time_series_cv_fold_summary.csv",
            index=False,
            encoding="utf-8-sig",
        )

    repeated_fold_aggregate = _summarize_repeated_cv_fold_metrics(all_fold_summary)
    repeated_fold_aggregate.to_csv(
        args.output_dir / "repeated_time_series_cv_25fold_aggregate_metrics.csv",
        index=False,
        encoding="utf-8-sig",
    )

    summary_payload = {
        "split_strategy": "repeated-time-series",
        "repeat_seeds": [int(seed) for seed in repeat_seeds],
        "repeat_count": int(len(repeat_seeds)),
        "tscv_n_splits": args.tscv_n_splits,
        "tscv_test_size": None if args.tscv_test_size is None else int(args.tscv_test_size),
        "tscv_gap": args.tscv_gap,
        "final_test_ratio": args.final_test_ratio,
        "propagation_model": args.propagation_model,
        "component_scale_mode": args.component_scale_mode,
        "component_scale_init": args.component_scale_init,
        "rd_fusion_gate_init": args.rd_fusion_gate_init,
        "initial_state_mode": args.initial_state_mode,
        "final_aggregation": (
            "final-fold-only" if bool(getattr(args, "evaluate_test", True))
            else "validation-only"
        ),
        "aggregate_metrics": json.loads(
            repeated_aggregate.to_json(orient="records", double_precision=15)
        ),
        "fold_aggregate_metrics": json.loads(
            repeated_fold_aggregate.to_json(orient="records", double_precision=15)
        ),
        "repeats": repeat_payloads,
    }
    with (args.output_dir / "repeated_time_series_cv_summary.json").open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(summary_payload, handle, indent=2, ensure_ascii=False)

    print(f"Saved repeated TimeSeriesSplit outputs to {args.output_dir}")


def train_supervised_structured_model_time_series(args: argparse.Namespace) -> None:
    """Run expanding-window TimeSeriesSplit CV with a fixed final holdout test block."""
    # 导入当前函数需要的模块，避免非相关流程提前加载。
    import copy

    # 执行 ensure_dir，完成当前步骤的副作用。
    ensure_dir(args.output_dir)
    # 计算并保存 supervision_weights，供后续逻辑复用。
    supervision_weights = build_supervision_weights_for_splits(args)
    # 计算并保存 fold_masks、fold_metadata，供后续逻辑复用。
    fold_masks, fold_metadata = build_time_series_cv_splits(
        supervision_weights=supervision_weights,
        n_splits=args.tscv_n_splits,
        test_size=args.tscv_test_size,
        gap=args.tscv_gap,
        final_test_ratio=args.final_test_ratio,
    )
    # 执行 to_csv，完成当前步骤的副作用。
    fold_metadata.to_csv(
        args.output_dir / "time_series_split_metadata.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # 初始化 summary_rows，明确后续步骤要累积的数据。
    summary_rows: List[Dict[str, Any]] = []
    # 遍历 fold_index, masks in enumerate(fold_masks)，逐项执行处理逻辑。
    for fold_index, masks in enumerate(fold_masks):
        # 计算并保存 fold_dir，供当前分支后续使用。
        fold_dir = args.output_dir / f"fold_{fold_index:02d}"
        # 计算并保存 fold_args，供当前分支后续使用。
        fold_args = copy.copy(args)
        # 计算并保存 output_dir，供当前分支后续使用。
        fold_args.output_dir = fold_dir
        # 计算并保存 split_strategy，供当前分支后续使用。
        fold_args.split_strategy = "fixed"
        # 计算并保存 split_override_masks，供当前分支后续使用。
        fold_args.split_override_masks = masks
        fold_args.evaluate_test = bool(getattr(args, "evaluate_test", True)) and (
            fold_index == len(fold_masks) - 1
        )
        # 判断 args.tensorboard_log_dir is not None，处理对应情况。
        if args.tensorboard_log_dir is not None:
            # 计算并保存 tensorboard_log_dir，供当前分支后续使用。
            fold_args.tensorboard_log_dir = args.tensorboard_log_dir / f"fold_{fold_index:02d}"

        # 执行 print，完成当前步骤的副作用。
        print(
            f"[tscv] fold={fold_index} "
            f"train={int(masks['train'].sum())} "
            f"val={int(masks['val'].sum())} "
            f"test={int(masks['test'].sum())} "
            f"evaluate_test={fold_args.evaluate_test}"
        )
        # 执行 train_supervised_structured_model，完成当前步骤的副作用。
        train_supervised_structured_model(fold_args)

        # 计算并保存 metrics_path，供当前分支后续使用。
        metrics_path = fold_dir / "structured_supervised_metrics.json"
        # 使用上下文管理器打开资源，结束后自动清理。
        with metrics_path.open("r", encoding="utf-8") as handle:
            # 计算并保存 metrics，供当前分支后续使用。
            metrics = json.load(handle)

        # 计算并保存 row，供当前分支后续使用。
        row = fold_metadata.iloc[fold_index].to_dict()
        # 执行 update，完成当前步骤的副作用。
        row.update(
            {
                "best_epoch": metrics.get("best_epoch"),
                "best_val_total_loss": metrics.get("best_val_total_loss"),
                "epochs_completed": metrics.get("epochs_completed"),
                "early_stopped": metrics.get("early_stopped"),
                "val_mse": metrics.get("splits", {}).get("val", {}).get("mse"),
                "val_rmse": metrics.get("splits", {}).get("val", {}).get("rmse"),
                "val_mae": metrics.get("splits", {}).get("val", {}).get("mae"),
                "val_r2": metrics.get("splits", {}).get("val", {}).get("r2"),
                "val_pearson": metrics.get("splits", {}).get("val", {}).get("pearson"),
                "val_spearman": metrics.get("splits", {}).get("val", {}).get("spearman"),
                "val_median_ae": metrics.get("splits", {}).get("val", {}).get("median_ae"),
                "val_p90_ae": metrics.get("splits", {}).get("val", {}).get("p90_ae"),
                "val_max_ae": metrics.get("splits", {}).get("val", {}).get("max_ae"),
                "val_bias": metrics.get("splits", {}).get("val", {}).get("bias"),
                "test_mse": metrics.get("splits", {}).get("test", {}).get("mse"),
                "test_rmse": metrics.get("splits", {}).get("test", {}).get("rmse"),
                "test_mae": metrics.get("splits", {}).get("test", {}).get("mae"),
                "test_r2": metrics.get("splits", {}).get("test", {}).get("r2"),
                "test_pearson": metrics.get("splits", {}).get("test", {}).get("pearson"),
                "test_spearman": metrics.get("splits", {}).get("test", {}).get("spearman"),
                "test_median_ae": metrics.get("splits", {}).get("test", {}).get("median_ae"),
                "test_p90_ae": metrics.get("splits", {}).get("test", {}).get("p90_ae"),
                "test_max_ae": metrics.get("splits", {}).get("test", {}).get("max_ae"),
                "test_bias": metrics.get("splits", {}).get("test", {}).get("bias"),
                "propagation_model": metrics.get("propagation_model"),
                "dcrnn_bidirectional": metrics.get("dcrnn_bidirectional"),
                "agcrn_embed_dim": metrics.get("agcrn_embed_dim"),
                "component_scale_mode": metrics.get("component_scale_mode"),
                "component_scale_init": metrics.get("component_scale_init"),
                "rd_fusion_gate_init": metrics.get("rd_fusion_gate_init"),
                "ts_baseline_lookback": metrics.get("ts_baseline_lookback"),
                "itransformer_heads": metrics.get("itransformer_heads"),
                "stgcn_kernel_size": metrics.get("stgcn_kernel_size"),
                "timesnet_kernel_sizes": metrics.get("timesnet_kernel_sizes"),
                "baseline_input": metrics.get("baseline_input"),
                "uses_supply_chain_graph": metrics.get("uses_supply_chain_graph"),
                "uses_adaptive_graph": metrics.get("uses_adaptive_graph"),
                "initial_state_mode": metrics.get("initial_state_mode"),
                "output_dir": str(fold_dir),
            }
        )
        # 执行 append，完成当前步骤的副作用。
        summary_rows.append(row)

    # 计算并保存 summary，供后续逻辑复用。
    summary = pd.DataFrame(summary_rows)
    if not bool(getattr(args, "evaluate_test", True)):
        summary = summary.drop(
            columns=[
                column
                for column in summary.columns
                if str(column).startswith("test_")
            ],
            errors="ignore",
        )
    # 计算并保存 aggregate_metrics，供后续逻辑复用。
    aggregate_metrics = summarize_time_series_cv_metrics(summary)
    # 计算并保存 aggregate_frame，供后续逻辑复用。
    aggregate_frame = flatten_time_series_cv_aggregate(aggregate_metrics)
    # 执行 to_csv，完成当前步骤的副作用。
    summary.to_csv(
        args.output_dir / "time_series_cv_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )
    # 执行 to_csv，完成当前步骤的副作用。
    aggregate_frame.to_csv(
        args.output_dir / "time_series_cv_aggregate_metrics.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # 计算并保存 summary_payload，供后续逻辑复用。
    summary_payload = {
        "split_strategy": "time-series",
        "tscv_n_splits": args.tscv_n_splits,
        "tscv_test_size": None if args.tscv_test_size is None else int(args.tscv_test_size),
        "tscv_gap": args.tscv_gap,
        "final_test_ratio": args.final_test_ratio,
        "propagation_model": args.propagation_model,
        "dcrnn_bidirectional": args.dcrnn_bidirectional,
        "agcrn_embed_dim": getattr(args, "agcrn_embed_dim", None),
        "component_scale_mode": args.component_scale_mode,
        "component_scale_init": args.component_scale_init,
        "rd_fusion_gate_init": args.rd_fusion_gate_init,
        "ts_baseline_lookback": getattr(args, "ts_baseline_lookback", None),
        "itransformer_heads": getattr(args, "itransformer_heads", None),
        "stgcn_kernel_size": getattr(args, "stgcn_kernel_size", None),
        "timesnet_kernel_sizes": getattr(args, "timesnet_kernel_sizes", None),
        "initial_state_mode": args.initial_state_mode,
        "uses_adaptive_graph": str(args.propagation_model).lower() == "agcrn",
        "evaluate_test": bool(getattr(args, "evaluate_test", True)),
        "aggregate_metrics": aggregate_metrics,
        "folds": json.loads(summary.to_json(orient="records")),
    }
    # 使用上下文管理器处理资源，确保结束后自动释放。
    with (args.output_dir / "time_series_cv_summary.json").open(
        "w",
        encoding="utf-8",
    ) as handle:
        # 执行 dump，完成当前步骤的副作用。
        json.dump(summary_payload, handle, indent=2, ensure_ascii=False)

    # 执行 print，完成当前步骤的副作用。
    print(f"Saved TimeSeriesSplit outputs to {args.output_dir}")


def _build_shifted_label_history_inputs(
    labels: Any,
    label_mask: Any,
) -> Tuple[Any, Any]:
    """Return causal ArticleScore-history inputs for target-time prediction."""
    import torch

    shifted_values = torch.zeros(
        labels.shape[0],
        labels.shape[1],
        1,
        dtype=labels.dtype,
        device=labels.device,
    )
    shifted_mask = torch.zeros_like(shifted_values)
    if labels.shape[0] > 1:
        shifted_values[1:, :, 0] = labels[:-1] * label_mask[:-1]
        shifted_mask[1:, :, 0] = (label_mask[:-1] > 0).to(labels.dtype)
    node_risk_sequence = torch.cat([shifted_values, shifted_mask], dim=2)
    node_risk_mask = torch.ones_like(shifted_mask)
    return node_risk_sequence, node_risk_mask


def _build_naive_label_history_predictions(
    labels: np.ndarray,
    label_mask: np.ndarray,
    mode: str,
) -> np.ndarray:
    """Build causal persistence or expanding historical-mean predictions."""
    labels = np.asarray(labels, dtype=np.float32)
    label_mask = np.asarray(label_mask, dtype=np.float32)
    predictions = np.zeros_like(labels, dtype=np.float32)
    last_values = np.zeros(labels.shape[1], dtype=np.float32)
    running_sum = np.zeros(labels.shape[1], dtype=np.float64)
    running_count = np.zeros(labels.shape[1], dtype=np.int64)

    for time_index in range(labels.shape[0]):
        if mode == "persistence":
            predictions[time_index] = last_values
        elif mode == "historical_mean":
            predictions[time_index] = np.divide(
                running_sum,
                running_count,
                out=np.zeros_like(running_sum),
                where=running_count > 0,
            ).astype(np.float32)
        else:
            raise ValueError(f"Unsupported naive label-history model: {mode}.")

        observed = label_mask[time_index] > 0
        last_values[observed] = labels[time_index, observed]
        running_sum[observed] += labels[time_index, observed]
        running_count[observed] += 1

    return predictions


def run_naive_label_history_baseline(args: argparse.Namespace) -> None:
    """Evaluate a deterministic causal label-history baseline."""
    mode = str(args.propagation_model).lower()
    data = build_supervised_label_history_training_data(args)
    ensure_dir(args.output_dir)
    predictions = _build_naive_label_history_predictions(
        data.label_values,
        data.label_mask,
        mode,
    )
    metrics: Dict[str, Any] = {
        "best_epoch": 0,
        "checkpoint_metric": None,
        "best_checkpoint_metric_value": None,
        "best_val_total_loss": None,
        "best_val_supervised_loss": None,
        "epochs_completed": 0,
        "early_stopped": False,
        "propagation_model": mode,
        "baseline_input": "strictly_shifted_article_score_history",
        "uses_supply_chain_graph": False,
        "uses_adaptive_graph": False,
        "evaluate_test": bool(getattr(args, "evaluate_test", True)),
        "splits": {},
    }
    for split_name, split_mask in data.split_by_time.items():
        if split_name == "test" and not bool(getattr(args, "evaluate_test", True)):
            continue
        metrics["splits"][split_name] = prediction_metrics(
            predictions=predictions,
            labels=data.label_values,
            supervision_weights=data.supervision_weights,
            time_mask=split_mask,
        )

    predictions_frame = build_training_predictions_frame(predictions, data)
    if not bool(getattr(args, "evaluate_test", True)):
        predictions_frame = predictions_frame[predictions_frame["split"] != "test"].copy()
    predictions_frame.to_csv(
        args.output_dir / "structured_supervised_predictions.csv",
        index=False,
        encoding="utf-8-sig",
    )
    data.time_metadata.to_csv(
        args.output_dir / "structured_supervised_time_metadata.csv",
        index=False,
        encoding="utf-8-sig",
    )
    with (args.output_dir / "structured_supervised_metrics.json").open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(metrics, handle, indent=2, ensure_ascii=False)


def train_label_history_baseline_model(args: argparse.Namespace) -> None:
    """Train iTransformer/TimesNet baselines on shifted ArticleScore history."""
    import copy
    import torch

    from src.scnews_sentiment.structured.propagation_models import (
        build_propagation_module,
        is_label_history_propagation_model,
        is_prospective_semantic_model,
    )

    if args.align_weight > 0:
        raise ValueError(
            "Label-history baselines must set align_weight=0.0 to avoid LLM alignment."
        )

    data = build_supervised_label_history_training_data(args)
    device = resolve_device(args.train_device)
    ensure_dir(args.output_dir)
    summary_writer = create_tensorboard_writer(args)

    labels = torch.tensor(data.label_values, dtype=torch.float32, device=device)
    label_mask = torch.tensor(data.label_mask, dtype=torch.float32, device=device)
    supervision_weights = torch.tensor(
        data.supervision_weights,
        dtype=torch.float32,
        device=device,
    )
    split_masks = {
        split: torch.tensor(mask, dtype=torch.bool, device=device)
        for split, mask in data.split_by_time.items()
    }
    node_risk_sequence, node_risk_mask = _build_shifted_label_history_inputs(
        labels,
        label_mask,
    )

    graph_module = build_propagation_module(args, data).to(device)
    graph_module.load_checkpoint(args.graph_checkpoint, device)
    evaluator = RiskEvaluationScorer(
        state_dim=args.graph_state_dim,
        hidden_sizes=parse_hidden_sizes(args.eval_hidden_sizes),
        dropout=args.eval_dropout,
    ).to(device)

    parameters = list(graph_module.module.parameters())
    parameters.extend(evaluator.module.parameters())
    optimizer = torch.optim.AdamW(
        parameters,
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    adjacency = graph_module.module.normalized_adjacency
    best_val = float("inf")
    best_state: Optional[Dict[str, Any]] = None
    history: List[Dict[str, Any]] = []
    epochs_without_improvement = 0
    early_stopped = False
    stop_epoch: Optional[int] = None

    def forward_predictions() -> Tuple[Any, Any]:
        propagated_states = graph_module.module(node_risk_sequence, node_risk_mask)
        predictions = evaluator.module(propagated_states).squeeze(-1)
        return predictions, propagated_states

    def loss_components(
        predictions: Any,
        propagated_states: Any,
        split_name: str,
    ) -> Tuple[Any, Any, Any, Any]:
        return compute_structured_total_loss(
            predictions=predictions,
            propagated_states=propagated_states,
            labels=labels,
            supervision_weights=supervision_weights,
            split_mask=split_masks[split_name],
            adjacency=adjacency,
            smooth_weight=args.smooth_weight,
            align_weight=0.0,
            align_temperature=args.align_temperature,
        )

    def loss_to_float(value: Any) -> float:
        return float(value.detach().cpu())

    for epoch in range(1, args.epochs + 1):
        graph_module.module.train()
        evaluator.module.train()
        optimizer.zero_grad(set_to_none=True)
        predictions, propagated_states = forward_predictions()
        total_loss, sup_loss, smooth_loss, align_loss = loss_components(
            predictions,
            propagated_states,
            "train",
        )
        total_loss.backward()
        if args.grad_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(parameters, args.grad_clip_norm)
        optimizer.step()

        graph_module.module.eval()
        evaluator.module.eval()
        with torch.no_grad():
            eval_predictions, eval_states = forward_predictions()
            train_total_t, train_sup_t, train_smooth_t, train_align_t = loss_components(
                eval_predictions,
                eval_states,
                "train",
            )
            val_total_t, val_sup_t, val_smooth_t, val_align_t = loss_components(
                eval_predictions,
                eval_states,
                "val",
            )
            train_total = loss_to_float(train_total_t)
            train_sup = loss_to_float(train_sup_t)
            train_smooth = loss_to_float(train_smooth_t)
            train_align = loss_to_float(train_align_t)
            val_total = loss_to_float(val_total_t)
            val_sup = loss_to_float(val_sup_t)
            val_smooth = loss_to_float(val_smooth_t)
            val_align = loss_to_float(val_align_t)

        checkpoint_metric_value = (
            val_sup if args.checkpoint_metric == "val_supervised" else val_total
        )
        improved = checkpoint_metric_value < best_val - args.early_stopping_min_delta
        if improved:
            best_val = checkpoint_metric_value
            best_state = {
                "graph_module": copy.deepcopy(graph_module.module.state_dict()),
                "evaluator": copy.deepcopy(evaluator.module.state_dict()),
                "epoch": epoch,
                "checkpoint_metric": args.checkpoint_metric,
                "checkpoint_metric_value": checkpoint_metric_value,
                "val_total_loss": val_total,
                "val_supervised_loss": val_sup,
                "val_smooth_loss": val_smooth,
                "val_alignment_loss": val_align,
            }
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        record = {
            "epoch": epoch,
            "train_total_loss": train_total,
            "train_supervised_loss": train_sup,
            "train_smooth_loss": train_smooth,
            "train_alignment_loss": train_align,
            "val_total_loss": val_total,
            "val_supervised_loss": val_sup,
            "val_smooth_loss": val_smooth,
            "val_alignment_loss": val_align,
            "checkpoint_metric": args.checkpoint_metric,
            "checkpoint_metric_value": checkpoint_metric_value,
            "is_best": improved,
            "epochs_without_improvement": epochs_without_improvement,
        }
        history.append(record)
        write_epoch_tensorboard(summary_writer, record, learning_rate=args.lr)

        wait_status = (
            "disabled"
            if args.early_stopping_patience == 0
            else f"{epochs_without_improvement}/{args.early_stopping_patience}"
        )
        if epoch == 1 or epoch % args.log_every == 0 or epoch == args.epochs:
            print(
                f"[train] epoch={epoch} "
                f"train={train_total:.6f} val={val_total:.6f} "
                f"sup={train_sup:.6f} smooth={train_smooth:.6f} "
                f"best_{args.checkpoint_metric}={best_val:.6f} "
                f"wait={wait_status}"
            )

        if (
            args.early_stopping_patience > 0
            and epochs_without_improvement >= args.early_stopping_patience
        ):
            early_stopped = True
            stop_epoch = epoch
            print(
                f"[early-stop] epoch={epoch} "
                f"best_epoch={best_state['epoch'] if best_state is not None else 'none'} "
                f"best_{args.checkpoint_metric}={best_val:.6f}"
            )
            break

    if best_state is not None:
        graph_module.module.load_state_dict(best_state["graph_module"])
        evaluator.module.load_state_dict(best_state["evaluator"])

    graph_module.module.eval()
    evaluator.module.eval()
    with torch.no_grad():
        final_predictions_tensor, _ = forward_predictions()
    final_predictions = final_predictions_tensor.detach().cpu().numpy().astype(np.float32)

    metrics = {
        "best_epoch": None if best_state is None else int(best_state["epoch"]),
        "checkpoint_metric": args.checkpoint_metric,
        "best_checkpoint_metric_value": best_val,
        "best_val_total_loss": (
            None if best_state is None else float(best_state["val_total_loss"])
        ),
        "best_val_supervised_loss": (
            None if best_state is None else float(best_state["val_supervised_loss"])
        ),
        "best_val_smooth_loss": (
            None if best_state is None else float(best_state["val_smooth_loss"])
        ),
        "best_val_alignment_loss": (
            None if best_state is None else float(best_state["val_alignment_loss"])
        ),
        "epochs_completed": len(history),
        "early_stopped": early_stopped,
        "stop_epoch": stop_epoch,
        "early_stopping_patience": args.early_stopping_patience,
        "early_stopping_min_delta": args.early_stopping_min_delta,
        "smooth_weight": args.smooth_weight,
        "align_weight": 0.0,
        "align_temperature": args.align_temperature,
        "align_dim": args.align_dim,
        "align_hidden_sizes": list(parse_hidden_sizes(args.align_hidden_sizes)),
        "align_dropout": args.align_dropout,
        "llm_forecast_csv": str(args.llm_forecast_csv),
        "llm_embedding_npz": str(args.llm_embedding_npz),
        "alignment_pair_count": 0,
        "alignment_skipped_unmatched_rows": 0,
        "alignment_topic_hs_remapped_rows": 0,
        "upstream_loss_weight": args.upstream_loss_weight,
        "propagation_model": args.propagation_model,
        "gat_heads": args.gat_heads,
        "dcrnn_diffusion_steps": args.dcrnn_diffusion_steps,
        "dcrnn_bidirectional": args.dcrnn_bidirectional,
        "agcrn_embed_dim": args.agcrn_embed_dim,
        "component_scale_mode": args.component_scale_mode,
        "component_scale_init": args.component_scale_init,
        "rd_fusion_gate_init": args.rd_fusion_gate_init,
        "ts_baseline_lookback": args.ts_baseline_lookback,
        "itransformer_heads": args.itransformer_heads,
        "stgcn_kernel_size": args.stgcn_kernel_size,
        "timesnet_kernel_sizes": args.timesnet_kernel_sizes,
        "baseline_input": "shifted_article_score_history",
        "uses_supply_chain_graph": args.propagation_model == "stgcn",
        "uses_adaptive_graph": False,
        "graph_direction": data.graph.graph_direction,
        "diffusion_mode": args.diffusion_mode,
        "initial_state_mode": args.initial_state_mode,
        "enable_graph_diffusion": args.enable_graph_diffusion,
        "enable_two_hop_diffusion": args.enable_two_hop_diffusion,
        "enable_recovery": args.enable_recovery,
        "time_count": int(len(data.time_metadata)),
        "node_count": int(len(data.graph.node_codes)),
        "input_news_count": int(len(data.news_metadata)),
        "supervised_time_count": int((data.supervision_weights.sum(axis=1) > 0).sum()),
        "evaluate_test": bool(getattr(args, "evaluate_test", True)),
        "splits": {},
        "history": history,
    }
    for split_name, split_mask in data.split_by_time.items():
        if split_name == "test" and not bool(getattr(args, "evaluate_test", True)):
            continue
        metrics["splits"][split_name] = prediction_metrics(
            predictions=final_predictions,
            labels=data.label_values,
            supervision_weights=data.supervision_weights,
            time_mask=split_mask,
        )
    write_final_metrics_tensorboard(summary_writer, metrics)

    predictions_frame = build_training_predictions_frame(final_predictions, data)
    if not bool(getattr(args, "evaluate_test", True)):
        predictions_frame = predictions_frame[predictions_frame["split"] != "test"].copy()
    predictions_frame.to_csv(
        args.output_dir / "structured_supervised_predictions.csv",
        index=False,
        encoding="utf-8-sig",
    )
    data.time_metadata.to_csv(
        args.output_dir / "structured_supervised_time_metadata.csv",
        index=False,
        encoding="utf-8-sig",
    )
    data.graph.node_metadata.to_csv(
        args.output_dir / "structured_supervised_node_metadata.csv",
        index=False,
        encoding="utf-8-sig",
    )
    with (args.output_dir / "structured_supervised_metrics.json").open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(metrics, handle, indent=2, ensure_ascii=False)

    torch.save(
        {
            "news_encoder_state_dict": None,
            "graph_module_state_dict": graph_module.module.state_dict(),
            "evaluator_state_dict": evaluator.module.state_dict(),
            "semantic_projector_state_dict": None,
            "struct_projector_state_dict": None,
            "gdelt_standardizer": data.gdelt_standardizer.to_dict(),
            "topic_hs_map": data.topic_hs_map,
            "graph_node_codes": data.graph.node_codes,
            "args": vars(args),
            "metrics": metrics,
        },
        args.output_dir / "structured_supervised_model.pt",
    )
    if summary_writer is not None:
        summary_writer.flush()
        summary_writer.close()
    print(f"Saved supervised training outputs to {args.output_dir}")


def train_supervised_structured_model(args: argparse.Namespace) -> None:
    """训练结构化风险传播模型。

    Parameters
    ----------
    args : argparse.Namespace
        命令行参数对象，包含输入文件、图文件、模型维度、训练超参数和输出路径。

    Returns
    -------
    None
        训练完成后将模型、指标和预测明细写入 ``args.output_dir``。

    Raises
    ------
    ValueError
        当数据加载、图构造、标签构造或时间切分不满足约束时抛出。
    RuntimeError
        当 PyTorch 前向、反向传播或权重加载失败时抛出。
    OSError
        当输出文件无法写入时抛出。

    Notes
    -----
    训练流程先用冻结文本编码器提取文本向量，再训练新闻初始化编码器、图传播
    模块和风险打分器。监督目标来自下一有效日期的 ``ArticleScore``，同时
    可加入图平滑正则项。训练过程中按验证集总损失保存最优权重，并可在
    验证集长期无改善时提前停止。
    """
    # 延迟导入 copy，避免未使用阶段强制加载依赖。
    import copy
    import torch
    # 延迟导入 torch，避免未使用阶段强制加载依赖。
    import torch
    from src.scnews_sentiment.structured.propagation_models import (
        build_propagation_module,
        is_label_history_propagation_model,
        is_prospective_semantic_model,
    )

    # 构造端到端监督训练所需的数据对象
    if (
        args.split_strategy == "time-series"
        and getattr(args, "split_override_masks", None) is None
    ):
        if _parse_repeat_seeds(getattr(args, "repeat_seeds", "")):
            train_supervised_structured_model_repeated_time_series(args)
            return
        # 执行 train_supervised_structured_model_time_series，完成当前步骤的副作用。
        train_supervised_structured_model_time_series(args)
        # 返回当前分支或函数的处理结果。
        return

    # 计算并保存 data，供后续逻辑复用。
    from src.scnews_sentiment.structured.propagation_models import (
        is_naive_label_history_model,
    )
    if is_naive_label_history_model(args.propagation_model):
        run_naive_label_history_baseline(args)
        return
    if is_label_history_propagation_model(args.propagation_model):
        train_label_history_baseline_model(args)
        return

    data = build_supervised_training_data(args)
    prospective_baseline = is_prospective_semantic_model(args.propagation_model)
    if not bool(getattr(args, "evaluate_test", True)):
        test_mask = data.split_by_time.pop("test", None)
        if test_mask is not None:
            data.supervision_weights[np.asarray(test_mask, dtype=bool)] = 0.0
    # 解析训练或编码设备，并保存 device
    device = resolve_device(args.train_device)
    # 确保目标目录存在
    ensure_dir(args.output_dir)
    # 计算并保存 summary_writer，供后续逻辑复用。
    summary_writer = create_tensorboard_writer(args)

    # 计算并保存 gdelt_tensor，供后续逻辑复用。
    gdelt_tensor = torch.tensor(data.gdelt_features, dtype=torch.float32, device=device)
    # 计算并保存 text_tensor，供后续逻辑复用。
    text_tensor = torch.tensor(data.text_embeddings, dtype=torch.float32, device=device)
    # 计算并保存 news_time_index，供后续逻辑复用。
    news_time_index = torch.tensor(data.news_time_index, dtype=torch.long, device=device)
    # 计算并保存 news_node_index，供后续逻辑复用。
    news_node_index = torch.tensor(data.news_node_index, dtype=torch.long, device=device)
    # 计算并保存 labels，供后续逻辑复用。
    labels = torch.tensor(data.label_values, dtype=torch.float32, device=device)
    # 计算并保存 supervision_weights，供后续逻辑复用。
    supervision_weights = torch.tensor(data.supervision_weights, dtype=torch.float32, device=device)
    # 计算并保存 split_masks，供后续逻辑复用。
    split_masks = {
        split: torch.tensor(mask, dtype=torch.bool, device=device)
        for split, mask in data.split_by_time.items()
    }

    # 计算并保存 news_encoder，供后续逻辑复用。
    news_encoder = NewsInitializationEncoder(
        gdelt_dim=data.gdelt_features.shape[1],
        text_dim=data.text_embeddings.shape[1],
        projection_dim=args.projection_dim,
        sent_hidden_sizes=parse_hidden_sizes(args.sent_hidden_sizes),   # 解析逗号分隔的 MLP 隐藏层维度
        representation_dim=args.representation_dim,
        dropout=args.dropout,
        final_layer_norm=not args.no_final_layer_norm,
    ).to(device)
    # 从 checkpoint 加载新闻初始化编码器权重，调用 news_encoder.load_checkpoint，执行当前步骤的副作用。
    news_encoder.load_checkpoint(args.checkpoint, device)

    # 计算并保存 graph_module，供后续逻辑复用。
    graph_module = build_propagation_module(args, data).to(device)
    # 调用 graph_module.load_checkpoint，执行当前步骤的副作用。
    graph_module.load_checkpoint(args.graph_checkpoint, device)

    # 计算并保存 evaluator，供后续逻辑复用。
    evaluator = RiskEvaluationScorer(
        state_dim=args.graph_state_dim,
        hidden_sizes=parse_hidden_sizes(args.eval_hidden_sizes),
        dropout=args.eval_dropout,
    ).to(device)

    # 计算并保存 parameters，供后续逻辑复用。
    alignment_batch = load_alignment_batch(
        args,
        data,
        force=prospective_baseline,
    )
    prospective_sequence: Optional[Any] = None
    prospective_mask: Optional[Any] = None
    if prospective_baseline:
        if alignment_batch is None:
            raise ValueError("Prospective-LLM + MLP requires cached LLM embeddings.")
        prospective_array, prospective_mask_array = (
            _build_prospective_semantic_sequence(
                alignment_batch,
                time_count=len(data.time_metadata),
                node_count=len(data.graph.node_codes),
            )
        )
        if prospective_array.shape[2] != int(args.prospective_embedding_dim):
            raise ValueError(
                "Cached Prospective-LLM embedding dimension "
                f"{prospective_array.shape[2]} does not match "
                f"--prospective-embedding-dim={args.prospective_embedding_dim}."
            )
        prospective_sequence = torch.tensor(
            prospective_array,
            dtype=torch.float32,
            device=device,
        )
        prospective_mask = torch.tensor(
            prospective_mask_array,
            dtype=torch.float32,
            device=device,
        )
    # 初始化 align_semantic_tensor，明确后续步骤要累积的数据。
    align_semantic_tensor: Optional[Any] = None
    # 初始化 align_time_index，明确后续步骤要累积的数据。
    align_time_index: Optional[Any] = None
    # 初始化 align_node_index，明确后续步骤要累积的数据。
    align_node_index: Optional[Any] = None
    align_pair_weights: Optional[Any] = None
    align_candidate_mask: Optional[Any] = None
    # 初始化 semantic_projector，明确后续步骤要累积的数据。
    semantic_projector: Optional[Any] = None
    # 初始化 struct_projector，明确后续步骤要累积的数据。
    struct_projector: Optional[Any] = None
    # 判断 alignment_batch is not None，决定是否进入对应分支。
    if alignment_batch is not None and args.align_weight > 0:
        # 计算并保存 align_semantic_tensor，供当前分支后续使用。
        align_semantic_tensor = torch.tensor(
            alignment_batch.semantic_embeddings,
            dtype=torch.float32,
            device=device,
        )
        # 计算并保存 align_time_index，供当前分支后续使用。
        align_time_index = torch.tensor(
            alignment_batch.time_index,
            dtype=torch.long,
            device=device,
        )
        # 计算并保存 align_node_index，供当前分支后续使用。
        align_node_index = torch.tensor(
            alignment_batch.node_index,
            dtype=torch.long,
            device=device,
        )
        align_pair_weights = torch.tensor(
            alignment_batch.pair_weights,
            dtype=torch.float32,
            device=device,
        )
        align_candidate_mask = torch.tensor(
            alignment_batch.candidate_mask,
            dtype=torch.bool,
            device=device,
        )
        # 计算并保存 semantic_projector，供当前分支后续使用。
        semantic_projector = make_mlp(
            input_dim=alignment_batch.semantic_embeddings.shape[1],
            hidden_sizes=parse_hidden_sizes(args.align_hidden_sizes),
            output_dim=args.align_dim,
            dropout=args.align_dropout,
            final_layer_norm=False,
        ).to(device)
        # 计算并保存 struct_projector，供当前分支后续使用。
        struct_projector = make_mlp(
            input_dim=args.graph_state_dim,
            hidden_sizes=parse_hidden_sizes(args.align_hidden_sizes),
            output_dim=args.align_dim,
            dropout=args.align_dropout,
            final_layer_norm=False,
        ).to(device)

    # 计算并保存 parameters，供后续逻辑复用。
    parameters = (
        []
        if prospective_baseline
        else list(news_encoder.module.parameters())
    )
    # 调用 parameters.extend，执行当前步骤的副作用。
    parameters.extend(graph_module.module.parameters())
    # 调用 parameters.extend，执行当前步骤的副作用。
    parameters.extend(evaluator.module.parameters())
    # 判断 semantic_projector is not None and struct_projector is...，决定是否进入对应分支。
    if semantic_projector is not None and struct_projector is not None:
        # 执行 extend，完成当前步骤的副作用。
        parameters.extend(semantic_projector.parameters())
        # 执行 extend，完成当前步骤的副作用。
        parameters.extend(struct_projector.parameters())
    # 计算并保存 optimizer，供后续逻辑复用。
    optimizer = torch.optim.AdamW(
        parameters,
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    # 计算并保存 node_count，供后续逻辑复用。
    node_count = len(data.graph.node_codes)
    # 计算并保存 time_count，供后续逻辑复用。
    time_count = len(data.time_metadata)
    # 计算并保存 adjacency，供后续逻辑复用。
    adjacency = graph_module.module.normalized_adjacency
    # 计算并保存 best_val，供后续逻辑复用。
    best_val = float("inf")
    # 初始化带类型标注的 best_state，明确后续存储内容。
    best_state: Optional[Dict[str, Any]] = None
    # 初始化带类型标注的 history，明确后续存储内容。
    history: List[Dict[str, Any]] = []
    # 记录验证集连续未改善的轮数，用于早停判断。
    epochs_without_improvement = 0
    # 标记训练是否因为早停条件而提前结束。
    early_stopped = False
    # 记录触发早停的 epoch；未触发时保持 None。
    stop_epoch: Optional[int] = None

    # 定义内部辅助函数 forward_predictions，复用当前作用域中的变量。
    def forward_predictions() -> Tuple[Any, Any]:
        # 计算并保存 news_representations，供后续逻辑复用。
        if prospective_baseline:
            if prospective_sequence is None or prospective_mask is None:
                raise RuntimeError("Prospective semantic inputs were not initialized.")
            node_risk_sequence = prospective_sequence
            node_risk_mask = prospective_mask
        else:
            news_representations = news_encoder.module(gdelt_tensor, text_tensor)
        # 计算并保存 node_risk_sequence、node_risk_mask，供后续逻辑复用。
            node_risk_sequence, node_risk_mask = aggregate_news_tensor_to_node_sequence(
                news_representations=news_representations,
                news_time_index=news_time_index,
                news_node_index=news_node_index,
                time_count=time_count,
                node_count=node_count,
            )
        # 计算并保存 propagated_states，供后续逻辑复用。
        propagated_states = graph_module.module(node_risk_sequence, node_risk_mask)
        if bool(getattr(args, "strict_pre_target", False)) and not prospective_baseline:
            propagation_module = graph_module.module
            initial_state_fn = getattr(propagation_module, "initial_state", None)
            initial_hidden_fn = getattr(propagation_module, "initial_hidden", None)
            if callable(initial_state_fn):
                initial_state = initial_state_fn()
            elif callable(initial_hidden_fn):
                # LSTM exposes its evaluator-facing hidden state separately
                # from the cell state.
                initial_state = initial_hidden_fn()
            else:
                # Feed-forward temporal baselines and zero-initialized
                # recurrent baselines do not expose a learned initial state.
                initial_state = propagated_states.new_zeros(
                    propagated_states.shape[1:]
                )
            propagated_states = torch.cat(
                [initial_state.unsqueeze(0), propagated_states[:-1]],
                dim=0,
            )
        # 计算并保存 predictions，供后续逻辑复用。
        predictions = evaluator.module(propagated_states).squeeze(-1)
        # 返回当前函数的处理结果。
        return predictions, propagated_states

    # 定义局部辅助函数，复用当前作用域中的上下文变量。
    def loss_components(
        predictions: Any,
        propagated_states: Any,
        split_name: str,
    ) -> Tuple[Any, Any, Any, Any]:
        # 返回当前函数的处理结果。
        return compute_structured_total_loss(
            predictions=predictions,
            propagated_states=propagated_states,
            labels=labels,
            supervision_weights=supervision_weights,
            split_mask=split_masks[split_name],
            adjacency=adjacency,
            smooth_weight=args.smooth_weight,
            align_weight=args.align_weight,
            semantic_embeddings=align_semantic_tensor,
            pair_time_index=align_time_index,
            pair_node_index=align_node_index,
            semantic_projector=semantic_projector,
            struct_projector=struct_projector,
            align_temperature=args.align_temperature,
            align_pair_weights=align_pair_weights,
            align_candidate_mask=align_candidate_mask,
            normalize_supervised_by_weight_sum=(
                args.normalize_supervised_by_weight_sum
            ),
        )

    # 定义局部辅助函数，复用当前作用域中的上下文变量。
    def loss_to_float(value: Any) -> float:
        # 返回当前函数的处理结果。
        return float(value.detach().cpu())

    # 遍历 range(1, args.epochs + 1)，逐项执行处理逻辑。
    for epoch in range(1, args.epochs + 1):
        # 执行 train，完成当前步骤的副作用。
        news_encoder.module.train()
        # 执行 train，完成当前步骤的副作用。
        graph_module.module.train()
        # 执行 train，完成当前步骤的副作用。
        evaluator.module.train()
        # 判断 semantic_projector is not None and struct_projector is...，处理对应情况。
        if semantic_projector is not None and struct_projector is not None:
            # 执行 train，完成当前步骤的副作用。
            semantic_projector.train()
            # 执行 train，完成当前步骤的副作用。
            struct_projector.train()

        # 执行 zero_grad，完成当前步骤的副作用。
        optimizer.zero_grad(set_to_none=True)
        # 计算并保存 predictions、propagated_states，供当前分支后续使用。
        predictions, propagated_states = forward_predictions()
        # 计算并保存 total_loss、sup_loss、smooth_loss，供当前分支后续使用。
        total_loss, sup_loss, smooth_loss, align_loss = loss_components(
            predictions,
            propagated_states,
            "train",
        )
        # 执行 backward，完成当前步骤的副作用。
        total_loss.backward()
        # 判断 args.grad_clip_norm > 0，处理对应情况。
        if args.grad_clip_norm > 0:
            # 执行 clip_grad_norm_，完成当前步骤的副作用。
            torch.nn.utils.clip_grad_norm_(parameters, args.grad_clip_norm)
        # 执行 step，完成当前步骤的副作用。
        optimizer.step()

        # 执行 eval，完成当前步骤的副作用。
        news_encoder.module.eval()
        # 执行 eval，完成当前步骤的副作用。
        graph_module.module.eval()
        # 执行 eval，完成当前步骤的副作用。
        evaluator.module.eval()
        # 判断 semantic_projector is not None and struct_projector is...，处理对应情况。
        if semantic_projector is not None and struct_projector is not None:
            # 执行 eval，完成当前步骤的副作用。
            semantic_projector.eval()
            # 执行 eval，完成当前步骤的副作用。
            struct_projector.eval()
        # 使用上下文管理器打开资源，结束后自动清理。
        with torch.no_grad():
            # 计算并保存 eval_predictions、eval_states，供当前分支后续使用。
            eval_predictions, eval_states = forward_predictions()
            # 计算并保存 train_total_t、train_sup_t、train_smooth_t，供当前分支后续使用。
            train_total_t, train_sup_t, train_smooth_t, train_align_t = loss_components(
                eval_predictions,
                eval_states,
                "train",
            )
            # 计算并保存 val_total_t、val_sup_t、val_smooth_t，供当前分支后续使用。
            val_total_t, val_sup_t, val_smooth_t, val_align_t = loss_components(
                eval_predictions,
                eval_states,
                "val",
            )
            # 计算并保存 train_total，供当前分支后续使用。
            train_total = loss_to_float(train_total_t)
            # 计算并保存 train_sup，供当前分支后续使用。
            train_sup = loss_to_float(train_sup_t)
            # 计算并保存 train_smooth，供当前分支后续使用。
            train_smooth = loss_to_float(train_smooth_t)
            # 计算并保存 train_align，供当前分支后续使用。
            train_align = loss_to_float(train_align_t)
            # 计算并保存 val_total，供当前分支后续使用。
            val_total = loss_to_float(val_total_t)
            # 计算并保存 val_sup，供当前分支后续使用。
            val_sup = loss_to_float(val_sup_t)
            # 计算并保存 val_smooth，供当前分支后续使用。
            val_smooth = loss_to_float(val_smooth_t)
            # 计算并保存 val_align，供当前分支后续使用。
            val_align = loss_to_float(val_align_t)

        # 计算并保存 checkpoint_metric_value，供当前分支后续使用。
        checkpoint_metric_value = (
            val_sup if args.checkpoint_metric == "val_supervised" else val_total
        )
        # 判断验证集损失是否达到足够幅度的下降，避免极小数值波动重置 patience。
        improved = checkpoint_metric_value < best_val - args.early_stopping_min_delta
        # 判断 improved，处理对应情况。
        if improved:
            # 计算并保存 best_val，供当前分支后续使用。
            best_val = checkpoint_metric_value
            # 计算并保存 best_state，供当前分支后续使用。
            best_state = {
                "news_encoder": copy.deepcopy(news_encoder.module.state_dict()),
                "graph_module": copy.deepcopy(graph_module.module.state_dict()),
                "evaluator": copy.deepcopy(evaluator.module.state_dict()),
                "epoch": epoch,
                "checkpoint_metric": args.checkpoint_metric,
                "checkpoint_metric_value": checkpoint_metric_value,
                "val_total_loss": val_total,
                "val_supervised_loss": val_sup,
                "val_smooth_loss": val_smooth,
                "val_alignment_loss": val_align,
            }
            # 判断 semantic_projector is not None and struct_projector is...，处理对应情况。
            if semantic_projector is not None and struct_projector is not None:
                # 计算并保存 best_state，供当前分支后续使用。
                best_state["semantic_projector"] = copy.deepcopy(
                    semantic_projector.state_dict()
                )
                # 计算并保存 best_state，供当前分支后续使用。
                best_state["struct_projector"] = copy.deepcopy(
                    struct_projector.state_dict()
                )
            # 验证集改善后重置早停计数。
            epochs_without_improvement = 0
        else:
            # 验证集没有达到 min_delta 改善时累积等待轮数。
            epochs_without_improvement += 1

        # 计算并保存 record，供当前分支后续使用。
        record = {
            "epoch": epoch,
            "train_total_loss": train_total,
            "train_supervised_loss": train_sup,
            "train_smooth_loss": train_smooth,
            "train_alignment_loss": train_align,
            "val_total_loss": val_total,
            "val_supervised_loss": val_sup,
            "val_smooth_loss": val_smooth,
            "val_alignment_loss": val_align,
            "checkpoint_metric": args.checkpoint_metric,
            "checkpoint_metric_value": checkpoint_metric_value,
            "is_best": improved,
            "epochs_without_improvement": epochs_without_improvement,
        }
        # 执行 append，完成当前步骤的副作用。
        history.append(record)
        # 执行 write_epoch_tensorboard，完成当前步骤的副作用。
        write_epoch_tensorboard(
            summary_writer,
            record,
            learning_rate=args.lr,
        )

        # 生成日志中的早停等待状态，patience 为 0 时明确显示已关闭。
        wait_status = (
            "disabled"
            if args.early_stopping_patience == 0
            else f"{epochs_without_improvement}/{args.early_stopping_patience}"
        )

        # 判断 epoch == 1 or epoch % args.log_every == 0 or epoch == ...，处理对应情况。
        if epoch == 1 or epoch % args.log_every == 0 or epoch == args.epochs:
            # 计算并保存 align_log，供当前分支后续使用。
            align_log = (
                f" align={train_align:.6f}/{val_align:.6f}"
                if args.align_weight > 0
                else ""
            )
            # 执行 print，完成当前步骤的副作用。
            print(
                f"[train] epoch={epoch} "
                f"train={train_total:.6f} val={val_total:.6f} "
                f"sup={train_sup:.6f} smooth={train_smooth:.6f} "
                f"{align_log} "
                f"best_{args.checkpoint_metric}={best_val:.6f} "
                f"wait={wait_status}"
            )

        # patience 为 0 时关闭早停；否则连续未改善达到 patience 后结束训练。
        if (
            args.early_stopping_patience > 0
            and epochs_without_improvement >= args.early_stopping_patience
        ):
            # 计算并保存 early_stopped，供当前分支后续使用。
            early_stopped = True
            # 计算并保存 stop_epoch，供当前分支后续使用。
            stop_epoch = epoch
            # 执行 print，完成当前步骤的副作用。
            print(
                f"[early-stop] epoch={epoch} "
                f"best_epoch={best_state['epoch'] if best_state is not None else 'none'} "
                f"best_{args.checkpoint_metric}={best_val:.6f}"
            )
            # 处理当前逻辑步骤：break
            break

    # 根据条件 best_state is not None 选择对应处理分支。
    if best_state is not None:
        # 执行 load_state_dict，完成当前步骤的副作用。
        news_encoder.module.load_state_dict(best_state["news_encoder"])
        # 执行 load_state_dict，完成当前步骤的副作用。
        graph_module.module.load_state_dict(best_state["graph_module"])
        # 执行 load_state_dict，完成当前步骤的副作用。
        evaluator.module.load_state_dict(best_state["evaluator"])
        # 判断 semantic_projector is not None and struct_projector is...，处理对应情况。
        if semantic_projector is not None and struct_projector is not None:
            # 执行 load_state_dict，完成当前步骤的副作用。
            semantic_projector.load_state_dict(best_state["semantic_projector"])
            # 执行 load_state_dict，完成当前步骤的副作用。
            struct_projector.load_state_dict(best_state["struct_projector"])

    # 调用 news_encoder.module.eval，执行当前步骤的副作用。
    news_encoder.module.eval()
    # 调用 graph_module.module.eval，执行当前步骤的副作用。
    graph_module.module.eval()
    # 调用 evaluator.module.eval，执行当前步骤的副作用。
    evaluator.module.eval()
    # 判断 semantic_projector is not None and struct_projector is...，决定是否进入对应分支。
    if semantic_projector is not None and struct_projector is not None:
        # 执行 eval，完成当前步骤的副作用。
        semantic_projector.eval()
        # 执行 eval，完成当前步骤的副作用。
        struct_projector.eval()
    # 使用上下文管理器处理资源，确保结束时自动释放。
    with torch.no_grad():
        # 计算并保存 final_predictions_tensor、final_states_tensor，供当前分支后续使用。
        final_predictions_tensor, final_states_tensor = forward_predictions()
    # 计算并保存 final_predictions，供后续逻辑复用。
    final_predictions = final_predictions_tensor.detach().cpu().numpy().astype(np.float32)

    # 计算并保存 metrics，供后续逻辑复用。
    metrics = {
        "best_epoch": None if best_state is None else int(best_state["epoch"]),
        "checkpoint_metric": args.checkpoint_metric,
        "best_checkpoint_metric_value": best_val,
        "best_val_total_loss": (
            None if best_state is None else float(best_state["val_total_loss"])
        ),
        "best_val_supervised_loss": (
            None if best_state is None else float(best_state["val_supervised_loss"])
        ),
        "best_val_smooth_loss": (
            None if best_state is None else float(best_state["val_smooth_loss"])
        ),
        "best_val_alignment_loss": (
            None if best_state is None else float(best_state["val_alignment_loss"])
        ),
        "epochs_completed": len(history),
        "early_stopped": early_stopped,
        "stop_epoch": stop_epoch,
        "early_stopping_patience": args.early_stopping_patience,
        "early_stopping_min_delta": args.early_stopping_min_delta,
        "smooth_weight": args.smooth_weight,
        "align_weight": args.align_weight,
        "align_temperature": args.align_temperature,
        "align_dim": args.align_dim,
        "align_hidden_sizes": list(parse_hidden_sizes(args.align_hidden_sizes)),
        "align_dropout": args.align_dropout,
        "llm_forecast_csv": str(args.llm_forecast_csv),
        "llm_embedding_npz": str(args.llm_embedding_npz),
        "alignment_pair_count": 0 if alignment_batch is None else int(len(alignment_batch.pair_metadata)),
        "alignment_skipped_unmatched_rows": (
            0
            if alignment_batch is None
            else int(alignment_batch.pair_metadata.attrs.get("skipped_unmatched_rows", 0))
        ),
        "alignment_topic_hs_remapped_rows": (
            0
            if alignment_batch is None
            else int(alignment_batch.pair_metadata.attrs.get("topic_hs_remapped_rows", 0))
        ),
        "upstream_loss_weight": args.upstream_loss_weight,
        "propagation_model": args.propagation_model,
        "gat_heads": args.gat_heads,
        "dcrnn_diffusion_steps": args.dcrnn_diffusion_steps,
        "dcrnn_bidirectional": args.dcrnn_bidirectional,
        "agcrn_embed_dim": args.agcrn_embed_dim,
        "component_scale_mode": args.component_scale_mode,
        "component_scale_init": args.component_scale_init,
        "rd_fusion_gate_init": args.rd_fusion_gate_init,
        "ts_baseline_lookback": args.ts_baseline_lookback,
        "itransformer_heads": args.itransformer_heads,
        "stgcn_kernel_size": args.stgcn_kernel_size,
        "timesnet_kernel_sizes": args.timesnet_kernel_sizes,
        "ode_steps": args.ode_steps,
        "ode_step_size": args.ode_step_size,
        "grand_heads": args.grand_heads,
        "grand_attention_dim": args.grand_attention_dim,
        "gread_reaction": args.gread_reaction,
        "gread_use_soft_adjacency": args.gread_use_soft_adjacency,
        "prospective_embedding_dim": args.prospective_embedding_dim,
        "baseline_input": (
            "target_aligned_prospective_llm_consensus_embeddings"
            if prospective_baseline
            else "aggregated_news_representations"
        ),
        "uses_supply_chain_graph": str(args.propagation_model).lower()
        not in {"agcrn", "lstm", "prospective_mlp"},
        "uses_adaptive_graph": str(args.propagation_model).lower() == "agcrn",
        "graph_direction": data.graph.graph_direction,
        "diffusion_mode": args.diffusion_mode,
        "initial_state_mode": args.initial_state_mode,
        "enable_graph_diffusion": args.enable_graph_diffusion,
        "enable_two_hop_diffusion": args.enable_two_hop_diffusion,
        "enable_recovery": args.enable_recovery,
        "time_count": int(time_count),
        "node_count": int(node_count),
        "input_news_count": int(len(data.news_metadata)),
        "supervised_time_count": int((data.supervision_weights.sum(axis=1) > 0).sum()),
        "evaluate_test": bool(getattr(args, "evaluate_test", True)),
        "consensus_hypothesis_limit": getattr(args, "consensus_hypothesis_limit", None),
        "splits": {},
        "history": history,
    }
    # 遍历 data.split_by_time.items()，逐项执行处理逻辑。
    for split_name, split_mask in data.split_by_time.items():
        # 计算并保存 结果，供当前分支后续使用。
        metrics["splits"][split_name] = prediction_metrics(
            predictions=final_predictions,
            labels=data.label_values,
            supervision_weights=data.supervision_weights,
            time_mask=split_mask,
        )
    # 执行 write_final_metrics_tensorboard，完成当前步骤的副作用。
    write_final_metrics_tensorboard(summary_writer, metrics)

    # 计算并保存 predictions_frame，供后续逻辑复用。
    predictions_frame = build_training_predictions_frame(final_predictions, data)
    # 调用 predictions_frame.to_csv，执行当前步骤的副作用。
    predictions_frame.to_csv(
        args.output_dir / "structured_supervised_predictions.csv",
        index=False,
        encoding="utf-8-sig",
    )
    # 调用 data.time_metadata.to_csv，执行当前步骤的副作用。
    data.time_metadata.to_csv(
        args.output_dir / "structured_supervised_time_metadata.csv",
        index=False,
        encoding="utf-8-sig",
    )
    # 调用 data.graph.node_metadata.to_csv，执行当前步骤的副作用。
    data.graph.node_metadata.to_csv(
        args.output_dir / "structured_supervised_node_metadata.csv",
        index=False,
        encoding="utf-8-sig",
    )
    # 判断 alignment_batch is not None，决定是否进入对应分支。
    if alignment_batch is not None:
        # 执行 to_csv，完成当前步骤的副作用。
        alignment_batch.pair_metadata.to_csv(
            args.output_dir / "structured_alignment_pairs.csv",
            index=False,
            encoding="utf-8-sig",
        )
    # 使用上下文管理器处理资源，确保结束时自动释放。
    with (args.output_dir / "structured_supervised_metrics.json").open(
        "w",
        encoding="utf-8",
    ) as handle:
        # 执行 dump，完成当前步骤的副作用。
        json.dump(metrics, handle, indent=2, ensure_ascii=False)

    # 调用 torch.save，执行当前步骤的副作用。
    torch.save(
        {
            "news_encoder_state_dict": news_encoder.module.state_dict(),
            "graph_module_state_dict": graph_module.module.state_dict(),
            "evaluator_state_dict": evaluator.module.state_dict(),
            "semantic_projector_state_dict": (
                None if semantic_projector is None else semantic_projector.state_dict()
            ),
            "struct_projector_state_dict": (
                None if struct_projector is None else struct_projector.state_dict()
            ),
            "gdelt_standardizer": data.gdelt_standardizer.to_dict(),
            "topic_hs_map": data.topic_hs_map,
            "graph_node_codes": data.graph.node_codes,
            "args": vars(args),
            "metrics": metrics,
        },
        args.output_dir / "structured_supervised_model.pt",
    )
    # 判断 summary_writer is not None，决定是否进入对应分支。
    if summary_writer is not None:
        # 执行 flush，完成当前步骤的副作用。
        summary_writer.flush()
        # 执行 close，完成当前步骤的副作用。
        summary_writer.close()
    # 调用 print，执行当前步骤的副作用。
    print(f"Saved supervised training outputs to {args.output_dir}")


def build_initial_representations(args: argparse.Namespace) -> RepresentationBatch:
    """构建新闻初始化表示。

    Parameters
    ----------
    args : argparse.Namespace
        参数对象，包含输入新闻、文本编码器、投影维度和编码设备配置。

    Returns
    -------
    RepresentationBatch
        新闻级和日级初始化表示及其元数据。

    Raises
    ------
    ValueError
        当新闻表、日期、文本或结构化特征不满足约束时抛出。
    RuntimeError
        当文本编码或 MLP 前向传播失败时抛出。
    OSError
        当文本编码器模型路径不可读取时抛出。

    Notes
    -----
    这是第一阶段入口：读取新闻、标准化 GDELT 特征、冻结编码正文，然后用
    未训练或已加载的 MLP 得到 ``r_{t,u}`` 和日级平均表示。
    """
    # 调用 set_seed，执行当前步骤的副作用。
    set_seed(args.seed)
    # 计算并保存 metadata, raw_gdelt, texts，供后续逻辑复用。
    metadata, raw_gdelt, texts = load_article_rows(args)
    # 计算并保存 standardizer，供后续逻辑复用。
    standardizer = FeatureStandardizer.fit(raw_gdelt)
    # 计算并保存 gdelt_features，供后续逻辑复用。
    gdelt_features = standardizer.transform(raw_gdelt)

    # 计算并保存 text_embeddings，供后续逻辑复用。
    text_embeddings = encode_texts(texts, args)
    # 计算并保存 device，供后续逻辑复用。
    device = resolve_device(args.train_device)
    # 计算并保存 sent_hidden_sizes，供后续逻辑复用。
    sent_hidden_sizes = parse_hidden_sizes(args.sent_hidden_sizes)

    # 计算并保存 encoder，供后续逻辑复用。
    encoder = NewsInitializationEncoder(
        gdelt_dim=gdelt_features.shape[1],
        text_dim=text_embeddings.shape[1],
        projection_dim=args.projection_dim,
        sent_hidden_sizes=sent_hidden_sizes,
        representation_dim=args.representation_dim,
        dropout=args.dropout,
        final_layer_norm=not args.no_final_layer_norm,
    ).to(device)
    # 调用 encoder.load_checkpoint，执行当前步骤的副作用。
    encoder.load_checkpoint(args.checkpoint, device)
    # 调用 encoder.eval，执行当前步骤的副作用。
    encoder.eval()

    # 计算并保存 news_representations，供后续逻辑复用。
    news_representations = encoder.encode(
        gdelt_features=gdelt_features,
        text_embeddings=text_embeddings,
        batch_size=args.representation_batch_size,
        device=device,
    )
    # 计算并保存 daily_metadata, daily_representations，供后续逻辑复用。
    daily_metadata, daily_representations = aggregate_by_groups(
        metadata=metadata,
        representations=news_representations,
        group_cols=parse_group_cols(args.daily_group_cols),
    )

    # 返回当前函数的处理结果。
    return RepresentationBatch(
        news_metadata=metadata,
        daily_metadata=daily_metadata,
        gdelt_features=gdelt_features,
        text_embeddings=text_embeddings,
        news_representations=news_representations,
        daily_representations=daily_representations,
        gdelt_standardizer=standardizer,
    )


def validate_inputs(args: argparse.Namespace) -> None:
    """执行轻量输入校验。

    Parameters
    ----------
    args : argparse.Namespace
        参数对象。

    Returns
    -------
    None
        校验通过后只打印摘要信息。

    Raises
    ------
    ValueError
        当新闻输入、图输入、Topic-HS4 映射或监督标签不满足约束时抛出。
    OSError
        当 CSV 文件不可读取时由 pandas 抛出。

    Notes
    -----
    该函数不会加载文本编码器，适合在正式训练前快速检查数据表结构和映射关系。
    """
    # 计算并保存 metadata, gdelt_features, texts，供后续逻辑复用。
    metadata, gdelt_features, texts = load_article_rows(args)
    # 计算并保存 group_cols，供后续逻辑复用。
    group_cols = parse_group_cols(args.daily_group_cols)
    # 调用 require_columns，执行当前步骤的副作用。
    require_columns(metadata, group_cols, "news metadata")
    # 计算并保存 standardizer，供后续逻辑复用。
    standardizer = FeatureStandardizer.fit(gdelt_features)
    # 计算并保存 transformed，供后续逻辑复用。
    transformed = standardizer.transform(gdelt_features)
    # 调用 print，执行当前步骤的副作用。
    print(f"Loaded rows: {len(metadata)}")
    # 调用 print，执行当前步骤的副作用。
    print(f"GDELT feature shape: {transformed.shape}")
    # 调用 print，执行当前步骤的副作用。
    print(f"Text rows: {len(texts)}")
    # 调用 print，执行当前步骤的副作用。
    print(f"Daily group columns: {group_cols}")
    # 根据条件 args.run_graph_propagation or args.train_s 选择对应处理分支。
    if args.run_graph_propagation or args.train_supervised:
        # 计算并保存 graph，供当前分支后续使用。
        graph = load_industry_graph(args)
        # 计算并保存 topic_hs_map，供当前分支后续使用。
        topic_hs_map = parse_topic_hs_map(args.topic_hs_map)
        # 计算并保存 topics，供当前分支后续使用。
        topics = set(metadata["topic"].astype(str))
        # 计算并保存 missing_topics，供当前分支后续使用。
        missing_topics = sorted(topics - set(topic_hs_map))
        # 判断 missing_topics，处理对应情况。
        if missing_topics:
            # 抛出异常，阻止无效数据继续向下游传播。
            raise ValueError(
                f"Missing HS4 mapping for topics {missing_topics}. "
                "Pass --topic-hs-map Topic=HS4."
            )
        # 计算并保存 missing_nodes，供当前分支后续使用。
        missing_nodes = sorted(
            {topic_hs_map[topic] for topic in topics if topic_hs_map[topic] not in graph.node_codes}
        )
        # 判断 missing_nodes，处理对应情况。
        if missing_nodes:
            # 抛出异常，阻止无效数据继续向下游传播。
            raise ValueError(
                "Mapped topic HS4 codes are not present in the current graph nodes. "
                f"Examples: {missing_nodes[:10]}"
            )
        # 执行 print，完成当前步骤的副作用。
        print(f"Graph nodes: {len(graph.node_codes)}")
        # 执行 print，完成当前步骤的副作用。
        print(f"Graph directed edges used: {graph.edge_count}")
        # 执行 print，完成当前步骤的副作用。
        print(f"Graph direction: {graph.graph_direction}")
        # 判断 args.train_supervised，处理对应情况。
        if args.train_supervised:
            # 计算并保存 labels，供当前分支后续使用。
            labels = pd.read_csv(args.label_csv, encoding=args.encoding)
            # 判断 args.max_label_rows is not None，处理对应情况。
            if args.max_label_rows is not None:
                # 计算并保存 labels，供当前分支后续使用。
                labels = labels.head(args.max_label_rows).copy()
            # 执行 require_columns，完成当前步骤的副作用。
            require_columns(
                labels,
                [args.topic_col, args.date_col, args.target_col],
                str(args.label_csv),
            )
            # 计算并保存 label_dates，供当前分支后续使用。
            label_dates = normalize_date_column(labels[args.date_col])["date_yyyymmdd"]
            # 执行 print，完成当前步骤的副作用。
            print(f"Label rows: {len(labels)}")
            # 执行 print，完成当前步骤的副作用。
            print(f"Global date union size: {len(set(metadata['date_yyyymmdd']) | set(label_dates))}")
    # 调用 print，执行当前步骤的副作用。
    print("Dry run finished without loading the text encoder.")


def run_structured_pipeline(args: argparse.Namespace) -> int:
    """Run the structured risk pipeline from an already-built config namespace."""
    if args.dry_run:
        validate_inputs(args)
        return 0

    if args.train_supervised:
        train_supervised_structured_model(args)
        return 0

    batch = build_initial_representations(args)
    print(f"Loaded news rows: {len(batch.news_metadata)}")
    print(f"x_gdelt shape: {batch.gdelt_features.shape}")
    print(f"x_text shape: {batch.text_embeddings.shape}")
    print(f"r_news shape: {batch.news_representations.shape}")
    print(f"r_daily shape: {batch.daily_representations.shape}")

    graph_batch: Optional[GraphPropagationBatch] = None
    if args.run_graph_propagation:
        graph_batch = run_graph_propagation(batch, args)
        print(f"graph nodes: {len(graph_batch.graph.node_codes)}")
        print(f"graph directed edges used: {graph_batch.graph.edge_count}")
        print(f"node risk sequence shape: {graph_batch.node_risk_sequence.shape}")
        print(f"h0 graph shape: {graph_batch.initial_node_states.shape}")
        print(f"h_struct sequence shape: {graph_batch.propagated_node_states.shape}")

    if args.save_representations:
        save_representations(batch, args)
        if graph_batch is not None:
            save_graph_propagation(graph_batch, args)
        print(f"Saved structured representations to {args.output_dir}")
    else:
        print("Representations were kept in memory only. Pass save_representations=true to write them.")

    return 0

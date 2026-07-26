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

def aggregate_news_tensor_to_node_sequence(
    news_representations: Any,
    news_time_index: Any,
    news_node_index: Any,
    time_count: int,
    node_count: int,
) -> Tuple[Any, Any]:
    """用可微 scatter 聚合新闻表示到节点时间序列。

    Parameters
    ----------
    news_representations : torch.Tensor
        新闻级表示矩阵，形状为 ``(n_news, risk_dim)``。
    news_time_index : torch.Tensor
        每条新闻对应的时间索引，形状为 ``(n_news,)``。
    news_node_index : torch.Tensor
        每条新闻对应的图节点索引，形状为 ``(n_news,)``。
    time_count : int
        全局时间步数量。
    node_count : int
        图节点数量。

    Returns
    -------
    node_risk_sequence : torch.Tensor
        节点风险序列，形状为 ``(T, N, risk_dim)``。
    node_risk_mask : torch.Tensor
        节点是否存在直接新闻冲击的掩码，形状为 ``(T, N, 1)``。

    Raises
    ------
    RuntimeError
        当索引越界、设备不一致或张量形状不兼容时由 PyTorch 抛出。

    Notes
    -----
    该函数使用 ``index_add_`` 对同一时间和节点的多条新闻求平均，保留从
    监督损失到新闻初始化编码器的梯度路径。
    """
    # 延迟导入 torch，避免未使用阶段强制加载依赖。
    import torch

    # 计算并保存 risk_dim，供后续逻辑复用。
    risk_dim = news_representations.shape[1]
    # 计算并保存 flat_size，供后续逻辑复用。
    flat_size = time_count * node_count
    # 计算并保存 flat_index，供后续逻辑复用。
    flat_index = news_time_index * node_count + news_node_index
    # 计算并保存 risk_flat，供后续逻辑复用。
    risk_flat = torch.zeros(
        flat_size,
        risk_dim,
        dtype=news_representations.dtype,
        device=news_representations.device,
    )
    # 计算并保存 count_flat，供后续逻辑复用。
    count_flat = torch.zeros(
        flat_size,
        1,
        dtype=news_representations.dtype,
        device=news_representations.device,
    )
    # 调用 risk_flat.index_add_，执行当前步骤的副作用。
    risk_flat.index_add_(0, flat_index, news_representations)
    # 调用 count_flat.index_add_，执行当前步骤的副作用。
    count_flat.index_add_(
        0,
        flat_index,
        torch.ones(
            news_representations.shape[0],
            1,
            dtype=news_representations.dtype,
            device=news_representations.device,
        ),
    )
    # 计算并保存 risk_flat，供后续逻辑复用。
    risk_flat = risk_flat / count_flat.clamp_min(1.0)
    # 计算并保存 risk_mask，供后续逻辑复用。
    risk_mask = (count_flat > 0).to(news_representations.dtype)
    # 返回当前函数的处理结果。
    return (
        risk_flat.reshape(time_count, node_count, risk_dim),
        risk_mask.reshape(time_count, node_count, 1),
    )


def compute_supervised_loss(
    predictions: Any,
    labels: Any,
    supervision_weights: Any,
    time_mask: Any,
    *,
    normalize_by_weight_sum: bool = False,
) -> Any:
    """计算带时间掩码和节点权重的监督 MSE 损失。

    Parameters
    ----------
    predictions : torch.Tensor
        模型预测分数，形状为 ``(T, N)``。
    labels : torch.Tensor
        ArticleScore 标签矩阵，形状为 ``(T, N)``。
    supervision_weights : torch.Tensor
        监督权重矩阵，形状为 ``(T, N)``。
    time_mask : torch.Tensor
        当前 split 的时间步布尔掩码，形状为 ``(T,)``。

    Returns
    -------
    torch.Tensor
        标量监督损失。

    Raises
    ------
    RuntimeError
        当输入形状无法广播或设备不一致时由 PyTorch 抛出。

    Notes
    -----
    仅在 ``supervision_weights`` 非零且属于当前 split 的时间步上计算损失。
    没有有效时间步时返回与预测相关的零损失，保证反向传播图仍然合法。
    """
    # 延迟导入 torch，避免未使用阶段强制加载依赖。
    import torch

    # 计算并保存 active_time_mask，供后续逻辑复用。
    active_time_mask = time_mask & (supervision_weights.sum(dim=1) > 0)
    # 根据条件 not bool(active_time_mask.any()) 选择对应处理分支。
    if not bool(active_time_mask.any()):
        # 返回当前分支或函数的处理结果。
        return predictions.sum() * 0.0
    # 计算并保存 mse，供后续逻辑复用。
    mse = (predictions - labels).pow(2)
    # 计算并保存 per_time_loss，供后续逻辑复用。
    weighted_mse = mse * supervision_weights
    if normalize_by_weight_sum:
        active_weights = supervision_weights * active_time_mask[:, None]
        denominator = active_weights.sum()
        if float(denominator.detach().cpu()) <= 0:
            return predictions.sum() * 0.0
        return (weighted_mse * active_time_mask[:, None]).sum() / denominator
    per_time_loss = weighted_mse.sum(dim=1)
    # 返回当前函数的处理结果。
    return per_time_loss[active_time_mask].mean()


def compute_graph_smoothness_loss(
    predictions: Any,
    adjacency: Any,
    time_mask: Any,
) -> Any:
    """计算图平滑正则项。

    Parameters
    ----------
    predictions : torch.Tensor
        每个时间步每个节点的预测分数，形状为 ``(T, N)``。
    adjacency : torch.Tensor
        图邻接或归一化邻接矩阵，形状为 ``(N, N)``。
    time_mask : torch.Tensor
        当前 split 的时间步布尔掩码。

    Returns
    -------
    torch.Tensor
        标量图平滑损失。

    Raises
    ------
    RuntimeError
        当预测矩阵和邻接矩阵维度不兼容时由 PyTorch 抛出。

    Notes
    -----
    本函数实现 ``0.5 * sum_ij A_ij ||z_i - z_j||^2``，并按边权总量归一化。
    正则项作用在预测值上，因此会对 ``f_eval`` 和前序传播模块产生梯度。
    """
    # 计算并保存 active_time_mask，供后续逻辑复用。
    active_time_mask = time_mask
    # 根据条件 not bool(active_time_mask.any()) 选择对应处理分支。
    if not bool(active_time_mask.any()):
        # 返回当前分支或函数的处理结果。
        return predictions.sum() * 0.0
    # 计算并保存 z，供后续逻辑复用。
    z = predictions[active_time_mask]
    # 计算并保存 edge_mass，供后续逻辑复用。
    edge_mass = adjacency.sum().clamp_min(1e-12)
    # 计算并保存 row_sum，供后续逻辑复用。
    row_sum = adjacency.sum(dim=1)
    # 计算并保存 col_sum，供后续逻辑复用。
    col_sum = adjacency.sum(dim=0)
    # 计算并保存 first，供后续逻辑复用。
    first = (z.pow(2) * row_sum).sum(dim=1)
    # 计算并保存 second，供后续逻辑复用。
    second = (z.pow(2) * col_sum).sum(dim=1)
    # 计算并保存 neighbor_values，供后续逻辑复用。
    neighbor_values = z @ adjacency.t()
    # 计算并保存 cross，供后续逻辑复用。
    cross = (z * neighbor_values).sum(dim=1)
    # 计算并保存 smooth，供后续逻辑复用。
    smooth = 0.5 * (first + second - 2.0 * cross)
    # 返回当前函数的处理结果。
    return (smooth / edge_mass).mean()


def compute_alignment_loss(
    semantic_embeddings: Any,
    propagated_states: Any,
    pair_time_index: Any,
    pair_node_index: Any,
    split_mask: Any,
    semantic_projector: Any,
    struct_projector: Any,
    temperature: float,
    pair_weights: Optional[Any] = None,
    candidate_mask: Optional[Any] = None,
) -> Any:
    """Compute one-way InfoNCE from frozen LLM semantics to structured states."""
    # 导入当前函数需要的模块，避免非相关流程提前加载。
    import torch
    # 导入当前函数需要的模块，避免非相关流程提前加载。
    import torch.nn.functional as functional

    # 判断 temperature <= 0，决定是否进入对应分支。
    if temperature <= 0:
        # 抛出异常，阻止无效数据继续向下游传播。
        raise ValueError("alignment temperature must be positive.")

    # 计算并保存 active_pair_mask，供后续逻辑复用。
    active_pair_mask = split_mask[pair_time_index]
    # 判断 int(active_pair_mask.sum().detach().cpu()) < 2，决定是否进入对应分支。
    if int(active_pair_mask.sum().detach().cpu()) < 2:
        # 返回当前分支或函数的处理结果。
        return propagated_states.sum() * 0.0

    # 计算并保存 active_time_index，供后续逻辑复用。
    active_time_index = pair_time_index[active_pair_mask]
    # 计算并保存 active_node_index，供后续逻辑复用。
    active_node_index = pair_node_index[active_pair_mask]
    # 计算并保存 semantic_input，供后续逻辑复用。
    semantic_input = semantic_embeddings[active_pair_mask]
    # 计算并保存 struct_input，供后续逻辑复用。
    struct_input = propagated_states[active_time_index, active_node_index]

    # 计算并保存 semantic_z，供后续逻辑复用。
    semantic_z = functional.normalize(semantic_projector(semantic_input), dim=1)
    # 计算并保存 struct_z，供后续逻辑复用。
    struct_z = functional.normalize(struct_projector(struct_input), dim=1)
    # 计算并保存 logits，供后续逻辑复用。
    logits = semantic_z @ struct_z.t() / float(temperature)
    # 计算并保存 labels，供后续逻辑复用。
    active_count = logits.shape[0]
    labels = torch.arange(active_count, device=logits.device)
    # 返回当前函数的处理结果。
    if candidate_mask is not None:
        active_candidate_mask = candidate_mask[active_pair_mask][:, active_pair_mask]
        eligible = active_candidate_mask.sum(dim=1) >= 2
        if not bool(eligible.any()):
            return propagated_states.sum() * 0.0
        logits = logits.masked_fill(~active_candidate_mask, float("-inf"))
        per_anchor = functional.cross_entropy(
            logits[eligible],
            labels[eligible],
            reduction="none",
        )
    else:
        eligible = torch.ones(active_count, dtype=torch.bool, device=logits.device)
        per_anchor = functional.cross_entropy(logits, labels, reduction="none")

    if pair_weights is None:
        active_weights = torch.ones(
            active_count, dtype=per_anchor.dtype, device=per_anchor.device
        )
    else:
        active_weights = pair_weights[active_pair_mask].to(per_anchor.dtype)
    active_weights = active_weights[eligible]
    return (per_anchor * active_weights).sum() / active_weights.sum().clamp_min(1e-12)


def compute_structured_total_loss(
    predictions: Any,
    propagated_states: Any,
    labels: Any,
    supervision_weights: Any,
    split_mask: Any,
    adjacency: Any,
    smooth_weight: float,
    *,
    align_weight: float,
    semantic_embeddings: Optional[Any] = None,
    pair_time_index: Optional[Any] = None,
    pair_node_index: Optional[Any] = None,
    semantic_projector: Optional[Any] = None,
    struct_projector: Optional[Any] = None,
    align_temperature: float = 0.07,
    align_pair_weights: Optional[Any] = None,
    align_candidate_mask: Optional[Any] = None,
    normalize_supervised_by_weight_sum: bool = False,
) -> Tuple[Any, Any, Any, Any]:
    """Return total, supervised, smoothness, and alignment losses."""
    # 计算并保存 sup，供后续逻辑复用。
    sup = compute_supervised_loss(
        predictions,
        labels,
        supervision_weights,
        split_mask,
        normalize_by_weight_sum=normalize_supervised_by_weight_sum,
    )
    # 计算并保存 smooth，供后续逻辑复用。
    smooth = compute_graph_smoothness_loss(predictions, adjacency, split_mask)
    # 计算并保存 align，供后续逻辑复用。
    align = predictions.sum() * 0.0

    # 判断 align_weight > 0，决定是否进入对应分支。
    if align_weight > 0:
        # 计算并保存 required，供当前分支后续使用。
        required = [
            semantic_embeddings,
            pair_time_index,
            pair_node_index,
            semantic_projector,
            struct_projector,
        ]
        # 判断 any(item is None for item in required)，处理对应情况。
        if any(item is None for item in required):
            # 抛出异常，阻止无效数据继续向下游传播。
            raise ValueError("Alignment is enabled but alignment tensors/modules are missing.")
        # 计算并保存 align，供当前分支后续使用。
        align = compute_alignment_loss(
            semantic_embeddings=semantic_embeddings,
            propagated_states=propagated_states,
            pair_time_index=pair_time_index,
            pair_node_index=pair_node_index,
            split_mask=split_mask,
            semantic_projector=semantic_projector,
            struct_projector=struct_projector,
            temperature=align_temperature,
            pair_weights=align_pair_weights,
            candidate_mask=align_candidate_mask,
        )

    # 计算并保存 total，供后续逻辑复用。
    total = sup + smooth_weight * smooth + align_weight * align
    # 返回当前函数的处理结果。
    return total, sup, smooth, align


def prediction_metrics(
    predictions: np.ndarray,
    labels: np.ndarray,
    supervision_weights: np.ndarray,
    time_mask: np.ndarray,
) -> Dict[str, Optional[float]]:
    """计算监督位置上的回归指标。

    Parameters
    ----------
    predictions : numpy.ndarray
        预测分数矩阵，形状为 ``(T, N)``。
    labels : numpy.ndarray
        标签矩阵，形状为 ``(T, N)``。
    supervision_weights : numpy.ndarray
        监督权重矩阵，形状为 ``(T, N)``。
    time_mask : numpy.ndarray
        当前 split 的时间步布尔掩码，形状为 ``(T,)``。

    Returns
    -------
    dict
        包含 ``count``、``mse``、``rmse`` 和 ``mae`` 的指标字典；无有效样本
        时数值指标为 ``None``。

    Raises
    ------
    IndexError
        当输入矩阵形状不一致导致布尔索引失败时由 NumPy 抛出。
    """
    # 计算并保存 active，供后续逻辑复用。
    active = (supervision_weights > 0) & time_mask[:, None]
    # 判断 not active.any()，决定是否进入对应分支。
    if not active.any():
        # 返回当前分支或函数的处理结果。
        return {
            "count": 0,
            "mse": None,
            "rmse": None,
            "mae": None,
            "r2": None,
            "pearson": None,
            "spearman": None,
            "median_ae": None,
            "p90_ae": None,
            "max_ae": None,
            "bias": None,
        }

    # 计算并保存 y_true，供后续逻辑复用。
    y_true = labels[active].astype(np.float64)
    # 计算并保存 y_pred，供后续逻辑复用。
    y_pred = predictions[active].astype(np.float64)
    # 计算并保存 error，供后续逻辑复用。
    error = y_pred - y_true
    # 计算并保存 mse，供后续逻辑复用。
    mse = float(np.mean(np.square(error)))
    # 计算并保存 abs_error，供后续逻辑复用。
    abs_error = np.abs(error)
    # 计算并保存 ss_res，供后续逻辑复用。
    ss_res = float(np.sum(np.square(error)))
    # 计算并保存 ss_tot，供后续逻辑复用。
    ss_tot = float(np.sum(np.square(y_true - float(np.mean(y_true)))))
    # 计算并保存 r2，供后续逻辑复用。
    r2 = None if len(y_true) <= 1 or ss_tot <= 0 else 1.0 - ss_res / ss_tot

    # 返回当前函数的处理结果。
    return {
        "count": int(active.sum()),
        "mse": mse,
        "rmse": float(np.sqrt(mse)),
        "mae": float(np.mean(abs_error)),
        "r2": None if r2 is None else float(r2),
        "pearson": correlation_value(y_true, y_pred),
        "spearman": correlation_value(rank_values(y_true), rank_values(y_pred)),
        "median_ae": float(np.median(abs_error)),
        "p90_ae": float(np.percentile(abs_error, 90)),
        "max_ae": float(np.max(abs_error)),
        "bias": float(np.mean(error)),
    }


def correlation_value(x_values: np.ndarray, y_values: np.ndarray) -> Optional[float]:
    """Return Pearson correlation when both vectors contain useful variance."""
    # 判断 len(x_values) <= 1 or len(y_values) <= 1，决定是否进入对应分支。
    if len(x_values) <= 1 or len(y_values) <= 1:
        # 返回当前分支或函数的处理结果。
        return None
    # 计算并保存 x，供后续逻辑复用。
    x = np.asarray(x_values, dtype=np.float64)
    # 计算并保存 y，供后续逻辑复用。
    y = np.asarray(y_values, dtype=np.float64)
    # 判断 float(np.std(x)) <= 0 or float(np.std(y)) <= 0，决定是否进入对应分支。
    if float(np.std(x)) <= 0 or float(np.std(y)) <= 0:
        # 返回当前分支或函数的处理结果。
        return None
    # 返回当前函数的处理结果。
    return float(np.corrcoef(x, y)[0, 1])


def rank_values(values: np.ndarray) -> np.ndarray:
    """Return average ranks for a one-dimensional array, with tie handling."""
    # 计算并保存 values，供后续逻辑复用。
    values = np.asarray(values, dtype=np.float64)
    # 计算并保存 order，供后续逻辑复用。
    order = np.argsort(values, kind="mergesort")
    # 计算并保存 ranks，供后续逻辑复用。
    ranks = np.empty(len(values), dtype=np.float64)
    # 计算并保存 start，供后续逻辑复用。
    start = 0
    # 在 start < len(values) 成立时循环调整状态。
    while start < len(values):
        # 计算并保存 end，供当前分支后续使用。
        end = start
        # 在 end + 1 < len(values) and values[order[end + 1]] ==... 成立时持续循环。
        while end + 1 < len(values) and values[order[end + 1]] == values[order[start]]:
            # 累积更新 end，记录当前迭代的贡献。
            end += 1
        # 计算并保存 average_rank，供当前分支后续使用。
        average_rank = (start + end + 2) / 2.0
        # 计算并保存 ranks，供当前分支后续使用。
        ranks[order[start : end + 1]] = average_rank
        # 计算并保存 start，供当前分支后续使用。
        start = end + 1
    # 返回当前函数的处理结果。
    return ranks


def summarize_time_series_cv_metrics(
    summary: pd.DataFrame,
) -> Dict[str, Dict[str, Dict[str, Optional[float]]]]:
    """Aggregate fold-level TimeSeriesSplit metrics into mean/std summaries."""
    # 计算并保存 metric_names，供后续逻辑复用。
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
    # 初始化 aggregate，明确后续步骤要累积的数据。
    aggregate: Dict[str, Dict[str, Dict[str, Optional[float]]]] = {}
    # 遍历 split_name in ["val", "test"]，逐项执行处理逻辑。
    for split_name in ["val", "test"]:
        # 计算并保存 aggregate，供当前分支后续使用。
        aggregate[split_name] = {}
        # 遍历 metric_name in metric_names，逐项处理数据。
        for metric_name in metric_names:
            # 计算并保存 column，供当前分支后续使用。
            column = f"{split_name}_{metric_name}"
            # 判断 column not in summary.columns，处理对应情况。
            if column not in summary.columns:
                # 处理当前逻辑步骤：continue
                continue
            # 计算并保存 values，供当前分支后续使用。
            values = pd.to_numeric(summary[column], errors="coerce").dropna().to_numpy(
                dtype=np.float64
            )
            # 判断 len(values) == 0，处理对应情况。
            if len(values) == 0:
                # 计算并保存 结果，供当前分支后续使用。
                aggregate[split_name][metric_name] = {
                    "count": 0,
                    "mean": None,
                    "std": None,
                    "min": None,
                    "max": None,
                    "mean_std": None,
                }
                # 处理当前逻辑步骤：continue
                continue
            # 计算并保存 mean_value，供当前分支后续使用。
            mean_value = float(np.mean(values))
            # 计算并保存 std_value，供当前分支后续使用。
            std_value = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
            # 计算并保存 结果，供当前分支后续使用。
            aggregate[split_name][metric_name] = {
                "count": int(len(values)),
                "mean": mean_value,
                "std": std_value,
                "min": float(np.min(values)),
                "max": float(np.max(values)),
                "mean_std": f"{mean_value:.6f} ± {std_value:.6f}",
            }
    # 返回当前函数的处理结果。
    return aggregate


def flatten_time_series_cv_aggregate(
    aggregate: Dict[str, Dict[str, Dict[str, Optional[float]]]]
) -> pd.DataFrame:
    """Convert nested TimeSeriesSplit aggregate metrics into a tabular frame."""
    # 初始化 rows，明确后续步骤要累积的数据。
    rows: List[Dict[str, Any]] = []
    # 遍历 split_name, split_metrics in aggregate.items()，逐项执行处理逻辑。
    for split_name, split_metrics in aggregate.items():
        # 遍历 metric_name, stats in split_metrics.items()，逐项处理数据。
        for metric_name, stats in split_metrics.items():
            # 初始化 row，用于后续累积或查找。
            row: Dict[str, Any] = {
                "split": split_name,
                "metric": metric_name,
            }
            # 执行 update，完成当前步骤的副作用。
            row.update(stats)
            # 执行 append，完成当前步骤的副作用。
            rows.append(row)
    # 返回当前函数的处理结果。
    return pd.DataFrame(rows)


def create_tensorboard_writer(args: argparse.Namespace) -> Optional[Any]:
    """Create a TensorBoard SummaryWriter for one supervised training run."""
    # 判断 getattr(args, "no_tensorboard", False)，决定是否进入对应分支。
    if getattr(args, "no_tensorboard", False):
        # 返回当前分支或函数的处理结果。
        return None
    # 尝试执行可能失败的逻辑，并在异常分支中兜底。
    try:
        # 导入当前分支需要的符号，供后续局部逻辑使用。
        from torch.utils.tensorboard import SummaryWriter
    except ImportError as exc:
        # 抛出异常，阻止无效数据继续向下游传播。
        raise ImportError(
            "TensorBoard logging requires tensorboard. "
            "Install it or pass --no-tensorboard to disable logging."
        ) from exc

    # 计算并保存 log_dir，供后续逻辑复用。
    log_dir = getattr(args, "tensorboard_log_dir", None)
    if log_dir is None:
        log_dir = args.output_dir / "tensorboard"
    log_dir = Path(log_dir)
    ensure_dir(log_dir)
    try:
        return SummaryWriter(log_dir=str(log_dir))
    except Exception as exc:
        raise RuntimeError(
            "Failed to create TensorBoard SummaryWriter at "
            f"{log_dir}. On Windows, TensorBoard/TensorFlow can fail on paths "
            "containing non-ASCII characters. Set tensorboard_log_dir to an "
            "ASCII-only directory such as C:/tensorboard_logs/SCRiskPrediction "
            "or set no_tensorboard=true to disable TensorBoard logging."
        ) from exc


def write_epoch_tensorboard(
    writer: Optional[Any],
    record: Dict[str, Any],
    *,
    learning_rate: float,
) -> None:
    """Write per-epoch supervised training losses to TensorBoard."""
    # 判断 writer is None，决定是否进入对应分支。
    if writer is None:
        # 返回当前分支或函数的处理结果。
        return
    # 计算并保存 epoch，供后续逻辑复用。
    epoch = int(record["epoch"])
    # 计算并保存 scalar_map，供后续逻辑复用。
    scalar_map = {
        "Loss/train_total": record["train_total_loss"],
        "Loss/train_supervised": record["train_supervised_loss"],
        "Loss/train_smooth": record["train_smooth_loss"],
        "Loss/train_alignment": record["train_alignment_loss"],
        "Loss/val_total": record["val_total_loss"],
        "Loss/val_supervised": record["val_supervised_loss"],
        "Loss/val_smooth": record["val_smooth_loss"],
        "Loss/val_alignment": record["val_alignment_loss"],
        "Training/best_epoch_flag": 1.0 if record["is_best"] else 0.0,
        "Training/epochs_without_improvement": record["epochs_without_improvement"],
        "Training/learning_rate": learning_rate,
    }
    # 遍历 tag, value in scalar_map.items()，逐项执行处理逻辑。
    for tag, value in scalar_map.items():
        # 执行 add_scalar，完成当前步骤的副作用。
        writer.add_scalar(tag, float(value), epoch)


def write_final_metrics_tensorboard(
    writer: Optional[Any],
    metrics: Dict[str, Any],
) -> None:
    """Write final split metrics and key hyperparameters to TensorBoard."""
    # 判断 writer is None，决定是否进入对应分支。
    if writer is None:
        # 返回当前分支或函数的处理结果。
        return
    # 计算并保存 step，供后续逻辑复用。
    step = int(metrics.get("best_epoch") or metrics.get("epochs_completed") or 0)
    # 遍历 split_name, split_metrics in metrics.get("splits", {}...，逐项执行处理逻辑。
    for split_name, split_metrics in metrics.get("splits", {}).items():
        # 遍历 metric_name, value in split_metrics.items()，逐项处理数据。
        for metric_name, value in split_metrics.items():
            # 判断 metric_name == "count" or value is None，处理对应情况。
            if metric_name == "count" or value is None:
                # 处理当前逻辑步骤：continue
                continue
            # 执行 add_scalar，完成当前步骤的副作用。
            writer.add_scalar(f"Metrics/{split_name}_{metric_name}", float(value), step)
    # 执行 add_text，完成当前步骤的副作用。
    writer.add_text(
        "Run/config",
        json.dumps(
            {
                "best_epoch": metrics.get("best_epoch"),
                "propagation_model": metrics.get("propagation_model"),
                "graph_direction": metrics.get("graph_direction"),
                "diffusion_mode": metrics.get("diffusion_mode"),
                "dcrnn_bidirectional": metrics.get("dcrnn_bidirectional"),
                "smooth_weight": metrics.get("smooth_weight"),
                "align_weight": metrics.get("align_weight"),
                "align_temperature": metrics.get("align_temperature"),
            },
            indent=2,
            ensure_ascii=False,
        ),
        step,
    )


def split_loss_value(
    predictions: Any,
    labels: Any,
    supervision_weights: Any,
    split_mask: Any,
    adjacency: Any,
    smooth_weight: float,
) -> Tuple[float, float, float]:
    """计算某个 split 的总损失、监督损失和图平滑损失。

    Parameters
    ----------
    predictions : torch.Tensor
        预测分数矩阵，形状为 ``(T, N)``。
    labels : torch.Tensor
        标签矩阵，形状为 ``(T, N)``。
    supervision_weights : torch.Tensor
        监督权重矩阵。
    split_mask : torch.Tensor
        当前 split 的时间步掩码。
    adjacency : torch.Tensor
        图邻接矩阵。
    smooth_weight : float
        图平滑正则权重。

    Returns
    -------
    total : float
        总损失。
    supervised : float
        监督 MSE 损失。
    smooth : float
        图平滑损失。

    Raises
    ------
    RuntimeError
        当底层损失函数输入形状不兼容时由 PyTorch 抛出。
    """
    # 计算并保存 sup，供后续逻辑复用。
    sup = compute_supervised_loss(predictions, labels, supervision_weights, split_mask)
    # 计算并保存 smooth，供后续逻辑复用。
    smooth = compute_graph_smoothness_loss(predictions, adjacency, split_mask)
    # 计算并保存 total，供后续逻辑复用。
    total = sup + smooth_weight * smooth
    # 返回当前函数的处理结果。
    return float(total.detach().cpu()), float(sup.detach().cpu()), float(smooth.detach().cpu())


def build_training_predictions_frame(
    predictions: np.ndarray,
    data: SupervisedTrainingData,
) -> pd.DataFrame:
    """构造监督预测明细表。

    Parameters
    ----------
    predictions : numpy.ndarray
        训练完成后的预测矩阵，形状为 ``(T, N)``。
    data : SupervisedTrainingData
        监督训练数据容器。

    Returns
    -------
    pandas.DataFrame
        每个被监督的时间-节点位置对应一行，包含 split、日期、HS4、真实值、
        预测值和误差。

    Raises
    ------
    IndexError
        当预测矩阵维度与数据容器中的时间或节点索引不一致时抛出。
    """
    # 初始化带类型标注的 rows，明确后续存储内容。
    rows: List[Dict[str, Any]] = []
    # 计算并保存 node_codes，供后续逻辑复用。
    node_codes = data.graph.node_codes
    # 初始化带类型标注的 split_lookup，明确后续存储内容。
    split_lookup: Dict[int, str] = {}
    # 遍历 data.split_by_time.items()，逐项执行处理逻辑。
    for split_name, split_mask in data.split_by_time.items():
        # 遍历 time_index in np.where(split_mask)[0]，逐项处理数据。
        for time_index in np.where(split_mask)[0]:
            # 计算并保存 split_lookup，供当前分支后续使用。
            split_lookup[int(time_index)] = split_name

    # 计算并保存 active_positions，供后续逻辑复用。
    active_positions = np.argwhere(data.supervision_weights > 0)
    # 遍历 active_positions，逐项执行处理逻辑。
    for time_index, node_index in active_positions:
        # 执行 append，完成当前步骤的副作用。
        rows.append(
            {
                "split": split_lookup.get(int(time_index), ""),
                "time_index": int(time_index),
                "date_yyyymmdd": data.time_metadata.loc[time_index, "date_yyyymmdd"],
                "date_iso": data.time_metadata.loc[time_index, "date_iso"],
                "hs_code_4d": node_codes[int(node_index)],
                "target_article_score": float(data.label_values[time_index, node_index]),
                "predicted_article_score": float(predictions[time_index, node_index]),
                "supervision_weight": float(data.supervision_weights[time_index, node_index]),
                "squared_error": float(
                    (predictions[time_index, node_index] - data.label_values[time_index, node_index])
                    ** 2
                ),
                "abs_error": float(
                    abs(predictions[time_index, node_index] - data.label_values[time_index, node_index])
                ),
            }
        )
    # 返回当前函数的处理结果。
    return pd.DataFrame(rows)

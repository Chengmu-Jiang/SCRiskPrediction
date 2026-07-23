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

def normalize_date_column(values: pd.Series) -> pd.DataFrame:
    """将日期列规范化为排序和展示所需的多种格式。

    Parameters
    ----------
    values : pandas.Series
        原始日期列，通常为 ``YYYYMMDD``、整数或可被 pandas 解析的日期。

    Returns
    -------
    pandas.DataFrame
        包含 ``date``、``date_yyyymmdd`` 和 ``date_iso`` 三列的数据表。

    Raises
    ------
    ValueError
        当存在无法解析的日期值时抛出，并给出最多 5 个示例。

    Notes
    -----
    该函数优先按 ``YYYYMMDD`` 解析；失败时退回到 pandas 通用日期解析。
    """
    # 计算并保存 raw，供后续逻辑复用。
    raw = values.astype(str).str.strip().str.replace(r"\.0$", "", regex=True)
    # 计算并保存 parsed，供后续逻辑复用。
    parsed = pd.to_datetime(raw, format="%Y%m%d", errors="coerce")
    # 根据条件 parsed.isna().any() 选择对应处理分支。
    if parsed.isna().any():
        # 计算并保存 parsed，供当前分支后续使用。
        parsed = pd.to_datetime(raw, errors="coerce")
    # 根据条件 parsed.isna().any() 选择对应处理分支。
    if parsed.isna().any():
        # 计算并保存 examples，供当前分支后续使用。
        examples = raw[parsed.isna()].head(5).tolist()
        # 抛出异常，阻止无效数据继续向下游传播。
        raise ValueError(f"Cannot parse Date values. Examples: {examples}")
    # 返回当前函数的处理结果。
    return pd.DataFrame(
        {
            "date": parsed,
            "date_yyyymmdd": parsed.dt.strftime("%Y%m%d"),
            "date_iso": parsed.dt.strftime("%Y-%m-%d"),
        }
    )


def load_article_rows(args: argparse.Namespace) -> Tuple[pd.DataFrame, np.ndarray, List[str]]:
    """读取新闻输入表并抽取结构化特征与正文文本。

    Parameters
    ----------
    args : argparse.Namespace
        命令行参数对象，至少需要包含 ``news_csv``、列名配置、编码和缺失值
        填充配置。

    Returns
    -------
    metadata : pandas.DataFrame
        新闻级元数据，包含 Topic、日期、新闻 ID 和可选原始字段。
    gdelt_values : numpy.ndarray
        5 个 GDELT 聚合指标组成的 ``float32`` 矩阵。
    texts : list of str
        由标题和正文拼接得到的文本列表，供冻结文本编码器使用。

    Raises
    ------
    ValueError
        当必需列缺失、正文为空、GDELT 指标不可转为数值，或特征矩阵包含
        NaN/Inf 时抛出。

    Notes
    -----
    这里读取的是 ``all_article_scores.csv``，一天允许多条新闻；后续会按
    Topic 和日期聚合为节点新闻冲击。
    """
    # 计算并保存 required，供后续逻辑复用。
    required = [
        args.topic_col,
        args.date_col,
        args.text_col,
        *GDELT_FEATURE_COLUMNS,
    ]
    # 读取并保存 news，作为后续处理的数据来源。
    news = pd.read_csv(args.news_csv, encoding=args.encoding)
    # 检查 DataFrame 是否包含指定列
    require_columns(news, required, str(args.news_csv))
    # 根据条件 args.max_rows is not None 选择对应处理分支。
    if args.max_rows is not None:
        # 计算并保存 news，供当前分支后续使用。
        news = news.head(args.max_rows).copy()
    else:
        # 计算并保存 news，供当前分支后续使用。
        news = news.copy()

    # 计算并保存 dates，供后续逻辑复用。
    dates = normalize_date_column(news[args.date_col])
    # 计算并保存 news，供后续逻辑复用。
    news["topic"] = news[args.topic_col].fillna("").astype(str)
    # 计算并保存 news，供后续逻辑复用。
    news["date"] = dates["date"]
    # 计算并保存 news，供后续逻辑复用。
    news["date_yyyymmdd"] = dates["date_yyyymmdd"]
    # 计算并保存 news，供后续逻辑复用。
    news["date_iso"] = dates["date_iso"]

    # 根据条件 args.title_col and args.title_col in news. 选择对应处理分支。
    if args.title_col and args.title_col in news.columns:
        # 计算并保存 titles，供当前分支后续使用。
        titles = news[args.title_col].fillna("").astype(str)
    else:
        # 计算并保存 titles，供当前分支后续使用。
        titles = pd.Series([""] * len(news), index=news.index)
    # 计算并保存 bodies，供后续逻辑复用。
    bodies = news[args.text_col].fillna("").astype(str)
    # 计算并保存 texts，供后续逻辑复用。
    texts = [
        f"Title: {title}\n\nArticle:\n{body}".strip() if title else body.strip()
        for title, body in zip(titles.tolist(), bodies.tolist())
    ]
    # 计算并保存 empty_text_mask，供后续逻辑复用。
    empty_text_mask = [len(text) == 0 for text in texts]
    # 根据条件 any(empty_text_mask) 选择对应处理分支。
    if any(empty_text_mask):
        # 抛出异常，阻止无效数据继续向下游传播。
        raise ValueError(f"Found {sum(empty_text_mask)} rows with empty article text.")

    # 计算并保存 gdelt，供后续逻辑复用。
    gdelt = news[GDELT_FEATURE_COLUMNS].apply(pd.to_numeric, errors="coerce")
    # 根据条件 gdelt.isna().any().any() 选择对应处理分支。
    if gdelt.isna().any().any():
        # 判断 args.fill_missing_gdelt is None，处理对应情况。
        if args.fill_missing_gdelt is None:
            # 计算并保存 bad_cols，供当前分支后续使用。
            bad_cols = gdelt.columns[gdelt.isna().any()].tolist()
            # 抛出异常，阻止无效数据继续向下游传播。
            raise ValueError(
                "GDELT feature columns contain non-numeric or missing values: "
                f"{bad_cols}. Pass --fill-missing-gdelt to use a numeric fallback."
            )
        # 计算并保存 gdelt，供当前分支后续使用。
        gdelt = gdelt.fillna(float(args.fill_missing_gdelt))

    # 计算并保存 gdelt_values，供后续逻辑复用。
    gdelt_values = gdelt.to_numpy(dtype=np.float32)
    # 根据条件 not np.isfinite(gdelt_values).all() 选择对应处理分支。
    if not np.isfinite(gdelt_values).all():
        # 抛出异常，阻止无效数据继续向下游传播。
        raise ValueError("GDELT feature matrix contains NaN or infinite values.")

    # 计算并保存 news，供后续逻辑复用。
    news["news_index"] = np.arange(len(news))
    # 计算并保存 news，供后续逻辑复用。
    news["news_id"] = (
        news["topic"].astype(str)
        + "|"
        + news["date_yyyymmdd"].astype(str)
        + "|"
        + news["news_index"].astype(str)
    )

    # 计算并保存 metadata_cols，供后续逻辑复用。
    metadata_cols = [
        "news_id",
        "news_index",
        "topic",
        "date",
        "date_yyyymmdd",
        "date_iso",
    ]
    # 计算并保存 optional_cols，供后续逻辑复用。
    optional_cols = [
        "ArticleScore",
        "SelectionScore",
        "MatchedEventIDs",
        "SourceURL",
        args.title_col,
    ]
    # 遍历 optional_cols，逐项执行处理逻辑。
    for col in optional_cols:
        # 判断 col and col in news.columns and col not in metadata_co...，处理对应情况。
        if col and col in news.columns and col not in metadata_cols:
            # 执行 append，完成当前步骤的副作用。
            metadata_cols.append(col)
    # 计算并保存 metadata，供后续逻辑复用。
    metadata = news[metadata_cols].copy()
    # 返回当前函数的处理结果。
    return metadata, gdelt_values, texts

def aggregate_by_groups(
    metadata: pd.DataFrame,
    representations: np.ndarray,
    group_cols: Sequence[str],
) -> Tuple[pd.DataFrame, np.ndarray]:
    """按元数据列对新闻表示做平均聚合。

    Parameters
    ----------
    metadata : pandas.DataFrame
        新闻级元数据，必须包含 ``news_index`` 和 ``group_cols`` 指定列。
    representations : numpy.ndarray
        新闻表示矩阵，第一维与 ``metadata`` 中的 ``news_index`` 对齐。
    group_cols : sequence of str
        分组列名，例如 ``["topic", "date_yyyymmdd"]``。

    Returns
    -------
    daily_metadata : pandas.DataFrame
        每个分组的元数据和新闻数量。
    daily_representations : numpy.ndarray
        每个分组的平均表示矩阵。

    Raises
    ------
    ValueError
        当 ``metadata`` 缺少分组列时抛出。
    IndexError
        当 ``news_index`` 超出 ``representations`` 第一维范围时由 NumPy 抛出。

    Notes
    -----
    该函数对应论文中对同一时刻所有新闻表示求平均的步骤；当前默认按
    Topic 和日期聚合，得到节点级新闻冲击前的日级表示。
    """
    # 调用 require_columns，执行当前步骤的副作用。
    require_columns(metadata, group_cols, "news metadata")

    # 初始化带类型标注的 rows，明确后续存储内容。
    rows: List[Dict[str, Any]] = []
    # 初始化带类型标注的 vectors，明确后续存储内容。
    vectors: List[np.ndarray] = []
    # 初始化带类型标注的 groupby_keys，明确后续存储内容。
    groupby_keys: Any
    # 根据条件 len(group_cols) == 1 选择对应处理分支。
    if len(group_cols) == 1:
        # 计算并保存 groupby_keys，供当前分支后续使用。
        groupby_keys = group_cols[0]
    else:
        # 计算并保存 groupby_keys，供当前分支后续使用。
        groupby_keys = list(group_cols)

    # 遍历 metadata.groupby(groupby_keys, sort=True)，逐项执行处理逻辑。
    for key, group in metadata.groupby(groupby_keys, sort=True):
        # 判断 not isinstance(key, tuple)，处理对应情况。
        if not isinstance(key, tuple):
            # 计算并保存 key，供当前分支后续使用。
            key = (key,)
        # 计算并保存 indices，供当前分支后续使用。
        indices = group["news_index"].to_numpy(dtype=int)
        # 计算并保存 row，供当前分支后续使用。
        row = {col: value for col, value in zip(group_cols, key)}
        # 计算并保存 row，供当前分支后续使用。
        row["news_count"] = int(len(indices))
        # 执行 append，完成当前步骤的副作用。
        rows.append(row)
        # 执行 append，完成当前步骤的副作用。
        vectors.append(representations[indices].mean(axis=0))

    # 计算并保存 daily_metadata，供后续逻辑复用。
    daily_metadata = pd.DataFrame(rows)
    # 计算并保存 daily_representations，供后续逻辑复用。
    daily_representations = np.vstack(vectors).astype(np.float32)
    # 返回当前函数的处理结果。
    return daily_metadata, daily_representations


def load_industry_graph(args: argparse.Namespace) -> IndustryGraph:
    """加载当前产业链图并构造归一化邻接矩阵。

    Parameters
    ----------
    args : argparse.Namespace
        参数对象，需包含节点表、边表、列名、边方向和编码配置。

    Returns
    -------
    IndustryGraph
        包含当前图节点元数据、HS4 顺序和归一化邻接矩阵的图对象。

    Raises
    ------
    ValueError
        当节点表或边表缺少必需列、边表为空、节点编码重复、边引用了节点表
        中不存在的 HS4、边权为负，或 IGPC 为负时抛出。

    Notes
    -----
    核心逻辑是只取边表端点作为当前图节点集合，再从节点表中查找 IGPC。
    邻接矩阵采用 ``adjacency[target, source]`` 的布局，因此左乘节点状态时
    表示 source 对 target 的传播影响。
    """
    # 读取并保存 nodes，作为后续处理的数据来源。
    nodes = pd.read_csv(args.nodes_csv, encoding=args.encoding)
    # 读取并保存 edges，作为后续处理的数据来源。
    edges = pd.read_csv(args.edges_csv, encoding=args.encoding)
    # 检查 DataFrame 是否包含指定列
    require_columns(nodes, [args.node_code_col, args.igpc_col], str(args.nodes_csv))
    # 检查 DataFrame 是否包含指定列
    require_columns(
        edges,
        [args.edge_upstream_col, args.edge_downstream_col],
        str(args.edges_csv),
    )

    # 计算并保存 nodes，供后续逻辑复用。
    nodes = nodes.copy()
    # 将任意 HS 编码值规范化为 4 位字符串，计算并保存 nodes，供后续逻辑复用。
    nodes["hs_code_4d_norm"] = nodes[args.node_code_col].apply(normalize_hs4)
    # 计算并保存 nodes，供后续逻辑复用。
    nodes = nodes[nodes["hs_code_4d_norm"] != ""].copy()
    # 计算并保存 duplicated，供后续逻辑复用。
    duplicated = nodes["hs_code_4d_norm"].duplicated(keep=False)
    # 根据条件 duplicated.any() 选择对应处理分支。
    if duplicated.any():
        # 计算并保存 examples，供当前分支后续使用。
        examples = nodes.loc[duplicated, "hs_code_4d_norm"].head(10).tolist()
        # 抛出异常，阻止无效数据继续向下游传播。
        raise ValueError(f"Node metadata contains duplicate HS4 codes: {examples}")

    # 计算并保存 edges，供后续逻辑复用。
    edges = edges.copy()
    # 将任意 HS 编码值规范化为 4 位字符串，计算并保存 edges，供后续逻辑复用。
    edges["upstream_hs4"] = edges[args.edge_upstream_col].apply(normalize_hs4)
    # 计算并保存 edges，供后续逻辑复用。
    edges["downstream_hs4"] = edges[args.edge_downstream_col].apply(normalize_hs4)
    # 计算并保存 edges，供后续逻辑复用。
    edges = edges[(edges["upstream_hs4"] != "") & (edges["downstream_hs4"] != "")].copy()
    # 根据条件 edges.empty 选择对应处理分支。
    if edges.empty:
        # 抛出异常，阻止无效数据继续向下游传播。
        raise ValueError(f"No valid edges found in {args.edges_csv}")

    # 根据条件 args.edge_weight_col 选择对应处理分支。
    if args.edge_weight_col:
        # 执行 require_columns，完成当前步骤的副作用。
        require_columns(edges, [args.edge_weight_col], str(args.edges_csv))
        # 计算并保存 weights，供当前分支后续使用。
        weights = pd.to_numeric(edges[args.edge_weight_col], errors="coerce").fillna(1.0)
        # 判断 (weights < 0).any()，处理对应情况。
        if (weights < 0).any():
            # 抛出异常，阻止无效数据继续向下游传播。
            raise ValueError("--edge-weight-col contains negative weights.")
        # 计算并保存 edges，供当前分支后续使用。
        edges["edge_weight"] = weights.astype(float)
    else:
        # 计算并保存 edges，供当前分支后续使用。
        edges["edge_weight"] = 1.0

    # 计算并保存 graph_codes，供后续逻辑复用。
    graph_codes = sorted(set(edges["upstream_hs4"]) | set(edges["downstream_hs4"]))
    # 计算并保存 nodes_by_code，供后续逻辑复用。
    nodes_by_code = nodes.set_index("hs_code_4d_norm", drop=False)
    # 计算并保存 missing_codes，供后续逻辑复用。
    missing_codes = [code for code in graph_codes if code not in nodes_by_code.index]
    # 根据条件 missing_codes 选择对应处理分支。
    if missing_codes:
        # 抛出异常，阻止无效数据继续向下游传播。
        raise ValueError(
            "Edges reference HS4 codes that are missing from node metadata. "
            f"Examples: {missing_codes[:10]}"
        )

    # 计算并保存 node_metadata，供后续逻辑复用。
    node_metadata = nodes_by_code.loc[graph_codes].reset_index(drop=True).copy()
    # 计算并保存 node_metadata，供后续逻辑复用。
    node_metadata["node_index"] = np.arange(len(node_metadata))
    # 计算并保存 node_metadata，供后续逻辑复用。
    node_metadata["hs_code_4d"] = node_metadata["hs_code_4d_norm"]
    # 计算并保存 node_metadata，供后续逻辑复用。
    node_metadata["igpc_value"] = pd.to_numeric(
        node_metadata[args.igpc_col], errors="coerce"
    ).fillna(0.0)
    # 根据条件 (node_metadata["igpc_value"] < 0).any() 选择对应处理分支。
    if (node_metadata["igpc_value"] < 0).any():
        # 抛出异常，阻止无效数据继续向下游传播。
        raise ValueError("IGPC values must be non-negative for log1p initialization.")

    # 计算并保存 node_to_index，供后续逻辑复用。
    node_to_index = {code: index for index, code in enumerate(graph_codes)}
    # 计算并保存 adjacency，供后续逻辑复用。
    adjacency = np.zeros((len(graph_codes), len(graph_codes)), dtype=np.float32)

    # 初始化带类型标注的 edge_records，明确后续存储内容。
    edge_records: List[Tuple[str, str, float]] = []
    # 根据条件 args.graph_direction in {"upstream-to-down 选择对应处理分支。
    if args.graph_direction in {"upstream-to-downstream", "bidirectional"}:
        # 执行 extend，完成当前步骤的副作用。
        edge_records.extend(
            zip(
                edges["upstream_hs4"].tolist(),
                edges["downstream_hs4"].tolist(),
                edges["edge_weight"].astype(float).tolist(),
            )
        )
    # 根据条件 args.graph_direction in {"downstream-to-up 选择对应处理分支。
    if args.graph_direction in {"downstream-to-upstream", "bidirectional"}:
        # 执行 extend，完成当前步骤的副作用。
        edge_records.extend(
            zip(
                edges["downstream_hs4"].tolist(),
                edges["upstream_hs4"].tolist(),
                edges["edge_weight"].astype(float).tolist(),
            )
        )

    # 遍历 edge_records，逐项执行处理逻辑。
    for source_code, target_code, weight in edge_records:
        # 计算并保存 source_index，供当前分支后续使用。
        source_index = node_to_index[source_code]
        # 计算并保存 target_index，供当前分支后续使用。
        target_index = node_to_index[target_code]
        # 累积更新 adjacency，记录当前迭代的贡献。
        adjacency[target_index, source_index] += float(weight)

    # 计算并保存 adjacency_with_self，供后续逻辑复用。
    adjacency_with_self = adjacency + np.eye(len(graph_codes), dtype=np.float32)
    # 计算并保存 degree，供后续逻辑复用。
    degree = adjacency_with_self.sum(axis=1).astype(np.float32)
    # 计算并保存 degree，供后续逻辑复用。
    degree = np.where(degree < 1e-12, 1.0, degree)
    # 计算并保存 normalizer，供后续逻辑复用。
    normalizer = np.sqrt(degree[:, None] * degree[None, :])
    # 计算并保存 normalized_adjacency，供后续逻辑复用。
    normalized_adjacency = (adjacency_with_self / normalizer).astype(np.float32)

    # 计算并保存 keep_cols，供后续逻辑复用。
    keep_cols = [
        "node_index",
        "hs_code_4d",
        "igpc_value",
    ]
    # 遍历 ["short_label", "full_description", "bec_d，逐项执行处理逻辑。
    for optional_col in ["short_label", "full_description", "bec_desc", "hs_section"]:
        # 判断 optional_col in node_metadata.columns，处理对应情况。
        if optional_col in node_metadata.columns:
            # 执行 append，完成当前步骤的副作用。
            keep_cols.append(optional_col)

    # 返回当前函数的处理结果。
    return IndustryGraph(
        node_metadata=node_metadata[keep_cols].copy(),
        node_codes=graph_codes,
        normalized_adjacency=normalized_adjacency,
        adjacency=adjacency.astype(np.float32),
        edge_count=len(edge_records),
        graph_direction=args.graph_direction,
    )


def build_node_risk_sequence(
    batch: RepresentationBatch,
    graph: IndustryGraph,
    topic_hs_map: Dict[str, str],
) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """将日级新闻表示映射到图节点风险序列。

    Parameters
    ----------
    batch : RepresentationBatch
        初始化阶段产生的日级新闻表示。
    graph : IndustryGraph
        当前产业链图。
    topic_hs_map : dict of str to str
        Topic 到 HS4 节点编码的映射。

    Returns
    -------
    time_metadata : pandas.DataFrame
        每个日期对应的时间步信息。
    risk_sequence : numpy.ndarray
        节点对齐的风险表示，形状为 ``(T, N, d_r)``。
    risk_mask : numpy.ndarray
        直接新闻冲击掩码，形状为 ``(T, N, 1)``。

    Raises
    ------
    ValueError
        当日级元数据缺少 Topic/日期列、Topic 无 HS4 映射，或映射后的 HS4
        不在当前图节点中时抛出。

    Notes
    -----
    如果同一天同一节点有多个 Topic 聚合行，本函数会再次求平均；没有新闻
    的节点风险向量保持为 0，mask 为 0。
    """
    # 调用 require_columns，执行当前步骤的副作用。
    require_columns(batch.daily_metadata, ["topic", "date_yyyymmdd"], "daily metadata")
    # 计算并保存 node_to_index，供后续逻辑复用。
    node_to_index = {code: index for index, code in enumerate(graph.node_codes)}
    # 计算并保存 dates，供后续逻辑复用。
    dates = sorted(batch.daily_metadata["date_yyyymmdd"].astype(str).unique().tolist())
    # 计算并保存 date_to_index，供后续逻辑复用。
    date_to_index = {date_text: index for index, date_text in enumerate(dates)}

    # 计算并保存 risk_dim，供后续逻辑复用。
    risk_dim = batch.daily_representations.shape[1]
    # 计算并保存 risk_sequence，供后续逻辑复用。
    risk_sequence = np.zeros((len(dates), len(graph.node_codes), risk_dim), dtype=np.float32)
    # 计算并保存 risk_counts，供后续逻辑复用。
    risk_counts = np.zeros((len(dates), len(graph.node_codes), 1), dtype=np.float32)

    # 计算并保存 missing_topics，供后续逻辑复用。
    missing_topics = sorted(
        set(batch.daily_metadata["topic"].astype(str)) - set(topic_hs_map)
    )
    # 根据条件 missing_topics 选择对应处理分支。
    if missing_topics:
        # 抛出异常，阻止无效数据继续向下游传播。
        raise ValueError(
            f"Missing HS4 mapping for topics {missing_topics}. "
            "Pass --topic-hs-map Topic=HS4."
        )

    # 初始化带类型标注的 missing_graph_nodes，明确后续存储内容。
    missing_graph_nodes: List[str] = []
    # 遍历 batch.daily_metadata.reset_index(drop=True，逐项执行处理逻辑。
    for row_index, row in batch.daily_metadata.reset_index(drop=True).iterrows():
        # 计算并保存 topic，供当前分支后续使用。
        topic = str(row["topic"])
        # 计算并保存 node_code，供当前分支后续使用。
        node_code = topic_hs_map[topic]
        # 判断 node_code not in node_to_index，处理对应情况。
        if node_code not in node_to_index:
            # 执行 append，完成当前步骤的副作用。
            missing_graph_nodes.append(node_code)
            # 处理当前逻辑步骤：continue
            continue
        # 计算并保存 time_index，供当前分支后续使用。
        time_index = date_to_index[str(row["date_yyyymmdd"])]
        # 计算并保存 node_index，供当前分支后续使用。
        node_index = node_to_index[node_code]
        # 累积更新 risk_sequence，记录当前迭代的贡献。
        risk_sequence[time_index, node_index] += batch.daily_representations[row_index]
        # 累积更新 risk_counts，记录当前迭代的贡献。
        risk_counts[time_index, node_index, 0] += 1.0

    # 根据条件 missing_graph_nodes 选择对应处理分支。
    if missing_graph_nodes:
        # 计算并保存 examples，供当前分支后续使用。
        examples = sorted(set(missing_graph_nodes))[:10]
        # 抛出异常，阻止无效数据继续向下游传播。
        raise ValueError(
            "Mapped topic HS4 codes are not present in the current graph nodes. "
            f"Examples: {examples}"
        )

    # 计算并保存 nonzero_counts，供后续逻辑复用。
    nonzero_counts = risk_counts > 0
    # 计算并保存 risk_sequence，供后续逻辑复用。
    risk_sequence = np.divide(
        risk_sequence,
        np.maximum(risk_counts, 1.0),
        out=risk_sequence,
        where=np.broadcast_to(nonzero_counts, risk_sequence.shape),
    ).astype(np.float32)
    # 计算并保存 risk_mask，供后续逻辑复用。
    risk_mask = nonzero_counts.astype(np.float32)

    # 计算并保存 parsed_dates，供后续逻辑复用。
    parsed_dates = pd.to_datetime(pd.Series(dates), format="%Y%m%d", errors="coerce")
    # 计算并保存 time_metadata，供后续逻辑复用。
    time_metadata = pd.DataFrame(
        {
            "time_index": np.arange(len(dates)),
            "date_yyyymmdd": dates,
            "date_iso": parsed_dates.dt.strftime("%Y-%m-%d"),
            "direct_news_node_count": risk_mask.sum(axis=1).reshape(-1).astype(int),
        }
    )
    # 返回当前函数的处理结果。
    return time_metadata, risk_sequence, risk_mask


def load_label_matrix(
    args: argparse.Namespace,
    graph: IndustryGraph,
    topic_hs_map: Dict[str, str],
    all_dates: Sequence[str],
) -> Tuple[np.ndarray, np.ndarray]:
    """读取监督标签并对齐到全局日期轴和图节点。

    Parameters
    ----------
    args : argparse.Namespace
        参数对象，需包含 ``label_csv``、列名、编码和可选截断配置。
    graph : IndustryGraph
        当前产业链图。
    topic_hs_map : dict of str to str
        Topic 到 HS4 节点编码的映射。
    all_dates : sequence of str
        全局日期并集，格式为 ``YYYYMMDD``。

    Returns
    -------
    label_values : numpy.ndarray
        ArticleScore 标签矩阵，形状为 ``(T, N)``。
    label_mask : numpy.ndarray
        标签存在掩码，形状为 ``(T, N)``。

    Raises
    ------
    ValueError
        当标签表缺少必需列、ArticleScore 不可转为数值、同一 Topic-Date 有
        重复标签、Topic 无 HS4 映射，或映射节点不在图中时抛出。

    Notes
    -----
    标签来自 ``daily_top_news.csv``，该文件应保证每个 Topic 每个有效日期
    只有一条标签新闻。
    """
    # 计算并保存 labels，供后续逻辑复用。
    labels = pd.read_csv(args.label_csv, encoding=args.encoding)
    # 调用 require_columns，执行当前步骤的副作用。
    require_columns(
        labels,
        [args.topic_col, args.date_col, args.target_col],
        str(args.label_csv),
    )
    # 根据条件 args.max_label_rows is not None 选择对应处理分支。
    if args.max_label_rows is not None:
        # 计算并保存 labels，供当前分支后续使用。
        labels = labels.head(args.max_label_rows).copy()
    else:
        # 计算并保存 labels，供当前分支后续使用。
        labels = labels.copy()

    # 计算并保存 dates，供后续逻辑复用。
    dates = normalize_date_column(labels[args.date_col])
    # 计算并保存 labels，供后续逻辑复用。
    labels["topic"] = labels[args.topic_col].fillna("").astype(str)
    # 计算并保存 labels，供后续逻辑复用。
    labels["date_yyyymmdd"] = dates["date_yyyymmdd"]
    # 计算并保存 labels，供后续逻辑复用。
    labels["article_score"] = pd.to_numeric(labels[args.target_col], errors="coerce")
    # 根据条件 labels["article_score"].isna().any() 选择对应处理分支。
    if labels["article_score"].isna().any():
        # 计算并保存 examples，供当前分支后续使用。
        examples = labels.loc[labels["article_score"].isna(), args.target_col].head(5).tolist()
        # 抛出异常，阻止无效数据继续向下游传播。
        raise ValueError(f"Label ArticleScore contains non-numeric values: {examples}")

    # 计算并保存 duplicated，供后续逻辑复用。
    duplicated = labels.duplicated(["topic", "date_yyyymmdd"], keep=False)
    # 根据条件 duplicated.any() 选择对应处理分支。
    if duplicated.any():
        # 计算并保存 examples，供当前分支后续使用。
        examples = labels.loc[duplicated, ["topic", "date_yyyymmdd"]].head(10)
        # 抛出异常，阻止无效数据继续向下游传播。
        raise ValueError(
            "Label CSV must contain one row per Topic-Date. "
            f"Duplicate examples: {examples.to_dict(orient='records')}"
        )

    # 计算并保存 missing_topics，供后续逻辑复用。
    missing_topics = sorted(set(labels["topic"]) - set(topic_hs_map))
    # 根据条件 missing_topics 选择对应处理分支。
    if missing_topics:
        # 抛出异常，阻止无效数据继续向下游传播。
        raise ValueError(
            f"Missing HS4 mapping for label topics {missing_topics}. "
            "Pass --topic-hs-map Topic=HS4."
        )

    # 计算并保存 node_to_index，供后续逻辑复用。
    node_to_index = {code: index for index, code in enumerate(graph.node_codes)}
    # 计算并保存 date_to_index，供后续逻辑复用。
    date_to_index = {date_text: index for index, date_text in enumerate(all_dates)}
    # 计算并保存 label_values，供后续逻辑复用。
    label_values = np.zeros((len(all_dates), len(graph.node_codes)), dtype=np.float32)
    # 计算并保存 label_mask，供后续逻辑复用。
    label_mask = np.zeros((len(all_dates), len(graph.node_codes)), dtype=np.float32)

    # 初始化带类型标注的 missing_nodes，明确后续存储内容。
    missing_nodes: List[str] = []
    # 遍历 labels.iterrows()，逐项执行处理逻辑。
    for _, row in labels.iterrows():
        # 计算并保存 node_code，供当前分支后续使用。
        node_code = topic_hs_map[str(row["topic"])]
        # 判断 node_code not in node_to_index，处理对应情况。
        if node_code not in node_to_index:
            # 执行 append，完成当前步骤的副作用。
            missing_nodes.append(node_code)
            # 处理当前逻辑步骤：continue
            continue
        # 计算并保存 date_text，供当前分支后续使用。
        date_text = str(row["date_yyyymmdd"])
        # 判断 date_text not in date_to_index，处理对应情况。
        if date_text not in date_to_index:
            # 处理当前逻辑步骤：continue
            continue
        # 计算并保存 time_index，供当前分支后续使用。
        time_index = date_to_index[date_text]
        # 计算并保存 node_index，供当前分支后续使用。
        node_index = node_to_index[node_code]
        # 计算并保存 label_values，供当前分支后续使用。
        label_values[time_index, node_index] = float(row["article_score"])
        # 计算并保存 label_mask，供当前分支后续使用。
        label_mask[time_index, node_index] = 1.0

    # 根据条件 missing_nodes 选择对应处理分支。
    if missing_nodes:
        # 计算并保存 examples，供当前分支后续使用。
        examples = sorted(set(missing_nodes))[:10]
        # 抛出异常，阻止无效数据继续向下游传播。
        raise ValueError(
            "Mapped label HS4 codes are not present in the current graph nodes. "
            f"Examples: {examples}"
        )
    # 返回当前函数的处理结果。
    return label_values, label_mask


def build_supervision_weights(
    label_mask: np.ndarray,
    graph: IndustryGraph,
    args: argparse.Namespace,
) -> np.ndarray:
    """根据上游/下游监督规则构造损失权重矩阵。

    Parameters
    ----------
    label_mask : numpy.ndarray
        标签存在掩码，形状为 ``(T, N)``。
    graph : IndustryGraph
        当前产业链图。
    args : argparse.Namespace
        参数对象，包含上游节点、下游节点和上游损失权重配置。

    Returns
    -------
    numpy.ndarray
        与 ``label_mask`` 同形状的监督权重矩阵。

    Raises
    ------
    ValueError
        当配置的上游或下游 HS4 节点不存在于当前图中时抛出。

    Notes
    -----
    核心逻辑是按确认后的三种情况选择下一时刻可监督节点：上游标签存在则
    监督上游；下游标签存在且下一时刻上游有标签或当前时刻曾有新闻标签时，
    监督下游。
    """
    # 计算并保存 node_to_index，供后续逻辑复用。
    node_to_index = {code: index for index, code in enumerate(graph.node_codes)}
    # 计算并保存 upstream_code，供后续逻辑复用。
    upstream_code = normalize_hs4(args.upstream_node_code)
    # 计算并保存 downstream_code，供后续逻辑复用。
    downstream_code = normalize_hs4(args.downstream_node_code)
    # 根据条件 upstream_code not in node_to_index 选择对应处理分支。
    if upstream_code not in node_to_index:
        # 抛出异常，阻止无效数据继续向下游传播。
        raise ValueError(f"Upstream node code {upstream_code} is not in the graph.")
    # 根据条件 downstream_code not in node_to_index 选择对应处理分支。
    if downstream_code not in node_to_index:
        # 抛出异常，阻止无效数据继续向下游传播。
        raise ValueError(f"Downstream node code {downstream_code} is not in the graph.")

    # 计算并保存 upstream_index，供后续逻辑复用。
    upstream_index = node_to_index[upstream_code]
    # 计算并保存 downstream_index，供后续逻辑复用。
    downstream_index = node_to_index[downstream_code]
    # 计算并保存 weights，供后续逻辑复用。
    weights = np.zeros_like(label_mask, dtype=np.float32)
    # 遍历 range(1, label_mask.shape[0])，逐项执行处理逻辑。
    for target_time in range(1, label_mask.shape[0]):
        # 计算并保存 previous_time，供当前分支后续使用。
        previous_time = target_time - 1
        # 计算并保存 omega_next_u，供当前分支后续使用。
        omega_next_u = label_mask[target_time, upstream_index] > 0
        # 计算并保存 omega_next_d，供当前分支后续使用。
        omega_next_d = label_mask[target_time, downstream_index] > 0
        # 计算并保存 omega_prev_any，供当前分支后续使用。
        omega_prev_any = (
            label_mask[previous_time, upstream_index]
            + label_mask[previous_time, downstream_index]
        ) > 0

        # 判断 omega_next_u，处理对应情况。
        if omega_next_u:
            # 计算并保存 weights，供当前分支后续使用。
            weights[target_time, upstream_index] = float(args.upstream_loss_weight)
        # 判断 omega_next_d and (omega_next_u or omega_prev_any)，处理对应情况。
        if omega_next_d and (omega_next_u or omega_prev_any):
            # 计算并保存 weights，供当前分支后续使用。
            weights[target_time, downstream_index] = 1.0
    # 返回当前函数的处理结果。
    return weights


def split_supervised_times(
    supervision_weights: np.ndarray,
    train_ratio: float,
    val_ratio: float,
) -> Dict[str, np.ndarray]:
    """按时间顺序切分监督时间步。

    Parameters
    ----------
    supervision_weights : numpy.ndarray
        监督权重矩阵，形状为 ``(T, N)``。
    train_ratio : float
        训练集时间步比例。
    val_ratio : float
        验证集时间步比例。

    Returns
    -------
    dict of str to numpy.ndarray
        ``train``、``val``、``test`` 三个布尔时间掩码。

    Raises
    ------
    ValueError
        当可监督时间步少于 3 个，或无法构造非空 train/val/test 切分时抛出。

    Notes
    -----
    该函数只切分存在监督权重的目标时间步，并跳过时间索引 0，因为训练目标
    是相邻有效日期的 ``t -> t+1``。
    """
    # 计算并保存 active_times，供后续逻辑复用。
    active_times = np.where(supervision_weights.sum(axis=1) > 0)[0]
    # 计算并保存 active_times，供后续逻辑复用。
    active_times = active_times[active_times > 0]
    # 根据条件 len(active_times) < 3 选择对应处理分支。
    if len(active_times) < 3:
        # 抛出异常，阻止无效数据继续向下游传播。
        raise ValueError(
            "Need at least three supervised target times to create train/val/test splits."
        )

    # 计算并保存 n_total，供后续逻辑复用。
    n_total = len(active_times)
    # 计算并保存 n_train，供后续逻辑复用。
    n_train = max(1, int(np.floor(n_total * train_ratio)))
    # 计算并保存 n_val，供后续逻辑复用。
    n_val = max(1, int(np.floor(n_total * val_ratio)))
    # 根据条件 n_train + n_val >= n_total 选择对应处理分支。
    if n_train + n_val >= n_total:
        # 计算并保存 n_val，供当前分支后续使用。
        n_val = max(1, n_total - n_train - 1)
    # 在 n_train + n_val >= n_total and n_train > 1 成立时持续调整当前状态。
    while n_train + n_val >= n_total and n_train > 1:
        # 累积更新 n_train，记录当前迭代的贡献。
        n_train -= 1
    # 根据条件 n_train + n_val >= n_total 选择对应处理分支。
    if n_train + n_val >= n_total:
        # 抛出异常，阻止无效数据继续向下游传播。
        raise ValueError("Unable to create non-empty train/val/test splits.")

    # 计算并保存 splits，供后续逻辑复用。
    splits = {
        "train": active_times[:n_train],
        "val": active_times[n_train : n_train + n_val],
        "test": active_times[n_train + n_val :],
    }
    # 初始化带类型标注的 masks，明确后续存储内容。
    masks: Dict[str, np.ndarray] = {}
    # 遍历 splits.items()，逐项执行处理逻辑。
    for split_name, indices in splits.items():
        # 计算并保存 mask，供当前分支后续使用。
        mask = np.zeros(supervision_weights.shape[0], dtype=bool)
        # 计算并保存 mask，供当前分支后续使用。
        mask[indices] = True
        # 计算并保存 masks，供当前分支后续使用。
        masks[split_name] = mask
    # 返回当前函数的处理结果。
    return masks


def supervised_active_times(supervision_weights: np.ndarray) -> np.ndarray:
    """Return supervised target time indices in chronological order."""
    # 计算并保存 active_times，供后续逻辑复用。
    active_times = np.where(supervision_weights.sum(axis=1) > 0)[0]
    # 返回当前函数的处理结果。
    return active_times[active_times > 0]


def make_split_masks(
    time_count: int,
    train_times: np.ndarray,
    val_times: np.ndarray,
    test_times: np.ndarray,
) -> Dict[str, np.ndarray]:
    """Build train/val/test boolean masks from explicit time-index arrays."""
    # 初始化 masks，明确后续步骤要累积的数据。
    masks: Dict[str, np.ndarray] = {}
    # 遍历 split_name, indices in {，逐项执行处理逻辑。
    for split_name, indices in {
        "train": train_times,
        "val": val_times,
        "test": test_times,
    }.items():
        # 计算并保存 mask，供当前分支后续使用。
        mask = np.zeros(time_count, dtype=bool)
        # 计算并保存 mask，供当前分支后续使用。
        mask[np.asarray(indices, dtype=np.int64)] = True
        # 计算并保存 masks，供当前分支后续使用。
        masks[split_name] = mask
    # 返回当前函数的处理结果。
    return masks


def build_time_series_cv_splits(
    supervision_weights: np.ndarray,
    n_splits: int,
    test_size: Optional[int],
    gap: int,
    final_test_ratio: float,
) -> Tuple[List[Dict[str, np.ndarray]], pd.DataFrame]:
    """Build expanding-window TimeSeriesSplit masks plus a final holdout test block."""
    # 计算并保存 active_times，供后续逻辑复用。
    active_times = supervised_active_times(supervision_weights)
    # 计算并保存 n_total，供后续逻辑复用。
    n_total = len(active_times)
    # 判断 n_total < 4，决定是否进入对应分支。
    if n_total < 4:
        # 抛出异常，阻止无效数据继续向下游传播。
        raise ValueError("Need at least four supervised target times for TimeSeriesSplit.")
    # 判断 not (0.0 < final_test_ratio < 1.0)，决定是否进入对应分支。
    if not (0.0 < final_test_ratio < 1.0):
        # 抛出异常，阻止无效数据继续向下游传播。
        raise ValueError("--final-test-ratio must be in (0, 1).")

    # 计算并保存 final_test_count，供后续逻辑复用。
    final_test_count = max(1, int(np.ceil(n_total * final_test_ratio)))
    # 计算并保存 cv_times，供后续逻辑复用。
    cv_times = active_times[:-final_test_count]
    # 计算并保存 final_test_times，供后续逻辑复用。
    final_test_times = active_times[-final_test_count:]
    # 判断 len(cv_times) < 3，决定是否进入对应分支。
    if len(cv_times) < 3:
        # 抛出异常，阻止无效数据继续向下游传播。
        raise ValueError(
            "Not enough pre-test supervised times after reserving final holdout test."
        )

    # 计算并保存 val_size，供后续逻辑复用。
    val_size = len(cv_times) // (n_splits + 1) if test_size is None else int(test_size)
    # 判断 val_size <= 0，决定是否进入对应分支。
    if val_size <= 0:
        # 抛出异常，阻止无效数据继续向下游传播。
        raise ValueError("TimeSeriesSplit validation size must be positive.")

    # 计算并保存 first_val_start，供后续逻辑复用。
    first_val_start = len(cv_times) - n_splits * val_size
    # 计算并保存 first_train_end，供后续逻辑复用。
    first_train_end = first_val_start - gap
    # 判断 first_train_end <= 0，决定是否进入对应分支。
    if first_train_end <= 0:
        # 抛出异常，阻止无效数据继续向下游传播。
        raise ValueError(
            "TimeSeriesSplit configuration leaves no training samples in the first fold. "
            "Reduce --tscv-n-splits, --tscv-test-size, or --tscv-gap."
        )

    # 初始化 split_masks，明确后续步骤要累积的数据。
    split_masks: List[Dict[str, np.ndarray]] = []
    # 初始化 metadata_rows，明确后续步骤要累积的数据。
    metadata_rows: List[Dict[str, Any]] = []
    # 计算并保存 time_count，供后续逻辑复用。
    time_count = supervision_weights.shape[0]
    # 遍历 fold_index in range(n_splits)，逐项执行处理逻辑。
    for fold_index in range(n_splits):
        # 计算并保存 val_start，供当前分支后续使用。
        val_start = first_val_start + fold_index * val_size
        # 计算并保存 val_end，供当前分支后续使用。
        val_end = val_start + val_size
        # 计算并保存 train_end，供当前分支后续使用。
        train_end = val_start - gap
        # 计算并保存 train_times，供当前分支后续使用。
        train_times = cv_times[:train_end]
        # 计算并保存 val_times，供当前分支后续使用。
        val_times = cv_times[val_start:val_end]
        # 判断 len(train_times) == 0 or len(val_times) == 0，处理对应情况。
        if len(train_times) == 0 or len(val_times) == 0:
            # 抛出异常，阻止无效数据继续向下游传播。
            raise ValueError(f"Empty train or validation split in fold {fold_index}.")

        # 执行 append，完成当前步骤的副作用。
        split_masks.append(
            make_split_masks(
                time_count=time_count,
                train_times=train_times,
                val_times=val_times,
                test_times=final_test_times,
            )
        )
        # 执行 append，完成当前步骤的副作用。
        metadata_rows.append(
            {
                "fold": fold_index,
                "train_count": int(len(train_times)),
                "val_count": int(len(val_times)),
                "test_count": int(len(final_test_times)),
                "gap": int(gap),
                "val_size": int(val_size),
                "train_start_time": int(train_times[0]),
                "train_end_time": int(train_times[-1]),
                "val_start_time": int(val_times[0]),
                "val_end_time": int(val_times[-1]),
                "test_start_time": int(final_test_times[0]),
                "test_end_time": int(final_test_times[-1]),
            }
        )

    # 返回当前函数的处理结果。
    return split_masks, pd.DataFrame(metadata_rows)


def build_supervised_training_data(args: argparse.Namespace) -> SupervisedTrainingData:
    """构造端到端监督训练所需的数据对象。

    Parameters
    ----------
    args : argparse.Namespace
        完整命令行参数对象。

    Returns
    -------
    SupervisedTrainingData
        包含输入新闻特征、文本嵌入、全局日期轴、标签矩阵、监督权重和时间
        切分的数据容器。

    Raises
    ------
    ValueError
        当图加载、新闻读取、标签读取、Topic-HS4 映射或时间切分任一阶段
        不满足约束时抛出。
    OSError
        当文本编码模型或 CSV 文件不可读取时由底层库抛出。

    Notes
    -----
    核心逻辑是使用 ``all_article_scores.csv`` 构造输入新闻冲击，使用
    ``daily_top_news.csv`` 构造标签，并在二者日期并集上建立统一时间轴。
    """
    # 调用 set_seed，设置 Python、NumPy 和 PyTorch 的随机种子
    set_seed(args.seed)
    # 加载当前产业链图并构造归一化邻接矩阵，并保存 graph
    graph = load_industry_graph(args)
    # 解析命令行传入的 Topic 到 HS4 映射，并保存 topic_hs_map
    topic_hs_map = parse_topic_hs_map(args.topic_hs_map)
    # 读取新闻输入表并抽取结构化特征与正文文本，并保存 metadata, raw_gdelt, texts
    metadata, raw_gdelt, texts = load_article_rows(args)

    # 按列对结构化 GDELT 特征做 z-score 标准化，并保存 standardizer
    standardizer = FeatureStandardizer.fit(raw_gdelt)
    # 应用已估计的 z-score 标准化，并保存 gdelt_features
    gdelt_features = standardizer.transform(raw_gdelt)
    # 使用冻结 Transformer 文本编码器生成新闻语义向量，并保存 text_embeddings。其中hidden_dim 是由 语言嵌入模型bge-base-en-v1.5 模型自身配置决定的
    text_embeddings = encode_texts(texts, args)

    # 计算并保存 input_dates，供后续逻辑复用。
    input_dates = set(metadata["date_yyyymmdd"].astype(str).tolist())
    # 计算并保存 label_dates_frame，供后续逻辑复用。
    label_dates_frame = pd.read_csv(
        args.label_csv,
        encoding=args.encoding,
        usecols=[args.date_col],
    )
    # 根据条件 args.max_label_rows is not None 选择对应处理分支。
    if args.max_label_rows is not None:
        # 计算并保存 label_dates_frame，供当前分支后续使用。
        label_dates_frame = label_dates_frame.head(args.max_label_rows)
    # 计算并保存 label_dates_raw，供后续逻辑复用。
    label_dates_raw = label_dates_frame[args.date_col]
    # 计算并保存 label_dates，供后续逻辑复用。
    label_dates = set(normalize_date_column(label_dates_raw)["date_yyyymmdd"].astype(str))
    # 计算并保存 all_dates，供后续逻辑复用。
    all_dates = sorted(input_dates | label_dates)
    # 计算并保存 date_to_index，供后续逻辑复用。
    date_to_index = {date_text: index for index, date_text in enumerate(all_dates)}
    # 计算并保存 node_to_index，供后续逻辑复用。
    node_to_index = {code: index for index, code in enumerate(graph.node_codes)}

    # 计算并保存 missing_topics，供后续逻辑复用。
    missing_topics = sorted(set(metadata["topic"].astype(str)) - set(topic_hs_map))
    # 根据条件 missing_topics 选择对应处理分支。
    if missing_topics:
        # 抛出异常，阻止无效数据继续向下游传播。
        raise ValueError(
            f"Missing HS4 mapping for input topics {missing_topics}. "
            "Pass --topic-hs-map Topic=HS4."
        )

    # 初始化带类型标注的 news_time_index，明确后续存储内容。
    news_time_index: List[int] = []
    # 初始化带类型标注的 news_node_index，明确后续存储内容。
    news_node_index: List[int] = []
    # 初始化带类型标注的 missing_nodes，明确后续存储内容。
    missing_nodes: List[str] = []
    # 遍历 metadata.iterrows()，逐项执行处理逻辑。
    for _, row in metadata.iterrows():
        # 计算并保存 node_code，供当前分支后续使用。
        node_code = topic_hs_map[str(row["topic"])]
        # 判断 node_code not in node_to_index，处理对应情况。
        if node_code not in node_to_index:
            # 执行 append，完成当前步骤的副作用。
            missing_nodes.append(node_code)
            # 处理当前逻辑步骤：continue
            continue
        # 执行 append，完成当前步骤的副作用。
        news_time_index.append(date_to_index[str(row["date_yyyymmdd"])])
        # 执行 append，完成当前步骤的副作用。
        news_node_index.append(node_to_index[node_code])
    # 根据条件 missing_nodes 选择对应处理分支。
    if missing_nodes:
        # 计算并保存 examples，供当前分支后续使用。
        examples = sorted(set(missing_nodes))[:10]
        # 抛出异常，阻止无效数据继续向下游传播。
        raise ValueError(
            "Mapped input HS4 codes are not present in the current graph nodes. "
            f"Examples: {examples}"
        )

    # 读取监督标签并对齐到全局日期轴和图节点，并保存 label_values, label_mask
    label_values, label_mask = load_label_matrix(args, graph, topic_hs_map, all_dates)
    # 根据上游/下游监督规则构造损失权重矩阵，并保存 supervision_weights
    supervision_weights = build_supervision_weights(label_mask, graph, args)
    # 按时间顺序切分监督时间步，并保存 split_by_time
    split_override = getattr(args, "split_override_masks", None)
    # 判断 split_override is None，决定是否进入对应分支。
    if split_override is None:
        # 计算并保存 split_by_time，供当前分支后续使用。
        split_by_time = split_supervised_times(
            supervision_weights=supervision_weights,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
        )
    else:
        # 计算并保存 split_by_time，供当前分支后续使用。
        split_by_time = {
            split_name: np.asarray(mask, dtype=bool)
            for split_name, mask in split_override.items()
        }

    # 计算并保存 parsed_dates，供后续逻辑复用。
    parsed_dates = pd.to_datetime(pd.Series(all_dates), format="%Y%m%d", errors="coerce")
    # 计算并保存 time_metadata，供后续逻辑复用。
    time_metadata = pd.DataFrame(
        {
            "time_index": np.arange(len(all_dates)),
            "date_yyyymmdd": all_dates,
            "date_iso": parsed_dates.dt.strftime("%Y-%m-%d"),
            "has_label": label_mask.sum(axis=1).astype(int),
            "has_supervision": (supervision_weights.sum(axis=1) > 0).astype(int),
        }
    )
    # 遍历 split_by_time.items()，逐项执行处理逻辑。
    for split_name, split_mask in split_by_time.items():
        # 计算并保存 time_metadata，供当前分支后续使用。
        time_metadata[f"is_{split_name}"] = split_mask.astype(int)

    # 返回当前函数的处理结果。
    return SupervisedTrainingData(
        graph=graph,
        topic_hs_map=topic_hs_map,
        news_metadata=metadata,
        time_metadata=time_metadata,
        gdelt_features=gdelt_features,
        text_embeddings=text_embeddings,
        news_time_index=np.asarray(news_time_index, dtype=np.int64),
        news_node_index=np.asarray(news_node_index, dtype=np.int64),
        label_values=label_values,
        label_mask=label_mask,
        supervision_weights=supervision_weights,
        split_by_time=split_by_time,
        gdelt_standardizer=standardizer,
    )


def build_supervised_label_history_training_data(
    args: argparse.Namespace,
) -> SupervisedTrainingData:
    """Build supervised data for baselines that only use ArticleScore history.

    The time axis intentionally matches the main structured pipeline by using the
    union of input-news dates and label dates. This keeps TimeSeriesSplit folds
    identical while avoiding BGE/text feature extraction for pure time-series
    baselines.
    """
    set_seed(args.seed)
    graph = load_industry_graph(args)
    topic_hs_map = parse_topic_hs_map(args.topic_hs_map)
    metadata, raw_gdelt, _ = load_article_rows(args)

    standardizer = FeatureStandardizer.fit(raw_gdelt)
    gdelt_features = standardizer.transform(raw_gdelt)

    input_dates = set(metadata["date_yyyymmdd"].astype(str).tolist())
    label_dates_frame = pd.read_csv(
        args.label_csv,
        encoding=args.encoding,
        usecols=[args.date_col],
    )
    if args.max_label_rows is not None:
        label_dates_frame = label_dates_frame.head(args.max_label_rows)
    label_dates = set(
        normalize_date_column(label_dates_frame[args.date_col])["date_yyyymmdd"].astype(str)
    )
    all_dates = sorted(input_dates | label_dates)

    label_values, label_mask = load_label_matrix(args, graph, topic_hs_map, all_dates)
    supervision_weights = build_supervision_weights(label_mask, graph, args)

    split_override = getattr(args, "split_override_masks", None)
    if split_override is None:
        split_by_time = split_supervised_times(
            supervision_weights=supervision_weights,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
        )
    else:
        split_by_time = {
            split_name: np.asarray(mask, dtype=bool)
            for split_name, mask in split_override.items()
        }

    parsed_dates = pd.to_datetime(pd.Series(all_dates), format="%Y%m%d", errors="coerce")
    time_metadata = pd.DataFrame(
        {
            "time_index": np.arange(len(all_dates)),
            "date_yyyymmdd": all_dates,
            "date_iso": parsed_dates.dt.strftime("%Y-%m-%d"),
            "has_label": label_mask.sum(axis=1).astype(int),
            "has_supervision": (supervision_weights.sum(axis=1) > 0).astype(int),
        }
    )
    for split_name, split_mask in split_by_time.items():
        time_metadata[f"is_{split_name}"] = split_mask.astype(int)

    return SupervisedTrainingData(
        graph=graph,
        topic_hs_map=topic_hs_map,
        news_metadata=metadata,
        time_metadata=time_metadata,
        gdelt_features=gdelt_features,
        text_embeddings=np.zeros((len(metadata), 0), dtype=np.float32),
        news_time_index=np.zeros((0,), dtype=np.int64),
        news_node_index=np.zeros((0,), dtype=np.int64),
        label_values=label_values,
        label_mask=label_mask,
        supervision_weights=supervision_weights,
        split_by_time=split_by_time,
        gdelt_standardizer=standardizer,
    )


def build_supervision_weights_for_splits(args: argparse.Namespace) -> np.ndarray:
    """Build supervision weights for split planning without loading the text encoder."""
    # 计算并保存 graph，供后续逻辑复用。
    graph = load_industry_graph(args)
    # 计算并保存 topic_hs_map，供后续逻辑复用。
    topic_hs_map = parse_topic_hs_map(args.topic_hs_map)
    # 计算并保存 metadata、_、_，供后续逻辑复用。
    metadata, _, _ = load_article_rows(args)

    # 计算并保存 input_dates，供后续逻辑复用。
    input_dates = set(metadata["date_yyyymmdd"].astype(str).tolist())
    # 读取并保存 label_dates_frame，作为后续处理的数据来源。
    label_dates_frame = pd.read_csv(
        args.label_csv,
        encoding=args.encoding,
        usecols=[args.date_col],
    )
    # 判断 args.max_label_rows is not None，决定是否进入对应分支。
    if args.max_label_rows is not None:
        # 计算并保存 label_dates_frame，供当前分支后续使用。
        label_dates_frame = label_dates_frame.head(args.max_label_rows)
    # 计算并保存 label_dates，供后续逻辑复用。
    label_dates = set(
        normalize_date_column(label_dates_frame[args.date_col])["date_yyyymmdd"].astype(str)
    )
    # 计算并保存 all_dates，供后续逻辑复用。
    all_dates = sorted(input_dates | label_dates)

    # 计算并保存 label_values、label_mask，供后续逻辑复用。
    label_values, label_mask = load_label_matrix(args, graph, topic_hs_map, all_dates)
    # 删除不再需要的中间变量，减少后续误用。
    del label_values
    # 返回当前函数的处理结果。
    return build_supervision_weights(label_mask, graph, args)


def load_alignment_batch(
    args: argparse.Namespace,
    data: SupervisedTrainingData,
) -> Optional[AlignmentBatch]:
    """Load frozen LLM semantic embeddings and align them by target date and HS4."""
    # 判断 args.align_weight <= 0，决定是否进入对应分支。
    if args.align_weight <= 0:
        # 返回当前分支或函数的处理结果。
        return None

    # 计算并保存 forecast_path，供后续逻辑复用。
    forecast_path = Path(args.llm_forecast_csv)
    # 计算并保存 embedding_path，供后续逻辑复用。
    embedding_path = Path(args.llm_embedding_npz)
    # 判断 not forecast_path.exists()，决定是否进入对应分支。
    if not forecast_path.exists():
        # 抛出异常，阻止无效数据继续向下游传播。
        raise FileNotFoundError(
            f"LLM forecast CSV does not exist: {forecast_path}. "
            "Pass --llm-forecast-csv or set --align-weight 0."
        )
    # 判断 not embedding_path.exists()，决定是否进入对应分支。
    if not embedding_path.exists():
        # 抛出异常，阻止无效数据继续向下游传播。
        raise FileNotFoundError(
            f"LLM embedding NPZ does not exist: {embedding_path}. "
            "Run llm_event_tone_pipeline.py --stage train/all, or pass --llm-embedding-npz."
        )

    # 读取并保存 forecasts，作为后续处理的数据来源。
    forecasts = pd.read_csv(forecast_path, encoding=args.encoding)
    # 执行 require_columns，完成当前步骤的副作用。
    require_columns(
        forecasts,
        ["sample_id", "target_date", "hs_code_4d"],
        str(forecast_path),
    )
    # 计算并保存 forecasts，供后续逻辑复用。
    forecasts = forecasts.copy()
    # 计算并保存 forecasts，供后续逻辑复用。
    forecasts["sample_id"] = forecasts["sample_id"].astype(str)
    # 计算并保存 duplicated，供后续逻辑复用。
    duplicated = forecasts["sample_id"].duplicated(keep=False)
    # 判断 duplicated.any()，决定是否进入对应分支。
    if duplicated.any():
        # 计算并保存 examples，供当前分支后续使用。
        examples = forecasts.loc[duplicated, "sample_id"].head(5).tolist()
        # 抛出异常，阻止无效数据继续向下游传播。
        raise ValueError(f"LLM forecast CSV has duplicated sample_id values: {examples}")

    # 使用上下文管理器处理资源，确保结束后自动释放。
    with np.load(embedding_path, allow_pickle=True) as npz:
        # 判断 "sample_id" not in npz.files or "embedding" not in npz...，处理对应情况。
        if "sample_id" not in npz.files or "embedding" not in npz.files:
            # 抛出异常，阻止无效数据继续向下游传播。
            raise ValueError(
                f"{embedding_path} must contain sample_id and embedding arrays."
            )
        # 计算并保存 embedding_sample_ids，供当前分支后续使用。
        embedding_sample_ids = np.asarray(npz["sample_id"]).astype(str)
        # 计算并保存 embeddings，供当前分支后续使用。
        embeddings = np.asarray(npz["embedding"], dtype=np.float32)

    # 判断 embeddings.ndim != 2，决定是否进入对应分支。
    if embeddings.ndim != 2:
        # 抛出异常，阻止无效数据继续向下游传播。
        raise ValueError(f"LLM embeddings must be 2-D, got shape {embeddings.shape}.")
    # 判断 len(embedding_sample_ids) != embeddings.shape[0]，决定是否进入对应分支。
    if len(embedding_sample_ids) != embeddings.shape[0]:
        # 抛出异常，阻止无效数据继续向下游传播。
        raise ValueError(
            "LLM embedding sample_id length does not match embedding row count."
        )
    # 判断 not np.isfinite(embeddings).all()，决定是否进入对应分支。
    if not np.isfinite(embeddings).all():
        # 抛出异常，阻止无效数据继续向下游传播。
        raise ValueError("LLM embeddings contain NaN or infinite values.")
    # 判断 len(set(embedding_sample_ids.tolist())) != len(embeddi...，决定是否进入对应分支。
    if len(set(embedding_sample_ids.tolist())) != len(embedding_sample_ids):
        # 抛出异常，阻止无效数据继续向下游传播。
        raise ValueError("LLM embedding NPZ has duplicated sample_id values.")

    # 计算并保存 embedding_lookup，供后续逻辑复用。
    embedding_lookup = {
        sample_id: index for index, sample_id in enumerate(embedding_sample_ids.tolist())
    }
    # 计算并保存 missing_embeddings，供后续逻辑复用。
    missing_embeddings = sorted(set(forecasts["sample_id"]) - set(embedding_lookup))
    # 判断 missing_embeddings，决定是否进入对应分支。
    if missing_embeddings:
        # 抛出异常，阻止无效数据继续向下游传播。
        raise ValueError(
            "LLM forecast CSV contains sample_id values without embeddings. "
            f"Examples: {missing_embeddings[:5]}"
        )

    # 计算并保存 normalized_dates，供后续逻辑复用。
    normalized_dates = normalize_date_column(forecasts["target_date"])
    # 计算并保存 node_to_index，供后续逻辑复用。
    node_to_index = {code: index for index, code in enumerate(data.graph.node_codes)}
    # 计算并保存 date_to_index，供后续逻辑复用。
    date_to_index = {
        str(row["date_yyyymmdd"]): int(row["time_index"])
        for _, row in data.time_metadata.iterrows()
    }
    # 初始化 split_lookup，明确后续步骤要累积的数据。
    split_lookup: Dict[int, str] = {}
    # 遍历 split_name, split_mask in data.split_by_time.items()，逐项执行处理逻辑。
    for split_name, split_mask in data.split_by_time.items():
        # 遍历 time_index in np.where(split_mask)[0]，逐项处理数据。
        for time_index in np.where(split_mask)[0]:
            # 计算并保存 split_lookup，供当前分支后续使用。
            split_lookup[int(time_index)] = split_name

    # 初始化 rows，明确后续步骤要累积的数据。
    rows: List[Dict[str, Any]] = []
    # 初始化 selected_embeddings，明确后续步骤要累积的数据。
    selected_embeddings: List[np.ndarray] = []
    # 初始化 time_indices，明确后续步骤要累积的数据。
    time_indices: List[int] = []
    # 初始化 node_indices，明确后续步骤要累积的数据。
    node_indices: List[int] = []
    # 计算并保存 skipped，供后续逻辑复用。
    skipped = 0
    # 计算并保存 topic_hs_remapped，供后续逻辑复用。
    topic_hs_remapped = 0

    # 遍历 row_number, row in forecasts.iterrows()，逐项执行处理逻辑。
    for row_number, row in forecasts.iterrows():
        # 计算并保存 sample_id，供当前分支后续使用。
        sample_id = str(row["sample_id"])
        # 计算并保存 date_yyyymmdd，供当前分支后续使用。
        date_yyyymmdd = str(normalized_dates.loc[row_number, "date_yyyymmdd"])
        # 计算并保存 raw_hs_code，供当前分支后续使用。
        raw_hs_code = normalize_hs4(row["hs_code_4d"])
        # 计算并保存 topic，供当前分支后续使用。
        topic = str(row.get("topic", ""))
        # 计算并保存 mapped_hs_code，供当前分支后续使用。
        mapped_hs_code = data.topic_hs_map.get(topic)
        # 计算并保存 hs_code，供当前分支后续使用。
        hs_code = normalize_hs4(mapped_hs_code) if mapped_hs_code else raw_hs_code
        # 判断 mapped_hs_code and raw_hs_code and raw_hs_code != hs_c...，处理对应情况。
        if mapped_hs_code and raw_hs_code and raw_hs_code != hs_code:
            # 累积更新 topic_hs_remapped，记录当前迭代的贡献。
            topic_hs_remapped += 1
        # 判断 date_yyyymmdd not in date_to_index or hs_code not in n...，处理对应情况。
        if date_yyyymmdd not in date_to_index or hs_code not in node_to_index:
            # 累积更新 skipped，记录当前迭代的贡献。
            skipped += 1
            # 处理当前逻辑步骤：continue
            continue

        # 计算并保存 time_index，供当前分支后续使用。
        time_index = date_to_index[date_yyyymmdd]
        # 计算并保存 node_index，供当前分支后续使用。
        node_index = node_to_index[hs_code]
        # 计算并保存 embedding_index，供当前分支后续使用。
        embedding_index = embedding_lookup[sample_id]
        # 执行 append，完成当前步骤的副作用。
        selected_embeddings.append(embeddings[embedding_index])
        # 执行 append，完成当前步骤的副作用。
        time_indices.append(time_index)
        # 执行 append，完成当前步骤的副作用。
        node_indices.append(node_index)
        # 执行 append，完成当前步骤的副作用。
        rows.append(
            {
                "sample_id": sample_id,
                "llm_split": str(row.get("split", "")),
                "topic": topic,
                "target_date": str(row["target_date"]),
                "date_yyyymmdd": date_yyyymmdd,
                "forecast_hs_code_4d": raw_hs_code,
                "hs_code_4d": hs_code,
                "topic_hs_remapped": int(mapped_hs_code is not None and raw_hs_code != hs_code),
                "time_index": time_index,
                "node_index": node_index,
                "structured_split": split_lookup.get(time_index, ""),
            }
        )

    # 判断 not rows，决定是否进入对应分支。
    if not rows:
        # 抛出异常，阻止无效数据继续向下游传播。
        raise ValueError(
            "No LLM forecast rows matched structured time-node positions. "
            f"Skipped rows: {skipped}."
        )

    # 计算并保存 pair_metadata，供后续逻辑复用。
    pair_metadata = pd.DataFrame(rows)
    # 计算并保存 结果，供后续逻辑复用。
    pair_metadata.attrs["skipped_unmatched_rows"] = skipped
    # 计算并保存 结果，供后续逻辑复用。
    pair_metadata.attrs["topic_hs_remapped_rows"] = topic_hs_remapped
    # 返回当前函数的处理结果。
    return AlignmentBatch(
        pair_metadata=pair_metadata,
        semantic_embeddings=np.vstack(selected_embeddings).astype(np.float32),
        time_index=np.asarray(time_indices, dtype=np.int64),
        node_index=np.asarray(node_indices, dtype=np.int64),
    )

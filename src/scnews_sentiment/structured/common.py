from __future__ import annotations

import argparse
import copy
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

DEFAULT_NEWS_CSV = Path("data") / "News" / "DailyTop_News" / "all_article_scores.csv"
DEFAULT_LABEL_CSV = Path("data") / "News" / "DailyTop_News" / "daily_top_news.csv"
DEFAULT_NODES_CSV = Path("data") / "IndustryChain" / "Filtered_Node_List_by_IGPC.csv"
DEFAULT_EDGES_CSV = Path("data") / "IndustryChain" / "Wafer2ElectricVehicle_network_edges.csv"
DEFAULT_OUTPUT_DIR = Path("outputs") / "structured_pipeline"
DEFAULT_LLM_FORECAST_CSV = Path("outputs") / "llm_event_tone" / "llm_event_tone" / "forecast_events.csv"
DEFAULT_LLM_EMBEDDINGS_NPZ = (
    Path("outputs") / "llm_event_tone" / "forecast_event_embeddings.npz"
)
DEFAULT_ENCODER_MODEL = Path(
    r"C:\Users\Administrator\.cache\modelscope\hub\models\BAAI\bge-base-en-v1.5"
)
DEFAULT_TOPIC_HS_MAP = {
    "Automaker": "8703",
    "Nexperia": "3818",
}

GDELT_FEATURE_COLUMNS = [
    "GoldsteinScale_AbsMax",
    "NumMentions_Sum",
    "NumSources_Sum",
    "NumArticles_Sum",
    "AvgTone_AbsMax",
]


@dataclass
class FeatureStandardizer:
    """按列对结构化 GDELT 特征做 z-score 标准化。

    Attributes
    ----------
    mean : numpy.ndarray
        每个结构化特征列的均值，形状为 ``(num_features,)``。
    std : numpy.ndarray
        每个结构化特征列的标准差，形状为 ``(num_features,)``。

    Notes
    -----
    核心逻辑是只在训练/当前输入数据上估计均值和标准差，并将极小标准差
    替换为 1，避免除零。该对象用于把 GDELT 聚合指标映射到更稳定的数值
    尺度，便于后续 MLP 学习。
    """

    mean: np.ndarray
    std: np.ndarray

    @classmethod
    def fit(cls, values: np.ndarray) -> "FeatureStandardizer":
        """从二维特征矩阵估计标准化参数。

        Parameters
        ----------
        values : numpy.ndarray
            待标准化的结构化特征矩阵，形状为 ``(n_samples, n_features)``。

        Returns
        -------
        FeatureStandardizer
            包含列均值和列标准差的标准化器。

        Raises
        ------
        ValueError
            本函数当前不显式抛出异常；如果输入不是可广播的数值数组，
            NumPy 会在均值或标准差计算时抛出相应异常。
        """
        # 计算并保存 mean，供后续逻辑复用。
        mean = values.mean(axis=0)
        # 计算并保存 std，供后续逻辑复用。
        std = values.std(axis=0)
        # numpy.where(condition[, x, y])：
        #   如果 condition 为 True，则返回 x；否则返回 y。
        std = np.where(std < 1e-12, 1.0, std)
        # 返回当前函数的处理结果。
        return cls(mean=mean.astype(np.float32), std=std.astype(np.float32))

    def transform(self, values: np.ndarray) -> np.ndarray:
        """应用已估计的 z-score 标准化。

        Parameters
        ----------
        values : numpy.ndarray
            待转换的特征矩阵，形状应与拟合时的特征维度一致。

        Returns
        -------
        numpy.ndarray
            标准化后的 ``float32`` 特征矩阵。

        Raises
        ------
        ValueError
            本函数不显式抛出异常；如果输入维度无法与 ``mean`` 或 ``std``
            广播，NumPy 会抛出广播相关异常。
        """
        # 返回当前函数的处理结果。
        return ((values - self.mean) / self.std).astype(np.float32)

    def to_dict(self) -> Dict[str, List[float]]:
        """将标准化参数转换成可 JSON 序列化的字典。

        Returns
        -------
        dict of str to list of float
            包含 ``mean`` 和 ``std`` 两个键的字典。

        Raises
        ------
        None
            本函数不抛出自定义异常。
        """
        # 返回当前函数的处理结果。
        return {
            "mean": self.mean.astype(float).tolist(),
            "std": self.std.astype(float).tolist(),
        }


@dataclass
@dataclass
class RepresentationBatch:
    """新闻初始化阶段的内存输出。

    Attributes
    ----------
    news_metadata : pandas.DataFrame
        新闻级元数据，包含新闻编号、Topic、日期等信息。
    daily_metadata : pandas.DataFrame
        按日期和 Topic 聚合后的日级元数据。
    gdelt_features : numpy.ndarray
        标准化后的 GDELT 结构化特征矩阵。
    text_embeddings : numpy.ndarray
        冻结文本编码器得到的新闻正文语义向量。
    news_representations : numpy.ndarray
        新闻级情绪感知表示 ``r_{t,u}``。
    daily_representations : numpy.ndarray
        日级平均表示，用于后续图节点风险注入。
    gdelt_standardizer : FeatureStandardizer
        结构化特征标准化器。

    Notes
    -----
    该容器不负责计算，只负责携带各阶段结果，避免后续图传播和训练函数
    重复解析 CSV 或重新编码文本。
    """

    news_metadata: pd.DataFrame
    daily_metadata: pd.DataFrame
    gdelt_features: np.ndarray
    text_embeddings: np.ndarray
    news_representations: np.ndarray
    daily_representations: np.ndarray
    gdelt_standardizer: FeatureStandardizer


@dataclass
@dataclass
class IndustryGraph:
    """与当前产业链图对齐的 HS4 产品图。

    Attributes
    ----------
    node_metadata : pandas.DataFrame
        当前边表涉及的节点元数据，包含 ``node_index``、``hs_code_4d`` 和
        ``igpc_value`` 等列。
    node_codes : list of str
        与邻接矩阵行列顺序一致的 HS4 节点编码列表。
    normalized_adjacency : numpy.ndarray
        加自环后的归一化邻接矩阵，形状为 ``(N, N)``。
    adjacency : numpy.ndarray
        未加自环的有向加权邻接矩阵，布局为 ``adjacency[target, source]``。
    edge_count : int
        按传播方向展开后的有向边数量。
    graph_direction : str
        图传播方向配置，例如 ``"upstream-to-downstream"``。

    Notes
    -----
    节点集合来自当前产业链边表端点，而不是节点表中的全部产品，保证传播
    模块只处理当前研究图。
    """

    node_metadata: pd.DataFrame
    node_codes: List[str]
    normalized_adjacency: np.ndarray
    adjacency: np.ndarray
    edge_count: int
    graph_direction: str


@dataclass
@dataclass
class GraphPropagationBatch:
    """产业链图传播阶段的内存输出。

    Attributes
    ----------
    graph : IndustryGraph
        已加载并归一化的产业链图。
    time_metadata : pandas.DataFrame
        每个传播时间步的日期和直接新闻节点数量。
    node_risk_sequence : numpy.ndarray
        节点对齐的新闻风险序列，形状为 ``(T, N, d_r)``。
    node_risk_mask : numpy.ndarray
        是否存在直接新闻冲击的掩码，形状为 ``(T, N, 1)``。
    initial_node_states : numpy.ndarray
        由 IGPC 初始化得到的节点状态 ``h_0``。
    propagated_node_states : numpy.ndarray
        每个时间步传播后的节点状态序列。
    final_node_states : numpy.ndarray
        最后一个时间步的节点状态。
    """

    graph: IndustryGraph
    time_metadata: pd.DataFrame
    node_risk_sequence: np.ndarray
    node_risk_mask: np.ndarray
    initial_node_states: np.ndarray
    propagated_node_states: np.ndarray
    final_node_states: np.ndarray


@dataclass
@dataclass
class SupervisedTrainingData:
    """监督训练阶段使用的数据和元数据。

    Attributes
    ----------
    graph : IndustryGraph
        当前产业链图。
    topic_hs_map : dict of str to str
        Topic 到 HS4 节点编码的映射。
    news_metadata : pandas.DataFrame
        输入新闻的元数据。
    time_metadata : pandas.DataFrame
        全局日期并集时间轴及 train/val/test 标记。
    gdelt_features : numpy.ndarray
        标准化后的新闻结构化特征。
    text_embeddings : numpy.ndarray
        冻结文本模型编码得到的新闻文本向量。
    news_time_index : numpy.ndarray
        每条输入新闻对应的全局时间索引。
    news_node_index : numpy.ndarray
        每条输入新闻映射到的图节点索引。
    label_values : numpy.ndarray
        ``daily_top_news.csv`` 中 ArticleScore 对齐后的标签矩阵。
    label_mask : numpy.ndarray
        标签是否存在的掩码矩阵。
    supervision_weights : numpy.ndarray
        按论文分支逻辑构造的监督损失权重矩阵。
    split_by_time : dict of str to numpy.ndarray
        train/val/test 的时间步布尔掩码。
    gdelt_standardizer : FeatureStandardizer
        训练数据使用的 GDELT 标准化器。
    """

    graph: IndustryGraph
    topic_hs_map: Dict[str, str]
    news_metadata: pd.DataFrame
    time_metadata: pd.DataFrame
    gdelt_features: np.ndarray
    text_embeddings: np.ndarray
    news_time_index: np.ndarray
    news_node_index: np.ndarray
    label_values: np.ndarray
    label_mask: np.ndarray
    supervision_weights: np.ndarray
    split_by_time: Dict[str, np.ndarray]
    gdelt_standardizer: FeatureStandardizer


@dataclass
@dataclass
class AlignmentBatch:
    """LLM semantic pairs aligned to structured time-node positions."""

    pair_metadata: pd.DataFrame
    semantic_embeddings: np.ndarray
    time_index: np.ndarray
    node_index: np.ndarray


def require_columns(df: pd.DataFrame, columns: Iterable[str], name: str) -> None:
    """检查 DataFrame 是否包含指定列。

    Parameters
    ----------
    df : pandas.DataFrame
        待检查的数据表。
    columns : iterable of str
        必须存在的列名集合。
    name : str
        数据源名称，通常为文件路径，用于错误信息。

    Returns
    -------
    None
        所有列存在时不返回值。

    Raises
    ------
    ValueError
        当 ``df`` 缺少任意必需列时抛出，并列出缺失列名。
    """
    # 计算并保存 missing，供后续逻辑复用。
    missing = [col for col in columns if col not in df.columns]
    # 根据条件 missing 选择对应处理分支。
    if missing:
        # 抛出异常，阻止无效数据继续向下游传播。
        raise ValueError(f"{name} is missing required columns: {missing}")


def normalize_hs4(value: Any) -> str:
    """将任意 HS 编码值规范化为 4 位字符串。

    Parameters
    ----------
    value : Any
        原始 HS 编码，可能是字符串、整数、浮点数或缺失值。

    Returns
    -------
    str
        规范化后的 4 位 HS4 编码；缺失或空值返回空字符串。

    Raises
    ------
    None
        本函数不抛出自定义异常。

    Notes
    -----
    核心逻辑是去除非数字字符并补齐前导零，保证节点表、边表和 Topic 映射
    使用同一种 HS4 表示方式。
    """
    # 根据条件 pd.isna(value) 选择对应处理分支。
    if pd.isna(value):
        # 返回当前分支或函数的处理结果。
        return ""
    # 计算并保存 text，供后续逻辑复用。
    text = str(value).strip()
    # 根据条件 not text 选择对应处理分支。
    if not text:
        # 返回当前分支或函数的处理结果。
        return ""
    # 尝试执行可能失败的逻辑，并在异常分支中兜底。
    try:
        # 计算并保存 text，供当前分支后续使用。
        text = str(int(float(text)))
    except ValueError:
        # 计算并保存 text，供当前分支后续使用。
        text = "".join(ch for ch in text if ch.isdigit())
    # 返回当前函数的处理结果。
    return text.zfill(4)[-4:] if text else ""


def parse_topic_hs_map(items: Sequence[str]) -> Dict[str, str]:
    """解析命令行传入的 Topic 到 HS4 映射。

    Parameters
    ----------
    items : sequence of str
        形如 ``"Topic=HS4"`` 的映射列表。传入值会覆盖默认映射。

    Returns
    -------
    dict of str to str
        Topic 名称到 4 位 HS4 编码的字典。

    Raises
    ------
    ValueError
        当某个映射项不包含等号，或 Topic/HS4 为空时抛出。
    """
    # 计算并保存 mapping，供后续逻辑复用。
    mapping = dict(DEFAULT_TOPIC_HS_MAP)
    # 遍历 items，逐项执行处理逻辑。
    for item in items:
        # 判断 "=" not in item，处理对应情况。
        if "=" not in item:
            # 抛出异常，阻止无效数据继续向下游传播。
            raise ValueError(f"Invalid topic-HS4 mapping {item!r}. Use Topic=HS4.")
        # 计算并保存 topic、code，供当前分支后续使用。
        topic, code = item.split("=", 1)
        # 计算并保存 topic，供当前分支后续使用。
        topic = topic.strip()
        # 计算并保存 code，供当前分支后续使用。
        code = normalize_hs4(code)
        # 判断 not topic or not code，处理对应情况。
        if not topic or not code:
            # 抛出异常，阻止无效数据继续向下游传播。
            raise ValueError(f"Invalid topic-HS4 mapping {item!r}. Use Topic=HS4.")
        # 计算并保存 mapping，供当前分支后续使用。
        mapping[topic] = code
    # 返回当前函数的处理结果。
    return mapping


def parse_hidden_sizes(text: str) -> Tuple[int, ...]:
    """解析逗号分隔的 MLP 隐藏层维度。

    Parameters
    ----------
    text : str
        例如 ``"256,64"`` 的维度配置；空字符串表示无隐藏层。

    Returns
    -------
    tuple of int
        隐藏层维度元组。

    Raises
    ------
    ValueError
        当任意维度不是正整数时抛出。
    """
    # 计算并保存 text，供后续逻辑复用。
    text = str(text or "").strip()
    # 根据条件 not text 选择对应处理分支。
    if not text:
        # 返回当前分支或函数的处理结果。
        return ()
    # 计算并保存 values，供后续逻辑复用。
    values = tuple(int(part.strip()) for part in text.split(",") if part.strip())
    # 根据条件 any(value <= 0 for value in values) 选择对应处理分支。
    if any(value <= 0 for value in values):
        # 抛出异常，阻止无效数据继续向下游传播。
        raise ValueError("hidden sizes must be positive integers.")
    # 返回当前函数的处理结果。
    return values


def parse_group_cols(text: str) -> List[str]:
    """解析用于日级聚合的分组列。

    Parameters
    ----------
    text : str
        逗号分隔的列名，例如 ``"topic,date_yyyymmdd"``。

    Returns
    -------
    list of str
        清理空白后的列名列表。

    Raises
    ------
    ValueError
        当没有解析出任何列名时抛出。
    """
    # 计算并保存 columns，供后续逻辑复用。
    columns = [part.strip() for part in str(text).split(",") if part.strip()]
    # 根据条件 not columns 选择对应处理分支。
    if not columns:
        # 抛出异常，阻止无效数据继续向下游传播。
        raise ValueError("--daily-group-cols must contain at least one column.")
    # 返回当前函数的处理结果。
    return columns


def ensure_dir(path: Path) -> None:
    """确保目标目录存在。

    Parameters
    ----------
    path : pathlib.Path
        需要创建的目录路径。

    Returns
    -------
    None
        目录存在或创建成功时不返回值。

    Raises
    ------
    OSError
        当文件系统无法创建目录时由 ``Path.mkdir`` 抛出。
    """
    # 调用 path.mkdir，执行当前步骤的副作用。
    path.mkdir(parents=True, exist_ok=True)


def set_seed(seed: int) -> None:
    """设置 Python、NumPy 和 PyTorch 的随机种子。

    Parameters
    ----------
    seed : int
        随机种子。

    Returns
    -------
    None
        设置完成后不返回值。

    Raises
    ------
    None
        本函数会忽略未安装 PyTorch 的情况。
    """
    # 调用 random.seed，执行当前步骤的副作用。
    random.seed(seed)
    # 调用 np.random.seed，执行当前步骤的副作用。
    np.random.seed(seed)
    # 尝试执行可能失败的逻辑，并在异常分支中兜底。
    try:
        # 导入当前分支需要的模块，避免提前加载额外依赖。
        import torch

        # 执行 manual_seed，完成当前步骤的副作用。
        torch.manual_seed(seed)
        # 判断 torch.cuda.is_available()，处理对应情况。
        if torch.cuda.is_available():
            # 执行 manual_seed_all，完成当前步骤的副作用。
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        # 忽略该异常或空分支，保持流程继续执行。
        pass


def resolve_device(device: str) -> str:
    """解析训练或编码设备。

    Parameters
    ----------
    device : str
        设备字符串，支持 ``"auto"``、``"cpu"``、``"cuda"`` 等 PyTorch
        可识别设备。

    Returns
    -------
    str
        实际使用的设备字符串。

    Raises
    ------
    ImportError
        当 ``device="auto"`` 且 PyTorch 不可导入时抛出。
    """
    # 计算并保存 device，供后续逻辑复用。
    device = str(device).lower()
    # 根据条件 device != "auto" 选择对应处理分支。
    if device != "auto":
        # 返回当前分支或函数的处理结果。
        return device
    # 延迟导入 torch，避免未使用阶段强制加载依赖。
    import torch

    # 返回当前函数的处理结果。
    return "cuda" if torch.cuda.is_available() else "cpu"

def mean_pool(last_hidden_state: Any, attention_mask: Any) -> Any:
    """对 Transformer token 表示执行 attention mask 加权平均池化。

    Parameters
    ----------
    last_hidden_state : torch.Tensor
        Transformer 最后一层隐状态，形状为 ``(batch, seq_len, hidden_dim)``。
    attention_mask : torch.Tensor
        attention mask，形状为 ``(batch, seq_len)``。

    Returns
    -------
    torch.Tensor
        平均池化后的句向量，形状为 ``(batch, hidden_dim)``。

    Raises
    ------
    RuntimeError
        当输入张量维度不兼容时由 PyTorch 抛出。
    """
    # 延迟导入 torch，避免未使用阶段强制加载依赖。
    import torch

    # 计算并保存 mask，供后续逻辑复用。
    mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
    # 计算并保存 summed，供后续逻辑复用。
    summed = torch.sum(last_hidden_state * mask, dim=1)
    # 计算并保存 counts，供后续逻辑复用。
    counts = torch.clamp(mask.sum(dim=1), min=1e-9)
    # 返回当前函数的处理结果。
    return summed / counts


def encode_texts(texts: Sequence[str], args: argparse.Namespace) -> np.ndarray:
    """使用冻结 Transformer 文本编码器生成新闻语义向量。

    Parameters
    ----------
    texts : sequence of str
        待编码新闻文本。
    args : argparse.Namespace
        包含 ``encoder_model``、``encoder_device``、``encoder_max_length``、
        ``embedding_batch_size``、``pooling`` 和归一化配置的参数对象。

    Returns
    -------
    numpy.ndarray
        文本嵌入矩阵，形状为 ``(len(texts), hidden_dim)``。

    Raises
    ------
    ValueError
        当 ``args.pooling`` 不是支持的池化方式时抛出。
    OSError
        当本地模型路径或 tokenizer/model 文件不可读取时由 transformers 抛出。
    ImportError
        当 PyTorch 或 transformers 未安装时抛出。

    Notes
    -----
    该函数在 ``torch.no_grad`` 下运行，不训练文本模型，只为后续结构化管线
    提供固定语义输入。
    """
    # 延迟导入 torch，避免未使用阶段强制加载依赖。
    import torch
    # 延迟导入 torch.nn.functional，避免未使用阶段强制加载依赖。
    import torch.nn.functional as functional
    # 从 transformers 导入 AutoModel, AutoTokenizer，供当前函数内部使用。
    from transformers import AutoModel, AutoTokenizer

    # 计算并保存 device，供后续逻辑复用。
    device = resolve_device(args.encoder_device)
    # 计算并保存 tokenizer，供后续逻辑复用。
    tokenizer = AutoTokenizer.from_pretrained(args.encoder_model)
    # 计算并保存 model，供后续逻辑复用。
    model = AutoModel.from_pretrained(args.encoder_model)
    # 调用 model.to，执行当前步骤的副作用。
    model.to(device)
    # 调用 model.eval，执行当前步骤的副作用。
    model.eval()

    # 初始化带类型标注的 embeddings，明确后续存储内容。
    embeddings: List[np.ndarray] = []
    # 使用上下文管理器处理资源，确保结束时自动释放。
    with torch.no_grad():
        # 遍历 start in range(0, len(texts), args.embedding_batch_si...，逐项处理数据。
        for start in range(0, len(texts), args.embedding_batch_size):
            # 计算并保存 batch，供当前分支后续使用。
            batch = list(texts[start : start + args.embedding_batch_size])
            # 计算并保存 encoded，供当前分支后续使用。
            encoded = tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=args.encoder_max_length,
                return_tensors="pt",
            )
            # 计算并保存 encoded，供当前分支后续使用。
            encoded = {key: value.to(device) for key, value in encoded.items()}
            # 计算并保存 output，供当前分支后续使用。
            output = model(**encoded)
            # 判断 args.pooling == "cls"，处理对应情况。
            if args.pooling == "cls":
                # 计算并保存 pooled，供当前分支后续使用。
                pooled = output.last_hidden_state[:, 0]
            # 判断 f args.pooling == "mean"，处理对应情况。
            elif args.pooling == "mean":
                # 计算并保存 pooled，供当前分支后续使用。
                pooled = mean_pool(output.last_hidden_state, encoded["attention_mask"])
            else:
                # 抛出异常，阻止无效数据继续向下游传播。
                raise ValueError(f"Unsupported pooling mode: {args.pooling}")
            # 判断 args.normalize_text_embeddings，处理对应情况。
            if args.normalize_text_embeddings:
                # 计算并保存 pooled，供当前分支后续使用。
                pooled = functional.normalize(pooled, p=2, dim=1)
            # 执行 append，完成当前步骤的副作用。
            embeddings.append(pooled.cpu().numpy())
    # 返回当前函数的处理结果。
    return np.vstack(embeddings).astype(np.float32)

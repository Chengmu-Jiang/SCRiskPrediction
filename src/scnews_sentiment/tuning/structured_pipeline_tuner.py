#!/usr/bin/env python3
"""Run stage-wise hyperparameter tuning for the Hydra structured pipeline.

本脚本按照“稳定性参数 -> 模型容量 -> 图传播参数”的顺序执行小规模网格搜索。
每个 trial 会写入独立目录，并在根目录持续更新汇总表，方便中断后查看已完成
结果。

Notes
-----
核心逻辑如下：

1. 为每个阶段生成候选参数组合。
2. 当前阶段基于上一阶段的最优参数继续搜索。
3. 每个 trial 调用一次 ``src/train.py``，并将输出目录自动命名。
4. 读取 TimeSeriesSplit 聚合指标，按指定 split 和 metric 选择最优 trial。
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


DEFAULT_OUTPUT_ROOT = Path("outputs") / "structured_pipeline_tuning"
DEFAULT_PIPELINE_SCRIPT = Path("src") / "train.py"

LOWER_IS_BETTER = {
    "mse",
    "rmse",
    "mae",
    "median_ae",
    "p90_ae",
    "max_ae",
}
HIGHER_IS_BETTER = {"r2", "pearson", "spearman"}

PARAM_SLUG_NAMES = {
    "align_temperature": "at",
    "align_weight": "aw",
    "dropout": "do",
    "early_stopping_patience": "esp",
    "eval_hidden_sizes": "eh",
    "graph_direction": "gd",
    "graph_dropout": "gdo",
    "graph_recovery_hidden_sizes": "grh",
    "graph_risk_hidden_sizes": "grh2",
    "graph_state_dim": "gs",
    "lr": "lr",
    "projection_dim": "pd",
    "representation_dim": "rd",
    "sent_hidden_sizes": "sh",
    "smooth_weight": "sw",
    "diffusion_mode": "dm",
    "weight_decay": "wd",
}


def grid_candidates(grid: Dict[str, Sequence[Any]]) -> List[Dict[str, Any]]:
    """Expand a parameter grid into a list of candidate dictionaries.

    Parameters
    ----------
    grid : dict of str to sequence
        网格定义。键是参数名，值是该参数要尝试的候选取值序列。

    Returns
    -------
    list of dict
        展开后的笛卡尔积组合。每个字典是一组可传给
        ``src/train.py`` 的参数覆盖项。

    Raises
    ------
    TypeError
        当 ``grid`` 的值不是可迭代序列时，``itertools.product`` 可能抛出该异常。

    Notes
    -----
    该函数只负责组合生成，不判断参数是否对模型有效；合法性由
    ``src/train.py`` 的 argparse 校验承担。
    """
    keys = list(grid)
    values = [grid[key] for key in keys]
    return [dict(zip(keys, combo)) for combo in itertools.product(*values)]


def stability_candidates() -> List[Dict[str, Any]]:
    """Build the first-stage training stability search space.

    Returns
    -------
    list of dict
        稳定性阶段候选参数列表，包含 baseline 和学习率、dropout、
        weight decay、early stopping patience 的小规模网格。

    Raises
    ------
    TypeError
        当内部网格配置不是可迭代对象时，可能由 ``grid_candidates`` 抛出。

    Notes
    -----
    这一阶段优先解决训练是否稳定和是否过拟合的问题，因此先调优化器与正则化
    参数，而不是直接改变模型结构。
    """
    return [{}] + grid_candidates(
        {
            "lr": [3e-4, 1e-3],
            "dropout": [0.1, 0.2],
            "weight_decay": [1e-5, 1e-4],
            "early_stopping_patience": [10, 20],
        }
    )


def capacity_candidates() -> List[Dict[str, Any]]:
    """Build the second-stage model capacity search space.

    Returns
    -------
    list of dict
        容量阶段候选参数列表。每个候选同时调整新闻表示维度、图状态维度、
        句子编码 MLP 和最终评价 MLP 的规模。

    Raises
    ------
    None
        本函数不主动抛出自定义异常。

    Notes
    -----
    容量阶段默认继承稳定性阶段的最优训练参数，只比较不同模型规模对验证集
    表现的影响。
    """
    return [
        {},
        {
            "projection_dim": 64,
            "representation_dim": 64,
            "graph_state_dim": 64,
            "sent_hidden_sizes": "128",
            "eval_hidden_sizes": "64",
        },
        {
            "projection_dim": 256,
            "representation_dim": 128,
            "graph_state_dim": 128,
            "sent_hidden_sizes": "256,128",
            "eval_hidden_sizes": "128",
        },
        {
            "projection_dim": 128,
            "representation_dim": 256,
            "graph_state_dim": 128,
            "sent_hidden_sizes": "256,128",
            "eval_hidden_sizes": "128",
        },
        {
            "projection_dim": 128,
            "representation_dim": 128,
            "graph_state_dim": 256,
            "sent_hidden_sizes": "256",
            "eval_hidden_sizes": "256,64",
        },
        {
            "projection_dim": 64,
            "representation_dim": 128,
            "graph_state_dim": 128,
            "sent_hidden_sizes": "128",
            "eval_hidden_sizes": "128,64",
        },
    ]


def graph_candidates() -> List[Dict[str, Any]]:
    """Build the third-stage graph propagation search space.

    Returns
    -------
    list of dict
        图传播阶段候选参数列表，包括图平滑权重、语义对齐权重、对齐温度、
        扩散模式和图方向。

    Raises
    ------
    None
        本函数不主动抛出自定义异常。

    Notes
    -----
    图传播参数的解释性较强，放在第三阶段可以在较稳定的训练配置和模型容量
    基础上观察产业链传播机制本身的贡献。
    """
    return [
        {},
        {"smooth_weight": 0.0},
        {"smooth_weight": 0.001},
        {"smooth_weight": 0.05},
        {"align_weight": 0.0},
        {"align_weight": 0.03},
        {"align_weight": 0.3},
        {"align_temperature": 0.1},
        {"diffusion_mode": "direct"},
        {"graph_direction": "bidirectional"},
        {"graph_direction": "downstream-to-upstream"},
    ]


STAGE_BUILDERS = {
    "stability": stability_candidates,
    "capacity": capacity_candidates,
    "graph": graph_candidates,
}


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """Parse command-line arguments for the tuning script.

    Parameters
    ----------
    argv : sequence of str, optional
        待解析的命令行参数。为 ``None`` 时使用真实的 ``sys.argv``。

    Returns
    -------
    argparse.Namespace
        解析后的配置对象，包含调参阶段、输出目录、选择指标、dry-run 开关和
        透传给 ``src/train.py`` 的额外参数。

    Raises
    ------
    ValueError
        当 ``--max-trials-per-stage`` 非空且不为正数时抛出。
    SystemExit
        当 argparse 解析失败或用户请求 ``--help`` 时由 argparse 抛出。

    Notes
    -----
    ``--pipeline-extra-args`` 使用 ``argparse.REMAINDER`` 接收剩余参数，因此它
    应放在命令末尾，用来把模型脚本自己的参数原样追加到每个 trial。
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stages",
        default="stability,capacity,graph",
        help="Comma-separated stages: stability,capacity,graph.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="Root directory for this tuning run.",
    )
    parser.add_argument(
        "--pipeline-script",
        type=Path,
        default=DEFAULT_PIPELINE_SCRIPT,
        help="Path to the Hydra training entrypoint.",
    )
    parser.add_argument(
        "--python-executable",
        default=sys.executable,
        help="Python executable used to run the training entrypoint.",
    )
    parser.add_argument(
        "--selection-split",
        choices=["val", "test"],
        default="val",
        help="Split used to select the best trial.",
    )
    parser.add_argument(
        "--selection-metric",
        default="rmse",
        help="Metric used to select the best trial, usually rmse or mae.",
    )
    parser.add_argument(
        "--direction",
        choices=["auto", "minimize", "maximize"],
        default="auto",
        help="Optimization direction. auto minimizes errors and maximizes correlations/R2.",
    )
    parser.add_argument(
        "--max-trials-per-stage",
        type=int,
        default=None,
        help="Optional cap for quick smoke tuning.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Write the tuning plan without launching training jobs.",
    )
    parser.add_argument(
        "--pipeline-extra-args",
        nargs=argparse.REMAINDER,
        default=[],
        help="Extra Hydra overrides appended to every training call.",
    )
    args = parser.parse_args(argv)
    if args.max_trials_per_stage is not None and args.max_trials_per_stage <= 0:
        raise ValueError("--max-trials-per-stage must be positive when provided.")
    return args


def selected_stages(raw_stages: str) -> List[str]:
    """Normalize and validate the requested tuning stages.

    Parameters
    ----------
    raw_stages : str
        逗号分隔的阶段名称，例如 ``"stability,capacity,graph"``。

    Returns
    -------
    list of str
        去除空白并校验后的阶段名称列表，顺序与输入一致。

    Raises
    ------
    ValueError
        当输入阶段不在 ``STAGE_BUILDERS`` 支持范围内时抛出。

    Notes
    -----
    阶段顺序会影响调参结果，因为后续阶段会继承前一阶段的最优参数。
    """
    stages = [item.strip() for item in raw_stages.split(",") if item.strip()]
    unknown = sorted(set(stages) - set(STAGE_BUILDERS))
    if unknown:
        raise ValueError(f"Unknown tuning stage(s): {unknown}")
    return stages


def metric_direction(metric_name: str, direction: str) -> str:
    """Resolve whether a metric should be minimized or maximized.

    Parameters
    ----------
    metric_name : str
        用于选择最优 trial 的指标名称，例如 ``"rmse"``、``"mae"`` 或
        ``"r2"``。
    direction : {"auto", "minimize", "maximize"}
        用户指定的优化方向。``"auto"`` 会根据指标名称自动判断。

    Returns
    -------
    str
        ``"minimize"`` 或 ``"maximize"``。

    Raises
    ------
    None
        本函数不主动抛出自定义异常。

    Notes
    -----
    误差类指标默认越小越好；相关性和 R² 默认越大越好。未知指标保守地按
    越小越好处理。
    """
    if direction != "auto":
        return direction
    metric_key = metric_name.lower()
    if metric_key in HIGHER_IS_BETTER:
        return "maximize"
    if metric_key in LOWER_IS_BETTER:
        return "minimize"
    return "minimize"


def is_better(score: Optional[float], best_score: Optional[float], direction: str) -> bool:
    """Compare a trial score against the current best score.

    Parameters
    ----------
    score : float or None
        当前 trial 的选择指标分数。``None`` 表示该 trial 无法参与比较。
    best_score : float or None
        当前阶段或全局已知的最优分数。
    direction : {"minimize", "maximize"}
        比较方向。

    Returns
    -------
    bool
        如果当前分数优于历史最优分数，返回 ``True``；否则返回 ``False``。

    Raises
    ------
    None
        本函数不主动抛出自定义异常。

    Notes
    -----
    该函数把 ``None`` score 视为不可用结果，避免失败 trial 被误选为最优。
    """
    if score is None:
        return False
    if best_score is None:
        return True
    if direction == "maximize":
        return score > best_score
    return score < best_score


def hydra_value(value: Any) -> str:
    """Serialize a Python value for Hydra command-line overrides."""
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def cli_flag(param_name: str) -> str:
    """Convert an argparse destination name to a command-line flag.

    Parameters
    ----------
    param_name : str
        Python 风格参数名，例如 ``"early_stopping_patience"``。

    Returns
    -------
    str
        命令行参数名，例如 ``"--early-stopping-patience"``。

    Raises
    ------
    None
        本函数不主动抛出自定义异常。

    Notes
    -----
    调参脚本内部统一使用 argparse 的 destination 名称，真正执行 trial 时再
    转换成 CLI flag。
    """
    return "--" + param_name.replace("_", "-")


def params_to_cli_args(params: Dict[str, Any]) -> List[str]:
    """Serialize parameter overrides into command-line arguments.

    Parameters
    ----------
    params : dict of str to Any
        参数覆盖字典。键为 argparse destination 名称，值为要传给模型脚本的
        参数值。

    Returns
    -------
    list of str
        可直接追加到 ``subprocess.run`` 命令中的参数列表。

    Raises
    ------
    None
        本函数不主动抛出自定义异常。

    Notes
    -----
    ``None`` 值会被跳过；布尔值只在为 ``True`` 时写入 flag，适合处理
    ``store_true`` 类型参数。
    """
    return [
        f"{key}={hydra_value(params[key])}"
        for key in sorted(params)
        if params[key] is not None
    ]


def value_slug(value: Any) -> str:
    """Convert a parameter value into a filesystem-friendly slug.

    Parameters
    ----------
    value : Any
        参数值，通常是数值、字符串或枚举值。

    Returns
    -------
    str
        适合放入目录名的短文本。

    Raises
    ------
    None
        本函数不主动抛出自定义异常。

    Notes
    -----
    为了让 Windows 路径更稳妥，函数会替换小数点、逗号、路径分隔符等字符，
    并截断过长文本。
    """
    if isinstance(value, float):
        text = f"{value:.6g}"
    else:
        text = str(value)
    replacements = {
        "-": "m",
        ".": "p",
        ",": "-",
        " ": "",
        "_": "-",
        "/": "-",
        "\\": "-",
        ":": "-",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text[:30]


def trial_hash(params: Dict[str, Any]) -> str:
    """Create a short stable hash for a trial parameter set.

    Parameters
    ----------
    params : dict of str to Any
        trial 的完整有效参数集合。

    Returns
    -------
    str
        8 位 SHA1 短哈希，用于避免目录名冲突。

    Raises
    ------
    TypeError
        当 ``params`` 中包含 JSON 无法序列化的值时，由 ``json.dumps`` 抛出。

    Notes
    -----
    目录名会包含人可读参数缩写和短哈希；短哈希基于完整参数而不是变更参数，
    因此能区分继承了不同前序最优配置的 trial。
    """
    payload = json.dumps(params, sort_keys=True, ensure_ascii=True)
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:8]


def trial_dir_name(
    stage: str,
    trial_index: int,
    changed_params: Dict[str, Any],
    params: Dict[str, Any],
) -> str:
    """Build the directory name for a tuning trial.

    Parameters
    ----------
    stage : str
        当前调参阶段名称。
    trial_index : int
        全局递增 trial 编号。
    changed_params : dict of str to Any
        当前 trial 相对于阶段基准新增或覆盖的参数。
    params : dict of str to Any
        当前 trial 的完整有效参数。

    Returns
    -------
    str
        包含阶段、trial 编号、参数摘要和短哈希的目录名。

    Raises
    ------
    TypeError
        当完整参数无法 JSON 序列化时，由 ``trial_hash`` 抛出。

    Notes
    -----
    空的 ``changed_params`` 会被命名为 ``baseline``，用于表示该阶段继承上一
    阶段最优参数而不额外修改。
    """
    if changed_params:
        parts = []
        for key in sorted(changed_params):
            short = PARAM_SLUG_NAMES.get(key, key)
            parts.append(f"{short}_{value_slug(changed_params[key])}")
        slug = "__".join(parts)
    else:
        slug = "baseline"
    slug = slug[:120].rstrip("_")
    return f"{stage}__trial_{trial_index:03d}__{slug}__{trial_hash(params)}"


def ensure_dir(path: Path) -> None:
    """Create a directory and its parents when missing.

    Parameters
    ----------
    path : pathlib.Path
        需要确保存在的目录路径。

    Returns
    -------
    None
        本函数只产生文件系统副作用。

    Raises
    ------
    OSError
        当目录无法创建或权限不足时，由 ``Path.mkdir`` 抛出。

    Notes
    -----
    所有 trial 输出目录和调参根目录都会通过该函数创建。
    """
    path.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, payload: Any) -> None:
    """Write a JSON payload using UTF-8 encoding.

    Parameters
    ----------
    path : pathlib.Path
        输出 JSON 文件路径。
    payload : Any
        可 JSON 序列化的对象。

    Returns
    -------
    None
        本函数只负责写文件。

    Raises
    ------
    TypeError
        当 ``payload`` 包含无法 JSON 序列化的对象时抛出。
    OSError
        当文件无法写入时抛出。

    Notes
    -----
    ``ensure_ascii=False`` 用于保留中文和路径文本，方便直接阅读调参结果。
    """
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def strip_pipeline_extra_args(extra_args: Sequence[str]) -> List[str]:
    """Normalize extra arguments forwarded to ``src/train.py``.

    Parameters
    ----------
    extra_args : sequence of str
        从 ``--pipeline-extra-args`` 捕获的剩余参数。

    Returns
    -------
    list of str
        去掉可选分隔符 ``"--"`` 后的透传参数。

    Raises
    ------
    None
        本函数不主动抛出自定义异常。

    Notes
    -----
    用户可以写 ``--pipeline-extra-args -- --encoder-device cpu``，函数会去掉
    第一个分隔符，只把真正属于模型脚本的参数传下去。
    """
    extra = list(extra_args)
    if extra and extra[0] == "--":
        extra = extra[1:]
    return extra


def run_trial(
    *,
    args: argparse.Namespace,
    run_root: Path,
    stage: str,
    trial_index: int,
    changed_params: Dict[str, Any],
    effective_params: Dict[str, Any],
) -> Dict[str, Any]:
    """Run one structured-pipeline tuning trial.

    Parameters
    ----------
    args : argparse.Namespace
        调参脚本的全局配置，包含 Python 解释器、模型脚本路径、选择指标和
        dry-run 开关。
    run_root : pathlib.Path
        当前调参批次的根目录。
    stage : str
        当前 trial 所属阶段。
    trial_index : int
        全局 trial 编号。
    changed_params : dict of str to Any
        当前 trial 在本阶段显式尝试的参数。
    effective_params : dict of str to Any
        当前 trial 实际传给 ``src/train.py`` 的完整参数集合。

    Returns
    -------
    dict
        trial 结果记录。包含状态、输出目录、选择指标分数、耗时、返回码和
        已抽取的关键指标。

    Raises
    ------
    OSError
        当 trial 目录、日志文件或 JSON 文件无法创建/写入时抛出。
    TypeError
        当参数或记录无法序列化为 JSON 时抛出。

    Notes
    -----
    核心逻辑是：先创建自动命名目录并写入参数和命令；如果不是 dry-run，就
    调用模型脚本，将 stdout/stderr 分别保存；训练结束后读取指标文件并把
    选择分数写回 ``trial_result.json``。
    """
    trial_dir = run_root / trial_dir_name(stage, trial_index, changed_params, effective_params)
    ensure_dir(trial_dir)
    write_json(trial_dir / "trial_params.json", effective_params)
    write_json(trial_dir / "changed_params.json", changed_params)

    pipeline_extra_args = strip_pipeline_extra_args(args.pipeline_extra_args)
    command = [
        args.python_executable,
        str(args.pipeline_script),
        *params_to_cli_args(effective_params),
        *pipeline_extra_args,
        f"output_dir={trial_dir}",
    ]
    write_json(trial_dir / "command.json", command)

    record: Dict[str, Any] = {
        "stage": stage,
        "trial_index": trial_index,
        "output_dir": str(trial_dir),
        "params_json": json.dumps(effective_params, sort_keys=True, ensure_ascii=False),
        "changed_params_json": json.dumps(changed_params, sort_keys=True, ensure_ascii=False),
        "status": "dry_run" if args.dry_run else "pending",
        "score": None,
        "selection_split": args.selection_split,
        "selection_metric": args.selection_metric,
        "duration_seconds": None,
    }

    if args.dry_run:
        write_json(trial_dir / "trial_result.json", record)
        return record

    started = time.time()
    with (trial_dir / "stdout.log").open("w", encoding="utf-8") as stdout_handle, (
        trial_dir / "stderr.log"
    ).open("w", encoding="utf-8") as stderr_handle:
        completed = subprocess.run(
            command,
            cwd=Path.cwd(),
            stdout=stdout_handle,
            stderr=stderr_handle,
            text=True,
            check=False,
        )
    duration = time.time() - started
    record["duration_seconds"] = round(duration, 3)
    record["returncode"] = completed.returncode
    if completed.returncode != 0:
        record["status"] = "failed"
        write_json(trial_dir / "trial_result.json", record)
        return record

    try:
        metrics_payload = load_metrics_payload(trial_dir)
        score = select_score(
            metrics_payload,
            split_name=args.selection_split,
            metric_name=args.selection_metric,
        )
        record["status"] = "completed"
        record["score"] = score
        add_selected_metrics(record, metrics_payload)
    except Exception as exc:  # noqa: BLE001 - record failures without hiding logs.
        record["status"] = "metrics_failed"
        record["metrics_error"] = str(exc)

    write_json(trial_dir / "trial_result.json", record)
    return record


def load_metrics_payload(trial_dir: Path) -> Dict[str, Any]:
    """Load the metrics JSON produced by a trial.

    Parameters
    ----------
    trial_dir : pathlib.Path
        单个 trial 的输出目录。

    Returns
    -------
    dict
        指标 JSON 内容。优先返回 TimeSeriesSplit 汇总指标；若不存在，则回退到
        单次监督训练指标。

    Raises
    ------
    FileNotFoundError
        当 trial 目录下没有可识别的指标 JSON 时抛出。
    json.JSONDecodeError
        当指标文件不是合法 JSON 时抛出。
    OSError
        当指标文件无法读取时抛出。

    Notes
    -----
    自动调参默认使用 ``time_series_cv_summary.json``，因为它包含跨折均值和
    标准差，更适合作为论文实验选择依据。
    """
    tscv_path = trial_dir / "time_series_cv_summary.json"
    supervised_path = trial_dir / "structured_supervised_metrics.json"
    if tscv_path.exists():
        with tscv_path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    if supervised_path.exists():
        with supervised_path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    raise FileNotFoundError(f"No metrics JSON found in {trial_dir}")


def select_score(
    metrics_payload: Dict[str, Any],
    *,
    split_name: str,
    metric_name: str,
) -> Optional[float]:
    """Extract the scalar score used to rank trials.

    Parameters
    ----------
    metrics_payload : dict
        从 trial 指标 JSON 读取出的指标内容。
    split_name : str
        用于选择最优 trial 的数据划分名称，例如 ``"val"`` 或 ``"test"``。
    metric_name : str
        用于选择最优 trial 的指标名称，例如 ``"rmse"``。

    Returns
    -------
    float or None
        选定 split 和 metric 的标量分数。TimeSeriesSplit 情况下返回跨折均值；
        无有效值时返回 ``None``。

    Raises
    ------
    KeyError
        当指标文件结构不是支持的 ``aggregate_metrics``、``folds`` 或
        ``splits`` 形态时抛出。
    ValueError
        当指标值无法转换为浮点数时抛出。

    Notes
    -----
    新版 ``src/train.py`` 会输出 ``aggregate_metrics``；为了兼容旧
    结果，本函数也支持从 ``folds`` 手动计算均值，或从单次 ``splits`` 中读取。
    """
    aggregate = metrics_payload.get("aggregate_metrics")
    if isinstance(aggregate, dict):
        value = (
            aggregate.get(split_name, {})
            .get(metric_name, {})
            .get("mean")
        )
        return None if value is None else float(value)

    folds = metrics_payload.get("folds")
    if isinstance(folds, list):
        column = f"{split_name}_{metric_name}"
        values = [
            float(row[column])
            for row in folds
            if isinstance(row, dict) and row.get(column) is not None
        ]
        return None if not values else sum(values) / len(values)

    splits = metrics_payload.get("splits")
    if isinstance(splits, dict):
        value = splits.get(split_name, {}).get(metric_name)
        return None if value is None else float(value)

    raise KeyError("Unsupported metrics payload shape.")


def add_selected_metrics(record: Dict[str, Any], metrics_payload: Dict[str, Any]) -> None:
    """Append commonly inspected metrics to a trial record.

    Parameters
    ----------
    record : dict
        需要就地更新的 trial 结果记录。
    metrics_payload : dict
        从 trial 指标 JSON 读取出的指标内容。

    Returns
    -------
    None
        本函数直接修改 ``record``。

    Raises
    ------
    None
        本函数不主动抛出自定义异常。

    Notes
    -----
    TimeSeriesSplit 输出会被展开为 ``val_rmse_mean``、``val_rmse_std`` 等列；
    单次训练输出会被展开为 ``train_rmse``、``val_rmse``、``test_rmse`` 等列。
    """
    metric_names = ["mse", "rmse", "mae", "r2", "pearson", "spearman", "median_ae", "p90_ae", "bias"]
    aggregate = metrics_payload.get("aggregate_metrics")
    if isinstance(aggregate, dict):
        for split_name in ["val", "test"]:
            for metric_name in metric_names:
                stats = aggregate.get(split_name, {}).get(metric_name, {})
                if isinstance(stats, dict):
                    record[f"{split_name}_{metric_name}_mean"] = stats.get("mean")
                    record[f"{split_name}_{metric_name}_std"] = stats.get("std")
        return

    splits = metrics_payload.get("splits")
    if isinstance(splits, dict):
        for split_name in ["train", "val", "test"]:
            for metric_name in metric_names:
                record[f"{split_name}_{metric_name}"] = splits.get(split_name, {}).get(metric_name)


def write_records_csv(path: Path, records: Sequence[Dict[str, Any]]) -> None:
    """Write trial records into a CSV summary table.

    Parameters
    ----------
    path : pathlib.Path
        输出 CSV 文件路径。
    records : sequence of dict
        trial 结果记录序列。

    Returns
    -------
    None
        本函数只负责写文件。

    Raises
    ------
    OSError
        当 CSV 文件无法写入时抛出。
    csv.Error
        当记录内容无法由 ``csv.DictWriter`` 正常写出时抛出。

    Notes
    -----
    基础字段固定放在前面，其他指标字段按字母序追加。这样即使不同 trial 的
    状态不同，汇总表也能包含所有已观察到的列。
    """
    base_fields = [
        "stage",
        "trial_index",
        "status",
        "score",
        "selection_split",
        "selection_metric",
        "duration_seconds",
        "returncode",
        "output_dir",
        "params_json",
        "changed_params_json",
    ]
    extra_fields = sorted({key for record in records for key in record} - set(base_fields))
    fieldnames = base_fields + extra_fields
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            writer.writerow(record)


def candidate_plan(
    stages: Sequence[str],
    max_trials_per_stage: Optional[int],
) -> Dict[str, List[Dict[str, Any]]]:
    """Build the complete tuning candidate plan.

    Parameters
    ----------
    stages : sequence of str
        要执行的阶段名称序列。
    max_trials_per_stage : int or None
        每个阶段最多保留的候选数量。为 ``None`` 时使用阶段完整候选空间。

    Returns
    -------
    dict of str to list of dict
        每个阶段对应的候选参数列表。

    Raises
    ------
    KeyError
        当 ``stages`` 包含未注册到 ``STAGE_BUILDERS`` 的阶段名时抛出。

    Notes
    -----
    该函数通常在 ``selected_stages`` 之后调用，因此正常命令行路径下不会遇到
    未知阶段。
    """
    plan: Dict[str, List[Dict[str, Any]]] = {}
    for stage in stages:
        candidates = STAGE_BUILDERS[stage]()
        if max_trials_per_stage is not None:
            candidates = candidates[:max_trials_per_stage]
        plan[stage] = candidates
    return plan


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Run the full stage-wise tuning workflow.

    Parameters
    ----------
    argv : sequence of str, optional
        可选命令行参数，主要用于测试。为 ``None`` 时读取真实命令行。

    Returns
    -------
    int
        进程退出码。成功完成调参计划时返回 ``0``。

    Raises
    ------
    ValueError
        当命令行参数或阶段名称不合法时抛出。
    OSError
        当输出目录、日志或汇总文件无法创建/写入时抛出。
    TypeError
        当调参计划或结果记录无法序列化为 JSON 时抛出。

    Notes
    -----
    主流程会为本次调参创建时间戳目录；每完成一个 trial，就立即刷新
    ``tuning_trials.csv`` 和 ``tuning_trials.json``。阶段结束后，脚本把该
    阶段最优参数作为下一阶段的起点，最终写出 ``final_best_params.json``。
    """
    args = parse_args(argv)
    stages = selected_stages(args.stages)
    direction = metric_direction(args.selection_metric, args.direction)

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_root = args.output_root / run_id
    ensure_dir(run_root)

    plan = candidate_plan(stages, args.max_trials_per_stage)
    write_json(
        run_root / "tuning_plan.json",
        {
            "stages": stages,
            "selection_split": args.selection_split,
            "selection_metric": args.selection_metric,
            "direction": direction,
            "pipeline_script": str(args.pipeline_script),
            "python_executable": args.python_executable,
            "pipeline_extra_args": strip_pipeline_extra_args(args.pipeline_extra_args),
            "candidates": plan,
        },
    )

    all_records: List[Dict[str, Any]] = []
    best_by_stage: List[Dict[str, Any]] = []
    current_best_params: Dict[str, Any] = {"split_strategy": "time-series"}
    trial_index = 0

    for stage in stages:
        stage_best_record: Optional[Dict[str, Any]] = None
        stage_best_params: Optional[Dict[str, Any]] = None
        for changed_params in plan[stage]:
            trial_index += 1
            effective_params = dict(current_best_params)
            effective_params.update(changed_params)
            print(
                f"[tune] stage={stage} trial={trial_index} "
                f"params={json.dumps(changed_params, sort_keys=True, ensure_ascii=False)}"
            )
            record = run_trial(
                args=args,
                run_root=run_root,
                stage=stage,
                trial_index=trial_index,
                changed_params=changed_params,
                effective_params=effective_params,
            )
            all_records.append(record)
            write_json(run_root / "tuning_trials.json", all_records)
            write_records_csv(run_root / "tuning_trials.csv", all_records)

            if record.get("status") == "completed" and is_better(
                None if record.get("score") is None else float(record["score"]),
                None if stage_best_record is None or stage_best_record.get("score") is None else float(stage_best_record["score"]),
                direction,
            ):
                stage_best_record = record
                stage_best_params = effective_params

        if stage_best_record is not None and stage_best_params is not None:
            current_best_params = dict(stage_best_params)
            best_by_stage.append(
                {
                    "stage": stage,
                    "score": stage_best_record.get("score"),
                    "output_dir": stage_best_record.get("output_dir"),
                    "params": current_best_params,
                }
            )
            write_json(run_root / "best_trials_by_stage.json", best_by_stage)

    completed_records = [record for record in all_records if record.get("status") == "completed"]
    completed_records.sort(
        key=lambda item: (
            float("-inf") if direction == "maximize" else float("inf")
        )
        if item.get("score") is None
        else (-float(item["score"]) if direction == "maximize" else float(item["score"]))
    )
    write_json(
        run_root / "final_best_params.json",
        {
            "selection_split": args.selection_split,
            "selection_metric": args.selection_metric,
            "direction": direction,
            "best_overall": None if not completed_records else completed_records[0],
            "best_by_stage": best_by_stage,
        },
    )
    print(f"[tune] saved results to {run_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

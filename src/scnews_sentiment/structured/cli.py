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
from src.scnews_sentiment.structured.pipeline import *

def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """解析命令行参数并执行基础合法性检查。

    Parameters
    ----------
    argv : sequence of str or None, optional
        命令行参数列表；为 ``None`` 时使用 ``sys.argv``。

    Returns
    -------
    argparse.Namespace
        已解析且经过基础校验的参数对象。

    Raises
    ------
    ValueError
        当维度、比例、学习率、Dropout、正参数初始化或批大小等配置不合法时
        抛出。
    SystemExit
        当 argparse 解析失败或用户请求帮助信息时由 argparse 抛出。
    """
    # 计算并保存 parser，供后续逻辑复用。
    parser = argparse.ArgumentParser(description=__doc__)
    # 执行 add_argument，完成当前步骤的副作用。
    parser.add_argument("--news-csv", type=Path, default=DEFAULT_NEWS_CSV)
    # 执行 add_argument，完成当前步骤的副作用。
    parser.add_argument("--label-csv", type=Path, default=DEFAULT_LABEL_CSV)
    # 执行 add_argument，完成当前步骤的副作用。
    parser.add_argument("--nodes-csv", type=Path, default=DEFAULT_NODES_CSV)
    # 执行 add_argument，完成当前步骤的副作用。
    parser.add_argument("--edges-csv", type=Path, default=DEFAULT_EDGES_CSV)
    # 执行 add_argument，完成当前步骤的副作用。
    parser.add_argument("--encoding", default="utf-8-sig")
    # 执行 add_argument，完成当前步骤的副作用。
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    # 执行 add_argument，完成当前步骤的副作用。
    parser.add_argument("--tensorboard-log-dir", type=Path, default=None)
    # 执行 add_argument，完成当前步骤的副作用。
    parser.add_argument("--no-tensorboard", action="store_true")
    # 执行 add_argument，完成当前步骤的副作用。
    parser.add_argument("--save-representations", action="store_true")
    # 执行 add_argument，完成当前步骤的副作用。
    parser.add_argument("--run-graph-propagation", action="store_true")
    # 执行 set_defaults，完成当前步骤的副作用。
    parser.set_defaults(train_supervised=True)
    # 执行 add_argument，完成当前步骤的副作用。
    parser.add_argument(
        "--train-supervised",
        dest="train_supervised",
        action="store_true",
        help="Run supervised training. This is the default when no mode flag is given.",
    )
    # 执行 add_argument，完成当前步骤的副作用。
    parser.add_argument(
        "--no-train-supervised",
        dest="train_supervised",
        action="store_false",
        help="Disable supervised training and run representation or graph propagation mode.",
    )
    # 执行 add_argument，完成当前步骤的副作用。
    parser.add_argument("--dry-run", action="store_true")
    # 执行 add_argument，完成当前步骤的副作用。
    parser.add_argument("--max-rows", type=int, default=None)
    # 执行 add_argument，完成当前步骤的副作用。
    parser.add_argument("--max-label-rows", type=int, default=None)

    # 执行 add_argument，完成当前步骤的副作用。
    parser.add_argument("--topic-col", default="Topic")
    # 执行 add_argument，完成当前步骤的副作用。
    parser.add_argument("--date-col", default="Date")
    # 执行 add_argument，完成当前步骤的副作用。
    parser.add_argument("--target-col", default="ArticleScore")
    # 执行 add_argument，完成当前步骤的副作用。
    parser.add_argument("--title-col", default="Article_Title")
    # 执行 add_argument，完成当前步骤的副作用。
    parser.add_argument("--text-col", default="Article_Text")
    # 执行 add_argument，完成当前步骤的副作用。
    parser.add_argument(
        "--daily-group-cols",
        default="topic,date_yyyymmdd",
        help="Comma-separated metadata columns used to average r_{t,u}.",
    )
    # 执行 add_argument，完成当前步骤的副作用。
    parser.add_argument(
        "--fill-missing-gdelt",
        type=float,
        default=0.0,
        help="Numeric fallback for missing structured features. Use empty value is unsupported.",
    )

    # 执行 add_argument，完成当前步骤的副作用。
    parser.add_argument("--encoder-model", type=Path, default=DEFAULT_ENCODER_MODEL)
    # 执行 add_argument，完成当前步骤的副作用。
    parser.add_argument("--encoder-device", default="auto")
    # 执行 add_argument，完成当前步骤的副作用。
    parser.add_argument("--encoder-max-length", type=int, default=512)
    # 执行 add_argument，完成当前步骤的副作用。
    parser.add_argument("--embedding-batch-size", type=int, default=16)
    # 执行 add_argument，完成当前步骤的副作用。
    parser.add_argument("--pooling", choices=["cls", "mean"], default="cls")
    # 执行 add_argument，完成当前步骤的副作用。
    parser.add_argument("--normalize-text-embeddings", action="store_true", default=True)
    # 执行 add_argument，完成当前步骤的副作用。
    parser.add_argument(
        "--no-normalize-text-embeddings",
        action="store_false",
        dest="normalize_text_embeddings",
    )

    # 执行 add_argument，完成当前步骤的副作用。
    parser.add_argument("--train-device", default="auto")
    # 执行 add_argument，完成当前步骤的副作用。
    parser.add_argument("--projection-dim", type=int, default=128)
    # 执行 add_argument，完成当前步骤的副作用。
    parser.add_argument("--sent-hidden-sizes", default="256,128")
    # 执行 add_argument，完成当前步骤的副作用。
    parser.add_argument("--representation-dim", type=int, default=256)
    # 执行 add_argument，完成当前步骤的副作用。
    parser.add_argument("--dropout", type=float, default=0.1)
    # 执行 add_argument，完成当前步骤的副作用。
    parser.add_argument("--representation-batch-size", type=int, default=256)
    # 执行 add_argument，完成当前步骤的副作用。
    parser.add_argument("--checkpoint", type=Path, default=None)
    # 执行 add_argument，完成当前步骤的副作用。
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--repeat-seeds",
        default="",
        help=(
            "Comma-separated seeds for repeated TimeSeriesSplit CV. "
            "Empty value keeps the historical single-CV behavior."
        ),
    )
    # 执行 add_argument，完成当前步骤的副作用。
    parser.add_argument("--no-final-layer-norm", action="store_true")

    # 执行 add_argument，完成当前步骤的副作用。
    parser.add_argument("--node-code-col", default="hs_code_4d")
    # 执行 add_argument，完成当前步骤的副作用。
    parser.add_argument("--igpc-col", default="IGPC")
    # 执行 add_argument，完成当前步骤的副作用。
    parser.add_argument("--edge-upstream-col", default="hs2022_code_upstream")
    # 执行 add_argument，完成当前步骤的副作用。
    parser.add_argument("--edge-downstream-col", default="hs2022_code_downstream")
    # 执行 add_argument，完成当前步骤的副作用。
    parser.add_argument("--edge-weight-col", default="")
    # 执行 add_argument，完成当前步骤的副作用。
    parser.add_argument(
        "--graph-direction",
        choices=["upstream-to-downstream", "downstream-to-upstream", "bidirectional"],
        default="downstream-to-upstream",
    )
    # 执行 add_argument，完成当前步骤的副作用。
    parser.add_argument(
        "--topic-hs-map",
        action="append",
        default=[],
        help="Topic to HS4 mapping used for node-level news injection, e.g. Automaker=8703.",
    )
    # 执行 add_argument，完成当前步骤的副作用。
    parser.add_argument("--graph-state-dim", type=int, default=128)
    parser.add_argument(
        "--propagation-model",
        choices=[
            "reaction_diffusion",
            "gcn",
            "gat",
            "dcrnn",
            "agcrn",
            "lstm",
            "stgcn",
            "itransformer",
            "timesnet",
            "gru_gated_reaction_diffusion",
            "lstm_gated_reaction_diffusion",
            "gru_z_gated_reaction_diffusion",
            "lstm_z_gated_reaction_diffusion",
            "lstm_component_gated_reaction_diffusion",
            "lstm_component_fusion_gated_reaction_diffusion",
        ],
        default="reaction_diffusion",
    )
    parser.add_argument("--gat-heads", type=int, default=4)
    parser.add_argument("--dcrnn-diffusion-steps", type=int, default=2)
    parser.add_argument("--dcrnn-bidirectional", action="store_true", default=False)
    parser.add_argument("--agcrn-embed-dim", type=int, default=10)
    parser.add_argument("--ts-baseline-lookback", type=int, default=5)
    parser.add_argument("--itransformer-heads", type=int, default=4)
    parser.add_argument("--stgcn-kernel-size", type=int, default=3)
    parser.add_argument("--timesnet-kernel-sizes", default="1,3,5")
    parser.add_argument(
        "--component-scale-mode",
        choices=["learnable", "none"],
        default="learnable",
    )
    parser.add_argument("--component-scale-init", type=float, default=0.01)
    parser.add_argument("--rd-fusion-gate-init", type=float, default=0.1)
    # 执行 add_argument，完成当前步骤的副作用。
    parser.add_argument("--graph-risk-hidden-sizes", default="")
    # 执行 add_argument，完成当前步骤的副作用。
    parser.add_argument("--graph-recovery-hidden-sizes", default="256")
    # 执行 add_argument，完成当前步骤的副作用。
    parser.add_argument("--graph-dropout", type=float, default=0.1)
    # 执行 add_argument，完成当前步骤的副作用。
    parser.add_argument("--propagation-dt", type=float, default=1.0)
    # 执行 add_argument，完成当前步骤的副作用。
    parser.add_argument("--alpha-init", type=float, default=0.1)
    # 执行 add_argument，完成当前步骤的副作用。
    parser.add_argument("--beta-init", type=float, default=0.1)
    # 执行 add_argument，完成当前步骤的副作用。
    parser.add_argument("--gamma-init", type=float, default=0.1)
    # 执行 add_argument，完成当前步骤的副作用。
    parser.add_argument("--eta-init", type=float, default=0.1)
    # 执行 add_argument，完成当前步骤的副作用。
    parser.add_argument("--recovery-eps", type=float, default=1e-6)
    # 执行 add_argument，完成当前步骤的副作用。
    parser.add_argument(
        "--diffusion-mode",
        choices=["deviation", "direct"],
        default="deviation",
        help="Use h-h0 diffusion by default; direct diffuses h itself.",
    )
    parser.add_argument(
        "--initial-state-mode",
        choices=["igpc", "learnable", "fixed_zero"],
        default="igpc",
        help="Choose how h0 is initialized: IGPC prior, learnable non-IGPC h0, or fixed zero.",
    )
    parser.add_argument(
        "--enable-graph-diffusion",
        dest="enable_graph_diffusion",
        action="store_true",
        default=True,
        help="Enable graph diffusion h_diff in the propagation update.",
    )
    parser.add_argument(
        "--disable-graph-diffusion",
        dest="enable_graph_diffusion",
        action="store_false",
        help="Disable graph diffusion h_diff for ablation.",
    )
    parser.add_argument(
        "--enable-two-hop-diffusion",
        dest="enable_two_hop_diffusion",
        action="store_true",
        default=True,
        help="Enable the gamma-weighted two-hop diffusion term.",
    )
    parser.add_argument(
        "--disable-two-hop-diffusion",
        dest="enable_two_hop_diffusion",
        action="store_false",
        help="Disable the gamma-weighted two-hop diffusion term for ablation.",
    )
    parser.add_argument(
        "--enable-recovery",
        dest="enable_recovery",
        action="store_true",
        default=True,
        help="Enable recovery/reaction h_react in the propagation update.",
    )
    parser.add_argument(
        "--disable-recovery",
        dest="enable_recovery",
        action="store_false",
        help="Disable recovery/reaction h_react for ablation.",
    )
    # 执行 add_argument，完成当前步骤的副作用。
    parser.add_argument("--graph-checkpoint", type=Path, default=None)

    # 执行 add_argument，完成当前步骤的副作用。
    parser.add_argument("--upstream-node-code", default="3818")
    # 执行 add_argument，完成当前步骤的副作用。
    parser.add_argument("--downstream-node-code", default="8703")
    # 执行 add_argument，完成当前步骤的副作用。
    parser.add_argument("--upstream-loss-weight", type=float, default=1.0)
    # 执行 add_argument，完成当前步骤的副作用。
    parser.add_argument("--smooth-weight", type=float, default=0.01)
    # 执行 add_argument，完成当前步骤的副作用。
    parser.add_argument("--align-weight", type=float, default=0.1)
    # 执行 add_argument，完成当前步骤的副作用。
    parser.add_argument("--align-temperature", type=float, default=0.07)
    # 执行 add_argument，完成当前步骤的副作用。
    parser.add_argument("--align-dim", type=int, default=128)
    # 执行 add_argument，完成当前步骤的副作用。
    parser.add_argument("--align-hidden-sizes", default="")
    # 执行 add_argument，完成当前步骤的副作用。
    parser.add_argument("--align-dropout", type=float, default=0.1)
    # 执行 add_argument，完成当前步骤的副作用。
    parser.add_argument("--llm-forecast-csv", type=Path, default=DEFAULT_LLM_FORECAST_CSV)
    # 执行 add_argument，完成当前步骤的副作用。
    parser.add_argument("--llm-embedding-npz", type=Path, default=DEFAULT_LLM_EMBEDDINGS_NPZ)
    # 执行 add_argument，完成当前步骤的副作用。
    parser.add_argument("--eval-hidden-sizes", default="128")
    # 执行 add_argument，完成当前步骤的副作用。
    parser.add_argument("--eval-dropout", type=float, default=0.1)
    # 执行 add_argument，完成当前步骤的副作用。
    parser.add_argument("--epochs", type=int, default=50)
    # 执行 add_argument，完成当前步骤的副作用。
    parser.add_argument("--lr", type=float, default=1e-3)
    # 执行 add_argument，完成当前步骤的副作用。
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    # 执行 add_argument，完成当前步骤的副作用。
    parser.add_argument("--grad-clip-norm", type=float, default=5.0)
    # 执行 add_argument，完成当前步骤的副作用。
    parser.add_argument(
        "--checkpoint-metric",
        choices=["val_supervised", "val_total"],
        default="val_supervised",
        help="Validation loss used for best-checkpoint selection and early stopping.",
    )
    # 执行 add_argument，完成当前步骤的副作用。
    parser.add_argument("--train-ratio", type=float, default=0.70)
    # 执行 add_argument，完成当前步骤的副作用。
    parser.add_argument("--val-ratio", type=float, default=0.15)
    # 执行 add_argument，完成当前步骤的副作用。
    parser.add_argument("--split-strategy", choices=["fixed", "time-series"], default="time-series")
    # 执行 add_argument，完成当前步骤的副作用。
    parser.add_argument("--final-test-ratio", type=float, default=0.15)
    # 执行 add_argument，完成当前步骤的副作用。
    parser.add_argument("--tscv-n-splits", type=int, default=5)
    # 执行 add_argument，完成当前步骤的副作用。
    parser.add_argument("--tscv-test-size", type=int, default=None)
    # 执行 add_argument，完成当前步骤的副作用。
    parser.add_argument("--tscv-gap", type=int, default=0)
    # 执行 add_argument，完成当前步骤的副作用。
    parser.add_argument(
        "--early-stopping-patience",
        type=int,
        default=0,
        help="Stop training after this many epochs without validation improvement; 0 disables it.",
    )
    # 执行 add_argument，完成当前步骤的副作用。
    parser.add_argument(
        "--early-stopping-min-delta",
        type=float,
        default=1e-4,
        help="Minimum validation loss decrease required to reset early-stopping patience.",
    )
    # 执行 add_argument，完成当前步骤的副作用。
    parser.add_argument("--log-every", type=int, default=5)

    # 计算并保存 args，供后续逻辑复用。
    args = parser.parse_args(argv)
    # 判断 args.max_rows is not None and args.max_rows <= 0，决定是否进入对应分支。
    if args.max_rows is not None and args.max_rows <= 0:
        # 抛出异常，阻止无效数据继续向下游传播。
        raise ValueError("--max-rows must be positive when provided.")
    # 判断 args.max_label_rows is not None and args.max_label_row...，决定是否进入对应分支。
    if args.max_label_rows is not None and args.max_label_rows <= 0:
        # 抛出异常，阻止无效数据继续向下游传播。
        raise ValueError("--max-label-rows must be positive when provided.")
    # 判断 args.projection_dim <= 0 or args.representation_dim <=...，决定是否进入对应分支。
    if args.projection_dim <= 0 or args.representation_dim <= 0:
        # 抛出异常，阻止无效数据继续向下游传播。
        raise ValueError("--projection-dim and --representation-dim must be positive.")
    # 判断 args.embedding_batch_size <= 0 or args.representation_...，决定是否进入对应分支。
    if args.embedding_batch_size <= 0 or args.representation_batch_size <= 0:
        # 抛出异常，阻止无效数据继续向下游传播。
        raise ValueError("batch sizes must be positive.")
    # 判断 not (0.0 <= args.dropout < 1.0)，决定是否进入对应分支。
    if not (0.0 <= args.dropout < 1.0):
        # 抛出异常，阻止无效数据继续向下游传播。
        raise ValueError("--dropout must be in [0, 1).")
    # 判断 args.graph_state_dim <= 0，决定是否进入对应分支。
    if args.graph_state_dim <= 0:
        # 抛出异常，阻止无效数据继续向下游传播。
        raise ValueError("--graph-state-dim must be positive.")
    if args.gat_heads <= 0:
        raise ValueError("--gat-heads must be positive.")
    if args.propagation_model == "gat" and args.graph_state_dim % args.gat_heads != 0:
        raise ValueError("--graph-state-dim must be divisible by --gat-heads.")
    if args.dcrnn_diffusion_steps <= 0:
        raise ValueError("--dcrnn-diffusion-steps must be positive.")
    if args.agcrn_embed_dim <= 0:
        raise ValueError("--agcrn-embed-dim must be positive.")
    if args.ts_baseline_lookback <= 0:
        raise ValueError("--ts-baseline-lookback must be positive.")
    if args.itransformer_heads <= 0:
        raise ValueError("--itransformer-heads must be positive.")
    if (
        args.propagation_model == "itransformer"
        and args.graph_state_dim % args.itransformer_heads != 0
    ):
        raise ValueError("--graph-state-dim must be divisible by --itransformer-heads.")
    if args.stgcn_kernel_size <= 0:
        raise ValueError("--stgcn-kernel-size must be positive.")
    if args.propagation_model == "timesnet":
        for item in str(args.timesnet_kernel_sizes).split(","):
            item = item.strip()
            if item and int(item) <= 0:
                raise ValueError("--timesnet-kernel-sizes must contain positive integers.")
    if args.component_scale_init <= 0:
        raise ValueError("--component-scale-init must be positive.")
    if not (0.0 < args.rd_fusion_gate_init < 1.0):
        raise ValueError("--rd-fusion-gate-init must be in (0, 1).")
    # 判断 not (0.0 <= args.graph_dropout < 1.0)，决定是否进入对应分支。
    if not (0.0 <= args.graph_dropout < 1.0):
        # 抛出异常，阻止无效数据继续向下游传播。
        raise ValueError("--graph-dropout must be in [0, 1).")
    # 判断 args.propagation_dt <= 0，决定是否进入对应分支。
    if args.propagation_dt <= 0:
        # 抛出异常，阻止无效数据继续向下游传播。
        raise ValueError("--propagation-dt must be positive.")
    # 遍历 ["alpha_init", "beta_init", "gamma_init", ，逐项执行处理逻辑。
    for name in ["alpha_init", "beta_init", "gamma_init", "eta_init", "recovery_eps"]:
        # 判断 getattr(args, name) <= 0，处理对应情况。
        if getattr(args, name) <= 0:
            # 抛出异常，阻止无效数据继续向下游传播。
            raise ValueError(f"--{name.replace('_', '-')} must be positive.")
    # 判断 args.upstream_loss_weight <= 0，决定是否进入对应分支。
    if args.upstream_loss_weight <= 0:
        # 抛出异常，阻止无效数据继续向下游传播。
        raise ValueError("--upstream-loss-weight must be positive.")
    # 判断 args.smooth_weight < 0，决定是否进入对应分支。
    if args.smooth_weight < 0:
        # 抛出异常，阻止无效数据继续向下游传播。
        raise ValueError("--smooth-weight must be non-negative.")
    # 判断 args.align_weight < 0，决定是否进入对应分支。
    if args.align_weight < 0:
        # 抛出异常，阻止无效数据继续向下游传播。
        raise ValueError("--align-weight must be non-negative.")
    # 判断 args.align_temperature <= 0，决定是否进入对应分支。
    if args.align_temperature <= 0:
        # 抛出异常，阻止无效数据继续向下游传播。
        raise ValueError("--align-temperature must be positive.")
    # 判断 args.align_dim <= 0，决定是否进入对应分支。
    if args.align_dim <= 0:
        # 抛出异常，阻止无效数据继续向下游传播。
        raise ValueError("--align-dim must be positive.")
    # 判断 not (0.0 <= args.align_dropout < 1.0)，决定是否进入对应分支。
    if not (0.0 <= args.align_dropout < 1.0):
        # 抛出异常，阻止无效数据继续向下游传播。
        raise ValueError("--align-dropout must be in [0, 1).")
    # 执行 parse_hidden_sizes，完成当前步骤的副作用。
    parse_hidden_sizes(args.align_hidden_sizes)
    # 判断 not (0.0 <= args.eval_dropout < 1.0)，决定是否进入对应分支。
    if not (0.0 <= args.eval_dropout < 1.0):
        # 抛出异常，阻止无效数据继续向下游传播。
        raise ValueError("--eval-dropout must be in [0, 1).")
    # 判断 args.epochs <= 0，决定是否进入对应分支。
    if args.epochs <= 0:
        # 抛出异常，阻止无效数据继续向下游传播。
        raise ValueError("--epochs must be positive.")
    # 判断 args.lr <= 0 or args.weight_decay < 0，决定是否进入对应分支。
    if args.lr <= 0 or args.weight_decay < 0:
        # 抛出异常，阻止无效数据继续向下游传播。
        raise ValueError("--lr must be positive and --weight-decay must be non-negative.")
    # 判断 args.grad_clip_norm < 0，决定是否进入对应分支。
    if args.grad_clip_norm < 0:
        # 抛出异常，阻止无效数据继续向下游传播。
        raise ValueError("--grad-clip-norm must be non-negative.")
    # 判断 args.early_stopping_patience < 0，决定是否进入对应分支。
    if args.early_stopping_patience < 0:
        # 抛出异常，阻止无效数据继续向下游传播。
        raise ValueError("--early-stopping-patience must be non-negative.")
    # 判断 args.early_stopping_min_delta < 0，决定是否进入对应分支。
    if args.early_stopping_min_delta < 0:
        # 抛出异常，阻止无效数据继续向下游传播。
        raise ValueError("--early-stopping-min-delta must be non-negative.")
    # 判断 args.log_every <= 0，决定是否进入对应分支。
    if args.log_every <= 0:
        # 抛出异常，阻止无效数据继续向下游传播。
        raise ValueError("--log-every must be positive.")
    # 判断 args.train_ratio <= 0 or args.val_ratio < 0 or args.tr...，决定是否进入对应分支。
    if args.train_ratio <= 0 or args.val_ratio < 0 or args.train_ratio + args.val_ratio >= 1:
        # 抛出异常，阻止无效数据继续向下游传播。
        raise ValueError("Require 0 < train_ratio and train_ratio + val_ratio < 1.")
    # 判断 not (0.0 < args.final_test_ratio < 1.0)，决定是否进入对应分支。
    if not (0.0 < args.final_test_ratio < 1.0):
        # 抛出异常，阻止无效数据继续向下游传播。
        raise ValueError("--final-test-ratio must be in (0, 1).")
    # 判断 args.tscv_n_splits <= 0，决定是否进入对应分支。
    if args.tscv_n_splits <= 0:
        # 抛出异常，阻止无效数据继续向下游传播。
        raise ValueError("--tscv-n-splits must be positive.")
    # 判断 args.tscv_test_size is not None and args.tscv_test_siz...，决定是否进入对应分支。
    if args.tscv_test_size is not None and args.tscv_test_size <= 0:
        # 抛出异常，阻止无效数据继续向下游传播。
        raise ValueError("--tscv-test-size must be positive when provided.")
    # 判断 args.tscv_gap < 0，决定是否进入对应分支。
    if args.tscv_gap < 0:
        # 抛出异常，阻止无效数据继续向下游传播。
        raise ValueError("--tscv-gap must be non-negative.")
    if str(getattr(args, "repeat_seeds", "")).strip():
        try:
            repeat_seeds = [
                int(item.strip())
                for item in str(args.repeat_seeds).split(",")
                if item.strip()
            ]
        except ValueError as exc:
            raise ValueError("--repeat-seeds must contain comma-separated integers.") from exc
        if not repeat_seeds:
            raise ValueError("--repeat-seeds must contain at least one integer when set.")

    # 返回当前函数的处理结果。
    return args


def main(argv: Optional[Sequence[str]] = None) -> int:
    """脚本主入口。

    Parameters
    ----------
    argv : sequence of str or None, optional
        可选命令行参数；为 ``None`` 时使用系统命令行。

    Returns
    -------
    int
        进程退出码。当前成功路径返回 0。

    Raises
    ------
    ValueError
        当参数或输入数据不满足约束时由下游函数抛出。
    RuntimeError
        当 PyTorch 模型运行失败时由下游函数抛出。
    OSError
        当文件读写失败时由下游函数抛出。

    Notes
    -----
    根据开关选择 dry-run、监督训练、初始化表示生成或图传播导出路径。
    """
    # 计算并保存 args，供后续逻辑复用。
    args = parse_args(argv)
    
    # 轻量级输入检查，不真正加载 BGE 文本模型、不生成 embedding、不训练模型、不保存训练结果
    # 在正式训练前快速确认数据和配置有没有问题
    if args.dry_run:
        # 执行 validate_inputs，完成当前步骤的副作用。
        validate_inputs(args)
        # 返回当前分支或函数的处理结果。
        return 0

    # 加载 BGE、生成 embedding、执行图传播、反向传播训练模型，并保存预测、指标和模型权重
    if args.train_supervised:
        # 执行 train_supervised_structured_model，完成当前步骤的副作用。
        train_supervised_structured_model(args)
        # 返回当前分支或函数的处理结果。
        return 0

    # 计算并保存 batch，供后续逻辑复用。
    batch = build_initial_representations(args)
    # 执行 print，完成当前步骤的副作用。
    print(f"Loaded news rows: {len(batch.news_metadata)}")
    # 执行 print，完成当前步骤的副作用。
    print(f"x_gdelt shape: {batch.gdelt_features.shape}")
    # 执行 print，完成当前步骤的副作用。
    print(f"x_text shape: {batch.text_embeddings.shape}")
    # 执行 print，完成当前步骤的副作用。
    print(f"r_news shape: {batch.news_representations.shape}")
    # 执行 print，完成当前步骤的副作用。
    print(f"r_daily shape: {batch.daily_representations.shape}")

    # 初始化带类型标注的 graph_batch，明确后续存储内容。
    graph_batch: Optional[GraphPropagationBatch] = None
    # 根据条件 args.run_graph_propagation 选择对应处理分支。
    if args.run_graph_propagation:
        # 计算并保存 graph_batch，供当前分支后续使用。
        graph_batch = run_graph_propagation(batch, args)
        # 执行 print，完成当前步骤的副作用。
        print(f"graph nodes: {len(graph_batch.graph.node_codes)}")
        # 执行 print，完成当前步骤的副作用。
        print(f"graph directed edges used: {graph_batch.graph.edge_count}")
        # 执行 print，完成当前步骤的副作用。
        print(f"node risk sequence shape: {graph_batch.node_risk_sequence.shape}")
        # 执行 print，完成当前步骤的副作用。
        print(f"h0 graph shape: {graph_batch.initial_node_states.shape}")
        # 执行 print，完成当前步骤的副作用。
        print(f"h_struct sequence shape: {graph_batch.propagated_node_states.shape}")

    # 根据条件 args.save_representations 选择对应处理分支。
    if args.save_representations:
        # 执行 save_representations，完成当前步骤的副作用。
        save_representations(batch, args)
        # 判断 graph_batch is not None，处理对应情况。
        if graph_batch is not None:
            # 执行 save_graph_propagation，完成当前步骤的副作用。
            save_graph_propagation(graph_batch, args)
        # 执行 print，完成当前步骤的副作用。
        print(f"Saved structured representations to {args.output_dir}")
    else:
        # 执行 print，完成当前步骤的副作用。
        print("Representations were kept in memory only. Pass --save-representations to write them.")

    # 返回当前函数的处理结果。
    return 0



if __name__ == "__main__":
    raise SystemExit(main())

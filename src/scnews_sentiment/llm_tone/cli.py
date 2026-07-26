from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, Sequence

from src.scnews_sentiment.llm_tone.pipeline import run_event_tone_pipeline
from src.scnews_sentiment.llm_tone.utils import (
    DEFAULT_NEWS_CSV,
    DEFAULT_NODES_CSV,
    DEFAULT_OUTPUT_DIR,
    coerce_config_arg_types,
    load_config_defaults,
    parse_hidden_sizes,
)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Optional YAML/JSON config file. CLI arguments override config values.",
    )
    config_args, _ = config_parser.parse_known_args(argv)
    config_defaults = load_config_defaults(config_args.config)

    parser = argparse.ArgumentParser(
        description="LLM event-tone forecasting and MLP regression pipeline.",
        parents=[config_parser],
    )
    parser.add_argument(
        "--stage",
        choices=[
            "build-samples",
            "compress",
            "forecast",
            "train",
            "all",
            "optimize-prompt",
            "eval-fixed-scorers",
            "optimize-prompt-and-eval-fixed-scorers",
        ],
        default="build-samples",
        help="Default only builds sample metadata and does not call LLM or encoder.",
    )
    parser.add_argument("--news-csv", type=Path, default=DEFAULT_NEWS_CSV)
    parser.add_argument("--nodes-csv", type=Path, default=DEFAULT_NODES_CSV)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--topic-col", default="Topic")
    parser.add_argument("--date-col", default="Date")
    parser.add_argument("--target-col", default="ArticleScore")
    parser.add_argument("--title-col", default="Article_Title")
    parser.add_argument("--text-col", default="Article_Text")
    parser.add_argument(
        "--topic-hs-map",
        nargs="*",
        default=[],
        help="Optional Topic=HS4 mappings. Defaults include Automaker=8703 and Nexperia=3818.",
    )
    parser.add_argument("--win", type=int, default=5)
    parser.add_argument(
        "--num-hypotheses",
        type=int,
        default=1,
        help="Number of independent prospective event decodes generated per target sample.",
    )
    parser.add_argument(
        "--split-strategy",
        choices=["per-topic", "global", "time-series"],
        default="per-topic",
    )
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--final-test-ratio", type=float, default=0.15)
    parser.add_argument("--tscv-n-splits", type=int, default=5)
    parser.add_argument("--tscv-test-size", type=int, default=None)
    parser.add_argument("--tscv-gap", type=int, default=0)

    parser.add_argument("--llm-mode", choices=["api", "local"], default="api")
    parser.add_argument("--api-key-env", default="LLM_API_KEY")
    parser.add_argument("--api-base", default="https://api.deepseek.com")
    parser.add_argument("--api-model", default="deepseek-v4-pro")
    parser.add_argument(
        "--local-model-path",
        default=r"C:\Users\Administrator\.cache\modelscope\hub\models\Qwen\Qwen3-8B",
    )
    parser.add_argument("--temperature", type=float, default=0.5)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument(
        "--thinking",
        choices=["auto", "enabled", "disabled"],
        default="auto",
        help="DeepSeek thinking switch. auto means do not send the thinking field.",
    )
    parser.add_argument(
        "--reasoning-effort",
        choices=["none", "high", "max"],
        default="none",
        help="DeepSeek reasoning effort. none means do not send the reasoning_effort field.",
    )
    parser.add_argument("--compress-max-tokens", type=int, default=4096)
    parser.add_argument("--forecast-max-tokens", type=int, default=4096)
    parser.add_argument("--request-sleep", type=float, default=0.0)
    parser.add_argument(
        "--llm-empty-retries",
        type=int,
        default=3,
        help="Retry count when an LLM call returns empty text after cleaning.",
    )
    parser.add_argument(
        "--llm-empty-retry-sleep",
        type=float,
        default=2.0,
        help="Seconds to sleep between empty LLM response retries.",
    )
    parser.add_argument("--force-llm", action="store_true")
    parser.add_argument(
        "--forecast-prompt-guidance",
        default="",
        help="Optional extra global guidance inserted into P_pre.",
    )
    parser.add_argument(
        "--forecast-prompt-path",
        type=Path,
        default=None,
        help="Optional YAML/JSON forecast prompt template file.",
    )
    parser.add_argument(
        "--optimizer-prompt-path",
        type=Path,
        default=None,
        help="Optional YAML/JSON prompt-improvement agent template file.",
    )
    parser.add_argument(
        "--prompt-optimization-rounds",
        type=int,
        default=3,
        help="Number of language-feedback prompt optimization rounds.",
    )
    parser.add_argument(
        "--prompt-loss-top-fraction",
        type=float,
        default=0.20,
        help="Fraction of train samples with largest squared_error used for prompt feedback.",
    )
    parser.add_argument("--prompt-agent-max-tokens", type=int, default=4096)
    parser.add_argument("--prompt-feedback-text-chars", type=int, default=900)
    parser.add_argument("--prompt-feedback-history-chars", type=int, default=600)
    parser.add_argument("--prompt-guidance-max-bullets", type=int, default=12)

    parser.add_argument(
        "--encoder-model",
        default=r"C:\Users\Administrator\.cache\modelscope\hub\models\BAAI\bge-base-en-v1.5",
    )
    parser.add_argument("--encoder-device", default="cuda")
    parser.add_argument("--encoder-max-length", type=int, default=512)
    parser.add_argument("--embedding-batch-size", type=int, default=16)
    parser.add_argument("--pooling", choices=["cls", "mean"], default="cls")
    parser.add_argument("--train-device", default="cuda")
    parser.add_argument("--hidden-sizes", default="256,64")
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--min-delta", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ridge-alphas", default="0.01,0.1,1,10,100,1000")
    parser.add_argument("--huber-alpha", type=float, default=1e-4)
    parser.add_argument("--huber-epsilon", type=float, default=1.35)
    parser.add_argument("--huber-max-iter", type=int, default=5000)
    parser.add_argument(
        "--fixed-scorer-include-final",
        action="store_true",
        help="Also evaluate prompt_optimization/final when running eval-fixed-scorers.",
    )
    parser.add_argument(
        "--fixed-scorer-kinds",
        default="ridge,huber,mlp,mlp_original_arch",
        help="Comma-separated fixed scorers to train and evaluate.",
    )
    parser.add_argument("--fixed-mlp-hidden-sizes", default="32")
    parser.add_argument("--fixed-mlp-dropout", type=float, default=0.2)
    parser.add_argument("--fixed-mlp-batch-size", type=int, default=16)
    parser.add_argument("--fixed-mlp-epochs", type=int, default=500)
    parser.add_argument("--fixed-mlp-lr", type=float, default=1e-3)
    parser.add_argument("--fixed-mlp-weight-decay", type=float, default=1e-4)
    parser.add_argument("--fixed-mlp-patience", type=int, default=50)
    parser.add_argument("--fixed-mlp-min-delta", type=float, default=1e-5)

    allowed_config_keys = {action.dest for action in parser._actions}
    unknown_config_keys = sorted(set(config_defaults) - allowed_config_keys)
    if unknown_config_keys:
        raise ValueError(
            f"Unknown config key(s): {unknown_config_keys}. "
            "Use argparse destination names such as llm_mode or api_model."
        )
    if config_defaults:
        parser.set_defaults(**config_defaults)

    args = parser.parse_args(argv)
    args = coerce_config_arg_types(args)
    if args.num_hypotheses <= 0:
        raise ValueError("--num-hypotheses must be positive.")
    if not (0.0 < args.top_p <= 1.0):
        raise ValueError("--top-p must be in (0, 1].")
    if args.win < 1:
        raise ValueError("--win must be >= 1.")
    if args.train_ratio <= 0 or args.val_ratio < 0 or args.train_ratio + args.val_ratio >= 1:
        raise ValueError("Require 0 < train_ratio and train_ratio + val_ratio < 1.")
    if not (0.0 < args.final_test_ratio < 1.0):
        raise ValueError("--final-test-ratio must be in (0, 1).")
    if args.tscv_n_splits <= 0:
        raise ValueError("--tscv-n-splits must be positive.")
    if args.tscv_test_size is not None and args.tscv_test_size <= 0:
        raise ValueError("--tscv-test-size must be positive when provided.")
    if args.tscv_gap < 0:
        raise ValueError("--tscv-gap must be non-negative.")
    if args.prompt_optimization_rounds < 0:
        raise ValueError("--prompt-optimization-rounds must be >= 0.")
    if not 0 < args.prompt_loss_top_fraction <= 1:
        raise ValueError("--prompt-loss-top-fraction must be in (0, 1].")
    if args.prompt_guidance_max_bullets < 1:
        raise ValueError("--prompt-guidance-max-bullets must be >= 1.")
    if args.llm_empty_retries < 1:
        raise ValueError("--llm-empty-retries must be >= 1.")
    if args.llm_empty_retry_sleep < 0:
        raise ValueError("--llm-empty-retry-sleep must be >= 0.")
    if args.huber_alpha < 0:
        raise ValueError("--huber-alpha must be >= 0.")
    if args.huber_epsilon <= 1:
        raise ValueError("--huber-epsilon must be > 1.")
    if args.huber_max_iter < 1:
        raise ValueError("--huber-max-iter must be >= 1.")
    if isinstance(args.fixed_scorer_kinds, (list, tuple)):
        fixed_scorer_kinds = [str(value).strip() for value in args.fixed_scorer_kinds]
    else:
        fixed_scorer_kinds = [
            value.strip()
            for value in str(args.fixed_scorer_kinds).split(",")
            if value.strip()
        ]
    supported_fixed_scorers = {"ridge", "huber", "mlp", "mlp_original_arch"}
    invalid_fixed_scorers = sorted(set(fixed_scorer_kinds) - supported_fixed_scorers)
    if not fixed_scorer_kinds or invalid_fixed_scorers:
        raise ValueError(
            "--fixed-scorer-kinds must contain one or more of "
            f"{sorted(supported_fixed_scorers)}; invalid values: {invalid_fixed_scorers}."
        )
    args.fixed_scorer_kinds = fixed_scorer_kinds
    fixed_mlp_hidden_sizes = parse_hidden_sizes(args.fixed_mlp_hidden_sizes)
    if any(size < 1 for size in fixed_mlp_hidden_sizes):
        raise ValueError("--fixed-mlp-hidden-sizes must contain positive integers.")
    if not 0 <= args.fixed_mlp_dropout < 1:
        raise ValueError("--fixed-mlp-dropout must be in [0, 1).")
    if args.fixed_mlp_batch_size < 1:
        raise ValueError("--fixed-mlp-batch-size must be >= 1.")
    if args.fixed_mlp_epochs < 1:
        raise ValueError("--fixed-mlp-epochs must be >= 1.")
    if args.fixed_mlp_lr <= 0:
        raise ValueError("--fixed-mlp-lr must be > 0.")
    if args.fixed_mlp_weight_decay < 0:
        raise ValueError("--fixed-mlp-weight-decay must be >= 0.")
    if args.fixed_mlp_patience < 1:
        raise ValueError("--fixed-mlp-patience must be >= 1.")
    if args.fixed_mlp_min_delta < 0:
        raise ValueError("--fixed-mlp-min-delta must be >= 0.")
    args.output_dir = args.output_dir.resolve()
    return args


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    run_event_tone_pipeline(args)


if __name__ == "__main__":
    main()

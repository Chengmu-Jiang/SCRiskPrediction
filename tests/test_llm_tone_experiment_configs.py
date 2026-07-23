from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pandas as pd

from src.scnews_sentiment.llm_tone.cli import parse_args
from src.scnews_sentiment.llm_tone.pipeline import run_event_tone_pipeline


EXPERIMENT_CONFIGS = {
    "forecast_baseline__optimizer_legacy.yaml": ("baseline.yaml", "legacy.yaml"),
    "forecast_baseline__optimizer_anti_overfit_v1.yaml": (
        "baseline.yaml",
        "anti_overfit_v1.yaml",
    ),
    "forecast_baseline_one_shot_v1__optimizer_legacy.yaml": (
        "baseline_one_shot_v1.yaml",
        "legacy.yaml",
    ),
    "forecast_baseline_one_shot_v1__optimizer_anti_overfit_v1.yaml": (
        "baseline_one_shot_v1.yaml",
        "anti_overfit_v1.yaml",
    ),
}


def test_llm_tone_experiment_configs_use_matching_256_64_mlps() -> None:
    config_dir = Path("configs/llm_tone/experiments")

    for filename, (forecast_prompt, optimizer_prompt) in EXPERIMENT_CONFIGS.items():
        args = parse_args(["--config", str(config_dir / filename)])

        assert args.stage == "optimize-prompt-and-eval-fixed-scorers"
        assert args.hidden_sizes == "256,64"
        assert args.fixed_scorer_kinds == ["mlp_original_arch"]
        assert args.fixed_scorer_include_final is True
        assert args.forecast_prompt_path.name == forecast_prompt
        assert args.optimizer_prompt_path.name == optimizer_prompt
        assert "llm_event_tone_experiments_expanded" in str(args.output_dir)


def test_combined_stage_optimizes_prompt_before_fixed_scorers(tmp_path: Path) -> None:
    args = SimpleNamespace(
        stage="optimize-prompt-and-eval-fixed-scorers",
        output_dir=tmp_path,
    )
    news = pd.DataFrame({"news": [1]})
    samples = pd.DataFrame({"sample_id": ["sample-a"]})
    compressions = pd.DataFrame({"daily_id": ["topic|20250101"]})
    call_order: list[str] = []
    optimize = Mock(side_effect=lambda *unused: call_order.append("optimize"))
    evaluate = Mock(side_effect=lambda *unused: call_order.append("evaluate"))

    with (
        patch("src.scnews_sentiment.llm_tone.pipeline.load_daily_news", return_value=news),
        patch("src.scnews_sentiment.llm_tone.pipeline.build_samples", return_value=samples),
        patch("src.scnews_sentiment.llm_tone.pipeline.write_sample_metadata"),
        patch(
            "src.scnews_sentiment.llm_tone.pipeline.ensure_daily_compressions",
            return_value=compressions,
        ),
        patch("src.scnews_sentiment.llm_tone.pipeline.write_daily_compressions_real"),
        patch(
            "src.scnews_sentiment.llm_tone.pipeline.optimize_prompt_with_language_feedback",
            optimize,
        ),
        patch("src.scnews_sentiment.llm_tone.pipeline.evaluate_fixed_scorers", evaluate),
    ):
        run_event_tone_pipeline(args)

    assert call_order == ["optimize", "evaluate"]

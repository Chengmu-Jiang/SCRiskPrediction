# SCRiskPrediction

Structured supply-chain risk prediction project refactored into a compact
Hydra-based deep learning project.

The previous implementation lived mostly in one large script. The current layout
separates configuration, data processing, model components, training utilities,
pipeline orchestration, LLM event-tone preprocessing, news aggregation, and
tuning.

## Quick Start

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the best known structured-risk experiment:

```bash
python src/train.py experiment=structured_pipeline_8703
```

Run a lightweight dry-run that validates inputs without loading the transformer
encoder:

```bash
python src/train.py dry_run=true max_rows=100 max_label_rows=100 output_dir=outputs/dry_run
```

TensorBoard logs use `C:/tensorboard_logs/SCRiskPrediction/structured_pipeline`
by default to avoid TensorFlow/TensorBoard path handling issues on Windows when
the project path contains Chinese characters.

Print metrics from a completed run:

```bash
python src/eval.py output_dir=outputs/structured_pipeline_8703_align005_small_reg
```

## Project Layout

- `src/train.py`: Hydra training entrypoint.
- `src/eval.py`: metrics inspection entrypoint.
- `src/aggregate_daily_top_news.py`: news aggregation entrypoint.
- `src/llm_event_tone.py`: LLM event-tone entrypoint.
- `src/tune_structured.py`: staged tuning entrypoint.
- `src/scnews_sentiment/structured/common.py`: constants, dataclasses, shared parsing and text encoding helpers.
- `src/scnews_sentiment/structured/data.py`: CSV loading, graph loading, label alignment, split construction, and alignment-pair loading.
- `src/scnews_sentiment/structured/models.py`: news encoder, graph propagation module, risk scorer, and MLP builders.
- `src/scnews_sentiment/structured/training_utils.py`: tensor aggregation, losses, metrics, TensorBoard logging, and prediction-frame builders.
- `src/scnews_sentiment/structured/pipeline.py`: end-to-end orchestration for dry-run, representation export, graph propagation, and supervised training.
- `src/scnews_sentiment/llm/event_tone_pipeline.py`: LLM event compression/forecasting and tone regressor pipeline.
- `src/scnews_sentiment/news/aggregate_daily_top_news.py`: GDELT/news aggregation into daily top-news inputs.
- `src/scnews_sentiment/tuning/structured_pipeline_tuner.py`: staged hyperparameter tuning.
- `configs/data/scnews_structured.yaml`: source CSV paths and schema.
- `configs/model/structured_risk.yaml`: encoder, graph, alignment, and scorer settings.
- `configs/experiment/structured_pipeline_8703.yaml`: tuned configuration corresponding to the best command supplied before the refactor.

Additional operational details are in `SCNEWS_REFACTOR.md`.

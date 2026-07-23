# SC News Sentiment Refactor

This project now runs the structured supply-chain risk pipeline through a
compact Hydra-based layout.

## Main Training

Best known configuration:

```bash
python src/train.py experiment=structured_pipeline_8703
```

Quick dry-run without loading the transformer encoder:

```bash
python src/train.py dry_run=true max_rows=100 max_label_rows=100 output_dir=outputs/dry_run
```

TensorBoard logs default to an ASCII-only path under
`C:/tensorboard_logs/SCRiskPrediction/structured_pipeline`. This avoids a
Windows TensorFlow/TensorBoard issue where paths containing Chinese characters
can be reported as "not a directory" even when the directory exists.

The tuned LLM alignment files used by `experiment=structured_pipeline_8703` are
stored locally under `outputs/llm_event_tone_8703/`.

The data paths are configured in `configs/data/scnews_structured.yaml`. They
currently point to this project's local data directory:

```text
${paths.root_dir}/data
```

## Output Files

The structured training pipeline preserves the original core artifact names:

- `time_series_split_metadata.csv`
- `time_series_cv_summary.csv`
- `time_series_cv_aggregate_metrics.csv`
- `time_series_cv_summary.json`
- `fold_XX/structured_supervised_predictions.csv`
- `fold_XX/structured_supervised_time_metadata.csv`
- `fold_XX/structured_supervised_node_metadata.csv`
- `fold_XX/structured_alignment_pairs.csv`
- `fold_XX/structured_supervised_metrics.json`
- `fold_XX/structured_supervised_model.pt`

## Supporting Pipelines

Aggregate raw news into daily top-news CSVs:

```bash
python src/aggregate_daily_top_news.py
```

Run the LLM event-tone pipeline:

```bash
python src/llm_event_tone.py
```

Run the stage-wise tuner. It now launches `src/train.py` with Hydra overrides:

```bash
python src/tune_structured.py
```

## Configuration

- `configs/train.yaml`: default structured training entry.
- `configs/experiment/structured_pipeline_8703.yaml`: best known tuned run.
- `configs/model/structured_risk.yaml`: encoder, graph, alignment, and scorer dimensions.
- `configs/data/scnews_structured.yaml`: CSV paths and schema column names.
- `configs/eval.yaml`: prints a saved metrics JSON from an output directory.

## Cleaned Template Files

Unused Lightning template directories were removed. The retained source tree is
centered on `src/scnews_sentiment`, with top-level entry scripts for training,
evaluation, aggregation, LLM event-tone processing, and tuning.

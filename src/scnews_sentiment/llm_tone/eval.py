from __future__ import annotations

import math

import pandas as pd

from src.scnews_sentiment.llm_tone.utils import metric_dict, require_columns


def select_high_loss_predictions(
    predictions: pd.DataFrame,
    top_fraction: float,
) -> pd.DataFrame:
    require_columns(predictions, ["split", "squared_error"], "tone_predictions.csv")
    train_rows = predictions[predictions["split"].astype(str) == "train"].copy()
    if train_rows.empty:
        raise ValueError("No train rows available for prompt feedback.")
    count = max(1, int(math.ceil(len(train_rows) * top_fraction)))
    return train_rows.sort_values("squared_error", ascending=False).head(count)


__all__ = ["metric_dict", "select_high_loss_predictions"]

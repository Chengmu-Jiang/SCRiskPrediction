from pathlib import Path

import numpy as np
import pandas as pd

from src.scnews_sentiment.llm_tone.scorers.fixed_scorers import (
    save_sample_aligned_real_news_embeddings,
)


def test_save_sample_aligned_real_news_embeddings_maps_by_daily_id(tmp_path: Path) -> None:
    real_ids = ["topic|20250101", "topic|20250102", "topic|history"]
    embeddings = np.asarray([[1.0, 1.5], [2.0, 2.5], [3.0, 3.5]], dtype=np.float32)
    mapping = pd.DataFrame(
        {
            "daily_id": ["topic|20250102", "topic|20250101", "topic|history"],
            "sample_id": ["sample-b", "sample-a", ""],
        }
    )
    output_path = tmp_path / "real_news_sample_embeddings.npz"

    save_sample_aligned_real_news_embeddings(
        real_ids=real_ids,
        real_embeddings=embeddings,
        sample_mapping=mapping,
        output_path=output_path,
    )

    with np.load(output_path, allow_pickle=True) as result:
        assert result["sample_id"].astype(str).tolist() == ["sample-b", "sample-a"]
        assert result["daily_id"].astype(str).tolist() == [
            "topic|20250102",
            "topic|20250101",
        ]
        np.testing.assert_array_equal(result["embedding"], embeddings[[1, 0]])
